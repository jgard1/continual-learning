import abc
import torch
from torch import nn
from torch.nn import functional as F
import utils
import copy
import numpy as np


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []   #--> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = 2000
        self.norm_exemplars = True
        self.herding = True

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass


    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True
            dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k>0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum)/(k+1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k][0].numpy())

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        self.train(mode=mode)


    ####----CLASSIFICATION----####

    def classify_with_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        # logging.info("entered classify ith exemplars")
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)    # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        x_features = feature.unsqueeze(2)             # (batch_size, feature_size, 1)


        jaunes= []
        # incredibly inefficient, but this operation is only called once per item in test set at end, no other time
        for x_idx, x_feature in enumerate(feature):
            x_feature = x_feature.unsqueeze(0)
            cur_min = float("inf")
            pred_class = 0
            for set_idx, P_y in enumerate(self.exemplar_sets):
                    exemplars = []
                    # Collect all exemplars in P_y into a <tensor> and extract their features
                    for ex in P_y:
                        # logging.info("ex.shape: "+str(ex.shape))
                        exemplars.append(torch.from_numpy(ex))
                    exemplars = torch.stack(exemplars).to(self._device())
                    # logging.info("exemplars.shape: "+str(exemplars.shape))
                    with torch.no_grad():
                        exemplar_features = self.feature_extractor(exemplars)
                    if self.norm_exemplars:
                        exemplar_features = F.normalize(exemplar_features, p=2, dim=1)
                    
                    # logging.info("x_feature.shape: "+str(x_feature.shape))
                    # logging.info("exemplar_features.shape: "+str(exemplar_features.shape))
                    feature_expanded = x_feature.expand_as(exemplar_features)         # (batch_size, feature_size, n_classes)

                    # logging.info("feature_expanded.shape: "+str(feature_expanded.shape))
                    # For each data-point in [x], find which exemplar-mean is closest to its extracted features
                    dists = (feature_expanded - exemplar_features).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
                    np_dists = dists.cpu().numpy()
                    # logging.info("np_dists.shape: "+str(np_dists.shape))
                    pred = np.argmin(np_dists)
                    val = np_dists[pred]
                    if(val <= cur_min):
                        pred_class = set_idx
                        cur_min = val
            jaunes.append(pred_class)

        torch_preds = torch.IntTensor(jaunes).to(self._device())
        # Set mode of model back
        self.train(mode=mode)

        return torch_preds

