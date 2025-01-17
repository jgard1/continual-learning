import abc
import torch
from torch import nn
from torch.nn import functional as F
import utils
import copy
import numpy as np
from kmeans_pytorch import kmeans, kmeans_predict
import logging 

# 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)





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
    # get the point nearest the centroid
    def get_point_nearest_centroid(self, centroid, cluster_features, original_idxs):
        # https://discuss.pytorch.org/t/among-a-set-of-reference-vectors-how-to-return-the-closest-one-to-a-given-vector/20423
        # logging.info("get_point_nearest_centroid: cluster_features = "+str(cluster_features))
        # logging.info("centroid = "+str(centroid))
        distances = torch.sqrt(torch.sum((cluster_features.squeeze(1) - centroid) ** 2, dim=1)) 
        # logging.info("distances = "+str(distances))
        # logging.info("get_point_nearest_centroid: cluster_features.shape = "+str(cluster_features.shape))
        # logging.info("centroid.shape = "+str(centroid.shape))
        # logging.info("distances.shape = "+str(distances.shape))
        # memes = torch.unsqueeze(distances, 0)
        # logging.info("memes.shape = "+str(memes.shape))
        min_index = np.argmin(distances.numpy())

        original_idx = original_idxs[str(min_index)]

        return cluster_features[min_index], original_idx


    #  returns list of tupples of form [(feature, original dataset idx),...]
    def get_all_points_in_cluster(self, features, cluster_ids_x, cluster_number):
        # error memes here. This line from 
        # https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
        # logging.info("get_all_points_in_cluster: cluster_number="+str(cluster_number))
        # logging.info("get_all_points_in_cluster: cluster_ids_x="+str(cluster_ids_x))
        original_dataset_idxs = ((cluster_ids_x == cluster_number).nonzero(as_tuple=False))
        
        # logging.info("kmeans_exemplars.py: original_datset_idxs = "+str(original_dataset_idxs))
        
        ret_features = features[original_dataset_idxs]

        cluster_to_original_idxs = {}
        for cluster_idx, original_idx in enumerate(original_dataset_idxs):
            cluster_to_original_idxs[str(cluster_idx)] = original_idx
     

        return ret_features, cluster_to_original_idxs



    def get_cluster_exemplars(self, features, num_clusters):
        logging.info("total number of features: "+str(len(features)))
        indices = torch.randperm(len(features))[:3000]
        subsample = features[indices]
        cluster_ids_x, cluster_centers = kmeans(
        X=subsample, num_clusters=num_clusters, distance='euclidean', device=self._device()
        )

        original_idx_map = {}
        ret_features = []
        for cluster_number, centroid in enumerate(cluster_centers):
            
            cluster_features, cluster_to_original_idxs = self.get_all_points_in_cluster(subsample, cluster_ids_x, cluster_number)
            selected_feature, selected_feature_idx = self.get_point_nearest_centroid(centroid, cluster_features, cluster_to_original_idxs)
            ret_features.append(selected_feature)
            
            # maps back to idx in entire dataset of features 
            original_idx_map[str(cluster_number)] = selected_feature_idx
        
        return torch.stack(ret_features), original_idx_map


    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        logging.info("entered  ExemplarHandler.construct_exemplar_set(self, dataset, n)")
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding:
            logging.info("herding enabled")
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

            # josh memes mod: here the features become just the near centroids 
            logging.info("Doing herding, creating a total of "+str(min(n, n_max))+" clusters.")
            features_kmeans, original_idxs_map = self.get_cluster_exemplars(features, min(n, n_max))


            # calculate mean of all features
            class_mean = torch.mean(features_kmeans, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features_kmeans[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k>0:
                    # logging.info("k>0")
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features_kmeans + exemplar_sum)/(k+1)
                    features_dists = features_means - class_mean
                    # logging.info("exemplar_sum: "+str(exemplar_sum))
                    # logging.info("features_dists: "+str(features_dists))
                else:
                    # logging.info("k=0")
                    features_dists = features_kmeans - class_mean
                    # logging.info("exemplar_sum: "+str(exemplar_sum))
                    # logging.info("features_dists: "+str(features_dists))


                #####################################################################################
                #####################################################################################
                #  Josh memes mod: changed index_selected so that it uses our next level shit 
                # index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                features_dists = features_dists.squeeze(1)
                # logging.info("features_dists.shape: "+str(features_dists.shape))
                # logging.info("torch.norm(features_dists, p=2, dim=1).shape: "+str(torch.norm(features_dists, p=2, dim=1).shape))
                shortlist_idx_selected = (np.argmin(torch.norm(features_dists, p=2, dim=1))).item()
                # logging.info("shortlist_idxs_selected.shape: "+str(shortlist_idxs_selected.shape))
                # logging.info("shortlist_idxs_selected: "+str(shortlist_idxs_selected))
                # logging.info("original_idxs_map: "+str(original_idxs_map))
                index_selected = original_idxs_map[str(shortlist_idx_selected)].item()
                # logging.info("just selected: index_selected: "+str(index_selected))
                # END JOSH Memes mod ################################################################
                #####################################################################################

                if index_selected in list_of_selected:
                    logging.info("error: index_selected: "+str(index_selected))
                    logging.info("error: list_of_selected: "+str(list_of_selected))
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features_kmeans[shortlist_idx_selected])

                # make sure this example won't be selected again
                features_kmeans[shortlist_idx_selected] = features_kmeans[shortlist_idx_selected] + 100000000
        else:
            logging.info("herding not enabled")
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
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        if self.compute_means:
            exemplar_means = []  #--> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for P_y in self.exemplar_sets:
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())       # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)        # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)              # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)    # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)             # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)         # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

