import os
import argparse
from sklearn.metrics import confusion_matrix 
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion(y_true, y_pred, out_file, plot_title)
    labels = np.unique(y_true)
    labels = [str(i) for i in labels]
    array = confusion_matrix(y_true, y_pred, labels)
    df_cm = pd.DataFrame(array, index = labels,
                      columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

    plt.title(plot_title)
    file_name = str(out_file).replace(" ","")
    plt.savefig(file_name)
    plt.close()