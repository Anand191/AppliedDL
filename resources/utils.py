from sklearn.metrics import confusion_matrix, classification_report
import itertools
import numpy as np
from scipy.io import loadmat
import pandas as pd

def conf_matrix(Y_true, Y_pred, fig, z=1, labels=[1, -1], target_names=['Regular', 'Anomaly'], split_name="Test Set",
                model_name="Isolation forest"):

    cm = confusion_matrix(Y_true, Y_pred, labels)
    # print(classification_report(Y_true, Y_pred, target_names=target_names))
    ax = fig.add_subplot(1, 2, z)
    cax = ax.matshow(cm, cmap='YlGn', interpolation='nearest')  # plt.cm.Blues
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], fontsize=15)
    ax.set_title('Confusion matrix for {} with model = {}'.format(split_name, model_name), fontsize=12)
    ax.set_xticklabels([''] + target_names)
    ax.set_yticklabels([''] + target_names)
    ax.set_xlabel('Predicted', fontsize='large')
    ax.set_ylabel('True', fontsize='large')
    return ax


def read_mat(matfile):
    mat = loadmat(matfile)
    Xdata = mat['X']
    Ydata = mat['y']
    columns = ['feat-{}'.format(i) for i in range(1,Xdata.shape[1]+1)]
    df = pd.DataFrame(Xdata, columns=columns)
    df['label'] = Ydata
    return df


