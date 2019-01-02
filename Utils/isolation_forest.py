import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix,f1_score
from sklearn.externals import joblib
from .numenta_features import statistical_features

class isolationForest(object):
    '''

    '''
    def __init__(self, path, preprocess=True):
        '''
        preprocess the data here
        :param data: path
        '''
        self.path = path
        if preprocess:
            self._preprocess()
        else:
            self.df = pd.read_csv(self.path, sep=',', index_col=0)
            self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))

        self.clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, \
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0, behaviour="new")

        self.ensemble = []
        self. accuracy = 0.
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.roc_auc = 0.

    def _preprocess(self):
        add_features = statistical_features(self.path)
        add_features.lagged_cols('value', 5)
        self.df = add_features.df

    def train(self, ensembleSize=5, sampleSize=500):
        mdlLst = []
        for n in range(ensembleSize):
            X = df_data.sample(sampleSize)
            clf.fit(X)
            mdlLst.append(clf)
        return mdlLst

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for clf in mdlLst:
            y_pred = np.add(y_pred, clf.decision_function(X).reshape(X.shape[0], ))
        y_pred = (y_pred * 1.0) / len(mdlLst)
        return y_pred

    def save_model(self):
        raise NotImplementedError

    def load_predict(self):
        raise NotImplementedError

    def confusion_matrix(self):
        raise NotImplementedError

    def ROC_curve(self):
        raise NotImplementedError



