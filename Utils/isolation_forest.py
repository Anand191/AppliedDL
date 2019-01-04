import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.externals import joblib
from Utils.numenta_features import statistical_features, calendar_features

random_state = np.random.RandomState(42)

class isolationForest(object):
    '''

    '''
    def __init__(self, path, eval_mode = False,print_report=False,preprocess=True, preprocess_args={"add_statistical": True, "add_calendar": True,
                                                               "statistical":['lagged_cols'],"calendar":['add_weekdays']}):
        '''
        preprocess the data here
        :param path: path
        '''
        if isinstance(path, pd.DataFrame):
            self.df = path
        else:
            self.df = pd.read_csv(path, sep=',', index_col=0)
            self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))

        if preprocess:
            self._preprocess(**preprocess_args)

        self.report = print_report
        self.eval_mode = eval_mode
        self.accuracy = 0.
        self.roc_auc = 0.
        self.cm = 0.

    def train(self, test_split = 0.2, ensembleSize=5, sampleSize=500, save=True, save_path=None):
        clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01,
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0, behaviour="new")
        Data = self.df.iloc[:,1:-1 ].values
        Target = self.df.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Data, Target, test_size=test_split,
                                                            random_state=random_state, stratify=Target)
        Xidxs = list(range(0,Data.shape[0]))
        self.ensemble = []
        for n in range(ensembleSize):
            sample = np.random.choice(Xidxs,sampleSize)
            X = Data[sample,:]
            clf.fit(X)
            self.ensemble.append(clf)

        if save:
            self._save_model(save_path)

    def eval(self, loadpath=None):
        if self.eval_mode:
            if loadpath is None:
                raise ValueError('A path to saved model must be provided if eval mode is on')
            else:
                self.ensemble = []
                if isinstance(loadpath, list):
                    for lp in loadpath:
                        self.ensemble.append(joblib.load(lp))
                else:
                    self.ensemble.append(joblib.load(loadpath))
                self.X_test = self.df.iloc[:, 1:-1].values
                self.y_test = self.df.iloc[:, -1].values
                self._predict()
        else:
            self._predict()

    def plot(self, plot_data=False, plot_tsne=False, plot_conf=False):
        #Timeseries plot
        if plot_data:
            self.df.plot(x='timestamp', y='value')

        # TSNE Plot of data
        if plot_tsne:
            X_embedded, labels = self._tsne_data()
            plt.figure(figsize=(12, 8))
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap=plt.cm.get_cmap("Paired", 2))
            plt.colorbar(ticks=range(2))

        # Confusion Matrix
        if plot_conf:
            df_cm = pd.DataFrame(self.cm,
                                 ['Normal', 'Anomaly'], ['Pred Normal', 'Pred Anomaly'])
            plt.figure(figsize=(10,6))
            sns.set(font_scale=1.2)  # for label size
            sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='g')

        plt.show()

    def _predict(self):
        y_pred = np.zeros(self.X_test.shape[0])
        for clf in self.ensemble:
            y_pred = np.add(y_pred, clf.decision_function(self.X_test).reshape(self.X_test.shape[0], ))
        y_pred = (y_pred * 1.0) / len(self.ensemble)
        y_pred = 1 - y_pred

        self.y_pred_class = y_pred.copy()
        self.y_pred_class[y_pred >= np.percentile(y_pred, 95)] = 1
        self.y_pred_class[y_pred < np.percentile(y_pred, 95)] = 0
        self._get_report()

    def _preprocess(self, add_statistical, add_calendar, statistical, calendar):
        if add_statistical:
            add_stat_features = statistical_features(self.df)
            statistical_methods = [getattr(add_stat_features, methodname) for methodname in statistical]
            for method in statistical_methods:
                method()
            self.df = add_stat_features.df

        if add_calendar:
            add_cal_features = calendar_features(self.df)
            calnendar_methods = [getattr(add_cal_features, methodname) for methodname in calendar]
            for method in calnendar_methods:
                method()
            self.df = add_cal_features.df

    def _save_model(self, path):
        for i, model in enumerate(self.ensemble):
            fname = os.path.join(path, 'iFmodel_{}.sav'.format(i))
            joblib.dump(model, fname)

    def _get_report(self):
        self.roc_auc = roc_auc_score(self.y_test, self.y_pred_class)
        self.accuracy = accuracy_score(self.y_test, self.y_pred_class)
        self.cm = confusion_matrix(self.y_test, self.y_pred_class)
        if self.report:
            print(classification_report(self.y_test,self.y_pred_class))
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_class)
            auc_sc = auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % auc_sc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

    def _tsne_data(self):
        y_plt = self.df['anomaly'].values
        X_plt = self.df.iloc[:,1:-1].values
        return(TSNE(n_components=2).fit_transform(X_plt), y_plt)

# filepath = '../resources/Numenta/cleaned_data/window/realAdExchange-exchange-2_cpc_results.csv'
# savepath = '../resources/Misc./Isolation_Forest/'
# iF = isolationForest(filepath)
# iF.train(save=True, save_path=savepath)
# iF.eval()
# iF.plot(plot_conf=True, plot_data=True)

# filepath = '../resources/Numenta/cleaned_data/window/realAdExchange-exchange-2_cpm_results.csv'
# loadpath = ['../resources/Misc./Isolation_Forest/iFmodel_0.sav']
# iF = isolationForest(filepath,eval_mode=True, print_report=False)
# iF.eval(loadpath)
# iF.plot(plot_conf=True)
