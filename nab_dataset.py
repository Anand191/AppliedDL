import pandas as pd
import numpy as np
import json
import os
from collections import Sequence


class add_labels(object):
    '''
    This class receives the path(s) to comma/tab/semicolon/space delimited file(s), which are organized as timestamp
    and value and enriches this dataframe with an additional column of anomaly labels. The anomaly labels are passed as
    json file(s).

    Args:
        files (str)(scalar/collections.Sequence/np.ndarray) : path to the data files
        label_file (str)(scalar/collections.Sequence/np.ndarray) : path to label json files
        filetype (str) : csv/tsv/space/semicolon
        window (bool) : whether to label individual anomalies or all points given a time window.
    Attributes:
         DELIMS (dict) : delimiter mapping filetype to the actual delimiter
    '''

    DELIMS = {'csv':',' , 'tsv': '\t', 'space': ' ', 'semicolon': ';'}

    def __init__(self, files, label_file, filetype, window=False):
        self.files = files
        self.labels = label_file
        self.ftype = filetype
        self.window = window

    def _format_datetime(self, timestamp):
        return(timestamp.split('.')[0])

    def _windowless(self, data, anomaly_labels):
        rows = list(data.loc[data['timestamp'].isin(anomaly_labels)].index)
        data['anomaly'].iloc[rows] = 1
        return data

    def _label_window(self, data, anomaly_labels):
        if not any(isinstance(el, Sequence) for el in anomaly_labels):
            window_start = data.loc[data['timestamp'] == self._format_datetime(anomaly_labels[0])].index.values[0]
            window_end = data.loc[data['timestamp'] == self._format_datetime(anomaly_labels[1])].index.values[0]
            rows = list(range(window_start, window_end + 1))
            data['anomaly'].iloc[rows] = 1
        else:
            for alabel in anomaly_labels:
                window_start = data.loc[data['timestamp'] == self._format_datetime(alabel[0])].index.values[0]
                window_end = data.loc[data['timestamp'] == self._format_datetime(alabel[1])].index.values[0]
                rows = list(range(window_start, window_end+1))
                data['anomaly'].iloc[rows] = 1
        return data


    def _preprocess(self, fpath, label_path):
        df = pd.read_csv(fpath, sep=add_labels.DELIMS[self.ftype], index_col=None)
        df['anomaly'] = [0]*df.shape[0]
        dir = os.path.dirname(fpath).split('/')[-1]
        fname = os.path.split(fpath)[-1]
        key = dir+'/'+fname
        with open(label_path) as f:
            json_labels = json.load(f)
            anomaly_timestamp = json_labels[key]

        if self.window:
            df = self._label_window(df, anomaly_timestamp)
        else:
            df = self._windowless(df, anomaly_timestamp)

        return df


    def process(self):
        if not isinstance(self.files, (Sequence, np.ndarray)) :
            final_df = self._preprocess(self.files, self.labels)
        else:
            final_df = [self._preprocess(f, self.labels) for f in self.files]

        return final_df