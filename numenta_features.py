import pandas as pd
import numpy as np
from collections import Counter


class calendar_features(object):
    '''

    '''

    ALL_DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    ALL_WEEKS = list(range(1,53))
    ALL_MONTHS = list(range(1,13))
    ALL_QUARTERS = list(range(1,5))
    KEYS = {'days':ALL_DAYS, 'weeks':ALL_WEEKS, 'months':ALL_MONTHS, 'quarters':ALL_QUARTERS}

    def __init__(self, path, dummy=True):
        self.dummy = dummy
        self.path = path
        self.reset()

    def _makeCategorical(self, col_name, prefix):
        self.df = pd.get_dummies(self.df, prefix=[prefix], columns=[col_name], drop_first=self.dummy)

    def _dummyVsOnehot(self,col_name, key):
        if not Counter(np.unique(self.df[col_name].values).tolist()) == Counter(calendar_features.KEYS[key]):
            self.dummy = False

    def add_weekdays(self):
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self._dummyVsOnehot('day_of_week', 'days')
        self._makeCategorical('day_of_week', 'day')

    def add_week(self):
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))
        self.df['week_of_year'] = self.df['timestamp'].dt.week
        self._dummyVsOnehot('week_of_year', 'weeks')
        self._makeCategorical('week_of_year', 'week')

    def add_month(self):
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))
        self.df['month_of_year'] = self.df['timestamp'].dt.month
        self._dummyVsOnehot('month_of_year', 'months')
        self._makeCategorical('month_of_year', 'month')

    def add_quarter(self):
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))
        self.df['quarter_of_year'] = self.df['timestamp'].dt.quarter
        self._dummyVsOnehot('quarter_of_year', 'quarters')
        self._makeCategorical('quarter_of_year', 'quarter')

    def reset(self):
        self.df = pd.read_csv(self.path, sep=',', index_col=0)



class statistical_features(object):
    '''

    '''
    def __init__(self, path):
        self.path = path
        self.reset()

    def lagged_cols(self, col_name,num_lags, all_lags=True):
        if all_lags:
            for lag in range(1, num_lags+1):
                self.df['lag_{}'.format(lag)] = self.df[col_name].shift(lag)
        else:
            self.df['lag_{}'.format(num_lags)] = self.df[col_name].shift(num_lags)

    def n_stddevs(self,col_name):
        mean = np.mean(self.df[col_name].values)
        std = np.std(self.df[col_name].values)
        self.df['n_stddevs'] = self.df[col_name].apply(lambda x: np.absolute(x-mean)/std)

    def reset(self):
        self.df = pd.read_csv(self.path, sep=',', index_col=0)





