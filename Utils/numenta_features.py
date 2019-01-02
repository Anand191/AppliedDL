import pandas as pd
import numpy as np
from collections import Counter


class calendar_features(object):
    '''
    This class receives the path to a csv file which is organized as timestamp, values and appends to it calender
    features viz. days of week, week of year etc. The added calendar features are categorical and are added either using
    dummy encoding (k-1 features to avoid multicollinearity) or one-hot encoding (all k features). If for ex the data
    doesn't span the entire space of the categorical variable then one-hot encoding is used by default.

    If at any point we wish to go back to the initial state of the data-frame (as read from csv) we just need to use
    the reset method.


    **Add hourly features, holidays and dummy/one-hot based on type of algorithms used. e.g. regression uses dummy
    xgboost uses one-hot.

    Args:
        path (str)(scalar) : path to the data file
        dummy (bool) : whether to use dummy encoding or one-hot encoding
    Attributes:
         ALL_DAYS (List): List of all days of week
         ALL_WEEKS (list): List of all the weeks in a year
         ALL_Months (List): 1-12
         ALL_QUARTERS (List): 1-4
         KEYS (dict): map strings to the above mentioned data structures
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
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self._dummyVsOnehot('day_of_week', 'days')
        self._makeCategorical('day_of_week', 'day')
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def add_week(self):
        self.df['week_of_year'] = self.df['timestamp'].dt.week
        self._dummyVsOnehot('week_of_year', 'weeks')
        self._makeCategorical('week_of_year', 'week')
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def add_month(self):
        self.df['month_of_year'] = self.df['timestamp'].dt.month
        self._dummyVsOnehot('month_of_year', 'months')
        self._makeCategorical('month_of_year', 'month')
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def add_quarter(self):
        self.df['quarter_of_year'] = self.df['timestamp'].dt.quarter
        self._dummyVsOnehot('quarter_of_year', 'quarters')
        self._makeCategorical('quarter_of_year', 'quarter')
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def reset(self):
        self.df = pd.read_csv(self.path, sep=',', index_col=0)
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))



class statistical_features(object):
    '''
    This class recieves the path to a csv file which is organized as timestamp, values and appends to it statistical
    features viz. moving averages, expanding averages, exponential smoothing etc.

    If at any point we wish to go back to the initial state of the data-frame (as read from csv) we just need to use
    the reset method.
    Args:
        path (str)(scalar): path to the csv file
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

        self.df.dropna(inplace=True)
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def n_stddevs(self,col_name):
        mean = np.mean(self.df[col_name].values)
        std = np.std(self.df[col_name].values)
        self.df['n_stddevs'] = self.df[col_name].apply(lambda x: np.absolute(x-mean)/std)
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def rolling(self, col_name, window, summary='mean'):
        #add win_type feature
        window = self.df[col_name].rolling(window=window, min_periods=1)
        statistic = getattr(window, summary)
        self.df['rolling_{}'.format(summary)] = statistic()
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def expanding(self, col_name, summary='mean', center=False):
        '''
        expanding transformation. e.g. with summary = mean, we get the mean of first 2 elements, then first 3
        then first 4 and so on.
        '''
        window = self.df[col_name].expanding(min_periods=1, center=center)
        statistic = getattr(window, summary)
        self.df['expanding_{}'.format(summary)] = statistic()
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def exponential_weighted(self, col_name, smoothing = 0.2, summary='mean'):
        window = self.df[col_name].ewm(alpha = smoothing)
        statistic = getattr(window, summary)
        self.df['exponential_{}'.format(summary)] = statistic()
        tgt_col = self.df.pop('anomaly')
        self.df['anomaly'] = tgt_col

    def reset(self):
        self.df = pd.read_csv(self.path, sep=',', index_col=0)
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: pd.to_datetime(x))






