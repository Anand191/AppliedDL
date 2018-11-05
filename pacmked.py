import pandas as pd
import numpy as np
import datetime

path = 'resources/pacemed/'

def convert_datetime(row):
    return (datetime.datetime.fromtimestamp(row))

df1 = pd.read_csv(path+'age.csv', index_col=None, sep=';')
df2 = pd.read_csv(path+'admission.csv', index_col=None, sep=';')
df3 = pd.read_csv(path+'signal.csv', index_col=None, sep=';')

df1.drop(['Unnamed: 0'], axis=1, inplace=True)
df2.drop(['Unnamed: 0'], axis=1, inplace=True)
df3.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

df_signal = df3.groupby(['pat_id', 'parameter']).mean()['value']
df_signal = df_signal.unstack().reset_index()
df_signal.columns.name = None


df2['date_admission'] = df2['date_admission'].astype('datetime64')
df2['date_discharge']= df2['date_discharge'].astype('datetime64')
df_admission = df2.groupby('pat_id')['date_admission'].agg('count').to_frame('Count').reset_index()
df_admission = pd.merge(df2, df_admission, on='pat_id')
print(df_admission)

