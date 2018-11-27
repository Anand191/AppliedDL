from nab_dataset import add_labels
import os
import json
import pandas as pd
from numenta_features import calendar_features

# data_files = os.listdir('./resources/Numenta/data/realAdExchange')
# file_paths = [os.path.join('./resources/Numenta/data/realAdExchange',f) for f in data_files]
# label_path = './resources/Numenta/labels/combined_windows.json'
# with open(label_path) as f:
#     json_labels = json.load(f)
#
#
#
# columns = ['timestamp', 'value', 'anomaly']
# destination = './resources/Numenta/cleaned_data/window'
# if not os.path.exists(destination):
#     os.makedirs(destination)
# process_data = add_labels(file_paths, label_path, 'csv', window=True)
# all_dataframes = process_data.process()
# for i, df in enumerate(all_dataframes):
#     # print(data_files[i])
#     # print(json_labels['realAdExchange'+'/'+data_files[i]])
#     # print(df.loc[df['anomaly'] == 1])
#     dirname = os.path.dirname(file_paths[i]).split('/')[-1]
#     df.to_csv(os.path.join(destination, "{}-{}".format(dirname,data_files[i])), sep=',', index_label=None, columns=columns)

path = 'resources/Numenta/cleaned_data/window/realAdExchange-exchange-2_cpc_results.csv'
add_calender_features = calendar_features(path)
print(add_calender_features.df.columns)

add_calender_features.add_weekdays()
print(add_calender_features.df.columns)
add_calender_features.reset()

add_calender_features.add_week()
print(add_calender_features.df.columns)
add_calender_features.reset()

add_calender_features.add_month()
print(add_calender_features.df.columns)
add_calender_features.reset()

add_calender_features.add_quarter()
print(add_calender_features.df.columns)
add_calender_features.reset()
