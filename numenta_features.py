import pandas as pd

df = pd.read_csv('resources/Numenta/cleaned_data/window/realAdExchange-exchange-2_cpc_results.csv', sep=',', index_col=0)
df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x))
print(df['timestamp'].iloc[1].strftime('%A'))