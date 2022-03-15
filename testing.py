import pandas as pd

df = pd.read_csv('gps_data/preprocessed_data/plant2-plant1/2021-09.csv')
df['start_plant2']  = pd.to_datetime(df['start_plant2'])
print(df.info())