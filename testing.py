import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

df = pd.read_csv('merged_data/train.csv')
plot_pacf(df['travel_time'])

