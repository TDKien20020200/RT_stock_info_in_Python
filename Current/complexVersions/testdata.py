import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
sns.set_style('whitegrid')

dataYears = yf.download("AAPL", period='7y')
print("dataYears:")
print(dataYears.shape)
print(dataYears)

data2 = dataYears.copy(deep=True)
print("data2:")
print(data2.shape)
print(data2)

scaler = MinMaxScaler(feature_range=(0, 15))
data2[['Close', 'Open', 'High', 'Low']] = scaler.fit_transform(data2[['Close', 'Open', 'High', 'Low']])
print("After data2:")
print(data2.shape)
print(data2)

dataUse = data2[['Open', 'High', 'Low', 'Close']].values
print("dataUse:")
print(dataUse.shape)
print(dataUse)