import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
sns.set_style('whitegrid')

dataYears = yf.download("AAPL", period='7y')
print("dataYears:")
print(dataYears.shape)
print(dataYears)

with open('saved_weights.pt', 'w') as f:
    pass