import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year - 2, end.month, end.day)

data = {}

for stock in tech_list:
    data[stock] = yf.download(stock, start, end)

company_list = tech_list  # Use tech_list as company_list
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    data[company]['company_name'] = com_name

df = pd.concat(data.values(), keys=data.keys(), axis=0)

####################################################################################################
# Thống kê:

# print(df.tail(50))

# # Summary Stats
# print(df.loc["AAPL"].describe())  # Use df.loc["AAPL"] to access the AAPL DataFrame

# print(df.loc["AAPL"].info())

# # Giá đóng cửa
# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(top=1.25, bottom=1.2)
#
# for i, company in enumerate(company_list, 1):
#     plt.subplot(2, 2, i)
#     df.loc[company]['Adj Close'].plot()
#     plt.ylabel('Adj Close')
#     plt.xlabel(None)
#     plt.title(f"Closing Price of {tech_list[i - 1]}")
#
# plt.tight_layout()
# plt.show()

# # Tổng khối lượng cổ phiếu được giao dịch mỗi ngày
# plt.figure(figsize=(15, 10))
# plt.subplots_adjust(top=1.25, bottom=1.2)
#
# for i, company in enumerate(company_list, 1):
#     plt.subplot(2, 2, i)
#     df.loc[company]['Volume'].plot()
#     plt.ylabel('Volume')
#     plt.xlabel(None)
#     plt.title(f"Sales Volume for {tech_list[i - 1]}")
#
# plt.tight_layout()
# plt.show()

####################################################################################################
# Phân tích:

# Sử dụng đường MA (moving avage) (đường trung bình)
ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        mean_value = df.loc[company, 'Adj Close'].rolling(ma).mean()
        df.loc[(company, slice(None)), column_name] = mean_value.values

# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_figheight(10)
# fig.set_figwidth(15)
#
# df.loc["AAPL"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[0, 0])
# axes[0, 0].set_title('APPLE')
#
# df.loc["GOOG"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[0, 1])
# axes[0, 1].set_title('GOOGLE')
#
# df.loc["MSFT"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[1, 0])
# axes[1, 0].set_title('MICROSOFT')
#
# df.loc["AMZN"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[1, 1])
# axes[1, 1].set_title('AMAZON')
#
# # Add the legend for each subplot
# axes[0, 0].legend()
# axes[0, 1].legend()
# axes[1, 0].legend()
# axes[1, 1].legend()
#
# fig.tight_layout()
# # plt.show()

####################################################################################################
# Phân tích Price (%tăng hoặc % giảm mỗi ngày):
# We'll use pct_change to find the percent change for each day
for company in company_list:
    # df.loc[(company, slice(None)), 'Daily Return'] = df.loc[company, 'Adj Close'].pct_change()
    change_pct = df.loc[company, 'Adj Close'].pct_change()
    df.loc[(company, slice(None)), 'Daily Return'] = change_pct.values

# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_figheight(10)
# fig.set_figwidth(15)
#
# df.loc["AAPL"]['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
# axes[0,0].set_title('APPLE')
#
# df.loc["GOOG"]['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
# axes[0,1].set_title('GOOGLE')
#
# df.loc["MSFT"]['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
# axes[1,0].set_title('MICROSOFT')
#
# df.loc["AMZN"]['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
# axes[1,1].set_title('AMAZON')
#
# fig.tight_layout()
# # plt.show()

# Bằng biểu đồ cột
# plt.figure(figsize=(12, 9))
#
# for i, company in enumerate(company_list, 1):
#     plt.subplot(2, 2, i)
#     df.loc[company, 'Daily Return'].hist(bins=50)
#     plt.xlabel('Daily Return')
#     plt.ylabel('Counts')
#     plt.title(f'{company_name[i - 1]}')
#
# plt.tight_layout()
# # plt.show()

###################################################################################################
# Phân tích sự tương quan giữa các giá trị Adj Close
# Lưu giá trị Adj Close của 4 công ty thằng 1 data frame riêng
closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
# print(tech_rets.head(50))

# # Comparing Google to itself should show a perfectly linear relationship
# sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')
# plt.show()
#
# # use joinplot to compare the daily returns of Google and Microsoft
# sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')
# plt.show()
#
# # use pairplot for automatic visual analysis of all the comparisons
# sns.pairplot(tech_rets, kind='reg')
# plt.show()

# # Sett up our figure by naming it returns_fig, call PairPLot on the DataFrame
# return_fig = sns.PairGrid(tech_rets.dropna())
# # Using map_upper we can specify what the upper triangle will look like.
# return_fig.map_upper(plt.scatter, color='purple')
# # We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
# return_fig.map_lower(sns.kdeplot, cmap='cool_d')
# # Finally we'll define the diagonal as a series of histogram plots of the daily return
# return_fig.map_diag(plt.hist, bins=30)
# plt.show()

# # Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
# returns_fig = sns.PairGrid(closing_df)
# # Using map_upper we can specify what the upper triangle will look like.
# returns_fig.map_upper(plt.scatter,color='purple')
# # We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
# returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
# # Finally we'll define the diagonal as a series of histogram plots of the daily return
# returns_fig.map_diag(plt.hist,bins=30)
# plt.show()

# plt.figure(figsize=(12, 10))
#
# plt.subplot(2, 2, 1)
# sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
# plt.title('Correlation of stock return')
#
# plt.subplot(2, 2, 2)
# sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
# plt.title('Correlation of stock closing price')
# plt.show()

###############################################################################################
# # Rủi ro với từng công ty
# rets = tech_rets.dropna()
#
# area = np.pi * 20
#
# plt.figure(figsize=(10, 8))
# plt.scatter(rets.mean(), rets.std(), s=area)
# plt.xlabel('Expected return')
# plt.ylabel('Risk')
#
# for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#     plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
#                  arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
#
# plt.show()

###############################################################################################
# dự đoán
# Get the stock quote
df = pdr.get_data_yahoo('AAPL', start='2022-01-01', end=datetime.now())

# Bieu dien Close
# plt.figure(figsize=(16,6))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

# print(training_data_len)

# Quy mo/Mo hinh hoa du lieu
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# print(scaled_data)


# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    # if i <= 61:
    #     print(x_train)
    #     print(y_train)
    #     print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))




# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
# print(rmse)


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
print(valid)

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

