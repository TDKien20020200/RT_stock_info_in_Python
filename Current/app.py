import yfinance as yf
from flask import request, render_template, jsonify, Flask, url_for
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd

import seaborn as sns

sns.set_style('whitegrid')
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from datetime import datetime
yf.pdr_override()
import os

from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__, template_folder='templates')


# Xây dựng model Xác định kiến trúc: vì là vấn đề về chuỗi thời gian nên sẽ sử dụng Long Short-term Memory (LSTM) để nắm
# bắt thông tin tuần tự:
class NeuralNetwork(nn.Module):
    def __init__(self, num_feature):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(num_feature, 64, batch_first=True)
        self.fc = nn.Linear(64, num_feature)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden)
        return x


# #############################################################################################################
model = NeuralNetwork(4)
# push to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Hàm Optimize Adam
optimizer = optim.Adam(model.parameters())
# Hàm Loss MSELoss
mse = nn.MSELoss()


# Mô hình Training: xác định quá trình thuận và nghịch (forward and backward) để train mạng lưới Neural:
def train(dataloader):
    epoch_loss = 0
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        pred = model(x)
        loss = mse(pred[0], y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss


# Đánh giá hiệu suất mô hình
def evaluate(dataloader):
    epoch_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            pred = model(x)
            loss = mse(pred[0], y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def printTrainVal(train_dataloader, valid_dataloader):
    n_epochs = 150
    best_valid_loss = float('inf')
    min_epoch = 0
    min_train_loss = 1000
    for epoch in range(1, n_epochs + 1):
        train_loss = train(train_dataloader)
        valid_loss = evaluate(valid_dataloader)

        # save the best model
        if valid_loss < best_valid_loss:
            min_epoch = epoch
            min_train_loss = train_loss
            best_valid_loss = valid_loss
            torch.save(model, 'saved_weights.pt')

        print("Epoch ", epoch + 1)
        print(f'\tTrain Loss: {train_loss:.5f} | ' + f'\tVal Loss: {valid_loss:.5f}\n')
    print("epoch ", min_epoch, " có valid_loss = ", best_valid_loss, " có train_loss = ", min_train_loss)

    return best_valid_loss


def bestModel():
    modelUse = torch.load('saved_weights.pt')
    return modelUse

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tickers/analysis')
def tickers_analysis():
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)
    data = {}

    now = datetime.now()
    date = now.strftime('%d') + now.strftime('%m')

    image_directory = './static/image/tickers/'
    for filename in os.listdir(image_directory):
        if filename.endswith('.png'):
            file_date = filename[-8:]
            if file_date[0:4] != date:
                file_path = os.path.join(image_directory, filename)
                os.remove(file_path)

    for stock in tech_list:
        data[stock] = yf.download(stock, start, end)
    company_list = tech_list
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

    for company, com_name in zip(company_list, company_name):
        data[company]['company_name'] = com_name

    df = pd.concat(data.values(), keys=data.keys(), axis=0)
    # Close chart
    filename1 = f'adj_close_chart_{date}.png'
    filepath1 = os.path.join('./static/image/tickers/', filename1)
    fileurl1 = url_for('static', filename=f'image/tickers/{filename1}')
    if not os.path.exists(filepath1):
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=1.25, bottom=1.2)
        for i, company in enumerate(company_list, 1):
            plt.subplot(2, 2, i)
            df.loc[company]['Adj Close'].plot()
            plt.ylabel('Adj Close')
            plt.xlabel(None)
            plt.title(f"Closing Price of {tech_list[i - 1]}")
        plt.tight_layout()
        plt.savefig(filepath1)
    # Volume chart
    filename2 = f'volume_chart_{date}.png'
    filepath2 = os.path.join('./static/image/tickers/', filename2)
    fileurl2 = url_for('static', filename=f'image/tickers/{filename2}')
    if not os.path.exists(filepath2):
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=1.25, bottom=1.2)
        for i, company in enumerate(company_list, 1):
            plt.subplot(2, 2, i)
            df.loc[company]['Volume'].plot()
            plt.ylabel('Volume')
            plt.xlabel(None)
            plt.title(f"Sales Volume for {tech_list[i - 1]}")
        plt.tight_layout()
        plt.savefig(filepath2)
    # MA (moving avage)
    ma_day = [10, 20, 50]
    for ma in ma_day:
        for company in company_list:
            column_name = f"MA for {ma} days"
            mean_value = df.loc[company, 'Adj Close'].rolling(ma).mean()
            df.loc[(company, slice(None)), column_name] = mean_value.values

    filename3 = f'MAs_chart_{date}.png'
    filepath3 = os.path.join('./static/image/tickers/', filename3)
    fileurl3 = url_for('static', filename=f'image/tickers/{filename3}')
    if not os.path.exists(filepath3):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        df.loc["AAPL"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[0, 0])
        axes[0, 0].set_title('APPLE')
        df.loc["GOOG"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[0, 1])
        axes[0, 1].set_title('GOOGLE')
        df.loc["MSFT"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[1, 0])
        axes[1, 0].set_title('MICROSOFT')
        df.loc["AMZN"].plot(y=['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=axes[1, 1])
        axes[1, 1].set_title('AMAZON')
        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[1, 0].legend()
        axes[1, 1].legend()
        fig.tight_layout()
        plt.savefig(filepath3)
    # Daily return
    for company in company_list:
        change_pct = df.loc[company, 'Adj Close'].pct_change()
        df.loc[(company, slice(None)), 'Daily Return'] = change_pct.values

    filename4 = f'PC_line_chart_{date}.png'
    filepath4 = os.path.join('./static/image/tickers/', filename4)
    fileurl4 = url_for('static', filename=f'image/tickers/{filename4}')
    if not os.path.exists(filepath4):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        df.loc["AAPL"]['Daily Return'].plot(ax=axes[0, 0], legend=True, linestyle='--', marker='o')
        axes[0, 0].set_title('APPLE')
        df.loc["GOOG"]['Daily Return'].plot(ax=axes[0, 1], legend=True, linestyle='--', marker='o')
        axes[0, 1].set_title('GOOGLE')
        df.loc["MSFT"]['Daily Return'].plot(ax=axes[1, 0], legend=True, linestyle='--', marker='o')
        axes[1, 0].set_title('MICROSOFT')
        df.loc["AMZN"]['Daily Return'].plot(ax=axes[1, 1], legend=True, linestyle='--', marker='o')
        axes[1, 1].set_title('AMAZON')
        fig.tight_layout()
        plt.savefig(filepath4)

    filename5 = f'PC_column_chart_{date}.png'
    filepath5 = os.path.join('./static/image/tickers/', filename5)
    fileurl5 = url_for('static', filename=f'image/tickers/{filename5}')
    if not os.path.exists(filepath5):
        plt.figure(figsize=(12, 9))
        for i, company in enumerate(company_list, 1):
            plt.subplot(2, 2, i)
            df.loc[company, 'Daily Return'].hist(bins=50)
            plt.xlabel('Daily Return')
            plt.ylabel('Counts')
            plt.title(f'{company_name[i - 1]}')
        plt.tight_layout()
        plt.savefig(filepath5)
    # Correlation of adj close prices
    closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']
    tech_rets = closing_df.pct_change()

    filename6 = f'comparations_visual_analysis_chart_{date}.png'
    filepath6 = os.path.join('./static/image/tickers/', filename6)
    fileurl6 = url_for('static', filename=f'image/tickers/{filename6}')
    if not os.path.exists(filepath6):
        sns.pairplot(tech_rets, kind='reg')
        plt.savefig(filepath6)

    filename7 = f'comparations_daily_return_chart_{date}.png'
    filepath7 = os.path.join('./static/image/tickers/', filename7)
    fileurl7 = url_for('static', filename=f'image/tickers/{filename7}')
    if not os.path.exists(filepath7):
        return_fig = sns.PairGrid(tech_rets.dropna())
        return_fig.map_upper(plt.scatter, color='purple')
        return_fig.map_lower(sns.kdeplot, cmap='cool_d')
        return_fig.map_diag(plt.hist, bins=30)
        plt.savefig(filepath7)

    filename8 = f'comparations_close_chart_{date}.png'
    filepath8 = os.path.join('./static/image/tickers/', filename8)
    fileurl8 = url_for('static', filename=f'image/tickers/{filename8}')
    if not os.path.exists(filepath8):
        returns_fig = sns.PairGrid(closing_df)
        returns_fig.map_upper(plt.scatter, color='purple')
        returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
        returns_fig.map_diag(plt.hist, bins=30)
        plt.savefig(filepath8)

    filename9 = f'comparations_correlation_chart_{date}.png'
    filepath9 = os.path.join('./static/image/tickers/', filename9)
    fileurl9 = url_for('static', filename=f'image/tickers/{filename9}')
    if not os.path.exists(filepath9):
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
        plt.title('Correlation of stock return')
        plt.subplot(2, 2, 2)
        sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
        plt.title('Correlation of stock closing price')
        plt.savefig(filepath9)
    # Risk
    filename10 = f'risks_chart_{date}.png'
    filepath10 = os.path.join('./static/image/tickers/', filename10)
    fileurl10 = url_for('static', filename=f'image/tickers/{filename10}')
    if not os.path.exists(filepath10):
        rets = tech_rets.dropna()
        area = np.pi * 20
        plt.figure(figsize=(10, 8))
        plt.scatter(rets.mean(), rets.std(), s=area)
        plt.xlabel('Expected return')
        plt.ylabel('Risk')
        for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
            plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
                         arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
        plt.savefig(filepath10)

    return render_template('tickers_analysis.html', tech_list=tech_list, fileurl1=fileurl1, fileurl2=fileurl2,
                           fileurl3=fileurl3, fileurl4=fileurl4, fileurl5=fileurl5, fileurl6=fileurl6, fileurl7=fileurl7,
                           fileurl8=fileurl8, fileurl9=fileurl9, fileurl10=fileurl10)


@app.route('/ticker/<ticker>')
def ticker_detail(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    dataYears = yf.Ticker(ticker).history(period='7y')
    current_price = data.iloc[-1].Close
    open_price = data.iloc[-1].Open
    volume = data.iloc[-1].Volume
    chart_data = {
        'x': data.index.strftime('%Y-%m-%d').tolist(),
        'open': data['Open'].tolist(),
        'high': data['High'].tolist(),
        'low': data['Low'].tolist(),
        'close': data['Close'].tolist()
    }
    now = datetime.now()
    date = now.strftime('%d') + now.strftime('%m')
    image_directory = './static/image/'
    for filename in os.listdir(image_directory):
        if filename.endswith('.png'):
            file_date = filename[-13:]
            if file_date[0:4] != date or file_date[6:10] != ticker:
                file_path = os.path.join(image_directory, filename)
                os.remove(file_path)

    # Filter and evaluate data
    # Daily Chart
    filename1 = f'daily_chart_{date}_{ticker}.png'
    filepath1 = os.path.join('./static/image/', filename1)
    fileurl1 = url_for('static', filename=f'image/{filename1}')
    if not os.path.exists(filepath1):
        data.plot(subplots=True, figsize=(14.5, 6.5))
        plt.title('Daily Chart', y=8.2)
        plt.savefig(filepath1)
    # Weekly Chart
    filename2 = f'weekly_chart_{date}_{ticker}.png'
    filepath2 = os.path.join('./static/image/', filename2)
    fileurl2 = url_for('static', filename=f'image/{filename2}')
    if not os.path.exists(filepath2):
        data.asfreq('W', method='ffill').plot(subplots=True, figsize=(14.5, 6.5), style='-')
        plt.title('Weekly Chart', y=8.2)
        plt.savefig(filepath2)

    # Calculate MAs
    ma_day = [10, 20, 50]
    for ma in ma_day:
        col_name = f'MA for {ma} days'
        data[col_name] = data['Close'].rolling(ma).mean()
    filename3 = f'MAs_chart_{date}_{ticker}.png'
    filepath3 = os.path.join('./static/image/', filename3)
    fileurl3 = url_for('static', filename=f'image/{filename3}')
    if not os.path.exists(filepath3):
        data[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(14.5, 6.5))
        plt.title('Comparison of Moving Averages and Close')
        plt.savefig(filepath3)

    # Dùng hàm pct_change() để tìm phần trăm thay đổi của giá Close mỗi ngày
    data['Daily_Return'] = data['Close'].pct_change()
    filename4 = f'daily_return_chart_{date}_{ticker}.png'
    filepath4 = os.path.join('./static/image/', filename4)
    fileurl4 = url_for('static', filename=f'image/{filename4}')
    if not os.path.exists(filepath4):
        data[['Daily_Return']].plot(legend=True, figsize=(14.5, 6.5))
        plt.title('Daily Return Percentage')
        plt.savefig(filepath4)

    ###############################################################################################################
    # Tạo dataset theo batch size trong pytorch
    # Chuẩn hóa dữ liệu
    data2 = dataYears.copy(deep=True)
    scaler = MinMaxScaler(feature_range=(0, 15)).fit(data2['Low'].values.reshape(-1, 1))
    data2['Close'] = scaler.transform(data2['Close'].values.reshape(-1, 1))
    data2['Open'] = scaler.transform(data2['Open'].values.reshape(-1, 1))
    data2['High'] = scaler.transform(data2['High'].values.reshape(-1, 1))
    data2['Low'] = scaler.transform(data2['Low'].values.reshape(-1, 1))
    dataUse = data2[['Open', 'High', 'Low', 'Close']].values
    # print(dataUse.shape)
    # print(dataUse)

    # chuẩn bị dữ liệu cho bài toán dự đoán giá cổ phiếu dựa trên giá cổ phiếu từ 10 ngày trước để dự
    # đoán giá cổ phiếu vào ngày tiếp theo
    seq_len = 11
    sequences = []
    for index in range(len(dataUse) - seq_len + 1):
        sequences.append(dataUse[index: index + seq_len])
    sequences = np.array(sequences)

    # Customize dataset
    # Tách dữ liệu toàn bộ tập dữ liệu thành ba phần. 80% cho tập huấn luyện (train), 10% cho tập xác thực (valid) và 10% còn lại cho tập kiểm thử (test):
    valid_set_size_percentage = 10
    test_set_size_percentage = 10
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * sequences.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * sequences.shape[0]))
    train_set_size = sequences.shape[0] - (valid_set_size + test_set_size)
    x_train = sequences[:train_set_size, :-1, :]
    y_train = sequences[:train_set_size, -1, :]
    x_valid = sequences[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = sequences[train_set_size:train_set_size + valid_set_size, -1, :]
    x_test = sequences[train_set_size + test_set_size:, :-1, :]
    y_test = sequences[train_set_size + test_set_size:, -1, :]
    # DataLoader
    # Tạo Trình tải dữ liệu: xác định các trình tải dữ liệu để tải tập dữ liệu theo từng batch với batch size = 32
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_valid = torch.tensor(x_valid).float()
    y_valid = torch.tensor(y_valid).float()

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    valid_dataset = TensorDataset(x_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Mô hình huấn luyện
    best_valid_loss = 1000
    count = 1
    while count < 15:
        print("Lần lặp", count)
        valid_loss = printTrainVal(train_dataloader, valid_dataloader)
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
        count = count + 1
        if best_valid_loss < valid_loss:
            break
    # for i in range(1, 10):
    #     printTrainVal(train_dataloader, valid_dataloader)

    modelUse = bestModel()
    x_test = torch.tensor(x_test).float()
    with torch.no_grad():
        y_test_pred = modelUse(x_test)

    y_test_pred = y_test_pred.numpy()[0]

    idx = 0
    filename5 = f'prediction_chart_{date}_{ticker}.png'
    filepath5 = os.path.join('./static/image/', filename5)
    fileurl5 = url_for('static', filename=f'image/{filename5}')
    if not os.path.exists(filepath5):
        plt.figure(figsize=(14.5, 6.5))
        plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
                 y_test[:, idx], color='black', label='test target')

        plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test_pred.shape[0]),
                 y_test_pred[:, idx], color='green', label='test prediction')

        plt.title('Predicted stock prices')
        plt.xlabel('time [days]')
        plt.ylabel('normalized price')
        plt.legend(loc='best')
        plt.savefig(filepath5)

    dataShowHis = data.tail(30)
    data_list = dataShowHis.reset_index().to_dict(orient='records')

    index_values = dataYears[len(dataYears) - len(y_test):].index
    col_values = ['Open', 'Low', 'High', 'Close']
    dataYears_predicted = pd.DataFrame(data=y_test_pred, index=index_values, columns=col_values)

    data_chart_predicted = {
        'x': dataYears_predicted.index.strftime('%Y-%m-%d').tolist(),
        'open': dataYears_predicted['Open'].tolist(),
        'high': dataYears_predicted['High'].tolist(),
        'low': dataYears_predicted['Low'].tolist(),
        'close': dataYears_predicted['Close'].tolist()
    }

    # Dự đoán 10 ngày tiếp theo
    # Get the last sequence of historical data as features for predicting the next 10 days
    last_sequence = sequences[-1:, 1:, :]
    last_sequence = torch.from_numpy(last_sequence).float()

    # Generate predictions for the next 10 days
    PRED_DAYS = 10
    with torch.no_grad():
        for i in range(PRED_DAYS):
            pred_i = modelUse(last_sequence)
            last_sequence = torch.cat((last_sequence, pred_i), dim=1)
            last_sequence = last_sequence[:, 1:, :]
    pred_days = last_sequence.reshape(PRED_DAYS, 4).numpy()
    # inverse transform the predicted values
    pred_days = scaler.inverse_transform(pred_days)
    date_range = pd.bdate_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=PRED_DAYS, freq='C', tz='America/New_York')

    data_pred = pd.DataFrame(
        data=pred_days,
        columns=['Open', 'High', 'Low', 'Close']
    )
    data_pred['Date'] = date_range
    col_close = data_pred.pop('Date')
    data_pred.insert(0, 'Date', col_close)
    pred_data_list = data_pred.reset_index().to_dict(orient='records')

    # # LSTM2
    # # Get the stock quote
    # df = pdr.get_data_yahoo(ticker, start='2012-01-01', end=datetime.now())
    # # Create a new dataframe with only the 'Close column
    # data = df.filter(['Close'])
    # # Convert the dataframe to a numpy array
    # dataset = data.values
    # # Get the number of rows to train the model on
    # training_data_len = int(np.ceil(len(dataset) * .95))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(dataset)
    # # Create the training data set
    # # Create the scaled training data set
    # train_data = scaled_data[0:int(training_data_len), :]
    # # Split the data into x_train and y_train data sets
    # x_train = []
    # y_train = []
    # for i in range(60, len(train_data)):
    #     x_train.append(train_data[i - 60:i, 0])
    #     y_train.append(train_data[i, 0])
    #     # if i <= 61:
    #     #     print(x_train)
    #     #     print(y_train)
    #     #     print()
    # # Convert the x_train and y_train to numpy arrays
    # x_train, y_train = np.array(x_train), np.array(y_train)
    # # Reshape the data
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #
    # count = 1
    # minRmse = 20
    # predictionsUse = 0
    # while count < 5:
    #     # Build the LSTM model
    #     model = Sequential()
    #     model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    #     model.add(LSTM(64, return_sequences=False))
    #     model.add(Dense(25))
    #     model.add(Dense(1))
    #     # Compile the model
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     # Train the model
    #     model.fit(x_train, y_train, batch_size=1, epochs=1)
    #     # Create the testing data set
    #     # Create a new array containing scaled values
    #     test_data = scaled_data[training_data_len - 60:, :]
    #     # Create the data sets x_test and y_test
    #     x_test = []
    #     y_test = dataset[training_data_len:, :]
    #     for i in range(60, len(test_data)):
    #         x_test.append(test_data[i - 60:i, 0])
    #     # Convert the data to a numpy array
    #     x_test = np.array(x_test)
    #     # Reshape the data
    #     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #     # Get the models predicted price values
    #     predictions = model.predict(x_test)
    #     predictions = scaler.inverse_transform(predictions)
    #     # Get the root mean squared error (RMSE)
    #     rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    #     if rmse < minRmse:
    #         minRmse = rmse
    #         predictionsUse = predictions
    #     print("lần ", count, ": rmse= ", rmse)
    #     count = count + 1
    #
    # # Plot the data
    # train = data[:training_data_len]
    # valid = data[training_data_len:]
    # valid['Predictions'] = predictionsUse
    # print(valid)
    # # Visualize the data
    # filename6 = f'prediction_chart_lstm2_{date}_{ticker}.png'
    # filepath6 = os.path.join('./static/image/', filename6)
    # fileurl6 = url_for('static', filename=f'image/{filename6}')
    # if not os.path.exists(filepath6):
    #     plt.figure(figsize=(14.5, 6.5))
    #     plt.title('Model')
    #     plt.xlabel('Date', fontsize=18)
    #     plt.ylabel('Close Price USD ($)', fontsize=18)
    #     plt.plot(train['Close'])
    #     plt.plot(valid[['Close', 'Predictions']])
    #     plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    #     plt.savefig(filepath6)

    # Truyền dữ liệu cho template
    return render_template('ticker_detail.html', ticker=ticker, current_price=current_price, open_price=open_price,
                           volume=volume, data_list=data_list, chart_data=chart_data,
                           data_chart_predicted=data_chart_predicted, pred_data_list=pred_data_list,
                           fileurl1=fileurl1, fileurl2=fileurl2, fileurl3=fileurl3, fileurl4=fileurl4, fileurl5=fileurl5)


@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.get_json()['ticker']
    data = yf.Ticker(ticker).history(period='1y')
    return jsonify({'currentPrice': data.iloc[-1].Close,
                    'openPrice': data.iloc[-1].Open})


if __name__ == '__main__':
    app.run(debug=True)
