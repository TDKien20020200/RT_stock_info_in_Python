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
import json

from sklearn.metrics import (explained_variance_score, r2_score, mean_squared_error, mean_absolute_error)

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
    n_epochs = 100
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

        # print("Epoch ", epoch + 1)
        # print(f'\tTrain Loss: {train_loss:.5f} | ' + f'\tVal Loss: {valid_loss:.5f}\n')
    print("epoch ", min_epoch, " có valid_loss = ", best_valid_loss, " có train_loss = ", min_train_loss)

    return best_valid_loss


def bestModel():
    modelUse = torch.load('saved_weights.pt')
    return modelUse


def resetSavedWeights():
    if os.path.exists('saved_weights.pt'):
        os.remove('saved_weights.pt')
    with open('saved_weights.pt', 'w') as f:
        pass


@app.route('/')
def index():
    return render_template('index.html')


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
            if file_date[0:4] != date:
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
    scaler = MinMaxScaler(feature_range=(0, 15))
    data2[['Close', 'Open', 'High', 'Low']] = scaler.fit_transform(data2[['Close', 'Open', 'High', 'Low']])
    dataUse = data2[['Open', 'High', 'Low', 'Close']].values

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
    modelUse = 0
    numModel = 0
    resetSavedWeights()

    while count < 16:
        print("Lần lặp", count)
        valid_loss = printTrainVal(train_dataloader, valid_dataloader)
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            modelUse = bestModel()
            numModel = count
        count = count + 1

    print("Với", count - 1, "lần chạy cho ", ticker, ", lần chạy thứ", numModel, "cho ra kết quả tối ưu nhất với "
                                                                                 "valid_loss =", best_valid_loss)
    # modelUse = bestModel()
    x_test = torch.tensor(x_test).float()
    with torch.no_grad():
        y_test_pred = modelUse(x_test)

    y_test_pred = y_test_pred.numpy()[0]

    y_test_array = np.ndarray.tolist(y_test[:, 0])
    y_test_pred_array = np.ndarray.tolist(y_test_pred[:, 0])

    y_test_array = json.loads(json.dumps(y_test_array))
    y_test_pred_array = json.loads(json.dumps(y_test_pred_array))

    y_test_array = [float(value) for value in y_test_array]
    y_test_pred_array = [float(value) for value in y_test_pred_array]
    # print(y_test_array)
    # print(y_test_pred_array)
    print("evs = ", explained_variance_score(y_test_array, y_test_pred_array))
    print("r2score = ", r2_score(y_test_array, y_test_pred_array))
    print("mse = ", mean_squared_error(y_test_array, y_test_pred_array))
    print("mae = ", mean_absolute_error(y_test_array, y_test_pred_array))

    idx = 0
    filename5 = f'prediction_chart_{date}_{ticker}.png'
    filepath5 = os.path.join('./static/image/', filename5)
    fileurl5 = url_for('static', filename=f'image/{filename5}')
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
    date_range = pd.bdate_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=PRED_DAYS, freq='C',
                                tz='America/New_York')

    data_pred = pd.DataFrame(
        data=pred_days,
        columns=['Open', 'High', 'Low', 'Close']
    )
    data_pred['Date'] = date_range
    col_close = data_pred.pop('Date')
    data_pred.insert(0, 'Date', col_close)
    pred_data_list = data_pred.reset_index().to_dict(orient='records')

    # Truyền dữ liệu cho template
    return render_template('ticker_detail.html', ticker=ticker, current_price=current_price, open_price=open_price,
                           volume=volume, data_list=data_list, chart_data=chart_data,
                           data_chart_predicted=data_chart_predicted, pred_data_list=pred_data_list,
                           fileurl1=fileurl1, fileurl2=fileurl2, fileurl3=fileurl3, fileurl4=fileurl4,
                           fileurl5=fileurl5)


@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.get_json()['ticker']
    data = yf.Ticker(ticker).history(period='1y')
    return jsonify({'currentPrice': data.iloc[-1].Close,
                    'openPrice': data.iloc[-1].Open})


if __name__ == '__main__':
    app.run(debug=True)
