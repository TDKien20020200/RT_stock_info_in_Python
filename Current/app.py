import yfinance as yf
from flask import request, render_template, jsonify, Flask
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
    n_epochs = 50
    best_valid_loss = float('inf')

    for epoch in range(1, n_epochs + 1):

        train_loss = train(train_dataloader)
        valid_loss = evaluate(valid_dataloader)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, 'saved_weights.pt')

        print("Epoch ", epoch + 1)
        print(f'\tTrain Loss: {train_loss:.5f} | ' + f'\tVal Loss: {valid_loss:.5f}\n')


def bestModel():
    modelUse = torch.load('saved_weights.pt')
    return modelUse

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tickers/analysis')
def tickers_analysis():
    return render_template('tickers_analysis.html')


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

    # Filter and evaluate data
    # Daily Chart
    data.plot(subplots=True, figsize=(14.5, 6.5))
    plt.title('Daily Chart', y=8.2)
    plt.savefig('./static/image/daily_chart.png')
    # Weekly Chart
    data.asfreq('W', method='ffill').plot(subplots=True, figsize=(14.5, 6.5), style='-')
    plt.title('Weekly Chart', y=8.2)
    plt.savefig('./static/image/weekly_chart.png')

    # Calculate MAs
    ma_day = [10, 20, 50]
    for ma in ma_day:
        col_name = f'MA for {ma} days'
        data[col_name] = data['Close'].rolling(ma).mean()
    data[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(14.5, 6.5))
    plt.title('Comparison of Moving Averages and Close')
    plt.savefig('./static/image/MAs_chart.png')

    # Dùng hàm pct_change() để tìm phần trăm thay đổi của giá Close mỗi ngày
    data['Daily_Return'] = data['Close'].pct_change()
    data[['Daily_Return']].plot(legend=True, figsize=(14.5, 6.5))
    plt.title('Daily Return Percentage')
    plt.savefig('./static/image/daily_return_chart.png')

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
    printTrainVal(train_dataloader, valid_dataloader)

    modelUse = bestModel()

    x_test = torch.tensor(x_test).float()

    with torch.no_grad():
        y_test_pred = modelUse(x_test)

    y_test_pred = y_test_pred.numpy()[0]

    idx = 0
    plt.figure(figsize=(14.5, 6.5))
    plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
             y_test[:, idx], color='black', label='test target')

    plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test_pred.shape[0]),
             y_test_pred[:, idx], color='green', label='test prediction')

    plt.title('Future stock prices')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price')
    plt.legend(loc='best')
    plt.savefig('./static/image/prediction_chart.png')

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
    print(pred_data_list)

    # Truyền dữ liệu cho template
    return render_template('ticker_detail.html', ticker=ticker, current_price=current_price, open_price=open_price,
                           volume=volume, data_list=data_list, chart_data=chart_data,
                           data_chart_predicted=data_chart_predicted, pred_data_list=pred_data_list)


@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.get_json()['ticker']
    data = yf.Ticker(ticker).history(period='1y')
    return jsonify({'currentPrice': data.iloc[-1].Close,
                    'openPrice': data.iloc[-1].Open})


if __name__ == '__main__':
    app.run(debug=True)
