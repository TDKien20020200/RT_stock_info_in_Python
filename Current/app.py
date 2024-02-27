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

    # Filter and evaluate data
    # Daily Chart
    data.plot(subplots=True, figsize=(14.5, 6.5))
    plt.savefig('./static/image/daily_chart.png')
    # Weekly Chart
    data.asfreq('W', method='ffill').plot(subplots=True, figsize=(14.5, 6.5), style='-')
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
    data.Daily_Return.plot(legend=True, figsize=(14.5, 6.5))
    plt.title('Daily Return Percentage')
    plt.savefig('./static/image/daily_return_chart.png')

    ###############################################################################################################
    # Tạo dataset theo batch size trong pytorch
    # Chuẩn hóa dữ liệu
    data2 = data.copy(deep=True)
    scaler = MinMaxScaler()
    data2['Close'] = scaler.fit_transform(data2['Close'].values.reshape(-1, 1))
    data2['Open'] = scaler.transform(data2['Open'].values.reshape(-1, 1))
    data2['High'] = scaler.transform(data2['High'].values.reshape(-1, 1))
    data2['Low'] = scaler.transform(data2['Low'].values.reshape(-1, 1))

    # Tạo dữ liệu huấn luyện và kiểm tra
    train_data = data2.iloc[:-30]
    test_data = data2.iloc[-30:]

    # Chuyển đổi dữ liệu thành dạng tensor
    train_tensor = torch.tensor(train_data[['Close', 'Open', 'High', 'Low']].values).float()
    test_tensor = torch.tensor(test_data[['Close', 'Open', 'High', 'Low']].values).float()

    # Tạo dataset và dataloader
    train_dataset = TensorDataset(train_tensor[:-1], train_tensor[1:])
    test_dataset = TensorDataset(test_tensor[:-1], test_tensor[1:])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Huấn luyện mô hình
    num_epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = train(train_dataloader)
        test_loss = evaluate(test_dataloader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Dự đoán giá đóng cửa
    with torch.no_grad():
        model.eval()
        predicted = model(test_tensor[:-1].to(device)).cpu().numpy()
        predicted = scaler.inverse_transform(predicted)

    # Plot dự đoán và giá thực tế
    plt.figure(figsize=(14.5, 6.5))
    x_values = test_data.index[1:len(predicted) + 1]
    y_values_actual = test_data['Close'].values[1:len(predicted) + 1]
    y_values_predicted = predicted.flatten()

    plt.plot(x_values, y_values_actual, label='Actual Close Price')
    plt.plot(x_values, y_values_predicted[:len(x_values)], label='Predicted Close Price')
    plt.title('Actual Close Price vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig('./static/image/prediction_chart.png')

    # Truyền dữ liệu cho template
    return render_template('ticker.html', ticker=ticker, current_price=current_price, open_price=open_price,
                           volume=volume, chart_data=chart_data, train_losses=train_losses, test_losses=test_losses)


@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.get_json()['ticker']
    data = yf.Ticker(ticker).history(period='1y')
    return jsonify({'currentPrice': data.iloc[-1].Close,
                    'openPrice': data.iloc[-1].Open})


if __name__ == '__main__':
    app.run(debug=True)
