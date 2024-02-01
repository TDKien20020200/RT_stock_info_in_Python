import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yfinance as yf
import torch.nn as nn
import torch.functional as F
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo

from tqdm.notebook import tqdm
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# create a ticker object for Google (GOOGL)
ticker = yf.Ticker('GOOG')

# Define the start and end dates
start_date = "2023-01-01"

# Get historical data for the specified date range
df = ticker.history(start=start_date, end=None)
print(df.tail(20))

candlestick_trace = go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Candlestick'
)

# Create the layout
layout = go.Layout(
    title='GOOG Candlestick Chart',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price', rangemode='normal')
)

# Create the figure and add the candlestick trace and layout
fig = go.Figure(data=[candlestick_trace], layout=layout)

# # Use plot function from plotly.offline to display the chart
# pyo.plot(fig)

# Move column 'Close' to the first position
col_close = df.pop('Close')
df.insert(0, 'Close', col_close)

# # Kiểm tra có cột dữ liệu nào nhận giá trị 'NaN'
# print(df.isna().sum().to_frame(name='num_of_NaN'))

# # Kiểm tra có những dữ liệu nào bị trùng lặp
# print(df.duplicated().sum())
#
# # Biểu đồ biểu diễn các cột Open, Close,... của df theo ngày
# df.plot(subplots=True, figsize=(17, 7))
# plt.suptitle('Google stock attributes', y=0.91)
# plt.show()

# # timely (theo tuần -> freq: 'W'. theo tháng -> freq: 'M', '3M', '6M')
# df.asfreq('W', method='ffill').plot(subplots=True, figsize=(17,7), style='-')
# plt.suptitle('Google Stock attributes (Weekly frequency)', y=0.91)
# plt.show()

# print(df[['Close']])

# # Tính toán các đường trung bình theo 10d, 20d, 50d moving average(ma)
# ma_day = [10, 20, 50]
#
# for ma in ma_day:
#     col_name = f'MA for {ma} days'
#     df[col_name] = df['Close'].rolling(ma).mean()
#
# # df[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(17,7))
# # plt.title('Comparision some MA and Close of Google stock')
# # plt.show()
#
# # Dùng hàm pct_change() để tìm phần trăm thay đổi của giá Close mỗi ngày
# df['Daily_Return'] = df['Close'].pct_change()
#
# # # Biểu diễn
# # df.Daily_Return.plot(legend=True, figsize=(15,5))
# # plt.title('Daily return percentage of Google stock')
# # plt.show()
#
# ###############################################################################################################
# # Tạo dataset theo batch size trong pytorch
# # Chuẩn hóa dữ liệu
# df2 = df.copy(deep=True)
# scaler = MinMaxScaler(feature_range=(0, 15)).fit(df2.Low.values.reshape(-1, 1))
# df2['Open'] = scaler.transform(df2.Open.values.reshape(-1, 1))
# df2['High'] = scaler.transform(df2.High.values.reshape(-1, 1))
# df2['Low'] = scaler.transform(df2.Low.values.reshape(-1, 1))
# df2['Close'] = scaler.transform(df2.Close.values.reshape(-1, 1))
# data = df2[['Open', 'High', 'Low', 'Close']].values
# # print(data.shape)
# # print(data)
#
#
# # chuẩn bị dữ liệu cho bài toán dự đoán giá cổ phiếu dựa trên chuỗi thời gian. Nó tạo ra các chuỗi (sequences) dựa
# # trên giá cổ phiếu từ 10 ngày trước để dự đoán giá cổ phiếu vào ngày tiếp theo
# seq_len = 11
# sequences = []
# for index in range(len(data) - seq_len + 1):
#     sequences.append(data[index: index + seq_len])
# sequences = np.array(sequences)
#
#
# # Customize dataset Tách dữ liệu toàn bộ tập dữ liệu thành ba phần. 80% cho tập huấn luyện (train), 10% cho tập xác
# # thực (valid) và 10% còn lại cho tập kiểm thử (test):
# valid_set_size_percentage = 10
# test_set_size_percentage = 10
#
# valid_set_size = int(np.round(valid_set_size_percentage / 100 * sequences.shape[0]))
# test_set_size = int(np.round(test_set_size_percentage / 100 * sequences.shape[0]))
# train_set_size = sequences.shape[0] - (valid_set_size + test_set_size)
#
# x_train = sequences[:train_set_size, :-1, :]
# y_train = sequences[:train_set_size, -1, :]
#
# x_valid = sequences[train_set_size:train_set_size + valid_set_size, :-1, :]
# y_valid = sequences[train_set_size:train_set_size + valid_set_size, -1, :]
#
# x_test = sequences[train_set_size + test_set_size:, :-1, :]
# y_test = sequences[train_set_size + test_set_size:, -1, :]
#
#
# # DataLoader
# # Tạo Trình tải dữ liệu: xác định các trình tải dữ liệu để tải tập dữ liệu theo từng batch với batch size = 32
# x_train = torch.tensor(x_train).float()
# y_train = torch.tensor(y_train).float()
#
# x_valid = torch.tensor(x_valid).float()
# y_valid = torch.tensor(y_valid).float()
#
# train_dataset = TensorDataset(x_train, y_train)
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
#
# valid_dataset = TensorDataset(x_valid, y_valid)
# valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
#
# # ############################################################################################################## Xây
# # dựng model Xác định kiến trúc: vì là vấn đề về chuỗi thời gian nên sẽ sử dụng Long Short-term Memory (LSTM) để nắm
# # bắt thông tin tuần tự:
# class NeuralNetwork(nn.Module):
#     def __init__(self, num_feature):
#         super(NeuralNetwork, self).__init__()
#         self.lstm = nn.LSTM(num_feature, 64, batch_first=True)
#         self.fc = nn.Linear(64, num_feature)
#
#     def forward(self, x):
#         output, (hidden, cell) = self.lstm(x)
#         x = self.fc(hidden)
#         return x
#
#
# model = NeuralNetwork(4)
#
# # push to cuda if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
#
# # summary(model, (4,))
#
# # Hàm Optimize Adam
# optimizer = optim.Adam(model.parameters())
# # Hàm Loss MSELoss
# mse = nn.MSELoss()
#
#
# # Mô hình Training: xác định quá trình thuận và nghịch (forward and backward) để train mạng lưới Neural:
# def train(dataloader):
#     epoch_loss = 0
#     model.train()
#
#     for batch in dataloader:
#         optimizer.zero_grad()
#         x, y = batch
#         pred = model(x)
#         loss = mse(pred[0], y)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#
#     return epoch_loss
#
#
# # Đánh giá hiệt suất mô hình
# def evaluate(dataloader):
#     epoch_loss = 0
#     model.eval()
#
#     with torch.no_grad():
#         for batch in dataloader:
#             x, y = batch
#             pred = model(x)
#             loss = mse(pred[0], y)
#             epoch_loss += loss.item()
#
#     return epoch_loss / len(dataloader)
#
#
# # Train mô hình trong 50 data sau đó sẽ lưu mô hình tốt nhất trong quá trình đào tạo dựa trên việc xác định Giá trị
# # bị mất (Val Loss):
# n_epochs = 100
# best_valid_loss = float('inf')
#
# for epoch in range(1, n_epochs + 1):
#
#     train_loss = train(train_dataloader)
#     valid_loss = evaluate(valid_dataloader)
#
#     # save the best model
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model, 'saved_weights.pt')
#
#     # print("Epoch ",epoch+1)
#     # print(f'\tTrain Loss: {train_loss:.5f} | ' + f'\tVal Loss: {valid_loss:.5f}\n')
#
# # Từ kết quả trên suy ra đuợc mô hình tốt nhất, từ đó đưa ra vài dự đoán ban đầu
# model=torch.load('saved_weights.pt')
#
# x_test= torch.tensor(x_test).float()
#
# with torch.no_grad():
#   y_test_pred = model(x_test)
#
# y_test_pred = y_test_pred.numpy()[0]
# # visualize các dự đoán và so sánh chúng với thực tế:
# idx=0
# plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
#          y_test[:,idx], color='black', label='test target')
#
# plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test_pred.shape[0]),
#          y_test_pred[:,idx], color='green', label='test prediction')
#
# plt.title('future stock prices')
# plt.xlabel('time [days]')
# plt.ylabel('normalized price')
# plt.legend(loc='best')
# plt.show()
#
# #######################################################################################################3
# index_values = df[len(df) - len(y_test):].index
# col_values = ['Open', 'Low', 'High', 'Close']
# df_results = pd.DataFrame(data=y_test_pred, index=index_values, columns=col_values)
#
# # Create a trace for the candlestick chart
# candlestick_trace = go.Candlestick(
#     x=df_results.index,
#     open=df_results['Open'],
#     high=df_results['High'],
#     low=df_results['Low'],
#     close=df_results['Close'],
#     name='Candlestick'
# )
#
# # Create the layout
# layout = go.Layout(
#     title='GOOG Candlestick Chart',
#     xaxis=dict(title='Date'),
#     yaxis=dict(title='Price', rangemode='normal')
# )
#
# # Create the figure and add the candlestick trace and layout
# fig = go.Figure(data=[candlestick_trace], layout=layout)
#
# # Update the layout of the figure
# fig.update_layout(xaxis_rangeslider_visible=False)
#
# # # Show the figure
# # fig.show()
#
# # Dự đoán 10 ngày tiếp theo
# # Get the last sequence of historical data as features for predicting the next 10 days
# last_sequence = sequences[-1:, 1:, :]
# last_sequence = torch.from_numpy(last_sequence).float()
#
#
# # Generate predictions for the next 10 days
# PRED_DAYS = 10
# with torch.no_grad():
#     for i in range(PRED_DAYS):
#         pred_i = model(last_sequence)
#         last_sequence = torch.cat((last_sequence, pred_i), dim=1)
#         last_sequence = last_sequence[:, 1:, :]
#
#
# pred_days = last_sequence.reshape(PRED_DAYS, 4).numpy()
#
# # inverse transform the predicted values
# pred_days = scaler.inverse_transform(pred_days)
#
# df_pred = pd.DataFrame(
#     data=pred_days,
#     columns=['Open', 'High', 'Low', 'Close']
# )
#
# print(df_pred)



chương 1: giới thiệu vài toán, giá trị mang lại, hệ thống liên quan,...
chương 2: nền tảng công nghệ( mọi công nghệ về vuild hệ thống, tiền xử lí, techinical, công cụ, thư viện)
chuơng 3: gộp 2 cái kia lại: c
chương 4: cần sự liên quan và chuyển tiếp với chương 3
chương 5: triển khai và thử nghiệm hệ thống: xem hệ thống hoạt động như thế nào

