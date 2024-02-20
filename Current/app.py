import yfinance as yf
from flask import request, render_template, jsonify, Flask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, template_folder='templates')

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


    data = data.tail(30)
    data_list = data.reset_index().to_dict(orient='records')
    
    return render_template('ticker_detail.html', ticker=ticker, current_price=current_price, open_price=open_price,
                           volume=volume, data_list=data_list, chart_data=chart_data)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.get_json()['ticker']
    data = yf.Ticker(ticker).history(period='1y')
    return jsonify({'currentPrice': data.iloc[-1].Close,
                    'openPrice': data.iloc[-1].Open})

if __name__ == '__main__':
    app.run(debug=True)