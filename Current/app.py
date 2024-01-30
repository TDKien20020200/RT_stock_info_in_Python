import yfinance as yf
from flask import request, render_template, jsonify, Flask

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ticker/<ticker>')
def ticker_detail(ticker):
    # Retrieve the current stock data for the ticker
    data = yf.Ticker(ticker).history(period='1d')
    current_price = data.iloc[-1].Close
    open_price = data.iloc[-1].Open
    volume = data.iloc[-1].Volume

    return render_template('ticker_detail.html', ticker=ticker, current_price=current_price, open_price=open_price, volume=volume)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.get_json()['ticker']
    data = yf.Ticker(ticker).history(period='1y')
    return jsonify({'currentPrice': data.iloc[-1].Close,
                    'openPrice': data.iloc[-1].Open})

if __name__ == '__main__':
    app.run(debug=True)