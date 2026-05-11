import yfinance as yf


# adj close: div & split adjusted — clean ret series
def load_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]


# last price per calendar month
def get_monthly_prices(data):
    return data.resample("ME").last()


# r(t) = P(t)/P(t-1) − 1
def get_monthly_returns(prices):
    return prices.pct_change()
