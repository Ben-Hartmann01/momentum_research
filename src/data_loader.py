import yfinance as yf


# adj close — splits and dividends handled automatically
def load_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]


# last trading day per calendar month
def get_monthly_prices(data):
    return data.resample("ME").last()


def get_monthly_returns(prices):
    return prices.pct_change()
