import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import yaml

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]

def fetch_global_stocks(config):
    tickers = config['data']['tickers']
    years = config['data']['years']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    all_data = []
    for ticker in tickers:
        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            all_data.append(data)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    df = pd.concat(all_data, ignore_index=True)
    df.to_csv(config['data']['raw_data_path'], index=False)
    return df

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    fetch_global_stocks(config)