import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import yaml

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            print(f"Brak danych dla {ticker}, pomijam.")
            return None
        df.reset_index(inplace=True)
        df['Ticker'] = ticker
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_global_stocks(config):
    tickers = config['data']['tickers']
    years = config['data']['years']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    all_data = []
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            all_data.append(data)
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    if not df.empty:
        df.to_csv(config['data']['raw_data_path'], index=False)
    return df

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    fetch_global_stocks(config)