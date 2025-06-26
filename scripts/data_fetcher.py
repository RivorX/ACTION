import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from .config_manager import ConfigManager
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.tickers_file = Path(config.get('data.tickers_file', 'config/training_tickers.yaml'))
        self.years = config.get('data.years', 3)
        self.raw_data_path = Path(config.get('data.raw_data_path', 'data/stock_data.csv'))

    def _load_tickers(self, region: str = None) -> list:
        try:
            with open(self.tickers_file, 'r') as f:
                tickers_config = yaml.safe_load(f)
                if region and region in tickers_config['tickers']:
                    return tickers_config['tickers'][region]
                return [ticker for region_tickers in tickers_config['tickers'].values() for ticker in region_tickers]
        except Exception as e:
            logger.error(f"Błąd wczytywania tickerów: {e}")
            return []

    def fetch_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                logger.warning(f"Brak danych dla {ticker}")
                return pd.DataFrame()
            df.reset_index(inplace=True)
            df['Ticker'] = ticker
            logger.info(f"Ostatnie 5 wierszy danych dla {ticker}:\n{df.tail(5).to_string()}")
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
        except Exception as e:
            logger.error(f"Błąd pobierania danych dla {ticker}: {e}")
            return pd.DataFrame()

    def fetch_global_stocks(self, region: str = None) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)
        tickers = self._load_tickers(region)
        all_data = []
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, start_date, end_date)
            if not data.empty and all(col in data.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']):
                all_data.append(data)
            else:
                logger.warning(f"Pominięto ticker {ticker} z powodu niekompletnych danych.")
        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        if not df.empty:
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Dane zapisane do {self.raw_data_path}")
        else:
            logger.error("Nie udało się pobrać żadnych danych giełdowych.")
        return df