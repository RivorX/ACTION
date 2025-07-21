import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from .config_manager import ConfigManager
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.tickers_file = Path(config.get('data.tickers_file', 'config/training_tickers.yaml'))
        self.years = config.get('data.years', 3)
        self.raw_data_path = Path(config.get('data.raw_data_path', 'data/stock_data.csv'))
        self.extra_days = 100  # Zwiększono z 50 do 100 dla większej pewności

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
            # Ensure start_date and end_date are timezone-naive
            if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)

            # Fetch additional 100 days of data before start_date
            adjusted_start_date = start_date - timedelta(days=self.extra_days)
            logger.info(f"Pobieranie danych dla {ticker} od {adjusted_start_date} do {end_date}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=adjusted_start_date, end=end_date)

            if df.empty:
                logger.warning(f"Brak danych dla {ticker}")
                return pd.DataFrame()

            # Logowanie liczby dni i sprawdzenie luk w danych
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            expected_days = (end_date - adjusted_start_date).days
            actual_days = len(df)
            logger.info(f"Liczba dni w danych dla {ticker}: {actual_days}, oczekiwano: {expected_days}")
            if actual_days < expected_days * 0.9:  # Jeśli brakuje więcej niż 10% dni
                logger.warning(f"Potencjalne luki w danych dla {ticker}: otrzymano {actual_days} dni, oczekiwano {expected_days}")

            df['Ticker'] = ticker

            # Initialize columns
            df['PE_ratio'] = np.nan
            df['PB_ratio'] = np.nan
            df['EPS'] = np.nan

            # Fetch fundamental data
            info = stock.info
            trailing_pe = info.get('trailingPE')
            pb_ratio = info.get('priceToBook')
            shares_outstanding = info.get('sharesOutstanding')
            total_equity = info.get('totalStockholderEquity')

            # Calculate EPS based on price and PE (if PE available)
            if trailing_pe and trailing_pe != 0:
                df['PE_ratio'] = trailing_pe
                df['EPS'] = df['Close'] / trailing_pe
            else:
                logger.warning(f"Brak trailingPE dla {ticker}, ustawiam EPS na NaN")

            # Calculate PB_ratio based on book value per share
            if total_equity and shares_outstanding and shares_outstanding != 0:
                book_value_per_share = total_equity / shares_outstanding
                df['PB_ratio'] = df['Close'] / book_value_per_share
            elif pb_ratio and pb_ratio != 0:
                df['PB_ratio'] = pb_ratio
            else:
                logger.warning(f"Brak danych do obliczenia PB_ratio dla {ticker}")

            # Fill missing values with forward and backward fill
            df['EPS'] = df['EPS'].ffill().bfill()
            df['PE_ratio'] = df['PE_ratio'].ffill().bfill()
            df['PB_ratio'] = df['PB_ratio'].ffill().bfill()

            # Check if fundamental data is missing
            if df['EPS'].isna().all() or df['PE_ratio'].isna().all() or df['PB_ratio'].isna().all():
                logger.warning(f"Brak danych fundamentalnych dla {ticker}, używane będą tylko dane techniczne")

            logger.info(
                f"{ticker} - unikalne EPS: {df['EPS'].nunique()}, "
                f"PE: {df['PE_ratio'].iloc[0] if not df['PE_ratio'].isna().all() else 'brak'}, "
                f"PB: {df['PB_ratio'].iloc[0] if not df['PB_ratio'].isna().all() else 'brak'}"
            )

            # Trim data to original date range
            df = df[df['Date'] >= pd.Timestamp(start_date, tz='UTC')].reset_index(drop=True)
            logger.info(f"Długość danych po przycięciu dla {ticker}: {len(df)}")

            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                       'Ticker', 'PE_ratio', 'PB_ratio', 'EPS']]

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
                fundamental_cols = ['PE_ratio', 'PB_ratio', 'EPS']
                if not all(data[f].isna().all() for f in fundamental_cols):
                    logger.info(f"Dane dla {ticker} zawierają dane fundamentalne")
                else:
                    logger.warning(f"Dane dla {ticker} nie zawierają danych fundamentalnych, używane będą tylko dane techniczne")
                all_data.append(data)
            else:
                logger.warning(f"Pominięto ticker {ticker} z powodu niekompletnych danych technicznych")
        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        if not df.empty:
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Dane zapisane do {self.raw_data_path}")
        else:
            logger.error("Nie udało się pobrać żadnych danych giełdowych.")
        return df