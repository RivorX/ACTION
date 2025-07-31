import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from .config_manager import ConfigManager
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lista wszystkich możliwych sektorów, w tym Unknown
ALL_SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 'Consumer Staples',
    'Energy', 'Utilities', 'Industrials', 'Materials', 'Communication Services',
    'Real Estate', 'Unknown'
]

class DataFetcher:
    def __init__(self, config_manager: ConfigManager, years: int):
        """
        Inicjalizuje DataFetcher z config_managerem i liczbą lat danych do pobrania.

        Args:
            config_manager (ConfigManager): Obiekt ConfigManager z konfiguracją.
            years (int): Liczba lat danych historycznych do pobrania.
        """
        self.config = config_manager.config
        self.years = years
        try:
            self.tickers_file = Path(self.config['data']['tickers_file'])
            self.raw_data_path = Path(self.config['data']['raw_data_path'])
        except KeyError as e:
            logger.error(f"Missing key in config: {e}")
            raise ValueError(f"Configuration error: missing key {e} in config.yaml")
        self.extra_days = 50  # Bufor na dodatkowe dni
        self.executor = ThreadPoolExecutor(max_workers=10)
        logger.info(f"Inicjalizacja DataFetcher z {self.years} latami danych.")

    def _load_tickers(self, region: str = None) -> list:
        """
        Wczytuje tickery dla podanego regionu z pliku YAML.

        Args:
            region (str): Region, dla którego wczytywane są tickery (np. 'poland', 'usa').

        Returns:
            list: Lista tickerów dla danego regionu.
        """
        try:
            with open(self.tickers_file, 'r') as f:
                tickers_config = yaml.safe_load(f)
                if region and region in tickers_config['tickers']:
                    return [item['ticker'] for item in tickers_config['tickers'][region]]
                return [item['ticker'] for region_tickers in tickers_config['tickers'].values() for item in region_tickers]
        except Exception as e:
            logger.error(f"Błąd wczytywania tickerów: {e}")
            return []

    async def fetch_stock_data(self, ticker: str, start_date: datetime, end_date: datetime, session: aiohttp.ClientSession) -> pd.DataFrame:
        """
        Pobiera dane giełdowe dla pojedynczego tickera.

        Args:
            ticker (str): Symbol tickera.
            start_date (datetime): Data początkowa.
            end_date (datetime): Data końcowa.
            session (aiohttp.ClientSession): Sesja aiohttp.

        Returns:
            pd.DataFrame: Dane giełdowe dla tickera.
        """
        try:
            # Ensure start_date and end_date are timezone-naive
            if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)

            # Fetch additional 50 days of data before start_date
            adjusted_start_date = start_date - timedelta(days=self.extra_days)

            # Run yfinance calls in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(self.executor, lambda: yf.Ticker(ticker))
            info = await loop.run_in_executor(self.executor, lambda: stock.info)
            first_trade_date = info.get('firstTradeDateEpochUtc', None)
            if first_trade_date:
                first_trade_date = pd.to_datetime(first_trade_date / 1000, unit='s', utc=True).replace(tzinfo=None)
                if first_trade_date > adjusted_start_date:
                    logger.warning(f"Ticker {ticker} ma dane dopiero od {first_trade_date}, dostosowuję start_date")
                    adjusted_start_date = first_trade_date

            # Pobieranie danych z opcją naprawy luk
            df = await loop.run_in_executor(self.executor, lambda: stock.history(start=adjusted_start_date, end=end_date, repair=True, auto_adjust=True))
            if df.empty:
                logger.warning(f"Brak danych dla {ticker}")
                return pd.DataFrame()

            # Reset indeksu i zmiana nazw kolumn
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df.columns = [col.replace(' ', '_') for col in df.columns]
            df = df.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                                    'Close': 'Close', 'Volume': 'Volume'})
            df['Ticker'] = ticker

            # Dodanie sektora z listy ALL_SECTORS
            sector = info.get('sector', 'Unknown')
            if sector not in ALL_SECTORS:
                sector = 'Unknown'
            df['Sector'] = sector
            logger.info(f"Przypisano sektor dla {ticker}: {sector}")

            # Logowanie liczby dni i sprawdzenie luk w danych
            expected_days = (end_date - adjusted_start_date).days * 0.6  # Minimum 60% dni handlowych
            actual_days = len(df)
            logger.info(f"Liczba dni w danych dla {ticker}: {actual_days}, minimum oczekiwane: {expected_days}")
            if actual_days < expected_days:
                logger.warning(f"Mała liczba dni dla {ticker}: {actual_days} dni, próbuję pobrać dłuższy zakres")
                # Próba pobrania dłuższego zakresu danych
                extended_start_date = adjusted_start_date - timedelta(days=365)  # Dodatkowy rok wstecz
                df_extended = await loop.run_in_executor(self.executor, lambda: stock.history(start=extended_start_date, end=end_date, repair=True, auto_adjust=True))
                if not df_extended.empty:
                    df_extended.reset_index(inplace=True)
                    df_extended['Date'] = pd.to_datetime(df_extended['Date'], utc=True)
                    df_extended.columns = [col.replace(' ', '_') for col in df_extended.columns]
                    df_extended = df_extended.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 
                                                             'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
                    df_extended['Ticker'] = ticker
                    df_extended['Sector'] = sector
                    actual_days_extended = len(df_extended)
                    if actual_days_extended > actual_days:
                        logger.info(f"Rozszerzono dane dla {ticker} do {actual_days_extended} dni")
                        df = df_extended

            # Przycięcie danych do oryginalnego zakresu dat
            df = df[df['Date'] >= pd.Timestamp(start_date, tz='UTC')].reset_index(drop=True)
            logger.info(f"Długość danych po przycięciu dla {ticker}: {len(df)}")

            # Filtruj wymagane kolumny
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Sector']
            df = df[required_cols]

            return df

        except Exception as e:
            logger.error(f"Błąd pobierania danych dla {ticker}: {e}")
            return pd.DataFrame()

    async def fetch_global_stocks(self, region: str = None) -> pd.DataFrame:
        """
        Pobiera dane giełdowe dla wszystkich tickerów z wybranego regionu.

        Args:
            region (str, optional): Region, dla którego pobierane są dane. Jeśli None, używa tickerów z configu.

        Returns:
            pd.DataFrame: Połączony DataFrame z danymi giełdowymi.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)
        tickers = self._load_tickers(region)
        all_data = []

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_stock_data(ticker, start_date, end_date, session) for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for ticker, result in zip(tickers, results):
                if isinstance(result, pd.DataFrame) and not result.empty and all(col in result.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Sector']):
                    all_data.append(result)
                else:
                    logger.warning(f"Pominięto ticker {ticker} z powodu niekompletnych danych lub błędu")

        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        if not df.empty:
            # Upewnij się, że kolumna Sector jest kategoryczna z pełnym zestawem kategorii
            df['Sector'] = pd.Categorical(df['Sector'], categories=ALL_SECTORS, ordered=False)
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Dane zapisane do {self.raw_data_path}")
        else:
            logger.error("Nie udało się pobrać żadnych danych giełdowych.")
        return df

    async def _fetch_stock_data_sync_helper(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            return await self.fetch_stock_data(ticker, start_date, end_date, session)

    def fetch_stock_data_sync(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Synchronous wrapper for fetching stock data."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._fetch_stock_data_sync_helper(ticker, start_date, end_date))

    def __del__(self):
        # Clean up the thread pool executor
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    config_manager = ConfigManager()
    # Domyślna liczba lat dla testów
    fetcher = DataFetcher(config_manager, years=3)
    asyncio.run(fetcher.fetch_global_stocks())