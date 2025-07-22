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
        self.extra_days = 50  # Bufor na dodatkowe dni

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

            # Fetch additional 50 days of data before start_date
            adjusted_start_date = start_date - timedelta(days=self.extra_days)

            # Sprawdzenie dostępności danych w metadanych tickera
            stock = yf.Ticker(ticker)
            info = stock.info
            first_trade_date = info.get('firstTradeDateEpochUtc', None)
            if first_trade_date:
                first_trade_date = pd.to_datetime(first_trade_date / 1000, unit='s', utc=True).replace(tzinfo=None)
                if first_trade_date > adjusted_start_date:
                    logger.warning(f"Ticker {ticker} ma dane dopiero od {first_trade_date}, dostosowuję start_date")
                    adjusted_start_date = first_trade_date

            # Pobieranie danych z opcją naprawy luk
            df = stock.history(start=adjusted_start_date, end=end_date, repair=True, auto_adjust=True)
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

            # Logowanie liczby dni i sprawdzenie luk w danych
            expected_days = (end_date - adjusted_start_date).days * 0.6  # Minimum 60% dni handlowych
            actual_days = len(df)
            logger.info(f"Liczba dni w danych dla {ticker}: {actual_days}, minimum oczekiwane: {expected_days}")
            if actual_days < expected_days:
                logger.warning(f"Mała liczba dni dla {ticker}: {actual_days} dni, próbuję pobrać dłuższy zakres")
                # Próba pobrania dłuższego zakresu danych
                extended_start_date = adjusted_start_date - timedelta(days=365)  # Dodatkowy rok wstecz
                df_extended = stock.history(start=extended_start_date, end=end_date, repair=True, auto_adjust=True)
                if not df_extended.empty:
                    df_extended.reset_index(inplace=True)
                    df_extended['Date'] = pd.to_datetime(df_extended['Date'], utc=True)
                    df_extended.columns = [col.replace(' ', '_') for col in df_extended.columns]
                    df_extended = df_extended.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 
                                                             'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
                    df_extended['Ticker'] = ticker
                    actual_days_extended = len(df_extended)
                    if actual_days_extended > actual_days:
                        logger.info(f"Rozszerzono dane dla {ticker} do {actual_days_extended} dni")
                        df = df_extended

            # Inicjalizacja kolumn fundamentalnych
            df['PE_ratio'] = 0.0
            df['PB_ratio'] = 0.0
            df['EPS'] = 0.0
            df['PE_ratio_missing'] = 1
            df['PB_ratio_missing'] = 1
            df['EPS_missing'] = 1

            # Pobieranie danych fundamentalnych
            trailing_pe = info.get('trailingPE')
            pb_ratio = info.get('priceToBook')
            shares_outstanding = info.get('sharesOutstanding')
            total_equity = info.get('totalStockholderEquity')

            # Obliczanie EPS na podstawie ceny i PE (jeśli dostępne)
            if trailing_pe and trailing_pe != 0 and not np.isinf(trailing_pe) and not np.isnan(trailing_pe):
                df['PE_ratio'] = trailing_pe
                # Bezpieczne obliczanie EPS z sprawdzeniem na NaN i inf
                eps_calculated = df['Close'] / trailing_pe
                # Sprawdź czy obliczenia są poprawne
                if not eps_calculated.isna().any() and not np.isinf(eps_calculated).any():
                    df['EPS'] = eps_calculated
                    df['PE_ratio_missing'] = 0
                    df['EPS_missing'] = 0
                else:
                    logger.warning(f"Niepoprawne obliczenie EPS dla {ticker}, pozostawiam jako 0")
                    df['EPS'] = 0.0
                    df['PE_ratio'] = 0.0
                    df['PE_ratio_missing'] = 1
                    df['EPS_missing'] = 1
            else:
                logger.warning(f"Brak lub niepoprawne trailingPE dla {ticker}, pozostawiam PE_ratio i EPS jako 0")

            # Obliczanie PB_ratio na podstawie wartości księgowej na akcję
            if total_equity and shares_outstanding and shares_outstanding != 0:
                book_value_per_share = total_equity / shares_outstanding
                # Sprawdź czy book_value_per_share jest poprawne
                if book_value_per_share > 0 and not np.isinf(book_value_per_share) and not np.isnan(book_value_per_share):
                    pb_ratio_calculated = df['Close'] / book_value_per_share
                    # Sprawdź czy obliczenia PB są poprawne
                    if not pb_ratio_calculated.isna().any() and not np.isinf(pb_ratio_calculated).any() and (pb_ratio_calculated > 0).any():
                        df['PB_ratio'] = pb_ratio_calculated
                        df['PB_ratio_missing'] = 0
                        logger.info(f"Obliczono PB_ratio dla {ticker} na podstawie total_equity i shares_outstanding")
                    else:
                        logger.warning(f"Niepoprawne obliczenie PB_ratio dla {ticker}, pozostawiam jako 0")
                else:
                    logger.warning(f"Niepoprawna book_value_per_share dla {ticker}, pozostawiam PB_ratio jako 0")
            elif pb_ratio and pb_ratio != 0 and not np.isinf(pb_ratio) and not np.isnan(pb_ratio):
                df['PB_ratio'] = pb_ratio
                df['PB_ratio_missing'] = 0
            else:
                logger.warning(f"Brak danych do obliczenia PB_ratio dla {ticker}, pozostawiam jako 0")

            # Sprawdzenie poprawności danych fundamentalnych
            if df['PE_ratio_missing'].all() and df['PB_ratio_missing'].all() and df['EPS_missing'].all():
                logger.warning(f"Brak jakichkolwiek danych fundamentalnych dla {ticker}, używane będą tylko dane techniczne")
            else:
                logger.info(f"Dane fundamentalne dostępne dla {ticker}: "
                           f"PE_ratio: {df['PE_ratio'].iloc[0] if not df['PE_ratio_missing'].all() else 'brak'}, "
                           f"PB_ratio: {df['PB_ratio'].iloc[0] if not df['PB_ratio_missing'].all() else 'brak'}, "
                           f"EPS: {df['EPS'].iloc[0] if not df['EPS_missing'].all() else 'brak'}")

            # Przycięcie danych do oryginalnego zakresu dat
            df = df[df['Date'] >= pd.Timestamp(start_date, tz='UTC')].reset_index(drop=True)
            logger.info(f"Długość danych po przycięciu dla {ticker}: {len(df)}")

            # Filtruj wymagane kolumny
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker',
                             'PE_ratio', 'PB_ratio', 'EPS', 'PE_ratio_missing', 'PB_ratio_missing', 'EPS_missing']
            df = df[required_cols]

            # Sprawdzenie na NaN lub nieskończoność - bardziej rygorystyczne
            for col in ['PE_ratio', 'PB_ratio', 'EPS']:
                # Sprawdź wszystkie typy problemów: NaN, inf, -inf, oraz wartości <= 0 dla PE i PB
                mask_invalid = (
                    df[col].isna() | 
                    np.isinf(df[col]) | 
                    (df[col] <= 0) |  # PE i PB nie mogą być ujemne lub zero
                    (df[col] > 1000)  # Bardzo wysokie wartości też mogą być błędne
                )
                
                if mask_invalid.any():
                    invalid_count = mask_invalid.sum()
                    total_count = len(df)
                    logger.warning(f"Kolumna {col} dla {ticker} zawiera {invalid_count}/{total_count} niepoprawnych wartości (NaN, inf, <=0, >1000), wypełniam zerami")
                    
                    # Ustaw niepoprawne wartości na 0
                    df.loc[mask_invalid, col] = 0.0
                    df.loc[mask_invalid, f'{col}_missing'] = 1
                
                # Sprawdź czy wszystkie wartości to zera - jeśli tak, oznacz jako missing
                if (df[col] == 0.0).all():
                    df[f'{col}_missing'] = 1
                    logger.info(f"Wszystkie wartości {col} dla {ticker} zostały ustawione na 0, oznaczono jako missing")
                
                # Finalne sprawdzenie
                final_check = df[col].isna().any() or np.isinf(df[col]).any()
                if final_check:
                    logger.error(f"BŁĄD: Nadal są NaN/inf w kolumnie {col} dla {ticker} po czyszczeniu!")
                    df[col] = 0.0
                    df[f'{col}_missing'] = 1

            return df

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
                logger.warning(f"Pominięto ticker {ticker} z powodu niekompletnych danych technicznych")
        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        if not df.empty:
            # Finalne sprawdzenie czystości danych przed zapisaniem
            fundamental_cols = ['PE_ratio', 'PB_ratio', 'EPS']
            for col in fundamental_cols:
                if col in df.columns:
                    # Sprawdź NaN i inf
                    nan_count = df[col].isna().sum()
                    inf_count = np.isinf(df[col]).sum()
                    if nan_count > 0 or inf_count > 0:
                        logger.warning(f"FINALNE CZYSZCZENIE: Kolumna {col} zawiera {nan_count} NaN i {inf_count} inf wartości")
                        df[col] = df[col].fillna(0.0)
                        df.loc[np.isinf(df[col]), col] = 0.0
                        # Ustaw missing flag dla wszystkich wypełnionych wartości
                        missing_col = f'{col}_missing'
                        if missing_col in df.columns:
                            df.loc[(df[col] == 0.0), missing_col] = 1
                    
                    # Statystyki końcowe
                    zero_count = (df[col] == 0.0).sum()
                    total_count = len(df)
                    logger.info(f"Kolumna {col}: {zero_count}/{total_count} ({100*zero_count/total_count:.1f}%) wartości to zera")
            
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Dane zapisane do {self.raw_data_path}")
        else:
            logger.error("Nie udało się pobrać żadnych danych giełdowych.")
        return df