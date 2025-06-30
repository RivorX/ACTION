import pandas as pd
import numpy as np
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
import torch
import pickle
import logging
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Klasa do inżynierii cech dla danych giełdowych."""
    
    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Oblicza wskaźnik RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices: pd.Series) -> tuple:
        """Oblicza MACD i linię sygnałową."""
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    @staticmethod
    def calculate_stochastic_k(group: pd.DataFrame) -> pd.Series:
        """Oblicza Stochastic %K."""
        low_14 = group['Low'].rolling(window=14).min()
        high_14 = group['High'].rolling(window=14).max()
        return 100 * (group['Close'] - low_14) / (high_14 - low_14)

    @staticmethod
    def calculate_true_range(group: pd.DataFrame) -> pd.Series:
        """Oblicza True Range."""
        high_low = group['High'] - group['Low']
        high_close_prev = abs(group['High'] - group['Close'].shift(1))
        low_close_prev = abs(group['Low'] - group['Close'].shift(1))
        return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    @staticmethod
    def calculate_obv(group: pd.DataFrame) -> pd.Series:
        """Oblicza On-Balance Volume."""
        return (np.sign(group['Close'].diff()) * group['Volume']).cumsum()

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
        """Usuwa wartości odstające na podstawie z-score."""
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje nowe cechy do ramki danych z grupowaniem po Ticker."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)

        def apply_features(group):
            group['MA10'] = group['Close'].rolling(window=10).mean()
            group['MA50'] = group['Close'].rolling(window=50).mean()
            group['RSI'] = self.compute_rsi(group['Close'])
            group['Volatility'] = group['Close'].rolling(window=20).std()
            group['MACD'], group['MACD_Signal'] = self.calculate_macd(group['Close'])
            group['BB_Middle'] = group['Close'].rolling(window=20).mean()
            group['BB_Std'] = group['Close'].rolling(window=20).std()
            group['BB_Upper'] = group['BB_Middle'] + 2 * group['BB_Std']
            group['BB_Lower'] = group['BB_Middle'] - 2 * group['BB_Std']
            group['Stochastic_K'] = self.calculate_stochastic_k(group)
            group['Stochastic_D'] = group['Stochastic_K'].rolling(window=3).mean()
            group['TR'] = self.calculate_true_range(group)
            group['ATR'] = group['TR'].rolling(window=14).mean()
            group['OBV'] = self.calculate_obv(group)
            group['Price_Change'] = group['Close'].pct_change()
            group['Day_of_Week'] = group['Date'].dt.dayofweek.astype(str)
            group['Month'] = group['Date'].dt.month.astype(str)
            return group

        df = df.groupby('Ticker').apply(apply_features).reset_index(drop=True)
        return df.ffill().bfill().dropna()

class DataPreprocessor:
    """Klasa odpowiedzialna za preprocessing danych giełdowych i tworzenie zbioru danych TimeSeriesDataSet."""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.normalizers_path = Path(config['data']['normalizers_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])

    def preprocess_data(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        """Preprocessuje dane i tworzy zbiór danych TimeSeriesDataSet."""
        if df.empty:
            raise ValueError("Ramka danych jest pusta. Sprawdź dane wejściowe.")

        # Czyszczenie i inżynieria cech
        df = self.feature_engineer.add_features(df)
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])]
        df = self.feature_engineer.remove_outliers(df, 'Close')

        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
        df['group_id'] = df['Ticker']

        # Transformacja logarytmiczna
        log_features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "BB_Middle", "BB_Upper", "BB_Lower", "ATR"]
        for feature in log_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        # Normalizacja
        numeric_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
            "MACD", "MACD_Signal", "BB_Middle", "BB_Upper", "BB_Lower", "Stochastic_K",
            "Stochastic_D", "ATR", "OBV", "Price_Change"
        ]
        normalizers = {}
        for feature in numeric_features:
            if feature in df.columns:
                normalizers[feature] = TorchNormalizer()
                df[feature] = normalizers[feature].fit_transform(df[feature].values)

        # Zapisz normalizery
        with open(self.normalizers_path, 'wb') as f:
            pickle.dump(normalizers, f)
        logger.info(f"Normalizery zapisane do: {self.normalizers_path}")

        # Utwórz dataset
        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="Close",
            group_ids=["group_id"],
            min_encoder_length=self.config['model']['min_encoder_length'],  # Minimalna długość enkodera
            max_encoder_length=self.config['model']['max_encoder_length'],  # Maksymalna długość enkodera
            max_prediction_length=self.config['model']['max_prediction_length'],
            time_varying_known_reals=[f for f in numeric_features if f in df.columns and f != "Close"],
            time_varying_known_categoricals=["Day_of_Week", "Month"],
            time_varying_unknown_reals=["Close"],
            target_normalizer=normalizers.get("Close", TorchNormalizer()),
            allow_missing_timesteps=True,
            add_encoder_length=False
        )
        logger.info(f"Target normalizer: {dataset.target_normalizer}")
        dataset.save(self.processed_data_path)
        return dataset