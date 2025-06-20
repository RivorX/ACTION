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
    def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
        """Usuwa wartości odstające na podstawie z-score."""
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje nowe cechy do ramki danych."""
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.compute_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=20).std()

        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Wstęgi Bollingera
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

        # ATR
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close_Prev'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close_Prev'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High_Low', 'High_Close_Prev', 'Low_Close_Prev']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df = df.drop(['High_Low', 'High_Close_Prev', 'Low_Close_Prev'], axis=1)

        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()

        # Procentowa zmiana ceny
        df['Price_Change'] = df['Close'].pct_change()

        # Dzień tygodnia i miesiąc
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Day_of_Week'] = df['Date'].dt.dayofweek.astype(str)
        df['Month'] = df['Date'].dt.month.astype(str)

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
            min_encoder_length=self.config['model']['min_encoder_length'],
            max_encoder_length=int(df['time_idx'].max()),
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