import pandas as pd
import numpy as np
import torch
import pickle
import logging
from pathlib import Path
from pytorch_forecasting.data import TimeSeriesDataSet, NaNLabelEncoder
from scripts.utils.config_manager import ConfigManager
import torch
from pytorch_forecasting.data.encoders import TorchNormalizer

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
        """Oblicza MACD, linię sygnałową i histogram MACD."""
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def calculate_stochastic_k(group: pd.DataFrame) -> pd.Series:
        """Oblicza Stochastic %K z zabezpieczeniem przed dzieleniem przez zero."""
        low_14 = group['Low'].rolling(window=14).min()
        high_14 = group['High'].rolling(window=14).max()
        denominator = high_14 - low_14
        stochastic_k = 100 * (group['Close'] - low_14) / denominator.where(denominator != 0, 1e-10)
        stochastic_k = stochastic_k.replace([np.inf, -np.inf], 0)
        return stochastic_k

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
    def calculate_adx(group: pd.DataFrame, period: int = 14) -> pd.Series:
        """Oblicza Average Directional Index (ADX)."""
        tr = FeatureEngineer.calculate_true_range(group)
        plus_dm = group['High'].diff().where(lambda x: x > 0, 0)
        minus_dm = (-group['Low'].diff()).where(lambda x: x > 0, 0)
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / tr.ewm(span=period, adjust=False).mean())
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / tr.ewm(span=period, adjust=False).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    @staticmethod
    def calculate_cci(group: pd.DataFrame, period: int = 20) -> pd.Series:
        """Oblicza Commodity Channel Index (CCI)."""
        typical_price = (group['High'] + group['Low'] + group['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = (typical_price - sma_tp).abs().rolling(window=period).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        return cci

    @staticmethod
    def calculate_ichimoku(group: pd.DataFrame) -> tuple:
        """Oblicza linie Ichimoku Cloud: Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B."""
        high_9 = group['High'].rolling(window=9).max()
        low_9 = group['Low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2

        high_26 = group['High'].rolling(window=26).max()
        low_26 = group['Low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2

        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = (group['High'].rolling(window=52).max() + group['Low'].rolling(window=52).min()) / 2

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 20) -> pd.Series:
        """Oblicza Price Rate of Change (ROC)."""
        return 100 * (prices - prices.shift(period)) / prices.shift(period)

    @staticmethod
    def calculate_vwap(group: pd.DataFrame) -> pd.Series:
        """Oblicza Volume Weighted Average Price (VWAP)."""
        typical_price = (group['High'] + group['Low'] + group['Close']) / 3
        vwap = (typical_price * group['Volume']).cumsum() / group['Volume'].cumsum()
        return vwap

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
            group = group.sort_values('Date')

            group['MA10'] = group['Close'].rolling(window=10).mean()
            group['MA50'] = group['Close'].rolling(window=50).mean()
            
            group['BB_upper'] = group['Close'].rolling(window=20).mean() + 2 * group['Close'].rolling(window=20).std()
            group['BB_lower'] = group['Close'].rolling(window=20).mean() - 2 * group['Close'].rolling(window=20).std()
            group['BB_width'] = group['BB_upper'] - group['BB_lower']
            group['Close_to_BB_upper'] = group['Close'] / group['BB_upper']
            group['Close_to_BB_lower'] = group['Close'] / group['BB_lower']

            group['RSI'] = self.compute_rsi(group['Close'])
            group['MACD'], group['MACD_Signal'], group['MACD_Histogram'] = self.calculate_macd(group['Close'])
            group['Stochastic_K'] = self.calculate_stochastic_k(group)
            group['Stochastic_D'] = group['Stochastic_K'].rolling(window=3).mean()
            group['TR'] = self.calculate_true_range(group)
            group['ATR'] = group['TR'].rolling(window=14).mean()
            group['OBV'] = self.calculate_obv(group)
            group['ADX'] = self.calculate_adx(group)
            group['CCI'] = self.calculate_cci(group)
            group['Tenkan_sen'], group['Kijun_sen'], group['Senkou_Span_A'], group['Senkou_Span_B'] = self.calculate_ichimoku(group)
            group['ROC'] = self.calculate_roc(group['Close'])
            group['VWAP'] = self.calculate_vwap(group)
            group['Momentum_20d'] = group['Close'] - group['Close'].shift(20)
            group['Close_to_MA_ratio'] = group['Close'] / ((group['MA10'] + group['MA50']) / 2)

            group['Relative_Returns'] = group['Close'].pct_change().shift(-1)
            group['Log_Returns'] = np.log(group['Close'] / group['Close'].shift(1)).shift(-1)
            group['Future_Volume'] = group['Volume'].shift(-1)
            group['Future_Volatility'] = group['Close'].rolling(window=20).std().shift(-1)

            for col in ['Relative_Returns', 'Log_Returns', 'Future_Volume', 'Future_Volatility']:
                group[col] = group[col].fillna(0)

            technical_features = [
                'MA10', 'MA50', 'BB_upper', 'BB_lower', 'BB_width', 'Close_to_BB_upper', 'Close_to_BB_lower',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Stochastic_K', 'Stochastic_D', 'TR', 'ATR', 'OBV',
                'ADX', 'CCI', 'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', 'ROC', 'VWAP',
                'Momentum_20d', 'Close_to_MA_ratio'
            ]
            for col in technical_features:
                if col in group.columns:
                    if group[col].isna().all():
                        logger.warning(f"Kolumna {col} zawiera tylko NaN dla {group['Ticker'].iloc[0]}, wypełniam zerami")
                        group[col] = group[col].fillna(0)
                    else:
                        group[col] = group[col].ffill().bfill()
                        if group[col].isna().any():
                            logger.warning(f"Kolumna {col} nadal zawiera NaN po ffill/bfill, wypełniam średnią")
                            group[col] = group[col].fillna(group[col].mean())

            return group

        df = df.groupby('Ticker').apply(apply_features).reset_index(drop=True)
        df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Sector'])
        logger.info(f"Długość danych po dropna kluczowych kolumn: {len(df)}")
        return df

class PreprocessingUtils:
    """Klasa do współdzielenia logiki preprocessingu danych giełdowych."""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.normalizers_path = Path(config['data']['normalizers_path'])
        self.config_manager = ConfigManager()
        self.day_of_week_categories = [str(i) for i in range(7)]
        self.month_categories = [str(i) for i in range(1, 13)]
        self.sectors = self.config_manager.get_sectors()
        self.numeric_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI",
            "MACD", "MACD_Signal", "MACD_Histogram", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
            "ADX", "CCI", "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "ROC", "VWAP",
            "Momentum_20d", "Close_to_MA_ratio", "BB_width", "Close_to_BB_upper", "Close_to_BB_lower",
            "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility"
        ]
        self.log_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR", "BB_width",
            "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "VWAP"
        ]
        self.categorical_features = ["Day_of_Week", "Month"]

    def load_normalizers(self) -> dict:
        """Wczytuje normalizery z pliku."""
        try:
            with open(self.normalizers_path, 'rb') as f:
                normalizers = pickle.load(f)
            logger.info(f"Wczytano normalizery z: {self.normalizers_path}")
            return normalizers
        except Exception as e:
            logger.error(f"Błąd wczytywania normalizerów: {e}")
            return {}

    def preprocess_dataframe(self, df: pd.DataFrame, ticker: str = None, historical_mode: bool = False, trim_days: int = 0) -> pd.DataFrame:
        """Preprocesuje ramkę danych, dodając cechy i normalizując."""
        if df.empty:
            raise ValueError("Ramka danych jest pusta.")

        if ticker:
            df = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
        
        # Zapisz oryginalne Close przed preprocessingiem
        original_close = df['Close'].copy()
        logger.info(f"Początkowa długość df: {len(df)}, original_close: {len(original_close)}")
        
        if historical_mode and trim_days > 0:
            df = df.iloc[:-trim_days].copy()
            original_close = original_close.iloc[:-trim_days].copy()
            logger.info(f"Po przycięciu (historical_mode): df: {len(df)}, original_close: {len(original_close)}")

        # Dodaj cechy
        df = self.feature_engineer.add_features(df)
        logger.info(f"Po add_features: df: {len(df)}")
        
        # Zachowaj oryginalne indeksy przed dropna
        original_indices = df.index
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        logger.info(f"Po dropna: df: {len(df)}, usunięto rekordy: {set(original_indices) - set(df.index)}")
        # Dopasuj original_close do przefiltrowanych indeksów
        original_close = original_close.loc[df.index]
        
        df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])]
        logger.info(f"Po filtrze Close > 0 i High >= Low: df: {len(df)}")
        # Ponownie dopasuj original_close
        original_close = original_close.loc[df.index]
        
        df = self.feature_engineer.remove_outliers(df, 'Close')
        logger.info(f"Po remove_outliers: df: {len(df)}")
        # Ostateczne dopasowanie original_close
        original_close = original_close.loc[df.index]
        logger.info(f"Ostateczna długość df: {len(df)}, original_close: {len(original_close)}")

        # Ustaw kategorie i time_idx
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
        df['group_id'] = df['Ticker']
        
        df['Day_of_Week'] = pd.Categorical(df['Date'].dt.dayofweek.astype(str), categories=self.day_of_week_categories, ordered=False)
        df['Month'] = pd.Categorical(df['Date'].dt.month.astype(str), categories=self.month_categories, ordered=False)
        df['Sector'] = pd.Categorical(df['Sector'], categories=self.sectors, ordered=False)

        # Logarytmowanie cech
        for feature in self.log_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        # Normalizacja
        normalizers = self.load_normalizers()
        for feature in self.numeric_features:
            if feature in df.columns and feature in normalizers:
                df[feature] = normalizers[feature].transform(df[feature].values)

        # Konwersja kategorycznych
        for cat_col in self.categorical_features:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str)

        logger.info(f"Kolumny po preprocessingu: {df.columns.tolist()}")
        logger.info(f"Długość df po preprocessingu: {len(df)}, długość original_close: {len(original_close)}")
        return df, original_close

    def create_dataset(self, df: pd.DataFrame, dataset_params: dict = None, predict_mode: bool = False) -> TimeSeriesDataSet:
        """Tworzy TimeSeriesDataSet z preprocesowanej ramki danych."""
        normalizers = self.load_normalizers()
        valid_numeric_features = [
            f for f in self.numeric_features 
            if f in df.columns and f in normalizers and not df[f].isna().any() and not np.isinf(df[f]).any()
        ]
        valid_categorical_features = [f for f in self.categorical_features if f in df.columns]

        dataset_args = {
            "data": df,
            "time_idx": "time_idx",
            "target": "Relative_Returns",
            "group_ids": ["group_id"],
            "min_encoder_length": self.config['model']['min_encoder_length'],
            "max_encoder_length": self.config['model']['max_encoder_length'],
            "max_prediction_length": self.config['model']['max_prediction_length'],
            "static_categoricals": ["Sector"],
            "time_varying_known_reals": [f for f in valid_numeric_features if f not in ["Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility"]],
            "time_varying_known_categoricals": valid_categorical_features,
            "time_varying_unknown_reals": ["Relative_Returns"],
            "target_normalizer": normalizers.get("Relative_Returns", TorchNormalizer()),
            "allow_missing_timesteps": True,
            "add_encoder_length": False,
            "categorical_encoders": {
                'Sector': NaNLabelEncoder(add_nan=False),
                'Day_of_Week': NaNLabelEncoder(add_nan=False),
                'Month': NaNLabelEncoder(add_nan=False)
            }
        }

        if dataset_params:
            dataset = TimeSeriesDataSet.from_parameters(dataset_params, df, predict_mode=predict_mode)
        else:
            dataset = TimeSeriesDataSet(**dataset_args)
            dataset.save(self.config['data']['processed_data_path'])

        return dataset