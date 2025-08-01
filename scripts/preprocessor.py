import pandas as pd
import numpy as np
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
import pytorch_forecasting
import torch
import pickle
import logging
from pathlib import Path
from scripts.config_manager import ConfigManager

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
        # Zabezpieczenie przed dzieleniem przez zero
        stochastic_k = 100 * (group['Close'] - low_14) / denominator.where(denominator != 0, 1e-10)
        # Zamiana inf na 0
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

            # Podstawowe średnie kroczące
            group['MA10'] = group['Close'].rolling(window=10).mean()
            group['MA50'] = group['Close'].rolling(window=50).mean()
            
            # Bollinger Bands
            group['BB_upper'] = group['Close'].rolling(window=20).mean() + 2 * group['Close'].rolling(window=20).std()
            group['BB_lower'] = group['Close'].rolling(window=20).mean() - 2 * group['Close'].rolling(window=20).std()
            group['BB_width'] = group['BB_upper'] - group['BB_lower']
            group['Close_to_BB_upper'] = group['Close'] / group['BB_upper']
            group['Close_to_BB_lower'] = group['Close'] / group['BB_lower']

            # Wskaźniki techniczne
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

            # Targety
            group['Relative_Returns'] = group['Close'].pct_change().shift(-1)
            group['Log_Returns'] = np.log(group['Close'] / group['Close'].shift(1)).shift(-1)
            group['Future_Volume'] = group['Volume'].shift(-1)
            group['Future_Volatility'] = group['Close'].rolling(window=20).std().shift(-1)

            # Cechy kategoryczne
            group['Day_of_Week'] = group['Date'].dt.dayofweek.astype(str)
            group['Month'] = group['Date'].dt.month.astype(str)
            # Wypełnianie NaN w Day_of_Week i Month przed przekształceniem na typ kategoryczny
            if group['Day_of_Week'].isna().any():
                logger.warning(f"Znaleziono {group['Day_of_Week'].isna().sum()} NaN w Day_of_Week dla tickera {group['Ticker'].iloc[0]}, wypełniam wartością '0'")
                group['Day_of_Week'] = group['Day_of_Week'].fillna('0')
            if group['Month'].isna().any():
                logger.warning(f"Znaleziono {group['Month'].isna().sum()} NaN w Month dla tickera {group['Ticker'].iloc[0]}, wypełniam wartością '1'")
                group['Month'] = group['Month'].fillna('1')

            # Przekształcenie na typ kategoryczny
            group['Day_of_Week'] = pd.Categorical(group['Day_of_Week'], categories=[str(i) for i in range(7)], ordered=False)
            group['Month'] = pd.Categorical(group['Month'], categories=[str(i) for i in range(1, 13)], ordered=False)

            # Wypełnianie NaN dla targetów
            for col in ['Relative_Returns', 'Log_Returns', 'Future_Volume', 'Future_Volatility']:
                group[col] = group[col].fillna(0)

            # Wypełnianie NaN dla cech technicznych
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

        # Usuwamy tylko wiersze z brakami w kluczowych danych
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Sector']
        df = df.dropna(subset=required_cols)
        logger.info(f"Długość danych po dropna kluczowych kolumn: {len(df)}")

        # Upewnij się, że Sector jest kategoryczny
        config_manager = ConfigManager()
        df['Sector'] = pd.Categorical(df['Sector'], categories=config_manager.get_sectors(), ordered=False)
        logger.info(f"Kategorie sektorów: {df['Sector'].cat.categories.tolist()}")

        # Logowanie brakujących danych w innych kolumnach
        for col in df.columns:
            if col not in required_cols and df[col].isna().any():
                logger.warning(f"Kolumna {col} zawiera wartości NaN dla tickera {df['Ticker'].iloc[0] if 'Ticker' in df else 'nieznany'} (liczba NaN: {df[col].isna().sum()})")

        return df

class DataPreprocessor:
    """Klasa odpowiedzialna za preprocessing danych giełdowych i tworzenie zbioru danych TimeSeriesDataSet."""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.normalizers_path = Path(config['data']['normalizers_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        self.day_of_week_categories = [str(i) for i in range(7)]
        self.config_manager = ConfigManager()

    def preprocess_data(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        if df.empty:
            raise ValueError("Ramka danych jest pusta. Sprawdź dane wejściowe.")

        df = self.feature_engineer.add_features(df)
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])]
        df = self.feature_engineer.remove_outliers(df, 'Close')

        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
        df['group_id'] = df['Ticker']

        df['Day_of_Week'] = df['Date'].dt.dayofweek.astype(str)
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=self.day_of_week_categories, ordered=False)

        df['Sector'] = pd.Categorical(df['Sector'], categories=self.config_manager.get_sectors(), ordered=False)
        logger.info(f"Kategorie sektorów w preprocess_data: {df['Sector'].cat.categories.tolist()}")

        numeric_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI",
            "MACD", "MACD_Signal", "MACD_Histogram", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
            "ADX", "CCI", "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "ROC", "VWAP",
            "Momentum_20d", "Close_to_MA_ratio", "BB_width", "Close_to_BB_upper", "Close_to_BB_lower",
            "Relative_Returns"
        ]

        log_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR", "BB_width",
            "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "VWAP"
        ]
        for feature in log_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        valid_numeric_features = []
        normalizers = {}
        for feature in numeric_features:
            if feature in df.columns:
                has_nan = df[feature].isna().any()
                has_inf = np.isinf(df[feature]).any()
                unique_count = df[feature].nunique()
                
                if has_nan or has_inf:
                    logger.warning(f"Cecha {feature} zawiera NaN ({has_nan}) lub inf ({has_inf}), pomijam w time_varying_known_reals")
                elif unique_count <= 1:
                    logger.warning(f"Cecha {feature} ma tylko {unique_count} unikalnych wartości, może powodować problemy z normalizacją")
                    valid_numeric_features.append(feature)
                else:
                    try:
                        normalizer = TorchNormalizer()
                        values = df[feature].values
                        df[feature] = normalizer.fit_transform(values)
                        normalizers[feature] = normalizer
                        
                        if df[feature].isna().any() or np.isinf(df[feature]).any():
                            logger.error(f"Normalizacja cechy {feature} spowodowała NaN lub inf, usuwam tę cechę")
                            del normalizers[feature]
                            if feature in valid_numeric_features:
                                valid_numeric_features.remove(feature)
                        else:
                            logger.info(f"Normalizacja cechy {feature} zakończona pomyślnie: min={df[feature].min():.6f}, max={df[feature].max():.6f}")
                            valid_numeric_features.append(feature)
                    except Exception as e:
                        logger.error(f"Błąd podczas normalizacji cechy {feature}: {e}")
                        if feature in valid_numeric_features:
                            valid_numeric_features.remove(feature)

        with open(self.normalizers_path, 'wb') as f:
            pickle.dump(normalizers, f)
        logger.info(f"Normalizery zapisane do: {self.normalizers_path}")

        targets = ["Relative_Returns", "Future_Volume", "Future_Volatility"]
        
        categorical_features = ["Day_of_Week", "Month"]
        valid_categorical_features = [f for f in categorical_features if f in df.columns]
        
        logger.info(f"Kategorie dla Day_of_Week: {self.day_of_week_categories}")
        logger.info(f"Kategorie dla Sector: {self.config_manager.get_sectors()}")

        logger.info(f"Finalna lista cech numerycznych ({len(valid_numeric_features)}): {valid_numeric_features}")
        logger.info(f"Finalna lista cech kategorycznych ({len(valid_categorical_features)}): {valid_categorical_features}")

        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="Relative_Returns",
            group_ids=["group_id"],
            min_encoder_length=self.config['model']['min_encoder_length'],
            max_encoder_length=self.config['model']['max_encoder_length'],
            max_prediction_length=self.config['model']['max_prediction_length'],
            static_categoricals=["Sector"],
            time_varying_known_reals=[f for f in valid_numeric_features if f not in targets],
            time_varying_known_categoricals=valid_categorical_features,
            time_varying_unknown_reals=["Relative_Returns"],
            target_normalizer=normalizers.get("Relative_Returns", TorchNormalizer()),
            allow_missing_timesteps=True,
            add_encoder_length=False,
            categorical_encoders={
                'Sector': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=False),
                'Day_of_Week': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=False),
                'Month': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=False)
            }
        )
        logger.info(f"Target normalizer: {dataset.target_normalizer}")
        
        dataset.save(self.processed_data_path)
        return dataset