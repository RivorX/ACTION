import pandas as pd
import numpy as np
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer, NaNLabelEncoder
import pytorch_forecasting
import torch
import pickle
import logging
from pathlib import Path
import time

import sys
import os
# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
    def calculate_macd(prices: pd.Series) -> pd.Series:
        """Oblicza MACD."""
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        return macd

    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 20) -> pd.Series:
        """Oblicza Price Rate of Change (ROC)."""
        return 100 * (prices - prices.shift(period)) / prices.shift(period)

    @staticmethod
    def calculate_vwap(group: pd.DataFrame) -> pd.Series:
        """Oblicza Volume Weighted Average Price (VWAP)."""
        typical_price = (group['Close'] + group['Close'] + group['Close']) / 3
        vwap = (typical_price * group['Volume']).cumsum() / group['Volume'].cumsum()
        return vwap

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
        """Usuwa wartości odstające na podstawie z-score."""
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]

    def add_features(self, df: pd.DataFrame, sectors_list=None) -> pd.DataFrame:
        """Dodaje nowe cechy do ramki danych z grupowaniem po Ticker."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)

        def apply_features(group):
            group = group.sort_values('Date')

            # Podstawowe średnie kroczące
            group['MA10'] = group['Close'].rolling(window=10).mean()
            group['MA50'] = group['Close'].rolling(window=50).mean()
            
            # Bollinger Bands (tylko górna granica)
            group['BB_upper'] = group['Close'].rolling(window=20).mean() + 2 * group['Close'].rolling(window=20).std()
            group['Close_to_BB_upper'] = group['Close'] / group['BB_upper']

            # Wskaźniki techniczne
            group['RSI'] = self.compute_rsi(group['Close'])
            group['MACD'] = self.calculate_macd(group['Close'])
            group['ROC'] = self.calculate_roc(group['Close'])
            group['VWAP'] = self.calculate_vwap(group)

            # Dodatkowe cechy
            group['Momentum_20d'] = group['Close'] - group['Close'].shift(20)
            group['Close_to_MA_ratio'] = group['Close'] / group['MA50']
            group['Relative_Returns'] = group['Close'].pct_change()

            group['Month'] = group['Date'].dt.month.astype(str)
            group['Day_of_Week'] = group['Date'].dt.dayofweek.astype(str)

            # Wypełnianie brakujących wartości dla Relative_Returns
            nan_count = group['Relative_Returns'].isna().sum()
            if nan_count > 0:
                group['Relative_Returns'] = group['Relative_Returns'].fillna(0)
            
            # Wypełnianie brakujących wartości dla innych cech
            features_to_fill = [
                'MA10', 'MA50', 'BB_upper', 'Close_to_BB_upper',
                'RSI', 'MACD', 'ROC', 'VWAP',
                'Momentum_20d', 'Close_to_MA_ratio'
            ]
            for feature in features_to_fill:
                if feature in group.columns:
                    group[feature] = group[feature].ffill().bfill()
            
            return group

        df = df.groupby('Ticker').apply(apply_features).reset_index(drop=True)
        
        if sectors_list:
            df['Sector'] = pd.Categorical(df['Sector'], categories=sectors_list, ordered=False)
        
        return df

class DataPreprocessor:
    def __init__(self, config: dict):
        """Inicjalizuje DataPreprocessor z konfiguracją."""
        self.config = config
        self.model_name = config['model_name']
        self.config_manager = ConfigManager()
        self.day_of_week_categories = [str(i) for i in range(7)]
        self.train_processed_df_path = Path(config['data']['train_processed_df_path'])
        self.val_processed_df_path = Path(config['data']['val_processed_df_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        self.gap_days = 10  # Stała wartość gap, można dodać do config jeśli potrzeba

    def _split_with_gap(self, df: pd.DataFrame) -> tuple:
        """Dzieli dane na train i val z gap'em 10 dni per ticker."""
        def split_group(group):
            group = group.sort_values('Date')
            total_days = (group['Date'].max() - group['Date'].min()).days
            train_days = int(0.8 * total_days)
            split_date = group['Date'].min() + pd.Timedelta(days=train_days)
            train = group[group['Date'] <= split_date]
            val_start_date = split_date + pd.Timedelta(days=self.gap_days + 1)
            val = group[group['Date'] >= val_start_date]
            return train, val

        trains = []
        vals = []
        for name, group in df.groupby('Ticker'):
            t, v = split_group(group)
            if not t.empty:
                trains.append(t)
            if not v.empty:
                vals.append(v)

        train_df = pd.concat(trains).reset_index(drop=True)
        val_df = pd.concat(vals).reset_index(drop=True)
        return train_df, val_df

    def process_data(self, mode: str = 'train', df: pd.DataFrame = None, normalizers: dict = None, ticker: str = None, historical_mode: bool = False, trim_days: int = 0):
        """Przetwarza dane dla trybu treningu lub predykcji."""
        start_time = time.time()
        numeric_features = [
            "Close", "Volume", "MA10", "MA50", "RSI", "MACD", "ROC", "VWAP",
            "Momentum_20d", "Close_to_MA_ratio", "Close_to_BB_upper", "Relative_Returns"
        ]

        if mode == 'train':
            if df is None:
                logger.error("Brak danych wejściowych dla trybu 'train'")
                raise ValueError("DataFrame musi być dostarczony dla trybu 'train'")
            if historical_mode:
                logger.warning("historical_mode=True w trybie 'train' zostanie zignorowane")
            if trim_days > 0:
                logger.warning("trim_days>0 w trybie 'train' zostanie zignorowane")
        elif mode == 'predict':
            if df is None or ticker is None:
                logger.error("Brak danych lub tickera dla trybu 'predict'")
                raise ValueError("DataFrame i ticker muszą być dostarczone dla trybu 'predict'")
            if historical_mode and trim_days > 0:
                df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=trim_days)]
            original_close = df['Close'].copy()
            df['Ticker'] = ticker

        feature_engineer = FeatureEngineer()
        df = feature_engineer.add_features(df, sectors_list=self.config['model']['sectors'])

        df['group_id'] = ticker if mode == 'predict' else df['Ticker']
        df = df.sort_values(['group_id', 'Date'])
        df['time_idx'] = df.groupby('group_id').cumcount()

        df['Day_of_Week'] = df['Date'].dt.dayofweek.astype(str)
        if df['Day_of_Week'].isna().any():
            logger.warning(f"Znaleziono NaN w Day_of_Week, wypełniam wartością '0'")
            df['Day_of_Week'] = df['Day_of_Week'].fillna('0')
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=self.day_of_week_categories, ordered=False)
        
        df['Sector'] = pd.Categorical(df['Sector'], categories=self.config['model']['sectors'], ordered=False)
        
        log_features = [
            "Close", "Volume", "MA10", "MA50", "VWAP"
        ]
        for feature in log_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        # Obsługa normalizerów
        if mode == 'train':
            # Split na train i val z gap
            train_df, val_df = self._split_with_gap(df)
            if train_df.empty or val_df.empty:
                raise ValueError(f"Zbiory po splicie są puste: train={len(train_df)}, val={len(val_df)}")

            normalizers_path = Path(f"models/normalizers/{self.model_name}_normalizers.pkl")
            if normalizers_path.exists():
                normalizers = self.config_manager.load_normalizers(self.model_name)
                logger.info(f"Załadowano istniejące normalizery dla modelu {self.model_name} – używam transform.")
                valid_numeric_features = []
                for feature in numeric_features:
                    if feature in train_df.columns and feature in normalizers:
                        try:
                            train_df[feature] = normalizers[feature].transform(train_df[feature].values)
                            val_df[feature] = normalizers[feature].transform(val_df[feature].values)
                            if train_df[feature].isna().any() or np.isinf(train_df[feature]).any() or val_df[feature].isna().any() or np.isinf(val_df[feature]).any():
                                logger.error(f"Transformacja cechy {feature} spowodowała NaN lub inf")
                            else:
                                valid_numeric_features.append(feature)
                        except Exception as e:
                            logger.error(f"Błąd podczas transformacji cechy {feature}: {e}")
                    elif feature in train_df.columns:
                        logger.warning(f"Cecha {feature} nie znajduje się w istniejących normalizerach, pomijam normalizację.")
            else:
                logger.info(f"Tworzenie nowych normalizerów i zapis dla modelu: {self.model_name}")
                normalizers = {}
                valid_numeric_features = []
                for feature in numeric_features:
                    if feature in train_df.columns:
                        has_nan = train_df[feature].isna().any()
                        has_inf = np.isinf(train_df[feature]).any()
                        unique_count = train_df[feature].nunique()
                        
                        if has_nan or has_inf:
                            logger.warning(f"Cecha {feature} zawiera NaN ({has_nan}) lub inf ({has_inf}), pomijam w time_varying_known_reals")
                        elif unique_count <= 1:
                            logger.warning(f"Cecha {feature} ma tylko {unique_count} unikalnych wartości, może powodować problemy z normalizacją")
                            valid_numeric_features.append(feature)
                        else:
                            try:
                                normalizer = TorchNormalizer()
                                values = train_df[feature].values
                                train_df[feature] = normalizer.fit_transform(values)
                                normalizers[feature] = normalizer
                                
                                # Transform val
                                val_df[feature] = normalizer.transform(val_df[feature].values)
                                
                                if train_df[feature].isna().any() or np.isinf(train_df[feature]).any() or val_df[feature].isna().any() or np.isinf(val_df[feature]).any():
                                    logger.error(f"Normalizacja cechy {feature} spowodowała NaN lub inf, usuwam tę cechę")
                                    del normalizers[feature]
                                    if feature in valid_numeric_features:
                                        valid_numeric_features.remove(feature)
                                else:
                                    logger.info(f"Normalizacja cechy {feature} zakończona pomyślnie: min={train_df[feature].min():.6f}, max={train_df[feature].max():.6f}")
                                    valid_numeric_features.append(feature)
                                

                            except Exception as e:
                                logger.error(f"Błąd podczas normalizacji cechy {feature}: {e}")
                                if feature in valid_numeric_features:
                                    valid_numeric_features.remove(feature)
                    else:
                        logger.warning(f"Cecha {feature} nie znajduje się w danych, pomijam")
                
                # Zapisz normalizery
                self.config_manager.save_normalizers(self.model_name, normalizers)
                logger.info(f"Normalizery zapisane dla modelu: {self.model_name}")

            categorical_columns = ['Day_of_Week', 'Month']
            for cat_col in categorical_columns:
                if cat_col in train_df.columns:
                    train_df[cat_col] = train_df[cat_col].astype(str)
                if cat_col in val_df.columns:
                    val_df[cat_col] = val_df[cat_col].astype(str)

            # Zapisz przetworzone df
            train_df.to_pickle(self.train_processed_df_path)
            val_df.to_pickle(self.val_processed_df_path)
            logger.info(f"Przetworzony train DataFrame zapisany do: {self.train_processed_df_path}")
            logger.info(f"Przetworzony val DataFrame zapisany do: {self.val_processed_df_path}")

            targets = ["Relative_Returns"]
            
            categorical_features = ["Day_of_Week", "Month"]
            valid_categorical_features = [f for f in categorical_features if f in train_df.columns]
            
            logger.info(f"Kategorie dla Day_of_Week: {self.day_of_week_categories}")
            logger.info(f"Kategorie dla Sector: {self.config['model']['sectors']}")
            logger.info(f"Finalna lista cech numerycznych ({len(valid_numeric_features)}): {valid_numeric_features}")
            logger.info(f"Finalna lista cech kategorycznych ({len(valid_categorical_features)}): {valid_categorical_features}")

            # Twórz dataset tylko na train_df (dla parametrów bez wycieku)
            dataset = TimeSeriesDataSet(
                train_df,
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
                    'Sector': NaNLabelEncoder(add_nan=False),
                    'Day_of_Week': NaNLabelEncoder(add_nan=False),
                    'Month': NaNLabelEncoder(add_nan=False)
                }
            )
            logger.info(f"Target normalizer: {dataset.target_normalizer}")
            
            dataset.save(self.processed_data_path)
            total_duration = time.time() - start_time
            logger.info(f"Całkowity czas process_data (train): {total_duration:.3f} sekundy")
            logger.info(f"Kolumny przetworzonego train_df: {train_df.columns.tolist()}")
            return dataset

        elif mode == 'predict':
            # Użyj istniejących normalizerów
            for feature in numeric_features:
                if feature in df.columns and feature in normalizers:
                    try:
                        df[feature] = normalizers[feature].transform(df[feature].values)
                        if df[feature].isna().any() or np.isinf(df[feature]).any():
                            logger.error(f"Transformacja cechy {feature} spowodowała NaN lub inf")
                    except Exception as e:
                        logger.error(f"Błąd podczas transformacji cechy {feature}: {e}")

            categorical_columns = ['Day_of_Week', 'Month']
            for cat_col in categorical_columns:
                if cat_col in df.columns:
                    df[cat_col] = df[cat_col].astype(str)

            total_duration = time.time() - start_time
            logger.info(f"Całkowity czas process_data (predict): {total_duration:.3f} sekundy")
            logger.info(f"Kolumny przetworzonego df: {df.columns.tolist()}")
            return df, original_close