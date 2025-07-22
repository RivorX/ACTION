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

            # Dodanie zmiennych wskazujących brakujące dane fundamentalne
            for col in ['PE_ratio', 'PB_ratio', 'EPS']:
                if col in group.columns:
                    group[f'{col}_missing'] = group[col].isna().astype(int)  # Dodajemy kolumnę wskazującą NaN
                    if group[col].isna().all():
                        logger.warning(f"Wypełniam {col} zerami dla {group['Ticker'].iloc[0]}")
                        group[col] = 0.0  # Wypełniamy zerami, jeśli wszystkie wartości są NaN
                    else:
                        group[col] = group[col].interpolate(method='linear').ffill().bfill()  # Interpolacja dla częściowych braków

            # Sprawdzenie poprawności danych fundamentalnych
            if group['EPS'].isna().all() or group['PE_ratio'].isna().all() or group['PB_ratio'].isna().all():
                logger.warning(f"Brak danych fundamentalnych dla grupy {group['Ticker'].iloc[0]}, używane będą tylko dane techniczne")

            # Wskaźniki techniczne
            group['RSI'] = self.compute_rsi(group['Close'])
            group['Volatility'] = group['Close'].rolling(window=20).std()
            group['MACD'], group['MACD_Signal'] = self.calculate_macd(group['Close'])
            group['Stochastic_K'] = self.calculate_stochastic_k(group)
            group['Stochastic_D'] = group['Stochastic_K'].rolling(window=3).mean()
            group['TR'] = self.calculate_true_range(group)
            group['ATR'] = group['TR'].rolling(window=14).mean()
            group['OBV'] = self.calculate_obv(group)

            # Feature engineering na Close
            group['Close_momentum_1d'] = group['Close'] - group['Close'].shift(1)
            group['Close_momentum_5d'] = group['Close'] - group['Close'].shift(5)
            group['Close_vs_MA10'] = group['Close'] / group['MA10']
            group['Close_vs_MA50'] = group['Close'] / group['MA50']
            group['Close_percentile_20d'] = group['Close'].rolling(window=20).rank(pct=True)
            group['Close_volatility_5d'] = group['Close'].rolling(window=5).std()
            close_trend = group['Close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            rsi_trend = group['RSI'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            group['Close_RSI_divergence'] = close_trend - rsi_trend
            group['Relative_Returns'] = group['Close'].pct_change().shift(-1)
            group['Log_Returns'] = np.log(group['Close'] / group['Close'].shift(1)).shift(-1)
            group['Future_Volume'] = group['Volume'].shift(-1)
            group['Future_Volatility'] = group['Volatility'].shift(-1)
            group['Day_of_Week'] = group['Date'].dt.dayofweek.astype(str)
            group['Month'] = group['Date'].dt.month.astype(str)

            # Wypełnianie NaN dla kolumn z shift(-1)
            for col in ['Relative_Returns', 'Log_Returns', 'Future_Volume', 'Future_Volatility']:
                group[col] = group[col].fillna(0)

            # Wypełnianie NaN dla cech technicznych
            technical_features = [
                'MA10', 'MA50', 'BB_upper', 'BB_lower', 'BB_width', 'Close_to_BB_upper', 'Close_to_BB_lower',
                'RSI', 'Volatility', 'MACD', 'MACD_Signal', 'Stochastic_K', 'Stochastic_D', 'TR', 'ATR', 'OBV',
                'Close_momentum_1d', 'Close_momentum_5d', 'Close_vs_MA10', 'Close_vs_MA50',
                'Close_percentile_20d', 'Close_volatility_5d', 'Close_RSI_divergence'
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

        # Usuwamy tylko wiersze z brakami w kluczowych danych technicznych
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.dropna(subset=required_cols)
        logger.info(f"Długość danych po dropna kluczowych kolumn: {len(df)}")

        # Logowanie brakujących danych w innych kolumnach
        for col in df.columns:
            if col not in required_cols and df[col].isna().any():
                logger.warning(f"Kolumna {col} zawiera wartości NaN dla tickera {df['Ticker'].iloc[0] if 'Ticker' in df else 'nieznany'} (liczba NaN: {df[col].isna().sum()})")

        # Dodatkowe logowanie pierwszych 5 wierszy dla debugowania
        logger.info(f"Pierwsze 5 wierszy po dodaniu cech:\n{df.head().to_string()}")

        return df

class DataPreprocessor:
    """Klasa odpowiedzialna za preprocessing danych giełdowych i tworzenie zbioru danych TimeSeriesDataSet."""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.normalizers_path = Path(config['data']['normalizers_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])

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

        # Rozszerzona lista cech logarytmowanych - specjalna obsługa dla cech fundamentalnych
        log_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR", "BB_width"
        ]
        fundamental_features = ["PE_ratio", "PB_ratio", "EPS"]
        
        # Standardowa transformacja log dla cech technicznych
        for feature in log_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))
        
        # Specjalna obsługa dla cech fundamentalnych
        for feature in fundamental_features:
            if feature in df.columns:
                # Sprawdź czy cecha ma jakiekolwiek niezerowe wartości
                nonzero_count = (df[feature] != 0.0).sum()
                total_count = len(df[feature])
                
                if nonzero_count == 0:
                    logger.warning(f"Cecha {feature} zawiera wyłącznie zera, pomijam transformację log")
                    # Pozostaw jako zera - nie rób transformacji log
                else:
                    logger.info(f"Cecha {feature}: {nonzero_count}/{total_count} ({100*nonzero_count/total_count:.1f}%) niezerowych wartości")
                    # Zastosuj transformację log tylko do wartości niezerowych
                    mask = df[feature] > 0
                    if mask.any():
                        df.loc[mask, feature] = np.log1p(df.loc[mask, feature])
                    # Wartości zerowe pozostają zerami

        # Lista cech numerycznych
        numeric_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
            "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
            "Close_momentum_1d", "Close_momentum_5d", "Close_vs_MA10", "Close_vs_MA50",
            "Close_percentile_20d", "Close_volatility_5d", "Close_RSI_divergence",
            "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility",
            "BB_width", "Close_to_BB_upper", "Close_to_BB_lower", "PE_ratio", "PB_ratio", "EPS"
        ]

        # Filtruj cechy numeryczne, usuwając te, które mają problemy
        valid_numeric_features = []
        for feature in numeric_features:
            if feature in df.columns:
                # Sprawdź podstawowe problemy
                has_nan = df[feature].isna().any()
                has_inf = np.isinf(df[feature]).any()
                all_zeros = (df[feature] == 0.0).all()
                unique_count = df[feature].nunique()
                
                if has_nan or has_inf:
                    logger.warning(f"Cecha {feature} zawiera NaN ({has_nan}) lub inf ({has_inf}), pomijam w time_varying_known_reals")
                elif all_zeros and feature in ['PE_ratio', 'PB_ratio', 'EPS']:
                    logger.warning(f"Cecha {feature} zawiera wyłącznie zera, pomijam w time_varying_known_reals ale zachowuję zmienną missing")
                elif unique_count <= 1:
                    logger.warning(f"Cecha {feature} ma tylko {unique_count} unikalnych wartości, może powodować problemy z normalizacją")
                    # Dla cech fundamentalnych z jedną wartością, lepiej je pominąć
                    if feature in ['PE_ratio', 'PB_ratio', 'EPS']:
                        logger.warning(f"Pomijam cechę fundamentalną {feature} z powodu braku różnorodności wartości")
                        continue
                    else:
                        valid_numeric_features.append(feature)
                else:
                    valid_numeric_features.append(feature)

        normalizers = {}
        for feature in valid_numeric_features:
            if feature in df.columns:
                try:
                    normalizer = TorchNormalizer()
                    
                    # Pobierz wartości dla wszystkich cech
                    values = df[feature].values
                    
                    # Specjalna obsługa dla cech fundamentalnych
                    if feature in ['PE_ratio', 'PB_ratio', 'EPS']:
                        # Sprawdź czy wartości są w rozsądnym zakresie
                        if len(np.unique(values)) < 2:
                            logger.warning(f"Cecha {feature} ma mniej niż 2 unikalne wartości, pomijam normalizację")
                            continue
                        
                        # Sprawdź zakres wartości po transformacji log
                        min_val, max_val = values.min(), values.max()
                        if max_val - min_val < 1e-6:  # Bardzo mały zakres
                            logger.warning(f"Cecha {feature} ma bardzo mały zakres wartości ({min_val:.6f} - {max_val:.6f}), może powodować problemy")
                    
                    df[feature] = normalizer.fit_transform(values)
                    normalizers[feature] = normalizer
                    
                    # Sprawdź wynik normalizacji
                    if df[feature].isna().any() or np.isinf(df[feature]).any():
                        logger.error(f"Normalizacja cechy {feature} spowodowała NaN lub inf, usuwam tę cechę")
                        del normalizers[feature]
                        # Usuń tę cechę z valid_numeric_features
                        if feature in valid_numeric_features:
                            valid_numeric_features.remove(feature)
                    else:
                        logger.info(f"Normalizacja cechy {feature} zakończona pomyślnie: min={df[feature].min():.6f}, max={df[feature].max():.6f}")
                        
                except Exception as e:
                    logger.error(f"Błąd podczas normalizacji cechy {feature}: {e}")
                    if feature in valid_numeric_features:
                        valid_numeric_features.remove(feature)

        with open(self.normalizers_path, 'wb') as f:
            pickle.dump(normalizers, f)
        logger.info(f"Normalizery zapisane do: {self.normalizers_path}")

        targets = ["Relative_Returns", "Future_Volume", "Future_Volatility"]
        
        # Dodajemy zmienne wskazujące brakujące dane do cech kategorycznych
        categorical_features = ["Day_of_Week", "Month", "PE_ratio_missing", "PB_ratio_missing", "EPS_missing"]
        valid_categorical_features = [f for f in categorical_features if f in df.columns]
        
        # WAŻNE: Konwertuj missing features na typ string (wymagane przez PyTorch Forecasting)
        for missing_col in ['PE_ratio_missing', 'PB_ratio_missing', 'EPS_missing']:
            if missing_col in df.columns:
                df[missing_col] = df[missing_col].astype(str)
                logger.info(f"Skonwertowano {missing_col} na typ string: {df[missing_col].unique()}")

        # Logowanie finalnej listy cech
        logger.info(f"Finalna lista cech numerycznych ({len(valid_numeric_features)}): {valid_numeric_features}")
        logger.info(f"Finalna lista cech kategorycznych ({len(valid_categorical_features)}): {valid_categorical_features}")
        
        # Sprawdź czy mamy cechy fundamentalne
        fundamental_in_features = [f for f in ['PE_ratio', 'PB_ratio', 'EPS'] if f in valid_numeric_features]
        missing_features = [f for f in ['PE_ratio_missing', 'PB_ratio_missing', 'EPS_missing'] if f in valid_categorical_features]
        logger.info(f"Cechy fundamentalne w modelu: {fundamental_in_features}")
        logger.info(f"Zmienne missing fundamentalne: {missing_features}")

        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="Relative_Returns",
            group_ids=["group_id"],
            min_encoder_length=self.config['model']['min_encoder_length'],
            max_encoder_length=self.config['model']['max_encoder_length'],
            max_prediction_length=self.config['model']['max_prediction_length'],
            time_varying_known_reals=[f for f in valid_numeric_features if f not in targets],
            time_varying_known_categoricals=valid_categorical_features,
            time_varying_unknown_reals=["Relative_Returns"],
            target_normalizer=normalizers.get("Relative_Returns", TorchNormalizer()),
            allow_missing_timesteps=True,
            add_encoder_length=False
        )
        logger.info(f"Target normalizer: {dataset.target_normalizer}")
        
        dataset.save(self.processed_data_path)
        return dataset