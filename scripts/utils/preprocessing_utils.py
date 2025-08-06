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
    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 30) -> pd.Series:
        """Oblicza Rate of Change (ROC) dla zadanego okresu."""
        return (prices / prices.shift(period) - 1).fillna(0)

    @staticmethod
    def calculate_dmi(group: pd.DataFrame, period: int = 14) -> tuple:
        """Oblicza DMI+ i DMI- (Directional Movement Index)."""
        up_move = group['High'].diff()
        down_move = group['Low'].diff().abs()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = FeatureEngineer.calculate_true_range(group)
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=1).sum() / tr.rolling(window=period, min_periods=1).sum()
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=1).sum() / tr.rolling(window=period, min_periods=1).sum()
        return plus_di.fillna(0), minus_di.fillna(0)

    @staticmethod
    def calculate_up_days_rolling(prices: pd.Series, window: int = 30) -> pd.Series:
        """Oblicza liczbę dni wzrostowych w oknie rolling."""
        up_days = (prices.diff() > 0).astype(int)
        return up_days.rolling(window=window, min_periods=1).sum().fillna(0)

    @staticmethod
    def calculate_rolling_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
        """Oblicza rolling volatility (std log returns) dla okna."""
        log_returns = np.log(prices / prices.shift(1)).fillna(0)
        return log_returns.rolling(window=window, min_periods=1).std().fillna(0)

    @staticmethod
    def calculate_atr(group: pd.DataFrame, period: int = 14) -> pd.Series:
        """Oblicza Average True Range (ATR)."""
        tr = FeatureEngineer.calculate_true_range(group)
        return tr.rolling(window=period, min_periods=1).mean().fillna(0)
    """Klasa do inżynierii cech dla danych giełdowych."""
    
    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Oblicza wskaźnik RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(0).clip(lower=0, upper=100)  # Zapewnienie zakresu [0, 100]
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series) -> tuple:
        """Oblicza MACD, linię sygnałową i histogram MACD z ograniczeniem zakresu."""
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        # Ograniczenie typowego zakresu MACD (np. -10 do 10)
        macd = macd.clip(-10, 10)
        signal = signal.clip(-10, 10)
        histogram = histogram.clip(-10, 10)
        return macd, signal, histogram

    @staticmethod
    def calculate_stochastic_k(group: pd.DataFrame) -> pd.Series:
        """Oblicza Stochastic %K z zabezpieczeniem przed dzieleniem przez zero."""
        low_14 = group['Low'].rolling(window=14).min()
        high_14 = group['High'].rolling(window=14).max()
        denominator = high_14 - low_14
        stochastic_k = 100 * (group['Close'] - low_14) / denominator.where(denominator != 0, 1e-10)
        stochastic_k = stochastic_k.replace([np.inf, -np.inf], 0).clip(lower=0, upper=100)  # Zapewnienie zakresu [0, 100]
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
    def calculate_ichimoku(group: pd.DataFrame) -> tuple:
        """Oblicza linie Ichimoku Cloud: Tenkan-sen, Kijun-sen, Senkou Span A."""
        high_9 = group['High'].rolling(window=9).max()
        low_9 = group['Low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2

        high_26 = group['High'].rolling(window=26).max()
        low_26 = group['Low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2

        senkou_span_a = (tenkan_sen + kijun_sen) / 2

        return tenkan_sen, kijun_sen, senkou_span_a

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
        """Usuwa wartości odstające na podstawie z-score."""
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        return df[abs(z_scores) < threshold]

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje nowe cechy do ramki danych z grupowaniem po Ticker, z pominięciem wybranych cech."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)

        def apply_features(group):
            group = group.sort_values('Date')
            group = group[(group['Close'] > 0) & (group['High'] >= group['Low'])]
            group = group.reset_index(drop=True)

            # Rolling mean/std z fillna przed clippingiem (usunięto MA50, BB_width)
            # ROC 30d
            group['ROC_30d'] = self.calculate_roc(group['Close'], period=30)
            # DMI+ i DMI-
            group['DMI_plus'], group['DMI_minus'] = self.calculate_dmi(group, period=14)
            # Up_Days_30d
            group['Up_Days_30d'] = self.calculate_up_days_rolling(group['Close'], window=30)
            # Rolling Volatility 30d
            group['Rolling_Volatility_30d'] = self.calculate_rolling_volatility(group['Close'], window=30)
            # ATR 14
            group['ATR_14'] = self.calculate_atr(group, period=14)
            group['BB_upper'] = (group['Close'].rolling(window=20, min_periods=1).mean() + 2 * group['Close'].rolling(window=20, min_periods=1).std()).bfill()
            group['BB_lower'] = (group['Close'].rolling(window=20, min_periods=1).mean() - 2 * group['Close'].rolling(window=20, min_periods=1).std()).bfill()

            group['RSI'] = self.compute_rsi(group['Close'])
            # MACD tylko histogram (bez sygnału)
            _, _, histogram = self.calculate_macd(group['Close'])
            group['MACD_Histogram'] = histogram
            # Stochastic_D: rolling mean z min_periods=1, fillna, potem przeskalowanie do [0,1] (bez Stochastic_K)
            # Stochastic_D bazuje na Stochastic_K, ale nie dodajemy Stochastic_K do cech
            stochastic_k = self.calculate_stochastic_k(group)
            group['Stochastic_D'] = stochastic_k.rolling(window=3, min_periods=1).mean().bfill()
            min_sd = group['Stochastic_D'].min()
            max_sd = group['Stochastic_D'].max()
            if max_sd > min_sd:
                group['Stochastic_D'] = (group['Stochastic_D'] - min_sd) / (max_sd - min_sd)
            group['Stochastic_D'] = group['Stochastic_D'].clip(0, 1)

            group['ADX'] = self.calculate_adx(group)
            # Ichimoku tylko Kijun_sen i Senkou_Span_A (bez Tenkan_sen)
            _, group['Kijun_sen'], group['Senkou_Span_A'] = self.calculate_ichimoku(group)
            group['Momentum_20d'] = (group['Close'] - group['Close'].shift(20)).clip(-1000, 1000)

            # --- AGRESYWNY CLIPPING I TRANSFORMACJE ---
            # Relative_Returns: clipping, sign-preserving log, fillna
            group['Relative_Returns'] = group['Close'].pct_change().shift(-1)
            group['Relative_Returns'] = group['Relative_Returns'].clip(-5, 5)
            group['Relative_Returns'] = group['Relative_Returns'].clip(-0.2, 0.2)

            # Log_Returns: clipping, sign-preserving log
            group['Log_Returns'] = np.log(group['Close'] / group['Close'].shift(1)).shift(-1)
            group['Log_Returns'] = group['Log_Returns'].clip(-0.2, 0.2)
            group['Log_Returns'] = np.sign(group['Log_Returns']) * np.log1p(np.abs(group['Log_Returns']))

            # Future_Volume: shift(-1), mocny clipping do percentyla 95, log1p
            group['Future_Volume'] = group['Volume'].shift(-1)
            fv_95 = group['Future_Volume'].quantile(0.95)
            group['Future_Volume'] = group['Future_Volume'].clip(1, fv_95)
            group['Future_Volume'] = np.log1p(group['Future_Volume'])

            # Future_Volatility: rolling std, shift(-1), clipping do percentyla 60, log1p
            group['Future_Volatility'] = group['Close'].rolling(window=20, min_periods=1).std().shift(-1)
            fv_60 = group['Future_Volatility'].quantile(0.60)
            group['Future_Volatility'] = group['Future_Volatility'].clip(1e-6, fv_60)
            group['Future_Volatility'] = np.log1p(group['Future_Volatility'])

            # Fillna na końcu dla wszystkich cech
            fillna_cols = [
                'Relative_Returns', 'Log_Returns', 'Future_Volume', 'Future_Volatility',
                'Stochastic_D', 'BB_upper', 'BB_lower',
                'RSI', 'MACD_Histogram', 'ADX', 'Kijun_sen', 'Senkou_Span_A', 'Momentum_20d',
                'ROC_30d', 'DMI_plus', 'DMI_minus', 'Up_Days_30d', 'Rolling_Volatility_30d', 'ATR_14'
            ]
            for col in fillna_cols:
                if col in group.columns:
                    group[col] = group[col].fillna(0)

            technical_features = [
                'BB_upper', 'BB_lower',
                'RSI', 'MACD_Histogram', 'Stochastic_D',
                'ADX', 'Kijun_sen', 'Senkou_Span_A', 'Momentum_20d',
                'ROC_30d', 'DMI_plus', 'DMI_minus', 'Up_Days_30d', 'Rolling_Volatility_30d', 'ATR_14'
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
        df = df.dropna(subset=['Date', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Sector'])
        logger.info(f"Długość danych po dropna kluczowych kolumn: {len(df)}")
        return df

class PreprocessingUtils:
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.normalizers_path = Path(config['data']['normalizers_path'])
        self.normalized_data_path = Path(config['data']['raw_data_path']).parent / 'normalized_data.csv'
        self.config_manager = ConfigManager()
        self.day_of_week_categories = [str(i) for i in range(7)]
        self.month_categories = [str(i) for i in range(1, 13)]
        self.sectors = self.config_manager.get_sectors()
        self.numeric_features = [
            "High", "Low", "Close", "Volume", "RSI",
            "MACD_Histogram", "Stochastic_D",
            "ADX", "Kijun_sen", "Senkou_Span_A", "Momentum_20d",
            "BB_upper", "BB_lower", "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility",
            "ROC_30d", "DMI_plus", "DMI_minus", "Up_Days_30d", "Rolling_Volatility_30d", "ATR_14"
        ]
        self.log_features = [
            "High", "Low", "Close", "Volume",
            "BB_upper", "BB_lower", "Kijun_sen", "Senkou_Span_A"
        ]
        self.categorical_features = ["Day_of_Week", "Month"]
        self.robust_features = self.numeric_features  # Wszystkie cechy używają RobustScaler

    def load_normalizers(self) -> dict:
        """Wczytuje normalizery z pliku."""
        if self.normalizers_path.exists():
            try:
                with open(self.normalizers_path, 'rb') as f:
                    normalizers = pickle.load(f)
                return normalizers
            except Exception as e:
                logger.error(f"Błąd wczytywania normalizerów: {e}")
                return {}
        else:
            logger.info(f"Plik normalizerów {self.normalizers_path} nie istnieje, zwracam pusty słownik")
            return {}

    def save_normalizers(self, normalizers: dict):
        """Zapisuje normalizery do pliku, jeśli plik jeszcze nie istnieje."""
        if not self.normalizers_path.exists():
            try:
                self.normalizers_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.normalizers_path, 'wb') as f:
                    pickle.dump(normalizers, f)
                logger.info(f"Zapisano normalizery do: {self.normalizers_path}")
            except Exception as e:
                logger.error(f"Błąd zapisu normalizerów: {e}")
                raise
        else:
            logger.info(f"Plik normalizerów {self.normalizers_path} już istnieje, pomijam zapis")

    def preprocess_dataframe(self, df: pd.DataFrame, ticker: str = None, historical_mode: bool = False, trim_days: int = 0) -> tuple:
        """Preprocesuje ramkę danych, dodając cechy i normalizując."""
        if df.empty:
            raise ValueError("Ramka danych jest pusta.")

        if ticker:
            df = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
        else:
            df = df.copy().reset_index(drop=True)  # Reset indeksów na początku

        # Zapisz oryginalne Close przed preprocessingiem
        original_close = df['Close'].copy()
        logger.info(f"Początkowa długość df: {len(df)}, original_close: {len(original_close)}")
        
        if historical_mode and trim_days > 0:
            df = df.iloc[:-trim_days].copy().reset_index(drop=True)
            original_close = original_close.iloc[:-trim_days].copy()
            logger.info(f"Po przycięciu (historical_mode): df: {len(df)}, original_close: {len(original_close)}")

        # Dodaj cechy
        df = self.feature_engineer.add_features(df).reset_index(drop=True)
        logger.info(f"Po add_features: df: {len(df)}")

        # Automatyczne czyszczenie inf/NaN/dużych wartości po dodaniu cech
        df = df.replace([np.inf, -np.inf], np.nan)
        # Bardzo duże wartości (np. >1e6) zamień na NaN tylko w numerycznych
        for col in df.select_dtypes(include=[np.number]).columns:
            df.loc[df[col].abs() > 1e6, col] = np.nan

        # Wypełnianie NaN: numeryczne -> 0, kategoryczne: Day_of_Week/Month -> '0', Sector -> 'Unknown'
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["category"]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in cat_cols:
            if col == 'Sector':
                if df[col].isnull().any():
                    if 'Unknown' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(['Unknown'])
                    df[col] = df[col].fillna('Unknown')
            elif col == 'Day_of_Week' or col == 'Month':
                if df[col].isnull().any():
                    if '0' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(['0'])
                    df[col] = df[col].fillna('0')
            else:
                # fallback: fillna na pierwszą kategorię
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].cat.categories[0])

        # Zachowaj oryginalne indeksy przed dropna
        original_indices = df.index
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).reset_index(drop=True)
        logger.info(f"Po dropna: df: {len(df)}, usunięto rekordy: {set(original_indices) - set(df.index)}")
        # Dopasuj original_close do przefiltrowanych indeksów
        original_close = original_close.loc[original_indices].reindex(df.index).fillna(0)
        
        df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])].reset_index(drop=True)
        logger.info(f"Po filtrze Close > 0 i High >= Low: df: {len(df)}")
        # Ponownie dopasuj original_close
        original_close = original_close.reindex(df.index).fillna(0)
        
        df = self.feature_engineer.remove_outliers(df, 'Close').reset_index(drop=True)
        logger.info(f"Po remove_outliers: df: {len(df)}")
        # Ostateczne dopasowanie original_close
        original_close = original_close.reindex(df.index).fillna(0)
        logger.info(f"Ostateczna długość df: {len(df)}, original_close: {len(original_close)}")

        # Ustaw kategorie i time_idx
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
        df['group_id'] = df['Ticker']
        
        df['Day_of_Week'] = pd.Categorical(df['Date'].dt.dayofweek.astype(str), categories=self.day_of_week_categories, ordered=False)
        df['Month'] = pd.Categorical(df['Date'].dt.month.astype(str), categories=self.month_categories, ordered=False)
        df['Sector'] = pd.Categorical(df['Sector'], categories=self.sectors, ordered=False)


        # --- AUTOMATYCZNE TRANSFORMACJE LOGARYTMICZNE DLA SKOŚNYCH CECH ---
        # Volume, ATR_14, Rolling_Volatility_30d: log1p
        log1p_features = ["Volume", "ATR_14", "Rolling_Volatility_30d"]
        for feature in log1p_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        # DMI_plus, DMI_minus: log1p(x - min + 1) (mogą być ujemne)
        for feature in ["DMI_plus", "DMI_minus"]:
            if feature in df.columns:
                min_val = df[feature].min()
                df[feature] = np.log1p(df[feature] - min_val + 1)

        # BB_upper, BB_lower, Kijun_sen, Senkou_Span_A: log1p(x - min + 1)
        for feature in ["BB_upper", "BB_lower", "Kijun_sen", "Senkou_Span_A"]:
            if feature in df.columns:
                min_val = df[feature].min()
                df[feature] = np.log1p(df[feature] - min_val + 1)

        # Momentum_20d: sign-preserving log1p
        if "Momentum_20d" in df.columns:
            df["Momentum_20d"] = np.sign(df["Momentum_20d"]) * np.log1p(np.abs(df["Momentum_20d"]))

        # BB_width: log1p
        if "BB_width" in df.columns:
            df["BB_width"] = np.log1p(df["BB_width"].clip(lower=0))

        # Pozostałe log_features (np. High, Low, Close): log1p jeśli nie są już przetransformowane
        for feature in self.log_features:
            if feature in df.columns and feature not in ["Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility", "OBV", "Volume", "BB_upper", "BB_lower", "Kijun_sen", "Senkou_Span_A"]:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        # Per-feature normalizery: MinMaxScaler dla wybranych, RobustScaler dla reszty


        from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer

        minmax_features = [
            'Future_Volume', 'Future_Volatility', 'BB_width', 'RSI', 'Stochastic_K', 'Stochastic_D'
        ]
        robust_features = [f for f in self.numeric_features if f not in minmax_features]

        normalizers = self.load_normalizers()
        new_normalizers = {}
        normalized_df = df.copy()  # Kopia do zapisu znormalizowanych danych
        quantile_features = [
            'ATR_14', 'Rolling_Volatility_30d', 'ADX', 'Future_Volatility', 'Future_Volume',
            'Senkou_Span_A', 'Close', 'BB_upper', 'BB_lower', 'Kijun_sen',
            'Relative_Returns', 'ROC_30d', 'High', 'DMI_minus',
            'MACD_Histogram', 'Low', 'Log_Returns', 'DMI_plus', 'Volume'
        ]
        # Indywidualne zakresy clippingu dla wybranych cech
        quantile_clip_map = {
            'Relative_Returns': (-2.5, 2.5),
            'ROC_30d': (-3, 3),
            'High': (-2.5, 2.5),
            'DMI_minus': (-3, 3),
            'Future_Volume': (-3, 3),
            'MACD_Histogram': (-3, 3),
            'Low': (-2.5, 2.5),
            'Log_Returns': (-3, 3),
            'DMI_plus': (-3, 3),
            'Volume': (-4, 4),
        }
        for feature in self.numeric_features:
            if feature in df.columns:
                if feature in quantile_features:
                    # Dla Relative_Returns: sign-preserving log1p przed QuantileTransformer
                    if feature == 'Relative_Returns':
                        df[feature] = np.sign(df[feature]) * np.log1p(np.abs(df[feature]))
                    normalizer = normalizers.get(feature, QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42, subsample=1_000_000))
                    if feature not in normalizers:
                        df[feature] = normalizer.fit_transform(df[[feature]])
                    else:
                        df[feature] = normalizer.transform(df[[feature]])
                    # Clipping po QuantileTransformer, indywidualny zakres dla wybranych cech
                    if feature in quantile_clip_map:
                        min_clip, max_clip = quantile_clip_map[feature]
                        df[feature] = df[feature].clip(min_clip, max_clip)
                    else:
                        df[feature] = df[feature].clip(-4, 4)
                    new_normalizers[feature] = normalizer
                    normalized_df[feature] = df[feature]
                elif feature == "Volume":
                    normalizer = normalizers.get(feature, QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42, subsample=1_000_000))
                    if feature not in normalizers:
                        df[feature] = normalizer.fit_transform(df[[feature]])
                    else:
                        df[feature] = normalizer.transform(df[[feature]])
                    df[feature] = df[feature].clip(-4, 4)
                    new_normalizers[feature] = normalizer
                    normalized_df[feature] = df[feature]
                elif feature in minmax_features:
                    normalizer = normalizers.get(feature, MinMaxScaler())
                    if feature not in normalizers:
                        df[feature] = normalizer.fit_transform(df[[feature]])
                    else:
                        df[feature] = normalizer.transform(df[[feature]])
                    new_normalizers[feature] = normalizer
                    normalized_df[feature] = df[feature]
                else:
                    normalizer = normalizers.get(feature, RobustScaler())
                    if feature not in normalizers:
                        df[feature] = normalizer.fit_transform(df[[feature]])
                    else:
                        df[feature] = normalizer.transform(df[[feature]])
                    new_normalizers[feature] = normalizer
                    normalized_df[feature] = df[feature]

        # Zapisz nowe normalizery
        if new_normalizers:
            self.save_normalizers(new_normalizers)

        # Zapisz znormalizowane dane do pliku normalized_data.csv dla debugowania
        try:
            normalized_df.to_csv(self.normalized_data_path, index=False)
            logger.info(f"Znormalizowane dane zapisano do: {self.normalized_data_path}")
        except Exception as e:
            logger.error(f"Błąd zapisu znormalizowanych danych: {e}")

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
            # Wymuszamy TorchNormalizer dla targetu
            "target_normalizer": TorchNormalizer(method='robust'),
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