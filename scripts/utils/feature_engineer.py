import pandas as pd
import numpy as np
import logging

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