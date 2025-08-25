import pandas as pd
import numpy as np
import yfinance as yf
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

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Oblicza ADX ręcznie (poprawione: zadeklarowany indeks dla Series)."""
        # upewnij się, że używamy typów numerycznych i zachowujemy indeks
        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)

        tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
        tr = pd.Series(tr, index=high.index)
        atr = tr.rolling(window=period, min_periods=period).mean()
        # unikaj dzielenia przez zero
        atr = atr.replace(0, np.nan)

        up = (high - high.shift()).fillna(0)
        down = (low.shift() - low).fillna(0)

        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)

        pos_dm = pd.Series(pos_dm, index=high.index).astype(float)
        neg_dm = pd.Series(neg_dm, index=high.index).astype(float)

        pos_di = 100 * pos_dm.rolling(window=period, min_periods=period).mean() / atr
        neg_di = 100 * neg_dm.rolling(window=period, min_periods=period).mean() / atr

        # zabezpiecz przed dzieleniem przez zero w mianowniku pos_di+neg_di
        denom = (pos_di + neg_di).replace(0, np.nan)
        dx = 100 * (pos_di - neg_di).abs() / denom
        adx = dx.rolling(window=period, min_periods=period).mean()

        return adx

    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Oblicza CCI ręcznie."""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - ma) / (0.015 * md)
        return cci

    @staticmethod
    def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 25) -> tuple:
        """Oblicza Aroon Up i Down ręcznie."""
        aroon_up = 100 * (period - high.rolling(period + 1).apply(np.argmax)) / period
        aroon_down = 100 * (period - low.rolling(period + 1).apply(np.argmin)) / period
        return aroon_up, aroon_down

    @staticmethod
    def calculate_parabolic_sar(high: pd.Series, low: pd.Series, af_step: float = 0.015, af_max: float = 0.20) -> pd.Series:
        """Oblicza Parabolic SAR ręcznie (uproszczona implementacja)."""
        sar = pd.Series(np.nan, index=high.index)
        ep = pd.Series(np.nan, index=high.index)
        af = pd.Series(np.nan, index=high.index)
        
        # Inicjalizacja (zakładamy trend up na start)
        trend = 1  # 1 dla up, -1 dla down
        sar.iloc[0] = low.iloc[0]
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = af_step
        
        for i in range(1, len(high)):
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
            
            if trend == 1:  # Up trend
                if low.iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = af_step
                else:
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af_max, af.iloc[i-1] + af_step)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Down trend
                if high.iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = af_step
                else:
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af_max, af.iloc[i-1] + af_step)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return sar

    @staticmethod
    def calculate_dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        """Oblicza DMI+ i DMI- ręcznie (poprawione: zachowany indeks Series)."""
        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)

        up = (high - high.shift()).fillna(0)
        down = (low.shift() - low).fillna(0)

        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)

        tr = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
        tr = pd.Series(tr, index=high.index)
        atr = tr.rolling(window=period, min_periods=period).mean().replace(0, np.nan)

        pos_dm = pd.Series(pos_dm, index=high.index).astype(float)
        neg_dm = pd.Series(neg_dm, index=high.index).astype(float)

        pos_di = 100 * pos_dm.rolling(window=period, min_periods=period).mean() / atr
        neg_di = 100 * neg_dm.rolling(window=period, min_periods=period).mean() / atr

        return pos_di, neg_di

    def add_features(self, df: pd.DataFrame, sectors_list=None) -> pd.DataFrame:
        """Dodaje nowe cechy do ramki danych z grupowaniem po Ticker."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)

        # Pobierz statyczne cechy dla każdego unikalnego tickera
        unique_tickers = df['Ticker'].unique()
        market_cap_dict = {}
        dividend_yield_dict = {}
        for ticker in unique_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', np.nan)
                if np.isnan(market_cap):
                    market_cap_category = 'Unknown'
                elif market_cap < 2e9:
                    market_cap_category = 'Small'
                elif market_cap < 10e9:
                    market_cap_category = 'Mid'
                else:
                    market_cap_category = 'Large'
                market_cap_dict[ticker] = market_cap_category
                
                dividend_yield = info.get('dividendYield', np.nan)
                if np.isnan(dividend_yield):
                    dividend_category = 'None'
                elif dividend_yield < 0.02:
                    dividend_category = 'Low'
                elif dividend_yield < 0.05:
                    dividend_category = 'Medium'
                else:
                    dividend_category = 'High'
                dividend_yield_dict[ticker] = dividend_category
            except Exception as e:
                logger.warning(f"Błąd pobierania info dla {ticker}: {e}")
                market_cap_dict[ticker] = 'Unknown'
                dividend_yield_dict[ticker] = 'None'

        # Dodaj statyczne cechy do df
        df['Market_Cap_Category'] = df['Ticker'].map(market_cap_dict)
        df['Dividend_Yield_Category'] = df['Ticker'].map(dividend_yield_dict)

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

            # Dodatkowe cechy istniejące
            group['Momentum_20d'] = group['Close'] - group['Close'].shift(20)
            group['Close_to_MA_ratio'] = group['Close'] / group['MA50']
            group['Relative_Returns'] = group['Close'].pct_change()

            group['Month'] = group['Date'].dt.month.astype(str)
            group['Day_of_Week'] = group['Date'].dt.dayofweek.astype(str)

            # Nowe cechy techniczne
            group['ADX'] = self.calculate_adx(group['High'], group['Low'], group['Close'])
            group['CCI'] = self.calculate_cci(group['High'], group['Low'], group['Close'])
            group['Aroon_Up'], group['Aroon_Down'] = self.calculate_aroon(group['High'], group['Low'])
            group['Parabolic_SAR'] = self.calculate_parabolic_sar(group['High'], group['Low'])
            group['DMI_plus'], group['DMI_minus'] = self.calculate_dmi(group['High'], group['Low'], group['Close'])
            group['Up_Days_30d'] = (group['Close'] > group['Close'].shift(1)).rolling(window=30).sum().astype(float)

            # Wypełnianie brakujących wartości dla Relative_Returns
            nan_count = group['Relative_Returns'].isna().sum()
            if nan_count > 0:
                group['Relative_Returns'] = group['Relative_Returns'].fillna(0)
            
            # Wypełnianie brakujących wartości dla innych cech (w tym nowych)
            features_to_fill = [
                'MA10', 'MA50', 'BB_upper', 'Close_to_BB_upper',
                'RSI', 'MACD', 'ROC', 'VWAP',
                'Momentum_20d', 'Close_to_MA_ratio',
                'ADX', 'CCI', 'Aroon_Up', 'Aroon_Down', 'Parabolic_SAR',
                'DMI_plus', 'DMI_minus', 'Up_Days_30d'
            ]
            for feature in features_to_fill:
                if feature in group.columns:
                    group[feature] = group[feature].ffill().bfill()
            
            return group

        df = df.groupby('Ticker').apply(apply_features).reset_index(drop=True)
        
        if sectors_list:
            df['Sector'] = pd.Categorical(df['Sector'], categories=sectors_list, ordered=False)
        
        # Ustaw kategorie dla nowych cech statycznych
        df['Market_Cap_Category'] = pd.Categorical(df['Market_Cap_Category'], categories=['Small', 'Mid', 'Large', 'Unknown'])
        df['Dividend_Yield_Category'] = pd.Categorical(df['Dividend_Yield_Category'], categories=['None', 'Low', 'Medium', 'High'])
        
        return df