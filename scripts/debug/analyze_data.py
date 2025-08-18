import pandas as pd
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import aiohttp
import torch
from pytorch_forecasting.data.encoders import TorchNormalizer

# Dodaj katalog główny do ścieżek systemowych
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.preprocessor import DataPreprocessor, FeatureEngineer

# Konfiguracja logowania zgodna z innymi modułami
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Klasa do globalnej analizy danych giełdowych pod kątem nietypowych wartości i problemów z normalizacją."""

    def __init__(self, config: dict, years: int = 10):
        """Inicjalizuje DataAnalyzer z konfiguracją i liczbą lat danych."""
        self.config = config
        self.years = years
        self.config_manager = ConfigManager()
        self.model_name = config['model_name']
        normalizers_path = Path(self.config['paths']['normalizers_dir']) / f"{self.model_name}_normalizers.pkl"
        self.config['data']['normalizers_path'] = str(normalizers_path)
        logger.info(f"Ścieżka normalizerów ustawiona na: {normalizers_path}")
        self.data_fetcher = DataFetcher(self.config_manager, years=years)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer()
        self.output_dir = Path(self.config['paths']['logs_dir']) / 'debug'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Lista cech numerycznych zgodna z preprocessor.py
        self.numeric_features = [
            "Close", "Volume", "MA10", "MA50", "RSI", "MACD", "ROC", "VWAP",
            "Momentum_20d", "Close_to_MA_ratio", "Close_to_BB_upper", "Relative_Returns"
        ]
        self.expected_ranges = {
            "RSI": (0, 100),
            "MACD": (float('-inf'), float('inf')),
            "ROC": (-100, 100),
            "VWAP": (0, float('inf')),
            "Momentum_20d": (float('-inf'), float('inf')),
            "Close_to_MA_ratio": (0, float('inf')),
            "Close_to_BB_upper": (0, float('inf')),
            "Relative_Returns": (float('-inf'), float('inf')),
            "Close": (0, float('inf')),
            "Volume": (0, float('inf')),
            "MA10": (0, float('inf')),
            "MA50": (0, float('inf'))
        }

    async def fetch_data(self, tickers: list) -> pd.DataFrame:
        """Pobiera dane dla podanych tickerów."""
        end_date = datetime.now(tz=None)  # Zgodność z data_fetcher.py (timezone-naive)
        start_date = end_date - timedelta(days=self.years * 365)
        all_data = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.data_fetcher.fetch_stock_data(ticker, start_date, end_date, session) for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ticker, result in zip(tickers, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_data.append(result)
                else:
                    logger.warning(f"Brak danych lub błąd dla tickera {ticker}")
        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        if not df.empty:
            df['Sector'] = pd.Categorical(df['Sector'], categories=self.config['model']['sectors'], ordered=False)
            logger.info(f"Pobrano dane dla {len(all_data)} tickerów, liczba wierszy: {len(df)}")
        return df

    def plot_feature_distribution(self, data: pd.Series, feature: str, normalized: bool = False):
        """Tworzy histogram rozkładu cechy i zapisuje go jako PNG."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=50, kde=True, color='blue' if not normalized else 'green')
        plt.title(f"Rozkład cechy {feature} {'(znormalizowana)' if normalized else ''}")
        plt.xlabel(feature)
        plt.ylabel("Liczba")
        output_path = self.output_dir / f"{feature}_{'normalized' if normalized else 'raw'}_all.png"
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Zapisano histogram dla {feature} ({'znormalizowana' if normalized else 'surowa'}) do: {output_path}")

    def analyze_data(self, df: pd.DataFrame):
        """Analizuje rozkład cech: zapisuje histogramy, statystyki do CSV i heatmapę korelacji."""
        logger.info("Preprocesowanie danych globalnych...")
        
        # Dodanie cech za pomocą FeatureEngineer
        df_processed = self.feature_engineer.add_features(df, sectors_list=self.config['model']['sectors'])

        # Wczytanie normalizerów
        normalizers = self.config_manager.load_normalizers(self.model_name)
        df_normalized = df_processed.copy()

        # Normalizacja danych
        for feature in self.numeric_features:
            if feature in df_normalized.columns and feature in normalizers:
                try:
                    df_normalized[feature] = normalizers[feature].transform(df_normalized[feature].values)
                    if df_normalized[feature].isna().any() or np.isinf(df_normalized[feature]).any():
                        logger.warning(f"Normalizacja cechy {feature} spowodowała NaN lub inf")
                except Exception as e:
                    logger.error(f"Błąd podczas normalizacji cechy {feature}: {e}")

        stats = []
        for feature in self.numeric_features:
            if feature in df_processed.columns:
                feature_data = df_processed[feature].dropna()
                feature_data_normalized = df_normalized[feature].dropna() if feature in df_normalized.columns else pd.Series()

                # Histogramy dla danych surowych i znormalizowanych
                if not feature_data.empty:
                    self.plot_feature_distribution(feature_data, feature, normalized=False)
                if not feature_data_normalized.empty:
                    self.plot_feature_distribution(feature_data_normalized, feature, normalized=True)

                # Statystyki
                stat = {
                    'Feature': feature,
                    'NaN_count': df_processed[feature].isna().sum(),
                    'Inf_count': np.isinf(df_processed[feature]).sum() if np.isinf(df_processed[feature]).any() else 0,
                    'Zero_count': (df_processed[feature] == 0).sum(),
                    'Negative_count': (df_processed[feature] < 0).sum(),
                    'Min': float(feature_data.min()) if not feature_data.empty else None,
                    'Max': float(feature_data.max()) if not feature_data.empty else None,
                    'Mean': float(feature_data.mean()) if not feature_data.empty else None,
                    'Std': float(feature_data.std()) if not feature_data.empty else None,
                    'Normalized_Min': float(feature_data_normalized.min()) if not feature_data_normalized.empty else None,
                    'Normalized_Max': float(feature_data_normalized.max()) if not feature_data_normalized.empty else None,
                    'Normalized_Mean': float(feature_data_normalized.mean()) if not feature_data_normalized.empty else None,
                    'Normalized_Std': float(feature_data_normalized.std()) if not feature_data_normalized.empty else None
                }

                # Sprawdzenie oczekiwanych zakresów
                if feature in self.expected_ranges:
                    min_val, max_val = self.expected_ranges[feature]
                    out_of_range = ((feature_data < min_val) | (feature_data > max_val)).sum() if not feature_data.empty else 0
                    stat['Out_of_range_count'] = out_of_range
                    if out_of_range > 0:
                        logger.warning(f"Cecha {feature} ma {out_of_range} wartości poza zakresem ({min_val}, {max_val})")

                stats.append(stat)

        # Zapis statystyk do CSV
        stats_df = pd.DataFrame(stats)
        stats_path = self.output_dir / 'feature_stats.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Zapisano statystyki cech do: {stats_path}")

        # Macierz korelacji
        existing_numeric_features = [f for f in self.numeric_features if f in df_processed.columns]
        numeric_df = df_processed[existing_numeric_features].dropna()
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            corr_output_path = self.output_dir / "correlation_matrix_all.csv"
            correlation_matrix.to_csv(corr_output_path)
            logger.info(f"Zapisano macierz korelacji do: {corr_output_path}")
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title("Macierz korelacji dla wszystkich tickerów")
            corr_plot_path = self.output_dir / "correlation_heatmap_all.png"
            plt.savefig(corr_plot_path)
            plt.close()
            logger.info(f"Zapisano heatmapę korelacji do: {corr_plot_path}")

    async def run_analysis(self, tickers: list):
        """Uruchamia globalną analizę dla podanych tickerów."""
        logger.info(f"Rozpoczynanie analizy danych dla {len(tickers)} tickerów...")
        df = await self.fetch_data(tickers)
        if df.empty:
            logger.error("Nie udało się pobrać danych.")
            return
        self.analyze_data(df)

async def main():
    config_manager = ConfigManager()
    config = config_manager.config
    analyzer = DataAnalyzer(config, years=10)
    
    tickers = [
        'CDR.WA', 'PLW.WA', 'TEN.WA', 'BLO.WA', 'CIG.WA', 'PKO.WA', 'PEO.WA', 'ING.WA', 
        'MBK.WA', 'ALR.WA', 'PKN.WA', 'TPE.WA', 'ENA.WA', 'PGE.WA', 'KGH.WA', 'LPP.WA', 
        'JSW.WA', 'DNP.WA', 'CPS.WA', 'PZU.WA', 'KRK.WA', 'ACP.WA', 'BMW.DE', 'SIE.DE', 
        'SAN.PA', 'TTE.PA', 'BP.L', 'HSBA.L', 'VOW3.DE', 'RNO.PA', 'NG.L', 'DB1.DE', 
        'AIR.PA', 'BAS.DE', 'SAP.DE', 'BNP.PA', 'AZN.L', 'NOVN.SW', 'ROG.SW', 'NESN.SW', 
        'DTE.DE', 'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NKE', 'JPM', 'XOM', 'PFE', 
        'NVDA', 'META', 'V', 'MA', 'DIS', 'NFLX', 'INTC', 'AMD', 'CSCO', 'KO', 'PG', 
        'WFC', '7203.T', '9984.T', '005930.KS', '2330.TW', 'BABA', 'JD', 'INFY.NS', 
        'RELIANCE.NS', 'TM', 'SONY', 'TCTZF', 'HDB', 'BRK-B', 'WMT', '005380.KS', 
        '1211.HK', '8035.T', '3690.HK', '0700.HK', 'NTTYY', 'TCS.NS'
    ]
    
    await analyzer.run_analysis(tickers)

if __name__ == "__main__":
    asyncio.run(main())