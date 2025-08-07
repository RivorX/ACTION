import pandas as pd
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import aiohttp
import json

# Dodaj katalog główny do ścieżek systemowych
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.data_fetcher import DataFetcher
from scripts.utils.config_manager import ConfigManager
from scripts.utils.preprocessing_utils import PreprocessingUtils

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
        self.config = config
        self.years = years
        self.config_manager = ConfigManager()
        model_name = config['model_name']
        normalizers_path = Path(f"models/normalizers/{model_name}_normalizers.pkl")
        self.config['data']['normalizers_path'] = str(normalizers_path)
        logger.info(f"Ścieżka normalizerów ustawiona na: {normalizers_path}")
        self.data_fetcher = DataFetcher(self.config_manager, years=years)
        self.preprocessing_utils = PreprocessingUtils(self.config)
        self.output_dir = Path('logs/debug')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.numeric_features = [
            "High", "Low", "Close", "Volume", "RSI",
            "MACD_Histogram", "Stochastic_D",
            "ADX", "Kijun_sen", "Senkou_Span_A", "Momentum_20d",
            "BB_upper", "BB_lower", "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility",
            "ROC_30d", "DMI_plus", "DMI_minus", "Up_Days_30d", "Rolling_Volatility_30d", "ATR_14"
        ]
        self.expected_ranges = {
            "RSI": (0, 100),
            # "Stochastic_K": (0, 100),
            "Stochastic_D": (0, 100),
            "ADX": (0, 100),
            # "BB_width": (0, float('inf')),
            "Future_Volatility": (0, float('inf')),
            "ROC_30d": (-1, 1),
            "DMI_plus": (0, 100),
            "DMI_minus": (0, 100),
            "Up_Days_30d": (0, 30),
            "Rolling_Volatility_30d": (0, 1),
            "ATR_14": (0, float('inf'))
        }

    async def fetch_data(self, tickers: list) -> pd.DataFrame:
        """Pobiera dane dla podanych tickerów."""
        end_date = datetime.now()
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
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

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
        logger.info(f"Zapisano histogram dla {feature} do: {output_path}")

    def analyze_data(self, df: pd.DataFrame):
        """Analizuje rozkład cech: zapisuje histogramy, statystyki do CSV i heatmapę korelacji."""
        logger.info("Preprocesowanie danych globalnych...")
        df_processed, _ = self.preprocessing_utils.preprocess_dataframe(df, None)

        stats = []
        for feature in self.numeric_features:
            if feature in df_processed.columns:
                feature_data = df_processed[feature].dropna()
                # Histogram surowy
                self.plot_feature_distribution(feature_data, feature, normalized=False)
                # Statystyki
                stats.append({
                    'Feature': feature,
                    'NaN_count': df_processed[feature].isna().sum(),
                    'Inf_count': np.isinf(df_processed[feature]).sum(),
                    'Zero_count': (df_processed[feature] == 0).sum(),
                    'Negative_count': (df_processed[feature] < 0).sum(),
                    'Min': float(feature_data.min()) if not feature_data.empty else None,
                    'Max': float(feature_data.max()) if not feature_data.empty else None,
                    'Mean': float(feature_data.mean()) if not feature_data.empty else None,
                    'Std': float(feature_data.std()) if not feature_data.empty else None
                })

        # Zapis statystyk do CSV
        stats_df = pd.DataFrame(stats)
        stats_path = self.output_dir / 'feature_stats.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Zapisano statystyki cech do: {stats_path}")

        # Heatmapa korelacji
        numeric_df = df_processed[self.numeric_features].dropna()
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            corr_output_path = self.output_dir / "correlation_matrix_all.csv"
            correlation_matrix.to_csv(corr_output_path)
            logger.info(f"Zapisano macierz korelacji do: {corr_output_path}")
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Macierz korelacji dla wszystkich tickerów")
            corr_plot_path = self.output_dir / "correlation_heatmap_all.png"
            plt.savefig(corr_plot_path)
            plt.close()
            logger.info(f"Zapisano heatmapę korelacji do: {corr_plot_path}")

    async def run_analysis(self, tickers: list):
        """Uruchamia globalną analizę dla podanych tickerów."""
        logger.info("Rozpoczynanie analizy danych...")
        df = await self.fetch_data(tickers)
        if df.empty:
            logger.error("Nie udało się pobrać danych.")
            return

        # Analiza globalna
        self.analyze_data(df)

    def save_results(self, results: dict, filename: str):
        """Zapisuje wyniki analizy do pliku JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Wyniki zapisane do: {output_path}")

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