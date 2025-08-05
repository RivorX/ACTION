import pandas as pd
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler, StandardScaler
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
            "Open", "High", "Low", "Close", "Volume", "MA50", "RSI",
            "MACD_Signal", "MACD_Histogram", "Stochastic_K", "Stochastic_D", "OBV",
            "ADX", "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Momentum_20d",
            "BB_width", "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility"
        ]
        self.expected_ranges = {
            "RSI": (0, 100),
            "Stochastic_K": (0, 100),
            "Stochastic_D": (0, 100),
            "ADX": (0, 100),
            "BB_width": (0, float('inf')),
            "Future_Volatility": (0, float('inf'))
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

    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Analizuje dane globalnie pod kątem nietypowych wartości i normalizacji."""
        analysis_results = {}
        
        # Preprocessing danych
        logger.info("Preprocesowanie danych globalnych...")
        df_processed, _ = self.preprocessing_utils.preprocess_dataframe(df, None)
        
        # Analiza dla każdej cechy numerycznej
        for feature in self.numeric_features:
            if feature in df_processed.columns:
                feature_data = df_processed[feature].dropna()
                analysis_results[feature] = {
                    'NaN_count': df_processed[feature].isna().sum(),
                    'Inf_count': np.isinf(df_processed[feature]).sum(),
                    'Zero_count': (df_processed[feature] == 0).sum(),
                    'Negative_count': (df_processed[feature] < 0).sum(),
                    'Outliers': [],
                    'Min': float(feature_data.min()) if not feature_data.empty else None,
                    'Max': float(feature_data.max()) if not feature_data.empty else None,
                    'Mean': float(feature_data.mean()) if not feature_data.empty else None,
                    'Std': float(feature_data.std()) if not feature_data.empty else None,
                    'Range_violation': []
                }

                # Wykrywanie outlierów za pomocą z-score (>3 lub <-3)
                if not feature_data.empty and feature_data.std() > 0:
                    z_scores = zscore(feature_data)
                    outliers = df_processed[abs(z_scores) > 3][['Date', 'Ticker', feature]]
                    analysis_results[feature]['Outliers'] = [
                        {'Date': row['Date'], 'Ticker': row['Ticker'], 'Value': float(row[feature])}
                        for _, row in outliers.iterrows()
                    ]

                # Sprawdzenie zakresu dla wybranych cech
                if feature in self.expected_ranges:
                    min_val, max_val = self.expected_ranges[feature]
                    range_violations = df_processed[
                        (df_processed[feature] < min_val) | (df_processed[feature] > max_val)
                    ][['Date', 'Ticker', feature]]
                    analysis_results[feature]['Range_violation'] = [
                        {'Date': row['Date'], 'Ticker': row['Ticker'], 'Value': float(row[feature])}
                        for _, row in range_violations.iterrows()
                    ]

                # Analiza normalizacji
                normalizers = self.preprocessing_utils.load_normalizers()
                normalizer_key = 'global'
                if normalizer_key in normalizers and feature in normalizers[normalizer_key]:
                    normalizer = normalizers[normalizer_key][feature]
                    try:
                        transformed = normalizer.transform(df_processed[feature].values)
                        analysis_results[feature]['Normalizer_stats'] = {
                            'Mean_transformed': float(np.mean(transformed[~np.isnan(transformed)])),
                            'Std_transformed': float(np.std(transformed[~np.isnan(transformed)])),
                            'NaN_transformed': np.isnan(transformed).sum(),
                            'Inf_transformed': np.isinf(transformed).sum(),
                            'Method': normalizer.method if hasattr(normalizer, 'method') else 'unknown'
                        }
                        # Tworzenie histogramu dla znormalizowanej cechy
                        self.plot_feature_distribution(pd.Series(transformed), feature, normalized=True)
                    except Exception as e:
                        analysis_results[feature]['Normalizer_error'] = str(e)
                        logger.error(f"Błąd podczas stosowania normalizera dla {feature} (global): {e}")
                else:
                    analysis_results[

feature]['Normalizer_error'] = f"Brak normalizera dla {feature} w global"
                    logger.warning(f"Brak normalizera dla {feature} w global")

                # Tworzenie histogramu dla surowej cechy
                self.plot_feature_distribution(feature_data, feature, normalized=False)

                # Sprawdzenie ekstremalnych wartości dla cen
                if feature in ['Close', 'Open', 'High', 'Low']:
                    extreme_values = df_processed[
                        (df_processed[feature] > 10000) | (df_processed[feature] < 0.01)
                    ][['Date', 'Ticker', feature]]
                    analysis_results[feature]['Extreme_values'] = [
                        {'Date': row['Date'], 'Ticker': row['Ticker'], 'Value': float(row[feature])}
                        for _, row in extreme_values.iterrows()
                    ]

        # Sprawdzenie korelacji między cechami
        numeric_df = df_processed[self.numeric_features].dropna()
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            analysis_results['Correlation_matrix'] = correlation_matrix.to_dict()
            # Zapis korelacji do pliku
            corr_output_path = self.output_dir / "correlation_matrix_all.csv"
            correlation_matrix.to_csv(corr_output_path)
            logger.info(f"Zapisano macierz korelacji do: {corr_output_path}")

        return analysis_results

    async def run_analysis(self, tickers: list):
        """Uruchamia globalną analizę dla podanych tickerów."""
        logger.info("Rozpoczynanie analizy danych...")
        df = await self.fetch_data(tickers)
        if df.empty:
            logger.error("Nie udało się pobrać danych.")
            return

        # Analiza globalna
        overall_results = self.analyze_data(df)
        self.save_results(overall_results, 'overall_analysis.json')
        
        # Zapis statystyk do CSV
        overall_stats = []
        for feature, stats in overall_results.items():
            if feature != 'Correlation_matrix':
                overall_stats.append({
                    'Feature': feature,
                    'NaN_count': stats['NaN_count'],
                    'Inf_count': stats['Inf_count'],
                    'Zero_count': stats['Zero_count'],
                    'Negative_count': stats['Negative_count'],
                    'Outliers_count': len(stats['Outliers']),
                    'Range_violation_count': len(stats.get('Range_violation', [])),
                    'Min': stats['Min'],
                    'Max': stats['Max'],
                    'Mean': stats['Mean'],
                    'Std': stats['Std'],
                    'Normalizer_mean': stats.get('Normalizer_stats', {}).get('Mean_transformed', None),
                    'Normalizer_std': stats.get('Normalizer_stats', {}).get('Std_transformed', None),
                    'Normalizer_method': stats.get('Normalizer_stats', {}).get('Method', None)
                })
        pd.DataFrame(overall_stats).to_csv(self.output_dir / 'overall_stats.csv', index=False)
        logger.info(f"Zapisano statystyki do: {self.output_dir / 'overall_stats.csv'}")

        # Logowanie kluczowych wyników
        for feature, stats in overall_results.items():
            if feature != 'Correlation_matrix':
                logger.info(f"\nAnaliza cechy {feature}:")
                logger.info(f"  NaN: {stats['NaN_count']}, Inf: {stats['Inf_count']}, Zeros: {stats['Zero_count']}")
                logger.info(f"  Negatives: {stats['Negative_count']}")
                logger.info(f"  Min: {stats['Min']:.2f}, Max: {stats['Max']:.2f}, Mean: {stats['Mean']:.2f}, Std: {stats['Std']:.2f}")
                if stats['Outliers']:
                    logger.info(f"  Outliers: {len(stats['Outliers'])}")
                    for outlier in stats['Outliers'][:5]:
                        logger.info(f"    {outlier['Date']} | {outlier['Ticker']} | {outlier['Value']:.2f}")
                if 'Range_violation' in stats and stats['Range_violation']:
                    logger.info(f"  Range violations: {len(stats['Range_violation'])}")
                    for violation in stats['Range_violation'][:5]:
                        logger.info(f"    {violation['Date']} | {violation['Ticker']} | {violation['Value']:.2f}")
                if 'Normalizer_stats' in stats:
                    logger.info(f"  Normalizer: Mean={stats['Normalizer_stats']['Mean_transformed']:.2f}, "
                                f"Std={stats['Normalizer_stats']['Std_transformed']:.2f}, "
                                f"Method={stats['Normalizer_stats']['Method']}")

        # Wykres korelacji dla kluczowych cech
        df_processed, _ = self.preprocessing_utils.preprocess_dataframe(df, None)
        key_features = ['RSI', 'Stochastic_K', 'Stochastic_D', 'BB_upper', 'BB_lower', 'OBV', 'Relative_Returns']
        numeric_df = df_processed[key_features].dropna()
        if not numeric_df.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Macierz korelacji dla wszystkich tickerów")
            corr_plot_path = self.output_dir / "correlation_heatmap_all.png"
            plt.savefig(corr_plot_path)
            plt.close()
            logger.info(f"Zapisano heatmap korelacji do: {corr_plot_path}")

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