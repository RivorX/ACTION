import pandas as pd
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import torch
import aiohttp
import os
import sys
import json

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.data_fetcher import DataFetcher
from scripts.utils.config_manager import ConfigManager
from scripts.utils.preprocessing_utils import PreprocessingUtils
from scipy.stats import zscore

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
    """Klasa do analizy danych giełdowych pod kątem nietypowych wartości."""

    def __init__(self, config: dict, years: int = 10):
        self.config = config
        self.years = years
        self.config_manager = ConfigManager()
        # Pobierz nazwę modelu z configu
        model_name = config['model_name']
        # Ustaw ścieżkę normalizerów na podstawie nazwy modelu
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

    def analyze_data(self, df: pd.DataFrame, ticker: str = None) -> dict:
        """Analizuje dane pod kątem nietypowych wartości."""
        analysis_results = {}
        if ticker:
            df = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
        
        # Preprocessing danych
        logger.info(f"Preprocesowanie danych dla {ticker if ticker else 'wszystkich tickerów'}...")
        df_processed, _ = self.preprocessing_utils.preprocess_dataframe(df, ticker)
        
        # Analiza dla każdej cechy numerycznej
        for feature in self.numeric_features:
            if feature in df_processed.columns:
                feature_data = df_processed[feature]
                analysis_results[feature] = {
                    'NaN_count': feature_data.isna().sum(),
                    'Inf_count': np.isinf(feature_data).sum(),
                    'Zero_count': (feature_data == 0).sum(),
                    'Negative_count': (feature_data < 0).sum(),
                    'Outliers': [],
                    'Min': float(feature_data.min()) if not feature_data.empty else None,
                    'Max': float(feature_data.max()) if not feature_data.empty else None,
                    'Mean': float(feature_data.mean()) if not feature_data.empty else None,
                    'Std': float(feature_data.std()) if not feature_data.empty else None
                }

                # Wykrywanie outlierów za pomocą z-score (>3 lub <-3)
                if not feature_data.empty and feature_data.std() > 0:
                    z_scores = zscore(feature_data)
                    outliers = df_processed[abs(z_scores) > 3][['Date', 'Ticker', feature]]
                    analysis_results[feature]['Outliers'] = [
                        {'Date': row['Date'], 'Ticker': row['Ticker'], 'Value': row[feature]}
                        for _, row in outliers.iterrows()
                    ]

                # Dodatkowe sprawdzenie dla ekstremalnych wartości
                if feature in ['Close', 'Open', 'High', 'Low']:
                    extreme_values = df_processed[
                        (feature_data > 10000) | (feature_data < 0.01)
                    ][['Date', 'Ticker', feature]]
                    analysis_results[feature]['Extreme_values'] = [
                        {'Date': row['Date'], 'Ticker': row['Ticker'], 'Value': row[feature]}
                        for _, row in extreme_values.iterrows()
                    ]

        # Sprawdzenie spójności normalizerów
        normalizers = self.preprocessing_utils.load_normalizers()
        for feature in self.numeric_features:
            if feature in df_processed.columns:
                # Wybierz odpowiedni normalizer: per ticker lub globalny
                normalizer_key = ticker if ticker and ticker in normalizers else 'global'
                if normalizer_key in normalizers and feature in normalizers[normalizer_key]:
                    normalizer = normalizers[normalizer_key][feature]
                    try:
                        transformed = normalizer.transform(df_processed[feature].values)
                        analysis_results[feature]['Normalizer_stats'] = {
                            'Mean_transformed': float(np.mean(transformed)),
                            'Std_transformed': float(np.std(transformed)),
                            'NaN_transformed': np.isnan(transformed).sum(),
                            'Inf_transformed': np.isinf(transformed).sum()
                        }
                        logger.info(f"Normalizer dla {feature} ({normalizer_key}) zastosowany poprawnie")
                    except Exception as e:
                        analysis_results[feature]['Normalizer_error'] = str(e)
                        logger.error(f"Błąd podczas stosowania normalizera dla {feature} ({normalizer_key}): {e}")
                else:
                    analysis_results[feature]['Normalizer_error'] = f"Brak normalizera dla {feature} w {normalizer_key}"
                    logger.warning(f"Brak normalizera dla {feature} w {normalizer_key}")

        return analysis_results

    async def run_analysis(self, tickers: list):
        """Uruchamia pełną analizę dla podanych tickerów."""
        logger.info("Rozpoczynanie analizy danych...")
        df = await self.fetch_data(tickers)
        if df.empty:
            logger.error("Nie udało się pobrać danych.")
            return

        # Analiza dla wszystkich tickerów razem
        overall_results = self.analyze_data(df)
        self.save_results(overall_results, 'overall_analysis.json')

        # Analiza dla każdego tickera osobno
        ticker_results = {}
        for ticker in tickers:
            ticker_results[ticker] = self.analyze_data(df, ticker)
            self.save_results(ticker_results[ticker], f'ticker_{ticker}_analysis.json')

        # Logowanie kluczowych wyników
        for feature, stats in overall_results.items():
            logger.info(f"\nAnaliza cechy {feature}:")
            logger.info(f"  NaN: {stats['NaN_count']}, Inf: {stats['Inf_count']}, Zeros: {stats['Zero_count']}")
            logger.info(f"  Negatives: {stats['Negative_count']}")
            logger.info(f"  Min: {stats['Min']:.2f}, Max: {stats['Max']:.2f}, Mean: {stats['Mean']:.2f}, Std: {stats['Std']:.2f}")
            if stats['Outliers']:
                logger.info(f"  Outliers: {len(stats['Outliers'])}")
                for outlier in stats['Outliers'][:5]:  # Pokazuj max 5 outlierów
                    logger.info(f"    {outlier['Date']} | {outlier['Ticker']} | {outlier['Value']:.2f}")
            if 'Extreme_values' in stats and stats['Extreme_values']:
                logger.info(f"  Extreme values: {len(stats['Extreme_values'])}")
                for extreme in stats['Extreme_values'][:5]:
                    logger.info(f"    {extreme['Date']} | {extreme['Ticker']} | {extreme['Value']:.2f}")

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
    
    # Lista tickerów z logów
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