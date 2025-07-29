import streamlit as st
import pandas as pd
import torch
import logging
import os
import asyncio
import aiohttp
import nest_asyncio
from datetime import datetime, timedelta
import sys
import tempfile
import glob
import time

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Configure logging to suppress Streamlit LocalSourcesWatcher warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.prediction_engine import load_data_and_model, preprocess_data, generate_predictions
from app.config_loader import load_config, load_tickers_and_names, load_benchmark_tickers
from app.plot_utils import create_stock_plot
from app.benchmark_utils import create_benchmark_plot, save_benchmark_to_csv, load_benchmark_history

def clean_temp_dir(temp_dir):
    """Cleans all CSV files in the specified temporary directory."""
    try:
        csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        for file in csv_files:
            for _ in range(3):  # Try up to 3 times to handle file locking
                try:
                    os.remove(file)
                    logger.info(f"Removed temporary file: {file}")
                    break
                except PermissionError as e:
                    logger.warning(f"Failed to remove temporary file {file}: {e}. Retrying...")
                    time.sleep(0.1)  # Short delay to allow file release
                except Exception as e:
                    logger.error(f"Unexpected error while removing {file}: {e}")
                    break
    except Exception as e:
        logger.error(f"Error cleaning temporary directory {temp_dir}: {e}")

class StockPredictor:
    """Handles business logic for stock price prediction."""
    def __init__(self, config, years):
        self.config = config
        self.years = years
        self.fetcher = DataFetcher(ConfigManager(), years)
        self.temp_dir = os.path.join(project_root, 'data', 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        clean_temp_dir(self.temp_dir)

    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetches stock data without caching."""
        return self.fetcher.fetch_stock_data_sync(ticker, start_date, end_date)

    def predict(self, ticker, start_date, end_date):
        """Generates predictions for a given ticker."""
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=self.temp_dir)
            temp_raw_data_path = temp_file.name
            temp_file.close()
            new_data = self.fetch_stock_data(ticker, start_date, end_date)
            if new_data.empty:
                raise ValueError(f"No data available for {ticker}")

            new_data.to_csv(temp_raw_data_path, index=False)
            logger.info(f"Data for {ticker} saved to {temp_raw_data_path}")

            _, dataset, normalizers, model = load_data_and_model(self.config, ticker, temp_raw_data_path)
            ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                median, lower_bound, upper_bound = generate_predictions(self.config, dataset, model, ticker_data)
            
            return ticker_data, original_close, median, lower_bound, upper_bound
        except Exception as e:
            logger.error(f"Error generating predictions for {ticker}: {e}")
            raise
        finally:
            if temp_file is not None and os.path.exists(temp_raw_data_path):
                for _ in range(3):
                    try:
                        os.remove(temp_raw_data_path)
                        logger.info(f"Temporary file {temp_raw_data_path} removed.")
                        break
                    except PermissionError as e:
                        logger.warning(f"Failed to remove temporary file {temp_raw_data_path}: {e}. Retrying...")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Unexpected error while removing {temp_raw_data_path}: {e}")
                        break

    def predict_historical(self, ticker, start_date, end_date, trim_date):
        """Compares predictions with historical data."""
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=self.temp_dir)
            temp_raw_data_path = temp_file.name
            temp_file.close()
            full_data = self.fetch_stock_data(ticker, start_date, end_date)
            if full_data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            full_data = full_data[full_data['Ticker'] == ticker].copy()
            full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
            new_data = full_data[full_data['Date'] <= trim_date].copy()
            if new_data.empty:
                raise ValueError(f"No data before {trim_date} for {ticker}")
            
            new_data.to_csv(temp_raw_data_path, index=False)
            logger.info(f"Data for {ticker} saved to {temp_raw_data_path}")
            
            _, dataset, normalizers, model = load_data_and_model(self.config, ticker, temp_raw_data_path, historical_mode=True)
            ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers, historical_mode=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                median, lower_bound, upper_bound = generate_predictions(self.config, dataset, model, ticker_data)
            
            full_data.set_index('Date', inplace=True)
            historical_close = full_data['Close']
            return ticker_data, original_close, median, lower_bound, upper_bound, historical_close
        except Exception as e:
            logger.error(f"Error comparing predictions with history for {ticker}: {e}")
            raise
        finally:
            if temp_file is not None and os.path.exists(temp_raw_data_path):
                for _ in range(3):
                    try:
                        os.remove(temp_raw_data_path)
                        logger.info(f"Temporary file {temp_raw_data_path} removed.")
                        break
                    except PermissionError as e:
                        logger.warning(f"Failed to remove temporary file {temp_raw_data_path}: {e}. Retrying...")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Unexpected error while removing {temp_raw_data_path}: {e}")
                        break

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    config_manager = ConfigManager()  # Singleton
    config = config_manager.config
    years = config['prediction']['years']

    predictor = StockPredictor(config, years)
    benchmark_tickers = load_benchmark_tickers(config)

    page = st.sidebar.selectbox("Wybierz stronę", ["Predykcje przyszłości", "Porównanie predykcji z historią", "Benchmark"])

    if page == "Predykcje przyszłości":
        ticker_options = load_tickers_and_names(config)
        default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")

        ticker_option = st.selectbox(
            "Wybierz spółkę z listy lub wpisz własną:",
            options=["Wpisz ręcznie"] + list(ticker_options.values()),
            index=0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
        )

        ticker_input = default_ticker
        if ticker_option == "Wpisz ręcznie":
            ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker)
        else:
            ticker_input = [k for k, v in ticker_options.items() if v == ticker_option][0]

        if st.button("Generuj predykcje"):
            with st.spinner('Trwa generowanie predykcji...'):
                try:
                    start_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=years * 365)
                    ticker_data, original_close, median, lower_bound, upper_bound = predictor.predict(
                        ticker_input, start_date, datetime.now()
                    )
                    create_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input)
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas generowania predykcji dla {ticker_input}: {str(e)}")

    elif page == "Porównanie predykcji z historią":
        ticker_options = load_tickers_and_names(config)
        default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")

        ticker_option = st.selectbox(
            "Wybierz spółkę z listy lub wpisz własną:",
            options=["Wpisz ręcznie"] + list(ticker_options.values()),
            index=0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
        )

        ticker_input = default_ticker
        if ticker_option == "Wpisz ręcznie":
            ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker)
        else:
            ticker_input = [k for k, v in ticker_options.items() if v == ticker_option][0]

        if st.button("Porównaj predykcje z historią"):
            with st.spinner('Trwa porównywanie predykcji z historią...'):
                try:
                    max_prediction_length = config['model']['max_prediction_length']
                    trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
                    start_date = trim_date - pd.Timedelta(days=years * 365)
                    
                    ticker_data, original_close, median, lower_bound, upper_bound, historical_close = predictor.predict_historical(
                        ticker_input, start_date, datetime.now(), trim_date
                    )
                    create_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_close)
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas porównywania predykcji z historią dla {ticker_input}: {str(e)}")

    elif page == "Benchmark":
        st.write("Spółki użyte w benchmarku:", " ".join(benchmark_tickers))

        if st.button("Generuj benchmark"):
            with st.spinner('Trwa generowanie benchmarku...'):
                try:
                    all_results = asyncio.run(create_benchmark_plot(config, benchmark_tickers, {}, years))
                    benchmark_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    save_benchmark_to_csv(benchmark_date, all_results, config['model_name'])
                except Exception as e:
                    logger.error(f"Error generating benchmark: {e}")
                    st.error(f"Wystąpił błąd podczas generowania benchmarku: {str(e)}")

        st.subheader("Historia benchmarków")
        benchmark_history = load_benchmark_history(benchmark_tickers)
        format_dict = {
            ('Podstawowe', 'Date'): '{}',
            ('Podstawowe', 'Model_Name'): '{}'
        }
        for ticker in benchmark_tickers:
            format_dict[(ticker, 'Acc')] = '{:.2f}%'
            format_dict[(ticker, 'MAPE')] = '{:.2f}%'
            format_dict[(ticker, 'MAE')] = '{:.2f}'
            format_dict[(ticker, 'DirAcc')] = '{:.2f}%'
        for metric in ['Acc', 'MAPE', 'DirAcc']:
            format_dict[('Średnie', metric)] = '{:.2f}%'
        format_dict[('Średnie', 'MAE')] = '{:.2f}'
        st.dataframe(benchmark_history.style.format(format_dict))

if __name__ == "__main__":
    main()