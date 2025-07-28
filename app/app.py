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
# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.prediction_engine import load_data_and_model, preprocess_data, generate_predictions
from app.config_loader import load_config, load_tickers_and_names, load_benchmark_tickers
from app.plot_utils import create_plot, create_historical_plot
from app.benchmark_utils import create_benchmark_plot, save_benchmark_to_csv, load_benchmark_history

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    """Handles business logic for stock price prediction."""
    def __init__(self, config):
        self.config = config
        self.fetcher = DataFetcher(ConfigManager())  # Przekazujemy ConfigManager

    def fetch_stock_data(self, ticker, start_date, end_date):
        """Fetches stock data without caching."""
        return self.fetcher.fetch_stock_data_sync(ticker, start_date, end_date)

    def predict(self, ticker, start_date, end_date, temp_raw_data_path):
        """Generates predictions for a given ticker."""
        try:
            # Fetch data
            new_data = self.fetch_stock_data(ticker, start_date, end_date)
            if new_data.empty:
                raise ValueError(f"No data available for {ticker}")

            new_data.to_csv(temp_raw_data_path, index=False)
            logger.info(f"Data for {ticker} saved to {temp_raw_data_path}")

            # Load model and preprocess data
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
            if os.path.exists(temp_raw_data_path):
                os.remove(temp_raw_data_path)
                logger.info(f"Temporary file {temp_raw_data_path} removed.")

    def predict_historical(self, ticker, start_date, end_date, trim_date, temp_raw_data_path):
        """Compares predictions with historical data."""
        try:
            # Fetch full data
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
            
            # Load model and preprocess data
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
            if os.path.exists(temp_raw_data_path):
                os.remove(temp_raw_data_path)
                logger.info(f"Temporary file {temp_raw_data_path} removed.")

def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    config = load_config()
    temp_raw_data_path = 'data/temp_stock_data.csv'
    predictor = StockPredictor(config)  # Przekazujemy słownik config
    benchmark_tickers = load_benchmark_tickers(config)  # Przekazujemy config

    page = st.sidebar.selectbox("Wybierz stronę", ["Predykcje przyszłości", "Porównanie predykcji z historią", "Benchmark"])

    if page == "Predykcje przyszłości":
        ticker_options = load_tickers_and_names(config)  # Przekazujemy config
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
                    start_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=730)
                    ticker_data, original_close, median, lower_bound, upper_bound = predictor.predict(
                        ticker_input, start_date, datetime.now(), temp_raw_data_path
                    )
                    create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input)
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas generowania predykcji dla {ticker_input}: {str(e)}")

    elif page == "Porównanie predykcji z historią":
        ticker_options = load_tickers_and_names(config)  # Przekazujemy config
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
                    start_date = trim_date - pd.Timedelta(days=720)
                    
                    ticker_data, original_close, median, lower_bound, upper_bound, historical_close = predictor.predict_historical(
                        ticker_input, start_date, datetime.now(), trim_date, temp_raw_data_path
                    )
                    create_historical_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_close)
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas porównywania predykcji z historią dla {ticker_input}: {str(e)}")

    elif page == "Benchmark":
        st.write("Spółki użyte w benchmarku:", " ".join(benchmark_tickers))

        if st.button("Generuj benchmark"):
            with st.spinner('Trwa generowanie benchmarku...'):
                try:
                    # Use existing event loop for Streamlit compatibility
                    loop = asyncio.get_event_loop()
                    all_results = loop.run_until_complete(
                        create_benchmark_plot(config, benchmark_tickers, {})
                    )
                    benchmark_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    save_benchmark_to_csv(benchmark_date, all_results)
                except Exception as e:
                    logger.error(f"Error generating benchmark: {e}")
                    st.error(f"Wystąpił błąd podczas generowania benchmarku: {str(e)}")
                finally:
                    if os.path.exists(temp_raw_data_path):
                        os.remove(temp_raw_data_path)
                        logger.info(f"Temporary file {temp_raw_data_path} removed.")

        st.subheader("Historia benchmarków")
        benchmark_history = load_benchmark_history(benchmark_tickers)
        st.dataframe(benchmark_history)

if __name__ == "__main__":
    main()