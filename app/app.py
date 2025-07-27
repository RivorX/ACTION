import streamlit as st
import pandas as pd
import torch
import logging
import os
from datetime import datetime, timedelta
import sys
# Dodaj katalog główny projektu do sys.path, żeby import scripts działał
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.prediction_engine import load_data_and_model, preprocess_data, generate_predictions
from app.config_loader import load_config, load_tickers_and_names, load_benchmark_tickers
from app.plot_utils import create_plot, create_historical_plot
from app.benchmark_utils import create_benchmark_plot, save_benchmark_to_csv, load_benchmark_history

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Główna funkcja aplikacji Streamlit."""
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    config = load_config()
    temp_raw_data_path = 'data/temp_stock_data.csv'

    benchmark_tickers = load_benchmark_tickers()

    page = st.sidebar.selectbox("Wybierz stronę", ["Predykcje przyszłości", "Porównanie predykcji z historią", "Benchmark"])

    if page == "Predykcje przyszłości":
        ticker_options = load_tickers_and_names()
        default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")

        ticker_option = st.selectbox(
            "Wybierz spółkę z listy lub wpisz własną:",
            options=["Wpisz ręcznie"] + list(ticker_options.values()),
            index=0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
        )

        # Pokazuj pole tekstowe tylko dla opcji "Wpisz ręcznie"
        ticker_input = default_ticker
        if ticker_option == "Wpisz ręcznie":
            ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker)
        else:
            ticker_input = [k for k, v in ticker_options.items() if v == ticker_option][0]

        if st.button("Generuj predykcje"):
            try:
                new_data, dataset, normalizers, model = load_data_and_model(config, ticker_input, temp_raw_data_path)
                ticker_data, original_close = preprocess_data(config, new_data, ticker_input, normalizers)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    median, lower_bound, upper_bound = generate_predictions(config, dataset, model, ticker_data)
                create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input)
            except Exception as e:
                logger.error(f"Błąd podczas generowania predykcji: {e}")
                st.error("Wystąpił błąd podczas generowania predykcji.")
            finally:
                if os.path.exists(temp_raw_data_path):
                    os.remove(temp_raw_data_path)
                    logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

    elif page == "Porównanie predykcji z historią":
        ticker_options = load_tickers_and_names()
        default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")

        ticker_option = st.selectbox(
            "Wybierz spółkę z listy lub wpisz własną:",
            options=["Wpisz ręcznie"] + list(ticker_options.values()),
            index=0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
        )

        # Pokazuj pole tekstowe tylko dla opcji "Wpisz ręcznie"
        ticker_input = default_ticker
        if ticker_option == "Wpisz ręcznie":
            ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker)
        else:
            ticker_input = [k for k, v in ticker_options.items() if v == ticker_option][0]

        if st.button("Porównaj predykcje z historią"):
            try:
                max_prediction_length = config['model']['max_prediction_length']
                trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
                start_date = trim_date - pd.Timedelta(days=720)
                
                fetcher = DataFetcher(ConfigManager())
                full_data = fetcher.fetch_stock_data(ticker_input, start_date, datetime.now())
                if full_data.empty:
                    raise ValueError("Brak pełnych danych historycznych")
                
                full_data = full_data[full_data['Ticker'] == ticker_input].copy()
                full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
                new_data = full_data[full_data['Date'] <= trim_date].copy()
                if new_data.empty:
                    raise ValueError(f"Brak danych przed {trim_date} dla {ticker_input}")
                
                new_data.to_csv(temp_raw_data_path, index=False)
                logger.info(f"Dane dla {ticker_input} zapisane do {temp_raw_data_path}")
                
                _, dataset, normalizers, model = load_data_and_model(config, ticker_input, temp_raw_data_path, historical_mode=True)
                ticker_data, original_close = preprocess_data(config, new_data, ticker_input, normalizers, historical_mode=True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    median, lower_bound, upper_bound = generate_predictions(config, dataset, model, ticker_data)
                
                full_data.set_index('Date', inplace=True)
                historical_close = full_data['Close']
                create_historical_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_close)
            except Exception as e:
                logger.error(f"Błąd podczas porównywania predykcji z historią: {e}")
                st.error("Wystąpił błąd podczas porównywania predykcji z historią.")
            finally:
                if os.path.exists(temp_raw_data_path):
                    os.remove(temp_raw_data_path)
                    logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

    elif page == "Benchmark":
        st.write("Spółki użyte w benchmarku:", " ".join(benchmark_tickers))

        if st.button("Generuj benchmark"):
            with st.spinner('Trwa generowanie benchmarku...'):
                try:
                    max_prediction_length = config['model']['max_prediction_length']
                    trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
                    start_date = trim_date - pd.Timedelta(days=720)
                    
                    fetcher = DataFetcher(ConfigManager())
                    historical_close_dict = {}
                    for ticker in benchmark_tickers:
                        full_data = fetcher.fetch_stock_data(ticker, start_date, datetime.now())
                        if full_data.empty:
                            logger.error(f"Brak danych dla {ticker}")
                            continue
                        full_data = full_data[full_data['Ticker'] == ticker].copy()
                        full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
                        full_data.set_index('Date', inplace=True)
                        historical_close_dict[ticker] = full_data['Close']
                    
                    accuracy_scores = create_benchmark_plot(config, benchmark_tickers, historical_close_dict)
                    benchmark_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    save_benchmark_to_csv(benchmark_date, accuracy_scores)
                except Exception as e:
                    logger.error(f"Błąd podczas generowania benchmarku: {e}")
                    st.error("Wystąpił błąd podczas generowania benchmarku.")
                finally:
                    if os.path.exists(temp_raw_data_path):
                        os.remove(temp_raw_data_path)
                        logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

        st.subheader("Historia benchmarków")
        benchmark_history = load_benchmark_history(benchmark_tickers)
        st.dataframe(benchmark_history)

if __name__ == "__main__":
    main()