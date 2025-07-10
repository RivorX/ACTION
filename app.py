import streamlit as st
import pandas as pd
import numpy as np
import torch
import yaml
import plotly.graph_objs as go
import logging
from datetime import datetime, timedelta
import os
from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.prediction_engine import load_data_and_model, preprocess_data, generate_predictions

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Wczytuje konfigurację z pliku YAML."""
    try:
        return ConfigManager().config
    except Exception as e:
        logger.error(f"Błąd wczytywania config.yaml: {e}")
        st.error("Błąd wczytywania pliku konfiguracyjnego.")
        raise

def load_tickers_and_names():
    """Wczytuje listę tickerów i pełnych nazw firm z plików konfiguracyjnych."""
    try:
        config = ConfigManager()
        tickers_file = config.get('data.tickers_file', 'config/training_tickers.yaml')
        company_names_file = config.get('data.company_names_file', 'config/company_names.yaml')
        
        with open(tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            all_tickers = list(dict.fromkeys(all_tickers))  # Usuwa duplikaty

        with open(company_names_file, 'r') as f:
            company_names = yaml.safe_load(f)['company_names']
        
        ticker_options = {ticker: f"{ticker} - {company_names.get(ticker, 'Nieznana firma')}" for ticker in all_tickers}
        return ticker_options
    except Exception as e:
        logger.error(f"Błąd wczytywania tickerów lub nazw firm: {e}")
        st.error("Błąd wczytywania listy tickerów lub nazw firm.")
        return {}

def create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker):
    """Tworzy wykres cen i predykcji."""
    last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config['model']['max_prediction_length'], freq='D')
    historical_dates = ticker_data['Date'].tolist()
    
    all_dates = historical_dates + [d.to_pydatetime() for d in pred_dates]
    all_close = original_close.tolist() + median.tolist()
    all_lower_bound = [None] * len(original_close) + lower_bound.tolist()
    all_upper_bound = [None] * len(original_close) + upper_bound.tolist()
    
    plot_data = pd.DataFrame({
        'Date': all_dates,
        'Close': all_close,
        'Lower_Bound': all_lower_bound,
        'Upper_Bound': all_upper_bound
    })

    plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Close'],
        mode='lines',
        name='Cena zamknięcia (historyczna i przewidywana)',
        line=dict(color='#0000FF')
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Upper_Bound'],
        mode='lines',
        name='Górny kwantyl (90%)',
        line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Lower_Bound'],
        mode='lines',
        name='Dolny kwantyl (10%)',
        line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))

    split_date = pd.Timestamp(last_date).isoformat()
    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=split_date,
        y=1.05,
        xref="x",
        yref="paper",
        text="Początek predykcji",
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    fig.update_layout(
        title=f"Ceny akcji dla {ticker}",
        xaxis_title="Data",
        yaxis_title="Cena zamknięcia",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    
    pred_df = pd.DataFrame({
        'Data': [d.to_pydatetime() for d in pred_dates],
        'Przewidywana cena': median.tolist(),
        'Dolny kwantyl (10%)': lower_bound.tolist(),
        'Górny kwantyl (90%)': upper_bound.tolist()
    })
    st.subheader("Przewidywane ceny")
    st.dataframe(pred_df.style.format({
        'Data': '{:%Y-%m-%d}',
        'Przewidywana cena': '{:.2f}',
        'Dolny kwantyl (10%)': '{:.2f}',
        'Górny kwantyl (90%)': '{:.2f}'
    }))

def create_historical_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker, historical_close):
    """Tworzy wykres porównujący predykcje z historią."""
    last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config['model']['max_prediction_length'], freq='D')
    historical_dates = ticker_data['Date'].tolist()
    
    pred_date_range = pd.DataFrame({'Date': pred_dates})
    pred_date_range['Date'] = pd.to_datetime(pred_date_range['Date'], utc=True)
    historical_close = historical_close.reindex(pred_date_range['Date'], method='ffill')
    
    logger.info(f"Długość historical_dates: {len(historical_dates)}")
    logger.info(f"Długość pred_dates: {len(pred_dates)}")
    logger.info(f"Długość original_close: {len(original_close)}")
    logger.info(f"Długość historical_close po reindex: {len(historical_close)}")
    logger.info(f"Długość median: {len(median)}")
    
    all_dates = historical_dates + pred_date_range['Date'].tolist()
    all_close = original_close.tolist() + historical_close.tolist()
    all_pred_close = [None] * len(historical_dates) + median.tolist()
    
    logger.info(f"Długość all_dates: {len(all_dates)}, all_close: {len(all_close)}, all_pred_close: {len(all_pred_close)}")
    if len(all_dates) != len(all_close) or len(all_dates) != len(all_pred_close):
        logger.error(f"Niezgodność długości: all_dates={len(all_dates)}, all_close={len(all_close)}, all_pred_close={len(all_pred_close)}")
        raise ValueError("Wszystkie tablice muszą mieć tę samą długość")

    plot_data = pd.DataFrame({
        'Date': all_dates,
        'Close': all_close,
        'Predicted_Close': all_pred_close
    })

    plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Close'],
        mode='lines',
        name='Cena zamknięcia (historyczna)',
        line=dict(color='#0000FF')
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Predicted_Close'],
        mode='lines',
        name='Przewidywana cena zamknięcia',
        line=dict(color='#FFA500', dash='dash')
    ))

    # Dodanie kwantyli z przekazanych danych
    all_lower_bound = [None] * len(historical_dates) + lower_bound.tolist()
    all_upper_bound = [None] * len(historical_dates) + upper_bound.tolist()
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=all_upper_bound,
        mode='lines',
        name='Górny kwantyl (90%)',
        line=dict(color='rgba(255, 165, 0, 0.3)', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=all_lower_bound,
        mode='lines',
        name='Dolny kwantyl (10%)',
        line=dict(color='rgba(255, 165, 0, 0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.1)'
    ))

    split_date = pd.Timestamp(last_date).isoformat()
    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=split_date,
        y=1.05,
        xref="x",
        yref="paper",
        text="Początek predykcji",
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    fig.update_layout(
        title=f"Porównanie predykcji z historią dla {ticker}",
        xaxis_title="Data",
        yaxis_title="Cena zamknięcia",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def create_benchmark_plot(config, benchmark_tickers, historical_close_dict):
    """Tworzy wykres benchmarku dla wszystkich spółek z możliwością wyboru widoczności w legendzie."""
    all_results = {}
    temp_raw_data_path = 'data/temp_benchmark_data.csv'
    accuracy_scores = {}
    
    for ticker in benchmark_tickers:
        try:
            new_data, dataset, normalizers, model = load_data_and_model(config, ticker, temp_raw_data_path, historical_mode=True)
            ticker_data, original_close = preprocess_data(config, new_data, ticker, normalizers, historical_mode=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                median, _, _ = generate_predictions(config, dataset, model, ticker_data)
            last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config['model']['max_prediction_length'], freq='D')
            
            # Poprawione dane historyczne
            historical_data = historical_close_dict[ticker].loc[:last_date].reindex(ticker_data['Date'], method='ffill')
            historical_close = historical_close_dict[ticker].reindex(pd.to_datetime(pred_dates), method='ffill')
            
            if len(median) == len(historical_close):
                differences = np.abs(median - historical_close)
                relative_diff = (differences / historical_close) * 100
                accuracy = 100 - np.mean(relative_diff)
                accuracy_scores[ticker] = accuracy
                logger.info(f"Procentowa zgodność dla {ticker}: {accuracy:.2f}%")
            else:
                logger.warning(f"Niezgodna długość predykcji i danych historycznych dla {ticker}")
                accuracy_scores[ticker] = 0.0
            
            all_results[ticker] = {
                'historical_dates': ticker_data['Date'].tolist(),
                'historical_close': historical_data.tolist(),
                'pred_dates': [d.to_pydatetime() for d in pred_dates],
                'predictions': median.tolist(),
                'historical_pred_close': historical_close.tolist()
            }
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania {ticker}: {e}")
            accuracy_scores[ticker] = 0.0
        finally:
            if os.path.exists(temp_raw_data_path):
                os.remove(temp_raw_data_path)
                logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

    fig = go.Figure()
    colors = ['#0000FF', '#00FF00', '#FF0000', '#800080', '#FFA500', '#00FFFF', '#FF00FF', '#FFFF00', '#A52A2A', '#808080']
    for idx, (ticker, data) in enumerate(all_results.items()):
        color_idx = idx % len(colors)
        # Linia ciągła dla predykcji
        fig.add_trace(go.Scatter(
            x=data['pred_dates'],
            y=data['predictions'],
            mode='lines',
            name=f'{ticker} (Predykcja)',
            line=dict(color=colors[color_idx]),
            legendgroup=ticker
        ))
        # Linia przerywana dla danych historycznych
        fig.add_trace(go.Scatter(
            x=data['historical_dates'],
            y=data['historical_close'],
            mode='lines',
            name=f'{ticker} (Historia)',
            line=dict(color=colors[color_idx], dash='dash'),
            legendgroup=ticker
        ))

    fig.update_layout(
        title="Benchmark predykcji z historią dla wszystkich spółek",
        xaxis_title="Data",
        yaxis_title="Cena zamknięcia",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Procentowa zgodność predykcji z historią")
    for ticker, accuracy in accuracy_scores.items():
        st.write(f"{ticker}: {accuracy:.2f}%")

    return accuracy_scores

def save_benchmark_to_csv(benchmark_date, accuracy_scores):
    """Zapisuje wyniki benchmarku do pliku CSV z historią."""
    csv_file = 'data/benchmarks_history.csv'  # Zapis do katalogu data
    average_accuracy = np.mean(list(accuracy_scores.values())) if accuracy_scores else 0.0
    
    new_data = {'Date': [benchmark_date], **{ticker: [accuracy] for ticker, accuracy in accuracy_scores.items()}, 'Average_Accuracy': [average_accuracy]}
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, dtype=str)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.to_csv(csv_file, index=False)
    logger.info(f"Wyniki benchmarku zapisane do {csv_file}")

def load_benchmark_history(benchmark_tickers):
    """Wczytuje historię benchmarków z pliku CSV."""
    csv_file = 'data/benchmarks_history.csv'  # Wczytywanie z katalogu data
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype=str)  # Wczytywanie wszystkiego jako stringi
        # Uzupełnij brakujące kolumny tickerami z zerami jako stringami
        missing_tickers = [t for t in benchmark_tickers if t not in df.columns]
        for ticker in missing_tickers:
            df[ticker] = '0.0'
        # Zwracamy czysty DataFrame
        return df[['Date'] + benchmark_tickers + ['Average_Accuracy']].fillna('0.0')
    return pd.DataFrame(columns=['Date'] + benchmark_tickers + ['Average_Accuracy']).fillna('0.0')

def load_benchmark_tickers():
    """Wczytuje listę tickerów dla benchmarku z pliku YAML."""
    try:
        config = ConfigManager()
        benchmark_tickers_file = 'config/benchmark_tickers.yaml'
        with open(benchmark_tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            return list(dict.fromkeys(all_tickers))  # Usuwa duplikaty
    except Exception as e:
        logger.error(f"Błąd wczytywania benchmark_tickers.yaml: {e}")
        st.error("Błąd wczytywania listy tickerów dla benchmarku.")
        return []

def main():
    """Główna funkcja aplikacji Streamlit."""
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    config = load_config()
    temp_raw_data_path = 'data/temp_stock_data.csv'

    # Lista tickerów dla benchmarku z nowego pliku
    benchmark_tickers = load_benchmark_tickers()

    page = st.sidebar.selectbox("Wybierz stronę", ["Predykcje przyszłości", "Porównanie predykcji z historią", "Benchmark"])

    if page == "Predykcje przyszłości":
        ticker_options = load_tickers_and_names()
        default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")

        ticker_option = st.selectbox(
            "Wybierz spółkę z listy lub wpisz własną:",
            options=["Wpisz ręcznie"] + list(ticker_options.values()),
            index=0 if not default_ticker in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
        )
        ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker if ticker_option == "Wpisz ręcznie" else [k for k, v in ticker_options.items() if v == ticker_option][0])

        if st.button("Generuj predykcje"):
            try:
                new_data, dataset, normalizers, model = load_data_and_model(config, ticker_input, temp_raw_data_path)
                ticker_data, original_close = preprocess_data(config, new_data, ticker_input, normalizers)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    median, lower_bound, upper_bound = generate_predictions(config, dataset, model, ticker_data)
                create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input)
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
            index=0 if not default_ticker in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
        )
        ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker if ticker_option == "Wpisz ręcznie" else [k for k, v in ticker_options.items() if v == ticker_option][0])

        if st.button("Porównaj predykcje z historią"):
            try:
                max_prediction_length = config['model']['max_prediction_length']
                trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
                start_date = trim_date - pd.Timedelta(days=365)
                
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
                    start_date = trim_date - pd.Timedelta(days=365)
                    
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

        # Wyświetlanie historii benchmarków
        st.subheader("Historia benchmarków")
        benchmark_history = load_benchmark_history(benchmark_tickers)
        st.dataframe(benchmark_history)

if __name__ == "__main__":
    main()