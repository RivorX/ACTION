import streamlit as st
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, metrics
from scripts.model import build_model
from scripts.preprocessor import add_features
import torch
import yaml
import plotly.graph_objs as go
import logging
from datetime import datetime, timedelta
import pickle
import os
from scripts.data_fetcher import fetch_stock_data

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Funkcja pomocnicza do wczytywania konfiguracji
def load_config():
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Błąd wczytywania config.yaml: {e}")
        st.error("Błąd wczytywania pliku konfiguracyjnego.")
        raise

# Funkcja do wczytywania danych i modelu
def load_data_and_model(config, selected_ticker, temp_raw_data_path, historical_mode=False, trim_days=0):
    # Pobierz dane
    start_date = datetime.now() - pd.Timedelta(days=365 + trim_days)
    new_data = fetch_stock_data(selected_ticker, start_date, datetime.now())
    if new_data is None or new_data.empty:
        logger.error(f"Nie udało się pobrać danych dla {selected_ticker}")
        st.error(f"Nie udało się pobrać danych dla {selected_ticker}")
        raise ValueError("Brak danych")

    new_data.to_csv(temp_raw_data_path, index=False)
    logger.info(f"Dane dla {selected_ticker} zapisane do {temp_raw_data_path}")

    # Wczytaj dataset
    try:
        dataset = torch.load(config['data']['processed_data_path'], weights_only=False, map_location=torch.device('cpu'))
        logger.info("Dataset wczytany poprawnie.")
    except Exception as e:
        logger.error(f"Błąd wczytywania datasetu: {e}")
        st.error("Błąd wczytywania danych. Sprawdź plik processed_data_path.")
        raise

    # Wczytaj normalizery
    try:
        with open(config['data']['normalizers_path'], 'rb') as f:
            normalizers = pickle.load(f)
        logger.info(f"Wczytano normalizery z: {config['data']['normalizers_path']}")
    except Exception as e:
        logger.error(f"Błąd wczytywania normalizerów: {e}")
        st.error("Błąd wczytywania normalizerów. Sprawdź plik normalizers_path.")
        raise

    # Wczytaj model
    try:
        checkpoint = torch.load(config['paths']['checkpoint_path'], map_location=torch.device('cpu'), weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if 'hidden_continuous_size' not in hyperparams:
            hyperparams['hidden_continuous_size'] = 64
            logger.warning("hidden_continuous_size nie znaleziono, ustawiono domyślnie na 64")
        if 'loss' in hyperparams and isinstance(hyperparams['loss'], str):
            hyperparams['loss'] = metrics.QuantileLoss(quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9])) if 'QuantileLoss' in hyperparams['loss'] else metrics.MAE()
        model = build_model(dataset, config, hyperparams=hyperparams)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to('cpu')
        logger.info("Model wczytany poprawnie.")
    except Exception as e:
        logger.error(f"Błąd wczytywania modelu: {e}")
        st.error(f"Błąd wczytywania modelu: {e}")
        raise

    return new_data, dataset, normalizers, model

# Funkcja do preprocessingu danych
def preprocess_data(config, ticker_data, selected_ticker, normalizers, historical_mode=False, trim_days=0):
    ticker_data = ticker_data[ticker_data['Ticker'] == selected_ticker].copy().reset_index(drop=True)
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date'], utc=True)
    
    # Zapisz oryginalne ceny
    original_close = ticker_data['Close'].copy()
    
    # Przytnij dane, jeśli w trybie historycznym
    if historical_mode and trim_days > 0:
        ticker_data = ticker_data.iloc[:-trim_days].copy()
        original_close = original_close.iloc[:-trim_days].copy()
    
    # Dodaj cechy
    ticker_data = add_features(ticker_data)
    ticker_data = ticker_data.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    ticker_data = ticker_data[(ticker_data['Close'] > 0) & (ticker_data['High'] >= ticker_data['Low'])]
    ticker_data['time_idx'] = range(len(ticker_data))
    ticker_data['group_id'] = selected_ticker

    # Transformacja logarytmiczna
    log_features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "BB_Middle", "BB_Upper", "BB_Lower", "ATR"]
    for feature in log_features:
        if feature in ticker_data.columns:
            ticker_data[feature] = np.log1p(ticker_data[feature].clip(lower=0))

    # Normalizacja
    numeric_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
        "MACD", "MACD_Signal", "BB_Middle", "BB_Upper", "BB_Lower", "Stochastic_K",
        "Stochastic_D", "ATR", "OBV", "Price_Change"
    ]
    for feature in numeric_features:
        if feature in ticker_data.columns and feature in normalizers:
            ticker_data[feature] = normalizers[feature].transform(ticker_data[feature].values)

    # Konwersja kolumn kategorycznych
    categorical_columns = ['Day_of_Week', 'Month']
    for cat_col in categorical_columns:
        if cat_col in ticker_data.columns:
            ticker_data[cat_col] = ticker_data[cat_col].astype(str)

    logger.info(f"Kolumny ticker_data: {ticker_data.columns.tolist()}")
    return ticker_data, original_close

# Funkcja do generowania predykcji
def generate_predictions(config, dataset, model, ticker_data):
    ticker_dataset = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(),
        ticker_data,
        predict_mode=True,
        max_prediction_length=config['model']['max_prediction_length']
    ).to_dataloader(train=False, batch_size=1, num_workers=4)

    with torch.no_grad():
        predictions = model.predict(ticker_dataset, mode="quantiles", return_x=True)
    logger.info(f"Kształt predictions.output: {predictions.output.shape}")

    pred_array = predictions.output.to('cpu')
    target_normalizer = dataset.target_normalizer
    pred_array = target_normalizer.inverse_transform(pred_array)
    pred_array = np.expm1(pred_array.numpy())

    if len(pred_array.shape) == 3:
        median = pred_array[0, :, 1]
        lower_bound = pred_array[0, :, 0]
        upper_bound = pred_array[0, :, 2]
    else:
        raise ValueError(f"Nieoczekiwany kształt pred_array: {pred_array.shape}")

    logger.info(f"Przewidywane ceny (pierwsze 5 dla mediany): {median[:5].tolist()}")
    return median, lower_bound, upper_bound

# Funkcja do tworzenia wykresu
def create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, selected_ticker, historical_mode=False, historical_close=None):
    last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config['model']['max_prediction_length'], freq='D')
    historical_dates = ticker_data['Date'].tolist()
    
    if historical_mode and historical_close is not None:
        # Przygotuj daty i ceny dla okresu historycznego i predykcji
        full_data_dates = historical_close.index.tolist()
        pred_start_idx = len(historical_dates)
        pred_end_idx = pred_start_idx + len(median)
        
        if pred_end_idx > len(full_data_dates):
            logger.warning("Predykcje przekraczają dostępne dane historyczne. Przycinanie predykcji.")
            median = median[:len(full_data_dates) - pred_start_idx]
            pred_dates = pred_dates[:len(full_data_dates) - pred_start_idx]

        # Dane historyczne (przycięte) + okres predykcji
        all_dates = historical_dates + [d.to_pydatetime() for d in pred_dates]
        all_close = original_close.tolist() + historical_close.iloc[pred_start_idx:pred_end_idx].tolist()
        all_pred_close = [None] * len(historical_dates) + median.tolist()

        if len(all_dates) != len(all_close) or len(all_dates) != len(all_pred_close):
            logger.error(f"Niezgodność długości: all_dates={len(all_dates)}, all_close={len(all_close)}, all_pred_close={len(all_pred_close)}")
            raise ValueError("Wszystkie tablice muszą mieć tę samą długość")

        plot_data = pd.DataFrame({
            'Date': all_dates,
            'Close': all_close,
            'Predicted_Close': all_pred_close
        })
    else:
        # Predykcje przyszłości
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
    
    # Suwak zakresu dat (domyślnie ostatnie 265 dni)
    min_date = plot_data['Date'].min()
    max_date = plot_data['Date'].max()
    default_start = max_date - pd.Timedelta(days=265)
    if default_start < min_date:
        default_start = min_date
    selected_range = st.slider(
        "Wybierz zakres dat",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(default_start.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM-DD"
    )

    mask = (plot_data['Date'] >= pd.to_datetime(selected_range[0], utc=True)) & (plot_data['Date'] <= pd.to_datetime(selected_range[1], utc=True))
    filtered_data = plot_data[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['Close'],
        mode='lines',
        name='Cena zamknięcia (historyczna)' if historical_mode else 'Cena zamknięcia (historyczna i przewidywana)',
        line=dict(color='blue')
    ))

    if historical_mode and 'Predicted_Close' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Predicted_Close'],
            mode='lines',
            name='Przewidywana cena zamknięcia',
            line=dict(color='orange', dash='dash')
        ))
    elif not historical_mode:
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Upper_Bound'],
            mode='lines',
            name='Górny kwantyl (90%)',
            line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data['Lower_Bound'],
            mode='lines',
            name='Dolny kwantyl (10%)',
            line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))

    # Dodaj linię rozdzielającą dane historyczne od predykcji
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
        title=f"{'Porównanie predykcji z historią' if historical_mode else 'Ceny akcji'} dla {selected_ticker}",
        xaxis_title="Data",
        yaxis_title="Cena zamknięcia (PLN)",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )

    st.plotly_chart(fig, use_container_width=True)
    
    if not historical_mode:
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

# Główna funkcja aplikacji
def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    config = load_config()
    tickers = config['data']['tickers']
    temp_raw_data_path = 'data/temp_stock_data.csv'

    # Nawigacja
    page = st.sidebar.selectbox("Wybierz stronę", ["Predykcje przyszłości", "Porównanie predykcji z historią"])
    selected_ticker = st.selectbox("Wybierz spółkę", tickers)

    if page == "Predykcje przyszłości":
        if st.button("Generuj predykcje"):
            try:
                new_data, dataset, normalizers, model = load_data_and_model(config, selected_ticker, temp_raw_data_path)
                ticker_data, original_close = preprocess_data(config, new_data, selected_ticker, normalizers)
                median, lower_bound, upper_bound = generate_predictions(config, dataset, model, ticker_data)
                create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, selected_ticker)
            finally:
                if os.path.exists(temp_raw_data_path):
                    os.remove(temp_raw_data_path)
                    logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

    elif page == "Porównanie predykcji z historią":
        if st.button("Porównaj predykcje z historią"):
            try:
                trim_days = 90
                new_data, dataset, normalizers, model = load_data_and_model(config, selected_ticker, temp_raw_data_path, historical_mode=True, trim_days=trim_days)
                ticker_data, original_close = preprocess_data(config, new_data, selected_ticker, normalizers, historical_mode=True, trim_days=trim_days)
                median, _, _ = generate_predictions(config, dataset, model, ticker_data)
                
                # Wczytaj pełne dane historyczne do porównania
                full_data = fetch_stock_data(selected_ticker, datetime.now() - pd.Timedelta(days=365), datetime.now())
                if full_data is None or full_data.empty:
                    raise ValueError("Brak pełnych danych historycznych")
                full_data = full_data[full_data['Ticker'] == selected_ticker].copy()
                full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
                full_data.set_index('Date', inplace=True)
                historical_close = full_data['Close']
                
                create_plot(config, ticker_data, original_close, median, None, None, selected_ticker, historical_mode=True, historical_close=historical_close)
            finally:
                if os.path.exists(temp_raw_data_path):
                    os.remove(temp_raw_data_path)
                    logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

if __name__ == "__main__":
    main()