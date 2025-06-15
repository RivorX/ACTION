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
import pkg_resources
from datetime import datetime, timedelta
import pickle

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Wyświetlanie w konsoli
def print_to_console(message):
    print(message)

# Weryfikacja wersji bibliotek
required_versions = {
    'numpy': '2.0.0',
    'pandas': '2.2.2'
}
for lib, expected_version in required_versions.items():
    try:
        installed_version = pkg_resources.get_distribution(lib).version
        logger.info(f"Wersja {lib}: {installed_version}")
        if installed_version != expected_version:
            logger.warning(f"Zalecana wersja {lib} to {expected_version}, zainstalowana: {installed_version}")
    except pkg_resources.DistributionNotFound:
        logger.error(f"Biblioteka {lib} nie jest zainstalowana")
        st.error(f"Biblioteka {lib} nie jest zainstalowana")
        raise

st.title("Stock Price Predictor")

# Wczytaj konfigurację
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Błąd wczytywania config.yaml: {e}")
    st.error("Błąd wczytywania pliku konfiguracyjnego.")
    raise

tickers = config['data']['tickers']
selected_ticker = st.selectbox("Wybierz spółkę", tickers)

# Przygotuj dane i model po naciśnięciu przycisku
if st.button("Generuj predykcje"):
    # Wczytaj dane
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
        logger.info(f"Hiperparametry z checkpointu: {hyperparams}")
        
        if 'hidden_continuous_size' not in hyperparams:
            hyperparams['hidden_continuous_size'] = 64
            logger.warning("hidden_continuous_size nie znaleziono w hiperparametrach, ustawiono domyślnie na 64")
        
        if 'loss' in hyperparams and isinstance(hyperparams['loss'], str):
            if 'QuantileLoss' in hyperparams['loss']:
                hyperparams['loss'] = metrics.QuantileLoss(quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9]))
            else:
                hyperparams['loss'] = metrics.MAE()
        
        model = build_model(dataset, config, hyperparams=hyperparams)
        logger.info(f"Hiperparametry użyte do budowy modelu: {dict(model.hparams)}")
        
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as e:
            logger.error(f"Błąd wczytywania state_dict: {e}")
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(checkpoint["state_dict"].keys())
            logger.info(f"Klucze obecne tylko w modelu: {model_keys - checkpoint_keys}")
            logger.info(f"Klucze obecne tylko w checkpointcie: {checkpoint_keys - model_keys}")
            raise
        
        model.eval()
        model.to('cpu')
        logger.info("Model wczytany poprawnie.")
    except Exception as e:
        logger.error(f"Błąd wczytywania modelu: {e}")
        st.error(f"Błąd wczytywania modelu: {e}")
        raise

    # Filtruj dane dla wybranego tickera
    try:
        data = pd.read_csv(config['data']['raw_data_path'])
        ticker_data = data[data['Ticker'] == selected_ticker].copy()
        ticker_data = ticker_data.reset_index(drop=True)
        ticker_data['Date'] = pd.to_datetime(ticker_data['Date'], utc=True)
        
        # Zapisz oryginalne ceny przed preprocessingu
        original_close = ticker_data['Close'].copy()
        
        # Dodaj wszystkie cechy za pomocą add_features
        ticker_data = add_features(ticker_data)
        ticker_data = ticker_data.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
        ticker_data = ticker_data[(ticker_data['Close'] > 0) & (ticker_data['High'] >= ticker_data['Low'])]
        ticker_data['time_idx'] = range(len(ticker_data))
        ticker_data['group_id'] = selected_ticker
        
        # Transformacja logarytmiczna
        log_features = [
            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", 
            "BB_Middle", "BB_Upper", "BB_Lower", "ATR"
        ]
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
        
        # Konwersja kolumn kategorycznych na stringi
        categorical_columns = ['Day_of_Week', 'Month']
        for cat_col in categorical_columns:
            if cat_col in ticker_data.columns:
                ticker_data[cat_col] = ticker_data[cat_col].astype(str)
        
        logger.info(f"Kolumny ticker_data: {ticker_data.columns.tolist()}")
        logger.info(f"Pierwsze 5 wierszy ticker_data:\n{ticker_data.head().to_string()}")
        logger.info(f"Liczba wierszy ticker_data po dropna: {len(ticker_data)}")
    except Exception as e:
        logger.error(f"Błąd wczytywania lub preprocessingu danych raw: {e}")
        st.error("Błąd wczytywania lub preprocessingu danych surowych. Sprawdź plik raw_data_path.")
        raise
    
    # Przygotuj dataset dla predykcji
    try:
        ticker_dataset = TimeSeriesDataSet.from_parameters(
            dataset.get_parameters(),
            ticker_data,
            predict_mode=True,
            max_prediction_length=config['model']['max_prediction_length']
        ).to_dataloader(train=False, batch_size=1, num_workers=4)
        logger.info(f"Liczba sekwencji w ticker_dataset: {len(ticker_dataset)}")
    except Exception as e:
        logger.error(f"Błąd przygotowania datasetu predykcyjnego: {e}")
        st.error(f"Błąd przygotowania danych do predykcji: {e}")
        raise
    
    # Wykonaj predykcję
    try:
        with torch.no_grad():
            predictions = model.predict(ticker_dataset, mode="quantiles", return_x=True)
        logger.info("Predykcja wykonana poprawnie.")
        logger.info(f"Kształt predictions.output: {predictions.output.shape}")
    except Exception as e:
        logger.error(f"Błąd predykcji: {e}")
        st.error(f"Błąd podczas wykonywania predykcji: {e}")
        raise
    
    # Konwertuj predykcje na numpy i odnormalizuj
    pred_array = predictions.output
    logger.info(f"Kształt pred_array przed odnormalizacją: {pred_array.shape}")
    print_to_console(f"Surowe predykcje (pierwsze 5 wartości dla mediany): {pred_array[0, :5, 1].tolist()}")

    # Pobierz normalizer z datasetu i odnormalizuj
    target_normalizer = dataset.target_normalizer
    try:
        pred_array = pred_array.to('cpu')
        # Odnormalizuj
        pred_array = target_normalizer.inverse_transform(pred_array)
        # Odwróć transformację logarytmiczną
        pred_array = np.expm1(pred_array.numpy())
        if len(pred_array.shape) == 3:
            median = pred_array[0, :, 1]
            lower_bound = pred_array[0, :, 0]
            upper_bound = pred_array[0, :, 2]
        else:
            logger.error(f"Nieoczekiwany kształt pred_array: {pred_array.shape}")
            st.error(f"Nieoczekiwany kształt tablicy predykcji: {pred_array.shape}")
            raise ValueError(f"Nieoczekiwany kształt pred_array: {pred_array.shape}")
        
        # Logowanie odnormalizowanych predykcji
        logger.info(f"Oryginalne ceny historyczne (ostatnie 5): {original_close.tail().tolist()}")
        logger.info(f"Przewidywane ceny (pierwsze 5 dla mediany): {median[:5].tolist()}")
        print_to_console(f"Przewidywane ceny (pierwsze 5 dla mediany): {median[:5].tolist()}")
    except Exception as e:
        logger.error(f"Błąd odnormalizacji: {e}")
        st.error(f"Błąd odnormalizacji predykcji: {e}")
        raise

    logger.info(f"Kształt pred_array po odnormalizacji: {pred_array.shape}")
    logger.info(f"Przykładowe wartości predykcji: median[0]={median[0]}, lower_bound[0]={lower_bound[0]}, upper_bound[0]={upper_bound[0]}")

    # Przygotuj dane do wykresu
    last_date = ticker_data['Date'].iloc[-1]
    pred_dates = [last_date + timedelta(days=i) for i in range(1, config['model']['max_prediction_length'] + 1)]
    historical_dates = ticker_data['Date'].tolist()
    historical_close = original_close.tolist()  # Użyj oryginalnych cen
    
    all_dates = historical_dates + pred_dates
    all_close = historical_close + median.tolist()
    all_lower_bound = [None] * len(historical_close) + lower_bound.tolist()
    all_upper_bound = [None] * len(historical_close) + upper_bound.tolist()
    
    plot_data = pd.DataFrame({
        'Date': all_dates,
        'Close': all_close,
        'Lower_Bound': all_lower_bound,
        'Upper_Bound': all_upper_bound
    })
    plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)

    # Suwak do wyboru zakresu czasowego
    min_date = plot_data['Date'].min()
    max_date = plot_data['Date'].max()
    default_start = min_date
    default_end = max_date
    if len(plot_data) > 100:
        default_start = max_date - timedelta(days=100)
    
    selected_range = st.slider(
        "Wybierz zakres dat",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
        format="YYYY-MM-DD"
    )
    
    mask = (plot_data['Date'] >= pd.to_datetime(selected_range[0], utc=True)) & (plot_data['Date'] <= pd.to_datetime(selected_range[1], utc=True))
    filtered_data = plot_data[mask]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['Close'],
        mode='lines',
        name='Cena zamknięcia (historyczna i przewidywana)',
        line=dict(color='blue')
    ))
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
    
    fig.update_layout(
        title=f"Ceny akcji dla {selected_ticker} (historyczne i przewidywane)",
        xaxis_title="Data",
        yaxis_title="Cena zamknięcia (PLN)",
        showlegend=True,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Wyświetl przewidywane ceny w tabeli
    pred_df = pd.DataFrame({
        'Data': pred_dates,
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