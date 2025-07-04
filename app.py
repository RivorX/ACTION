import streamlit as st
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, metrics
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
import torch
import yaml
import plotly.graph_objs as go
import logging
from datetime import datetime, timedelta
import pickle
import os
from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager

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

def load_data_and_model(config, ticker, temp_raw_data_path, historical_mode=False, trim_days=0):
    """Wczytuje dane i model na podstawie tickera oraz porównuje parametry normalizerów."""
    fetcher = DataFetcher(ConfigManager())
    start_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=365 + trim_days)
    new_data = fetcher.fetch_stock_data(ticker, start_date, datetime.now())
    if new_data.empty:
        logger.error(f"Nie udało się pobrać danych dla {ticker}")
        st.error(f"Nie udało się pobrać danych dla {ticker}")
        raise ValueError("Brak danych")

    new_data.to_csv(temp_raw_data_path, index=False)
    logger.info(f"Dane dla {ticker} zapisane do {temp_raw_data_path}")

    try:
        dataset = torch.load(config['data']['processed_data_path'], weights_only=False, map_location=torch.device('cpu'))
        logger.info("Dataset wczytany poprawnie.")
    except Exception as e:
        logger.error(f"Błąd wczytywania datasetu: {e}")
        st.error("Błąd wczytywania danych.")
        raise

    try:
        with open(config['data']['normalizers_path'], 'rb') as f:
            normalizers = pickle.load(f)
        logger.info(f"Wczytano normalizery z: {config['data']['normalizers_path']}")
    except Exception as e:
        logger.error(f"Błąd wczytywania normalizerów: {e}")
        st.error("Błąd wczytywania normalizerów.")
        raise

    # Porównanie parametrów normalizerów - ZAKTUALIZOWANE dla Relative_Returns
    try:
        relative_returns_normalizer_params = normalizers['Relative_Returns'].get_parameters() if 'Relative_Returns' in normalizers else {}
        target_normalizer_params = dataset.target_normalizer.get_parameters()
        logger.info(f"Parametry normalizera dla Relative_Returns (normalizers.pkl): {relative_returns_normalizer_params}")
        logger.info(f"Parametry normalizera dla target (dataset.target_normalizer): {target_normalizer_params}")
        # Poprawione porównanie
        if not torch.allclose(relative_returns_normalizer_params, target_normalizer_params, rtol=1e-5, atol=1e-8):
            logger.warning("Normalizery dla Relative_Returns różnią się! Może to powodować błędy w predykcjach.")
        else:
            logger.info("Normalizery dla Relative_Returns są zgodne.")
    except Exception as e:
        logger.error(f"Błąd podczas porównywania normalizerów: {e}")

    try:
        checkpoint = torch.load(config['paths']['checkpoint_path'], map_location=torch.device('cpu'), weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if 'hidden_continuous_size' not in hyperparams:
            hyperparams['hidden_continuous_size'] = config['model']['hidden_size'] // 2
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

def preprocess_data(config, ticker_data, ticker, normalizers, historical_mode=False, trim_days=0):
    """Preprocessuje dane dla wybranego tickera z logowaniem wartości przed i po normalizacji."""
    ticker_data = ticker_data[ticker_data['Ticker'] == ticker].copy().reset_index(drop=True)
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date'], utc=True)
    
    original_close = ticker_data['Close'].copy()
    if historical_mode and trim_days > 0:
        ticker_data = ticker_data.iloc[:-trim_days].copy()
        original_close = original_close.iloc[:-trim_days].copy()
    
    preprocessor = DataPreprocessor(config)
    ticker_data = preprocessor.feature_engineer.add_features(ticker_data)
    ticker_data = ticker_data.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    ticker_data = ticker_data[(ticker_data['Close'] > 0) & (ticker_data['High'] >= ticker_data['Low'])]
    ticker_data['time_idx'] = range(len(ticker_data))
    ticker_data['group_id'] = ticker

    # Transformacja logarytmiczna - USUNIĘTO BB cechy
    log_features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR"]
    for feature in log_features:
        if feature in ticker_data.columns:
            ticker_data[feature] = np.log1p(ticker_data[feature].clip(lower=0))

    # Normalizacja - ZAKTUALIZOWANA LISTA CECH
    numeric_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
        "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
        "Close_momentum_1d", "Close_momentum_5d", "Close_vs_MA10", "Close_vs_MA50",
        "Close_percentile_20d", "Close_volatility_5d", "Close_RSI_divergence",
        "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility"
    ]
    for feature in numeric_features:
        if feature in ticker_data.columns and feature in normalizers:
            ticker_data[feature] = normalizers[feature].transform(ticker_data[feature].values)

    categorical_columns = ['Day_of_Week', 'Month']
    for cat_col in categorical_columns:
        if cat_col in ticker_data.columns:
            ticker_data[cat_col] = ticker_data[cat_col].astype(str)

    logger.info(f"Kolumny ticker_data: {ticker_data.columns.tolist()}")
    return ticker_data, original_close

def generate_predictions(config, dataset, model, ticker_data):
    """Generuje predykcje dla podanych danych."""
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
    logger.info(f"Predykcje Relative Returns przed denormalizacją (pierwsze 5 dla mediany): {pred_array[0, :5, 1].tolist()}")
    
    # Denormalizacja Relative Returns
    pred_array = target_normalizer.inverse_transform(pred_array)
    logger.info(f"Predykcje Relative Returns po denormalizacji (pierwsze 5 dla mediany): {pred_array[0, :5, 1].tolist()}")
    
    # Konwersja Relative Returns na ceny Close
    last_close_price = ticker_data['Close'].iloc[-1]
    logger.info(f"Ostatnia cena Close (znormalizowana): {last_close_price}")
    
    try:
        with open(config['data']['normalizers_path'], 'rb') as f:
            normalizers = pickle.load(f)
        close_normalizer = normalizers.get('Close', target_normalizer)
    except:
        close_normalizer = target_normalizer
    
    last_close_denorm = close_normalizer.inverse_transform(torch.tensor([[last_close_price]]).float())
    last_close_denorm = np.expm1(last_close_denorm.numpy())[0, 0]
    logger.info(f"Ostatnia cena Close (denormalizowana): {last_close_denorm}")
    
    if len(pred_array.shape) == 3:
        relative_returns_median = pred_array[0, :, 1]
        relative_returns_lower = pred_array[0, :, 0]  
        relative_returns_upper = pred_array[0, :, 2]
        
        current_price = last_close_denorm
        median = []
        lower_bound = []
        upper_bound = []
        
        for i in range(len(relative_returns_median)):
            price_median = current_price * (1 + relative_returns_median[i])
            price_lower = current_price * (1 + relative_returns_lower[i])
            price_upper = current_price * (1 + relative_returns_upper[i])
            
            median.append(price_median)
            lower_bound.append(price_lower)
            upper_bound.append(price_upper)
            
            current_price = price_median
        
        median = np.array(median)
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
    else:
        raise ValueError(f"Nieoczekiwany kształt pred_array: {pred_array.shape}")

    logger.info(f"Przewidywane ceny (pierwsze 5 dla mediany): {median[:5].tolist()}")
    return median, lower_bound, upper_bound

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
        line=dict(color='blue')
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
        xaxis=dict(rangeslider=dict(visible=True), type='date')
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

def create_historical_plot(config, ticker_data, original_close, median, ticker, historical_close):
    """Tworzy wykres porównujący predykcje z historią."""
    last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config['model']['max_prediction_length'], freq='D')
    historical_dates = ticker_data['Date'].tolist()
    
    # Tworzenie pełnego zakresu dat dla okresu predykcji
    pred_date_range = pd.DataFrame({'Date': pred_dates})
    pred_date_range['Date'] = pd.to_datetime(pred_date_range['Date'], utc=True)
    
    # Przygotowanie historycznych cen z wypełnieniem brakujących dat
    historical_close = historical_close.reindex(pred_date_range['Date'], method='ffill')
    
    # Logowanie dla debugowania
    logger.info(f"Długość historical_dates: {len(historical_dates)}")
    logger.info(f"Długość pred_dates: {len(pred_dates)}")
    logger.info(f"Długość original_close: {len(original_close)}")
    logger.info(f"Długość historical_close po reindex: {len(historical_close)}")
    logger.info(f"Długość median: {len(median)}")
    
    # Tworzenie tablic dla wykresu
    all_dates = historical_dates + pred_date_range['Date'].tolist()
    all_close = original_close.tolist() + historical_close.tolist()
    all_pred_close = [None] * len(historical_dates) + median.tolist()

    # Sprawdzenie długości tablic
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
        line=dict(color='blue')  # Ciągła niebieska linia
    ))
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Predicted_Close'],
        mode='lines',
        name='Przewidywana cena zamknięcia',
        line=dict(color='orange', dash='dash')  # Przerywana pomarańczowa linia
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
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    """Główna funkcja aplikacji Streamlit."""
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    config = load_config()
    temp_raw_data_path = 'data/temp_stock_data.csv'

    ticker_options = load_tickers_and_names()
    default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")

    ticker_option = st.selectbox(
        "Wybierz spółkę z listy lub wpisz własną:",
        options=["Wpisz ręcznie"] + list(ticker_options.values()),
        index=0 if not default_ticker in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
    )
    ticker_input = st.text_input("Wpisz ticker spółki (np. AAPL, CDR.WA):", value=default_ticker if ticker_option == "Wpisz ręcznie" else [k for k, v in ticker_options.items() if v == ticker_option][0])

    page = st.sidebar.selectbox("Wybierz stronę", ["Predykcje przyszłości", "Porównanie predykcji z historią"])

    if page == "Predykcje przyszłości":
        if st.button("Generuj predykcje"):
            try:
                new_data, dataset, normalizers, model = load_data_and_model(config, ticker_input, temp_raw_data_path)
                ticker_data, original_close = preprocess_data(config, new_data, ticker_input, normalizers)
                median, lower_bound, upper_bound = generate_predictions(config, dataset, model, ticker_data)
                create_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input)
            finally:
                if os.path.exists(temp_raw_data_path):
                    os.remove(temp_raw_data_path)
                    logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

    elif page == "Porównanie predykcji z historią":
        if st.button("Porównaj predykcje z historią"):
            try:
                # Pobierz max_prediction_length z konfiguracji
                max_prediction_length = config['model']['max_prediction_length']
                
                # Ustaw datę początkową na dzisiaj - max_prediction_length - 365 dni
                trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
                start_date = trim_date - pd.Timedelta(days=365)
                
                # Pobierz dane historyczne
                fetcher = DataFetcher(ConfigManager())
                full_data = fetcher.fetch_stock_data(ticker_input, start_date, datetime.now())
                if full_data.empty:
                    raise ValueError("Brak pełnych danych historycznych")
                
                full_data = full_data[full_data['Ticker'] == ticker_input].copy()
                full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
                
                # Przytnij dane do trim_date dla predykcji
                new_data = full_data[full_data['Date'] <= trim_date].copy()
                if new_data.empty:
                    raise ValueError(f"Brak danych przed {trim_date} dla {ticker_input}")
                
                new_data.to_csv(temp_raw_data_path, index=False)
                logger.info(f"Dane dla {ticker_input} zapisane do {temp_raw_data_path}")
                
                # Wczytaj model i dataset
                _, dataset, normalizers, model = load_data_and_model(config, ticker_input, temp_raw_data_path, historical_mode=True)
                
                # Preprocessuj dane
                ticker_data, original_close = preprocess_data(config, new_data, ticker_input, normalizers, historical_mode=True)
                
                # Generuj predykcje
                median, _, _ = generate_predictions(config, dataset, model, ticker_data)
                
                # Przygotuj dane historyczne do porównania
                full_data.set_index('Date', inplace=True)
                historical_close = full_data['Close']
                
                create_historical_plot(config, ticker_data, original_close, median, ticker_input, historical_close)
            finally:
                if os.path.exists(temp_raw_data_path):
                    os.remove(temp_raw_data_path)
                    logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

if __name__ == "__main__":
    main()