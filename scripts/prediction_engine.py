import logging
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_forecasting.data.encoders
import pickle
from pathlib import Path
from scripts.data_fetcher import DataFetcher
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
from scripts.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lista wszystkich możliwych sektorów
ALL_SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 'Consumer Staples',
    'Energy', 'Utilities', 'Industrials', 'Materials', 'Communication Services',
    'Real Estate', 'Unknown'
]

def load_data_and_model(config, ticker, temp_raw_data_path, historical_mode=False, trim_days=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Używane urządzenie: {device}")

    fetcher = DataFetcher(ConfigManager())
    start_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=730 + trim_days)
    new_data = fetcher.fetch_stock_data(ticker, start_date, datetime.now())
    if new_data.empty:
        logger.error(f"Nie udało się pobrać danych dla {ticker}")
        raise ValueError("Brak danych")

    new_data.to_csv(temp_raw_data_path, index=False)
    logger.info(f"Dane dla {ticker} zapisane do {temp_raw_data_path}, długość: {len(new_data)}")

    try:
        dataset = torch.load(config['data']['processed_data_path'], weights_only=False, map_location=device)
        logger.info("Dataset wczytany poprawnie.")
    except Exception as e:
        logger.error(f"Błąd wczytywania datasetu: {e}")
        raise

    try:
        with open(config['data']['normalizers_path'], 'rb') as f:
            normalizers = pickle.load(f)
        logger.info(f"Wczytano normalizery z: {config['data']['normalizers_path']}")
    except Exception as e:
        logger.error(f"Błąd wczytywania normalizerów: {e}")
        raise

    # Pobierz parametry normalizera dla Relative_Returns
    relative_returns_normalizer_params = normalizers.get('Relative_Returns', None)
    target_normalizer_params = dataset.target_normalizer.get_parameters()
    logger.info(f"Parametry normalizera dla Relative_Returns (normalizers.pkl): {relative_returns_normalizer_params.get_parameters() if relative_returns_normalizer_params else 'Brak'}")
    logger.info(f"Parametry normalizera dla target (dataset.target_normalizer): {target_normalizer_params}")

    # Sprawdź zgodność normalizerów tylko jeśli relative_returns_normalizer_params istnieje
    if relative_returns_normalizer_params is not None:
        try:
            relative_returns_params_tensor = relative_returns_normalizer_params.get_parameters()
            if not torch.allclose(relative_returns_params_tensor, target_normalizer_params, rtol=1e-5, atol=1e-8):
                logger.warning("Normalizery dla Relative_Returns różnią się! Może to powodować błędy w predykcjach.")
            else:
                logger.info("Normalizery dla Relative_Returns są zgodne.")
        except Exception as e:
            logger.warning(f"Nie można porównać normalizerów dla Relative_Returns: {e}")
    else:
        logger.warning("Brak normalizera dla Relative_Returns w normalizers.pkl, pomijam porównanie.")

    try:
        checkpoint = torch.load(config['paths']['checkpoint_path'], map_location=device, weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if 'hidden_continuous_size' not in hyperparams:
            hyperparams['hidden_continuous_size'] = config['model']['hidden_size'] // 2
        model = build_model(dataset, config, hyperparams=hyperparams)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        logger.info(f"Model wczytany i przeniesiony na urządzenie: {device}")
        logger.info(f"Model parameters device: {next(model.parameters()).device if model.parameters() else 'No parameters'}")
    except Exception as e:
        logger.error(f"Błąd wczytywania modelu: {e}")
        raise

    return new_data, dataset, normalizers, model

def preprocess_data(config, ticker_data, ticker, normalizers, historical_mode=False, trim_days=0):
    ticker_data = ticker_data[ticker_data['Ticker'] == ticker].copy().reset_index(drop=True)
    logger.info(f"Długość ticker_data po filtrowaniu tickera: {len(ticker_data)}")
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date'], utc=True)

    # Zapisz original_close przed preprocessingiem
    original_close = ticker_data['Close'].copy()
    
    if historical_mode and trim_days > 0:
        ticker_data = ticker_data.iloc[:-trim_days].copy()
        original_close = original_close.iloc[:-trim_days].copy()
    
    preprocessor = DataPreprocessor(config)
    ticker_data = preprocessor.feature_engineer.add_features(ticker_data)
    logger.info(f"Długość ticker_data po dodaniu cech: {len(ticker_data)}")
    ticker_data = ticker_data.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    logger.info(f"Długość ticker_data po dropna: {len(ticker_data)}")
    ticker_data = ticker_data[(ticker_data['Close'] > 0) & (ticker_data['High'] >= ticker_data['Low'])]
    logger.info(f"Długość ticker_data po warunku: {len(ticker_data)}")
    
    # Przytnij original_close do tej samej długości co ticker_data po dropna
    original_close = original_close.loc[ticker_data.index].copy()
    logger.info(f"Długość original_close po przycięciu: {len(original_close)}")
    
    ticker_data['time_idx'] = range(len(ticker_data))
    ticker_data['group_id'] = ticker
    
    # Upewnij się, że Day_of_Week ma dokładnie 7 kategorii (0-6)
    ticker_data['Day_of_Week'] = ticker_data['Date'].dt.dayofweek.astype(str)
    # Wypełnianie NaN w Day_of_Week przed przekształceniem na kategoryczne
    if ticker_data['Day_of_Week'].isna().any():
        logger.warning(f"Znaleziono NaN w Day_of_Week, wypełniam wartością '0'")
        ticker_data['Day_of_Week'] = ticker_data['Day_of_Week'].fillna('0')
    ticker_data['Day_of_Week'] = pd.Categorical(ticker_data['Day_of_Week'], 
                                               categories=[str(i) for i in range(7)], 
                                               ordered=False)
    
    ticker_data['Sector'] = pd.Categorical(ticker_data['Sector'], categories=ALL_SECTORS, ordered=False)
    
    log_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR", "BB_width",
        "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "VWAP"
    ]
    for feature in log_features:
        if feature in ticker_data.columns:
            ticker_data[feature] = np.log1p(ticker_data[feature].clip(lower=0))

    numeric_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI",
        "MACD", "MACD_Signal", "MACD_Histogram", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
        "ADX", "CCI", "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "ROC", "VWAP",
        "Momentum_20d", "Close_to_MA_ratio", "BB_width", "Close_to_BB_upper", "Close_to_BB_lower",
        "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility"
    ]
    for feature in numeric_features:
        if feature in ticker_data.columns and feature in normalizers:
            ticker_data[feature] = normalizers[feature].transform(ticker_data[feature].values)

    categorical_columns = ['Day_of_Week', 'Month', 'Sector']
    for cat_col in categorical_columns:
        if cat_col in ticker_data.columns:
            ticker_data[cat_col] = ticker_data[cat_col].astype(str)

    logger.info(f"Kolumny ticker_data: {ticker_data.columns.tolist()}")
    return ticker_data, original_close

def generate_predictions(config, dataset, model, ticker_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Generowanie predykcji na urządzeniu: {device}")
    model = model.to(device)
    
    # Upewnienie się, że kolumny kategoryczne są stringami
    categorical_columns = ['Day_of_Week', 'Month', 'Sector']
    for cat_col in categorical_columns:
        if cat_col in ticker_data.columns:
            ticker_data[cat_col] = ticker_data[cat_col].astype(str)
            logger.info(f"Upewniono się że {cat_col} jest stringiem: {ticker_data[cat_col].dtype}")
    
    ticker_dataset = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(),
        ticker_data,
        predict_mode=True,
        max_prediction_length=config['model']['max_prediction_length'],
        categorical_encoders={
            'Sector': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=False),
            'Day_of_Week': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=False),
            'Month': pytorch_forecasting.data.encoders.NaNLabelEncoder(add_nan=False)
        }
    ).to_dataloader(train=False, batch_size=128, num_workers=4)

    # Użycie float16 w autocast
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32):
        predictions = model.predict(ticker_dataset, mode="quantiles", return_x=True)
    logger.info(f"Kształt predictions.output: {predictions.output.shape}")

    pred_array = predictions.output.to('cpu')
    target_normalizer = dataset.target_normalizer
    logger.info(f"Predykcje Relative Returns przed denormalizacją (pierwsze 5 dla mediany): {pred_array[0, :5, 1].tolist()}")
    
    pred_array = target_normalizer.inverse_transform(pred_array)
    logger.info(f"Predykcje Relative Returns po denormalizacji (pierwsze 5 dla mediany): {pred_array[0, :5, 1].tolist()}")
    
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