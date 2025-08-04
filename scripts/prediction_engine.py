import logging
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from pytorch_forecasting import TimeSeriesDataSet
from scripts.data_fetcher import DataFetcher
from scripts.model import build_model
from scripts.utils.config_manager import ConfigManager
from scripts.utils.preprocessing_utils import PreprocessingUtils
import asyncio
import aiohttp
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def load_data_and_model_async(config, ticker, temp_raw_data_path, historical_mode=False, trim_days=0, years=3):
    """Asynchroniczna wersja load_data_and_model."""
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Używane urządzenie: {device}")

    async with aiohttp.ClientSession() as session:
        fetcher = DataFetcher(ConfigManager(), years)
        start_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=years * 365 + trim_days)
        fetch_time = time.time()
        new_data = await fetcher.fetch_stock_data(ticker, start_date, datetime.now(), session)
        fetch_duration = time.time() - fetch_time
        logger.info(f"Pobieranie danych dla {ticker} zajęło: {fetch_duration:.3f} sekundy")
        if new_data.empty:
            logger.error(f"Nie udało się pobrać danych dla {ticker}")
            raise ValueError("Brak danych")

        new_data.to_csv(temp_raw_data_path, index=False)
        logger.info(f"Dane dla {ticker} zapisane do {temp_raw_data_path}, długość: {len(new_data)}")

    try:
        dataset_load_time = time.time()
        dataset = torch.load(config['data']['processed_data_path'], weights_only=False, map_location=device)
        dataset_load_duration = time.time() - dataset_load_time
        logger.info(f"Wczytywanie datasetu zajęło: {dataset_load_duration:.3f} sekundy")
    except Exception as e:
        logger.error(f"Błąd wczytywania datasetu: {e}")
        raise

    preprocessing_utils = PreprocessingUtils(config)
    normalizers = preprocessing_utils.load_normalizers()

    # Sprawdź, czy ticker ma swój normalizer, jeśli nie, użyj globalnego
    ticker_normalizers = normalizers.get(ticker, normalizers.get('global', {}))
    relative_returns_normalizer = ticker_normalizers.get('Relative_Returns', dataset.target_normalizer)

    try:
        model_load_time = time.time()
        model_name = config['model_name']
        model_path = os.path.join(config['paths']['models_dir'], f"{model_name}.pth")
        normalizers_path = os.path.join(config['paths']['models_dir'], 'normalizers', f"{model_name}_normalizers.pkl")
        config['data']['normalizers_path'] = normalizers_path
        if not os.path.exists(model_path):
            logger.error(f"Plik modelu {model_path} nie istnieje.")
            raise FileNotFoundError(f"Plik modelu {model_path} nie istnieje.")
        if not os.path.exists(normalizers_path):
            logger.error(f"Plik normalizerów {normalizers_path} nie istnieje.")
            raise FileNotFoundError(f"Plik normalizerów {normalizers_path} nie istnieje.")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if 'hidden_continuous_size' not in hyperparams:
            hyperparams['hidden_continuous_size'] = config['model']['hidden_size'] // 2
        model = build_model(dataset, config, hyperparams=hyperparams)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model_load_duration = time.time() - model_load_time
        logger.info(f"Wczytywanie modelu zajęło: {model_load_duration:.3f} sekundy")
        logger.info(f"Model {model_name} wczytany z {model_path} i przeniesiony na urządzenie: {device}")
        logger.info(f"Urządzenie parametrów modelu: {next(model.parameters()).device}")
    except Exception as e:
        logger.error(f"Błąd wczytywania modelu: {e}")
        raise

    total_duration = time.time() - start_time
    logger.info(f"Całkowity czas load_data_and_model_async: {total_duration:.3f} sekundy")
    return new_data, dataset, ticker_normalizers, model

def load_data_and_model(config, ticker, temp_raw_data_path, historical_mode=False, trim_days=0, years=3):
    """Synchroniczna wersja wywołująca asynchroniczną."""
    start_time = time.time()
    result = asyncio.get_event_loop().run_until_complete(
        load_data_and_model_async(config, ticker, temp_raw_data_path, historical_mode, trim_days, years)
    )
    total_duration = time.time() - start_time
    logger.info(f"Całkowity czas load_data_and_model: {total_duration:.3f} sekundy")
    return result

def preprocess_data(config, ticker_data, ticker, normalizers, historical_mode=False, trim_days=0):
    start_time = time.time()
    preprocessing_utils = PreprocessingUtils(config)
    ticker_data, original_close = preprocessing_utils.preprocess_dataframe(ticker_data, ticker, historical_mode, trim_days)
    ticker_data = ticker_data.reset_index(drop=True)
    original_close = original_close.reindex(ticker_data.index).fillna(0)
    total_duration = time.time() - start_time
    logger.info(f"Całkowity czas preprocess_data: {total_duration:.3f} sekundy")
    return ticker_data, original_close

def generate_predictions(config, dataset, model, ticker_data):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Generowanie predykcji na urządzeniu: {device}")
    logger.info(f"Urządzenie parametrów modelu: {next(model.parameters()).device}")
    
    model = model.to(device)
    preprocessing_utils = PreprocessingUtils(config)
    
    dataset_creation_time = time.time()
    ticker_dataset = preprocessing_utils.create_dataset(ticker_data, dataset.get_parameters(), predict_mode=True)
    batch_size = config['prediction']['batch_size']
    dataloader = ticker_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    dataset_creation_duration = time.time() - dataset_creation_time
    logger.info(f"Tworzenie TimeSeriesDataSet i dataloadera zajęło: {dataset_creation_duration:.3f} sekundy")
    
    prediction_time = time.time()
    with torch.inference_mode(), torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32):
        predictions = model.predict(dataloader, mode="quantiles", return_x=True)
    prediction_duration = time.time() - prediction_time
    logger.info(f"Wykonywanie predykcji zajęło: {prediction_duration:.3f} sekundy")
    logger.info(f"Kształt predictions.output: {predictions.output.shape}")
    
    transfer_time = time.time()
    pred_array = predictions.output.to('cpu').numpy()
    transfer_duration = time.time() - transfer_time
    logger.info(f"Transfer predykcji na CPU zajął: {transfer_duration:.3f} sekundy")
    
    denorm_time = time.time()
    target_normalizer = dataset.target_normalizer
    # Zapewnij, że pred_array jest tensorem przed denormalizacją
    if not isinstance(pred_array, torch.Tensor):
        pred_array = torch.from_numpy(pred_array)
    
    # Denormalizacja przy użyciu target_normalizer
    pred_array = target_normalizer.inverse_transform(pred_array)
    
    # Zapewnij, że pred_array jest tablicą NumPy po denormalizacji
    if isinstance(pred_array, torch.Tensor):
        pred_array = pred_array.numpy()
    
    # Wczytaj normalizery i wybierz odpowiedni dla 'Close'
    normalizers = preprocessing_utils.load_normalizers()
    ticker = ticker_data['Ticker'].iloc[0]
    close_normalizer = normalizers.get(ticker, normalizers.get('global', {})).get('Close', None)
    if close_normalizer is None:
        logger.warning(f"Brak normalizera dla 'Close' dla tickera {ticker}, używam globalnego normalizera")
        close_normalizer = normalizers.get('global', {}).get('Close', None)
        if close_normalizer is None:
            logger.warning(f"Brak globalnego normalizera dla 'Close', brak denormalizacji dla Close")
            close_normalizer = None
    
    # Denormalizacja ostatniej ceny zamknięcia
    last_close_price = ticker_data['Close'].iloc[-1]
    
    # Zapewnij, że last_close_price jest numpy float, nie tensorem
    if isinstance(last_close_price, torch.Tensor):
        last_close_price = last_close_price.cpu().numpy()
    if isinstance(last_close_price, np.ndarray):
        last_close_price = float(last_close_price)
    else:
        last_close_price = float(last_close_price)
    
    if close_normalizer is not None:
        # RobustScaler oczekuje numpy array w kształcie (n_samples, n_features)
        last_close_denorm = close_normalizer.inverse_transform(np.array([[last_close_price]]))[0, 0]
    else:
        last_close_denorm = last_close_price  # Bez normalizera, użyj surowej wartości
    if 'Close' in preprocessing_utils.log_features:
        last_close_denorm = np.expm1(last_close_denorm)
    
    # Zapewnij, że last_close_denorm jest liczbą typu float, nie tensorem
    if isinstance(last_close_denorm, torch.Tensor):
        last_close_denorm = last_close_denorm.item()
    elif isinstance(last_close_denorm, np.ndarray):
        last_close_denorm = float(last_close_denorm)
    else:
        last_close_denorm = float(last_close_denorm)
    
    denorm_duration = time.time() - denorm_time
    logger.info(f"Denormalizacja wyników zajęła: {denorm_duration:.3f} sekundy")
    logger.info(f"Denormalizowana ostatnia cena zamknięcia: {last_close_denorm}")
    
    price_calc_time = time.time()
    if len(pred_array.shape) == 3:
        relative_returns_median = pred_array[0, :, 1]
        relative_returns_lower = pred_array[0, :, 0]
        relative_returns_upper = pred_array[0, :, 2]
        
        # Zapewnij, że relative_returns są tablicami NumPy, nie tensorami
        if isinstance(relative_returns_median, torch.Tensor):
            relative_returns_median = relative_returns_median.cpu().numpy()
        if isinstance(relative_returns_lower, torch.Tensor):
            relative_returns_lower = relative_returns_lower.cpu().numpy()
        if isinstance(relative_returns_upper, torch.Tensor):
            relative_returns_upper = relative_returns_upper.cpu().numpy()
            
        # Dodatkowo upewnij się, że są to rzeczywiście numpy arrays
        relative_returns_median = np.asarray(relative_returns_median)
        relative_returns_lower = np.asarray(relative_returns_lower)
        relative_returns_upper = np.asarray(relative_returns_upper)
        
        logger.info(f"Pierwsze 5 Relative_Returns (mediana): {relative_returns_median[:5].tolist()}")
        
        current_price = float(last_close_denorm)  # Zapewnij, że current_price jest typu float
        median = []
        lower_bound = []
        upper_bound = []
        
        for i in range(len(relative_returns_median)):
            # Upewnij się że wszystkie wartości są float przed operacjami
            rel_ret_median = float(relative_returns_median[i])
            rel_ret_lower = float(relative_returns_lower[i])
            rel_ret_upper = float(relative_returns_upper[i])
            
            price_median = current_price * (1 + rel_ret_median)
            price_lower = current_price * (1 + rel_ret_lower)
            price_upper = current_price * (1 + rel_ret_upper)
            
            median.append(price_median)
            lower_bound.append(price_lower)
            upper_bound.append(price_upper)
            
            current_price = price_median
        
        median = np.array(median)
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
    else:
        raise ValueError(f"Nieoczekiwany kształt pred_array: {pred_array.shape}")
    
    price_calc_duration = time.time() - price_calc_time
    logger.info(f"Obliczanie cen zajęło: {price_calc_duration:.3f} sekundy")
    
    logger.info(f"Przewidywane ceny (pierwsze 5 dla mediany): {median[:5].tolist()}")
    total_duration = time.time() - start_time
    logger.info(f"Całkowity czas generate_predictions: {total_duration:.3f} sekundy")
    return median, lower_bound, upper_bound