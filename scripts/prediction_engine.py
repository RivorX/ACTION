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

    relative_returns_normalizer_params = normalizers.get('Relative_Returns', None)
    target_normalizer_params = dataset.target_normalizer.get_parameters()

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
        model_load_time = time.time()
        model_name = config['model_name']
        model_path = os.path.join(config['paths']['models_dir'], f"{model_name}.pth")
        if not os.path.exists(model_path):
            logger.error(f"Plik modelu {model_path} nie istnieje.")
            raise FileNotFoundError(f"Plik modelu {model_path} nie istnieje.")

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
    return new_data, dataset, normalizers, model

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
    pred_array = predictions.output.to('cpu')
    transfer_duration = time.time() - transfer_time
    logger.info(f"Transfer predykcji na CPU zajął: {transfer_duration:.3f} sekundy")
    
    denorm_time = time.time()
    target_normalizer = dataset.target_normalizer
    pred_array = target_normalizer.inverse_transform(pred_array)
    last_close_price = ticker_data['Close'].iloc[-1]
    
    normalizers = preprocessing_utils.load_normalizers()
    close_normalizer = normalizers.get('Close', target_normalizer)
    
    last_close_denorm = close_normalizer.inverse_transform(torch.tensor([[last_close_price]]).float())
    last_close_denorm = np.expm1(last_close_denorm.numpy())[0, 0]
    denorm_duration = time.time() - denorm_time
    logger.info(f"Denormalizacja wyników zajęła: {denorm_duration:.3f} sekundy")
    
    price_calc_time = time.time()
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
    price_calc_duration = time.time() - price_calc_time
    logger.info(f"Obliczanie cen zajęło: {price_calc_duration:.3f} sekundy")
    
    logger.info(f"Przewidywane ceny (pierwsze 5 dla mediany): {median[:5].tolist()}")
    total_duration = time.time() - start_time
    logger.info(f"Całkowity czas generate_predictions: {total_duration:.3f} sekundy")
    return median, lower_bound, upper_bound