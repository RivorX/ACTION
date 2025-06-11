import pandas as pd
import numpy as np
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
import yaml
import torch

def add_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_dataset(df, config):
    if df.empty:
        raise ValueError("DataFrame jest pusty. Sprawdź dane wejściowe.")
    
    # Czyszczenie danych
    df = df.dropna(subset=['Close'])  # Usuń wiersze z brakującymi wartościami w Close
    if (df['Close'] <= 0).any():
        print("Znaleziono ujemne wartości w Close. Zamieniam na NaN i usuwam.")
        df.loc[df['Close'] <= 0, 'Close'] = np.nan
        df = df.dropna(subset=['Close'])
    
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
    df['group_id'] = df['Ticker']
    
    # Transformacja logarytmiczna tylko dla dodatnich wartości
    df['Close'] = np.log1p(df['Close'].clip(lower=0))  # Clip usuwa ujemne wartości
    
    normalizer = TorchNormalizer()
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="Close",
        group_ids=["group_id"],
        min_encoder_length=config['model']['min_encoder_length'],
        max_encoder_length=int(df['time_idx'].max()),
        max_prediction_length=config['model']['max_prediction_length'],
        time_varying_known_reals=["time_idx", "Open", "High", "Low", "Volume", "MA10", "MA50", "RSI", "Volatility"],
        time_varying_unknown_reals=["Close"],
        target_normalizer=normalizer,
        allow_missing_timesteps=True,
        add_encoder_length=False
    )
    
    print("dataset.reals po uzyskaniu:", dataset.reals)
    dataset.save(config['data']['processed_data_path'])
    return dataset

def preprocess_data(config):
    df = pd.read_csv(config['data']['raw_data_path'])
    df = add_features(df)
    dataset = create_dataset(df, config)
    return dataset

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    preprocess_data(config)