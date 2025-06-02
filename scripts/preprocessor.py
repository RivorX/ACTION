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
    # Konwersja kolumny Date na datetime z ujednoliconą strefą czasową (UTC)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Obliczanie time_idx na podstawie różnicy dni i rzutowanie na int
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
    
    df['group_id'] = df['Ticker']

    # Debugowanie: Wyświetl kolumny przed utworzeniem datasetu
    print("Kolumny df przed utworzeniem datasetu:", df)
    
    # Upewnij się, że df nie zawiera encoder_length
    if 'encoder_length' in df.columns:
        df = df.drop(columns=['encoder_length'])
        print("Usunięto kolumnę encoder_length z df")

    # Tworzenie normalizatora
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

    # Debugowanie: Wyświetl dataset.reals
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