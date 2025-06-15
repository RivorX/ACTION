import torch
import pytorch_forecasting
from pytorch_forecasting.data import TimeSeriesDataSet
from .model import build_model
from .preprocessor import add_features
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger
import optuna
import yaml
import pandas as pd
import numpy as np
import pickle
import logging
import os

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomModelCheckpoint(Callback):
    def __init__(self, monitor, save_path, mode="min"):
        super().__init__()
        self.monitor = monitor
        self.save_path = save_path
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        if (self.mode == "min" and current_score < self.best_score) or (self.mode == "max" and current_score > self.best_score):
            self.best_score = current_score
            logger.info(f"Zapisywanie checkpointu z {self.monitor}={current_score} w {self.save_path}")
            # Zapisz state_dict i hiperparametry
            hyperparams = dict(pl_module.hparams)
            # Serializuj loss jako string
            if 'loss' in hyperparams:
                hyperparams['loss'] = str(hyperparams['loss'])
            checkpoint = {
                "state_dict": pl_module.state_dict(),
                "hyperparams": hyperparams
            }
            torch.save(checkpoint, self.save_path)

def objective(trial, train_dataset, val_dataset, config):
    model = build_model(train_dataset, config, trial)
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience']),
            CustomModelCheckpoint(monitor="val_loss", save_path=config['paths']['checkpoint_path'], mode="min")
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
    )
    for batch in val_dataloader:
        x, y = batch
        for key, val in x.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"Validation batch tensor {key} device: {val.device}")
        logger.info(f"Validation batch: y_hat[0, :5] = {y[0][:5].tolist()}, y_target[0, :5] = {y[1][:5].tolist()}")
        break
    trainer.fit(
        model,
        train_dataloaders=train_dataset.to_dataloader(
            train=True, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
        ),
        val_dataloaders=val_dataloader
    )
    return trainer.callback_metrics["val_loss"].item()

def train_model(dataset, config, use_optuna=True):
    # Wczytaj surowe dane z CSV
    df = pd.read_csv(config['data']['raw_data_path'])
    logger.info(f"Kolumny w raw_data: {df.columns.tolist()}")
    
    # Zastosuj te same transformacje, co w preprocessor.py
    df = add_features(df)
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])]
    
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
    df['group_id'] = df['Ticker']
    
    # Wczytaj normalizery z pliku
    with open(config['data']['normalizers_path'], 'rb') as f:
        normalizers = pickle.load(f)
    logger.info(f"Wczytano normalizery z: {config['data']['normalizers_path']}")
    
    # Transformacja logarytmiczna dla dodatnich cech
    log_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", 
        "BB_Middle", "BB_Upper", "BB_Lower", "ATR"
    ]
    for feature in log_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature].clip(lower=0))
    
    # Normalizacja wszystkich cech numerycznych
    numeric_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
        "MACD", "MACD_Signal", "BB_Middle", "BB_Upper", "BB_Lower", "Stochastic_K",
        "Stochastic_D", "ATR", "OBV", "Price_Change"
    ]
    for feature in numeric_features:
        if feature in df.columns and feature in normalizers:
            df[feature] = normalizers[feature].transform(df[feature].values)
    
    # Konwersja kolumn numerycznych na float
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # Upewnij się, że cechy kategoryczne są stringami
    categorical_columns = ['Day_of_Week', 'Month']
    for cat_col in categorical_columns:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)

    if df.empty:
        raise ValueError("DataFrame jest pusty. Sprawdź dane wejściowe lub preprocessing.")

    # Sprawdź liczbę rekordów dla każdej grupy
    group_counts = df.groupby('group_id').size().reset_index(name='count')
    logger.info(f"Liczba rekordów dla każdej grupy:\n{group_counts.to_string()}")
    
    # Określ minimalną liczbę rekordów w zbiorze walidacyjnym
    min_val_records = config['model'].get('min_prediction_length', 1) + config['model'].get('min_encoder_length', 1)
    max_time_idx = df['time_idx'].max()
    split_idx = int(max_time_idx * 0.8)
    
    # Podziel dane na zbiór treningowy i walidacyjny
    train_df = df[df['time_idx'] <= split_idx]
    val_df = df[df['time_idx'] > split_idx]
    
    # Sprawdź liczbę rekordów dla każdej grupy w zbiorach
    train_group_counts = train_df.groupby('group_id').size().reset_index(name='train_count')
    val_group_counts = val_df.groupby('group_id').size().reset_index(name='val_count')
    
    # Filtruj grupy, które mają wystarczającą liczbę rekordów w zbiorze walidacyjnym
    valid_groups = val_group_counts[val_group_counts['val_count'] >= min_val_records]['group_id']
    logger.info(f"Grupy z wystarczającą liczbą rekordów w zbiorze walidacyjnym ({len(valid_groups)} grup): {valid_groups.tolist()}")
    
    # Odfiltruj df, train_df i val_df, aby zawierały tylko ważne grupy
    df = df[df['group_id'].isin(valid_groups)]
    train_df = train_df[train_df['group_id'].isin(valid_groups)]
    val_df = val_df[val_df['group_id'].isin(valid_groups)]
    
    if df.empty:
        raise ValueError("DataFrame po filtrowaniu grup jest pusty. Sprawdź dane wejściowe.")
    if train_df.empty or val_df.empty:
        raise ValueError(f"Zbiór treningowy (rozmiar: {len(train_df)}) lub walidacyjny (rozmiar: {len(val_df)}) jest pusty po filtrowaniu grup.")

    logger.info(f"DataFrame info:\n{df.info()}")
    logger.info(f"Kolumny DataFrame: {df.columns.tolist()}")
    logger.info(f"Pierwsze 5 wierszy DataFrame:\n{df.head().to_string()}")
    logger.info(f"time_idx statystyki: {df['time_idx'].describe()}")
    logger.info(f"max_time_idx: {max_time_idx}, split_idx: {split_idx}")

    # Twórz nowe TimeSeriesDataSet
    train_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), train_df)
    val_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), val_df)

    if use_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        logger.info(f"Najlepsze parametry: {best_params}")
    else:
        best_params = None
        logger.info("Pomijanie optymalizacji Optuna, używanie domyślnych hiperparametrów.")

    final_model = build_model(dataset, config, hyperparams=best_params)

    checkpoint_path = config['paths']['checkpoint_path']
    if os.path.exists(checkpoint_path):
        logger.info(f"Wczytywanie checkpointu z {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if 'loss' in hyperparams and isinstance(hyperparams['loss'], str):
            if 'QuantileLoss' in hyperparams['loss']:
                hyperparams['loss'] = pytorch_forecasting.metrics.QuantileLoss(quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9]))
            else:
                hyperparams['loss'] = pytorch_forecasting.metrics.MAE()
        final_model = build_model(dataset, config, hyperparams=hyperparams)
        try:
            final_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as e:
            logger.error(f"Błąd wczytywania state_dict: {e}")
            raise
        final_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        logger.info("Brak checkpointu, trenowanie od zera")

    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience']),
            CustomModelCheckpoint(monitor="val_loss", save_path=config['paths']['checkpoint_path'], mode="min")
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    trainer.fit(
        model=final_model,
        train_dataloaders=train_dataset.to_dataloader(
            train=True, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
        ),
        val_dataloaders=val_dataset.to_dataloader(
            train=False, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
        )
    )
    # Zapisz cały model jako state_dict z hiperparametrami
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "hyperparams": dict(final_model.hparams)
    }
    if 'loss' in checkpoint['hyperparams']:
        checkpoint['hyperparams']['loss'] = str(checkpoint['hyperparams']['loss'])
    torch.save(checkpoint, config['paths']['model_save_path'])
    return final_model

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = torch.load(config['data']['processed_data_path'], weights_only=False)
    train_model(dataset, config, use_optuna=True)