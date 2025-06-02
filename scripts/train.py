import torch
import pytorch_forecasting
from pytorch_forecasting.data import TimeSeriesDataSet
from .model import build_model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import optuna
import yaml
import pandas as pd
import numpy as np
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, train_dataset, val_dataset, config):
    model = build_model(train_dataset, config, trial)
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience'])],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    # Debugowanie: Sprawdź urządzenia batchy w dataloaderze
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
    )
    for batch in val_dataloader:
        x, y = batch
        for key, val in x.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"Validation batch tensor {key} device: {val.device}")
        break  # Sprawdź tylko pierwszy batch
    trainer.fit(
        model,
        train_dataloaders=train_dataset.to_dataloader(
            train=True, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
        ),
        val_dataloaders=val_dataloader
    )
    return trainer.callback_metrics["val_loss"].item()

def train_model(dataset, config):
    # Pobierz oryginalne dane z datasetu i przekształć tensory na numpy
    reals = dataset.data['reals'].detach().clone().cpu().numpy()  # Przenieś na CPU dla preprocessing
    groups = dataset.data['groups'].detach().clone().cpu().numpy()
    time_idx = dataset.data['time'].detach().clone().cpu().numpy()

    # Spłaszcz groups do 1D, jeśli ma więcej wymiarów
    if groups.ndim > 1:
        groups = groups.squeeze()
        if groups.ndim > 1:
            groups = groups.flatten()

    # Przekształć groups i time_idx na 2D, aby pasowały do reals
    groups = np.expand_dims(groups, axis=1)
    time_idx = np.expand_dims(time_idx, axis=1)

    # Filtruj kolumny reals, aby wykluczyć time_idx i encoder_length
    expected_reals = ["Open", "High", "Low", "Volume", "MA10", "MA50", "RSI", "Volatility", "Close"]
    real_columns = [col for col in dataset.reals if col in expected_reals]
    columns = real_columns + ['group_id', 'time_idx']

    # Debugowanie: Wyświetl dataset.reals
    print("dataset.reals:", dataset.reals)

    # Przywróć oryginalny format danych jako DataFrame
    df = pd.DataFrame(
        np.concatenate([reals[:, :len(real_columns)], groups, time_idx], axis=1),
        columns=columns
    )
    df['group_id'] = df['group_id'].astype(str)
    df['time_idx'] = df['time_idx'].astype(int)

    # Sprawdź, czy DataFrame nie jest pusty
    if df.empty:
        raise ValueError("DataFrame jest pusty. Sprawdź dane wejściowe lub preprocessing.")

    # Debugowanie: Wyświetl informacje o DataFrame
    print("DataFrame info:", df.info())
    print("Kolumny DataFrame:", df.columns.tolist())
    print("time_idx unikalne wartości:", df['time_idx'].unique())
    print("time_idx statystyki:", df['time_idx'].describe())

    # Podział na zbiór treningowy i walidacyjny (80% trening, 20% walidacja)
    max_time_idx = df['time_idx'].max()
    if pd.isna(max_time_idx) or not np.isfinite(max_time_idx):
        raise ValueError("max_time_idx jest NaN lub nieokreślony. Sprawdź kolumnę time_idx.")

    split_idx = int(max_time_idx * 0.8)
    print(f"max_time_idx: {max_time_idx}, split_idx: {split_idx}")

    train_df = df[df['time_idx'] <= split_idx]
    val_df = df[df['time_idx'] > split_idx]

    # Sprawdź, czy zbiory treningowy i walidacyjny nie są puste
    if train_df.empty or val_df.empty:
        raise ValueError(f"Zbiór treningowy (rozmiar: {len(train_df)}) lub walidacyjny (rozmiar: {len(val_df)}) jest pusty. Sprawdź podział danych.")

    # Stwórz nowe TimeSeriesDataSet dla treningu i walidacji
    train_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), train_df)
    val_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), val_df)

    # Optymalizacja hiperparametrów z Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])

    # Trenuj finalny model z najlepszymi parametrami
    best_params = study.best_params
    print(f"Najlepsze parametry: {best_params}")
    final_model = build_model(dataset, config)
    final_model.hparams.update(best_params)

    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience'])],
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
    final_model.save(config['paths']['model_save_path'])
    return final_model

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = TimeSeriesDataSet.load(config['data']['processed_data_path'])
    train_model(dataset, config)