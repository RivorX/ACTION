import torch
import pytorch_forecasting
from pytorch_forecasting.data import TimeSeriesDataSet
from .model import build_model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger
import optuna
import yaml
import pandas as pd
import numpy as np
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
    reals = dataset.data['reals'].detach().clone().cpu().numpy()
    groups = dataset.data['groups'].detach().clone().cpu().numpy()
    time_idx = dataset.data['time'].detach().clone().cpu().numpy()

    if groups.ndim > 1:
        groups = groups.squeeze()
        if groups.ndim > 1:
            groups = groups.flatten()

    groups = np.expand_dims(groups, axis=1)
    time_idx = np.expand_dims(time_idx, axis=1)

    expected_reals = ["Open", "High", "Low", "Volume", "MA10", "MA50", "RSI", "Volatility", "Close"]
    real_columns = [col for col in dataset.reals if col in expected_reals]
    columns = real_columns + ['group_id', 'time_idx']

    print("dataset.reals:", dataset.reals)

    df = pd.DataFrame(
        np.concatenate([reals[:, :len(real_columns)], groups, time_idx], axis=1),
        columns=columns
    )
    df['group_id'] = df['group_id'].astype(str)
    df['time_idx'] = df['time_idx'].astype(int)

    if df.empty:
        raise ValueError("DataFrame jest pusty. Sprawdź dane wejściowe lub preprocessing.")

    print("DataFrame info:", df.info())
    print("Kolumny DataFrame:", df.columns.tolist())
    print("time_idx statystyki:", df['time_idx'].describe())

    max_time_idx = df['time_idx'].max()
    if pd.isna(max_time_idx) or not np.isfinite(max_time_idx):
        raise ValueError("max_time_idx jest NaN lub nieokreślony. Sprawdź kolumnę time_idx.")

    split_idx = int(max_time_idx * 0.8)
    print(f"max_time_idx: {max_time_idx}, split_idx: {split_idx}")

    train_df = df[df['time_idx'] <= split_idx]
    val_df = df[df['time_idx'] > split_idx]

    if train_df.empty or val_df.empty:
        raise ValueError(f"Zbiór treningowy (rozmiar: {len(train_df)}) lub walidacyjny (rozmiar: {len(val_df)}) jest pusty. Sprawdź podział danych.")

    train_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), train_df)
    val_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), val_df)

    if use_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        print(f"Najlepsze parametry: {best_params}")
    else:
        best_params = None
        print("Pomijanie optymalizacji Optuna, używanie domyślnych hiperparametrów.")

    final_model = build_model(dataset, config, hyperparams=best_params)

    checkpoint_path = config['paths']['checkpoint_path']
    if os.path.exists(checkpoint_path):
        print(f"Wczytywanie checkpointu z {checkpoint_path}")
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
        print("Brak checkpointu, trenowanie od zera")

    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
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