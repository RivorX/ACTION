import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.backends.cudnn as cudnn
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
import pytorch_forecasting
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.model import build_model
from scripts.config_manager import ConfigManager
from scripts.utils.batch_size_estimator import estimate_batch_size
import optuna
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Wydajnościowe ustawienia CUDA/cuDNN
cudnn.benchmark = True

class CustomModelCheckpoint(pl.callbacks.Callback):
    """Niestandardowy callback do zapisywania checkpointów."""
    
    def __init__(self, monitor: str, save_path: str, mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.save_path = Path(save_path)
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        is_better = (self.mode == "min" and current_score < self.best_score) or (self.mode == "max" and current_score > self.best_score)
        if is_better:
            self.best_score = current_score
            logger.info(f"Zapisywanie checkpointu z {self.monitor}={current_score} w {self.save_path}")
            checkpoint = {
                "state_dict": pl_module.state_dict(),
                "hyperparams": dict(pl_module.hparams)
            }
            torch.save(checkpoint, self.save_path)

def objective(trial, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, config: dict):
    model = build_model(train_dataset, config, trial)
    
    # Estymacja batch size
    batch_size = estimate_batch_size(model, train_dataset, config)
    config['training']['batch_size'] = batch_size
    logger.info(f"Ustawiono batch_size w objective na: {batch_size}")

    num_workers = config['training']['num_workers']
    pin_memory = torch.cuda.is_available()
    prefetch_factor = config['training']['prefetch_factor']
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience']),
            CustomModelCheckpoint(monitor="val_loss", save_path=config['paths']['model_save_path'], mode="min")
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=config['training']['batch_size'], num_workers=num_workers,
        persistent_workers=True, pin_memory=pin_memory, prefetch_factor=prefetch_factor
    )
    for batch in val_dataloader:
        x, y = batch
        for key, val in x.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"Validation batch tensor {key} device: {val.device}")
        logger.info(f"Validation batch: y[0, :5] = {y[0][:5].tolist()}")
        break
    trainer.fit(model, train_dataloaders=train_dataset.to_dataloader(
        train=True, batch_size=config['training']['batch_size'], num_workers=num_workers,
        persistent_workers=True, pin_memory=pin_memory, prefetch_factor=prefetch_factor
    ), val_dataloaders=val_dataloader)
    return trainer.callback_metrics["val_loss"].item()

def train_model(dataset: tuple, config: dict, use_optuna: bool = True, continue_training: bool = False, new_lr: float = None):
    logger.info("Rozpoczynanie treningu modelu...")
    
    # Rozpakuj krotkę dataset na train_dataset i val_dataset
    train_dataset, val_dataset = dataset
    
    # Ładuj przetworzone train_df i val_df
    train_processed_df_path = Path(config['data']['train_processed_df_path']).with_suffix('.parquet')
    val_processed_df_path = Path(config['data']['val_processed_df_path']).with_suffix('.parquet')
    
    if not train_processed_df_path.exists() or not val_processed_df_path.exists():
        raise FileNotFoundError(f"Przetworzone DataFrame train/val nie istnieją w {train_processed_df_path} lub {val_processed_df_path}")
    
    train_df = pd.read_parquet(train_processed_df_path).copy()
    val_df = pd.read_parquet(val_processed_df_path).copy()
    
    logger.info(f"Wczytano przetworzony train DataFrame z {train_processed_df_path}, długość: {len(train_df)}")
    logger.info(f"Wczytano przetworzony val DataFrame z {val_processed_df_path}, długość: {len(val_df)}")
    
    if train_df.empty or val_df.empty:
        raise ValueError("Przetworzone DataFrame train/val są puste")
    
    # Upewnij się, że Sector i Day_of_Week są kategoryczne
    train_df['Sector'] = pd.Categorical(train_df['Sector'], categories=config['model']['sectors'], ordered=False)
    train_df['Day_of_Week'] = pd.Categorical(train_df['Day_of_Week'], categories=[str(i) for i in range(7)], ordered=False)
    val_df['Sector'] = pd.Categorical(val_df['Sector'], categories=config['model']['sectors'], ordered=False)
    val_df['Day_of_Week'] = pd.Categorical(val_df['Day_of_Week'], categories=[str(i) for i in range(7)], ordered=False)
    
    # Filtracja grup z wystarczającą liczbą rekordów
    min_val_records = config['model'].get('min_prediction_length', 1) + config['model'].get('min_encoder_length', 1)
    df = pd.concat([train_df, val_df])
    group_counts = df.groupby('group_id').size().reset_index(name='count')
    valid_groups = group_counts[group_counts['count'] >= min_val_records]['group_id']
    train_df = train_df[train_df['group_id'].isin(valid_groups)]
    val_df = val_df[val_df['group_id'].isin(valid_groups)]
    
    if train_df.empty or val_df.empty:
        raise ValueError(f"Zbiory danych są puste po filtrowaniu: train_df={len(train_df)}, val_df={len(val_df)}")
    
    if len(val_dataset) == 0 or len(train_dataset) == 0:
        raise ValueError(f"Zbiory danych są puste: train_dataset={len(train_dataset)}, val_dataset={len(val_dataset)}")

    # Ustal urządzenie
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Używane urządzenie: {device}")

    # Optymalizacja z Optuna (ustaw use_optuna=False po pierwszym tuningu dla prędkości)
    if use_optuna and not continue_training:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        logger.info(f"Najlepsze parametry: {best_params}")
    else:
        best_params = None
        logger.info("Pomijanie optymalizacji Optuna, używanie domyślnych hiperparametrów.")

    # Wczytywanie modelu
    model_save_path = Path(config['paths']['model_save_path'])
    logger.info(f"Ścieżka do modelu: {model_save_path}, istnieje: {model_save_path.exists()}")
    if continue_training and model_save_path.exists():
        logger.info(f"Wczytywanie modelu z {model_save_path}")
        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'), weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        
        if new_lr is not None:
            hyperparams['learning_rate'] = new_lr
            logger.info(f"Zmieniono learning rate na {new_lr} dla kontynuacji treningu.")
        
        final_model = build_model(train_dataset, config, hyperparams=hyperparams)
        try:
            final_model.load_state_dict(checkpoint["state_dict"])
            final_model.to(device)
            logger.info(f"Model wczytany i przeniesiony na urządzenie: {device}")
            logger.info(f"Model parameters device: {next(final_model.parameters()).device}")
        except RuntimeError as e:
            logger.error(f"Błąd wczytywania state_dict: {e}")
            raise
    else:
        logger.info("Brak modelu lub kontynuacja wyłączona, trenowanie od zera")
        final_model = build_model(train_dataset, config, hyperparams=best_params)

    # Estymacja batch size przed utworzeniem DataLoaderów
    batch_size = estimate_batch_size(final_model, train_dataset, config)
    config['training']['batch_size'] = batch_size  # Zaktualizuj config
    logger.info(f"Ustawiono batch_size na: {batch_size}")

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config['training']['early_stopping_patience']),
            CustomModelCheckpoint(monitor="val_loss", save_path=config['paths']['model_save_path'], mode="min")
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    num_workers = config['training']['num_workers']
    prefetch_factor = config['training']['prefetch_factor']
    pin_memory = torch.cuda.is_available()
    trainer.fit(
        model=final_model,
        train_dataloaders=train_dataset.to_dataloader(
            train=True, batch_size=config['training']['batch_size'], num_workers=num_workers,
            persistent_workers=True, pin_memory=pin_memory, prefetch_factor=prefetch_factor
        ),
        val_dataloaders=val_dataset.to_dataloader(
            train=False, batch_size=config['training']['batch_size'], num_workers=num_workers,
            persistent_workers=True, pin_memory=pin_memory, prefetch_factor=prefetch_factor
        )
    )
    
    # Zapisz model
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "hyperparams": dict(final_model.hparams)
    }
    torch.save(checkpoint, model_save_path)
    logger.info(f"Model zapisany w: {model_save_path}")
    return final_model