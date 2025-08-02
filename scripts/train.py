import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
from pytorch_forecasting import TimeSeriesDataSet
from scripts.model import build_model
from scripts.utils.config_manager import ConfigManager
from scripts.utils.preprocessing_utils import PreprocessingUtils
import optuna
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_combined_metric", patience=config['training']['early_stopping_patience'], mode="min"),
            CustomModelCheckpoint(monitor="val_combined_metric", save_path=config['paths']['model_save_path'], mode="min")
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False, batch_size=config['training']['batch_size'], num_workers=6, persistent_workers=True, pin_memory=True
    )
    for batch in val_dataloader:
        x, y = batch
        for key, val in x.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"Validation batch tensor {key} device: {val.device}")
        logger.info(f"Validation batch: y[0, :5] = {y[0][:5].tolist()}")
        break
    trainer.fit(model, train_dataloaders=train_dataset.to_dataloader(
        train=True, batch_size=config['training']['batch_size'], num_workers=6, persistent_workers=True, pin_memory=True
    ), val_dataloaders=val_dataloader)
    return trainer.callback_metrics["val_combined_metric"].item()

def train_model(dataset: TimeSeriesDataSet, config: dict, use_optuna: bool = True, continue_training: bool = False):
    logger.info("Rozpoczynanie treningu modelu...")
    
    df = pd.read_csv(config['data']['raw_data_path'])
    selected_tickers = config['data']['tickers']
    df = df[df['Ticker'].isin(selected_tickers)]
    if df.empty:
        raise ValueError(f"Brak danych dla wybranych tickerów: {selected_tickers}")

    preprocessing_utils = PreprocessingUtils(config)
    df, _ = preprocessing_utils.preprocess_dataframe(df)

    min_val_records = config['model'].get('min_prediction_length', 1) + config['model'].get('min_encoder_length', 1)
    group_counts = df.groupby('group_id').size().reset_index(name='count')
    valid_groups = group_counts[group_counts['count'] >= min_val_records]['group_id']
    df = df[df['group_id'].isin(valid_groups)]

    train_df = df[df['time_idx'] <= int(df['time_idx'].max() * 0.8)]
    val_df = df[df['time_idx'] > int(df['time_idx'].max() * 0.8)]
    
    if df.empty or train_df.empty or val_df.empty:
        raise ValueError(f"Zbiory danych są puste po filtrowaniu: df={len(df)}, train_df={len(train_df)}, val_df={len(val_df)}")

    logger.info(f"max_time_idx: {df['time_idx'].max()}, split_idx: {int(df['time_idx'].max() * 0.8)}")

    train_dataset = preprocessing_utils.create_dataset(train_df, dataset.get_parameters())
    val_dataset = preprocessing_utils.create_dataset(val_df, dataset.get_parameters())

    if len(val_dataset) == 0 or len(train_dataset) == 0:
        raise ValueError(f"Zbiory danych są puste: train_dataset={len(train_dataset)}, val_dataset={len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Używane urządzenie: {device}")

    if use_optuna and not continue_training:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        logger.info(f"Najlepsze parametry: {best_params}")
    else:
        best_params = None
        logger.info("Pomijanie optymalizacji Optuna, używanie domyślnych hiperparametrów.")

    model_save_path = Path(config['paths']['model_save_path'])
    logger.info(f"Ścieżka do modelu: {model_save_path}, istnieje: {model_save_path.exists()}")
    if continue_training and model_save_path.exists():
        logger.info(f"Wczytywanie modelu z {model_save_path}")
        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'), weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if config['model'].get('learning_rate') is not None:
            hyperparams['learning_rate'] = config['model']['learning_rate']
            logger.info(f"Zaktualizowano learning_rate w hiperparametrach modelu na: {hyperparams['learning_rate']}")
        final_model = build_model(dataset, config, hyperparams=hyperparams)
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
        final_model = build_model(dataset, config, hyperparams=best_params)

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_combined_metric", patience=config['training']['early_stopping_patience'], mode="min"),
            CustomModelCheckpoint(monitor="val_combined_metric", save_path=config['paths']['model_save_path'], mode="min")
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/")
    )
    trainer.fit(
        model=final_model,
        train_dataloaders=train_dataset.to_dataloader(
            train=True, batch_size=config['training']['batch_size'], num_workers=6, persistent_workers=True, pin_memory=True
        ),
        val_dataloaders=val_dataset.to_dataloader(
            train=False, batch_size=config['training']['batch_size'], num_workers=6, persistent_workers=True, pin_memory=True
        )
    )
    
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "hyperparams": dict(final_model.hparams)
    }
    torch.save(checkpoint, model_save_path)
    logger.info(f"Model zapisany w: {model_save_path}")
    return final_model