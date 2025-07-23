import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from pytorch_forecasting import TimeSeriesDataSet
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.model import build_model, CustomTemporalFusionTransformer
from scripts.config_manager import ConfigManager
import optuna
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Konfiguracja logowania
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
        precision="16-mixed" if torch.cuda.is_available() else "32-true",  # Zaktualizowano na float16
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
        logger.info(f"Validation batch: y[0, :5] = {y[0][:5].tolist()}")
        break
    trainer.fit(model, train_dataloaders=train_dataset.to_dataloader(
        train=True, batch_size=config['training']['batch_size'], num_workers=4, persistent_workers=True
    ), val_dataloaders=val_dataloader)
    return trainer.callback_metrics["val_loss"].item()

def train_model(dataset: TimeSeriesDataSet, config: dict, use_optuna: bool = True, continue_training: bool = False):
    logger.info("Rozpoczynanie treningu modelu...")
    
    # Pobierz dane z raw_data_path i odfiltruj tylko wybrane tickery
    df = pd.read_csv(config['data']['raw_data_path'])
    selected_tickers = config['data']['tickers']
    df = df[df['Ticker'].isin(selected_tickers)]
    if df.empty:
        raise ValueError(f"Brak danych dla wybranych tickerów: {selected_tickers}")
    
    preprocessor = DataPreprocessor(config)
    df = preprocessor.feature_engineer.add_features(df)
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])
    df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])]
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days.astype(int)
    df['group_id'] = df['Ticker']
    
    # Wczytaj normalizery
    with open(config['data']['normalizers_path'], 'rb') as f:
        normalizers = pickle.load(f)
    logger.info(f"Wczytano normalizery z: {config['data']['normalizers_path']}")
    
    # Transformacja logarytmiczna - specjalna obsługa dla cech fundamentalnych
    log_features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR"]
    fundamental_features = ["PE_ratio", "PB_ratio", "EPS"]
    
    # Standardowa transformacja log dla cech technicznych
    for feature in log_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature].clip(lower=0))
    
    # Specjalna obsługa dla cech fundamentalnych
    for feature in fundamental_features:
        if feature in df.columns:
            # Sprawdź czy cecha ma jakiekolwiek niezerowe wartości
            nonzero_count = (df[feature] != 0.0).sum()
            total_count = len(df[feature])
            
            if nonzero_count == 0:
                logger.warning(f"Cecha {feature} zawiera wyłącznie zera, pomijam transformację log")
                # Pozostaw jako zera - nie rób transformacji log
            else:
                logger.info(f"Cecha {feature}: {nonzero_count}/{total_count} ({100*nonzero_count/total_count:.1f}%) niezerowych wartości")
                # Zastosuj transformację log tylko do wartości niezerowych
                mask = df[feature] > 0
                if mask.any():
                    df.loc[mask, feature] = np.log1p(df.loc[mask, feature])
                # Wartości zerowe pozostają zerami
    
    # Normalizacja z sprawdzeniem dostępności cech
    numeric_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
        "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
        "Close_momentum_1d", "Close_momentum_5d", "Close_vs_MA10", "Close_vs_MA50",
        "Close_percentile_20d", "Close_volatility_5d", "Close_RSI_divergence",
        "Relative_Returns", "Log_Returns", "Future_Volume", "Future_Volatility",
        "PE_ratio", "PB_ratio", "EPS"
    ]
    
    # Sprawdź które cechy fundamentalne są dostępne w normalizerach
    available_fundamental = []
    for feature in ["PE_ratio", "PB_ratio", "EPS"]:
        if feature in normalizers and feature in df.columns:
            available_fundamental.append(feature)
        elif feature in df.columns:
            logger.warning(f"Cecha {feature} jest w danych ale nie ma dla niej normalizera, pomijam")
    
    logger.info(f"Dostępne cechy fundamentalne: {available_fundamental}")
    
    for feature in numeric_features:
        if feature in df.columns and feature in normalizers:
            try:
                df[feature] = normalizers[feature].transform(df[feature].values)
                # Sprawdź czy nie ma NaN po transformacji
                if df[feature].isna().any() or np.isinf(df[feature]).any():
                    logger.error(f"Transformacja cechy {feature} spowodowała NaN lub inf")
            except Exception as e:
                logger.error(f"Błąd transformacji cechy {feature}: {e}")
        elif feature in df.columns:
            logger.warning(f"Brak normalizera dla cechy {feature}")
    
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            # Sprawdź końcowy stan cechy
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Cecha {col} ma {nan_count} wartości NaN po konwersji")
    
    categorical_columns = ['Day_of_Week', 'Month']
    for cat_col in categorical_columns:
        if cat_col in df.columns:
            if cat_col == 'Day_of_Week':
                # Wymuś 7-dniowy tydzień, uzupełnij braki na poniedziałek (0)
                df[cat_col] = pd.Categorical(df[cat_col], categories=[0,1,2,3,4,5,6])
                df[cat_col] = df[cat_col].fillna(0)
            df[cat_col] = df[cat_col].astype(str)

    # Filtracja grup z wystarczającą liczbą rekordów
    min_val_records = config['model'].get('min_prediction_length', 1) + config['model'].get('min_encoder_length', 1)
    group_counts = df.groupby('group_id').size().reset_index(name='count')
    logger.info(f"Liczba rekordów dla każdej grupy:\n{group_counts.to_string()}")
    
    valid_groups = group_counts[group_counts['count'] >= min_val_records]['group_id']
    
    df = df[df['group_id'].isin(valid_groups)]
    train_df = df[df['time_idx'] <= int(df['time_idx'].max() * 0.8)]
    val_df = df[df['time_idx'] > int(df['time_idx'].max() * 0.8)]
    
    if df.empty or train_df.empty or val_df.empty:
        raise ValueError(f"Zbiory danych są puste po filtrowaniu: df={len(df)}, train_df={len(train_df)}, val_df={len(val_df)}")

    logger.info(f"Statystyki time_idx: {df['time_idx'].describe()}")
    logger.info(f"max_time_idx: {df['time_idx'].max()}, split_idx: {int(df['time_idx'].max() * 0.8)}")

    # Naprawienie typów dla flag missing w zbiorach danych
    fundamental_missing_flags = ['PE_ratio_missing', 'PB_ratio_missing', 'EPS_missing']
    for flag in fundamental_missing_flags:
        if flag in train_df.columns:
            train_df = train_df.copy()  # Zapewniamy, że mamy kopię do modyfikacji
            train_df[flag] = train_df[flag].astype(str)
            logger.info(f"Przekonwertowano {flag} na string w train_df: {train_df[flag].unique()}")
        if flag in val_df.columns:
            val_df = val_df.copy()  # Zapewniamy, że mamy kopię do modyfikacji
            val_df[flag] = val_df[flag].astype(str)
            logger.info(f"Przekonwertowano {flag} na string w val_df: {val_df[flag].unique()}")

    # Tworzenie datasetów treningowego i walidacyjnego
    train_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), train_df)
    val_dataset = TimeSeriesDataSet.from_parameters(dataset.get_parameters(), val_df)

    if len(val_dataset) == 0 or len(train_dataset) == 0:
        raise ValueError(f"Zbiory danych są puste: train_dataset={len(train_dataset)}, val_dataset={len(val_dataset)}")

    # Ustal urządzenie
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Używane urządzenie: {device}")

    # Optymalizacja z Optuna
    if use_optuna and not continue_training:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        logger.info(f"Najlepsze parametry: {best_params}")
    else:
        best_params = None
        logger.info("Pomijanie optymalizacji Optuna, używanie domyślnych hiperparametrów.")

    # Wczytywanie modelu
    checkpoint_path = Path(config['paths']['checkpoint_path'])
    logger.info(f"Ścieżka do checkpointu: {checkpoint_path}, istnieje: {checkpoint_path.exists()}")
    if continue_training and checkpoint_path.exists():
        logger.info(f"Wczytywanie checkpointu z {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        hyperparams = checkpoint["hyperparams"]
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
        logger.info("Brak checkpointu lub kontynuacja wyłączona, trenowanie od zera")
        final_model = build_model(dataset, config, hyperparams=best_params)

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",  # Zaktualizowano na float16
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
    
    # Zapisz model
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "hyperparams": dict(final_model.hparams)
    }
    torch.save(checkpoint, Path(config['paths']['model_save_path']))
    logger.info(f"Model zapisany w: {config['paths']['model_save_path']}")
    return final_model

if __name__ == "__main__":
    config = ConfigManager().config
    dataset = torch.load(config['data']['processed_data_path'], weights_only=False)
    train_model(dataset, config, use_optuna=True, continue_training=False)