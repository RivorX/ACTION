import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
import pytorch_forecasting
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

# Lista wszystkich możliwych sektorów
ALL_SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 'Consumer Staples',
    'Energy', 'Utilities', 'Industrials', 'Materials', 'Communication Services',
    'Real Estate', 'Unknown'
]

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
    
    # Upewnij się, że Sector i Day_of_Week są kategoryczne
    df['Sector'] = pd.Categorical(df['Sector'], categories=ALL_SECTORS, ordered=False)
    df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=[str(i) for i in range(7)], ordered=False)
    logger.info(f"Kategorie sektorów w train.py: {df['Sector'].cat.categories.tolist()}")
    logger.info(f"Kategorie dni tygodnia w train.py: {df['Day_of_Week'].cat.categories.tolist()}")
    
    # Wczytaj normalizery
    with open(config['data']['normalizers_path'], 'rb') as f:
        normalizers = pickle.load(f)
    logger.info(f"Wczytano normalizery z: {config['data']['normalizers_path']}")
    
    # Transformacja logarytmiczna
    log_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "ATR", "BB_width",
        "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "VWAP"
    ]
    
    for feature in log_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature].clip(lower=0))
    
    # Normalizacja z sprawdzeniem dostępności cech
    numeric_features = [
        "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI",
        "MACD", "MACD_Signal", "MACD_Histogram", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
        "ADX", "CCI", "Tenkan_sen", "Kijun_sen", "Senkou_Span_A", "Senkou_Span_B", "ROC", "VWAP",
        "Momentum_20d", "Close_to_MA_ratio", "BB_width", "Close_to_BB_upper", "Close_to_BB_lower",
        "Relative_Returns"
    ]
    
    for feature in numeric_features:
        if feature in df.columns and feature in normalizers:
            try:
                df[feature] = normalizers[feature].transform(df[feature].values)
                if df[feature].isna().any() or np.isinf(df[feature].any()):
                    logger.error(f"Transformacja cechy {feature} spowodowała NaN lub inf")
            except Exception as e:
                logger.error(f"Błąd transformacji cechy {feature}: {e}")
        elif feature in df.columns:
            logger.info(f"Pomijanie normalizacji dla cechy {feature}, ponieważ nie jest w normalizers.pkl")

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Cecha {col} ma {nan_count} wartości NaN po konwersji")
    
    categorical_columns = ['Day_of_Week', 'Month']
    for cat_col in categorical_columns:
        if cat_col in df.columns:
            if cat_col == 'Day_of_Week':
                df[cat_col] = pd.Categorical(df[cat_col], categories=[str(i) for i in range(7)], ordered=False)
            df[cat_col] = df[cat_col].astype(str)

    # Filtracja grup z wystarczającą liczbą rekordów
    min_val_records = config['model'].get('min_prediction_length', 1) + config['model'].get('min_encoder_length', 1)
    group_counts = df.groupby('group_id').size().reset_index(name='count')
    
    valid_groups = group_counts[group_counts['count'] >= min_val_records]['group_id']
    
    df = df[df['group_id'].isin(valid_groups)]
    train_df = df[df['time_idx'] <= int(df['time_idx'].max() * 0.8)]
    val_df = df[df['time_idx'] > int(df['time_idx'].max() * 0.8)]
    
    if df.empty or train_df.empty or val_df.empty:
        raise ValueError(f"Zbiory danych są puste po filtrowaniu: df={len(df)}, train_df={len(train_df)}, val_df={len(val_df)}")

    logger.info(f"max_time_idx: {df['time_idx'].max()}, split_idx: {int(df['time_idx'].max() * 0.8)}")

    # Tworzenie datasetów treningowego i walidacyjnego
    train_dataset = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(),
        train_df,
        static_categoricals=["Sector"], 
        categorical_encoders={
            'Sector': NaNLabelEncoder(add_nan=False),
            'Day_of_Week': NaNLabelEncoder(add_nan=False),
            'Month': NaNLabelEncoder(add_nan=False)
        }
    )
    val_dataset = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(),
        val_df,
        static_categoricals=["Sector"], 
        categorical_encoders={
            'Sector': NaNLabelEncoder(add_nan=False),
            'Day_of_Week': NaNLabelEncoder(add_nan=False),
            'Month': NaNLabelEncoder(add_nan=False)
        }
    )

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
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
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