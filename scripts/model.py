import pytorch_forecasting
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_lightning import LightningModule
import torch
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import pickle
from pathlib import Path
import numpy as np

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ustaw precyzję dla Tensor Cores na GPU
torch.set_float32_matmul_precision('medium')

def move_to_device(obj: Any, device: torch.device) -> Any:
    """Rekurencyjnie przenosi tensory na wskazane urządzenie."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(val, device) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    return obj

class ModelConfig:
    """Klasa zarządzająca konfiguracją modelu."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_quantile_loss = config['model'].get('use_quantile_loss', False)
        self.quantiles = config['model'].get('quantiles', [0.1, 0.5, 0.9]) if self.use_quantile_loss else None
        self.default_hyperparams = self._get_default_hyperparams()

    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Tworzy domyślne hiperparametry na podstawie konfiguracji."""
        return {
            "hidden_size": self.config['model']['hidden_size'],
            "lstm_layers": self.config['model']['lstm_layers'],
            "attention_head_size": self.config['model']['attention_head_size'],
            "dropout": self.config['model']['dropout'],
            "hidden_continuous_size": self.config['model']['hidden_size'] // 2,
            "output_size": len(self.quantiles) if self.use_quantile_loss else 1,
            "loss": QuantileLoss(quantiles=self.quantiles) if self.use_quantile_loss else MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": self.config['training']['early_stopping_patience'],
            "learning_rate": self.config['model']['learning_rate']
        }

    def get_filtered_params(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Filtruje parametry do przekazania do TemporalFusionTransformer."""
        valid_keys = [
            "hidden_size", "lstm_layers", "attention_head_size", "dropout", "hidden_continuous_size",
            "output_size", "loss", "log_interval", "reduce_on_plateau_patience", "learning_rate"
        ]
        return {k: v for k, v in hyperparams.items() if k in valid_keys}

class HyperparamFactory:
    """Klasa do generowania hiperparametrów na podstawie różnych źródeł."""
    @staticmethod
    def from_trial(trial, config: ModelConfig) -> Dict[str, Any]:
        """Generuje hiperparametry z trialu Optuna."""
        return {
            "hidden_size": trial.suggest_int("hidden_size", config.config['model']['min_hidden_size'], config.config['model']['max_hidden_size']),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "attention_head_size": trial.suggest_int("attention_head_size", config.config['model']['min_attention_head_size'], config.config['model']['max_attention_head_size']),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lstm_layers": trial.suggest_int("lstm_layers", config.config['model']['min_lstm_layers'], config.config['model']['max_lstm_layers']),
            "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 8, config.config['model']['max_hidden_size']),
            "output_size": len(config.quantiles) if config.use_quantile_loss else 1,
            "loss": QuantileLoss(quantiles=config.quantiles) if config.use_quantile_loss else MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": config.config['training']['early_stopping_patience']
        }

    @staticmethod
    def from_checkpoint(hyperparams: Dict[str, Any], config: ModelConfig) -> Dict[str, Any]:
        """Generuje hiperparametry z checkpointu, uzupełniając brakujące wartości."""
        required_keys = [
            "hidden_size", "learning_rate", "attention_head_size", "dropout",
            "lstm_layers", "hidden_continuous_size", "output_size",
            "log_interval", "reduce_on_plateau_patience"
        ]
        filtered_hyperparams = {}
        for key in required_keys:
            if key in hyperparams:
                filtered_hyperparams[key] = hyperparams[key]
            else:
                logger.warning(f"Brak klucza {key} w hiperparametrach, używam wartości domyślnej")
                filtered_hyperparams[key] = config.default_hyperparams[key]
        filtered_hyperparams['loss'] = config.default_hyperparams['loss']
        return filtered_hyperparams

class CustomTemporalFusionTransformer(LightningModule):
    def __init__(self, dataset, config: Dict[str, Any], hyperparams: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_config = ModelConfig(config)
        self.hyperparams = hyperparams if hyperparams else self.model_config.default_hyperparams
        self.normalizers_path = Path(config['data']['normalizers_path'])
        self.dataset = dataset  # Przechowuj dataset, aby uzyskać dostęp do target_normalizer
        self._load_normalizers()
        self._initialize_model(dataset)
        self._save_hyperparameters()

    def _load_normalizers(self):
        """Wczytuje normalizery z pliku."""
        try:
            with open(self.normalizers_path, 'rb') as f:
                self.normalizers = pickle.load(f)
            logger.info(f"Wczytano normalizery z: {self.normalizers_path}")
        except Exception as e:
            logger.error(f"Błąd wczytywania normalizerów: {e}")
            self.normalizers = {}

    def _initialize_model(self, dataset):
        """Inicjalizuje TemporalFusionTransformer z filtrowanymi parametrami."""
        filtered_params = self.model_config.get_filtered_params(self.hyperparams)
        logger.info(f"Parametry przekazywane do TemporalFusionTransformer: {filtered_params}")
        self.model = TemporalFusionTransformer.from_dataset(dataset, **filtered_params)

    def _save_hyperparameters(self):
        """Zapisuje hiperparametry, ignorując 'loss' i dodając informacje o quantile."""
        hparams_to_save = {k: v for k, v in self.hyperparams.items() if k != 'loss'}
        hparams_to_save.update({
            'quantiles': self.model_config.quantiles,
            'use_quantile_loss': self.model_config.use_quantile_loss
        })
        self.save_hyperparameters(hparams_to_save)

    def on_fit_start(self):
        """Przenosi model na GPU przed rozpoczęciem treningu."""
        self.model.to(self.device)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = move_to_device(x, self.device)
        output = self.model(x)
        # Jeśli to tuple/lista, zwróć pierwszy element (predykcje)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def predict(self, data, **kwargs):
        """Deleguje predykcję do wewnętrznego modelu."""
        predictions = self.model.predict(data, **kwargs)
        logger.info(f"Kształt zwracanych predykcji: {predictions.output.shape}")
        return predictions

    def interpret_output(self, x: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Deleguje interpretację wyjścia do wewnętrznego modelu TFT."""
        x = move_to_device(x, self.device)
        
        # Najpierw sprawdź, czy model ma wszystkie wymagane komponenty
        try:
            # Wywołaj model w trybie interpretacji - to zwraca pełne dane
            self.model.eval()
            with torch.no_grad():
                # Wywołaj model bezpośrednio, aby uzyskać pełne dane wyjściowe
                full_output = self.model(x)
                
                # Sprawdź, czy full_output ma wymagane klucze
                if isinstance(full_output, dict):
                    logger.info(f"Model zwrócił następujące klucze: {list(full_output.keys())}")
                    return self.model.interpret_output(full_output, **kwargs)
                else:
                    # Jeśli model zwraca tensor, musimy go przekonwertować na format słownikowy
                    logger.info("Model zwrócił tensor, konwertuję na format słownikowy")
                    
                    # Przygotuj pełne dane wyjściowe poprzez wywołanie _forward_full
                    if hasattr(self.model, '_forward_full'):
                        full_output = self.model._forward_full(x)
                    else:
                        # Wywołaj model bezpośrednio w trybie pełnym
                        full_output = self.model.forward(x)
                    
                    return self.model.interpret_output(full_output, **kwargs)
                    
        except Exception as e:
            logger.error(f"Błąd w interpret_output: {e}")
            # Spróbuj alternatywnego podejścia
            try:
                # Wywołaj model w trybie treningowym, aby uzyskać pełne dane
                self.model.train()
                full_output = self.model(x)
                self.model.eval()
                return self.model.interpret_output(full_output, **kwargs)
            except Exception as e2:
                logger.error(f"Alternatywna metoda również nie działa: {e2}")
                raise e

    def _shared_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int, stage: str) -> torch.Tensor:
        """Wspólna logika dla kroku treningowego i walidacyjnego."""
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
            y_hat = self(x)
            loss = self.model.loss(y_hat, y_target)
        batch_size = x['encoder_cont'].size(0)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Obliczanie l2_norm dla parametrów modelu
        l2_norm = sum(p.pow(2).sum() for p in self.parameters()).sqrt().item()
        self.log(f"{stage}_l2_norm", l2_norm, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

        if stage == 'val' and batch_idx % 50 == 0:
            # Logowanie wartości przed denormalizacją
            logger.info(f"Validation batch {batch_idx}: y_hat[0, :5] = {y_hat[0, :5].tolist()}, y_target[0, :5] = {y_target[0, :5].tolist()}")

            # Denormalizacja y_hat i y_target
            close_normalizer = self.normalizers.get('Close') or self.dataset.target_normalizer
            if close_normalizer:
                try:
                    # Przeniesienie na CPU i konwersja na float32 przed denormalizacją
                    y_hat_denorm = close_normalizer.inverse_transform(y_hat.float().cpu())
                    y_target_denorm = close_normalizer.inverse_transform(y_target.float().cpu())
                    # Logowanie po inverse_transform, przed np.expm1
                    logger.info(f"Validation batch {batch_idx}: y_hat_denorm_before_expm1[0, :5] = {y_hat_denorm[0, :5].tolist()}, y_target_denorm_before_expm1[0, :5] = {y_target_denorm[0, :5].tolist()}")
                    # Odwrócenie transformacji logarytmicznej
                    y_hat_denorm = np.expm1(y_hat_denorm.numpy())
                    y_target_denorm = np.expm1(y_target_denorm.numpy())
                    logger.info(
                        f"Validation batch {batch_idx}: "
                        f"y_hat_denorm[0, :5] = {y_hat_denorm[0, :5].tolist()}, "
                        f"y_target_denorm[0, :5] = {y_target_denorm[0, :5].tolist()}"
                    )
                except Exception as e:
                    logger.error(f"Błąd podczas denormalizacji: {e}")
                    logger.info(f"Validation batch {batch_idx}: y_hat[0, :5] = {y_hat[0, :5].tolist()}, y_target[0, :5] = {y_target[0, :5].tolist()}")
            else:
                logger.warning("Brak normalizera dla 'Close', logowanie znormalizowanych wartości.")
                logger.info(f"Validation batch {batch_idx}: y_hat[0, :5] = {y_hat[0, :5].tolist()}, y_target[0, :5] = {y_target[0, :5].tolist()}")

        return loss

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        """Loguje val_l2_norm i learning_rate na końcu każdej epoki walidacyjnej."""
        # Pobierz średnią val_l2_norm z epoki
        val_l2_norm = self.trainer.callback_metrics.get("val_l2_norm", None)
        if val_l2_norm is not None:
            logger.info(f"Validation epoch end: val_l2_norm = {val_l2_norm:.4f}")
        else:
            logger.warning("val_l2_norm nie jest dostępne w callback_metrics")

        # Pobierz aktualny learning rate z optymalizatora
        optimizer = self.optimizers()
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Validation epoch end: learning_rate = {current_lr:.6f}")
        else:
            logger.warning("Optimizer nie jest dostępny, brak learning_rate")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Konfiguruje optymalizator i scheduler."""
        learning_rate = self.hyperparams.get('learning_rate', self.model_config.config['model']['learning_rate'])
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.model_config.config['training']['reduce_lr_patience'],
            factor=self.model_config.config['training']['reduce_lr_factor'],
            mode='min'
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def build_model(dataset, config: Dict[str, Any], trial=None, hyperparams: Optional[Dict[str, Any]] = None) -> CustomTemporalFusionTransformer:
    """Buduje model z odpowiednimi hiperparametrami."""
    model_config = ModelConfig(config)
    if trial:
        hyperparams = HyperparamFactory.from_trial(trial, model_config)
    elif hyperparams:
        hyperparams = HyperparamFactory.from_checkpoint(hyperparams, model_config)
    else:
        hyperparams = model_config.default_hyperparams
    logger.info(f"Budowanie modelu z hiperparametrami: {hyperparams}")
    return CustomTemporalFusionTransformer(dataset, config, hyperparams)