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
import sys
import os
import time
import matplotlib.pyplot as plt

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.config_manager import ConfigManager 
from scripts.utils.model_config import ModelConfig, HyperparamFactory
from scripts.utils.validation_utils import log_validation_details, create_validation_plot, convert_to_prices

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ustaw precyzję dla Tensor Cores na GPU
torch.set_float32_matmul_precision('medium')

# Dodana optymalizacja: Włącz TF32 dla szybszych matmul w mixed precision
torch.backends.cuda.matmul.allow_tf32 = True

def move_to_device(obj: Any, device: torch.device) -> Any:
    """Rekurencyjnie przenosi tensory na wskazane urządzenie asynchronicznie z non_blocking=True."""
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {key: move_to_device(val, device) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    return obj

def sanitize_tensor(tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Usuwa NaN i Inf z tensora, zastępując je fill_value."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning("Wykryto NaN lub Inf w tensorze, zastępuję wartościami 0.0")
        return torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return tensor

class CustomTemporalFusionTransformer(LightningModule):
    def __init__(self, dataset, config: Dict[str, Any], hyperparams: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_config = ModelConfig(config)
        self.hyperparams = hyperparams if hyperparams else self.model_config.default_hyperparams
        self.model_name = config['model_name']
        self.dataset = dataset
        self.config_manager = ConfigManager()
        self._load_normalizers()
        self._initialize_model(dataset)
        self._save_hyperparameters()
        self.val_batch_count = 0
        self.enable_detailed_validation = config['validation']['enable_detailed_validation'] 
        self.max_val_batches_to_log = config['validation']['max_validation_batches_to_log']
        self.save_plots = config['validation']['save_plots']  
        self.max_plots_per_epoch = config['validation']['max_plots_per_epoch']
        self.logs_dir = config['paths']['logs_dir']
        self.plot_count = 0
        self.debug = config['validation']['debug']

    def _load_normalizers(self):
        """Wczytuje normalizery za pomocą ConfigManager."""
        try:
            self.normalizers = self.config_manager.load_normalizers(self.model_name)
            logger.info(f"Wczytano normalizery dla modelu: {self.model_name}")
        except Exception as e:
            logger.error(f"Błąd wczytywania normalizerów: {e}")
            self.normalizers = {}

    def _initialize_model(self, dataset):
        """Inicjalizuje TemporalFusionTransformer z filtrowanymi parametrami."""
        filtered_params = self.model_config.get_filtered_params(self.hyperparams)
        logger.info(f"Parametry przekazywane do TemporalFusionTransformer: {filtered_params}")
        # użyj podklasy z nadpisanym transfer_batch_to_device
        self.model = TFTWithTransfer.from_dataset(dataset, **filtered_params)

    def _save_hyperparameters(self):
        """Zapisuje hiperparametry, ignorując 'loss' i dodając informacje o quantile."""
        hparams_to_save = {k: v for k, v in self.hyperparams.items() if k != 'loss'}
        self.save_hyperparameters(hparams_to_save)

    def on_fit_start(self):
        """Przenosi model na GPU przed rozpoczęciem treningu."""
        self.model.to(self.device)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = move_to_device(x, self.device)
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def predict(self, data, **kwargs):
        """Deleguje predykcję do wewnętrznego modelu. Nie opakowujemy DataLoadera — Lightning przeniesie batch."""
        start_time = time.time()
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Uruchamianie predykcji na urządzeniu: {device}")

        # Nie opakowujemy DataLoadera — przekażemy go bezpośrednio
        if isinstance(data, torch.utils.data.DataLoader):
            predictions = self.model.predict(data, **kwargs)
        else:
            data_gpu = move_to_device(data, device)
            predictions = self.model.predict(data_gpu, **kwargs)

        prediction_duration = time.time() - start_time
        logger.info(f"Kształt zwracanych predykcji: {predictions.output.shape}")
        logger.info(f"Czas predykcji w metodzie predict: {prediction_duration:.3f} sekundy")
        return predictions

    def interpret_output(self, x: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Deleguje interpretację wyjścia do wewnętrznego modelu TFT."""
        x = move_to_device(x, self.device)
        try:
            self.model.eval()
            with torch.no_grad():
                full_output = self.model(x)
                if isinstance(full_output, dict):
                    logger.info(f"Model zwrócił następujące klucze: {list(full_output.keys())}")
                    return self.model.interpret_output(full_output, **kwargs)
                else:
                    logger.info("Model zwrócił tensor, konwertuję na format słownikowy")
                    if hasattr(self.model, '_forward_full'):
                        full_output = self.model._forward_full(x)
                    else:
                        full_output = self.model.forward(x)
                    return self.model.interpret_output(full_output, **kwargs)
        except Exception as e:
            logger.error(f"Błąd w interpret_output: {e}")
            try:
                self.model.train()
                full_output = self.model(x)
                self.model.eval()
                return self.model.interpret_output(full_output, **kwargs)
            except Exception as e2:
                logger.error(f"Alternatywna metoda również nie działa: {e2}")
                raise e
            
    def _shared_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int, stage: str) -> torch.Tensor:
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)
        
        if stage == 'train' and not y_target.requires_grad:
            y_target.requires_grad_(True)
        
        if self.debug:
            if torch.isnan(y_target).any() or torch.isinf(y_target).any():
                logger.warning(f"NaN/Inf w y_target w batch {batch_idx}")
                y_target = torch.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                y_hat = self(x)

                if self.debug:
                    if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                        logger.warning(f"NaN/Inf w y_hat w batch {batch_idx}")
                        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Obliczanie straty
                loss = self.model.loss(y_hat, y_target)
                
                # Obliczanie dodatkowych metryk tylko dla walidacji
                if stage == 'val':
                    # Wybierz medianę dla metryk (indeks 1 dla kwantyli [0.1, 0.5, 0.9])
                    y_hat_median = y_hat[:, :, 1] if y_hat.dim() == 3 else y_hat
                    
                    # MAPE
                    mape = torch.mean(torch.abs((y_target - y_hat_median) / (y_target + 1e-10))) * 100
                    self.log(f"{stage}_mape", mape, on_step=False, on_epoch=True, prog_bar=True, batch_size=x['encoder_cont'].size(0))
                    
                    # Directional Accuracy
                    direction_pred = torch.sign(y_hat_median)
                    direction_true = torch.sign(y_target)
                    directional_accuracy = (direction_pred == direction_true).float().mean() * 100
                    self.log(f"{stage}_directional_accuracy", directional_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=x['encoder_cont'].size(0))
                
                if not torch.isfinite(loss):
                    logger.warning(f"Loss nie jest skończony w batch {batch_idx}: {loss}")
                    loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
                
        except Exception as e:
            logger.error(f"Błąd podczas forward pass w batch {batch_idx}: {e}")
            loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
            y_hat = torch.zeros_like(y_target, requires_grad=True)
        
        batch_size = x['encoder_cont'].size(0)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        try:
            l2_norm = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad).sqrt().item()
            self.log(f"{stage}_l2_norm", l2_norm, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        except Exception as e:
            logger.warning(f"Nie można obliczyć l2_norm: {e}")
        
        if stage == 'val':
            # Szczegółowe logowanie walidacji (tylko jeśli włączone)
            if self.enable_detailed_validation and self.val_batch_count < self.max_val_batches_to_log:
                try:
                    log_validation_details(
                        x, y_hat, y_target, batch_idx,
                        self.normalizers, self.dataset,
                        self.save_plots, self.plot_count, self.max_plots_per_epoch,
                        self.logs_dir, self.current_epoch
                    )
                    self.val_batch_count += 1
                except Exception as e:
                    logger.error(f"Błąd w logowaniu szczegółów walidacji: {e}")
            # Generowanie wykresów niezależnie od enable_detailed_validation
            if self.save_plots and self.plot_count < self.max_plots_per_epoch:
                try:
                    relative_returns_normalizer = self.normalizers.get('Relative_Returns') or self.dataset.target_normalizer
                    if relative_returns_normalizer:
                        y_hat_denorm = relative_returns_normalizer.inverse_transform(y_hat.float().cpu())
                        y_target_denorm = relative_returns_normalizer.inverse_transform(y_target.float().cpu())
                        create_validation_plot(
                            y_hat_denorm, y_target_denorm, batch_idx,
                            self.logs_dir, self.current_epoch
                        )
                        self.plot_count += 1
                    else:
                        logger.warning("Brak normalizera dla 'Relative_Returns' do wykresu")
                except Exception as e:
                    logger.error(f"Błąd podczas tworzenia wykresu walidacyjnego: {e}")
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Loguje val_l2_norm i learning_rate na końcu każdej epoki walidacyjnej oraz resetuje licznik wykresów."""
        val_l2_norm = self.trainer.callback_metrics.get("val_l2_norm", None)
        if val_l2_norm is not None:
            logger.info(f"Validation epoch end: val_l2_norm = {val_l2_norm:.4f}")
        else:
            logger.warning("val_l2_norm nie jest dostępne w callback_metrics")

        optimizer = self.optimizers()
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Validation epoch end: learning_rate = {current_lr:.6f}")
        else:
            logger.warning("Optimizer nie jest dostępny, brak learning_rate")
        self.val_batch_count = 0
        self.plot_count = 0  # Resetowanie licznika wykresów na końcu epoki

    def configure_optimizers(self) -> Dict[str, Any]:
        """Konfiguruje optymalizator i scheduler."""
        learning_rate = self.hyperparams.get('learning_rate', self.model_config.config['model']['learning_rate'])
        weight_decay = self.model_config.config['training']['weight_decay']
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """
        Metoda wymagana przez PyTorch Lightning do treningu.
        """
        return self._shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """
        Metoda wymagana przez PyTorch Lightning do walidacji.
        """
        return self._shared_step(batch, batch_idx, stage='val')

# dodaj subclass, która użyje Twojej funkcji move_to_device
class TFTWithTransfer(TemporalFusionTransformer):
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_to_device(batch, device)

def build_model(dataset, config: Dict[str, Any], trial=None, hyperparams: Optional[Dict[str, Any]] = None) -> CustomTemporalFusionTransformer:
    """Buduje model z odpowiednimi hiperparametrami."""
    model_config = ModelConfig(config)
    if trial:
        hyperparams = HyperparamFactory.from_trial(trial, model_config)
    elif hyperparams:
        hyperparams = HyperparamFactory.from_checkpoint(hyperparams, model_config)
    else:
        hyperparams = model_config.default_hyperparams
    return CustomTemporalFusionTransformer(dataset, config, hyperparams)