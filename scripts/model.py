import pytorch_forecasting
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_lightning import LightningModule
import torch
import logging
import time
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
    """Rekurencyjnie przenosi tensory na wskazane urządzenie asynchronicznie z non_blocking=True."""
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj  # Już na właściwym urządzeniu
        return obj.to(device, non_blocking=True)  # Asynchroniczne przenoszenie tensorów
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

class ModelConfig:
    """Klasa zarządzająca konfiguracją modelu."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_quantile_loss = config['model']['use_quantile_loss']
        self.quantiles = config['model']['quantiles'] if self.use_quantile_loss else None
        self.weight_start = config['model']['weight_start']
        self.weight_end = config['model']['weight_end']
        self.directional_weight = config['model']['directional_weight']
        self.embedding_sizes = {
            'Sector': (12, 5),  # 12 kategorii, wymiar osadzenia 5
            'Day_of_Week': (7, 5),  # 7 kategorii, wymiar osadzenia 5
            'Month': (12, 5)  # 12 kategorii, wymiar osadzenia 5
        }
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
            "learning_rate": self.config['model']['learning_rate'],
            "embedding_sizes": self.embedding_sizes,
            "weight_start": self.weight_start,
            "weight_end": self.weight_end,
            "directional_weight": self.directional_weight
        }

    def get_filtered_params(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Filtruje parametry do przekazania do TemporalFusionTransformer."""
        valid_keys = [
            "hidden_size", "lstm_layers", "attention_head_size", "dropout", "hidden_continuous_size",
            "output_size", "loss", "log_interval", "reduce_on_plateau_patience", "learning_rate",
            "embedding_sizes"
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
            "reduce_on_plateau_patience": config.config['training']['early_stopping_patience'],
            "embedding_sizes": config.embedding_sizes,
            "weight_start": trial.suggest_float("weight_start", 1.0, 1.2),
            "weight_end": trial.suggest_float("weight_end", 1.3, 2.0),
            "directional_weight": trial.suggest_float("directional_weight", 0.1, 0.5)
        }

    @staticmethod
    def from_checkpoint(hyperparams: Dict[str, Any], config: ModelConfig) -> Dict[str, Any]:
        """Generuje hiperparametry z checkpointu, uzupełniając brakujące wartości."""
        required_keys = [
            "hidden_size", "learning_rate", "attention_head_size", "dropout",
            "lstm_layers", "hidden_continuous_size", "output_size",
            "log_interval", "reduce_on_plateau_patience", "embedding_sizes",
            "weight_start", "weight_end", "directional_weight"
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
        self.dataset = dataset
        self.val_batch_count = 0
        self.enable_detailed_validation = config['validation']['enable_detailed_validation']
        self.max_val_batches_to_log = config['validation']['max_validation_batches_to_log']
        self.validation_outputs = []  
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
        """Zapisuje hiperparametry, ignorując 'loss' i 'logging_metrics'."""
        hparams_to_save = {k: v for k, v in self.hyperparams.items() if k not in ['loss', 'logging_metrics']}
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
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def predict(self, data, **kwargs):
        """Deleguje predykcję do wewnętrznego modelu z optymalizacją transferu na GPU."""
        start_time = time.time()
        device = self.device
        
        # Sprawdź czy dane to DataLoader i przenieś na GPU
        if hasattr(data, '__iter__') and hasattr(data, 'dataset'):
            # To jest DataLoader - stwórz nowy z właściwym device
            original_dataloader = data
            
            # Stwórz nowy DataLoader z przenoszeniem na GPU
            class GPUDataLoader:
                def __init__(self, original_loader, target_device):
                    self.original_loader = original_loader
                    self.target_device = target_device
                    self.dataset = original_loader.dataset
                    self.batch_size = original_loader.batch_size
                
                def __iter__(self):
                    for batch in self.original_loader:
                        # Przenieś cały batch na GPU asynchronicznie
                        batch_gpu = move_to_device(batch, self.target_device)
                        yield batch_gpu
                
                def __len__(self):
                    return len(self.original_loader)
            
            gpu_dataloader = GPUDataLoader(original_dataloader, device)
            predictions = self.model.predict(gpu_dataloader, **kwargs)
        else:
            # Pojedynczy batch - przenieś na GPU
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
        # Uwaga: move_to_device używa non_blocking=True, co wymaga pin_memory=True w DataLoader
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)
        
        if stage == 'train' and not y_target.requires_grad:
            y_target.requires_grad_(True)
        
        y_target = sanitize_tensor(y_target)
        
        try:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                y_hat = self(x)
                y_hat = sanitize_tensor(y_hat)

                quantile_loss = self.model.loss(y_hat, y_target)
                y_hat_median = y_hat[:, :, 1] if y_hat.dim() == 3 else y_hat
                direction_pred = torch.sign(y_hat_median)
                direction_true = torch.sign(y_target)
                directional_accuracy = (direction_pred == direction_true).float().mean() * 100
                
                directional_weight = self.hyperparams.get('directional_weight', self.model_config.directional_weight)
                total_loss = (1.0 - directional_weight) * quantile_loss + directional_weight * (1.0 - directional_accuracy / 100)
                
                # Obliczanie val_combined_metric
                combined_metric = 0.7 * quantile_loss + 0.3 * (1.0 - directional_accuracy / 100)
                
                prediction_length = y_hat.shape[1]
                weight_start = self.hyperparams.get('weight_start', self.model_config.weight_start)
                weight_end = self.hyperparams.get('weight_end', self.model_config.weight_end)
                weights = torch.linspace(weight_start, weight_end, steps=prediction_length, device=self.device).view(1, -1, 1)
                weights = weights.expand_as(y_hat)
                weighted_loss = (total_loss * weights).mean()
                
                if stage == 'val':
                    mape = torch.mean(torch.abs((y_target - y_hat_median) / (y_target + 1e-10))) * 100
                    self.log(f"{stage}_mape", mape, on_step=False, on_epoch=True, prog_bar=True, batch_size=x['encoder_cont'].size(0))
                    self.log(f"{stage}_directional_accuracy", directional_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=x['encoder_cont'].size(0))
                    self.log(f"{stage}_combined_metric", combined_metric, on_step=True, on_epoch=True, prog_bar=True, batch_size=x['encoder_cont'].size(0))
                
                if not torch.isfinite(weighted_loss):
                    logger.warning(f"Weighted loss nie jest skończony w batch {batch_idx}: {weighted_loss}")
                    weighted_loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
                
        except Exception as e:
            logger.error(f"Błąd podczas forward pass w batch {batch_idx}: {e}")
            weighted_loss = torch.tensor(1e-6, device=self.device, requires_grad=True)
            y_hat = torch.zeros_like(y_target, requires_grad=True)
        
        batch_size = x['encoder_cont'].size(0)
        self.log(f"{stage}_loss", weighted_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        try:
            l2_norm = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad).sqrt().item()
            self.log(f"{stage}_l2_norm", l2_norm, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        except Exception as e:
            logger.warning(f"Nie można obliczyć l2_norm: {e}")
        
        if stage == 'val' and self.enable_detailed_validation and self.val_batch_count < self.max_val_batches_to_log:
            try:
                self._log_validation_details(x, y_hat, y_target, batch_idx)
                self.val_batch_count += 1
            except Exception as e:
                logger.error(f"Błąd w logowaniu szczegółów walidacji: {e}")
        
        if stage == 'val':
            return {
                'val_loss': weighted_loss,
                'val_l2_norm': torch.tensor(l2_norm, device=self.device),
                'val_directional_accuracy': directional_accuracy,
                'val_mape': mape,
                'val_combined_metric': combined_metric
            }
        return weighted_loss

    def _log_validation_details(self, x, y_hat, y_target, batch_idx):
        """Wydzielona funkcja do logowania szczegółów walidacji, zminimalizowana synchronizacja."""
        relative_returns_normalizer = self.normalizers.get('Relative_Returns') or self.dataset.target_normalizer
        if relative_returns_normalizer:
            try:
                # Minimalizujemy przenoszenie na CPU, wykonując tylko gdy konieczne
                y_hat_denorm = relative_returns_normalizer.inverse_transform(y_hat.float())
                y_target_denorm = relative_returns_normalizer.inverse_transform(y_target.float())
                
                if 'encoder_cont' in x:
                    encoder_cont = x['encoder_cont'][0]
                    close_normalizer = self.normalizers.get('Close')
                    if close_normalizer is not None:
                        try:
                            numeric_features = [
                                "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
                                "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
                                "Close_momentum_1d", "Close_momentum_5d", "Close_vs_MA10", "Close_vs_MA50",
                                "Close_percentile_20d", "Close_volatility_5d", "Close_RSI_divergence"
                            ]
                            close_idx = numeric_features.index("Close") if "Close" in numeric_features else None
                            
                            if close_idx is not None:
                                last_close_norm = encoder_cont[-1, close_idx]
                                last_close_denorm = close_normalizer.inverse_transform(torch.tensor([[last_close_norm]], device=self.device))
                                last_close_price = np.expm1(last_close_denorm.cpu().numpy())[0, 0]
                                
                                if last_close_price > 10000:
                                    logger.warning(f"Bardzo wysoka cena Close: {last_close_price:.2f}")
                                    last_close_price_alt = last_close_denorm.cpu().numpy()[0, 0]
                                    if 10 <= last_close_price_alt <= 1000:
                                        last_close_price = last_close_price_alt
                                
                                logger.info(f"Ostatnia cena Close z batcha: {last_close_price:.2f}")
                                self._convert_to_prices(y_hat_denorm, y_target_denorm, last_close_price, batch_idx)
                            else:
                                logger.warning("Nie można znaleźć indeksu kolumny Close")
                        except Exception as e:
                            logger.error(f"Błąd podczas konwersji na rzeczywiste ceny: {e}")
                    else:
                        logger.warning("Brak normalizera dla Close")
                else:
                    logger.warning("Brak danych encoder_cont w batchu")
            except Exception as e:
                logger.error(f"Błąd podczas denormalizacji Relative Returns: {e}")
        else:
            logger.warning("Brak normalizera dla 'Relative_Returns'")

    def _convert_to_prices(self, y_hat_denorm, y_target_denorm, last_close_price, batch_idx):
        def to_scalar(tensor_val):
            if hasattr(tensor_val, 'numel') and tensor_val.numel() == 1:
                return tensor_val.item()
            elif hasattr(tensor_val, 'cpu'):
                val = tensor_val.cpu().numpy()
                return val.item() if val.size == 1 else val.flatten()[0]
            else:
                return float(tensor_val)
        
        y_hat_prices = []
        y_target_prices = []
        current_price_pred = last_close_price
        current_price_target = last_close_price
        
        for i in range(min(5, y_hat_denorm.shape[1])):
            # Dla predykcji (quantiles)
            if y_hat_denorm.dim() == 3:  # Sprawdź, czy tensor ma 3 wymiary
                relative_return_pred = to_scalar(y_hat_denorm[0, i, 1])  # mediana
                relative_return_pred_lower = to_scalar(y_hat_denorm[0, i, 0])  # dolny
                relative_return_pred_upper = to_scalar(y_hat_denorm[0, i, 2])  # górny
            else:
                logger.warning(f"y_hat_denorm ma nieoczekiwany kształt: {y_hat_denorm.shape}")
                relative_return_pred = to_scalar(y_hat_denorm[0, i])
                relative_return_pred_lower = relative_return_pred
                relative_return_pred_upper = relative_return_pred
            
            next_price_pred = current_price_pred * (1 + relative_return_pred)
            next_price_pred_lower = current_price_pred * (1 + relative_return_pred_lower)
            next_price_pred_upper = current_price_pred * (1 + relative_return_pred_upper)
            
            y_hat_prices.append({
                'median': next_price_pred,
                'lower': next_price_pred_lower,
                'upper': next_price_pred_upper
            })
            current_price_pred = next_price_pred
            
            relative_return_target = to_scalar(y_target_denorm[0, i])
            next_price_target = current_price_target * (1 + relative_return_target)
            y_target_prices.append(next_price_target)
            current_price_target = next_price_target
        
        pred_medians = [f"{p['median']:.2f}" for p in y_hat_prices]
        pred_lowers = [f"{p['lower']:.2f}" for p in y_hat_prices]
        pred_uppers = [f"{p['upper']:.2f}" for p in y_hat_prices]
        target_prices_formatted = [f"{p:.2f}" for p in y_target_prices]
        
        logger.info(
            f"Validation batch {batch_idx} - RZECZYWISTE CENY:\n"
            f"  Predykcje (mediana): {pred_medians}\n"
            f"  Predykcje (dolny 10%): {pred_lowers}\n"
            f"  Predykcje (górny 90%): {pred_uppers}\n"
            f"  Rzeczywiste ceny: {target_prices_formatted}"
        )

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> None:
        output = self._shared_step(batch, batch_idx, 'val')
        self.validation_outputs.append(output)

    def on_validation_epoch_end(self) -> None:
        """Loguje średnie metryki na końcu każdej epoki walidacyjnej."""
        if self.validation_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
            avg_l2_norm = torch.stack([x['val_l2_norm'] for x in self.validation_outputs]).mean()
            avg_directional_accuracy = torch.stack([x['val_directional_accuracy'] for x in self.validation_outputs]).mean()
            avg_mape = torch.stack([x['val_mape'] for x in self.validation_outputs]).mean()
            avg_combined_metric = torch.stack([x['val_combined_metric'] for x in self.validation_outputs]).mean()
            self.log('val_loss_epoch', avg_loss, prog_bar=True)
            self.log('val_l2_norm_epoch', avg_l2_norm)
            self.log('val_directional_accuracy_epoch', avg_directional_accuracy, prog_bar=True)
            self.log('val_mape_epoch', avg_mape, prog_bar=True)
            self.log('val_combined_metric_epoch', avg_combined_metric, prog_bar=True)
            logger.info(f"Validation epoch end: val_l2_norm = {avg_l2_norm:.4f}")
            logger.info(f"Validation epoch end: val_combined_metric = {avg_combined_metric:.4f}")
            logger.info(f"Validation epoch end: learning_rate = {self.optimizers().param_groups[0]['lr']:.6f}")
        else:
            logger.warning("Brak wyników walidacji w validation_outputs")
        self.validation_outputs.clear()  # Czyszczenie wyników po epoce
        self.val_batch_count = 0  # Resetowanie licznika po epoce

    def configure_optimizers(self) -> Dict[str, Any]:
        """Konfiguruje optymalizator i scheduler."""
        learning_rate = self.hyperparams.get('learning_rate', self.model_config.config['model']['learning_rate'])
        weight_decay = self.model_config.config['training']['weight_decay']
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.model_config.config['training']['reduce_lr_patience'],
            factor=self.model_config.config['training']['reduce_lr_factor']
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_combined_metric",
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