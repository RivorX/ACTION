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
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)
        
        if stage == 'train' and not y_target.requires_grad:
            y_target.requires_grad_(True)
        
        if torch.isnan(y_target).any() or torch.isinf(y_target).any():
            logger.warning(f"NaN/Inf w y_target w batch {batch_idx}")
            y_target = torch.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                y_hat = self(x)
                
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
        
        if stage == 'val' and batch_idx % 50 == 0:
            try:
                self._log_validation_details(x, y_hat, y_target, batch_idx)
            except Exception as e:
                logger.error(f"Błąd w logowaniu szczegółów walidacji: {e}")
        
        return loss

    def _log_validation_details(self, x, y_hat, y_target, batch_idx):
        """Wydzielona funkcja do logowania szczegółów walidacji."""
        # Denormalizacja y_hat i y_target (Relative Returns)
        relative_returns_normalizer = self.normalizers.get('Relative_Returns') or self.dataset.target_normalizer
        if relative_returns_normalizer:
            try:
                # Przeniesienie na CPU i konwersja na float32 przed denormalizacją
                y_hat_denorm = relative_returns_normalizer.inverse_transform(y_hat.float().cpu())
                y_target_denorm = relative_returns_normalizer.inverse_transform(y_target.float().cpu())
                
                # KONWERSJA RELATIVE RETURNS NA RZECZYWISTE CENY
                if 'encoder_cont' in x:
                    encoder_cont = x['encoder_cont'][0].cpu()  # Pierwszy przykład z batcha
                    close_normalizer = self.normalizers.get('Close')
                    if close_normalizer is not None:
                        try:
                            # Znajdź pozycję Close w numeric_features
                            numeric_features = [
                                "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
                                "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
                                "Close_momentum_1d", "Close_momentum_5d", "Close_vs_MA10", "Close_vs_MA50",
                                "Close_percentile_20d", "Close_volatility_5d", "Close_RSI_divergence"
                            ]
                            close_idx = numeric_features.index("Close") if "Close" in numeric_features else None
                            
                            if close_idx is not None:
                                # Pobierz ostatnią wartość Close z encodera
                                last_close_norm = encoder_cont[-1, close_idx]
                                last_close_denorm = close_normalizer.inverse_transform(torch.tensor([[last_close_norm]]))
                                last_close_price = np.expm1(last_close_denorm.numpy())[0, 0]
                                
                                # Sprawdź rozsądność ceny
                                if last_close_price > 10000:
                                    logger.warning(f"Bardzo wysoka cena Close: {last_close_price:.2f}")
                                    last_close_price_alt = last_close_denorm.numpy()[0, 0]
                                    if 10 <= last_close_price_alt <= 1000:
                                        last_close_price = last_close_price_alt
                                
                                logger.info(f"Ostatnia cena Close z batcha: {last_close_price:.2f}")
                                
                                # Konwertuj tylko pierwsze 5 predykcji dla czytelności
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
            
            # Oblicz ceny
            next_price_pred = current_price_pred * (1 + relative_return_pred)
            next_price_pred_lower = current_price_pred * (1 + relative_return_pred_lower)
            next_price_pred_upper = current_price_pred * (1 + relative_return_pred_upper)
            
            y_hat_prices.append({
                'median': next_price_pred,
                'lower': next_price_pred_lower,
                'upper': next_price_pred_upper
            })
            current_price_pred = next_price_pred
            
            # Dla wartości rzeczywistych
            relative_return_target = to_scalar(y_target_denorm[0, i])
            next_price_target = current_price_target * (1 + relative_return_target)
            y_target_prices.append(next_price_target)
            current_price_target = next_price_target
        
        # Formatowanie wyników
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