import logging
from typing import Dict, Any, Optional
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss

logger = logging.getLogger(__name__)

class ModelConfig:
    """Klasa zarządzająca konfiguracją modelu."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_quantile_loss = config['model'].get('use_quantile_loss', False)
        self.quantiles = config['model'].get('quantiles', [0.1, 0.5, 0.9]) if self.use_quantile_loss else None
        # Pobieranie embedding_sizes z konfiguracji
        self.embedding_sizes = config['model']['embedding_sizes']
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
            "embedding_sizes": self.embedding_sizes
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
        tuning_config = config.config['training']['tuning']
        return {
            "hidden_size": trial.suggest_int("hidden_size", tuning_config['min_hidden_size'], tuning_config['max_hidden_size']),
            "learning_rate": trial.suggest_float("learning_rate", tuning_config['min_learning_rate'], tuning_config['max_learning_rate'], log=True),
            "attention_head_size": trial.suggest_int("attention_head_size", tuning_config['min_attention_head_size'], tuning_config['max_attention_head_size']),
            "dropout": trial.suggest_float("dropout", tuning_config['min_dropout'], tuning_config['max_dropout']),
            "lstm_layers": trial.suggest_int("lstm_layers", tuning_config['min_lstm_layers'], tuning_config['max_lstm_layers']),
            "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", tuning_config['min_hidden_continuous_size'], tuning_config['max_hidden_continuous_size']),
            "output_size": len(config.quantiles) if config.use_quantile_loss else 1,
            "loss": QuantileLoss(quantiles=config.quantiles) if config.use_quantile_loss else MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": config.config['training']['early_stopping_patience'],
            "embedding_sizes": config.embedding_sizes
        }

    @staticmethod
    def from_checkpoint(hyperparams: Dict[str, Any], config: ModelConfig) -> Dict[str, Any]:
        """Generuje hiperparametry z checkpointu, uzupełniając brakujące wartości."""
        required_keys = [
            "hidden_size", "learning_rate", "attention_head_size", "dropout",
            "lstm_layers", "hidden_continuous_size", "output_size",
            "log_interval", "reduce_on_plateau_patience", "embedding_sizes"
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