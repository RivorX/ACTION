import pytorch_forecasting
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_lightning import LightningModule
import torch
import logging

# Ustaw precyzję dla Tensor Cores na GPU
torch.set_float32_matmul_precision('medium')

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def move_to_device(obj, device):
    """Rekurencyjnie przenosi tensory na wskazane urządzenie."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(val, device) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    return obj

class CustomTemporalFusionTransformer(LightningModule):
    def __init__(self, dataset, config, hyperparams=None):
        super().__init__()
        # Zapisz hiperparametry, ignorując dataset oraz problematyczne klucze
        self.save_hyperparameters(ignore=['dataset', 'loss'])
        self.config = config
        # Określ, czy używasz QuantileLoss
        use_quantile_loss = config['model'].get('use_quantile_loss', False)
        quantiles = config['model'].get('quantiles', [0.1, 0.5, 0.9]) if use_quantile_loss else None
        
        # Parametry modelu
        params = hyperparams if hyperparams is not None else {
            "hidden_size": config['model']['hidden_size'],
            "lstm_layers": config['model']['lstm_layers'],
            "attention_head_size": config['model']['attention_head_size'],
            "dropout": config['model']['dropout'],
            "hidden_continuous_size": config['model']['hidden_size'] // 2,
            "output_size": len(quantiles) if use_quantile_loss else 1,
            "loss": QuantileLoss(quantiles=quantiles) if use_quantile_loss else MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": config['training']['early_stopping_patience'],
            "learning_rate": config['model']['learning_rate']
        }
        
        # Filtruj parametry
        valid_keys = [
            "hidden_size", "lstm_layers", "attention_head_size", "dropout", "hidden_continuous_size",
            "output_size", "loss", "log_interval", "reduce_on_plateau_patience", "learning_rate"
        ]
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        logger.info(f"Parametry przekazywane do TemporalFusionTransformer: {filtered_params}")
        
        # Inicjalizacja modelu
        self.model = TemporalFusionTransformer.from_dataset(dataset, **filtered_params)

    def on_fit_start(self):
        """Przenosi model na GPU przed rozpoczęciem treningu."""
        device = self.device
        self.model.to(device)

    def forward(self, x):
        x = move_to_device(x, self.device)
        return self.model(x)[0]

    def predict(self, data, **kwargs):
        """Deleguje predykcję do wewnętrznego modelu TemporalFusionTransformer."""
        predictions = self.model.predict(data, **kwargs)
        logger.info(f"Kształt zwracanych predykcji: {predictions.output.shape}")
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)
        # Użyj AMP z bf16, jeśli dostępne
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
            y_hat = self(x)
            loss = self.model.loss(y_hat, y_target)
        batch_size = x['encoder_cont'].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)
        # Użyj AMP z bf16, jeśli dostępne
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
            y_hat = self(x)
            loss = self.model.loss(y_hat, y_target)
        batch_size = x['encoder_cont'].size(0)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Logowanie dla batchu walidacyjnego
        if batch_idx % 50 == 0:
            logger.info(f"Validation batch {batch_idx}: y_hat[0, :5] = {y_hat[0, :5].tolist()}, y_target[0, :5] = {y_target[0, :5].tolist()}")
        
        return loss

    def configure_optimizers(self):
        learning_rate = self.hparams.get('learning_rate', self.config['model']['learning_rate'])
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config['training']['reduce_lr_patience'],
            factor=self.config['training']['reduce_lr_factor'],
            mode='min'
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def build_model(dataset, config, trial=None, hyperparams=None):
    if trial:
        hyperparams = {
            "hidden_size": trial.suggest_int("hidden_size", config['model']['min_hidden_size'], config['model']['max_hidden_size']),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "attention_head_size": trial.suggest_int("attention_head_size", config['model']['min_attention_head_size'], config['model']['max_attention_head_size']),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lstm_layers": trial.suggest_int("lstm_layers", config['model']['min_lstm_layers'], config['model']['max_lstm_layers']),
            "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 8, config['model']['max_hidden_size']),
            "output_size": len(config['model'].get('quantiles', [0.1, 0.5, 0.9])) if config['model'].get('use_quantile_loss', False) else 1,
            "loss": QuantileLoss(quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9])) if config['model'].get('use_quantile_loss', False) else MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": config['training']['early_stopping_patience']
        }
    elif hyperparams:
        # Użyj hiperparametrów z checkpointu
        required_keys = [
            "hidden_size", "learning_rate", "attention_head_size", "dropout",
            "lstm_layers", "hidden_continuous_size", "output_size", "loss",
            "log_interval", "reduce_on_plateau_patience"
        ]
        filtered_hyperparams = {}
        for key in required_keys:
            if key in hyperparams:
                filtered_hyperparams[key] = hyperparams[key]
            else:
                logger.warning(f"Brak klucza {key} w hiperparametrach, używam wartości domyślnej")
                if key == "output_size":
                    filtered_hyperparams[key] = len(config['model'].get('quantiles', [0.1, 0.5, 0.9])) if config['model'].get('use_quantile_loss', False) else 1
                elif key == "loss":
                    filtered_hyperparams[key] = QuantileLoss(quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9])) if config['model'].get('use_quantile_loss', False) else MAE()
                elif key == "log_interval":
                    filtered_hyperparams[key] = 10
                elif key == "reduce_on_plateau_patience":
                    filtered_hyperparams[key] = config['training']['early_stopping_patience']
                else:
                    filtered_hyperparams[key] = config['model'].get(key, 0.01 if key == "learning_rate" else 1)
        hyperparams = filtered_hyperparams
    else:
        # Domyślne hiperparametry z config.yaml
        hyperparams = {
            "hidden_size": config['model']['hidden_size'],
            "learning_rate": config['model']['learning_rate'],
            "attention_head_size": config['model']['attention_head_size'],
            "dropout": config['model']['dropout'],
            "lstm_layers": config['model']['lstm_layers'],
            "hidden_continuous_size": config['model']['hidden_size'] // 2,
            "output_size": len(config['model'].get('quantiles', [0.1, 0.5, 0.9])) if config['model'].get('use_quantile_loss', False) else 1,
            "loss": QuantileLoss(quantiles=config['model'].get('quantiles', [0.1, 0.5, 0.9])) if config['model'].get('use_quantile_loss', False) else MAE(),
            "log_interval": 10,
            "reduce_on_plateau_patience": config['training']['early_stopping_patience']
        }
    logger.info(f"Budowanie modelu z hiperparametrami: {hyperparams}")
    return CustomTemporalFusionTransformer(dataset, config, hyperparams)