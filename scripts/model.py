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
logging.basicConfig(level=logging.INFO)
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
        self.save_hyperparameters(ignore=['loss', 'logging_metrics', 'dataset', 'model.loss', 'model.logging_metrics'])
        # Określ, czy używasz QuantileLoss, czy MAE
        use_quantile_loss = config['model'].get('use_quantile_loss', False)
        quantiles = config['model'].get('quantiles', [0.1, 0.5, 0.9]) if use_quantile_loss else None
        # Mapowanie typów straty
        loss_mapping = {
            "MAE": MAE(),
            "QuantileLoss": QuantileLoss(quantiles=quantiles) if use_quantile_loss else MAE(),
        }
        # Parametry modelu
        params = {
            "hidden_size": config['model']['hidden_size'],
            "lstm_layers": config['model']['lstm_layers'],
            "attention_head_size": config['model']['attention_head_size'],
            "dropout": config['model']['dropout'],
            "hidden_continuous_size": min(config['model']['hidden_size'] // 2, 16),
            "output_size": len(quantiles) if use_quantile_loss else 1,  # Liczba kwantyli lub 1 dla MAE
            "loss": loss_mapping.get(config['model'].get('loss', 'MAE'), MAE()),
            "log_interval": 10,
            "reduce_on_plateau_patience": config['training']['early_stopping_patience'],
        }
        if hyperparams:
            params.update(hyperparams)
        logger.info(f"Parametry przekazywane do TemporalFusionTransformer: {params}")
        # Inicjalizacja modelu
        self.model = TemporalFusionTransformer.from_dataset(dataset, **params)
        self.config = config

    def on_fit_start(self):
        """Przenosi model na GPU przed rozpoczęciem treningu."""
        device = self.device  # Urządzenie ustawione przez PyTorch Lightning
        self.model.to(device)

    def forward(self, x):
        # Przenieś wszystkie tensory w x na to samo urządzenie co model
        x = move_to_device(x, self.device)
        # Zwracamy tylko tensor predykcji
        return self.model(x)[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Przenieś dane na to samo urządzenie co model
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)  # Wyciągnij tensor docelowy z krotki
        y_hat = self(x)
        loss = self.model.loss(y_hat, y_target)
        batch_size = x['encoder_cont'].size(0)  # Pobierz batch_size z danych
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Przenieś dane na to samo urządzenie co model
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)  # Wyciągnij tensor docelowy z krotki
        y_hat = self(x)
        loss = self.model.loss(y_hat, y_target)
        batch_size = x['encoder_cont'].size(0)  # Pobierz batch_size z danych
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['model']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.config['training']['early_stopping_patience'], factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def build_model(dataset, config, trial=None):
    if trial:
        hyperparams = {
            "hidden_size": trial.suggest_int("hidden_size", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "attention_head_size": trial.suggest_int("attention_head_size", 1, 8),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 4),
        }
    else:
        hyperparams = {
            "hidden_size": config['model']['hidden_size'],
            "learning_rate": config['model']['learning_rate'],
            "attention_head_size": config['model']['attention_head_size'],
            "dropout": config['model']['dropout'],
            "lstm_layers": config['model']['lstm_layers'],
        }
    return CustomTemporalFusionTransformer(dataset, config, hyperparams)