import torch
import pandas as pd
import logging
from pytorch_forecasting import TimeSeriesDataSet
from pathlib import Path
import numpy as np

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lista wszystkich możliwych sektorów
ALL_SECTORS = [
    'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 'Consumer Staples',
    'Energy', 'Utilities', 'Industrials', 'Materials', 'Communication Services',
    'Real Estate', 'Unknown'
]

def debug_dataset(dataset_path: str = "data/processed_dataset.pt"):
    """
    Skrypt debugujący dla pliku processed_dataset.pt.
    Sprawdza zawartość datasetu, jego atrybuty oraz szczegóły zmiennych statycznych i kategorycznych.
    """
    try:
        # Wczytaj dataset
        logger.info(f"Wczytywanie datasetu z: {dataset_path}")
        dataset = torch.load(dataset_path, weights_only=False, map_location=torch.device('cpu'))
        logger.info("Dataset wczytany poprawnie.")

        # Sprawdź, czy dataset jest instancją TimeSeriesDataSet
        if not isinstance(dataset, TimeSeriesDataSet):
            logger.error(f"Dataset nie jest instancją TimeSeriesDataSet, tylko: {type(dataset)}")
            return

        # Wyświetl podstawowe informacje o dataset
        logger.info(f"Parametry datasetu: {dataset.get_parameters()}")

        # Sprawdź atrybuty datasetu
        attributes_to_check = [
            'reals', 'categoricals', 'static_categoricals', 'static_reals',
            'time_varying_known_reals', 'time_varying_known_categoricals',
            'time_varying_unknown_reals', 'target', 'group_ids', 'time_idx'
        ]

        for attr in attributes_to_check:
            if hasattr(dataset, attr):
                value = getattr(dataset, attr)
                logger.info(f"Atrybut {attr}: {value}")
            else:
                logger.warning(f"Atrybut {attr} nie istnieje w dataset.")

        # Sprawdź normalizer celu
        if hasattr(dataset, 'target_normalizer'):
            logger.info(f"Target normalizer: {dataset.target_normalizer}")
            if hasattr(dataset.target_normalizer, 'get_parameters'):
                logger.info(f"Parametry normalizera celu: {dataset.target_normalizer.get_parameters()}")
        else:
            logger.warning("Brak target_normalizer w dataset.")

        # Sprawdź szczegóły zmiennych statycznych
        if hasattr(dataset, 'static_categoricals') and dataset.static_categoricals:
            logger.info(f"Statyczne zmienne kategoryczne: {dataset.static_categoricals}")
            for cat in dataset.static_categoricals:
                try:
                    categories = dataset.categorical_encoders[cat].classes_
                    logger.info(f"Kategorie dla {cat}: {categories}")
                    if cat == 'Sector':
                        missing_sectors = set(ALL_SECTORS) - set(categories)
                        if missing_sectors:
                            logger.warning(f"Brakujące sektory w {cat}: {missing_sectors}")
                        else:
                            logger.info(f"Wszystkie sektory z ALL_SECTORS są obecne w {cat}")
                except Exception as e:
                    logger.warning(f"Nie udało się pobrać kategorii dla {cat}: {e}")
        else:
            logger.warning("Brak statycznych zmiennych kategorycznych w dataset.")

        if hasattr(dataset, 'static_reals') and dataset.static_reals:
            logger.info(f"Statyczne zmienne rzeczywiste: {dataset.static_reals}")
        else:
            logger.warning("Brak statycznych zmiennych rzeczywistych w dataset.")

        # Sprawdź szczegóły zmiennych kategorycznych
        if hasattr(dataset, 'categoricals') and dataset.categoricals:
            logger.info(f"Zmienne kategoryczne: {dataset.categoricals}")
            for cat in dataset.categoricals:
                try:
                    categories = dataset.categorical_encoders[cat].classes_
                    logger.info(f"Kategorie dla {cat}: {categories}")
                    if cat == 'Day_of_Week':
                        expected_categories = [str(i) for i in range(7)]
                        missing_categories = set(expected_categories) - set(categories)
                        if missing_categories:
                            logger.warning(f"Brakujące kategorie w {cat}: {missing_categories}")
                        else:
                            logger.info(f"Wszystkie kategorie dla {cat} są obecne")
                except Exception as e:
                    logger.warning(f"Nie udało się pobrać kategorii dla {cat}: {e}")
        else:
            logger.warning("Brak zmiennych kategorycznych w dataset.")

        # Sprawdź szczegóły zmiennych rzeczywistych
        if hasattr(dataset, 'reals') and dataset.reals:
            logger.info(f"Zmienne rzeczywiste (reals): {dataset.reals}")
        else:
            logger.warning("Brak zmiennych rzeczywistych (reals) w dataset.")

        # Sprawdź, czy dataset zawiera dane
        try:
            dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
            x, y = next(iter(dataloader))
            logger.info(f"Przykładowy batch: x.keys={list(x.keys())}, y.shape={y[0].shape}")
            
            # Sprawdź szczegóły batcha
            for key, value in x.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"Tensor {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        logger.warning(f"Tensor {key} zawiera NaN lub inf")
                else:
                    logger.info(f"Klucz {key}: wartość={value[:5] if isinstance(value, list) else value}")
            
            # Sprawdź wartości statycznych zmiennych kategorycznych w batchu
            if 'static_categoricals' in x:
                logger.info(f"Statyczne zmienne kategoryczne w batchu: {x['static_categoricals']}")
                if 'Sector' in dataset.static_categoricals:
                    sector_values = x['static_categoricals'][:, dataset.static_categoricals.index('Sector')]
                    unique_sectors = np.unique(sector_values)
                    logger.info(f"Unikalne wartości Sector w batchu: {unique_sectors}")
                    decoded_sectors = dataset.categorical_encoders['Sector'].inverse_transform(unique_sectors)
                    logger.info(f"Zdekodowane wartości Sector: {decoded_sectors}")
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia dataloadera lub pobierania batcha: {e}")

        # Sprawdź statystyki zmiennych rzeczywistych w batchu
        if 'encoder_cont' in x:
            encoder_cont = x['encoder_cont']
            for i, feature in enumerate(dataset.reals):
                feature_values = encoder_cont[:, :, i]
                logger.info(f"Statystyki dla {feature}: min={feature_values.min().item():.4f}, max={feature_values.max().item():.4f}, mean={feature_values.mean().item():.4f}")

    except Exception as e:
        logger.error(f"Błąd podczas wczytywania datasetu: {e}")
        raise

if __name__ == "__main__":
    dataset_path = "data/processed_dataset.pt"
    logger.info("Uruchamianie skryptu debugującego...")
    debug_dataset(dataset_path)
    logger.info("Debugowanie zakończone.")