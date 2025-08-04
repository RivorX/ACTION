import torch
import sys
import logging
import os
from pathlib import Path

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.model import build_model, CustomTemporalFusionTransformer
from scripts.utils.config_manager import ConfigManager

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transfer_weights(old_checkpoint_path: str, new_model: CustomTemporalFusionTransformer, device: str = 'cpu') -> CustomTemporalFusionTransformer:
    """
    Przenosi kompatybilne wagi z checkpointu starego modelu do nowego modelu TFT.

    Args:
        old_checkpoint_path (str): Ścieżka do checkpointu starego modelu (.pth).
        new_model (CustomTemporalFusionTransformer): Nowy model, do którego przenosimy wagi.
        device (str): Urządzenie, na którym operujemy ('cpu' lub 'cuda').

    Returns:
        CustomTemporalFusionTransformer: Nowy model z przeniesionymi wagami.
    """
    # Wczytaj checkpoint starego modelu
    try:
        old_checkpoint = torch.load(old_checkpoint_path, map_location=device)
        old_state_dict = old_checkpoint['state_dict']
        logger.info(f"Wczytano checkpoint starego modelu z: {old_checkpoint_path}")
    except Exception as e:
        logger.error(f"Błąd wczytywania checkpointu: {e}")
        raise

    # Pobierz state_dict nowego modelu
    new_state_dict = new_model.state_dict()

    # Liczniki do obliczania procentu przeniesionych wag
    total_keys = len(new_state_dict)
    transferred_keys = 0

    # Utwórz nowy state_dict z przeniesionymi wagami
    transferred_state_dict = {}

    # Porównaj i przenieś wagi
    for key in new_state_dict.keys():
        if key in old_state_dict:
            # Sprawdź zgodność wymiarów
            if old_state_dict[key].shape == new_state_dict[key].shape:
                transferred_state_dict[key] = old_state_dict[key]
                transferred_keys += 1
            else:
                transferred_state_dict[key] = new_state_dict[key]
        else:
            # Użyj domyślnej inicjalizacji dla brakujących kluczy
            transferred_state_dict[key] = new_state_dict[key]
            logger.info(f"Brak klucza {key} w starym modelu - użyto domyślnej inicjalizacji")

    # Oblicz procent przeniesionych wag
    transfer_percentage = (transferred_keys / total_keys) * 100 if total_keys > 0 else 0
    logger.info(f"Przeniesiono {transferred_keys} z {total_keys} wag ({transfer_percentage:.2f}%)")

    # Załaduj przeniesione wagi do nowego modelu
    try:
        new_model.load_state_dict(transferred_state_dict)
        logger.info("Wagi przeniesione pomyślnie do nowego modelu")
    except Exception as e:
        logger.error(f"Błąd podczas ładowania wag do nowego modelu: {e}")
        raise

    return new_model