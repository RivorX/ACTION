import torch
import sys
import logging
import os

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.model import build_model, CustomTemporalFusionTransformer
from scripts.config_manager import ConfigManager


# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Ścieżka do checkpointu starego modelu
OLD_CHECKPOINT_PATH = 'models/Gen_4_1_mini_checkpoint.pth'

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

    # Utwórz nowy state_dict z przeniesionymi wagami
    transferred_state_dict = {}

    # Porównaj i przenieś wagi
    for key in new_state_dict.keys():
        if key in old_state_dict:
            # Sprawdź zgodność wymiarów
            if old_state_dict[key].shape == new_state_dict[key].shape:
                transferred_state_dict[key] = old_state_dict[key]
                logger.info(f"Przeniesiono wagę dla klucza: {key}")
            else:
                transferred_state_dict[key] = new_state_dict[key]
                logger.warning(f"Pominięto {key} - niezgodność wymiarów: "
                              f"stary={old_state_dict[key].shape}, nowy={new_state_dict[key].shape}")
        else:
            # Użyj domyślnej inicjalizacji dla brakujących kluczy
            transferred_state_dict[key] = new_state_dict[key]
            logger.info(f"Brak klucza {key} w starym modelu - użyto domyślnej inicjalizacji")

    # Załaduj przeniesione wagi do nowego modelu
    try:
        new_model.load_state_dict(transferred_state_dict)
        logger.info("Wagi przeniesione pomyślnie do nowego modelu")
    except Exception as e:
        logger.error(f"Błąd podczas ładowania wag do nowego modelu: {e}")
        raise

    return new_model

def main():
    # Inicjalizacja konfiguracji
    config_manager = ConfigManager()
    config = config_manager.config

    # Ustal urządzenie
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Używane urządzenie: {device}")

    # Wczytaj dataset
    try:
        dataset = torch.load(config['data']['processed_data_path'], map_location=device, weights_only=False)
        logger.info(f"Wczytano dataset z: {config['data']['processed_data_path']}")
    except Exception as e:
        logger.error(f"Błąd wczytywania datasetu: {e}")
        raise

    # Zbuduj nowy model
    new_model = build_model(dataset, config)
    new_model.to(device)
    logger.info("Zbudowano nowy model TemporalFusionTransformer")

    # Przenieś wagi
    new_model = transfer_weights(OLD_CHECKPOINT_PATH, new_model, device)

    # Zapisz nowy model z przeniesionymi wagami
    output_path = f"models/{config['model_name']}_transferred.pth"
    try:
        torch.save({
            'state_dict': new_model.state_dict(),
            'hyperparams': dict(new_model.hparams)
        }, output_path)
        logger.info(f"Nowy model zapisano w: {output_path}")
    except Exception as e:
        logger.error(f"Błąd zapisu modelu: {e}")
        raise

if __name__ == "__main__":
    main()