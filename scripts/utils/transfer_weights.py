import torch
import sys
import logging
import os
from pathlib import Path
import shutil

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.model import CustomTemporalFusionTransformer

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transfer_weights(old_checkpoint_path: str, new_model: CustomTemporalFusionTransformer, config: dict, normalizers_path: Path, device: str = 'cpu') -> tuple[CustomTemporalFusionTransformer, dict]:
    """
    Przenosi kompatybilne wagi z checkpointu starego modelu do nowego modelu TFT oraz kopiuje normalizery starego modelu.

    Args:
        old_checkpoint_path (str): Ścieżka do checkpointu starego modelu (.pth).
        new_model (CustomTemporalFusionTransformer): Nowy model, do którego przenosimy wagi.
        config (dict): Konfiguracja modelu.
        normalizers_path (Path): Ścieżka do zapisu nowych normalizerów.
        device (str): Urządzenie, na którym operujemy ('cpu' lub 'cuda').

    Returns:
        tuple[CustomTemporalFusionTransformer, dict]: Nowy model z przeniesionymi wagami oraz zaktualizowana konfiguracja.
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
                logger.warning(f"Pominięto {key} - niezgodność wymiarów: "
                              f"stary={old_state_dict[key].shape}, nowy={new_state_dict[key].shape}")
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

    # Transfer normalizerów
    models_dir = Path(config['paths']['models_dir'])
    old_normalizers_path = models_dir / 'normalizers' / f"{Path(old_checkpoint_path).stem}_normalizers.pkl"

    if old_normalizers_path.exists():
        if not normalizers_path.exists():
            shutil.copy(old_normalizers_path, normalizers_path)
            logger.info(f"Skopiowano normalizery z {old_normalizers_path} do {normalizers_path}")
        else:
            logger.info(f"Normalizery w {normalizers_path} już istnieją, nadpisywanie.")
            shutil.copy(old_normalizers_path, normalizers_path)
            logger.info(f"Nadpisano normalizery w {normalizers_path} z {old_normalizers_path}")
    else:
        logger.error(f"Plik normalizerów {old_normalizers_path} nie istnieje. Brak normalizerów do skopiowania.")
        raise FileNotFoundError(f"Plik normalizerów {old_normalizers_path} nie istnieje.")

    # Zapis modelu z przeniesionymi wagami
    config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{config['model_name']}.pth")
    checkpoint = {
        'state_dict': new_model.state_dict(),
        'hyperparams': dict(new_model.hparams)
    }
    torch.save(checkpoint, config['paths']['model_save_path'])
    logger.info(f"Model z przeniesionymi wagami zapisano w: {config['paths']['model_save_path']}")

    return new_model, config