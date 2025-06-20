import yaml
import logging
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """Klasa do zarządzania konfiguracją z pliku YAML."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Inicjalizuje menedżera konfiguracji.
        
        Args:
            config_path (str): Ścieżka do pliku konfiguracyjnego.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Wczytuje konfigurację z pliku YAML.
        
        Returns:
            dict: Słownik z konfiguracją.
        
        Raises:
            FileNotFoundError: Jeśli plik konfiguracyjny nie istnieje.
            yaml.YAMLError: Jeśli plik YAML jest nieprawidłowy.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Konfiguracja wczytana z {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Plik konfiguracyjny {self.config_path} nie istnieje")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Błąd parsowania pliku YAML: {e}")
            raise

    def get(self, key: str, default=None):
        """Pobiera wartość z konfiguracji według klucza.
        
        Args:
            key (str): Klucz w formacie 'sekcja.podsekcja'.
            default: Wartość domyślna, jeśli klucz nie istnieje.
        
        Returns:
            Wartość dla podanego klucza lub default, jeśli klucz nie istnieje.
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Klucz {key} nie znaleziony w konfiguracji, zwracam {default}")
            return default