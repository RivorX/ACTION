import yaml
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None

    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config_path = Path(config_path)
            cls._instance.config = cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> dict:
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

    def get(self, key: str):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.error(f"Klucz {key} nie znaleziony w konfiguracji")
            raise KeyError(f"Klucz {key} nie znaleziony w konfiguracji")

    def load_normalizers(self, model_name: str) -> dict:
        normalizers_path = Path(self.get('paths.normalizers_dir')) / f"{model_name}_normalizers.pkl"
        if normalizers_path.exists():
            try:
                with open(normalizers_path, "rb") as f:
                    normalizers = pickle.load(f)
                logger.info(f"Normalizery wczytane z {normalizers_path}")
                return normalizers
            except Exception as e:
                logger.error(f"Błąd podczas ładowania normalizerów: {e}")
                return {}
        else:
            logger.warning(f"Plik normalizerów {normalizers_path} nie istnieje")
            return {}

    def save_normalizers(self, model_name: str, normalizers: dict):
        normalizers_path = Path(self.get('paths.normalizers_dir')) / f"{model_name}_normalizers.pkl"
        normalizers_path.parent.mkdir(parents=True, exist_ok=True)
        if normalizers_path.exists():
            logger.warning(f"Normalizery dla modelu {model_name} już istnieją w {normalizers_path}. Nie zapisuję ponownie.")
            return
        try:
            with open(normalizers_path, "wb") as f:
                pickle.dump(normalizers, f)
            logger.info(f"Normalizery zapisane do {normalizers_path}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisu normalizerów: {e}")