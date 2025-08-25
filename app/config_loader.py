import yaml
import logging
from scripts.config_manager import ConfigManager

logger = logging.getLogger(__name__)

def load_config():
    """Wczytuje konfigurację z pliku YAML za pomocą Singleton."""
    return ConfigManager().config

def load_tickers_and_names(config):
    """Wczytuje tickery i nazwy spółek z pliku YAML."""
    try:
        with open(config['data']['tickers_file'], 'r') as f:
            tickers_config = yaml.safe_load(f)
        ticker_dict = {}
        for region in tickers_config['tickers']:
            for ticker, name in tickers_config['tickers'][region].items():
                ticker_dict[ticker] = name
        return ticker_dict
    except Exception as e:
        logger.error(f"Błąd wczytywania tickerów i nazw: {e}")
        return {}

def load_benchmark_tickers(config):
    """Wczytuje tickery benchmarkowe z pliku konfiguracyjnego."""
    try:
        benchmark_tickers_file = config['data']['benchmark_tickers_file']
        with open(benchmark_tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(list(region.keys()))  # Użyj kluczy (tickery) zamiast wartości
            return list(dict.fromkeys(all_tickers))  # Usuń duplikaty
    except KeyError as e:
        logger.error(f"Brak klucza w konfiguracji: {e}")
        raise ValueError(f"Błąd konfiguracji: brak klucza {e} w config.yaml")
    except Exception as e:
        logger.error(f"Błąd wczytywania benchmark_tickers.yaml: {e}")
        raise

def save_history_range(config, history_range_days):
    """Zapisuje wybrany zakres historycznych danych do pliku konfiguracyjnego."""
    try:
        config_manager = ConfigManager()
        config_manager.config['prediction']['history_range_days'] = history_range_days
        with open(config_manager.config_file, 'w') as f:
            yaml.safe_dump(config_manager.config, f)
        logger.info(f"Zaktualizowano history_range_days do {history_range_days} w pliku konfiguracyjnym.")
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania history_range_days: {e}")
        raise