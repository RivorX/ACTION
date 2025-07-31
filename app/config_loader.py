import yaml
import logging
from scripts.config_manager import ConfigManager

logger = logging.getLogger(__name__)

def load_config():
    """Loads configuration from YAML file using Singleton."""
    return ConfigManager().config

def load_tickers_and_names(config):
    """Ładuje tickery i nazwy spółek z pliku YAML."""
    try:
        with open(config['data']['tickers_file'], 'r') as f:
            tickers_config = yaml.safe_load(f)
        ticker_dict = {}
        for region in tickers_config['tickers']:
            for item in tickers_config['tickers'][region]:
                ticker_dict[item['ticker']] = item['name']
        return ticker_dict
    except Exception as e:
        logger.error(f"Błąd wczytywania tickerów i nazw: {e}")
        return {}

def load_benchmark_tickers(config):
    """Loads benchmark tickers from configuration file."""
    try:
        benchmark_tickers_file = config['data']['benchmark_tickers_file']
        with open(benchmark_tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            return list(dict.fromkeys(all_tickers))
    except KeyError as e:
        logger.error(f"Missing key in config: {e}")
        raise ValueError(f"Configuration error: missing key {e} in config.yaml")
    except Exception as e:
        logger.error(f"Error loading benchmark_tickers.yaml: {e}")
        raise