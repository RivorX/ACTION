import yaml
import logging
from scripts.config_manager import ConfigManager

logger = logging.getLogger(__name__)

def load_config():
    """Loads configuration from YAML file."""
    try:
        return ConfigManager().config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        raise

def load_tickers_and_names(config):
    """Loads tickers and company names from configuration files."""
    try:
        # Pobierz ścieżki bezpośrednio z config, bez domyślnych wartości
        tickers_file = config['data']['tickers_file']
        company_names_file = config['data']['company_names_file']
        
        # Wczytaj tickery
        with open(tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            all_tickers = list(dict.fromkeys(all_tickers))  # Usuwa duplikaty

        # Wczytaj nazwy spółek
        with open(company_names_file, 'r') as f:
            company_names = yaml.safe_load(f)['company_names']
        
        ticker_options = {ticker: f"{ticker} - {company_names.get(ticker, 'Nieznana firma')}" for ticker in all_tickers}
        return ticker_options
    except KeyError as e:
        logger.error(f"Missing key in config: {e}")
        raise ValueError(f"Configuration error: missing key {e} in config.yaml")
    except Exception as e:
        logger.error(f"Error loading tickers or company names: {e}")
        raise

def load_benchmark_tickers(config):
    """Loads benchmark tickers from configuration file."""
    try:
        # Pobierz ścieżkę bezpośrednio z config, bez domyślnych wartości
        benchmark_tickers_file = config['data']['benchmark_tickers_file']
        with open(benchmark_tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            return list(dict.fromkeys(all_tickers))  # Usuwa duplikaty
    except KeyError as e:
        logger.error(f"Missing key in config: {e}")
        raise ValueError(f"Configuration error: missing key {e} in config.yaml")
    except Exception as e:
        logger.error(f"Error loading benchmark_tickers.yaml: {e}")
        raise