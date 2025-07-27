import yaml
import logging
from scripts.config_manager import ConfigManager

# Konfiguracja logowania
logger = logging.getLogger(__name__)

def load_config():
    """Wczytuje konfigurację z pliku YAML."""
    try:
        return ConfigManager().config
    except Exception as e:
        logger.error(f"Błąd wczytywania config.yaml: {e}")
        raise

def load_tickers_and_names():
    """Wczytuje listę tickerów i pełnych nazw firm z plików konfiguracyjnych."""
    try:
        config = ConfigManager()
        tickers_file = config.get('data.tickers_file', 'config/training_tickers.yaml')
        company_names_file = config.get('data.company_names_file', 'config/company_names.yaml')
        
        with open(tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            all_tickers = list(dict.fromkeys(all_tickers))  # Usuwa duplikaty

        with open(company_names_file, 'r') as f:
            company_names = yaml.safe_load(f)['company_names']
        
        ticker_options = {ticker: f"{ticker} - {company_names.get(ticker, 'Nieznana firma')}" for ticker in all_tickers}
        return ticker_options
    except Exception as e:
        logger.error(f"Błąd wczytywania tickerów lub nazw firm: {e}")
        return {}

def load_benchmark_tickers():
    """Wczytuje listę tickerów dla benchmarku z pliku YAML."""
    try:
        config = ConfigManager()
        benchmark_tickers_file = 'config/benchmark_tickers.yaml'
        with open(benchmark_tickers_file, 'r') as f:
            tickers_config = yaml.safe_load(f)
            all_tickers = []
            for region in tickers_config['tickers'].values():
                all_tickers.extend(region)
            return list(dict.fromkeys(all_tickers))  # Usuwa duplikaty
    except Exception as e:
        logger.error(f"Błąd wczytywania benchmark_tickers.yaml: {e}")
        return []