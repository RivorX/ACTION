import os
import yaml
import asyncio
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.train import train_model
from scripts.config_manager import ConfigManager
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    directories = ['data', 'models', 'config', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

async def start_training(regions: str = 'global', years: int = 3, use_optuna: bool = False, continue_training: bool = True):
    try:
        create_directories()

        # Wczytaj konfigurację
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Walidacja liczby lat
        if years < 3:
            logger.warning(f"Podano {years} lat. Minimalna liczba lat to 3. Ustawiam domyślnie na 3 lata.")
            years = 3
        logger.info(f"Ustawiono liczbę lat danych: {years}")

        # Przetwarzanie wybranych regionów
        regions_list = [r.strip().lower() for r in regions.split(',')]
        valid_regions = ['poland', 'europe', 'usa', 'global', 'all']
        selected_regions = [r for r in regions_list if r in valid_regions]
        if not selected_regions:
            logger.warning("Nieprawidłowe regiony. Domyślnie wybrano 'global'.")
            selected_regions = ['global']

        logger.info(f"Pobieranie danych dla regionów: {', '.join(selected_regions)}...")

        # Wczytaj tickery dla wybranych regionów
        fetcher = DataFetcher(config_manager, years=years)  # Przekazanie years do DataFetcher
        all_tickers = []
        if 'all' in selected_regions:
            with open(config['data']['tickers_file'], 'r') as f:
                tickers_config = yaml.safe_load(f)
            for region in tickers_config['tickers']:
                all_tickers.extend([item['ticker'] for item in tickers_config['tickers'][region]])
        else:
            for region in selected_regions:
                tickers = fetcher._load_tickers(region)
                all_tickers.extend(tickers)
        
        # Usuń duplikaty tickerów
        all_tickers = list(dict.fromkeys(all_tickers))
        logger.info(f"Wybrane tickery: {all_tickers}")

        # Aktualizacja konfiguracji z wybranymi tickerami
        config['data']['tickers'] = all_tickers

        # Pobierz dane asynchronicznie
        df = await fetcher.fetch_global_stocks(region=None)  # region=None, bo tickery są już wybrane
        if df.empty:
            raise ValueError("Nie udało się pobrać danych giełdowych.")

        # Preprocessuj dane
        logger.info("Preprocessing danych...")
        preprocessor = DataPreprocessor(config)
        dataset = preprocessor.preprocess_data(df)

        # Dynamiczne ustawienie ścieżek na podstawie model_name
        model_name = config['model_name']
        config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
        logger.info(f"Ścieżka zapisu modelu: {config['paths']['model_save_path']}")

        # Trenuj model
        logger.info("Trenowanie modelu...")
        train_model(dataset, config, use_optuna=use_optuna, continue_training=continue_training)

        logger.info("Trening zakończony. Uruchom `streamlit run app.py`, aby użyć aplikacji.")
    
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas treningu: {str(e)}")
        raise

if __name__ == "__main__":
    regions = input(f"Wybierz region(y) ({', '.join(['poland', 'europe', 'usa', 'global', 'all'])}, oddziel przecinkami, np. poland,europe) [domyślnie: global]: ").lower() or 'global'
    
    years_input = input("Podaj liczbę lat danych historycznych [minimum: 3, domyślnie: 3]: ").lower() or '3'
    try:
        years = int(years_input)
        if years < 3:
            logger.warning(f"Podano {years} lat. Minimalna liczba lat to 3. Używam domyślnej wartości 3 lata.")
            years = 3
    except ValueError as e:
        logger.error(f"Błąd: {e}. Używam domyślnej wartości 3 lata.")
        years = 3

    use_optuna_input = input("Użyć Optuna do optymalizacji? (tak/nie) [domyślnie: nie]: ").lower() or 'nie'
    use_optuna = use_optuna_input == 'tak'

    continue_training_input = input("Kontynuować trening z checkpointu? (tak/nie) [domyślnie: tak]: ").lower() or 'tak'
    continue_training = continue_training_input != 'nie'

    # Uruchom asynchroniczną funkcję start_training
    asyncio.run(start_training(regions, years, use_optuna, continue_training))