import os
import yaml
import asyncio
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.train import train_model
from scripts.model import build_model
from scripts.utils.config_manager import ConfigManager
from scripts.utils.transfer_weights import transfer_weights
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    directories = ['data', 'models', 'config', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

async def start_training(regions: str = 'global', years: int = 3, use_optuna: bool = False, continue_training: bool = True, new_learning_rate: float = None):
    try:
        create_directories()

        config_manager = ConfigManager()
        config = config_manager.config
        
        if years < 3:
            logger.warning(f"Podano {years} lat. Minimalna liczba lat to 3. Ustawiam domyślnie na 3 lata.")
            years = 3
        logger.info(f"Ustawiono liczbę lat danych: {years}")

        regions_list = [r.strip().lower() for r in regions.split(',')]
        valid_regions = ['poland', 'europe', 'usa', 'global', 'all']
        selected_regions = [r for r in regions_list if r in valid_regions]
        if not selected_regions:
            logger.warning("Nieprawidłowe regiony. Domyślnie wybrano 'global'.")
            selected_regions = ['global']

        logger.info(f"Pobieranie danych dla regionów: {', '.join(selected_regions)}...")

        fetcher = DataFetcher(config_manager, years=years)
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
        
        all_tickers = list(dict.fromkeys(all_tickers))
        logger.info(f"Wybrane tickery: {all_tickers}")

        config['data']['tickers'] = all_tickers

        if continue_training and new_learning_rate is not None:
            config['model']['learning_rate'] = new_learning_rate
            logger.info(f"Zaktualizowano learning rate na: {new_learning_rate}")

        df = await fetcher.fetch_global_stocks(region=None)
        if df.empty:
            raise ValueError("Nie udało się pobrać danych giełdowych.")

        logger.info("Preprocessing danych...")
        preprocessor = DataPreprocessor(config)
        dataset = preprocessor.preprocess_data(df)

        model_name = config['model_name']
        config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
        logger.info(f"Ścieżka zapisu modelu: {config['paths']['model_save_path']}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not continue_training:
            use_transfer_learning = input("Czy użyć transfer learningu z istniejącego modelu? (tak/nie) [domyślnie: nie]: ").lower() or 'nie'
            if use_transfer_learning == 'tak':
                old_model_filename = input("Podaj nazwę pliku starego modelu z katalogu models (np. Gen_4_1_mini.pth): ").strip()
                if not old_model_filename:
                    logger.error("Nie podano nazwy pliku starego modelu.")
                    raise ValueError("Nazwa pliku starego modelu nie może być pusta.")
                
                models_dir = Path(config['paths']['models_dir'])
                old_checkpoint_path = models_dir / old_model_filename

                if not old_checkpoint_path.exists():
                    logger.error(f"Plik {old_checkpoint_path} nie istnieje w katalogu {models_dir}.")
                    raise FileNotFoundError(f"Plik {old_checkpoint_path} nie istnieje.")

                logger.info("Budowanie modelu dla transfer learningu...")
                new_model = build_model(dataset, config)
                new_model = transfer_weights(old_checkpoint_path, new_model, device)
                logger.info("Wagi przeniesione pomyślnie, zapis modelu przed treningiem...")
                
                checkpoint = {
                    'state_dict': new_model.state_dict(),
                    'hyperparams': dict(new_model.hparams)
                }
                torch.save(checkpoint, config['paths']['model_save_path'])
                logger.info(f"Model z przeniesionymi wagami zapisano w: {config['paths']['model_save_path']}")

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

    new_learning_rate = None
    if continue_training:
        reduce_lr_input = input("Czy obniżyć learning rate? (tak/nie) [domyślnie: nie]: ").lower() or 'nie'
        if reduce_lr_input == 'tak':
            lr_input = input("Podaj nową wartość learning rate (aktualnie: 0.001): ")
            try:
                new_learning_rate = float(lr_input)
                if new_learning_rate <= 0:
                    logger.error("Learning rate musi być większy od 0. Używam domyślnego learning rate.")
                    new_learning_rate = None
                else:
                    logger.info(f"Nowa wartość learning rate: {new_learning_rate}")
            except ValueError as e:
                logger.error(f"Błąd: {e}. Używam domyślnego learning rate.")
                new_learning_rate = None

    asyncio.run(start_training(regions, years, use_optuna, continue_training, new_learning_rate))