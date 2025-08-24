import os
import yaml
import asyncio
import logging
from pathlib import Path
import torch
import shutil
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.train import train_model
from scripts.model import build_model, CustomTemporalFusionTransformer
from scripts.config_manager import ConfigManager
from scripts.utils.transfer_weights import transfer_weights
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Tworzy wymagane katalogi, jeśli nie istnieją."""
    directories = ['data', 'data/train', 'models', 'models/normalizers', 'config', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Utworzono katalog: {directory}")

async def start_training(regions: str = 'global', years: int = 3, use_optuna: bool = False, continue_training: bool = True, use_transfer_learning: bool = False, old_model_filename: str = None, new_lr: float = None, ticker_percentage: float = 1.0):
    """Uruchamia proces treningu modelu, w tym pobieranie danych, preprocessing, transfer wag i trening."""
    try:
        create_directories()

        # Wczytaj konfigurację
        config_manager = ConfigManager()
        config = config_manager.config
        config['data']['years'] = years
        logger.info(f"Ustawiono liczbę lat danych: {years}")

        # Walidacja regionów
        regions_list = [r.strip().lower() for r in regions.split(',')]
        valid_regions = ['poland', 'europe', 'usa', 'global', 'all']
        selected_regions = [r for r in regions_list if r in valid_regions]
        if not selected_regions:
            logger.warning("Nieprawidłowe regiony. Domyślnie wybrano 'global'.")
            selected_regions = ['global']
        logger.info(f"Pobieranie danych dla regionów: {', '.join(selected_regions)}...")

        # Pobieranie tickerów
        fetcher = DataFetcher(config_manager, years=years)
        all_tickers = []
        tickers_config = None
        with open(config['data']['tickers_file'], 'r') as f:
            tickers_config = yaml.safe_load(f)
        
        if 'all' in selected_regions:
            for region in tickers_config['tickers']:
                region_tickers = [item for item in tickers_config['tickers'][region]]
                num_to_select = max(1, int(len(region_tickers) * ticker_percentage))
                selected = random.sample(region_tickers, num_to_select)
                all_tickers.extend(selected)
        else:
            for region in selected_regions:
                region_tickers = [item for item in tickers_config['tickers'][region]]
                num_to_select = max(1, int(len(region_tickers) * ticker_percentage))
                selected = random.sample(region_tickers, num_to_select)
                all_tickers.extend(selected)
        
        all_tickers = list(dict.fromkeys(all_tickers))  # Usunięcie duplikatów
        logger.info(f"Wybrane tickery (używając {ticker_percentage*100}%): {all_tickers}")
        config['data']['tickers'] = all_tickers

        # Pobieranie danych
        logger.info("Pobieranie danych giełdowych...")
        df = await fetcher.fetch_global_stocks(region=None)
        if df.empty:
            raise ValueError("Nie udało się pobrać danych giełdowych.")

        data_path = Path(config['data']['raw_data_path'])
        df.to_csv(data_path, index=False)
        logger.info(f"Dane zapisane do {data_path}")

        # Preprocessing danych
        logger.info("Preprocessing danych...")
        model_name = config['model_name']
        normalizers_path = Path(config['paths']['models_dir']) / 'normalizers' / f"{model_name}_normalizers.pkl"
        logger.info(f"Ścieżka normalizerów: {normalizers_path}")

        # Jeśli transfer learning, skopiuj stare normalizery do nowej ścieżki
        if use_transfer_learning and old_model_filename:
            old_model_name = old_model_filename.replace('.pth', '')  # Wyciągnij nazwę modelu bez rozszerzenia
            old_normalizers_path = Path(config['paths']['models_dir']) / 'normalizers' / f"{old_model_name}_normalizers.pkl"
            if old_normalizers_path.exists():
                shutil.copy(old_normalizers_path, normalizers_path)
                logger.info(f"Skopiowano stare normalizery z {old_normalizers_path} do {normalizers_path} dla transfer learningu.")
            else:
                logger.warning(f"Stare normalizery {old_normalizers_path} nie istnieją – preprocessing stworzy nowe.")

        preprocessor = DataPreprocessor(config)
        dataset = preprocessor.process_data(mode='train', df=df)
        train_dataset, val_dataset = dataset  # Rozpakowanie krotki

        # Transfer learning
        if use_transfer_learning and not continue_training:
            models_dir = Path(config['paths']['models_dir'])
            old_checkpoint_path = models_dir / old_model_filename
            if not old_checkpoint_path.exists():
                logger.error(f"Plik {old_checkpoint_path} nie istnieje w katalogu {models_dir}.")
                raise FileNotFoundError(f"Plik {old_checkpoint_path} nie istnieje.")

            logger.info("Budowanie modelu dla transfer learningu...")
            new_model = build_model(train_dataset, config)  # Używamy train_dataset
            new_model, config = transfer_weights(
                old_checkpoint_path=old_checkpoint_path,
                new_model=new_model,
                config=config,
                normalizers_path=normalizers_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info(f"Model z przeniesionymi wagami zapisano w: {config['paths']['model_save_path']}")
            # Po transferze, trenuj model
            config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
            final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=True, new_lr=new_lr)
        else:
            # Trening modelu
            logger.info("Trenowanie modelu...")
            config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
            final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=continue_training, new_lr=new_lr)

        logger.info("Trening zakończony. Uruchom `streamlit run app.py`, aby użyć aplikacji.")
        return final_model

    except Exception as e:
        logger.error(f"Wystąpił błąd podczas treningu: {str(e)}")
        raise

if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.config
    model_name = config['model_name']
    model_path = Path(config['paths']['models_dir']) / f"{model_name}.pth"
    
    # Odczytaj aktualne lr (z configu lub checkpointu, jeśli istnieje)
    current_lr = config['model']['learning_rate']
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            current_lr = checkpoint['hyperparams']['learning_rate']
            logger.info(f"Odczytano learning rate z checkpointu: {current_lr}")
        except Exception as e:
            logger.warning(f"Nie można odczytać learning rate z checkpointu: {e}. Używam domyślnego z configu: {current_lr}")

    # Pobieranie danych od użytkownika na początku
    regions = input(f"Wybierz region(y) ({', '.join(['poland', 'europe', 'usa', 'global', 'all'])}, oddziel przecinkami, np. poland,europe) [domyślnie: global]: ").lower() or 'global'
    
    ticker_percentage_input = input("Podaj procent tickerów do użycia (30-100) [domyślnie: 100]: ").lower() or '100'
    try:
        ticker_percentage = float(ticker_percentage_input) / 100
        if ticker_percentage < 0.3 or ticker_percentage > 1.0:
            logger.warning(f"Podano {ticker_percentage*100}%. Wartość musi być między 30-100%. Używam domyślnej wartości 100%.")
            ticker_percentage = 1.0
    except ValueError as e:
        logger.error(f"Błąd: {e}. Używam domyślnej wartości 100%.")
        ticker_percentage = 1.0
    
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

    # Nowa logika: zapytaj o zmianę lr tylko przy kontynuacji treningu
    custom_lr = None
    if continue_training:
        change_lr_input = input(f"Aktualne learning rate to {current_lr}, czy chcesz je zmienić? (tak/nie) [domyślnie: nie]: ").lower() or 'nie'
        if change_lr_input == 'tak':
            new_lr_str = input("Podaj nową wartość learning rate (np. 0.0001): ").strip()
            try:
                custom_lr = float(new_lr_str)
                logger.info(f"Nowa wartość learning rate ustawiona na: {custom_lr}")
            except ValueError:
                logger.error("Nieprawidłowa wartość learning rate. Używam aktualnej.")

    use_transfer_learning = False
    old_model_filename = None
    if not continue_training:
        use_transfer_learning_input = input("Czy użyć transfer learningu z istniejącego modelu? (tak/nie) [domyślnie: nie]: ").lower() or 'nie'
        use_transfer_learning = use_transfer_learning_input == 'tak'
        if use_transfer_learning:
            old_model_filename = input("Podaj nazwę pliku starego modelu z katalogu models (np. model.pth): ").strip()
            if not old_model_filename:
                logger.error("Nie podano nazwy pliku starego modelu.")
                raise ValueError("Nazwa pliku starego modelu nie może być pusta.")

    # Uruchomienie treningu z nowym parametrem custom_lr i ticker_percentage
    asyncio.run(start_training(regions, years, use_optuna, continue_training, use_transfer_learning, old_model_filename, new_lr=custom_lr, ticker_percentage=ticker_percentage))