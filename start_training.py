import os
import yaml
import asyncio
import logging
from pathlib import Path
import shutil
import torch
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.train import train_model
from scripts.model import build_model
from scripts.utils.config_manager import ConfigManager
from scripts.utils.transfer_weights import transfer_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Tworzy wymagane katalogi, jeśli nie istnieją."""
    directories = ['data', 'models', 'config', 'logs', 'models/normalizers']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

async def start_training(regions: str = 'global', years: int = 3, use_optuna: bool = False, continue_training: bool = True, new_learning_rate: float = None):
    """Uruchamia proces treningu modelu, w tym pobieranie danych, preprocessing i trening."""
    try:
        create_directories()

        # Inicjalizacja konfiguracji
        config_manager = ConfigManager()
        config = config_manager.config
        config['data']['years'] = years
        logger.info(f"Ustawiono liczbę lat danych: {years}")

        # Pytanie o transfer learning tylko jeśli nie kontynuujemy treningu
        use_transfer_learning = False
        old_model_filename = None
        if not continue_training:
            use_transfer_learning = input("Czy użyć transfer learningu z istniejącego modelu? (tak/nie) [domyślnie: nie]: ").lower() == 'tak'
            if use_transfer_learning:
                old_model_filename = input("Podaj nazwę pliku starego modelu z katalogu models (np. Gen4_1_mini.pth): ").strip()
                if not old_model_filename:
                    logger.error("Nie podano nazwy pliku starego modelu.")
                    raise ValueError("Nazwa pliku starego modelu nie może być pusta.")

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
        if 'all' in selected_regions:
            with open(config['data']['tickers_file'], 'r') as f:
                tickers_config = yaml.safe_load(f)
            for region in tickers_config['tickers']:
                all_tickers.extend([item['ticker'] for item in tickers_config['tickers'][region]])
        else:
            for region in selected_regions:
                tickers = fetcher._load_tickers(region)
                all_tickers.extend(tickers)
        
        all_tickers = list(dict.fromkeys(all_tickers))  # Usunięcie duplikatów
        logger.info(f"Wybrane tickery: {all_tickers}")
        config['data']['tickers'] = all_tickers

        # Aktualizacja learning rate, jeśli podano
        if continue_training and new_learning_rate is not None:
            config['model']['learning_rate'] = new_learning_rate
            logger.info(f"Zaktualizowano learning rate na: {new_learning_rate}")

        # Pobieranie danych
        df = await fetcher.fetch_global_stocks(region=None)
        if df.empty:
            raise ValueError("Nie udało się pobrać danych giełdowych.")

        # Preprocessing danych
        logger.info("Preprocessing danych...")
        model_name = config['model_name']
        normalizers_path = Path(config['paths']['models_dir']) / 'normalizers' / f"{model_name}_normalizers.pkl"
        config['data']['normalizers_path'] = str(normalizers_path)
        logger.info(f"Ścieżka normalizerów: {normalizers_path}")

        preprocessor = DataPreprocessor(config)
        dataset = preprocessor.preprocess_data(df)

        # Transfer learning
        if use_transfer_learning:
            models_dir = Path(config['paths']['models_dir'])
            old_checkpoint_path = models_dir / old_model_filename
            old_normalizers_path = models_dir / 'normalizers' / f"{old_model_filename.replace('.pth', '')}_normalizers.pkl"

            if not old_checkpoint_path.exists():
                logger.error(f"Plik {old_checkpoint_path} nie istnieje w katalogu {models_dir}.")
                raise FileNotFoundError(f"Plik {old_checkpoint_path} nie istnieje.")
            
            if old_normalizers_path.exists():
                logger.warning(
                    f"Plik normalizerów {old_normalizers_path} istnieje. "
                    f"Kopiowanie starych normalizerów może spowodować niezgodność z nową metodą normalizacji (np. robust dla RSI, Stochastic_K, Stochastic_D). "
                    f"Zaleca się wygenerowanie nowych normalizerów dla spójności. Czy kontynuować z kopiowaniem? (tak/nie) [domyślnie: nie]: "
                )
                copy_normalizers = input().lower() == 'tak'
                if copy_normalizers and not normalizers_path.exists():
                    shutil.copy(old_normalizers_path, normalizers_path)
                    logger.info(f"Skopiowano normalizery z {old_normalizers_path} do {normalizers_path}")
                else:
                    logger.info(f"Pominięto kopiowanie normalizerów. Nowe normalizery zostaną wygenerowane podczas preprocessingu.")
            else:
                logger.info(f"Plik normalizerów {old_normalizers_path} nie istnieje. Nowe normalizery zostaną wygenerowane.")

            logger.info("Budowanie modelu dla transfer learningu...")
            new_model = build_model(dataset, config)
            new_model = transfer_weights(
                old_checkpoint_path=old_checkpoint_path,
                new_model=new_model,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info("Wagi przeniesione pomyślnie, zapis modelu przed treningiem...")
            
            config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
            checkpoint = {
                'state_dict': new_model.state_dict(),
                'hyperparams': dict(new_model.hparams)
            }
            torch.save(checkpoint, config['paths']['model_save_path'])
            logger.info(f"Model z przeniesionymi wagami zapisano w: {config['paths']['model_save_path']}")

        # Trening modelu
        logger.info("Trenowanie modelu...")
        config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
        final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=continue_training)
        logger.info("Trening zakończony. Uruchom `streamlit run app.py`, aby użyć aplikacji.")
        return final_model

    except Exception as e:
        logger.error(f"Wystąpił błąd podczas treningu: {str(e)}")
        raise

if __name__ == "__main__":
    # Pobieranie danych od użytkownika
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

    # Uruchomienie treningu
    asyncio.run(start_training(regions, years, use_optuna, continue_training, new_learning_rate))