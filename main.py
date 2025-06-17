import os
import yaml
from scripts.data_fetcher import fetch_global_stocks
from scripts.preprocessor import preprocess_data
from scripts.train import train_model

# Przełącznik do włączania/wyłączania optymalizacji hiperparametrów z Optuna
optime = False  # Ustaw na True, aby użyć Optuna, False, aby pominąć Optuna i użyć domyślnych hiperparametrów
continue_training = True  # Ustaw na True, aby kontynuować trening z istniejącego checkpointu, False, aby trenować od zera

def create_directories():
    directories = ['data', 'models', 'config']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def main_pipeline():
    create_directories()

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Step 1: Fetch data
    print("Fetching data...")
    df = fetch_global_stocks(config)

    # Step 2: Preprocess data
    print("Preprocessing data...")
    dataset = preprocess_data(config)

    # Step 3: Train model
    print("Training model...")
    train_model(dataset, config, use_optuna=optime, continue_training=continue_training)

    print("Pipeline completed. Run `streamlit run app.py` to launch the app.")

if __name__ == "__main__":
    main_pipeline()