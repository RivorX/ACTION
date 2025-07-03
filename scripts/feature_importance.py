import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import pickle
import yaml
import sys
import os

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from scripts.model import build_model
from scripts.config_manager import ConfigManager

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """
    Klasa do analizy ważności cech w modelach Temporal Fusion Transformer.
    Wykorzystuje InterpretationVisualizer z pytorch_forecasting 1.4.0.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicjalizuje analizator ważności cech.
        
        Args:
            config_path (str): Ścieżka do pliku konfiguracyjnego.
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        # Na potrzeby analizy ważności cech użyjmy CPU zamiast GPU, by uniknąć problemów z urządzeniami
        self.device = torch.device('cpu')
        logger.info(f"Używane urządzenie: {self.device}")
        
    def load_model_and_data(self):
        """
        Wczytuje model oraz dataset.
        
        Returns:
            tuple: (model, dataset) - wytrenowany model oraz dataset.
        """
        try:
            # Wczytaj dataset - wymuszone na CPU
            dataset_path = Path(self.config['data']['processed_data_path'])
            dataset = torch.load(dataset_path, weights_only=False, map_location='cpu')
            logger.info(f"Dataset wczytany z: {dataset_path}")
            
            # Wczytaj model - wymuszone na CPU
            checkpoint_path = Path(self.config['paths']['checkpoint_path'])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint nie istnieje: {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            hyperparams = checkpoint["hyperparams"]
            if 'hidden_continuous_size' not in hyperparams:
                hyperparams['hidden_continuous_size'] = self.config['model']['hidden_size'] // 2
                
            model = build_model(dataset, self.config, hyperparams=hyperparams)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            model = model.cpu()
            logger.info("Model wczytany poprawnie i przeniesiony na CPU.")
            
            return model, dataset
            
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania modelu lub datasetu: {e}")
            raise

    def analyze_feature_importance(self, output_csv: str = "data/feature_importance.csv"):
        """
        Oblicza ważność cech i zapisuje wyniki do pliku CSV.
        
        Args:
            output_csv (str): Ścieżka do pliku CSV, gdzie zapisane zostaną wyniki.
        
        Returns:
            tuple: (importance_df, compact_df) - pełna i skrócona wersja DataFrame z wynikami.
        """
        try:
            model, dataset = self.load_model_and_data()
            
            # Pobierz nazwy cech z datasetu
            feature_mapping = self._get_feature_names_from_dataset(dataset)
            
            # Przygotuj dataloader
            dataloader = dataset.to_dataloader(
                train=False, 
                batch_size=self.config['training']['batch_size'],
                num_workers=0
            )
            
            # Pobierz dane
            x, _ = next(iter(dataloader))
            
            # Przenieś dane na właściwe urządzenie
            x = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
            
            # Oblicz ważność cech
            logger.info("Obliczanie ważności cech...")
            with torch.no_grad():
                interpretation = model.interpret_output(x)
                logger.info(f"Dostępne klucze w interpretacji: {list(interpretation.keys())}")
            
            # Przygotuj dane do zapisania
            importance_data = []
            
            # Przetwarzanie variable_importance
            if 'variable_importance' in interpretation:
                for var_name, importance in interpretation['variable_importance'].items():
                    feature_name = var_name.replace('_encoder', '').replace('_decoder', '')
                    if '__' in feature_name:
                        feature_name = feature_name.split('__')[0]
                    importance_data.append({
                        'Feature': feature_name,
                        'Variable_Importance': float(importance),
                        'Static_Importance': 0.0,
                        'Encoder_Importance': 0.0,
                        'Decoder_Importance': 0.0,
                        'Attention_Importance': 0.0,
                        'Total_Importance': float(importance),
                        'Type': 'Variable'
                    })
            
            # Przetwarzanie static_variables
            if 'static_variables' in interpretation:
                static_vars = interpretation['static_variables']
                if isinstance(static_vars, dict):
                    for var_name, importance in static_vars.items():
                        importance_data.append({
                            'Feature': var_name,
                            'Variable_Importance': 0.0,
                            'Static_Importance': float(importance),
                            'Encoder_Importance': 0.0,
                            'Decoder_Importance': 0.0,
                            'Attention_Importance': 0.0,
                            'Total_Importance': float(importance),
                            'Type': 'Static'
                        })
                elif isinstance(static_vars, torch.Tensor):
                    logger.info(f"static_variables jest tensorem o kształcie: {static_vars.shape}")
                    # Jeśli to tensor, spróbuj przekonwertować na słownik
                    if static_vars.dim() > 0 and static_vars.shape[-1] > 0:
                        # Oblicz średnią po wymiarze batch
                        static_mean = static_vars.mean(dim=0) if static_vars.dim() > 1 else static_vars
                        feature_names = feature_mapping.get('static_variables', [])
                        
                        for i in range(static_mean.shape[0]):
                            # Użyj prawdziwej nazwy cechy jeśli dostępna
                            if i < len(feature_names):
                                feature_name = feature_names[i]
                            else:
                                feature_name = f'static_var_{i}'
                                
                            importance_data.append({
                                'Feature': feature_name,
                                'Variable_Importance': 0.0,
                                'Static_Importance': float(static_mean[i]),
                                'Encoder_Importance': 0.0,
                                'Decoder_Importance': 0.0,
                                'Attention_Importance': 0.0,
                                'Total_Importance': float(static_mean[i]),
                                'Type': 'Static'
                            })
                    else:
                        logger.info("static_variables jest pustym tensorem, pomijam")
            
            # Przetwarzanie encoder_variables i decoder_variables
            for key in ['encoder_variables', 'decoder_variables']:
                if key in interpretation:
                    variables = interpretation[key]
                    if isinstance(variables, dict):
                        for var_name, importance in variables.items():
                            feature_name = var_name.replace('_encoder', '').replace('_decoder', '')
                            if '__' in feature_name:
                                feature_name = feature_name.split('__')[0]
                            entry = {
                                'Feature': feature_name,
                                'Variable_Importance': 0.0,
                                'Static_Importance': 0.0,
                                'Encoder_Importance': float(importance) if key == 'encoder_variables' else 0.0,
                                'Decoder_Importance': float(importance) if key == 'decoder_variables' else 0.0,
                                'Attention_Importance': 0.0,
                                'Total_Importance': float(importance),
                                'Type': key.capitalize().replace('_variables', '')
                            }
                            importance_data.append(entry)
                    elif isinstance(variables, torch.Tensor):
                        logger.info(f"{key} jest tensorem o kształcie: {variables.shape}")
                        # Jeśli to tensor, spróbuj przekonwertować na słownik
                        if variables.dim() > 0 and variables.shape[-1] > 0:
                            # Oblicz średnią po wymiarze batch jeśli istnieje
                            vars_mean = variables.mean(dim=0) if variables.dim() > 1 else variables
                            feature_names = feature_mapping.get(key, [])
                            
                            for i in range(vars_mean.shape[0]):
                                # Użyj prawdziwej nazwy cechy jeśli dostępna
                                if i < len(feature_names):
                                    feature_name = feature_names[i]
                                else:
                                    feature_name = f'{key}_{i}'
                                    
                                entry = {
                                    'Feature': feature_name,
                                    'Variable_Importance': 0.0,
                                    'Static_Importance': 0.0,
                                    'Encoder_Importance': float(vars_mean[i]) if key == 'encoder_variables' else 0.0,
                                    'Decoder_Importance': float(vars_mean[i]) if key == 'decoder_variables' else 0.0,
                                    'Attention_Importance': 0.0,
                                    'Total_Importance': float(vars_mean[i]),
                                    'Type': key.capitalize().replace('_variables', '')
                                }
                                importance_data.append(entry)
                        else:
                            logger.info(f"{key} jest pustym tensorem, pomijam")
            
            # Przetwarzanie attention
            if 'attention' in interpretation:
                attention_data = interpretation['attention']
                if isinstance(attention_data, dict):
                    for var_name, importance in attention_data.items():
                        feature_name = var_name.replace('_encoder', '').replace('_decoder', '')
                        if '__' in feature_name:
                            feature_name = feature_name.split('__')[0]
                        entry = {
                            'Feature': feature_name,
                            'Variable_Importance': 0.0,
                            'Static_Importance': 0.0,
                            'Encoder_Importance': 0.0,
                            'Decoder_Importance': 0.0,
                            'Attention_Importance': float(importance),
                            'Total_Importance': float(importance),
                            'Type': 'Attention'
                        }
                        importance_data.append(entry)
                elif isinstance(attention_data, torch.Tensor):
                    logger.info(f"attention jest tensorem o kształcie: {attention_data.shape}")
                    # Jeśli to tensor, oblicz średnią wagę attention dla każdej cechy
                    if attention_data.dim() >= 2:
                        # Tensor attention ma kształt [batch_size, seq_len, features] lub podobny
                        # Oblicz średnią po wszystkich wymiarach oprócz ostatniego
                        attention_mean = attention_data.mean(dim=tuple(range(attention_data.dim() - 1)))
                        
                        # Spróbuj zmapować na nazwy cech - attention może odpowiadać encoder_variables
                        encoder_names = feature_mapping.get('encoder_variables', [])
                        
                        for i in range(attention_mean.shape[0]):
                            # Użyj prawdziwej nazwy cechy jeśli dostępna
                            if i < len(encoder_names):
                                feature_name = f"{encoder_names[i]}_attention"
                            else:
                                feature_name = f'attention_feature_{i}'
                                
                            entry = {
                                'Feature': feature_name,
                                'Variable_Importance': 0.0,
                                'Static_Importance': 0.0,
                                'Encoder_Importance': 0.0,
                                'Decoder_Importance': 0.0,
                                'Attention_Importance': float(attention_mean[i]),
                                'Total_Importance': float(attention_mean[i]),
                                'Type': 'Attention'
                            }
                            importance_data.append(entry)
            
            # Utwórz DataFrame
            importance_df = pd.DataFrame(importance_data)
            
            if importance_df.empty:
                logger.warning("Brak danych o ważności cech do zapisania.")
                return None, None
                
            # Normalizuj wartości całkowitej ważności
            total_sum = importance_df['Total_Importance'].sum()
            if total_sum > 0:
                importance_df['Total_Importance_Normalized'] = importance_df['Total_Importance'] / total_sum
            else:
                importance_df['Total_Importance_Normalized'] = 0.0
                
            # Sortuj według całkowitej ważności
            importance_df = importance_df.sort_values('Total_Importance', ascending=False)
                
            # Utwórz kompaktową wersję
            compact_df = importance_df[['Feature', 'Total_Importance', 'Total_Importance_Normalized']].copy()
                
            # Zapisz do CSV
            output_path = Path(output_csv)
            importance_df.to_csv(output_path, index=False)
            logger.info(f"Ważność cech zapisana do: {output_path}")
            
            compact_path = output_path.parent / 'feature_importance_compact.csv'
            compact_df.to_csv(compact_path, index=False)
            logger.info(f"Kompaktowa wersja ważności cech zapisana do: {compact_path}")
            
            # Wypisz najważniejsze cechy
            logger.info("Top 10 najważniejszych cech:")
            for i, (feature, importance) in enumerate(
                zip(compact_df['Feature'].head(10), compact_df['Total_Importance_Normalized'].head(10))
            ):
                logger.info(f"{i+1}. {feature}: {importance:.4f}")
                
            return importance_df, compact_df
                
        except Exception as e:
            logger.error(f"Błąd podczas obliczania ważności cech: {e}")
            raise
            
    def plot_feature_importance(self, compact_df=None, output_path="data/feature_importance_plot.png"):
        """
        Tworzy wykres ważności cech.
        
        Args:
            compact_df (pd.DataFrame, optional): DataFrame z kompaktową wersją ważności cech.
                                                Jeśli None, próbuje wczytać z pliku.
            output_path (str): Ścieżka do pliku z wykresem.
        """
        try:
            if compact_df is None:
                # Jeśli nie podano DataFrame, spróbuj wczytać z pliku
                compact_path = Path("data/feature_importance_compact.csv")
                if compact_path.exists():
                    compact_df = pd.read_csv(compact_path)
                else:
                    # Jeśli nie istnieje, wykonaj analizę
                    _, compact_df = self.analyze_feature_importance()
                    if compact_df is None:
                        raise ValueError("Nie udało się utworzyć danych do wykresu.")
            
            # Filtruj tylko prawdziwe cechy (usuń attention i attention_feature_X)
            real_features_df = compact_df[
                (~compact_df['Feature'].str.contains('attention', case=False, na=False)) &
                (~compact_df['Feature'].str.contains('attention_feature_', case=False, na=False))
            ].copy()
            
            if real_features_df.empty:
                logger.warning("Brak prawdziwych cech do wyświetlenia na wykresie.")
                return
            
            # Posortuj cechy według ważności od największej do najmniejszej
            real_features_df = real_features_df.sort_values('Total_Importance_Normalized', ascending=True)
            
            # Wybierz top N cech (maksymalnie 20 dla czytelności)
            top_n = min(20, len(real_features_df))
            plot_df = real_features_df.tail(top_n)
            
            # Utwórz wykres
            plt.figure(figsize=(14, 10))
            sns.set_theme(style="whitegrid")
            
            # Utwórz barplot
            ax = sns.barplot(
                x='Total_Importance_Normalized',
                y='Feature',
                data=plot_df,
                palette="viridis",
                hue='Feature',
                legend=False
            )
            
            # Dodaj tytuł i etykiety
            plt.title('Ważność cech w modelu Temporal Fusion Transformer', fontsize=16, pad=20)
            plt.xlabel('Znormalizowana ważność', fontsize=12)
            plt.ylabel('Cecha', fontsize=12)
            
            # Dodaj wartości na końcach słupków
            for i, v in enumerate(plot_df['Total_Importance_Normalized']):
                ax.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=10)
            
            # Dostosuj layout
            plt.tight_layout()
            
            # Zapisz wykres
            plt_path = Path(output_path)
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            plt.close()  # Zamknij figure aby zwolnić pamięć
            logger.info(f"Wykres ważności cech zapisany do {plt_path}")
            
            # Wypisz statystyki
            logger.info(f"Wykres zawiera {len(plot_df)} najważniejszych prawdziwych cech")
            
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia wykresu: {e}")
            raise

    def _get_feature_names_from_dataset(self, dataset):
        """
        Pobiera nazwy cech z datasetu.
        
        Returns:
            dict: Słownik mapujący indeksy na nazwy cech
        """
        feature_mapping = {
            'encoder_variables': [],
            'decoder_variables': [],
            'static_variables': []
        }
        
        try:
            # Pobierz nazwy cech z datasetu
            if hasattr(dataset, 'reals'):
                feature_mapping['encoder_variables'] = dataset.reals.copy()
                feature_mapping['decoder_variables'] = dataset.reals.copy()
                
            if hasattr(dataset, 'categoricals'):
                feature_mapping['encoder_variables'].extend(dataset.categoricals)
                feature_mapping['decoder_variables'].extend(dataset.categoricals)
                
            if hasattr(dataset, 'static_categoricals'):
                feature_mapping['static_variables'].extend(dataset.static_categoricals)
                
            if hasattr(dataset, 'static_reals'):
                feature_mapping['static_variables'].extend(dataset.static_reals)
                
            logger.info(f"Znaleziono nazwy cech:")
            logger.info(f"  encoder_variables: {len(feature_mapping['encoder_variables'])} cech")
            logger.info(f"  decoder_variables: {len(feature_mapping['decoder_variables'])} cech")
            logger.info(f"  static_variables: {len(feature_mapping['static_variables'])} cech")
            
        except Exception as e:
            logger.warning(f"Nie udało się pobrać nazw cech z datasetu: {e}")
            
        return feature_mapping
        
def calculate_feature_importance(config_path: str = "config/config.yaml", output_csv: str = "data/feature_importance.csv"):
    """
    Oblicza ważność cech dla modelu TemporalFusionTransformer i zapisuje wyniki do pliku CSV.
    
    Args:
        config_path (str): Ścieżka do pliku konfiguracyjnego.
        output_csv (str): Ścieżka do pliku CSV, gdzie zapisane zostaną wyniki.
    """
    analyzer = FeatureImportanceAnalyzer(config_path)
    importance_df, compact_df = analyzer.analyze_feature_importance(output_csv)
    analyzer.plot_feature_importance(compact_df)
    return importance_df, compact_df

if __name__ == "__main__":
    logger.info("Rozpoczynanie analizy ważności cech...")
    calculate_feature_importance()
    logger.info("Analiza ważności cech zakończona.")