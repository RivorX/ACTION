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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from scripts.model import build_model
from scripts.utils.config_manager import ConfigManager

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
        self.device = torch.device('cpu')
        logger.info(f"Używane urządzenie: {self.device}")
        
    def load_model_and_data(self):
        """
        Wczytuje model oraz dataset.
        
        Returns:
            tuple: (model, dataset) - wytrenowany model oraz dataset.
        """
        try:
            dataset_path = Path(self.config['data']['processed_data_path'])
            dataset = torch.load(dataset_path, weights_only=False, map_location=self.device)
            logger.info(f"Dataset wczytany z: {dataset_path}")
            
            # Dynamiczne budowanie ścieżki do modelu na podstawie model_name i models_dir
            model_name = self.config['model_name']
            checkpoint_path = Path(self.config['paths']['models_dir']) / f"{model_name}.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint nie istnieje: {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            hyperparams = checkpoint["hyperparams"]
            if 'hidden_continuous_size' not in hyperparams:
                hyperparams['hidden_continuous_size'] = self.config['model']['hidden_size'] // 2
                
            model = build_model(dataset, self.config, hyperparams=hyperparams)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            model = model.to(self.device)
            logger.info(f"Model wczytany poprawnie z {checkpoint_path} i przeniesiony na {self.device}.")
            
            return model, dataset
            
        except Exception as e:
            logger.error(f"Błąd podczas wczytywania modelu lub datasetu: {e}")
            raise

    def analyze_feature_importance(self, output_csv: str = "data/feature_importance_analysis/feature_importance.csv"):
        """
        Oblicza ważność cech i zapisuje wyniki do pliku CSV.
        
        Args:
            output_csv (str): Ścieżka do pliku CSV, gdzie zapisane zostaną wyniki.
        
        Returns:
            tuple: (importance_df, compact_df) - pełna i skrócona wersja DataFrame z wynikami.
        """
        try:
            model, dataset = self.load_model_and_data()
            
            feature_mapping = self._get_feature_names_from_dataset(dataset)
            
            dataloader = dataset.to_dataloader(
                train=False, 
                batch_size=self.config['training']['batch_size'],
                num_workers=0
            )
            
            x, _ = next(iter(dataloader))
            x = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
            
            logger.info("Obliczanie ważności cech...")
            with torch.no_grad():
                interpretation = model.interpret_output(x)
                logger.info(f"Dostępne klucze w interpretacji: {list(interpretation.keys())}")
            
            importance_data = []
            
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
                    static_mean = static_vars.mean(dim=0) if static_vars.dim() > 1 else static_vars
                    feature_names = feature_mapping.get('static_variables', [])
                    
                    for i in range(static_mean.shape[0]):
                        feature_name = feature_names[i] if i < len(feature_names) else f'static_var_{i}'
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
                        vars_mean = variables.mean(dim=0) if variables.dim() > 1 else variables
                        feature_names = feature_mapping.get(key, [])
                        
                        for i in range(vars_mean.shape[0]):
                            feature_name = feature_names[i] if i < len(feature_names) else f'{key}_{i}'
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
                    attention_mean = attention_data.mean(dim=tuple(range(attention_data.dim() - 1)))
                    encoder_names = feature_mapping.get('encoder_variables', [])
                    
                    for i in range(attention_mean.shape[0]):
                        feature_name = f"{encoder_names[i]}_attention" if i < len(encoder_names) else f'attention_feature_{i}'
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
            
            importance_df = pd.DataFrame(importance_data)
            
            if importance_df.empty:
                logger.warning("Brak danych o ważności cech do zapisania.")
                return None, None
                
            total_sum = importance_df['Total_Importance'].sum()
            if total_sum > 0:
                importance_df['Total_Importance_Normalized'] = importance_df['Total_Importance'] / total_sum
            else:
                importance_df['Total_Importance_Normalized'] = 0.0
                
            importance_df = importance_df.sort_values('Total_Importance', ascending=False)
            compact_df = importance_df[['Feature', 'Total_Importance', 'Total_Importance_Normalized', 'Type']].copy()
                
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Tworzenie katalogu, jeśli nie istnieje
            importance_df.to_csv(output_path, index=False)
            logger.info(f"Ważność cech zapisana do: {output_path}")
            
            compact_path = output_path.parent / 'feature_importance_compact.csv'
            compact_df.to_csv(compact_path, index=False)
            logger.info(f"Kompaktowa wersja ważności cech zapisana do: {compact_path}")
            
            # Zapisz pełny DataFrame bez cech attention do oddzielnego pliku
            non_attention_df = importance_df[
                (~importance_df['Feature'].str.contains('attention', case=False, na=False)) &
                (~importance_df['Feature'].str.contains('attention_feature_', case=False, na=False))
            ].copy()
            non_attention_path = output_path.parent / 'feature_importance_no_attention.csv'
            non_attention_df.to_csv(non_attention_path, index=False)
            logger.info(f"Pełna wersja bez cech attention zapisana do: {non_attention_path}")
            
            # Wyświetlanie top 10 i bottom 10 cech w konsoli
            logger.info("Top 10 najważniejszych cech:")
            for i, row in compact_df.head(10).iterrows():
                logger.info(f"{i+1}. {row['Feature']}: {row['Total_Importance_Normalized']:.4f} ({row['Type']})")
            
            return importance_df, compact_df
                
        except Exception as e:
            logger.error(f"Błąd podczas obliczania ważności cech: {e}")
            raise
            
    def plot_feature_importance(self, compact_df=None, output_dir="data/feature_importance_analysis"):
        """
        Tworzy dwa wykresy: dla 10 najważniejszych cech i 10 najmniej istotnych cech.
        
        Args:
            compact_df (pd.DataFrame, optional): DataFrame z kompaktową wersją ważności cech.
            output_dir (str): Katalog, gdzie zapisane zostaną wykresy.
        """
        try:
            if compact_df is None:
                compact_path = Path(output_dir) / "feature_importance_compact.csv"
                if compact_path.exists():
                    compact_df = pd.read_csv(compact_path)
                else:
                    _, compact_df = self.analyze_feature_importance()
                    if compact_df is None:
                        raise ValueError("Nie udało się utworzyć danych do wykresów.")
            
            # Filtracja rzeczywistych cech (usunięcie artefaktów typu 'attention')
            real_features_df = compact_df[
                (~compact_df['Feature'].str.contains('attention', case=False, na=False)) &
                (~compact_df['Feature'].str.contains('attention_feature_', case=False, na=False))
            ].copy()
            
            if real_features_df.empty:
                logger.warning("Brak prawdziwych cech do wyświetlenia na wykresach.")
                return
            
            # Sortowanie i wybór dokładnie 10 najlepszych cech
            real_features_df = real_features_df.sort_values('Total_Importance_Normalized', ascending=False)
            top_10_df = real_features_df.head(10)  # 10 najważniejszych cech
            
            # Tworzenie katalogu wyjściowego
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Wykres 1: Top 10 najważniejszych cech
            plt.figure(figsize=(14, 8))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(
                x='Total_Importance_Normalized',
                y='Feature',
                data=top_10_df,
                palette="viridis",
                hue='Feature',
                dodge=False,
                legend=False
            )
            plt.title('Top 10 najważniejszych cech w modelu Temporal Fusion Transformer', fontsize=16, pad=20)
            plt.xlabel('Znormalizowana ważność', fontsize=12)
            plt.ylabel('Cecha', fontsize=12)
            # Dodanie etykiet wartości z buforem, aby uniknąć nakładania
            max_value = top_10_df['Total_Importance_Normalized'].max()
            for i, v in enumerate(top_10_df['Total_Importance_Normalized']):
                ax.text(v + 0.0005, i, f"{v:.4f}", va='center', fontsize=10)
            # Ustawienie zakresu osi X z buforem
            plt.xlim(0, max_value * 1.1)  # Dodanie 10% bufora
            plt.tight_layout()
            top_plot_path = output_path / 'top_10_feature_importance.png'
            plt.savefig(top_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Wykres top 10 cech zapisany do {top_plot_path}")
            
            # Wykres 2: Top 10 najmniej istotnych cech
            bottom_10_df = real_features_df.tail(10).sort_values('Total_Importance_Normalized')  # 10 najmniej istotnych cech
            plt.figure(figsize=(14, 8))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(
                x='Total_Importance_Normalized',
                y='Feature',
                data=bottom_10_df,
                palette="magma",
                hue='Feature',
                dodge=False,
                legend=False
            )
            plt.title('Top 10 najmniej istotnych cech w modelu Temporal Fusion Transformer', fontsize=16, pad=20)
            plt.xlabel('Znormalizowana ważność', fontsize=12)
            plt.ylabel('Cecha', fontsize=12)
            # Dodanie etykiet wartości z buforem
            max_value_bottom = bottom_10_df['Total_Importance_Normalized'].max()
            for i, v in enumerate(bottom_10_df['Total_Importance_Normalized']):
                ax.text(v + 0.0005, i, f"{v:.4f}", va='center', fontsize=10)
            # Ustawienie zakresu osi X z buforem
            plt.xlim(0, max_value_bottom * 1.1)  # Dodanie 10% bufora
            plt.tight_layout()
            bottom_plot_path = output_path / 'bottom_10_feature_importance.png'
            plt.savefig(bottom_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Wykres najmniej istotnych cech zapisany do {bottom_plot_path}")
            
            logger.info(f"Wykresy zawierają 10 najważniejszych i 10 najmniej istotnych cech")
            
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia wykresów: {e}")
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
                
            # Dodaj Sector do static_categoricals z pełnym zestawem kategorii
            feature_mapping['static_variables'].append('Sector')
            
            logger.info(f"Znaleziono nazwy cech:")
            logger.info(f"  encoder_variables: {len(feature_mapping['encoder_variables'])} cech")
            logger.info(f"  decoder_variables: {len(feature_mapping['decoder_variables'])} cech")
            logger.info(f"  static_variables: {len(feature_mapping['static_variables'])} cech")
            
        except Exception as e:
            logger.warning(f"Nie udało się pobrać nazw cech z datasetu: {e}")
            
        return feature_mapping
        
def calculate_feature_importance(config_path: str = "config/config.yaml", output_csv: str = "data/feature_importance_analysis/feature_importance.csv"):
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