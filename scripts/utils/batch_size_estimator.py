import torch
import logging
import math
from pytorch_forecasting import TimeSeriesDataSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_batch_size(model, dataset: TimeSeriesDataSet, config: dict) -> int:
    """
    Estymuje batch size analitycznie na podstawie złożoności modelu, ilości danych, dostępnej VRAM i AMP.
    
    Args:
        model: Instancja modelu PyTorch (np. CustomTemporalFusionTransformer).
        dataset: Instancja TimeSeriesDataSet.
        config: Słownik konfiguracji (z 'training.auto_batch_size', 'training.max_vram_usage', 
                'model.max_encoder_length', 'model.max_prediction_length', 'model.hidden_size', 
                'model.attention_head_size', 'model.lstm_layers').
    
    Returns:
        int: Oszacowany batch size.
    """
    if not config['training'].get('auto_batch_size', False):
        logger.info("Auto-estymacja batch size wyłączona. Używam wartości z configu.")
        return config['training']['batch_size']
    
    # Sprawdź dostępność GPU
    if not torch.cuda.is_available():
        logger.warning("GPU niedostępne. Fallback na CPU z domyślnym batch size.")
        return config['training']['batch_size'] // 2  # Mniejszy batch na CPU dla bezpieczeństwa
    
    device = torch.device('cuda')
    
    # Oblicz złożoność modelu (liczba parametrów)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Liczba parametrów modelu: {num_params}")
    
    # Ilość danych (rozmiar datasetu)
    data_size = len(dataset)
    logger.info(f"Rozmiar datasetu: {data_size}")
    
    # Dostępna VRAM
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # w MB
    max_vram_usage = config['training'].get('max_vram_usage', 0.8)  # Domyślnie 80%
    available_vram = total_vram * max_vram_usage
    logger.info(f"Dostępna VRAM (po limicie {max_vram_usage*100}%): {available_vram:.2f} MB")
    
    # Zużycie pamięci przez model (stała wartość z danych)
    model_memory = 5.63  # MB, z podanych danych
    logger.info(f"Zużycie pamięci przez model: {model_memory:.2f} MB (z AMP: 0.5)")
    
    # Przybliżony rozmiar próbki: (max_encoder_length + max_prediction_length) * liczba cech
    num_features = len(dataset.reals) + len(dataset.categoricals)
    max_encoder_length = config['model']['max_encoder_length']  # 180
    max_prediction_length = config['model']['max_prediction_length']  # 60
    input_size = (max_encoder_length + max_prediction_length) * num_features
    overhead_factor = 1.5  # Zmniejszony dla embeddingów i bufferów w TFT
    input_memory_per_sample = (input_size * 4 * overhead_factor * 0.5) / (1024 ** 2)  # AMP factor = 0.5
    
    # Aktywacje dla multi-head attention (O(seq^2 * hidden * heads * 3 dla Q/K/V), korekta 1/8 dla TFT)
    hidden_size = config['model']['hidden_size']  # 128
    attention_head_size = config['model']['attention_head_size']  # 4
    attention_activation_per_sample = (max_encoder_length ** 2 * hidden_size * attention_head_size * 3) * 4 * 0.5 / (1024 ** 2) / 8.0
    
    # Aktywacje dla LSTM (O(seq * hidden * layers * 2))
    lstm_layers = config['model']['lstm_layers']  # 2
    lstm_activation_per_sample = lstm_layers * (max_encoder_length * hidden_size * 2) * 4 * 0.5 / (1024 ** 2)
    
    # Całkowita pamięć na próbkę (z faktorem 1.3 dla forward + backward)
    batch_memory_per_sample = (input_memory_per_sample + attention_activation_per_sample + lstm_activation_per_sample) * 1.3
    
    logger.info(f"Przybliżony rozmiar próbki: {input_size} elementów, liczba cech: {num_features}")
    logger.info(f"Zużycie pamięci na próbkę: {batch_memory_per_sample:.4f} MB")
    
    # Oblicz maksymalny batch size
    remaining_vram = available_vram - model_memory
    if remaining_vram <= 0:
        logger.warning("Niewystarczająca pamięć VRAM po zaalokowaniu modelu. Ustawiam minimalny batch size.")
        return 1
    
    estimated_batch_size = int(remaining_vram / batch_memory_per_sample)
    estimated_batch_size = min(estimated_batch_size, math.ceil(data_size / 10))  # Limit do 10% datasetu
    estimated_batch_size = max(1, estimated_batch_size)  # Minimum 1
    
    # Ogranicz batch_size do maksymalnie 256, aby celować w przedział 128–256
    estimated_batch_size = min(256, estimated_batch_size)
    
    # Oblicz oszacowane użycie VRAM dla tego batch_size
    estimated_vram = model_memory + estimated_batch_size * batch_memory_per_sample
    logger.info(f"Oszacowano użycie VRAM na {estimated_vram:.2f} MB dla batch_size {estimated_batch_size}")
    
    return estimated_batch_size