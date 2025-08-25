import torch
import logging
import math
from pytorch_forecasting import TimeSeriesDataSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_batch_size(model, dataset: TimeSeriesDataSet, config: dict) -> int:
    """
    Poprawiony estymator batch size — prostsze i bardziej konserwatywne heurystyki.
    Zamiast nadmiernego mnożenia przez liczbę cech,:
      - oddzielnie uwzględniam pamięć parametrów modelu,
      - bardziej realistycznie estymuję pamięć wejściową (features * seq_len),
      - estymuję pamięć dla attention jako ~seq^2 * heads,
      - estymuję pamięć dla LSTM jako ~seq * hidden * layers,
      - mnożę aktywacje przez czynnik 2 (forward+backward) oraz mały overhead.
    Wynik jest konserwatywny i zwykle mniejszy (dokładniej odpowiada realnemu zużyciu).
    """
    if not config['training'].get('auto_batch_size', False):
        logger.info("Auto-estymacja batch size wyłączona. Używam wartości z configu.")
        return config['training']['batch_size']

    if not torch.cuda.is_available():
        logger.warning("GPU niedostępne. Fallback na CPU z domyślnym batch size (połówka).")
        return max(1, config['training']['batch_size'] // 2)

    device = torch.device('cuda')
    # parametry modelu
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_params_bytes = num_params * 4  # float32
    model_params_mb = model_params_bytes / (1024 ** 2)
    logger.info(f"Liczba parametrów modelu: {num_params} ({model_params_mb:.2f} MB)")

    data_size = len(dataset)
    logger.info(f"Rozmiar datasetu: {data_size}")

    # VRAM dostępna (MB)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    max_vram_usage = config['training'].get('max_vram_usage', 0.8)
    usable_vram_mb = total_vram_mb * max_vram_usage
    logger.info(f"Dostępna VRAM (po limicie {max_vram_usage*100:.0f}%): {usable_vram_mb:.2f} MB")

    # Odejmij pamięć parametrów (i trochę overheadu dla optimizerów, np. moment i grad)
    optimizer_overhead_factor = 1.6  # RMSProp/Adam-like dodatkowe pamięci
    reserved_for_model_mb = model_params_mb * optimizer_overhead_factor
    usable_for_batches_mb = max(0.0, usable_vram_mb - reserved_for_model_mb)
    logger.info(f"Zarezerwowane dla modelu+opty: {reserved_for_model_mb:.2f} MB, dostępne dla batchy: {usable_for_batches_mb:.2f} MB")

    # Cecha / rozmiary sekwencji
    num_features = len(getattr(dataset, "reals", [])) + len(getattr(dataset, "categoricals", []))
    max_encoder_length = config['model'].get('max_encoder_length', 180)
    max_prediction_length = config['model'].get('max_prediction_length', 60)
    seq_len = max_encoder_length  # konserwatywnie bierzemy encoder length na potrzeby aktywacji
    total_time_steps = max_encoder_length + max_prediction_length

    # Prosty input memory (na próbkę) — bez dodatkowych overheadów
    input_elements = total_time_steps * max(1, num_features)
    bytes_per_element = 4  # float32
    input_memory_per_sample_mb = (input_elements * bytes_per_element) / (1024 ** 2)

    # Attention: przybliżenie pamięci dla score matrix: seq^2 * heads * 4 bytes
    hidden_size = config['model'].get('hidden_size', 128)
    attention_head_size = config['model'].get('attention_head_size', 16)
    # policz liczbę heads (konserwatywnie co najmniej 1)
    num_heads = max(1, hidden_size // max(1, attention_head_size))
    attention_scores_bytes = (seq_len ** 2) * num_heads * bytes_per_element
    attention_memory_per_sample_mb = attention_scores_bytes / (1024 ** 2)
    # uwzględnij Q/K/V oraz dodatkowe tymczasowe tensory => mnożnik
    attention_memory_per_sample_mb *= 3.0  # Q/K/V + scores + overhead

    # LSTM activations przybliżenie: seq * hidden * layers * bytes
    lstm_layers = config['model'].get('lstm_layers', 1)
    lstm_bytes = seq_len * hidden_size * lstm_layers * bytes_per_element
    lstm_memory_per_sample_mb = lstm_bytes / (1024 ** 2)
    # mały overhead
    lstm_memory_per_sample_mb *= 2.0

    # Całkowita pamięć aktywacji na próbkę (MB); multiplikator 2 dla forward+backward oraz dodatkowe overhead
    activations_per_sample_mb = (input_memory_per_sample_mb + attention_memory_per_sample_mb + lstm_memory_per_sample_mb)
    multipler_forward_backward = 2.0
    overhead_misc = 1.1
    batch_memory_per_sample_mb = activations_per_sample_mb * multipler_forward_backward * overhead_misc

    # bezpieczeństwo: minimalna estymacja próbek waży conajmniej input size
    batch_memory_per_sample_mb = max(batch_memory_per_sample_mb, input_memory_per_sample_mb * 1.2)

    logger.info(f"Przybliżony rozmiar próbki: {input_elements} elementów, liczba cech: {num_features}")
    logger.info(f"Składowe pamięci (MB) - input: {input_memory_per_sample_mb:.4f}, attention: {attention_memory_per_sample_mb:.4f}, lstm: {lstm_memory_per_sample_mb:.4f}")
    logger.info(f"Zużycie pamięci na próbkę (MB, est): {batch_memory_per_sample_mb:.4f}")

    if usable_for_batches_mb <= 0:
        logger.warning("Niewystarczająca pamięć VRAM po odjęciu modelu/opty. Ustawiam minimalny batch size = 1.")
        return 1

    # oszacuj batch
    estimated_batch = int(usable_for_batches_mb / batch_memory_per_sample_mb) if batch_memory_per_sample_mb > 0 else 1

    # ograniczenia praktyczne
    max_from_dataset = math.ceil(data_size / 10) if data_size > 10 else data_size
    estimated_batch = max(1, min(estimated_batch, 256, max_from_dataset))

    # Dodatkowe zabezpieczenie — jeśli estymacja zbyt duża, zmniejsz konserwatywnie
    if estimated_batch > 128:
        estimated_batch = min(estimated_batch, 128)

    # log szacowanego użycia VRAM
    estimated_vram_mb = estimated_batch * batch_memory_per_sample_mb + reserved_for_model_mb
    logger.info(f"Oszacowano użycie VRAM ~{estimated_vram_mb:.2f} MB dla batch_size {estimated_batch}")

    return int(estimated_batch)