import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def convert_to_prices(y_hat_denorm, y_target_denorm, last_close_price, batch_idx):
    def to_scalar(tensor_val):
        if hasattr(tensor_val, 'numel') and tensor_val.numel() == 1:
            return tensor_val.item()
        elif hasattr(tensor_val, 'cpu'):
            val = tensor_val.cpu().numpy()
            return val.item() if val.size == 1 else val.flatten()[0]
        else:
            return float(tensor_val)
    
    y_hat_prices = []
    y_target_prices = []
    current_price_pred = last_close_price
    current_price_target = last_close_price
    
    for i in range(min(5, y_hat_denorm.shape[1])):
        if y_hat_denorm.dim() == 3:
            relative_return_pred = to_scalar(y_hat_denorm[0, i, 1])
            relative_return_pred_lower = to_scalar(y_hat_denorm[0, i, 0])
            relative_return_pred_upper = to_scalar(y_hat_denorm[0, i, 2])
        else:
            logger.warning(f"y_hat_denorm ma nieoczekiwany kształt: {y_hat_denorm.shape}")
            relative_return_pred = to_scalar(y_hat_denorm[0, i])
            relative_return_pred_lower = relative_return_pred
            relative_return_pred_upper = relative_return_pred
        
        next_price_pred = current_price_pred * (1 + relative_return_pred)
        next_price_pred_lower = current_price_pred * (1 + relative_return_pred_lower)
        next_price_pred_upper = current_price_pred * (1 + relative_return_pred_upper)
        
        y_hat_prices.append({
            'median': next_price_pred,
            'lower': next_price_pred_lower,
            'upper': next_price_pred_upper
        })
        current_price_pred = next_price_pred
        
        relative_return_target = to_scalar(y_target_denorm[0, i])
        next_price_target = current_price_target * (1 + relative_return_target)
        y_target_prices.append(next_price_target)
        current_price_target = next_price_target
    
    pred_medians = [f"{p['median']:.2f}" for p in y_hat_prices]
    pred_lowers = [f"{p['lower']:.2f}" for p in y_hat_prices]
    pred_uppers = [f"{p['upper']:.2f}" for p in y_hat_prices]
    target_prices_formatted = [f"{p:.2f}" for p in y_target_prices]
    
    logger.info(
        f"Validation batch {batch_idx} - RZECZYWISTE CENY:\n"
        f"  Predykcje (mediana): {pred_medians}\n"
        f"  Predykcje (dolny 10%): {pred_lowers}\n"
        f"  Predykcje (górny 90%): {pred_uppers}\n"
        f"  Rzeczywiste ceny: {target_prices_formatted}"
    )

def create_validation_plot(y_hat_denorm, y_target_denorm, batch_idx, logs_dir, current_epoch):
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(y_hat_denorm.shape[1])
    if y_hat_denorm.dim() == 3:
        y_hat_median = y_hat_denorm[0, :, 1].numpy()
        y_hat_lower = y_hat_denorm[0, :, 0].numpy()
        y_hat_upper = y_hat_denorm[0, :, 2].numpy()
    else:
        y_hat_median = y_hat_denorm[0, :].numpy()
        y_hat_lower = y_hat_median
        y_hat_upper = y_hat_median
    y_target = y_target_denorm[0, :].numpy()
    plt.plot(time_steps, y_target, label='Rzeczywiste', color='blue', marker='o')
    plt.plot(time_steps, y_hat_median, label='Predykcja (mediana)', color='red', linestyle='--', marker='x')
    plt.fill_between(time_steps, y_hat_lower, y_hat_upper, color='red', alpha=0.1, label='Przedział ufności (10%-90%)')
    plt.title(f'Predykcja vs Rzeczywiste - Batch {batch_idx}')
    plt.xlabel('Krok czasowy')
    plt.ylabel('Zwrot względny')
    plt.legend()
    plt.grid(True)
    os.makedirs(logs_dir, exist_ok=True)
    plot_path = os.path.join(logs_dir, f'ValPlot_Batch_{batch_idx}_epoch_{current_epoch}.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Zapisano wykres walidacyjny: {plot_path}")

def log_validation_details(x, y_hat, y_target, batch_idx, normalizers, dataset, save_plots, plot_count, max_plots_per_epoch, logs_dir, current_epoch):
    relative_returns_normalizer = normalizers.get('Relative_Returns') or dataset.target_normalizer
    if relative_returns_normalizer:
        try:
            y_hat_denorm = relative_returns_normalizer.inverse_transform(y_hat.float().cpu())
            y_target_denorm = relative_returns_normalizer.inverse_transform(y_target.float().cpu())
            if 'encoder_cont' in x:
                encoder_cont = x['encoder_cont'][0].cpu()
                close_normalizer = normalizers.get('Close')
                if close_normalizer is not None:
                    try:
                        numeric_features = [
                            "Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI", "Volatility",
                            "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D", "ATR", "OBV",
                            "Close_momentum_1d", "Close_momentum_5d", "Close_vs_MA10", "Close_vs_MA50",
                            "Close_percentile_20d", "Close_volatility_5d", "Close_RSI_divergence"
                        ]
                        close_idx = numeric_features.index("Close") if "Close" in numeric_features else None
                        if close_idx is not None:
                            last_close_norm = encoder_cont[-1, close_idx]
                            last_close_denorm = close_normalizer.inverse_transform(torch.tensor([[last_close_norm]]))
                            last_close_price = np.expm1(last_close_denorm.numpy())[0, 0]
                            if last_close_price > 10000:
                                logger.warning(f"Bardzo wysoka cena Close: {last_close_price:.2f}")
                                last_close_price_alt = last_close_denorm.numpy()[0, 0]
                                if 10 <= last_close_price_alt <= 1000:
                                    last_close_price = last_close_price_alt
                            logger.info(f"Ostatnia cena Close z batcha: {last_close_price:.2f}")
                            convert_to_prices(y_hat_denorm, y_target_denorm, last_close_price, batch_idx)
                        else:
                            logger.warning("Nie można znaleźć indeksu kolumny Close")
                    except Exception as e:
                        logger.error(f"Błąd podczas konwersji na rzeczywiste ceny: {e}")
                else:
                    logger.warning("Brak normalizera dla Close")
            else:
                logger.warning("Brak danych encoder_cont w batchu")
        except Exception as e:
            logger.error(f"Błąd podczas denormalizacji Relative Returns: {e}")
    else:
        logger.warning("Brak normalizera dla 'Relative_Returns'")
    if save_plots and plot_count < max_plots_per_epoch:
        try:
            create_validation_plot(y_hat_denorm, y_target_denorm, batch_idx, logs_dir, current_epoch)
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia wykresu walidacyjnego: {e}")