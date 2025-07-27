import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import logging
import os
from datetime import datetime
from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.prediction_engine import load_data_and_model, preprocess_data, generate_predictions
import torch

logger = logging.getLogger(__name__)

def create_benchmark_plot(config, benchmark_tickers, historical_close_dict):
    """Tworzy wykres benchmarku porównujący predykcje z rzeczywistymi cenami zamknięcia dla ostatnich 3 miesięcy dla wielu firm."""
    all_results = {}
    temp_raw_data_path = 'data/temp_benchmark_data.csv'
    accuracy_scores = {}
    max_prediction_length = config['model']['max_prediction_length']
    trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
    start_date = trim_date - pd.Timedelta(days=720)

    for ticker in benchmark_tickers:
        try:
            # Wczytaj pełne dane historyczne
            fetcher = DataFetcher(ConfigManager())
            full_data = fetcher.fetch_stock_data(ticker, start_date, datetime.now())
            if full_data.empty:
                logger.error(f"Brak danych dla {ticker}")
                accuracy_scores[ticker] = 0.0
                continue
            full_data = full_data[full_data['Ticker'] == ticker].copy()
            full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
            full_data.set_index('Date', inplace=True)
            historical_close = full_data['Close']

            # Przytnij dane do trim_date dla modelu
            new_data = full_data[full_data.index <= trim_date].copy()
            if new_data.empty:
                logger.error(f"Brak danych przed {trim_date} dla {ticker}")
                accuracy_scores[ticker] = 0.0
                continue
            new_data.reset_index().to_csv(temp_raw_data_path, index=False)
            logger.info(f"Dane dla {ticker} zapisane do {temp_raw_data_path}")

            # Wczytaj dane i model
            _, dataset, normalizers, model = load_data_and_model(config, ticker, temp_raw_data_path, historical_mode=True)
            ticker_data, original_close = preprocess_data(config, new_data.reset_index(), ticker, normalizers, historical_mode=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                median, _, _ = generate_predictions(config, dataset, model, ticker_data)

            # Przygotuj daty i dane
            last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=max_prediction_length, freq='D')

            # Przytnij dane historyczne do okresu przed predykcją
            historical_dates = ticker_data['Date'].tolist()
            historical_close_trimmed = original_close.tolist()
            if len(historical_dates) != len(historical_close_trimmed):
                logger.error(f"Niezgodność długości historical_dates ({len(historical_dates)}) i historical_close_trimmed ({len(historical_close_trimmed)}) dla {ticker}")
                accuracy_scores[ticker] = 0.0
                continue

            # Pobierz dane dla okresu predykcji
            historical_pred_close = historical_close.loc[trim_date:]
            if historical_pred_close.empty:
                logger.error(f"Brak danych historycznych po {trim_date} dla {ticker}")
                accuracy_scores[ticker] = 0.0
                continue
            historical_pred_close = historical_pred_close.reindex(pd.to_datetime(pred_dates), method='ffill')
            if historical_pred_close.isna().any():
                logger.warning(f"Znaleziono NaN w historical_pred_close dla {ticker}. Wypełniam metodą ffill i bfill.")
                historical_pred_close = historical_pred_close.ffill().bfill()
            if historical_pred_close.isna().any():
                logger.error(f"Po wypełnieniu nadal istnieją NaN w historical_pred_close dla {ticker}")
                accuracy_scores[ticker] = 0.0
                continue
            historical_pred_close = historical_pred_close.tolist()

            # Oblicz dokładność
            if len(median) == len(historical_pred_close):
                median = np.array(median)
                historical_pred_close_array = np.array(historical_pred_close)
                if np.any(historical_pred_close_array == 0):
                    logger.warning(f"Znalezione zera w historical_pred_close dla {ticker}. Zastępuję zera wartością 1e-6.")
                    historical_pred_close_array = np.where(historical_pred_close_array == 0, 1e-6, historical_pred_close_array)
                differences = np.abs(median - historical_pred_close_array)
                relative_diff = (differences / historical_pred_close_array) * 100
                if np.any(np.isnan(relative_diff)):
                    logger.warning(f"NaN w relative_diff dla {ticker}. Pomijam wartości NaN w obliczaniu średniej.")
                    relative_diff = relative_diff[~np.isnan(relative_diff)]
                if len(relative_diff) > 0:
                    accuracy = 100 - np.mean(relative_diff)
                    accuracy_scores[ticker] = accuracy
                    logger.info(f"Procentowa zgodność dla {ticker}: {accuracy:.2f}%")
                else:
                    logger.warning(f"Brak ważnych danych do obliczenia zgodności dla {ticker}")
                    accuracy_scores[ticker] = 0.0
            else:
                logger.error(f"Niezgodna długość predykcji i danych historycznych dla {ticker}: median={len(median)}, historical_pred_close={len(historical_pred_close)}")
                accuracy_scores[ticker] = 0.0

            all_results[ticker] = {
                'historical_dates': historical_dates,
                'historical_close': historical_close_trimmed,
                'pred_dates': [d.to_pydatetime() for d in pred_dates],
                'predictions': median.tolist(),
                'historical_pred_close': historical_pred_close
            }
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania {ticker}: {e}")
            accuracy_scores[ticker] = 0.0
        finally:
            if os.path.exists(temp_raw_data_path):
                os.remove(temp_raw_data_path)
                logger.info(f"Tymczasowy plik {temp_raw_data_path} usunięty.")

    # Tworzenie wykresu
    fig = go.Figure()
    colors = ['#0000FF', '#00FF00', '#FF0000', '#800080', '#FFA500', '#00FFFF', '#FF00FF', '#FFFF00', '#A52A2A', '#808080']

    for idx, (ticker, data) in enumerate(all_results.items()):
        color_idx = idx % len(colors)
        historical_dates = data['historical_dates']
        pred_dates = data['pred_dates']
        historical_close = data['historical_close']
        historical_pred_close = data['historical_pred_close']
        predictions = data['predictions']

        # Połącz dane do wykresu
        all_dates = historical_dates + pred_dates
        all_close = historical_close + historical_pred_close
        all_pred_close = [None] * len(historical_dates) + predictions

        # Sprawdź zgodność długości
        if len(all_dates) != len(all_close) or len(all_dates) != len(all_pred_close):
            logger.error(f"Niezgodność długości dla {ticker}: all_dates={len(all_dates)}, all_close={len(all_close)}, all_pred_close={len(all_pred_close)}")
            continue

        plot_data = pd.DataFrame({
            'Date': all_dates,
            'Close': all_close,
            'Predicted_Close': all_pred_close
        })
        plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)

        # Linia ciągła dla rzeczywistych cen zamknięcia
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Close'],
            mode='lines',
            name=f'{ticker} (Historia)',
            line=dict(color=colors[color_idx]),
            legendgroup=ticker
        ))
        # Linia przerywana dla predykcji
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Predicted_Close'],
            mode='lines',
            name=f'{ticker} (Predykcja)',
            line=dict(color=colors[color_idx], dash='dash'),
            legendgroup=ticker
        ))

        # Linia oddzielająca początek okresu predykcji
        split_date = pd.Timestamp(pred_dates[0]).isoformat()
        fig.add_shape(
            type="line",
            x0=split_date,
            x1=split_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=split_date,
            y=1.05,
            xref="x",
            yref="paper",
            text="Początek predykcji",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )

    # Ustawienia wykresu
    fig.update_layout(
        title="Porównanie predykcji z historią dla wybranych spółek",
        xaxis_title="Data",
        yaxis_title="Cena zamknięcia",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Wyświetlanie dokładności
    st.subheader("Procentowa zgodność predykcji z historią")
    for ticker, accuracy in accuracy_scores.items():
        st.write(f"{ticker}: {accuracy:.2f}%")

    return accuracy_scores

def save_benchmark_to_csv(benchmark_date, accuracy_scores):
    """Zapisuje wyniki benchmarku do pliku CSV z historią."""
    csv_file = 'data/benchmarks_history.csv'
    average_accuracy = np.mean(list(accuracy_scores.values())) if accuracy_scores else 0.0
    
    new_data = {'Date': [benchmark_date], **{ticker: [accuracy] for ticker, accuracy in accuracy_scores.items()}, 'Average_Accuracy': [average_accuracy]}
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, dtype=str)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.to_csv(csv_file, index=False)
    logger.info(f"Wyniki benchmarku zapisane do {csv_file}")

def load_benchmark_history(benchmark_tickers):
    """Wczytuje historię benchmarków z pliku CSV."""
    csv_file = 'data/benchmarks_history.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype=str)
        missing_tickers = [t for t in benchmark_tickers if t not in df.columns]
        for ticker in missing_tickers:
            df[ticker] = '0.0'
        return df[['Date'] + benchmark_tickers + ['Average_Accuracy']].fillna('0.0')
    return pd.DataFrame(columns=['Date'] + benchmark_tickers + ['Average_Accuracy']).fillna('0.0')