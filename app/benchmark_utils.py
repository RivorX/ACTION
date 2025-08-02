import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import logging
import os
import asyncio
from datetime import datetime
import tempfile
import glob
import time
from scripts.data_fetcher import DataFetcher
from scripts.utils.config_manager import ConfigManager
from scripts.prediction_engine import load_data_and_model, preprocess_data, generate_predictions
import torch

logger = logging.getLogger(__name__)

def clean_temp_dir(temp_dir):
    """Cleans all CSV files in the specified temporary directory."""
    try:
        csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        for file in csv_files:
            for _ in range(3):  # Try up to 3 times to handle file locking
                try:
                    os.remove(file)
                    logger.info(f"Removed temporary file: {file}")
                    break
                except PermissionError as e:
                    logger.warning(f"Failed to remove temporary file {file}: {e}. Retrying...")
                    time.sleep(0.1)  # Short delay to allow file release
                except Exception as e:
                    logger.error(f"Unexpected error while removing {file}: {e}")
                    break
    except Exception as e:
        logger.error(f"Error cleaning temporary directory {temp_dir}: {e}")

async def fetch_ticker_data(ticker, start_date, end_date):
    """Asynchronously fetches data for a single ticker."""
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        years = config['prediction']['years']
        fetcher = DataFetcher(config_manager, years)
        data = fetcher.fetch_stock_data_sync(ticker, start_date, end_date)
        if data.empty:
            logger.error(f"No data for {ticker}")
            return ticker, None
        return ticker, data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return ticker, None

async def process_ticker(ticker, full_data, config, temp_raw_data_path, max_prediction_length, trim_date, dataset, normalizers, model):
    """Asynchronously processes data for a single ticker with loaded model."""
    try:
        if full_data is None:
            logger.error(f"No data for {ticker}")
            return ticker, None

        full_data = full_data[full_data['Ticker'] == ticker].copy()
        full_data['Date'] = pd.to_datetime(full_data['Date'], utc=True)
        full_data.set_index('Date', inplace=True)
        historical_close = full_data['Close']

        # Trim data to trim_date for model
        new_data = full_data[full_data.index <= trim_date].copy()
        if new_data.empty:
            logger.error(f"No data before {trim_date} for {ticker}")
            return ticker, None

        new_data.reset_index().to_csv(temp_raw_data_path, index=False)
        logger.info(f"Data for {ticker} saved to {temp_raw_data_path}, długość: {len(new_data)}")

        # Preprocess data
        ticker_data, original_close = preprocess_data(config, new_data.reset_index(), ticker, normalizers, historical_mode=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            median, _, _ = generate_predictions(config, dataset, model, ticker_data)

        # Prepare dates and data
        last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=max_prediction_length, freq='D')

        # Trim historical data to pre-prediction period
        historical_dates = ticker_data['Date'].tolist()
        historical_close_trimmed = original_close.tolist()
        if len(historical_dates) != len(historical_close_trimmed):
            logger.error(f"Length mismatch: historical_dates ({len(historical_dates)}) and historical_close_trimmed ({len(historical_close_trimmed)}) for {ticker}")
            return ticker, None

        # Fetch data for prediction period
        historical_pred_close = historical_close.loc[trim_date:]
        if historical_pred_close.empty:
            logger.error(f"No historical data after {trim_date} for {ticker}")
            return ticker, None
        historical_pred_close = historical_pred_close.reindex(pd.to_datetime(pred_dates), method='ffill')
        if historical_pred_close.isna().any():
            logger.warning(f"NaN found in historical_pred_close for {ticker}. Filling with ffill and bfill.")
            historical_pred_close = historical_pred_close.ffill().bfill()
        if historical_pred_close.isna().any():
            logger.error(f"NaN persists in historical_pred_close for {ticker}")
            return ticker, None
        historical_pred_close = historical_pred_close.tolist()

        # Calculate metrics
        if len(median) == len(historical_pred_close):
            median = np.array(median)
            historical_pred_close_array = np.array(historical_pred_close)

            # Avoid zero denominators
            historical_pred_close_array = np.where(historical_pred_close_array == 0, 1e-6, historical_pred_close_array)

            # Accuracy (100 - MAPE)
            differences = np.abs(median - historical_pred_close_array)
            relative_diff = (differences / historical_pred_close_array) * 100
            if np.any(np.isnan(relative_diff)):
                logger.warning(f"NaN in relative_diff for {ticker}. Skipping NaN values in mean calculation.")
                relative_diff = relative_diff[~np.isnan(relative_diff)]
            accuracy = 100 - np.mean(relative_diff) if len(relative_diff) > 0 else 0.0

            # MAPE
            mape = np.mean(relative_diff) if len(relative_diff) > 0 else np.inf

            # MAE
            mae = np.mean(differences)

            # Directional Accuracy
            pred_changes = np.sign(np.diff(median))
            actual_changes = np.sign(np.diff(historical_pred_close_array))
            directional_accuracy = np.mean(pred_changes == actual_changes) * 100 if len(pred_changes) > 0 else 0.0

            logger.info(f"Metrics for {ticker}: Accuracy={accuracy:.2f}%, MAPE={mape:.2f}%, MAE={mae:.2f}, Directional Accuracy={directional_accuracy:.2f}%")
        else:
            logger.error(f"Mismatched prediction and historical data lengths for {ticker}: median={len(median)}, historical_pred_close={len(historical_pred_close)}")
            return ticker, None

        return ticker, {
            'historical_dates': historical_dates,
            'historical_close': historical_close_trimmed,
            'pred_dates': [d.to_pydatetime() for d in pred_dates],
            'predictions': median.tolist(),
            'historical_pred_close': historical_pred_close,
            'metrics': {
                'Acc': accuracy,
                'MAPE': mape,
                'MAE': mae,
                'DirAcc': directional_accuracy
            }
        }

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return ticker, None

async def create_benchmark_plot(config, benchmark_tickers, historical_close_dict, years):
    """Tworzy wykres benchmarku i oblicza metryki dla wielu tickerów asynchronicznie."""
    all_results = {}
    accuracy_scores = {}
    max_prediction_length = config['model']['max_prediction_length']
    trim_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=max_prediction_length)
    start_date = trim_date - pd.Timedelta(days=years * 365)
    model_name = config['model_name']
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    # Clean temp directory at start
    clean_temp_dir(temp_dir)

    # Pobieranie danych dla wszystkich tickerów
    logger.info("Pobieranie danych dla wszystkich tickerów...")
    config_manager = ConfigManager()  # Singleton
    tasks = [fetch_ticker_data(ticker, start_date, datetime.now()) for ticker in benchmark_tickers]
    ticker_data_results = await asyncio.gather(*tasks)
    ticker_data_dict = {ticker: data for ticker, data in ticker_data_results if data is not None}

    if not ticker_data_dict:
        logger.error("Nie udało się pobrać danych dla żadnego tickera.")
        return accuracy_scores

    # Wczytanie modelu raz
    logger.info("Wczytywanie modelu i danych...")
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=temp_dir)
        temp_raw_data_path = temp_file.name
        temp_file.close()
        first_ticker = next(iter(ticker_data_dict))
        first_data = ticker_data_dict[first_ticker]
        first_data.reset_index().to_csv(temp_raw_data_path, index=False)
        _, dataset, normalizers, model = load_data_and_model(config, first_ticker, temp_raw_data_path, historical_mode=True)
        logger.info(f"Model, dataset i normalizatory wczytane pomyślnie dla {first_ticker}")

        # Przetwarzanie tickerów asynchronicznie
        tasks = [
            process_ticker(ticker, data, config, temp_raw_data_path, max_prediction_length, trim_date, dataset, normalizers, model)
            for ticker, data in ticker_data_dict.items()
        ]
        results = await asyncio.gather(*tasks)

        for ticker, result in results:
            if result is not None and isinstance(result, dict):
                all_results[ticker] = result
                accuracy_scores[ticker] = result['metrics']['Acc']
            else:
                logger.warning(f"Pominięto ticker {ticker} z powodu niepoprawnych danych.")
                accuracy_scores[ticker] = 0.0

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

            combined_dates = historical_dates + pred_dates
            combined_close = historical_close + historical_pred_close
            combined_pred_close = [None] * len(historical_dates) + predictions

            if len(combined_dates) != len(combined_close) or len(combined_dates) != len(combined_pred_close):
                logger.error(f"Niezgodność długości dla {ticker}: combined_dates={len(combined_dates)}, combined_close={len(combined_close)}, combined_pred_close={len(combined_pred_close)}")
                continue

            plot_data = pd.DataFrame({
                'Date': combined_dates,
                'Close': combined_close,
                'Predicted_Close': combined_pred_close
            })
            plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)

            fig.add_trace(go.Scatter(
                x=plot_data['Date'],
                y=plot_data['Close'],
                mode='lines',
                name=f'{ticker} (Historia)',
                line=dict(color=colors[color_idx]),
                legendgroup=ticker
            ))
            fig.add_trace(go.Scatter(
                x=plot_data['Date'],
                y=plot_data['Predicted_Close'],
                mode='lines',
                name=f'{ticker} (Predykcja)',
                line=dict(color=colors[color_idx], dash='dash'),
                legendgroup=ticker
            ))

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

        # Wyświetlanie metryk w zwykłej tabeli
        st.subheader("Metryki predykcji dla każdej spółki")
        metrics_data = []
        for ticker, data in all_results.items():
            metrics = data['metrics']
            metrics_data.append({
                'Ticker': ticker,
                'Acc': metrics['Acc'],
                'MAPE': metrics['MAPE'],
                'MAE': metrics['MAE'],
                'DirAcc': metrics['DirAcc']
            })
        metrics_df = pd.DataFrame(metrics_data)

        if not metrics_df.empty:
            format_dict = {
                'Acc': '{:.2f}%',
                'MAPE': '{:.2f}%',
                'MAE': '{:.2f}',
                'DirAcc': '{:.2f}%'
            }
            st.dataframe(metrics_df.style.format(format_dict))

        return all_results

    finally:
        if 'model' in locals():
            del model
        if 'dataset' in locals():
            del dataset
        if 'normalizers' in locals():
            del normalizers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Pamięć GPU po sprzątaniu: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        if temp_file is not None and os.path.exists(temp_raw_data_path):
            for _ in range(3):
                try:
                    os.remove(temp_raw_data_path)
                    logger.info(f"Temporary file {temp_raw_data_path} removed.")
                    break
                except PermissionError as e:
                    logger.warning(f"Failed to remove temporary file {temp_raw_data_path}: {e}. Retrying...")
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Unexpected error while removing {temp_raw_data_path}: {e}")
                    break
        clean_temp_dir(temp_dir)

def save_benchmark_to_csv(benchmark_date, all_results, model_name):
    """Saves benchmark results to CSV with history, including all metrics and model name."""
    csv_file = 'data/benchmarks_history.csv'
    metrics = ['Acc', 'MAPE', 'MAE', 'DirAcc']
    columns = ['Date', 'Model_Name']
    for ticker in all_results.keys():
        for metric in metrics:
            columns.append(f"{ticker}_{metric}")
    columns.extend(['Avg_' + metric for metric in metrics])
    
    metrics_data = {'Date': [benchmark_date], 'Model_Name': [model_name]}
    valid_metrics = {}
    
    for ticker, data in all_results.items():
        if isinstance(data, dict) and 'metrics' in data:
            valid_metrics[ticker] = data['metrics']
            for metric in metrics:
                value = data['metrics'].get(metric, 0.0)
                metrics_data[f"{ticker}_{metric}"] = [value]
        else:
            logger.warning(f"No valid metrics data for {ticker}, setting default values.")
            for metric in metrics:
                metrics_data[f"{ticker}_{metric}"] = [0.0]
    
    if valid_metrics:
        for metric in metrics:
            values = [m.get(metric, 0.0) for m in valid_metrics.values() if m.get(metric, 0.0) != 0.0]
            metrics_data[f"Avg_{metric}"] = [np.mean(values) if values else 0.0]
    else:
        for metric in metrics:
            metrics_data[f"Avg_{metric}"] = [0.0]
    
    new_data = pd.DataFrame(metrics_data)
    
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, dtype=str)
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        updated_df = new_data
    
    updated_df.to_csv(csv_file, index=False)
    logger.info(f"Benchmark results saved to {csv_file}")

def load_benchmark_history(benchmark_tickers):
    """Loads benchmark history from CSV with nested headers."""
    csv_file = 'data/benchmarks_history.csv'
    metrics = ['Acc', 'MAPE', 'MAE', 'DirAcc']
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype=str)
        
        # Create MultiIndex for columns in desired order: Podstawowe, Średnie, then per-ticker metrics
        columns = ['Date', 'Model_Name']
        multi_columns = [('Podstawowe', 'Date'), ('Podstawowe', 'Model_Name')]
        for metric in metrics:
            columns.append(f"Avg_{metric}")
            multi_columns.append(('Średnie', metric))
        for ticker in benchmark_tickers:
            for metric in metrics:
                columns.append(f"{ticker}_{metric}")
                multi_columns.append((ticker, metric))
        multi_columns = pd.MultiIndex.from_tuples(multi_columns)
        
        # Select and rename columns
        df = df[columns].fillna('0.0')
        
        # Create DataFrame with MultiIndex
        df.columns = multi_columns
        
        # Convert to numeric for proper formatting
        for col in df.columns:
            if col[0] != 'Podstawowe':  # Skip Date and Model_Name
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        return df
    else:
        # Create empty DataFrame with MultiIndex
        multi_columns = pd.MultiIndex.from_tuples([('Podstawowe', 'Date'), ('Podstawowe', 'Model_Name')] + 
                                                 [('Średnie', metric) for metric in metrics] +
                                                 [(ticker, metric) for ticker in benchmark_tickers for metric in metrics])
        df = pd.DataFrame(columns=multi_columns).fillna('0.0')
        return df