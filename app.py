import streamlit as st
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from scripts.model import build_model
import torch
import yaml
import plotly.graph_objs as go

st.title("Stock Price Predictor")

# Wczytaj konfigurację
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

tickers = config['data']['tickers']
selected_ticker = st.selectbox("Select a company", tickers)

if st.button("Predict"):
    # Wczytaj dane i model
    dataset = TimeSeriesDataSet.load(config['data']['processed_data_path'])
    model = build_model(dataset, config)
    model.load_state_dict(torch.load(config['paths']['model_save_path'], map_location=torch.device('cpu')))
    model.eval()
    model.to('cpu')  # Przenieś model na CPU dla predykcji

    # Filtruj dane dla wybranego tickera
    data = pd.read_csv(config['data']['raw_data_path'])
    ticker_data = data[data['Ticker'] == selected_ticker].tail(100)
    
    # Przygotuj dataset dla predykcji
    ticker_dataset = TimeSeriesDataSet.from_parameters(
        dataset.get_parameters(),
        ticker_data,
        predict=True,
        mode="prediction",
        max_prediction_length=config['model']['max_prediction_length']
    )
    
    # Wykonaj predykcję
    with torch.no_grad():
        predictions = model.predict(ticker_dataset, return_x=True)
    
    # Konwertuj predykcje na numpy
    pred_array = predictions.output.cpu().numpy()  # Kształt: (1, max_prediction_length, n_quantiles)
    
    # Pobierz kwantyle z konfiguracji
    quantiles = config['model'].get('quantiles', [0.1, 0.5, 0.9])
    n_quantiles = len(quantiles)
    
    # Przygotuj dane do wykresu
    time_steps = np.arange(1, config['model']['max_prediction_length'] + 1)
    median = pred_array[0, :, quantiles.index(0.5)] if 0.5 in quantiles else pred_array[0, :, 1]
    lower_bound = pred_array[0, :, quantiles.index(0.1)] if 0.1 in quantiles else median
    upper_bound = pred_array[0, :, quantiles.index(0.9)] if 0.9 in quantiles else median

    # Twórz wykres z przedziałem ufności
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=median,
        mode='lines',
        name='Mediana (50%)',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=upper_bound,
        mode='lines',
        name='Górny kwantyl (90%)',
        line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=lower_bound,
        mode='lines',
        name='Dolny kwantyl (10%)',
        line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    fig.update_layout(
        title=f"Przewidywane ceny dla {selected_ticker} (następne {config['model']['max_prediction_length']} dni)",
        xaxis_title="Dni w przyszłości",
        yaxis_title="Cena zamknięcia",
        showlegend=True
    )
    
    st.plotly_chart(fig)