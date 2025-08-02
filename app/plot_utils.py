import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def create_base_plot(title, xaxis_title="Data", yaxis_title="Cena zamknięcia", split_date=None):
    """Creates a base Plotly figure with common settings."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers")
    )
    if split_date:
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
    return fig

def create_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker, historical_close=None):
    """
    Creates a plot of stock prices and predictions, with optional historical comparison.

    Args:
        config (dict): Configuration dictionary.
        ticker_data (pd.DataFrame): DataFrame with ticker data.
        original_close (pd.Series): Historical close prices.
        median (np.ndarray): Predicted median prices.
        lower_bound (np.ndarray): Lower quantile predictions.
        upper_bound (np.ndarray): Upper quantile predictions.
        ticker (str): Ticker symbol.
        historical_close (pd.Series, optional): Historical close prices for comparison.

    Returns:
        None: Displays the plot and optional prediction table in Streamlit.
    """
    last_date = ticker_data['Date'].iloc[-1].to_pydatetime()
    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=config['model']['max_prediction_length'], freq='D')
    historical_dates = ticker_data['Date'].tolist()

    if historical_close is not None:
        # Historical comparison mode
        pred_date_range = pd.DataFrame({'Date': pred_dates})
        pred_date_range['Date'] = pd.to_datetime(pred_date_range['Date'], utc=True)
        historical_close = historical_close.reindex(pred_date_range['Date'], method='ffill')
        
        logger.info(f"Długość historical_dates: {len(historical_dates)}")
        logger.info(f"Długość pred_dates: {len(pred_dates)}")
        logger.info(f"Długość original_close: {len(original_close)}")
        logger.info(f"Długość historical_close po reindex: {len(historical_close)}")
        logger.info(f"Długość median: {len(median)}")
        
        combined_dates = historical_dates + pred_date_range['Date'].tolist()
        combined_close = original_close.tolist() + historical_close.tolist()
        combined_pred_close = [None] * len(historical_dates) + median.tolist()
        
        if len(combined_dates) != len(combined_close) or len(combined_dates) != len(combined_pred_close):
            logger.error(f"Niezgodność długości: combined_dates={len(combined_dates)}, combined_close={len(combined_close)}, combined_pred_close={len(combined_pred_close)}")
            raise ValueError("Wszystkie tablice muszą mieć tę samą długość")

        plot_data = pd.DataFrame({
            'Date': combined_dates,
            'Close': combined_close,
            'Predicted_Close': combined_pred_close
        })
        plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)
        
        fig = create_base_plot(f"Porównanie predykcji z historią dla {ticker}", split_date=pd.Timestamp(last_date).isoformat())
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Close'],
            mode='lines',
            name='Cena zamknięcia (historyczna)',
            line=dict(color='#0000FF')
        ))
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Predicted_Close'],
            mode='lines',
            name='Przewidywana cena zamknięcia',
            line=dict(color='#FFA500', dash='dash')
        ))
    else:
        # Future prediction mode
        logger.info(f"Długość historical_dates: {len(historical_dates)}")
        logger.info(f"Długość pred_dates: {len(pred_dates)}")
        logger.info(f"Długość original_close: {len(original_close)}")
        logger.info(f"Długość median: {len(median)}")
        logger.info(f"Długość lower_bound: {len(lower_bound)}")
        logger.info(f"Długość upper_bound: {len(upper_bound)}")
        
        # Dopasuj długości, obcinając początkowe dni, jeśli różnica <= 10
        max_trim = 10
        if len(original_close) > len(historical_dates) and len(original_close) - len(historical_dates) <= max_trim:
            trim_count = len(original_close) - len(historical_dates)
            logger.warning(f"Obcinanie {trim_count} początkowych dni z original_close, lower_bound i upper_bound dla {ticker}")
            original_close = original_close[trim_count:]
            combined_lower_bound = [None] * len(historical_dates) + lower_bound.tolist()
            combined_upper_bound = [None] * len(historical_dates) + upper_bound.tolist()
        else:
            combined_lower_bound = [None] * len(original_close) + lower_bound.tolist()
            combined_upper_bound = [None] * len(original_close) + upper_bound.tolist()
        
        combined_dates = historical_dates + [d.to_pydatetime() for d in pred_dates]
        combined_close = original_close.tolist() + median.tolist()
        
        if not (len(combined_dates) == len(combined_close) == len(combined_lower_bound) == len(combined_upper_bound)):
            logger.error(f"Niezgodność długości: combined_dates={len(combined_dates)}, combined_close={len(combined_close)}, combined_lower_bound={len(combined_lower_bound)}, combined_upper_bound={len(combined_upper_bound)}")
            raise ValueError("Wszystkie tablice muszą mieć tę samą długość")

        plot_data = pd.DataFrame({
            'Date': combined_dates,
            'Close': combined_close,
            'Lower_Bound': combined_lower_bound,
            'Upper_Bound': combined_upper_bound
        })
        plot_data['Date'] = pd.to_datetime(plot_data['Date'], utc=True)
        
        fig = create_base_plot(f"Ceny akcji dla {ticker}", split_date=pd.Timestamp(last_date).isoformat())
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Close'],
            mode='lines',
            name='Cena zamknięcia (historyczna i przewidywana)',
            line=dict(color='#0000FF')
        ))
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Upper_Bound'],
            mode='lines',
            name='Górny kwantyl (90%)',
            line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['Lower_Bound'],
            mode='lines',
            name='Dolny kwantyl (10%)',
            line=dict(color='rgba(0, 0, 255, 0.3)', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))

    st.plotly_chart(fig, use_container_width=True)
    
    if historical_close is None:
        pred_df = pd.DataFrame({
            'Data': [d.to_pydatetime() for d in pred_dates],
            'Przewidywana cena': median.tolist(),
            'Dolny kwantyl (10%)': lower_bound.tolist(),
            'Górny kwantyl (90%)': upper_bound.tolist()
        })
        st.subheader("Przewidywane ceny")
        st.dataframe(pred_df.style.format({
            'Data': '{:%Y-%m-%d}',
            'Przewidywana cena': '{:.2f}',
            'Dolny kwantyl (10%)': '{:.2f}',
            'Górny kwantyl (90%)': '{:.2f}'
        }))