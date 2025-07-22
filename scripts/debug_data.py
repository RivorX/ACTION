#!/usr/bin/env python3
"""
Narzędzie do debugowania i sprawdzania jakości danych przed treningiem.
Sprawdza problemy z NaN, inf, i innymi problemami w danych.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import sys
import os

# Dodaj katalog główny do ścieżek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config_manager import ConfigManager

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_quality(data_path: str, config_path: str = "config/config.yaml"):
    """Analizuje jakość danych i szuka problemów."""
    
    # Wczytaj konfigurację
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    
    # Wczytaj dane
    logger.info(f"Wczytywanie danych z: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Wczytano {len(df)} wierszy i {len(df.columns)} kolumn")
    
    # Podstawowe statystyki
    logger.info("\n=== PODSTAWOWE STATYSTYKI ===")
    logger.info(f"Zakres dat: {df['Date'].min()} - {df['Date'].max()}")
    logger.info(f"Liczba tickerów: {df['Ticker'].nunique()}")
    logger.info(f"Tickery: {df['Ticker'].unique().tolist()}")
    
    # Analiza cech fundamentalnych
    fundamental_cols = ['PE_ratio', 'PB_ratio', 'EPS']
    existing_fundamental = [col for col in fundamental_cols if col in df.columns]
    
    logger.info(f"\n=== ANALIZA CECH FUNDAMENTALNYCH ===")
    logger.info(f"Dostępne cechy fundamentalne: {existing_fundamental}")
    
    for col in existing_fundamental:
        logger.info(f"\n--- Analiza {col} ---")
        series = df[col]
        
        # Podstawowe statystyki
        logger.info(f"Rozmiar: {len(series)}")
        logger.info(f"Typ danych: {series.dtype}")
        logger.info(f"Wartości NaN: {series.isna().sum()} ({100*series.isna().sum()/len(series):.1f}%)")
        logger.info(f"Wartości inf: {np.isinf(series).sum()}")
        logger.info(f"Wartości zero: {(series == 0.0).sum()} ({100*(series == 0.0).sum()/len(series):.1f}%)")
        logger.info(f"Wartości ujemne: {(series < 0).sum()}")
        logger.info(f"Unikalne wartości: {series.nunique()}")
        
        if not series.isna().all():
            valid_values = series.dropna()
            if len(valid_values) > 0:
                logger.info(f"Min: {valid_values.min():.6f}")
                logger.info(f"Max: {valid_values.max():.6f}")
                logger.info(f"Średnia: {valid_values.mean():.6f}")
                logger.info(f"Mediana: {valid_values.median():.6f}")
                logger.info(f"Std: {valid_values.std():.6f}")
                
                # Top wartości
                value_counts = valid_values.value_counts().head(10)
                logger.info(f"Top 10 wartości:")
                for value, count in value_counts.items():
                    logger.info(f"  {value:.6f}: {count} razy ({100*count/len(valid_values):.1f}%)")
        
        # Sprawdź missing flags
        missing_col = f'{col}_missing'
        if missing_col in df.columns:
            missing_series = df[missing_col]
            logger.info(f"Kolumna {missing_col}: {missing_series.sum()} oznaczonych jako missing ({100*missing_series.sum()/len(missing_series):.1f}%)")
            
            # Sprawdź zgodność
            actual_zeros = (series == 0.0)
            marked_missing = (missing_series == 1)
            agreement = (actual_zeros == marked_missing).sum()
            logger.info(f"Zgodność między zerami a flagą missing: {agreement}/{len(series)} ({100*agreement/len(series):.1f}%)")
    
    # Analiza cech technicznych
    logger.info(f"\n=== ANALIZA CECH TECHNICZNYCH ===")
    technical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Volatility']
    for col in technical_cols:
        if col in df.columns:
            series = df[col]
            nan_count = series.isna().sum()
            inf_count = np.isinf(series).sum()
            zero_count = (series == 0.0).sum()
            
            if nan_count > 0 or inf_count > 0 or zero_count > len(series) * 0.5:  # Jeśli więcej niż 50% to zera
                logger.warning(f"{col}: NaN={nan_count}, inf={inf_count}, zeros={zero_count} ({100*zero_count/len(series):.1f}%)")
            else:
                logger.info(f"{col}: OK (NaN={nan_count}, inf={inf_count}, zeros={zero_count})")
    
    # Sprawdź duplikaty
    logger.info(f"\n=== ANALIZA DUPLIKATÓW ===")
    duplicates = df.duplicated()
    logger.info(f"Duplikaty wierszy: {duplicates.sum()}")
    
    # Sprawdź luki czasowe
    logger.info(f"\n=== ANALIZA LUK CZASOWYCH ===")
    df['Date'] = pd.to_datetime(df['Date'])
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Oblicz różnice między kolejnymi datami
        date_diffs = ticker_data['Date'].diff().dt.days
        large_gaps = date_diffs > 7  # Luki większe niż tydzień
        
        if large_gaps.any():
            gap_count = large_gaps.sum()
            max_gap = date_diffs.max()
            logger.warning(f"{ticker}: {gap_count} dużych luk czasowych (max: {max_gap} dni)")
        else:
            logger.info(f"{ticker}: Brak dużych luk czasowych")
    
    # Rekomendacje
    logger.info(f"\n=== REKOMENDACJE ===")
    
    issues_found = []
    
    for col in existing_fundamental:
        series = df[col]
        zero_percent = 100*(series == 0.0).sum()/len(series)
        
        if zero_percent > 95:
            issues_found.append(f"Cecha {col} ma {zero_percent:.1f}% zer - rozważ usunięcie z modelu")
        elif zero_percent > 80:
            issues_found.append(f"Cecha {col} ma {zero_percent:.1f}% zer - może powodować problemy z normalizacją")
        
        if series.nunique() <= 1:
            issues_found.append(f"Cecha {col} ma tylko {series.nunique()} unikalnych wartości - usuń z modelu")
    
    if issues_found:
        logger.warning("Znalezione problemy:")
        for issue in issues_found:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Nie znaleziono poważnych problemów z danymi!")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizuj jakość danych")
    parser.add_argument("--data", default="data/stock_data.csv", help="Ścieżka do pliku danych")
    parser.add_argument("--config", default="config/config.yaml", help="Ścieżka do konfiguracji")
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        logger.error(f"Plik danych nie istnieje: {args.data}")
        sys.exit(1)
    
    analyze_data_quality(args.data, args.config)
