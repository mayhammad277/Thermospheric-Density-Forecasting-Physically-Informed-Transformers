# src/improved_space_weather_pipeline.py
"""
Complete pipeline: load data, preprocess, train transformer, evaluate.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil, gc

# Add parent directory for imports (if run directly)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_engine import (
    load_csv_files, preprocess_omni, preprocess_goes,
    engineer_solar_wind_features, engineer_kp_features,
    engineer_goes_features, create_lags, add_solar_cycle_feature,
    add_dst_forecast_features, add_kp_forecast_features,
    align_and_merge, drop_flag_columns, smart_fill_nans,
    make_sequences
)
from src.transformer import TemporalDensityTransformer, train_model


def run_pipeline(data_dir: str, seq_len: int = 25, epochs: int = 100):
    print("Loading data...")
    champ_df    = load_csv_files(data_dir, exclude_prefixes=['initial_states', 'omni2', 'omni-', 'omni_', 'goes'])
    goes_df_raw = load_csv_files(data_dir, prefix='goes')
    omni_df_raw = load_csv_files(data_dir, prefix='omni')
    init_df     = load_csv_files(data_dir, prefix='initial_states')

    for name, df in [('champ', champ_df), ('goes', goes_df_raw),
                     ('omni', omni_df_raw), ('initial_states', init_df)]:
        if df.empty:
            raise RuntimeError(f"No files loaded for '{name}' in {data_dir}")

    omni_clean = preprocess_omni(omni_df_raw)
    goes_clean = preprocess_goes(goes_df_raw)

    print("Engineering features...")
    omni_clean = engineer_solar_wind_features(omni_clean)
    omni_clean = engineer_kp_features(omni_clean)
    omni_clean = add_solar_cycle_feature(omni_clean)
    omni_clean = add_dst_forecast_features(omni_clean)
    omni_clean = add_kp_forecast_features(omni_clean)
    omni_clean = create_lags(omni_clean, 'Kp_index', lags=[1, 2, 3, 4, 5, 7, 19, 38])
    goes_clean = engineer_goes_features(goes_clean)

    merged = align_and_merge(champ_df, omni_clean, goes_clean, init_df)
    merged = drop_flag_columns(merged)
    merged = smart_fill_nans(merged)

    target = 'Orbit Mean Density (kg/m^3)'
    if target not in merged.columns:
        raise KeyError(f"Target column '{target}' not found.")
    merged = merged.dropna(subset=[target])

    n = len(merged)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    train_df = merged.iloc[:train_end]
    val_df   = merged.iloc[train_end:val_end]
    test_df  = merged.iloc[val_end:]

    X_train, y_train, scaler = make_sequences(train_df, target, seq_len, fit_scaler=True)
    X_val,   y_val,   _      = make_sequences(val_df,   target, seq_len, scaler=scaler, fit_scaler=False)
    X_test,  y_test,  _      = make_sequences(test_df,  target, seq_len, scaler=scaler, fit_scaler=False)

    input_dim = X_train.shape[2]
    model = TemporalDensityTransformer(input_dim=input_dim, seq_len=seq_len,
                                       num_heads=4, num_layers=3,
                                       hidden_dim=128, dropout=0.1)
    print("Training...")
    model, history = train_model(model, X_train, y_train, X_val, y_val,
                                 epochs=epochs, log_target=True)

    plt.figure(figsize=(10, 4))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE (log-space)')
    plt.title('Training History — Thermospheric Density Model')
    plt.legend(); plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

    return model, scaler, merged


if __name__ == '__main__':
    DATA_DIR = '/content/'   # modify as needed
    model, scaler, merged = run_pipeline(DATA_DIR, seq_len=25, epochs=100)

    # Optional: memory usage check
    print(f"\nRAM usage: {psutil.Process().memory_info().rss / 1024**2:.0f} MB")
