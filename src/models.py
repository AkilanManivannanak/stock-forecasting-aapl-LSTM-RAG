# src/models.py
"""
Model helpers for Stock Forecasting API (Keras 3 compatible).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import joblib

# IMPORTANT: Keras 3 imports (NOT tf_keras)
from keras.models import load_model as keras_load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_lstm(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def get_model_path(ticker: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir / f"{ticker.upper()}_lstm.h5"


def load_trained_model(ticker: str):
    p = get_model_path(ticker)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}. Train first (python -m scripts.train).")
    # compile=False avoids metric warnings & reduces load issues
    return keras_load_model(p, compile=False)


def make_forecast(model, X_last, horizon: int = 1) -> float:
    X = np.array(X_last)
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    current_window = X.copy()
    last_pred = None

    for _ in range(horizon):
        y_hat = float(model.predict(current_window, verbose=0)[0, 0])
        last_pred = y_hat

        # autoregressive roll-forward
        new_step = current_window[:, -1, :].copy()
        new_step[:, 0] = y_hat
        current_window = np.concatenate([current_window[:, 1:, :], new_step.reshape(1, 1, -1)], axis=1)

    return float(last_pred)
