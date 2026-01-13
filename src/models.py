"""
Model definitions and inference helpers for the AAPL forecasting project.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from keras.models import load_model as keras_load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# ---------- LSTM model definition and training ----------

def build_lstm(input_shape: Tuple[int, int]) -> Sequential:
    """
    Build a simple LSTM model.

    input_shape: (time_steps, num_features)
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(model, X_train, y_train, epochs: int, batch_size: int, verbose: int = 1):
    """
    Train the given LSTM model.
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_split=0.1,
    )
    return history


# ---------- Model loading helpers for API ----------

def get_model_path(ticker: str) -> Path:
    """
    Return the path where the trained model for a given ticker is stored.
    Adjust this to match how/where you save models.
    """
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    # Example: models/AAPL_lstm.h5
    return models_dir / f"{ticker.upper()}_lstm.h5"


def load_trained_model(ticker: str):
    """
    Load the trained LSTM model for the given ticker.

    If you use a different format (e.g. joblib for RandomForest),
    adjust this function accordingly.
    """
    model_path = get_model_path(ticker)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found for {ticker} at {model_path}. "
            "Train and save the model before calling the API."
        )
    model = keras_load_model(model_path)
    return model


# ---------- Forecast helper ----------

def make_forecast(model, X_last, horizon: int = 1) -> float:
    """
    Multi-step forecast using an already prepared feature window X_last.

    Parameters
    ----------
    model : trained model
        Your LSTM (or other) model that predicts next-step price.
    X_last : np.ndarray or similar
        Latest input window with shape (1, time_steps, num_features).
    horizon : int
        How many steps ahead to forecast. Horizon=1 returns the next-step prediction.

    Returns
    -------
    float
        Predicted price at the given horizon.
    """
    # Ensure X_last has correct shape
    X = np.array(X_last)
    if X.ndim == 2:
        # assume (time_steps, num_features) -> add batch axis
        X = X[np.newaxis, ...]

    current_window = X.copy()
    last_pred = None

    for _ in range(horizon):
        # One-step prediction
        y_hat = model.predict(current_window, verbose=0)[0, 0]
        last_pred = float(y_hat)

        # Shift window and append prediction as the latest step
        # This is a simple autoregressive-style update; customize if needed.
        new_step = current_window[:, -1, :].copy()
        # If the first feature is Close price, update it with prediction
        new_step[:, 0] = y_hat
        current_window = np.concatenate(
            [current_window[:, 1:, :], new_step.reshape(1, 1, -1)],
            axis=1,
        )

    return last_pred

from datetime import date, timedelta
import pandas as pd


def forecast_next_days(ticker: str, days: int):
    """
    Simple helper for the API.

    For now:
    - loads the trained LSTM model from get_model_path(ticker)
    - loads the last window features from a CSV (you need to adapt this)
    - uses make_forecast to get a point forecast for each future day

    Replace the 'load X_last' part with your actual feature window loading.
    """
    if ticker.upper() != "AAPL":
        raise ValueError("Only AAPL is supported for now.")

    # TODO: replace this with your real last-window loading
    # For now, just raise if you don't have it
    last_window_path = Path("artifacts") / "last_window.npy"
    if not last_window_path.exists():
        raise ValueError(
            "last_window.npy not found in artifacts/. "
            "Save your latest input window there after training."
        )

    X_last = np.load(last_window_path)  # shape (1, time_steps, num_features)
    model = load_trained_model(ticker)

    # Build a simple date index: today + 1..days (business days only)
    today = date.today()
    dates = []
    d = today
    while len(dates) < days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri only
            dates.append(d)

    preds = []
    for i in range(days):
        y_hat = make_forecast(model, X_last, horizon=i + 1)
        preds.append({"date": dates[i], "value": float(y_hat)})

    meta = {
        "version": "lstm_lag_forecaster_v1",
        "train_start": None,
        "train_end": None,
        "cutoff": today,
    }

    return preds, meta
