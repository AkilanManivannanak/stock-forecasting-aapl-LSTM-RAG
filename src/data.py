# src/data.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from src.features import add_technical_indicators


# ---------- Data loading ----------

def load_data(stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLCV data for the given symbol and date range.
    """
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.sort_index(inplace=True)
    return df


# ---------- LSTM data prep (univariate on Close) ----------

def create_sequences(scaled_data: np.ndarray, time_step: int):
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def train_test_split_series(X, y, test_ratio: float = 0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def prepare_data(df: pd.DataFrame, time_step: int, test_ratio: float = 0.2):
    """
    Scale Close prices, create LSTM sequences, split train/test, return scaler.
    """
    close_df = df[["Close"]].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_df.values)

    X, y = create_sequences(scaled, time_step)
    X_train, X_test, y_train, y_test = train_test_split_series(X, y, test_ratio)

    return X_train, X_test, y_train, y_test, scaler


# ---------- Simple lag features (optional) ----------

def build_tabular_lag_features(df: pd.DataFrame, n_lags: int = 10):
    """
    Basic lag-only features on Close.
    """
    data = df["Close"].values.astype(float)
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i - n_lags:i])
        y.append(data[i])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    return X, y


# ---------- RF feature matrix with technical indicators (price target) ----------

def build_rf_feature_matrix(df: pd.DataFrame, n_lags: int = 5):
    """
    Create a feature matrix for RandomForest that combines:
    - OHLCV
    - returns, volatility
    - technical indicators (MA, RSI, MACD, Bollinger)
    - lagged closes
    Target is next-day Close.
    """
    data = add_technical_indicators(df)

    for lag in range(1, n_lags + 1):
        data[f"lag_{lag}"] = data["Close"].shift(lag)

    data["target_close"] = data["Close"].shift(-1)
    data = data.dropna()

    feature_cols = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "return_1d",
        "log_return_1d",
        "ma_10",
        "ma_20",
        "ma_50",
        "vol_10",
        "vol_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
    ] + [f"lag_{i}" for i in range(1, n_lags + 1)]

    X = data[feature_cols].values.astype(float)
    y = data["target_close"].values.astype(float)

    return X, y


# ---------- RF feature matrix for return prediction (using log_return_1d) ----------

def build_return_feature_matrix_from_indicators(
    df_with_indicators: pd.DataFrame,
    n_lags: int = 5,
):
    """
    Build X, y, idx for predicting next-day log return using:
    - existing indicator features
    - log_return_1d as target

    Here target_log_ret_1d is log_return_1d shifted by -1 day.[web:158]
    """
    data = df_with_indicators.copy()

    # target = next-day log return
    if "log_return_1d" not in data.columns:
        raise ValueError("log_return_1d not found. Ensure add_technical_indicators was applied.")

    data["target_log_ret_1d"] = data["log_return_1d"].shift(-1)

    base_cols = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "return_1d",
        "log_return_1d",
        "ma_10",
        "ma_20",
        "ma_50",
        "vol_10",
        "vol_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
    ]

    # keep only base columns that actually exist
    feature_cols = [c for c in base_cols if c in data.columns]

    # add lagged closes
    for lag in range(1, n_lags + 1):
        col = f"lag_{lag}"
        if "Close" in data.columns and col not in data.columns:
            data[col] = data["Close"].shift(lag)
        if col in data.columns:
            feature_cols.append(col)

    # --- FIX: build subset only from columns that really exist ---
    existing_cols = set(data.columns)
    subset_cols = [c for c in feature_cols if c in existing_cols]
    if "target_log_ret_1d" in existing_cols:
        subset_cols.append("target_log_ret_1d")

    # if nothing to subset on, raise clear error
    if not subset_cols:
        raise ValueError("No valid columns found to build return feature matrix.")

    data = data.dropna(subset=subset_cols)

    # recompute feature_cols after dropna, using only existing columns
    feature_cols = [c for c in feature_cols if c in data.columns]

    X = data[feature_cols].values.astype(float)
    y = data["target_log_ret_1d"].values.astype(float)
    index = data.index

    return X, y, index



# ---------- For future forecasting with LSTM ----------

def get_last_window(df: pd.DataFrame, time_step: int, scaler):
    """
    Return last `time_step` days as scaled 3D array [1, time_step, 1].
    """
    data = df[["Close"]].values
    scaled = scaler.transform(data)
    last_window = scaled[-time_step:]
    last_window = last_window.reshape(1, time_step, 1)
    return last_window
