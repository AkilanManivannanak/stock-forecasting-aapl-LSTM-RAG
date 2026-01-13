import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from src.data import (
    load_data,
    prepare_data,
    build_rf_feature_matrix,
    get_last_window,
)
from src.features import add_technical_indicators


# ---------------- Config ----------------

TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

TIME_STEP = 60
TEST_RATIO = 0.2
SEED = 42
N_LAGS = 10

np.random.seed(SEED)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ---------------- Utility ----------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_regime(close_series) -> str:
    """
    Simple regime classification based on last 252-day return.
    Robust to Series / DataFrame column / list input.
    """
    if isinstance(close_series, pd.Series):
        s = close_series
    elif isinstance(close_series, pd.DataFrame):
        s = close_series.iloc[:, 0]
    else:
        s = pd.Series(close_series)

    s = pd.to_numeric(s, errors="coerce")

    if s.shape[0] < 252:
        return "unknown"

    last = s.iloc[-1]
    prev = s.iloc[-252]

    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return "unknown"

    ret_1y = last / prev - 1.0

    if ret_1y > 0.2:
        return "bull"
    elif ret_1y < -0.2:
        return "bear"
    else:
        return "sideways"


# ---------------- Main ----------------

def main():
    # 1. Load data
    df_raw = load_data(TICKER, START_DATE, END_DATE)
    print(f"Downloaded {len(df_raw)} rows for {TICKER}.")

    # 2. LSTM price forecasting (Close)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df_raw, TIME_STEP, TEST_RATIO)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_pred_inv = scaler.inverse_transform(train_pred)
    test_pred_inv = scaler.inverse_transform(test_pred)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse_lstm = rmse(y_test_inv, test_pred_inv)
    print(f"LSTM RMSE (Close): {rmse_lstm:.4f}")

    # save LSTM model for serving in the format the API expects
    from src.models import get_model_path
    from config import STOCK_SYMBOL

    model_path = get_model_path(STOCK_SYMBOL)  # e.g., models/AAPL_lstm.h5
    model.save(model_path)
    print(f"Saved LSTM model to {model_path}")

    # plot LSTM actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(df_raw.index[-len(y_test_inv):], y_test_inv, label="Actual Close")
    plt.plot(df_raw.index[-len(test_pred_inv):], test_pred_inv, label="LSTM Predicted Close")
    plt.title(f"{TICKER} Close Price - LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_pred_lstm.jpg")
    plt.close()

    # 3. Naive baseline: last close
    last_close_pred = df_raw["Close"].shift(1).iloc[-len(y_test_inv):].values.reshape(-1, 1)
    rmse_naive = rmse(y_test_inv, last_close_pred)
    print(f"Naive RMSE (Close): {rmse_naive:.4f}")

    # 4. Simple MA_5 baseline
    ma_5 = df_raw["Close"].rolling(window=5).mean()
    ma_5_pred = ma_5.shift(1).iloc[-len(y_test_inv):].values.reshape(-1, 1)
    rmse_ma_5 = rmse(y_test_inv, ma_5_pred)
    print(f"MA_5 RMSE (Close): {rmse_ma_5:.4f}")

    # 5. RandomForest regression on indicators for price
    X_rf, y_rf = build_rf_feature_matrix(df_raw, n_lags=N_LAGS)
    split_rf = int(len(X_rf) * (1 - TEST_RATIO))
    X_rf_train, X_rf_test = X_rf[:split_rf], X_rf[split_rf:]
    y_rf_train, y_rf_test = y_rf[:split_rf], y_rf[split_rf:]

    rf_price = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1,
    )
    rf_price.fit(X_rf_train, y_rf_train)
    y_rf_pred = rf_price.predict(X_rf_test)
    rmse_rf = rmse(y_rf_test, y_rf_pred)
    print(f"RandomForest RMSE (Close): {rmse_rf:.4f}")

    # save RF model for serving
    joblib.dump(rf_price, "models/rf_price.pkl")
    print("Saved RF price model to models/rf_price.pkl")

    # 6. Save metrics for models
    metrics_df = pd.DataFrame(
        {
            "model": ["naive", "lstm", "random_forest", "ma_5"],
            "rmse": [rmse_naive, rmse_lstm, rmse_rf, rmse_ma_5],
        }
    )
    metrics_df.to_csv("metrics.csv", index=False)
    print("Saved metrics.csv")

    # 7. Future forecasting with LSTM
    FUTURE_DAYS = 30
    last_window = get_last_window(df_raw, TIME_STEP, scaler)
    future_preds_scaled = []
    curr_window = last_window.copy()

    for _ in range(FUTURE_DAYS):
        pred_scaled = model.predict(curr_window)
        future_preds_scaled.append(pred_scaled[0, 0])
        new_window = np.append(curr_window[0, 1:, 0], pred_scaled[0, 0])
        curr_window = new_window.reshape(1, TIME_STEP, 1)

    future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
    future_preds = scaler.inverse_transform(future_preds_scaled).flatten()

    last_date = df_raw.index[-1]
    future_dates = []
    d = last_date
    while len(future_dates) < FUTURE_DAYS:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)

    future_forecast_df = pd.DataFrame({"date": future_dates, "forecast_close": future_preds})
    future_forecast_df.to_csv("future_forecast_lstm.csv", index=False)
    print("Saved future_forecast_lstm.csv")

    # 8. Walk-forward RF vs naive (regime-aware RMSE)
    window_size = 252
    step_size = 126
    walk_metrics = []

    for start in range(0, len(df_raw) - window_size, step_size):
        end = start + window_size
        window_df = df_raw.iloc[start:end]
        regime = compute_regime(window_df["Close"])

        X_wf, y_wf = build_rf_feature_matrix(window_df, n_lags=N_LAGS)
        if len(X_wf) < 50:
            continue

        split_wf = int(len(X_wf) * 0.8)
        X_wf_train, X_wf_test = X_wf[:split_wf], X_wf[split_wf:]
        y_wf_train, y_wf_test = y_wf[:split_wf], y_wf[split_wf:]

        rf_wf = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=SEED,
            n_jobs=-1,
        )
        rf_wf.fit(X_wf_train, y_wf_train)
        y_wf_pred = rf_wf.predict(X_wf_test)
        rmse_wf = rmse(y_wf_test, y_wf_pred)

        walk_metrics.append(
            {
                "window_id": len(walk_metrics),
                "start_date": window_df.index[0].strftime("%Y-%m-%d"),
                "end_date": window_df.index[-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "model": "random_forest",
                "rmse": rmse_wf,
            }
        )

        naive_pred_wf = window_df["Close"].shift(1).iloc[-len(y_wf_test):].values
        rmse_naive_wf = rmse(y_wf_test, naive_pred_wf)
        walk_metrics.append(
            {
                "window_id": len(walk_metrics),
                "start_date": window_df.index[0].strftime("%Y-%m-%d"),
                "end_date": window_df.index[-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "model": "naive",
                "rmse": rmse_naive_wf,
            }
        )

    if walk_metrics:
        walk_df = pd.DataFrame(walk_metrics)
        walk_df.to_csv("walkforward_metrics.csv", index=False)
        print("Saved walkforward_metrics.csv")


if __name__ == "__main__":
    main()

