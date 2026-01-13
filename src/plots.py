# src/plots.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


def plot_actual_vs_pred(y_true, y_pred, title: str, plots_dir: str, test_start=None, test_end=None):
    os.makedirs(plots_dir, exist_ok=True)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    plt.figure(figsize=(16, 6))
    plt.plot(y_true, label="Actual Close", color="tab:blue", linewidth=1.5)
    plt.plot(y_pred, label="LSTM Predicted Close", color="tab:orange", linewidth=2, alpha=0.8)

    # error band
    error = np.abs(y_true - y_pred)
    plt.fill_between(
        np.arange(len(y_true)),
        y_true - error,
        y_true + error,
        color="tab:orange",
        alpha=0.08,
        label="|Error band|",
    )

    # build a richer title with dates
    if test_start is not None and test_end is not None:
        full_title = f"{title} – Test Set ({test_start:%Y-%m-%d} to {test_end:%Y-%m-%d})"
    else:
        full_title = f"{title} – Test Set"

    plt.title(full_title, fontsize=16)
    plt.xlabel("Test Time Index (days)", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)

    ymin, ymax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.ylim(ymin - 2, ymax + 2)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=12)

    mae = float(np.mean(error))
    text_lines = [
        f"Test length: {len(y_true)} points",
        f"Mean abs error: {mae:.2f} USD",
    ]
    if test_start is not None and test_end is not None:
        text_lines.insert(0, f"Test period: {test_start:%Y-%m-%d} → {test_end:%Y-%m-%d}")
    text = "\n".join(text_lines)

    plt.annotate(
        text,
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        alpha=0.8,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6),
    )

    out_path = os.path.join(plots_dir, "actual_vs_pred_lstm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")
def plot_future_forecast(df_history: pd.DataFrame, forecast_df: pd.DataFrame, plots_dir: str, ticker: str):
    """
    Plot recent historical closes plus LSTM future forecast.[file:151]
    df_history: full historical df with 'Close' and DateTimeIndex.
    forecast_df: dataframe from future_forecast_lstm.csv with 'date' and 'forecast_close'.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # use last 180 days of history
    hist_tail = df_history.tail(180)

    plt.figure(figsize=(16, 6))
    plt.plot(
        hist_tail.index,
        hist_tail["Close"].values,
        label="Historical Close",
        color="tab:blue",
        linewidth=1.5,
    )

    plt.plot(
        forecast_df["date"],
        forecast_df["forecast_close"],
        label="LSTM Forecast Close",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
    )

    plt.title(f"{ticker} – Recent History and {len(forecast_df)}‑Day LSTM Forecast", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)

    out_path = os.path.join(plots_dir, "future_forecast_lstm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved future forecast plot to {out_path}")

def plot_single_day_forecast(df_history: pd.DataFrame, forecast_df: pd.DataFrame, plots_dir: str, ticker: str):
    """
    Visualize the next single-day forecast: last actual close vs first forecasted close.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # ensure scalar floats
    last_actual_date = df_history.index[-1]
    last_actual_close = float(df_history["Close"].iloc[-1])

    first_forecast_row = forecast_df.iloc[0]
    next_date = pd.to_datetime(first_forecast_row["date"])
    next_close = float(first_forecast_row["forecast_close"])

    heights = [last_actual_close, next_close]

    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Last actual", "Next day forecast"],
        heights,
        color=["tab:blue", "tab:orange"],
        alpha=0.8,
    )

    plt.title(f"{ticker} – Next-Day LSTM Forecast", fontsize=14)
    plt.ylabel("Price (USD)", fontsize=12)

    change_pct = ((next_close / last_actual_close) - 1) * 100

    text = (
        f"Last actual ({last_actual_date:%Y-%m-%d}): {last_actual_close:.2f} USD\n"
        f"Forecast ({next_date:%Y-%m-%d}): {next_close:.2f} USD\n"
        f"Change: {change_pct:.2f}%"
    )
    plt.annotate(
        text,
        xy=(0.5, -0.25),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7),
    )

    plt.grid(axis="y", alpha=0.3)

    out_path = os.path.join(plots_dir, "single_day_forecast_lstm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved single-day forecast plot to {out_path}")

def plot_single_day_timeseries(df_history: pd.DataFrame, forecast_df: pd.DataFrame, plots_dir: str, ticker: str):
    """
    Time-series view of single-day forecast:
    last actual close and next-day forecast as two consecutive dates.
    """
    os.makedirs(plots_dir, exist_ok=True)

    last_actual_date = df_history.index[-1]
    last_actual_close = float(df_history["Close"].iloc[-1])

    first_forecast_row = forecast_df.iloc[0]
    next_date = pd.to_datetime(first_forecast_row["date"])
    next_close = float(first_forecast_row["forecast_close"])

    dates = [last_actual_date, next_date]
    values = [last_actual_close, next_close]

    plt.figure(figsize=(10, 5))
    plt.plot(
        dates,
        values,
        marker="o",
        linestyle="-",
        color="tab:orange",
        label="Last actual & forecast",
    )

    plt.axvline(last_actual_date, color="gray", linestyle="--", alpha=0.4, label="Cut-off date")

    plt.title(f"{ticker} – Last Actual vs Next-Day LSTM Forecast (Time-Series View)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)

    out_path = os.path.join(plots_dir, "single_day_timeseries_lstm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved single-day timeseries plot to {out_path}")
