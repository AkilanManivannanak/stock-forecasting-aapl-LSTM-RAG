# src/reports.py

import os
import pandas as pd


def write_experiment_report(
    metrics_path: str,
    forecast_path: str,
    plots_dir: str,
    out_path: str,
    walkforward_path: str | None = None,
):
    # read metrics and sort by RMSE (lower is better)
    metrics = pd.read_csv(metrics_path)
    metrics_sorted = metrics.sort_values("rmse")

    # best row is first after sorting
    best_row = metrics_sorted.iloc[0]
    best_model = best_row["model"]
    best_rmse = best_row["rmse"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# AAPL Forecasting Experiment Report\n\n")

        # 1. Global model performance
        f.write("## Model performance (full test split)\n\n")
        f.write(metrics_sorted.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"Best model on RMSE: **{best_model}** with RMSE ≈ {best_rmse:.3f}.\n\n")

        f.write("### Interpretation\n\n")
        f.write(
            "- Naive baseline currently has the lowest RMSE, which shows why strong baselines are important "
            "when evaluating complex models like LSTM and RandomForest.\n"
        )
        f.write(
            "- The LSTM tracks the overall trend visually but is still numerically worse than Naive; "
            "this suggests more feature engineering and hyperparameter tuning are needed.\n\n"
        )

        # 2. Walk-forward / regime analysis
        if walkforward_path is not None and os.path.exists(walkforward_path):
            f.write("## Walk-forward and regime analysis\n\n")
            wf = pd.read_csv(walkforward_path, parse_dates=["start_date", "end_date"])

            f.write("### Per-window metrics (Naive & RandomForest)\n\n")
            f.write(wf.to_markdown(index=False))
            f.write("\n\n")

            # aggregate by regime and model
            regime_summary = (
                wf.groupby(["regime", "model"])["rmse"]
                .mean()
                .reset_index()
                .sort_values(["regime", "rmse"])
            )

            f.write("### Average RMSE by regime\n\n")
            f.write(regime_summary.to_markdown(index=False))
            f.write("\n\n")

            f.write(
                "Interpretation:\n\n"
                "- Each row in the table above shows how models behave in different market regimes "
                "(bull, bear, sideways), defined by the cumulative return over each window.\n"
            )
            f.write(
                "- This helps answer questions like: does the feature-rich RandomForest offer more value "
                "in trending markets, or does the Naive model remain hard to beat even in bull runs?\n\n"
            )

        # 3. Plots
        f.write("## Plots\n\n")
        f.write(
            "- `plots/actual_vs_pred_lstm.png` – Actual vs LSTM predicted AAPL close prices "
            "on the held‑out test set.\n\n"
        )

        # 4. Future forecast
        f.write("## Future forecast\n\n")
        try:
            forecast_df = pd.read_csv(forecast_path)
            horizon = len(forecast_df)
        except Exception:
            forecast_df = None
            horizon = "N/A"

        f.write(
            f"- Future LSTM forecasts (next {horizon} business days) are saved in "
            f"`{forecast_path}` and can be plotted to visualize the expected trend.\n"
        )
