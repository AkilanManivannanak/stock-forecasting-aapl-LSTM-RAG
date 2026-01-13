# src/evaluation.py

import os
import csv


def save_metrics(metrics_dict, outputs_dir: str):
    """
    metrics_dict: {model_name: {"rmse": float}}
    """
    os.makedirs(outputs_dir, exist_ok=True)
    path = os.path.join(outputs_dir, "metrics.csv")

    fieldnames = ["model", "rmse"]

    # sort by rmse ascending
    items = sorted(metrics_dict.items(), key=lambda kv: kv[1]["rmse"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, vals in items:
            writer.writerow({"model": model_name, "rmse": vals["rmse"]})

    print("\nModel leaderboard (sorted by RMSE):")
    for model_name, vals in items:
        print(f"  {model_name:15s} RMSE = {vals['rmse']:.4f}")
    print(f"\nSaved metrics to {path}")
