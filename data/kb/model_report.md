# AAPL Stock Forecasting – Multi-Model Report

## Dataset
- Source: Yahoo Finance (AAPL Close prices, 2015–2024).
- Features: Close, Return, MA_5, MA_10, Vol_10.

## Models
- Naive (t+1 = t).
- Moving average (5-day).
- RandomForest with walk-forward validation.
- LSTM (PyTorch) with 60-day window.

## Metrics
- See outputs_metrics.csv for RMSE and MAE by model.

## Findings
- Fill this section with what you observed from your runs (which model is best, by how much, where LSTM fails, etc.).
