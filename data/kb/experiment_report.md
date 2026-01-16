# AAPL Forecasting Experiment Report

## Model performance (full test split)

| model         |     rmse |
|:--------------|---------:|
| naive         |  3.54425 |
| lstm          |  7.7914  |
| random_forest | 39.8252  |
| ma_5          | 50.9045  |

Best model on RMSE: **naive** with RMSE ≈ 3.544.

### Interpretation

- Naive baseline currently has the lowest RMSE, which shows why strong baselines are important when evaluating complex models like LSTM and RandomForest.
- The LSTM tracks the overall trend visually but is still numerically worse than Naive; this suggests more feature engineering and hyperparameter tuning are needed.

## Walk-forward and regime analysis

### Per-window metrics (Naive & RandomForest)

|   window_id | start_date          | end_date            | regime   | model         |      rmse |
|------------:|:--------------------|:--------------------|:---------|:--------------|----------:|
|           0 | 2015-01-02 00:00:00 | 2015-12-31 00:00:00 | sideways | naive         |  0.484548 |
|           0 | 2015-01-02 00:00:00 | 2015-12-31 00:00:00 | sideways | random_forest |  2.3769   |
|           1 | 2015-07-06 00:00:00 | 2016-07-01 00:00:00 | bear     | naive         |  0.378485 |
|           1 | 2015-07-06 00:00:00 | 2016-07-01 00:00:00 | bear     | random_forest |  2.46395  |
|           2 | 2016-01-04 00:00:00 | 2016-12-30 00:00:00 | bull     | naive         |  0.296849 |
|           2 | 2016-01-04 00:00:00 | 2016-12-30 00:00:00 | bull     | random_forest |  1.35657  |
|           3 | 2016-07-05 00:00:00 | 2017-07-03 00:00:00 | bull     | naive         |  0.352383 |
|           3 | 2016-07-05 00:00:00 | 2017-07-03 00:00:00 | bull     | random_forest |  6.22551  |
|           4 | 2017-01-03 00:00:00 | 2018-01-02 00:00:00 | bull     | naive         |  0.433686 |
|           4 | 2017-01-03 00:00:00 | 2018-01-02 00:00:00 | bull     | random_forest |  3.35154  |
|           5 | 2017-07-05 00:00:00 | 2018-07-03 00:00:00 | bull     | naive         |  0.606743 |
|           5 | 2017-07-05 00:00:00 | 2018-07-03 00:00:00 | bull     | random_forest |  2.48883  |
|           6 | 2018-01-03 00:00:00 | 2019-01-03 00:00:00 | bear     | naive         |  1.00663  |
|           6 | 2018-01-03 00:00:00 | 2019-01-03 00:00:00 | bear     | random_forest |  5.22148  |
|           7 | 2018-07-05 00:00:00 | 2019-07-05 00:00:00 | bull     | naive         |  0.695446 |
|           7 | 2018-07-05 00:00:00 | 2019-07-05 00:00:00 | bull     | random_forest |  1.34375  |
|           8 | 2019-01-04 00:00:00 | 2020-01-03 00:00:00 | bull     | naive         |  0.797528 |
|           8 | 2019-01-04 00:00:00 | 2020-01-03 00:00:00 | bull     | random_forest |  9.88688  |
|           9 | 2019-07-08 00:00:00 | 2020-07-06 00:00:00 | bull     | naive         |  2.22361  |
|           9 | 2019-07-08 00:00:00 | 2020-07-06 00:00:00 | bull     | random_forest |  7.16559  |
|          10 | 2020-01-06 00:00:00 | 2021-01-04 00:00:00 | bull     | naive         |  2.8188   |
|          10 | 2020-01-06 00:00:00 | 2021-01-04 00:00:00 | bull     | random_forest | 28.0809   |
|          11 | 2020-07-07 00:00:00 | 2021-07-06 00:00:00 | bull     | naive         |  2.15167  |
|          11 | 2020-07-07 00:00:00 | 2021-07-06 00:00:00 | bull     | random_forest |  3.55364  |
|          12 | 2021-01-05 00:00:00 | 2022-01-03 00:00:00 | bull     | naive         |  2.20419  |
|          12 | 2021-01-05 00:00:00 | 2022-01-03 00:00:00 | bull     | random_forest | 18.0071   |
|          13 | 2021-07-07 00:00:00 | 2022-07-06 00:00:00 | sideways | naive         |  3.43076  |
|          13 | 2021-07-07 00:00:00 | 2022-07-06 00:00:00 | sideways | random_forest |  6.19773  |
|          14 | 2022-01-04 00:00:00 | 2023-01-04 00:00:00 | bear     | naive         |  3.26921  |
|          14 | 2022-01-04 00:00:00 | 2023-01-04 00:00:00 | bear     | random_forest |  4.74817  |
|          15 | 2022-07-07 00:00:00 | 2023-07-07 00:00:00 | bull     | naive         |  2.06158  |
|          15 | 2022-07-07 00:00:00 | 2023-07-07 00:00:00 | bull     | random_forest | 12.0152   |
|          16 | 2023-01-05 00:00:00 | 2024-01-05 00:00:00 | bull     | naive         |  2.22968  |
|          16 | 2023-01-05 00:00:00 | 2024-01-05 00:00:00 | bull     | random_forest |  4.61835  |
|          17 | 2023-07-10 00:00:00 | 2024-07-09 00:00:00 | bull     | naive         |  2.84688  |
|          17 | 2023-07-10 00:00:00 | 2024-07-09 00:00:00 | bull     | random_forest |  8.47475  |
|          18 | 2024-01-08 00:00:00 | 2025-01-07 00:00:00 | bull     | naive         |  2.93578  |
|          18 | 2024-01-08 00:00:00 | 2025-01-07 00:00:00 | bull     | random_forest | 12.4854   |
|          19 | 2024-07-10 00:00:00 | 2025-07-11 00:00:00 | bear     | naive         |  5.1239   |
|          19 | 2024-07-10 00:00:00 | 2025-07-11 00:00:00 | bear     | random_forest | 12.4757   |

### Average RMSE by regime

| regime   | model         |    rmse |
|:---------|:--------------|--------:|
| bear     | naive         | 2.44456 |
| bear     | random_forest | 6.22732 |
| bull     | naive         | 1.6182  |
| bull     | random_forest | 8.50386 |
| sideways | naive         | 1.95765 |
| sideways | random_forest | 4.28732 |

Interpretation:

- Each row in the table above shows how models behave in different market regimes (bull, bear, sideways), defined by the cumulative return over each window.
- This helps answer questions like: does the feature-rich RandomForest offer more value in trending markets, or does the Naive model remain hard to beat even in bull runs?

## Plots

- `plots/actual_vs_pred_lstm.png` – Actual vs LSTM predicted AAPL close prices on the held‑out test set.

## Future forecast

- Future LSTM forecasts (next 30 business days) are saved in `outputs/future_forecast_lstm.csv` and can be plotted to visualize the expected trend.
