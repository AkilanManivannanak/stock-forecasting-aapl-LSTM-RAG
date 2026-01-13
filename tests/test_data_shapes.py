import pandas as pd
from src.data import load_data, prepare_data, build_rf_feature_matrix

def test_load_data():
    df = load_data("AAPL", "2020-01-01", "2026-01-01")
    assert not df.empty

    # Handle both simple columns and MultiIndex columns.[web:158]
    cols = df.columns
    if hasattr(cols, "levels") and len(cols.levels) == 2:
        # MultiIndex like ('Open', 'AAPL') -> take first level
        first_level = set(cols.get_level_values(0))
        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(first_level)
    else:
        assert {"Open", "High", "Low", "Close", "Volume"}.issubset(set(cols))

def test_prepare_data_shapes():
    df = load_data("AAPL", "2020-01-01", "2020-12-31")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, time_step=60, test_ratio=0.2)
    assert X_train.shape[1:] == (60, 1)
    assert X_test.shape[1:] == (60, 1)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_rf_feature_matrix_shapes():
    df = load_data("AAPL", "2020-01-01", "2020-12-31")
    X_rf, y_rf = build_rf_feature_matrix(df, n_lags=5)
    assert X_rf.shape[0] == y_rf.shape[0]
    assert X_rf.shape[1] > 5

