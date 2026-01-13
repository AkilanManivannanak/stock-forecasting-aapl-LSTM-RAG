import pandas as pd
from src.main_training import compute_regime  # already defined in main_training.py[conversation_history:1]


def test_compute_regime_handles_short_series():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    regime = compute_regime(df["Close"])
    assert regime in {"unknown", "bull", "bear", "sideways"}
