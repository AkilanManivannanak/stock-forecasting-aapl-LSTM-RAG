from datetime import date
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint


from config import STOCK_SYMBOL, START_DATE
from src.data import load_data
from src.models import load_trained_model, make_forecast

app = FastAPI(
    title="Stock Forecast API",
    description="Forecast future stock prices using ML models.",
    version="1.0.0",
)


# ---------- Pydantic models ----------


class ForecastRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    days: conint(ge=1, le=30) = Field(..., example=7)


class ForecastResponse(BaseModel):
    ticker: str
    days: int
    forecasts: List[float]


class HorizonRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    days: conint(ge=1, le=30) = Field(..., example=30)


class HorizonResult(BaseModel):
    horizon: int
    forecast: float


class MultiHorizonResponse(BaseModel):
    ticker: str
    requested_days: int
    best_horizon: int
    horizons: List[HorizonResult]


# ---------- Helper functions ----------

def get_latest_data(ticker: str) -> pd.DataFrame:
    """
    Load historical data for the given ticker from START_DATE up to today.
    Matches src.data.load_data(ticker, start_date, end_date).
    """
    end_date = date.today().strftime("%Y-%m-%d")
    df = load_data(ticker, START_DATE, end_date)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}.")
    return df


def build_latest_features(df: pd.DataFrame):
    """
    Build the latest feature window for the model.

    For your current LSTM, the input shape is (batch, TIME_STEP, 1)
    with Close prices normalized by the same scaler used in training.
    Here we approximate by taking the last TIME_STEP closes and casting
    to float32 so Keras does not see dtype=object.
    """
    from config import TIME_STEP  # reuse your TIME_STEP

    # Take last TIME_STEP Close values
    closes = df["Close"].astype("float32").values
    if len(closes) < TIME_STEP:
        raise HTTPException(
            status_code=400, detail=f"Not enough data to build a {TIME_STEP}-step window."
        )

    window = closes[-TIME_STEP:]                     # shape (TIME_STEP,)
    window = window.reshape(1, TIME_STEP, 1)         # (batch, time_steps, features)
    return window

# ---------- Simple forecast endpoint ----------

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    print("DEBUG /forecast req:", req)

    df = get_latest_data(req.ticker)
    model = load_trained_model(req.ticker)
    X_last = build_latest_features(df)

    forecasts: List[float] = []
    for h in range(1, req.days + 1):
        y_hat = make_forecast(model, X_last, horizon=h)
        forecasts.append(float(y_hat))

    return ForecastResponse(
        ticker=req.ticker,
        days=req.days,
        forecasts=forecasts,
    )


# ---------- Multi-horizon endpoint (1..days) ----------

@app.post("/multi_horizon_forecast", response_model=MultiHorizonResponse)
def multi_horizon_forecast(req: HorizonRequest):
    print("DEBUG /multi_horizon_forecast req:", req)

    df = get_latest_data(req.ticker)
    model = load_trained_model(req.ticker)
    X_last = build_latest_features(df)

    horizons: List[HorizonResult] = []
    for h in range(1, req.days + 1):
        y_hat = make_forecast(model, X_last, horizon=h)
        horizons.append(HorizonResult(horizon=h, forecast=float(y_hat)))

    best_horizon = req.days  # later you can use RMSE metrics

    return MultiHorizonResponse(
        ticker=req.ticker,
        requested_days=req.days,
        best_horizon=best_horizon,
        horizons=horizons,
    )


# ---------- Root endpoint ----------

@app.get("/")
def root():
    return {
        "message": "Stock Forecast API is running.",
        "endpoints": ["/forecast", "/multi_horizon_forecast"],
    }
