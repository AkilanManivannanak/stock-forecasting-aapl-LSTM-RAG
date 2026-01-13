from datetime import date
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Stock Forecasting API")


@app.get("/health")
def health():
    return {"status": "ok"}


class ForecastRequest(BaseModel):
    ticker: str = "AAPL"
    days: int = 30


class ForecastPoint(BaseModel):
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    model_version: str
    trained_on_start: Optional[date] = None
    trained_on_end: Optional[date] = None
    data_cutoff: Optional[date] = None
    points: List[ForecastPoint]


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.days <= 0 or req.days > 60:
        raise HTTPException(status_code=400, detail="days must be between 1 and 60")

    try:
        df = pd.read_csv("future_forecast_lstm.csv", parse_dates=["date"]).sort_values("date")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="future_forecast_lstm.csv not found. Run training first.",
        )

    df = df.head(req.days)

    points = [
        ForecastPoint(date=row["date"].date(), forecast=float(row["forecast_close"]))
        for _, row in df.iterrows()
    ]

    data_cutoff = df["date"].min().date() if not df.empty else None

    return ForecastResponse(
        model_version="lstm_from_future_csv_v1",
        trained_on_start=None,
        trained_on_end=None,
        data_cutoff=data_cutoff,
        points=points,
    )
