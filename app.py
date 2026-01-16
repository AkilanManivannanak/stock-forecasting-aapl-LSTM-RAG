# app.py

import os
import inspect
from datetime import date
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

from config import START_DATE, TIME_STEP
from src.models import load_trained_model, make_forecast
from src.rag_copilot.qa import answer_question

app = FastAPI(
    title="Stock Forecast API",
    description="Forecast future stock prices using ML models + provide RAG Q&A with citations.",
    version="1.2.1",
)

# ---------------------------
# Forecasting models (API)
# ---------------------------

class ForecastRequest(BaseModel):
    ticker: str = Field(default="AAPL", json_schema_extra={"example": "AAPL"})
    days: conint(ge=1, le=30) = Field(default=7, json_schema_extra={"example": 7})


class ForecastPoint(BaseModel):
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    ticker: str
    days: int
    points: List[ForecastPoint]


class HorizonRequest(BaseModel):
    ticker: str = Field(default="AAPL", json_schema_extra={"example": "AAPL"})
    days: conint(ge=1, le=30) = Field(default=30, json_schema_extra={"example": 30})


class HorizonResult(BaseModel):
    horizon: int
    date: date
    forecast: float


class MultiHorizonResponse(BaseModel):
    ticker: str
    requested_days: int
    best_horizon: int
    horizons: List[HorizonResult]


def _download_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Deterministic download: auto_adjust explicitly set.
    Returns: Open, High, Low, Close, Volume. Sorted index, no NaNs.
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}.")

    # Normalize MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns in yfinance data: {missing}")

    return df[needed].dropna().sort_index()


def get_latest_data(ticker: str) -> pd.DataFrame:
    end_date = date.today().strftime("%Y-%m-%d")
    return _download_ohlcv(ticker, START_DATE, end_date)


def _scaler_path(ticker: str) -> Path:
    return Path("models") / f"{ticker.upper()}_scaler.pkl"


def load_scaler(ticker: str):
    p = _scaler_path(ticker)
    if not p.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Scaler not found at {p}. Run training (python -m scripts.train) to generate it.",
        )
    return joblib.load(p)


def build_latest_window_scaled(df: pd.DataFrame, scaler):
    """
    Build latest scaled window: shape (1, TIME_STEP, 1)
    """
    closes = df["Close"].astype("float32").values.reshape(-1, 1)
    if len(closes) < TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to build TIME_STEP={TIME_STEP} window.",
        )

    scaled = scaler.transform(closes)
    return scaled[-TIME_STEP:].reshape(1, TIME_STEP, 1)


def business_day_dates_after(last_timestamp: pd.Timestamp, n: int) -> List[date]:
    start = last_timestamp + pd.Timedelta(days=1)
    return list(pd.bdate_range(start=start, periods=n).date)


# ---------------------------
# RAG models (API)
# ---------------------------

class AskRequest(BaseModel):
    ticker: str = Field(default="AAPL", json_schema_extra={"example": "AAPL"})
    question: str = Field(..., json_schema_extra={"example": "What is the goal of this project?"})
    k: conint(ge=1, le=12) = Field(default=6, json_schema_extra={"example": 6})


class AskResponse(BaseModel):
    answer: str
    citations: List[dict]
    retrieval_debug: dict


def _call_answer_question_safely(**kwargs):
    """
    Calls answer_question with ONLY the kwargs it actually supports.
    This prevents 'unexpected keyword argument' crashes when qa.py changes.
    """
    sig = inspect.signature(answer_question)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return answer_question(**filtered)


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    df = get_latest_data(req.ticker)
    scaler = load_scaler(req.ticker)

    # Model load can fail if saved/loaded with different Keras stacks.
    # Fix is in src/models.py (Keras3 load_model compile=False).
    model = load_trained_model(req.ticker)

    X_last = build_latest_window_scaled(df, scaler)

    forecasts_usd: List[float] = []
    for h in range(1, req.days + 1):
        y_hat_scaled = make_forecast(model, X_last, horizon=h)
        y_hat_usd = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])
        forecasts_usd.append(y_hat_usd)

    dates = business_day_dates_after(df.index[-1], req.days)
    points = [ForecastPoint(date=d, forecast=f) for d, f in zip(dates, forecasts_usd)]
    return ForecastResponse(ticker=req.ticker.upper(), days=req.days, points=points)


@app.post("/multi_horizon_forecast", response_model=MultiHorizonResponse)
def multi_horizon_forecast(req: HorizonRequest):
    df = get_latest_data(req.ticker)
    scaler = load_scaler(req.ticker)
    model = load_trained_model(req.ticker)

    X_last = build_latest_window_scaled(df, scaler)
    dates = business_day_dates_after(df.index[-1], req.days)

    horizons: List[HorizonResult] = []
    for h in range(1, req.days + 1):
        y_hat_scaled = make_forecast(model, X_last, horizon=h)
        y_hat_usd = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])
        horizons.append(HorizonResult(horizon=h, date=dates[h - 1], forecast=y_hat_usd))

    best_horizon = req.days  # placeholder
    return MultiHorizonResponse(
        ticker=req.ticker.upper(),
        requested_days=req.days,
        best_horizon=best_horizon,
        horizons=horizons,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    RAG endpoint: local embeddings + Chroma; answer may be generated by Ollama/OpenAI depending on qa.py.
    """
    persist_dir = os.getenv("RAG_PERSIST_DIR", "artifacts/chroma")

    # Provide *possible* knobs (only used if qa.answer_question supports them)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    hf_embed_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        out = _call_answer_question_safely(
            question=req.question,
            ticker=req.ticker,
            k=int(req.k),
            persist_dir=persist_dir,
            chat_model=chat_model,
            embed_model=embed_model,
            hf_embed_model=hf_embed_model,
        )
        return AskResponse(**out)

    except Exception as e:
        msg = str(e)
        # If Chroma gets corrupted / mismatch IDs, guide the user to rebuild the store.
        if "Error finding id" in msg or "chromadb.errors.InternalError" in msg:
            raise HTTPException(
                status_code=500,
                detail=(
                    "RAG index seems corrupted or out-of-sync. Fix: STOP uvicorn, then run:\n"
                    "rm -rf artifacts/chroma\n"
                    "python -m src.rag_copilot.ingest --ticker AAPL --docs_dir data/kb --persist_dir artifacts/chroma\n"
                    "Then restart uvicorn."
                ),
            )
        raise HTTPException(status_code=500, detail=f"RAG /ask failed: {msg}")


@app.get("/")
def root():
    return {
        "message": "Stock Forecast API is running.",
        "endpoints": ["/health", "/forecast", "/multi_horizon_forecast", "/ask"],
    }
