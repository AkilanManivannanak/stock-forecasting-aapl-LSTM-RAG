# Stock Forecasting System (AAPL)
**Baselines → LSTM → FastAPI → Strictly Grounded RAG Copilot (Ollama)**

A production-style **time-series forecasting and evaluation system** for **AAPL** that demonstrates how applied ML systems should be built:

**Start with strong baselines → measure honestly → generate reproducible artifacts → serve via API → expose results through a non-hallucinating RAG assistant.**

---

## Demo
- Video / screenshots: https://drive.google.com/drive/folders/1jweBVVM5sx1_OAHEBk9_Tknf9P59TeqS?usp=share_link

---

## Why this project exists

Many ML demos:
- skip baselines
- overstate deep models
- present metrics without reproducibility
- bolt on RAG without grounding guarantees

This repository is designed to do the opposite.

**Principles**
- Baselines are first-class citizens.
- Metrics are produced by code and written to artifacts.
- Artifacts are reproducible and reviewable.
- The RAG copilot follows a strict non-hallucination contract.

---

## System overview

### Forecasting pipeline
1. Download **AAPL OHLCV** via `yfinance`
2. Chronological split (train / validation / test)
3. Train:
   - naive baseline
   - moving-average baseline(s)
   - LSTM (Keras / TensorFlow)
4. Evaluate and export:
   - `metrics.csv` (leaderboard)
   - plots
   - `data/kb/experiment_report.md`
5. Serve forecasts via **FastAPI**

### RAG copilot (strict grounding)
1. Index only documents under `data/kb/`
2. Persist Chroma vector store → `artifacts/chroma` (`finance_copilot`)
3. Retrieve top-k chunks at query time
4. Answer using retrieved context only
5. If not explicitly present, return **`Not found in docs.`**

---

## Dataset
- **Source:** Yahoo Finance (`yfinance`)
- **Ticker:** AAPL
- **Frequency:** Daily (`1d`)
- **Fields:** OHLCV (Adj Close optional)
- **Range:** `START_DATE` (config) → latest trading day at run time

Typical scale:
- ~250 trading days per year  
- ~2,500 rows for ~10 years of daily data (approximate)

The exact range and row count for your run should be recorded in:
- `data/kb/experiment_report.md`

---

## Model details (LSTM)

Defined in `scripts/train.py`:

- **Input:** sliding window of the last `N` days (config-driven)
- **Target:** next-day close (rolled forward for multi-day forecasts)
- **Architecture:** LSTM encoder → Dense regression head
- **Loss:** MSE
- **Optimizer:** Adam
- **Validation:** chronological holdout

The experiment report is the source of truth for run-specific configuration and metrics:
- `data/kb/experiment_report.md`

---

## Evaluation and results

### Why baselines often win on close prices
Daily close prices are highly persistent and often behave like a **random walk**. For this target, simple baselines can be extremely strong. If an LSTM underperforms, the correct response is better targets/features/validation, not marketing.

### Verified metrics (AAPL)
Only metrics that were explicitly recorded are shown here.

| Model | MAE | MSE | RMSE | MAPE (%) |
|------|----:|----:|-----:|---------:|
| MA(7) baseline | 7.973786 | 107.4526 | 10.365936 | 3.342139 |

Full leaderboard:
- `metrics.csv` and/or `outputs/metrics.csv`
- `data/kb/experiment_report.md`

---

## Artifacts produced
- `models/` — saved LSTM weights
- `metrics.csv` — evaluation leaderboard
- `plots/actual_vs_pred_lstm.png`
- `data/kb/experiment_report.md` — used by the RAG copilot

---

## API

### Endpoints
- `GET /health`
- `POST /forecast`
- `POST /multi_horizon_forecast`
- `POST /ask` (strictly grounded RAG)

### Run the API
After training and building the Chroma index, start the service:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
# Open :
    - http: //127.0.0.1:8000/docs

Example : forecast:

    curl -s -X POST http://127.0.0.1:8000/forecast \
      -H "Content-Type: application/json" \
      -d '{"ticker":"AAPL","days":5}' | python -m json.tool

Example : RAG:

     curl -s -X POST http://127.0.0.1:8000/ask \
       -H "Content-Type: application/json" \
       -d '{"ticker":"AAPL","question":"Which model has the best RMSE?","k":6}' \
       | python -m json.tool

---

## How to run (end-to-end)
# 1. Create venv + install dependencies:

       python -m venv .venv
       source .venv/bin/activate
       python -m pip install -U pip setuptools wheel
       python -m pip install -r requirements.txt

# for markdown table rendering in reports
      python -m pip install -U tabulate

# helps avoid TF/Keras3 compatibility issues on some setups
      python -m pip install -U tf-keras

# 2. Run tests

     pytest -q

# 3. Train models + generate artifacts

     python -m scripts.train

# 4. Build the RAG index (chroma)

     rm -rf artifacts
     mkdir -p artifacts/chroma

    python -m src.rag_copilot.ingest \
    --ticker AAPL \
    --docs_dir data/kb \
    --persist_dir artifacts/chroma \
    --collection_name finance_copilot

# 5. Run Ollama + API

    # terminal 1
    ollama serve
    ollama pull llama3.1

    # terminal 2
    export RAG_PROVIDER=ollama
    export OLLAMA_BASE_URL=http://127.0.0.1:11434
    export OLLAMA_MODEL=llama3.1
    export RAG_PERSIST_DIR=$(pwd)/artifacts/chroma
    export RAG_COLLECTION_NAME=finance_copilot
    export HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
 
    uvicorn app:app --host 127.0.0.1 --port 8000

---

## Strict grounding contract

The RAG assistant follows one rule:
If the answer is not explicitly present in retrieved data/kb/ context, it returns:
Not found in docs.

---

## Roadmap

- Predict returns / log-returns instead of raw close price
- Walk-forward evaluation
- Regime slicing (bull / bear / sideways)
- Stronger baselines (ARIMA, GBM on engineered features)
- CI gates on metric regression
- API latency metrics (p50 / p95)

---
# Non-goals

- This project does not attempt to produce a profitable trading strategy.
- It does not optimize Sharpe ratio, drawdown, or transaction costs.
---


