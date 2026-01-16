# Project Goal

Build a production-style stock forecasting system for AAPL that demonstrates an end-to-end ML workflow:
1) Data ingestion (Yahoo Finance OHLCV)
2) Feature engineering (technical indicators + lag features)
3) Model training & evaluation (Naive baseline, Moving Average, RandomForest, LSTM)
4) Walk-forward regime evaluation (bull/bear/sideways windows)
5) Model serving via FastAPI (forecast + multi-horizon endpoints)
6) RAG Copilot that answers questions about results and code using a local vector store with citations.
