# Stock Forecasting LLM AAPL

## How to run

### 1. Train models and generate metrics
python main.py

### 2. Build Docker image
docker build -t stock-forecast-api .

### 3. Run FastAPI service
docker run -p 8000:8000 stock-forecast-api

search in website: http://localhost:8000/health
http://localhost:8000/docs#/default/forecast_forecast_post

### 4. Call API
Open http://localhost:8000/docs in a browser and use POST /forecast with:
{
  "ticker": "AAPL",
  "days": 5
}

## Model performance (RMSE)

| Model          | RMSE (Close) |
|----------------|--------------|
| Naive (lag 1)  | 3.24         |
| MA_5           | 5.09         |
| LSTM           | 4.94         |
| Random Forest  | 43.04        |


