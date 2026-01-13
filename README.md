## Architecture

![Architecture](docs/architecture.png)

Data (Yahoo Finance) → features (technical indicators + lags) → train (LSTM + RF + baselines) →
registry (models/ + metrics CSVs) → Docker + FastAPI serve (`/forecast`) → monitor via logs/CSV metrics.

## How to run

### Train and generate metrics
python main.py

### Build Docker image
docker build -t stock-forecast-api .

### Run API
docker run -p 8000:8000 stock-forecast-api

Then open http://localhost:8000/docs and call POST /forecast.


## Model performance (RMSE)

| Model          | RMSE (Close) |
|----------------|--------------|
| Naive (lag 1)  | see metrics.csv |
| MA_5           | see metrics.csv |
| LSTM           | see metrics.csv |
| Random Forest  | see metrics.csv |

