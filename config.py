# config.py

from datetime import date

STOCK_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")  # auto-updates each day

TIME_STEP = 60
TEST_SIZE_RATIO = 0.2

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 64

FORECAST_HORIZON_DAYS = 30

OUTPUTS_DIR = "outputs"
PLOTS_DIR = "plots"
