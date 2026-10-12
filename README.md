# STRIAX (Roblox-Deep-Learning)

Predicts whether RBLX will close **up or down on the next trading day** with a PyTorch LSTM,
and keeps a scored log of every prediction it makes.

## Pipeline

```
fetch  ->  data/RBLX_prices.csv            (yfinance, clean single-header CSV)
train  ->  models/lstm_model.pt            (LSTM weights)
           models/scaler.pkl               (feature scaler)
           models/metadata.json            (model version, features, data range, metrics)
predict -> out/predictions.csv             (every prediction, scored once the result is known)
           out/latest.json                 (newest prediction + live hit rate)
```

All code lives in the `striax/` package:

| File | Job |
|---|---|
| `config.py` | Paths, ticker, hyperparameters (paths overridable by env var) |
| `storage.py` | Reads/writes local paths or `s3://` URIs |
| `data.py` | Download and clean prices |
| `features.py` | Feature engineering, shared by training and prediction |
| `model.py` | `LSTMClassifier` and sequence windowing |
| `train.py` | Train/val/test split, training loop, artifacts + metrics |
| `predict.py` | Next-day prediction from saved artifacts |
| `predictions.py` | Predictions log schema, scoring, hit rate |

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m striax all       # fetch + train + predict
python -m striax daily     # fetch + predict (trains only if no model exists)
python -m striax fetch | train | predict
```

## Run in Docker

```bash
docker build -t striax .
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/out:/app/out" striax          # all
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/out:/app/out" striax daily
```

(PowerShell: use `${PWD}` instead of `$PWD`.)

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `STRIAX_DATA_DIR` | `data` | local folder or `s3://bucket/prefix` |
| `STRIAX_MODEL_DIR` | `models` | local folder or `s3://bucket/prefix` |
| `STRIAX_OUTPUT_DIR` | `out` | local folder or `s3://bucket/prefix` |
| `STRIAX_TICKER` | `RBLX` | |
| `STRIAX_PERIOD` | `5y` | yfinance history length |
| `STRIAX_EPOCHS` | `50` | lower it for quick test runs |

Pointing the three directories at S3 is all that's needed to run on AWS; no code changes.

## predictions.csv columns

`as_of_date, ticker, last_close, predicted_direction, prob_up, model_version, created_at,
target_date, actual_close, actual_direction, correct`. The last four are filled in automatically
on a later run once the next trading day's close exists. See `striax/predictions.py`.

## Evaluation notes

- Data is split chronologically 70/15/15. The best epoch is picked on **validation**; the
  **test** set is scored once at the end.
- `metadata.json` reports test accuracy next to two baselines (always predicting "up", and
  predicting the training set's majority class). The LSTM is only useful if it beats them.

## Other files

- `index.html`, `style.css`, `script.js`, `data.json`: frontend dashboard.
- `src/fetch_data.py` → `src/trainLRM.py` → `export_data.py`: linear-regression model that
  currently feeds `data.json` for the frontend.
- `notebooks/`: exploration and visualizations.
