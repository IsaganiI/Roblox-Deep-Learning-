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

## Run on AWS (ECS Fargate)

```
ECR (image) ──▶ ECS Fargate task ──reads/writes──▶ S3: s3://striax-<account>/models, /out
                       │
                       └──logs──▶ CloudWatch: /ecs/striax
```

One-time setup (Windows PowerShell, from the repo root; needs AWS CLI configured):

```powershell
$ACCOUNT  = aws sts get-caller-identity --query Account --output text
$REGISTRY = "$ACCOUNT.dkr.ecr.us-east-1.amazonaws.com"

aws ecr create-repository --repository-name striax --region us-east-1
aws s3 mb "s3://striax-$ACCOUNT" --region us-east-1

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REGISTRY
docker build -t striax .
docker tag striax:latest "$REGISTRY/striax:latest"
docker push "$REGISTRY/striax:latest"

powershell -ExecutionPolicy Bypass -File .\deploy\setup.ps1
```

Run it:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\run.ps1 -Command all   # first run: full retrain
powershell -ExecutionPolicy Bypass -File .\deploy\run.ps1                # daily: fetch + predict
```

Run it automatically (EventBridge Scheduler, New York time):

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1            # weekdays 17:15 daily, Saturdays 10:00 retrain
powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1 -TestNow   # one-time run in 2 minutes
powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1 -Pause     # turn schedules off
powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1 -Remove    # delete schedules
```

`deploy/` contents:

| File | Purpose |
|---|---|
| `setup.ps1` | Creates/updates log group, IAM roles, ECS cluster, task definition (safe to re-run) |
| `run.ps1` | Starts one Fargate task, waits, prints logs and the result |
| `taskdef.json` | Task definition template (1 vCPU, 2 GB, S3 settings) |
| `task-role-policy.json` | Least-privilege S3 access: only the STRIAX bucket |
| `ecs-tasks-trust.json` | Lets ECS tasks assume the two roles |
| `schedule.ps1` | Creates/updates the daily and weekly-retrain schedules |
| `scheduler-role-policy.json` | Scheduler may only start the STRIAX task |
| `scheduler-trust.json` | Lets EventBridge Scheduler (this account only) assume its role |

The task uses the default VPC with a public IP, so no NAT gateway (and its monthly cost) is needed.

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
