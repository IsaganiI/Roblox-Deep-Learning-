"""Step 3 of the pipeline: predict tomorrow's direction and update the predictions log.

Reads the saved model (no retraining), predicts from the latest SEQ_LEN days,
fills in outcomes for older predictions that can now be scored, and writes:
  out/predictions.csv   the full history (see predictions.py for columns)
  out/latest.json       the newest prediction plus the running hit rate
"""

import io
from datetime import datetime, timezone

import joblib
import torch

from . import config, predictions, storage
from .data import load_prices
from .features import FEATURE_COLS, build_features
from .model import LSTMClassifier


def load_artifacts():
    meta = storage.read_json(storage.join(config.MODEL_DIR, config.METADATA_FILE))
    if meta["feature_cols"] != FEATURE_COLS:
        raise RuntimeError("Saved model was trained on different features; retrain it with "
                           "`python -m striax train`.")
    scaler = joblib.load(io.BytesIO(storage.read_bytes(
        storage.join(config.MODEL_DIR, config.SCALER_FILE))))
    hp = meta["hyperparams"]
    model = LSTMClassifier(len(FEATURE_COLS), hp["hidden_size"], hp["num_layers"], hp["dropout"])
    state = torch.load(io.BytesIO(storage.read_bytes(
        storage.join(config.MODEL_DIR, config.MODEL_FILE))), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, scaler, meta


def run_predict() -> dict:
    prices = load_prices()
    feats = build_features(prices)
    model, scaler, meta = load_artifacts()

    seq_len = meta["seq_len"]
    X = scaler.transform(feats[FEATURE_COLS].values[-seq_len:])
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X, dtype=torch.float32).unsqueeze(0)), dim=1)[0]
    prob_up = float(probs[1])

    row = {
        "as_of_date": str(feats["Date"].iloc[-1].date()),
        "ticker": config.TICKER,
        "last_close": round(float(feats["Close"].iloc[-1]), 4),
        "predicted_direction": "UP" if prob_up >= 0.5 else "DOWN",
        "prob_up": round(prob_up, 4),
        "model_version": meta["model_version"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    log_uri = storage.join(config.OUTPUT_DIR, config.PREDICTIONS_FILE)
    log = (storage.read_csv(log_uri, dtype={"as_of_date": str, "target_date": str})
           if storage.exists(log_uri) else predictions.empty_log())
    log = predictions.score(predictions.upsert(log, row), prices)
    storage.write_csv(log, log_uri)

    latest = {
        "prediction": row,
        "track_record": predictions.summary(log),
        "model": {"version": meta["model_version"], "trained_at": meta["trained_at"],
                  "test_accuracy": meta["metrics"]["test_accuracy"],
                  "baseline_always_up_test_accuracy":
                      meta["metrics"]["baseline_always_up_test_accuracy"]},
    }
    storage.write_json(latest, storage.join(config.OUTPUT_DIR, config.LATEST_FILE))

    print(f"[predict] {config.TICKER} as of {row['as_of_date']} (close ${row['last_close']:.2f}): "
          f"next day {row['predicted_direction']}  P(up)={prob_up:.1%}  [{row['model_version']}]")
    tr = latest["track_record"]
    if tr["n_scored"]:
        print(f"[predict] live hit rate {tr['hit_rate']:.1%} over {tr['n_scored']} scored predictions")
    print(f"[predict] -> {log_uri}")
    return latest
