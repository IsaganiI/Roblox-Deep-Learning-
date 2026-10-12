"""Step 2 of the pipeline: train the LSTM and save model artifacts.

Changes from the original EndPriceLSTM.py:
  * Three chronological splits. The best epoch is chosen on VALIDATION data,
    and TEST data is scored once at the end. (Before, the checkpoint was
    picked by test accuracy, which makes the reported test number optimistic.)
  * Each window is labelled with its own last day's next-day direction, so
    training matches what predict.py does.
  * Saves metadata.json (model version, features, data range, metrics) next
    to the model so every prediction can be traced to the model that made it.
"""

import io
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config, storage
from .data import load_prices
from .features import FEATURE_COLS, build_features
from .model import LSTMClassifier, make_windows


def _predict(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(1).cpu().numpy(), logits


def run_train() -> dict:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    feats = build_features(load_prices())
    labeled = feats[feats["Target"].notna()].reset_index(drop=True)
    X_raw = labeled[FEATURE_COLS].values
    y_all = labeled["Target"].astype(int).values

    n = len(labeled)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    # Fit the scaler on training rows only, then apply it everywhere.
    scaler = StandardScaler().fit(X_raw[:n_train])
    X_scaled = scaler.transform(X_raw)

    windows = make_windows(X_scaled, config.SEQ_LEN)
    end_idx = np.arange(config.SEQ_LEN - 1, n)  # row each window ends on
    labels = y_all[end_idx]

    train_m = end_idx < n_train
    val_m = (end_idx >= n_train) & (end_idx < n_train + n_val)
    test_m = end_idx >= n_train + n_val

    X_tr, y_tr = windows[train_m], labels[train_m]
    X_va, y_va = windows[val_m], labels[val_m]
    X_te, y_te = windows[test_m], labels[test_m]
    print(f"[train] windows: train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(FEATURE_COLS), config.HIDDEN_SIZE,
                           config.NUM_LAYERS, config.DROPOUT).to(device)

    counts = np.bincount(y_tr, minlength=2)
    weights = torch.tensor(len(y_tr) / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=config.BATCH_SIZE, shuffle=True)

    best_val_acc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        val_pred, val_logits = _predict(model, X_va, device)
        val_loss = criterion(val_logits, torch.from_numpy(y_va).to(device)).item()
        val_acc = accuracy_score(y_va, val_pred)
        scheduler.step(val_loss)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{config.EPOCHS}  train_loss={train_loss / len(train_dl):.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{'  * best' if improved else ''}")

    # Score the chosen checkpoint on the untouched test set, once.
    model.load_state_dict(best_state)
    test_pred, _ = _predict(model, X_te, device)
    majority = int(np.bincount(y_tr, minlength=2).argmax())
    metrics = {
        "best_epoch": best_epoch,
        "val_accuracy": round(float(best_val_acc), 4),
        "test_accuracy": round(float(accuracy_score(y_te, test_pred)), 4),
        "baseline_always_up_test_accuracy": round(float((y_te == 1).mean()), 4),
        "baseline_train_majority_test_accuracy": round(float((y_te == majority).mean()), 4),
        "test_report": classification_report(y_te, test_pred, labels=[0, 1],
                                             target_names=["Down", "Up"],
                                             output_dict=True, zero_division=0),
    }

    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": "lstm-" + now.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": now.isoformat(timespec="seconds"),
        "ticker": config.TICKER,
        "feature_cols": FEATURE_COLS,
        "seq_len": config.SEQ_LEN,
        "hyperparams": {
            "hidden_size": config.HIDDEN_SIZE, "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT, "batch_size": config.BATCH_SIZE,
            "epochs": config.EPOCHS, "lr": config.LR, "seed": config.SEED,
        },
        "data_range": {
            "first_date": str(labeled["Date"].iloc[0].date()),
            "last_labeled_date": str(labeled["Date"].iloc[-1].date()),
            "test_start_date": str(labeled["Date"].iloc[n_train + n_val].date()),
        },
        "n_windows": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))},
        "metrics": metrics,
    }

    buf = io.BytesIO()
    torch.save(best_state, buf)
    storage.write_bytes(storage.join(config.MODEL_DIR, config.MODEL_FILE), buf.getvalue())
    buf = io.BytesIO()
    joblib.dump(scaler, buf)
    storage.write_bytes(storage.join(config.MODEL_DIR, config.SCALER_FILE), buf.getvalue())
    storage.write_json(metadata, storage.join(config.MODEL_DIR, config.METADATA_FILE))

    print(f"\n[train] {metadata['model_version']}  best epoch {best_epoch}")
    print(f"  val accuracy            {metrics['val_accuracy']:.4f}")
    print(f"  TEST accuracy           {metrics['test_accuracy']:.4f}")
    print(f"  baseline 'always up'    {metrics['baseline_always_up_test_accuracy']:.4f}")
    print(f"  baseline train majority {metrics['baseline_train_majority_test_accuracy']:.4f}")
    print(f"  artifacts -> {config.MODEL_DIR}")
    return metadata
