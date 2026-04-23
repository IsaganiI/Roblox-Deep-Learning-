import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from train import (
    load_and_engineer, LSTMClassifier,
    DATA_PATH, MODEL_PATH, SCALER_PATH,
    SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
)

def predict_latest():
    df = load_and_engineer(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("Date", "Target", "Close",
                                                         "High", "Low", "Open", "Volume")]

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(df[feature_cols].values)

    # Use the most recent SEQ_LEN rows as input
    seq = torch.tensor(X_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)  # (1, seq, features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMClassifier(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(seq.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(probs.argmax())

    last_date  = df["Date"].iloc[-1].date()
    last_close = df["Close"].iloc[-1]

    print(f"\n RBLX Prediction")
    print(f"Based on data through : {last_date}")
    print(f"Last close            : ${last_close:.2f}")
    print(f"Prediction (next day) : {'UP' if pred == 1 else 'DOWN'}")
    print(f"Confidence            : {probs[pred]*100:.1f}%  (down={probs[0]*100:.1f}%  up={probs[1]*100:.1f}%)")


if __name__ == "__main__":
    predict_latest()
