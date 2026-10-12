"""Feature engineering shared by training and prediction.

Training and prediction must build features the exact same way, so this is
the only place they are defined. The features are the same ones the original
EndPriceLSTM.py used.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Return", "HL_range", "OC_range", "Vol_change",
    "Return_lag1", "Return_lag2", "Return_lag3", "Return_lag5",
    "MA5", "Vol_MA5", "MA10", "Vol_MA10", "MA20", "Vol_MA20",
]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with Date, Close, FEATURE_COLS and Target.

    Target = 1 if the NEXT day's close is higher than this day's close.
    The most recent row has no next day yet, so its Target is NaN: it is
    excluded from training but is exactly the row we predict from.
    """
    df = prices.copy()

    df["Return"] = df["Close"].pct_change()
    df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["OC_range"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Vol_change"] = df["Volume"].pct_change()

    for lag in [1, 2, 3, 5]:
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)

    for w in [5, 10, 20]:
        df[f"MA{w}"] = df["Close"].rolling(w).mean() / df["Close"] - 1
        df[f"Vol_MA{w}"] = df["Volume"].rolling(w).mean() / df["Volume"] - 1

    next_close = df["Close"].shift(-1)
    df["Target"] = np.where(next_close.isna(), np.nan, (next_close > df["Close"]).astype(float))

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df[["Date", "Close"] + FEATURE_COLS + ["Target"]]
