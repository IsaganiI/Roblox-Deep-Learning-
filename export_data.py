import pandas as pd
import json
import joblib
import numpy as np

df = pd.read_csv("data/RBLX.csv", skiprows=2)
df.columns = ["Date","Close","High","Low","Open","Volume"]
df["Close"] = pd.to_numeric(df["Close"])
df["Volume"] = pd.to_numeric(df["Volume"])
df = df.dropna()

df["MA7"]  = df["Close"].rolling(7).mean()
df["MA30"] = df["Close"].rolling(30).mean()
df["Lag1"] = df["Close"].shift(1)
df["Lag2"] = df["Close"].shift(2)
df["Lag3"] = df["Close"].shift(3)
df["Lag5"] = df["Close"].shift(5)
df["MA5"]  = df["Close"].rolling(5).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df = df.dropna()

model = joblib.load("models/model.pkl")
last_row = df.iloc[-1]
features = pd.DataFrame([{
    "Lag1": last_row["Lag1"],
    "Lag2": last_row["Lag2"],
    "Lag3": last_row["Lag3"],
    "Lag5": last_row["Lag5"],
    "MA5":  last_row["MA5"],
    "MA20": last_row["MA20"]
}])

predicted  = float(model.predict(features)[0])
last_price = float(df["Close"].iloc[-1])
change     = last_price - float(df["Close"].iloc[-2])
signal     = "Bullish" if predicted > last_price else "Bearish" if predicted < last_price else "Neutral"

recent = df.tail(90)
output = {
    "labels":  recent["Date"].tolist(),
    "prices":  [round(x,2) for x in recent["Close"].tolist()],
    "ma7":     [round(x,2) if not np.isnan(x) else None for x in recent["MA7"].tolist()],
    "ma30":    [round(x,2) if not np.isnan(x) else None for x in recent["MA30"].tolist()],
    "volumes": [int(x) for x in recent["Volume"].tolist()],
    "latest": {
        "price":     round(last_price, 2),
        "change":    round(change, 2),
        "changePct": round(change / float(df["Close"].iloc[-2]) * 100, 2),
        "date":      df["Date"].iloc[-1],
        "volume":    int(df["Volume"].iloc[-1])
    },
    "signal": {
        "label":      signal,
        "predicted":  round(predicted, 2),
        "trend":      signal + " — model predicts next close at $" + str(round(predicted,2)) + ".",
        "volatility": "Medium",
        "maNote":     "Based on 7-day and 30-day moving averages.",
        "signalNote": signal + " — linear regression output based on previous close."
    }
}

with open("data.json", "w") as f:
    json.dump(output, f)
print("Done! data.json created.")