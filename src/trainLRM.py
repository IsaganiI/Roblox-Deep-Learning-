import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

df = pd.read_csv("data/RBLX.csv", skiprows=2)

df.columns = ["Date","Close","High","Low","Open","Volume"]

df["Close"] = pd.to_numeric(df["Close"])

df["Lag1"] = df["Close"].shift(1)
df["Lag2"] = df["Close"].shift(2)
df["Lag3"] = df["Close"].shift(3)
df["Lag5"] = df["Close"].shift(5)
df["MA5"] = df["Close"].rolling(5).mean()
df["MA20"] = df["Close"].rolling(20).mean()

df = df.dropna()

X = df[
    ["Lag1",
    "Lag2",
    "Lag3",
    "Lag5",
    "MA5",
    "MA20"]
    ]

y = df["Close"]


split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


model = LinearRegression()
model.fit(X_train, y_train)




joblib.dump(model, "models/model.pkl")
print("Model trained and saved")

from sklearn.metrics import mean_squared_error

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE:", rmse)
