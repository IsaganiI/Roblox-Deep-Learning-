import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

df = pd.read_csv("data/RBLX.csv", skiprows=2)

df.columns = ["Date","Close","High","Low","Open","Volume"]

df["Close"] = pd.to_numeric(df["Close"])

df["Prev_Close"] = df["Close"].shift(1)

df = df.dropna()

X = df[["Prev_Close"]]
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


preds = model.predict(X_test)
rmse = np.sqrt(mean_absolute_error(y_test, preds))

print("RMSE:", rmse)
