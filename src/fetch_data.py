import yfinance as yf
import pandas as pd

ticker = "RBLX"

data = yf.download(ticker, period="5y")

data.to_csv("data/RBLX.csv")

print("Roblox stock data saved.")