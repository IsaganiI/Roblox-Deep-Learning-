import pandas as pd
import joblib

DATA_PATH = "data/RBLX.csv"
MODEL_PATH = "models/model.pk1"

def main():
    df = pd.read_csv(DATA_PATH)
    needed = ["Open", "High", "Low", "Volume", "Close"]
    