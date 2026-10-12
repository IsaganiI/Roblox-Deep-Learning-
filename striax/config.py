"""Central configuration for the STRIAX pipeline.

Every location can be overridden with an environment variable, so the same
code runs on a laptop (local folders) and on AWS (s3:// URIs) with no edits:

    STRIAX_DATA_DIR    where fetched price data is stored     (default: data)
    STRIAX_MODEL_DIR   where model artifacts are stored       (default: models)
    STRIAX_OUTPUT_DIR  where predictions are written          (default: out)
    STRIAX_TICKER      stock symbol                           (default: RBLX)
    STRIAX_PERIOD      how much history to download           (default: 5y)
"""

import os

TICKER = os.environ.get("STRIAX_TICKER", "RBLX")
HISTORY_PERIOD = os.environ.get("STRIAX_PERIOD", "5y")

DATA_DIR = os.environ.get("STRIAX_DATA_DIR", "data")
MODEL_DIR = os.environ.get("STRIAX_MODEL_DIR", "models")
OUTPUT_DIR = os.environ.get("STRIAX_OUTPUT_DIR", "out")

# File names inside those locations
PRICES_FILE = f"{TICKER}_prices.csv"
MODEL_FILE = "lstm_model.pt"
SCALER_FILE = "scaler.pkl"
METADATA_FILE = "metadata.json"
PREDICTIONS_FILE = "predictions.csv"
LATEST_FILE = "latest.json"

# Model hyperparameters (unchanged from the original EndPriceLSTM.py)
SEQ_LEN = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("STRIAX_EPOCHS", "50"))
LR = 1e-3
SEED = 42

# Chronological split: train / validation (checkpoint selection) / test (reported once)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
