import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os



_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(_ROOT, "data", "RBLX.csv")
MODEL_PATH  = os.path.join(_ROOT, "models", "lstm_model.pt")
SCALER_PATH = os.path.join(_ROOT, "models", "scaler.pkl")


SEQ_LEN     = 20
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
TRAIN_RATIO = 0.80



def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, skiprows=[1, 2])
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)


    # feats.
    df["Return"]       = df["Close"].pct_change()                   # day by day change
    df["HL_range"]     = (df["High"] - df["Low"]) / df["Close"]      
    df["OC_range"]     = (df["Close"] - df["Open"]) / df["Open"]    
    df["Vol_change"]   = df["Volume"].pct_change()

    for lag in [1, 2, 3, 5]:
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)

    for w in [5, 10, 20]:
        df[f"MA{w}"]    = df["Close"].rolling(w).mean() / df["Close"] - 1       # moving avg
        df[f"Vol_MA{w}"] = df["Volume"].rolling(w).mean() / df["Volume"] - 1    # vol mov avg

    # if next day's close > today's close
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # cleaning
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)



class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)          # up or down
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last time-step
        return self.head(out)


def train():
    os.makedirs("models", exist_ok=True)

    df = load_and_engineer(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("Date", "Target", "Close",
                                                         "High", "Low", "Open", "Volume")]
    X_raw = df[feature_cols].values
    y_raw = df["Target"].values

    split = int(len(X_raw) * TRAIN_RATIO)
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    joblib.dump(scaler, SCALER_PATH)

    X_train, y_train = make_sequences(X_train_scaled, y_train_raw, SEQ_LEN)
    X_test,  y_test  = make_sequences(X_test_scaled,  y_test_raw,  SEQ_LEN)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMClassifier(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

    # handle imbalance
    counts  = np.bincount(y_train)
    weights = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)



    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1): # train loop
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

        model.eval()
        all_preds, all_true = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_true.extend(yb.cpu().tolist())

        val_acc = accuracy_score(all_true, all_preds)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss/len(train_dl):.4f}  "
                  f"val_loss={val_loss/len(test_dl):.4f}  "
                  f"val_acc={val_acc:.4f}  {'✓ saved' if val_acc == best_val_acc else ''}")


    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())

    print("\nFinal Test Results")
    print(f"Best Val Accuracy : {best_val_acc:.4f}")
    print(f"Test  Accuracy    : {accuracy_score(all_true, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=["Down", "Up"]))
    print(f"\nModel  saved >> {MODEL_PATH}")
    print(f"Scaler saved >> {SCALER_PATH}")


if __name__ == "__main__":
    train()