"""The LSTM classifier and the sequence-windowing helper."""

import numpy as np
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # down / up
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last time step
        return self.head(out)


def make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Window i covers rows [i, i+seq_len) and ends at row i+seq_len-1.

    Its label is that LAST row's Target (the next day's direction), which is
    exactly the setup used at prediction time: the latest seq_len rows in,
    tomorrow's direction out.
    """
    n = len(X) - seq_len + 1
    return np.stack([X[i:i + seq_len] for i in range(n)]).astype(np.float32)
