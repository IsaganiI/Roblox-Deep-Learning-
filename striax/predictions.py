"""The predictions log: one row per prediction, scored once the real result is known.

This file (predictions.csv) is the contract everything downstream reads:
the dashboard, an MCP server, evaluation reports.

Columns
  as_of_date          last trading day of data the prediction used (YYYY-MM-DD)
  ticker              e.g. RBLX
  last_close          close price on as_of_date
  predicted_direction UP or DOWN for the next trading day
  prob_up             model probability that the next close is higher (0-1)
  model_version       which trained model made it (see models/metadata.json)
  created_at          UTC timestamp the prediction was made
  target_date         the next trading day (filled in once it has happened)
  actual_close        close on target_date (filled in later)
  actual_direction    UP or DOWN (filled in later)
  correct             True/False (filled in later)
"""

import pandas as pd

COLUMNS = [
    "as_of_date", "ticker", "last_close", "predicted_direction", "prob_up",
    "model_version", "created_at",
    "target_date", "actual_close", "actual_direction", "correct",
]


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def upsert(log: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Add a prediction; re-running on the same day with the same model replaces it."""
    log = log.copy()
    same = (log["as_of_date"].astype(str) == str(row["as_of_date"])) & \
           (log["model_version"].astype(str) == str(row["model_version"]))
    log = log[~same]
    new = pd.DataFrame([{c: row.get(c) for c in COLUMNS}])
    log = new if log.empty else pd.concat([log, new], ignore_index=True)
    return log.sort_values(["as_of_date", "created_at"]).reset_index(drop=True)[COLUMNS]


def score(log: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Fill in the actual outcome for every prediction whose next trading day exists."""
    log = log.copy().astype({"target_date": "object", "actual_close": "object",
                             "actual_direction": "object", "correct": "object"})
    dates = prices["Date"].dt.strftime("%Y-%m-%d").tolist()
    closes = prices["Close"].tolist()
    position = {d: i for i, d in enumerate(dates)}

    for i, r in log.iterrows():
        if pd.notna(r["actual_direction"]) and str(r["actual_direction"]) != "":
            continue
        j = position.get(str(r["as_of_date"]))
        if j is None or j + 1 >= len(dates):
            continue
        actual = "UP" if closes[j + 1] > closes[j] else "DOWN"
        log.at[i, "target_date"] = dates[j + 1]
        log.at[i, "actual_close"] = round(float(closes[j + 1]), 4)
        log.at[i, "actual_direction"] = actual
        log.at[i, "correct"] = bool(actual == r["predicted_direction"])
    return log


def summary(log: pd.DataFrame) -> dict:
    scored = log[log["correct"].notna() & (log["correct"].astype(str) != "")]
    hits = scored["correct"].astype(str).str.lower().eq("true")
    return {
        "n_predictions": int(len(log)),
        "n_scored": int(len(scored)),
        "hit_rate": round(float(hits.mean()), 4) if len(scored) else None,
    }
