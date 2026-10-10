"""Step 1 of the pipeline: download daily prices and store them as a clean CSV.

The saved file has one header row: Date,Open,High,Low,Close,Volume
(no more skiprows tricks for yfinance's multi-row header).
"""

import pandas as pd

from . import config, storage

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def prices_uri() -> str:
    return storage.join(config.DATA_DIR, config.PRICES_FILE)


def fetch_prices(ticker: str = config.TICKER, period: str = config.HISTORY_PERIOD) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, period=period, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # Newer yfinance versions return (field, ticker) column pairs; keep just the field.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[PRICE_COLS].reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    return clean_prices(df)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    for col in PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"] + PRICE_COLS)
    df = df.drop_duplicates(subset="Date", keep="last")
    return df.sort_values("Date").reset_index(drop=True)[["Date"] + PRICE_COLS]


def save_prices(df: pd.DataFrame, uri: str = None) -> str:
    uri = uri or prices_uri()
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    storage.write_csv(out, uri)
    return uri


def load_prices(uri: str = None) -> pd.DataFrame:
    return clean_prices(storage.read_csv(uri or prices_uri()))


def run_fetch() -> pd.DataFrame:
    df = fetch_prices()
    uri = save_prices(df)
    print(f"[fetch] {len(df)} rows for {config.TICKER} through "
          f"{df['Date'].iloc[-1].date()} -> {uri}")
    return df
