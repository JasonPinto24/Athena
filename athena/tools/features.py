import numpy as np
import pandas as pd

from athena.tools.prices import get_prices

FEATURES = [
    "ret_5", "ret_20", "ret_60", "rsi_14", "macd_hist",
    "vol_20", "volume_ratio", "dist_52w_high", "rel_20",
]
MARKET_FEATURES = ["mkt_ret_60", "mkt_vol_20"]


def rsi(close, period=14):
    change = close.diff()
    gain = change.clip(lower=0).rolling(period).mean()
    loss = (-change.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(ticker, as_of, index_ticker="^NSEI", horizon=20):
    stock = get_prices(ticker, as_of).set_index("date")
    index = get_prices(index_ticker, as_of).set_index("date")
    close = stock["close"]
    index_close = index["close"].reindex(close.index).ffill()

    df = pd.DataFrame(index=close.index)

    # Stock clues
    df["ret_5"] = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)
    df["ret_60"] = close.pct_change(60)
    df["rsi_14"] = rsi(close)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - signal) / close
    df["vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    df["volume_ratio"] = stock["volume"] / stock["volume"].rolling(20).mean()
    df["dist_52w_high"] = close / close.rolling(252).max() - 1
    df["rel_20"] = df["ret_20"] - index_close.pct_change(20)

    # Market "weather" clues (same for every stock on a given day)
    df["mkt_ret_60"] = index_close.pct_change(60)
    df["mkt_vol_20"] = index_close.pct_change().rolling(20).std() * np.sqrt(252)

    # What actually happened over the NEXT 20 trading days
    future_stock = close.shift(-horizon) / close - 1
    future_index = index_close.shift(-horizon) / index_close - 1
    df["future_excess"] = future_stock - future_index
    df["label"] = (future_stock > future_index).astype(float)
    df.loc[future_stock.isna(), "label"] = np.nan

    df["ticker"] = ticker
    return df