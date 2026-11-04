FEATURE_COLS = ["ret_1d","ret_5d","ret_20d","ma_ratio","vol_20d","rsi_14","mom_10","ma_5_20","high_low","close_open","vol_ratio","ret_10d","range_vol","trend_20","dist_ma50"]
import pandas as pd
import numpy as np
def add_indicators(df):
    df = df.copy()
    c = df["close"]
    df["ret_1d"] = c.pct_change(1)
    df["ret_5d"] = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_20d"] = c.pct_change(20)
    df["ma_ratio"] = c / c.rolling(20).mean()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["mom_10"] = c - c.shift(10)
    df["ma_5_20"] = c.rolling(5).mean() / c.rolling(20).mean()
    df["high_low"] = (df["high"] - df["low"]) / c
    df["close_open"] = (c - df["open"]) / df["open"]
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["range_vol"] = (df["high"] - df["low"]) / df["close"].rolling(20).std()
    df["trend_20"] = c / c.shift(20) - 1
    df["dist_ma50"] = c / c.rolling(50).mean() - 1
    return df