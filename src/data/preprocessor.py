import pandas as pd
def make_label(df):
    df = df.copy()
    df["fwd_ret"] = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (df["fwd_ret"] > 0).astype(int)
    return df