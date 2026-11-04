"""Live pipeline used by app.py.

Real prices -> 15 indicators -> chronological split -> model -> metrics -> backtest -> explanation.
Every number the app shows is computed here at run time from data. Nothing is hard-coded.
Only models that are actually implemented are offered (XGBoost, Logistic Regression).
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data.preprocessor import make_label
from src.features.technical import FEATURE_COLS, add_indicators
from src.split import fixed_split

AVAILABLE_MODELS = ["XGBoost", "Logistic Regression"]

# Honest roadmap shown in the app: what exists and what does not (yet).
ROADMAP = [
    ("Price download + 15 indicators", "Implemented"),
    ("Chronological split + embargo, leakage tests", "Implemented"),
    ("XGBoost, Logistic Regression baselines", "Implemented"),
    ("Cost-adjusted backtest, significance test", "Implemented"),
    ("News download + FinBERT sentiment", "Not built yet"),
    ("PatchTST, TFT, Mamba, CNN-BiLSTM-Attn", "Not built yet"),
    ("FinPatch-Mamba (novel model)", "Not built yet"),
]


# ---------------------------------------------------------------- data
def _clean_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date", "open", "high", "low", "close", "volume"]]


def load_prices(ticker: str, refresh: bool = False, max_age_hours: float = 6.0):
    """Return (prices_df, info). Uses the local CSV cache; re-downloads from Yahoo Finance when
    the cache is older than `max_age_hours` or `refresh` is True. Falls back to the cache if the
    download fails (and says so in info['warning'])."""
    cache = config.PRICE_DIR / f"{ticker.replace('^', '')}.csv"
    cached = pd.read_csv(cache, parse_dates=["date"]) if cache.exists() else None
    age_h = (time.time() - cache.stat().st_mtime) / 3600 if cache.exists() else None
    info = {"source": "cache", "warning": None, "fetched_at": None}

    if refresh or cached is None or age_h > max_age_hours:
        try:
            import yfinance as yf
            raw = yf.download(ticker, start=config.START_DATE, auto_adjust=True, progress=False)
            fresh = _clean_download(raw)
            if fresh.empty:
                raise RuntimeError("Yahoo Finance returned no rows")
            out = fresh.assign(ticker=ticker, sector=config.STOCK_TO_SECTOR.get(ticker, "Index"))
            out.to_csv(cache, index=False)
            info.update(source="Yahoo Finance (live download)", fetched_at=pd.Timestamp.now())
            cached = out
        except Exception as e:                      # network, symbol, rate limit ...
            if cached is None:
                raise RuntimeError(f"Could not download {ticker} and no cached file exists: {e}")
            info["warning"] = f"Live download failed ({type(e).__name__}); showing cached data."
    if info["fetched_at"] is None:
        info["fetched_at"] = pd.Timestamp.fromtimestamp(cache.stat().st_mtime)
    prices = cached[["date", "open", "high", "low", "close", "volume"]]
    prices = prices.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    return prices.reset_index(drop=True), info


def build_dataset(prices: pd.DataFrame):
    """Return (labelled rows, latest row). The latest row has features but no label yet."""
    lab = make_label(add_indicators(prices)).replace([np.inf, -np.inf], np.nan)
    ready = lab.dropna(subset=FEATURE_COLS)
    latest = ready.iloc[[-1]]
    data = ready.dropna(subset=["target"]).reset_index(drop=True)
    return data, latest


def split_data(data: pd.DataFrame):
    tr, va, te = fixed_split(data)
    if len(tr) < 300 or len(te) < 60:
        raise ValueError(f"Not enough history to train/test (train={len(tr)}, test={len(te)}). "
                         "Pick a stock with a longer price history.")
    return tr, va, te


# ---------------------------------------------------------------- models
def make_model(name: str, seed: int = 42):
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss",
                             early_stopping_rounds=30, random_state=seed, n_jobs=1)
    if name == "Logistic Regression":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    raise ValueError(f"Model '{name}' is not implemented yet")


def fit_model(name, tr, va, seed=42):
    model = make_model(name, seed)
    Xtr, ytr = tr[FEATURE_COLS], tr["target"]
    if name == "XGBoost" and len(va) > 20:
        model.fit(Xtr, ytr, eval_set=[(va[FEATURE_COLS], va["target"])], verbose=False)
    elif name == "XGBoost":
        model.set_params(early_stopping_rounds=None)
        model.fit(Xtr, ytr, verbose=False)
    else:
        model.fit(Xtr, ytr)
    return model


def predict_up(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURE_COLS])[:, 1]


# ---------------------------------------------------------------- evaluation
def classification_metrics(y, p, thr: float = 0.5) -> dict:
    pred = (p > thr).astype(int)
    n, k = len(y), int((pred == y).sum())
    acc = k / n
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)
    return {
        "n_test": n,
        "accuracy": acc,
        "acc_ci_low": acc - half,
        "acc_ci_high": acc + half,
        "p_value_vs_50pct": binomtest(k, n, 0.5).pvalue,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
        "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
    }


def naive_baselines(te: pd.DataFrame) -> pd.DataFrame:
    y = te["target"].astype(int)
    prev_dir = (te["ret_1d"] > 0).astype(int)                 # repeat yesterday's direction
    always_up = pd.Series(1, index=te.index)
    rows = [("Repeat yesterday's direction", accuracy_score(y, prev_dir)),
            ("Always predict UP", accuracy_score(y, always_up))]
    return pd.DataFrame(rows, columns=["baseline", "accuracy"])


def _stats(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    sd = ret.std()
    return {"total_return": eq.iloc[-1] - 1,
            "sharpe": float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min())}


def backtest(te: pd.DataFrame, p: np.ndarray, cost: float = config.TRANSACTION_COST):
    """Long when P(up) > 0.5 else cash. Decision at close of day t, earns return t->t+1.
    `cost` is charged per side on every change of position."""
    pos = pd.Series((p > 0.5).astype(float), index=te.index)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0]
    strat = pos * te["fwd_ret"] - cost * turnover
    bh = te["fwd_ret"].copy()
    bh.iloc[0] -= cost                                         # buy-and-hold pays entry cost too
    curve = pd.DataFrame({"date": te["date"].values,
                          "Strategy": (1 + strat).cumprod().values,
                          "Buy and hold": (1 + bh).cumprod().values})
    s, b = _stats(strat), _stats(bh)
    s["trades"] = int((turnover > 0).sum())
    s["time_in_market"] = float(pos.mean())
    return curve, {"Strategy": s, "Buy and hold": b}


# ---------------------------------------------------------------- explanation
def explain(model, name, X_test: pd.DataFrame, x_latest: pd.DataFrame):
    """Return (global importance df, local contribution df, method text).
    Local contributions are toward UP (positive = pushes UP)."""
    X_test, x_latest = X_test[FEATURE_COLS], x_latest[FEATURE_COLS]
    if name == "Logistic Regression":
        scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
        c_test = scaler.transform(X_test) * clf.coef_[0]
        c_last = (scaler.transform(x_latest) * clf.coef_[0])[0]
        method = "Exact log-odds contribution (coefficient x standardised value)"
    else:
        try:
            import shap
            sv = shap.TreeExplainer(model).shap_values(X_test)
            sv = sv[1] if isinstance(sv, list) else sv
            c_test = np.asarray(sv)
            c_last = np.asarray(shap.TreeExplainer(model).shap_values(x_latest)).reshape(-1)
            method = "SHAP (TreeExplainer), in log-odds"
        except Exception:
            imp = model.feature_importances_
            g = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp}).sort_values("importance")
            return g, None, "XGBoost gain importance (install `shap` for SHAP values)"
    g = pd.DataFrame({"feature": FEATURE_COLS, "importance": np.abs(c_test).mean(0)}).sort_values("importance")
    loc = pd.DataFrame({"feature": FEATURE_COLS, "contribution": c_last})
    loc = loc.reindex(loc["contribution"].abs().sort_values().index)
    return g, loc, method


# ---------------------------------------------------------------- orchestration
def run_experiment(prices: pd.DataFrame, model_name: str, seed: int = 42,
                   cost: float = config.TRANSACTION_COST) -> dict:
    data, latest = build_dataset(prices)
    tr, va, te = split_data(data)
    model = fit_model(model_name, tr, va, seed)
    p = predict_up(model, te)
    curve, bt = backtest(te, p, cost)
    glob, loc, method = explain(model, model_name, te, latest)
    p_next = float(predict_up(model, latest)[0])
    return {
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "date_ranges": {k: (d["date"].min(), d["date"].max()) for k, d in
                        (("train", tr), ("val", va), ("test", te))},
        "metrics": classification_metrics(te["target"].astype(int), p),
        "baselines": naive_baselines(te),
        "curve": curve, "backtest": bt,
        "global_expl": glob, "local_expl": loc, "expl_method": method,
        "next_day": {"as_of": latest["date"].iloc[0], "close": float(latest["close"].iloc[0]),
                     "p_up": p_next},
        "test_start": te["date"].min(),
    }


def seed_stability(prices: pd.DataFrame, model_name: str, seeds=None) -> pd.DataFrame:
    seeds = seeds or config.SEEDS
    data, _ = build_dataset(prices)
    tr, va, te = split_data(data)
    rows = []
    for s in seeds:
        m = fit_model(model_name, tr, va, s)
        met = classification_metrics(te["target"].astype(int), predict_up(m, te))
        rows.append({"seed": s, "accuracy": met["accuracy"], "mcc": met["mcc"], "auc": met["auc"]})
    return pd.DataFrame(rows)