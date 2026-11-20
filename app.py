# Nifty-XAI - 4 Models (Naive / Logistic / XGBoost / LSTM) + SHAP XAI + Research Evidence
# Education / research purpose only. Run: streamlit run app.py
#
# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   scikit-learn
#   xgboost
#   tensorflow-cpu   (optional - LSTM venumna)
#
# Flow:
#   1. Top   : Nifty 50 last-close ticker (scrolling)
#   2. Middle: Amount enter -> "Predict & Suggest" button
#   3. Bottom: Evlo amount-ayum use panni, ella sector-layum irundhu entha stock,
#              evlo qty, evlo Rs - XAI reason oda small table

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# SETTINGS (delivery trade charges - un broker contract note paathu maathunga)
# ------------------------------------------------
YEARS = 5
MAX_STOCKS = 8          # max evlo stocks suggest pannanum
MAX_PER_SECTOR = 2      # oru sector-la max evlo stock (diversification)
BROKERAGE = 0.0         # Rs per order (Zerodha delivery = 0)
STT_PCT = 0.100         # buy + sell
STAMP_PCT = 0.015       # buy only
EXCH_PCT = 0.003        # NSE transaction charge
SEBI_PCT = 0.0001
DP_CHARGE = 15.93       # Rs per stock SOLD (GST serthu)

STOCKS = {
    "HDFCBANK.NS": ("HDFC Bank", "Banking"), "ICICIBANK.NS": ("ICICI Bank", "Banking"),
    "SBIN.NS": ("SBI", "Banking"), "AXISBANK.NS": ("Axis Bank", "Banking"),
    "KOTAKBANK.NS": ("Kotak Bank", "Banking"), "BAJFINANCE.NS": ("Bajaj Finance", "Finance"),
    "JIOFIN.NS": ("Jio Financial", "Finance"), "HDFCLIFE.NS": ("HDFC Life", "Finance"),
    "SBILIFE.NS": ("SBI Life", "Finance"), "BSE.NS": ("BSE Ltd", "Finance"),
    "INFY.NS": ("Infosys", "IT"), "TCS.NS": ("TCS", "IT"),
    "HCLTECH.NS": ("HCL Tech", "IT"), "TECHM.NS": ("Tech Mahindra", "IT"), "WIPRO.NS": ("Wipro", "IT"),
    "RELIANCE.NS": ("Reliance", "Energy"), "ONGC.NS": ("ONGC", "Energy"),
    "NTPC.NS": ("NTPC", "Energy"), "POWERGRID.NS": ("Power Grid", "Energy"),
    "COALINDIA.NS": ("Coal India", "Energy"), "TATAPOWER.NS": ("Tata Power", "Energy"),
    "MARUTI.NS": ("Maruti", "Auto"), "M&M.NS": ("M&M", "Auto"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto", "Auto"), "EICHERMOT.NS": ("Eicher Motors", "Auto"),
    "TMPV.NS": ("Tata Motors PV", "Auto"),
    "SUNPHARMA.NS": ("Sun Pharma", "Pharma"), "DRREDDY.NS": ("Dr Reddy", "Pharma"),
    "CIPLA.NS": ("Cipla", "Pharma"), "DIVISLAB.NS": ("Divis Lab", "Pharma"),
    "APOLLOHOSP.NS": ("Apollo Hospitals", "Healthcare"), "MAXHEALTH.NS": ("Max Healthcare", "Healthcare"),
    "ITC.NS": ("ITC", "FMCG"), "HINDUNILVR.NS": ("HUL", "FMCG"),
    "NESTLEIND.NS": ("Nestle", "FMCG"), "TATACONSUM.NS": ("Tata Consumer", "FMCG"),
    "TATASTEEL.NS": ("Tata Steel", "Metals"), "HINDALCO.NS": ("Hindalco", "Metals"),
    "JSWSTEEL.NS": ("JSW Steel", "Metals"), "GRASIM.NS": ("Grasim", "Cement"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Cement"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Telecom"), "LT.NS": ("L&T", "Infra"),
    "ASIANPAINT.NS": ("Asian Paints", "Paints"), "TITAN.NS": ("Titan", "Consumer"),
    "ADANIPORTS.NS": ("Adani Ports", "Infra"), "ADANIENT.NS": ("Adani Enterprises", "Infra"),
    "ETERNAL.NS": ("Eternal (Zomato)", "Consumer"), "INDIGO.NS": ("IndiGo", "Aviation"),
    "TRENT.NS": ("Trent", "Retail"),
}

FEATURES = ['Return_1d', 'Return_5d', 'Return_10d', 'Return_20d', 'MA_Ratio_5_20',
            'MA_Ratio_20_50', 'Price_vs_MA20', 'Vol_10', 'Vol_20', 'RSI', 'Vol_Change']


# ------------------------------------------------
# CHARGES
# ------------------------------------------------
def charges(value, side):
    """Oru order ku total charges (Rs). side = 'buy' / 'sell'."""
    exch = value * (EXCH_PCT + SEBI_PCT) / 100
    total = BROKERAGE + exch + 0.18 * (BROKERAGE + exch) + value * STT_PCT / 100
    if side == "buy":
        total += value * STAMP_PCT / 100
    else:
        total += DP_CHARGE
    return total


# ------------------------------------------------
# DATA
# ------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_ticker_data():
    """50 stocks last close + % change - oru download-la."""
    syms = list(STOCKS)
    out = []
    try:
        raw = yf.download(syms, period="7d", group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        for s in syms:
            try:
                c = raw[s]["Close"].dropna()
                if len(c) >= 2:
                    out.append((STOCKS[s][0], float(c.iloc[-1]),
                                (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100))
            except Exception:
                continue
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_all(years):
    syms = list(STOCKS)
    raw = yf.download(syms, period=f"{years}y", group_by="ticker", auto_adjust=True,
                      progress=False, threads=True)
    out = {}
    for s in syms:
        try:
            d = raw[s].dropna(subset=["Close"])[["Close", "Volume"]].copy()
            if len(d) >= 250:
                out[s] = d
        except Exception:
            continue
    return out


# ------------------------------------------------
# FEATURES + MODEL
# ------------------------------------------------
def make_features(raw):
    """Latest row DROP pannala - adhu dhaan naalaiku prediction row."""
    df = raw.copy()
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    ma5, ma20, ma50 = (df['Close'].rolling(n).mean() for n in (5, 20, 50))
    df['MA_Ratio_5_20'] = ma5 / ma20
    df['MA_Ratio_20_50'] = ma20 / ma50
    df['Price_vs_MA20'] = df['Close'] / ma20 - 1
    df['Vol_10'] = df['Return_1d'].rolling(10).std()
    df['Vol_20'] = df['Return_1d'].rolling(20).std()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 1e-10)
    df['RSI'] = (100 - 100 / (1 + gain / loss)).clip(0, 100)
    df['Vol_Change'] = df['Volume'].pct_change(1)
    df['NextRet'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['NextRet'] > 0).astype(float).where(df['NextRet'].notna())
    df = df.replace([np.inf, -np.inf], np.nan)
    df['Vol_Change'] = df['Vol_Change'].clip(-5, 5)
    return df.dropna(subset=FEATURES)


def describe(feat, v):
    """Feature value -> human readable."""
    if feat == 'RSI':
        return f"RSI {v:.0f}"
    if feat == 'Return_1d':
        return f"last day {v * 100:+.1f}%"
    if feat == 'Return_5d':
        return f"5-day {v * 100:+.1f}%"
    if feat == 'Return_10d':
        return f"10-day {v * 100:+.1f}%"
    if feat == 'Return_20d':
        return f"20-day {v * 100:+.1f}%"
    if feat == 'MA_Ratio_5_20':
        return "MA5 above MA20" if v > 1 else "MA5 below MA20"
    if feat == 'MA_Ratio_20_50':
        return "MA20 above MA50" if v > 1 else "MA20 below MA50"
    if feat == 'Price_vs_MA20':
        return f"price {v * 100:+.1f}% vs MA20"
    if feat in ('Vol_10', 'Vol_20'):
        return f"volatility {v * 100:.1f}%"
    if feat == 'Vol_Change':
        return f"volume {v * 100:+.0f}% vs prev day"
    return feat


import math
import importlib.util

LSTM_OK = importlib.util.find_spec("tensorflow") is not None
EMBARGO = 5          # train/test naduvula 5 naal gap (leakage thavirkka)
SEQ_LEN = 60
MODELS = [("naive", "1. Naive"), ("lr", "2. Logistic"), ("xgb", "3. XGBoost"), ("lstm", "4. LSTM")]


def new_xgb():
    return xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='logloss', random_state=42)


def run_lstm(X, y, cut, split, x_now):
    """LSTM: train = X[:cut], test = X[split:]. Scaler train-la mattum fit.
    Window = last 60 days (indru serthu) -> naalaiku Up/Down."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    tf.random.set_seed(42)
    np.random.seed(42)
    sc = StandardScaler().fit(X[:cut])
    data = sc.transform(np.vstack([X, x_now]))

    def windows(ends):
        return np.array([data[i - SEQ_LEN + 1:i + 1] for i in ends])

    tr_ends = list(range(SEQ_LEN - 1, cut))
    te_ends = list(range(split, len(X)))
    Xs_tr, ys_tr = windows(tr_ends), y[tr_ends].astype(float)
    Xs_te = windows(te_ends)
    model = Sequential([Input(shape=(SEQ_LEN, X.shape[1])),
                        LSTM(32, return_sequences=True), Dropout(0.2),
                        LSTM(32), Dropout(0.2),
                        Dense(16, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(Xs_tr, ys_tr, epochs=6, batch_size=64, verbose=0, validation_split=0.1)
    p_te = model.predict(Xs_te, verbose=0).flatten()
    p_now = float(model.predict(data[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[1]), verbose=0)[0][0])
    return p_te, p_now


def analyse(raw, use_lstm):
    f = make_features(raw)
    train = f.dropna(subset=['Target'])
    X, y = train[FEATURES].values, train['Target'].astype(int).values
    nr = train['NextRet'].values
    x_now = f[FEATURES].iloc[[-1]].values
    split = int(len(X) * 0.8)
    cut = split - EMBARGO
    Xtr, ytr, Xte, yte, nr_te = X[:cut], y[:cut], X[split:], y[split:], nr[split:]

    # ---------- Holdout (out-of-sample) ----------
    p_te, p_now = {}, {}
    up_tr = (Xtr[:, 0] > 0)                                   # Naive = "indru Up na naalaiku Up"
    pu = ytr[up_tr].mean() if up_tr.any() else 0.5
    pd_ = ytr[~up_tr].mean() if (~up_tr).any() else 0.5
    p_te["naive"] = (Xte[:, 0] > 0).astype(float)
    p_now["naive"] = float(pu if x_now[0, 0] > 0 else pd_)

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    p_te["lr"] = lr.predict_proba(sc.transform(Xte))[:, 1]
    xg = new_xgb().fit(Xtr, ytr)
    p_te["xgb"] = xg.predict_proba(Xte)[:, 1]

    # ---------- Final models (ella data-vilum train) -> naalaiku prediction ----------
    sc2 = StandardScaler().fit(X)
    lr2 = LogisticRegression(max_iter=1000).fit(sc2.transform(X), y)
    p_now["lr"] = float(lr2.predict_proba(sc2.transform(x_now))[0][1])
    xg2 = new_xgb().fit(X, y)
    p_now["xgb"] = float(xg2.predict_proba(x_now)[0][1])

    if use_lstm and LSTM_OK:
        try:
            p_te["lstm"], p_now["lstm"] = run_lstm(X, y, cut, split, x_now)
        except Exception:
            pass

    voters = [m for m in ("lr", "xgb", "lstm") if m in p_te]   # Naive = baseline, final vote-la illa
    p_te["ens"] = np.mean([p_te[m] for m in voters], axis=0)
    p_now["ens"] = float(np.mean([p_now[m] for m in voters]))

    out = {"n_test": int(len(yte)), "voters": "+".join(m.upper() for m in voters)}
    base_pred = int(ytr.mean() >= 0.5)
    out["k_base"] = int((yte == base_pred).sum())
    out["base_up_ret"] = float(nr_te.sum())                    # "ellaa naalum vaangu" strategy
    for m in ["naive", "lr", "xgb", "lstm", "ens"]:
        if m in p_te:
            pred = (p_te[m] > 0.5).astype(int)
            out[f"p_{m}"] = p_now[m]
            out[f"k_{m}"] = int((pred == yte).sum())
            out[f"up_{m}"] = int(pred.sum())
            out[f"sr_{m}"] = float(nr_te[pred == 1].sum())
        else:
            out[f"p_{m}"] = np.nan
            out[f"k_{m}"] = out[f"up_{m}"] = out[f"sr_{m}"] = np.nan
    out["acc"] = out["k_ens"] / out["n_test"]
    out["base"] = out["k_base"] / out["n_test"]
    out["prob"] = out["p_ens"]

    # ---------- XAI: XGBoost SHAP contributions (last column = bias) ----------
    contrib = xg2.get_booster().predict(xgb.DMatrix(x_now), pred_contribs=True)[0][:-1]
    order = np.argsort(contrib) if out["prob"] < 0.5 else np.argsort(-contrib)
    out["why"] = ", ".join(
        f"{describe(FEATURES[i], x_now[0][i])} ({'+' if contrib[i] > 0 else '-'})" for i in order[:2])

    # ---------- Expected return = prob x avg up move + (1-prob) x avg down move ----------
    avg_up = float(nr[nr > 0].mean()) if (nr > 0).any() else 0.0
    avg_dn = float(nr[nr <= 0].mean()) if (nr <= 0).any() else 0.0
    out["avg_up"], out["avg_dn"] = avg_up * 100, avg_dn * 100
    out["exp_ret"] = (out["prob"] * avg_up + (1 - out["prob"]) * avg_dn) * 100
    out["rsi"] = float(f["RSI"].iloc[-1])
    out["price"] = float(raw['Close'].iloc[-1])
    out["date"] = str(raw.index[-1].date())
    votes = [out[f"p_{m}"] > 0.5 for m in ("naive", "lr", "xgb", "lstm") if np.isfinite(out[f"p_{m}"])]
    out["votes_up"], out["n_models"] = int(sum(votes)), len(votes)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def analyse_symbol(sym, years, use_lstm):
    data = load_all(years)
    if sym not in data:
        return {"error": "data kidaikala (illa < 250 naal)"}
    try:
        return analyse(data[sym], use_lstm)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------- Research statistics ----------
def binom_p(k, n):
    """Two-sided p-value vs 50% (normal approx). Stocks correlated -> optimistic."""
    if n == 0:
        return np.nan
    z = (k - n / 2) / math.sqrt(n / 4)
    return math.erfc(abs(z) / math.sqrt(2))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def evidence_table(res, ref_cost_pct):
    """4 models + ensemble: pooled out-of-sample accuracy, significance, signal return vs cost."""
    n_total = int(res.n_test.sum())
    base_acc = res.k_base.sum() / n_total
    rows = []
    for key, label in MODELS + [("ens", "Final (vote)")]:
        k = res[f"k_{key}"]
        if k.isna().all():
            rows.append({"Model": label, "Accuracy": "N/A", "95% CI": "", "p (vs 50%)": "",
                         "Edge vs baseline": "", "Up signals": "", "Gross/signal": "", "Net/signal": ""})
            continue
        ok = k.notna()
        kk, nn = int(k[ok].sum()), int(res.n_test[ok].sum())
        lo, hi = wilson(kk, nn)
        ups = int(res[f"up_{key}"][ok].sum())
        gross = (res[f"sr_{key}"][ok].sum() / ups * 100) if ups else np.nan
        rows.append({
            "Model": label, "Accuracy": f"{kk / nn * 100:.2f}%", "95% CI": f"{lo * 100:.1f}-{hi * 100:.1f}%",
            "p (vs 50%)": f"{binom_p(kk, nn):.3f}",
            "Edge vs baseline": f"{(kk / nn - res.k_base[ok].sum() / nn) * 100:+.2f}%",
            "Up signals": f"{ups:,}",
            "Gross/signal": f"{gross:+.3f}%" if ups else "-",
            "Net/signal": f"{gross - ref_cost_pct:+.3f}%" if ups else "-",
        })
    return pd.DataFrame(rows), base_acc, n_total


def build_portfolio(df, budget):
    """df: analysis rows. Return list of dict (Company, Qty, ...) - total buy cost <= budget."""
    cand = df[(df.prob > 0.5) & (df.exp_ret > 0)].sort_values("exp_ret", ascending=False)
    if cand.empty:
        return []
    recs = cand.to_dict("records")

    picked, targets = [], []
    for n in range(min(MAX_STOCKS, len(recs)), 0, -1):
        pool = [r for r in recs if r["price"] <= budget / n]
        sel, cnt = [], {}
        for r in pool:
            if cnt.get(r["Sector"], 0) >= MAX_PER_SECTOR:
                continue
            sel.append(r)
            cnt[r["Sector"]] = cnt.get(r["Sector"], 0) + 1
            if len(sel) == n:
                break
        if len(sel) < n:
            continue
        w = np.array([max(r["prob"] - 0.5, 0.005) for r in sel])
        t = budget * 0.998 * w / w.sum()
        if all(t[i] >= sel[i]["price"] for i in range(n)):
            picked, targets = sel, list(t)
            break
    if not picked:
        return []

    qty = [int(targets[i] // picked[i]["price"]) for i in range(len(picked))]

    def total_cost(q):
        return sum(q[i] * picked[i]["price"] + charges(q[i] * picked[i]["price"], "buy")
                   for i in range(len(picked)) if q[i] > 0)

    while total_cost(qty) > budget:                      # safety - charges thaandina kuraikkum
        j = max(range(len(qty)), key=lambda i: qty[i] * picked[i]["price"])
        qty[j] -= 1
    while True:                                          # meedhi cash-ku innum share vaangum
        order = sorted(range(len(qty)),
                       key=lambda i: qty[i] * picked[i]["price"] / targets[i])
        for i in order:
            trial = qty.copy()
            trial[i] += 1
            if total_cost(trial) <= budget:
                qty = trial
                break
        else:
            break

    out = []
    for i, r in enumerate(picked):
        if qty[i] > 0:
            r = dict(r)
            r["qty"] = qty[i]
            r["amount"] = qty[i] * r["price"]
            out.append(r)
    return out


# ===== UI =====
def pct(v):
    return "N/A" if v is None or not np.isfinite(v) else f"{v * 100:.1f}%"


st.set_page_config(page_title="Nifty-XAI", page_icon="📈", layout="wide")
st.title("📈 Nifty-XAI")
st.caption("Nifty 50 stocks-ku naalaiku Up/Down predict panni (4 models), un amount-ku edha vaanganum nu sollum. Education / research project.")

# ---- 1. TOP: scrolling ticker ----
tick = get_ticker_data()
if tick:
    items = "".join(
        f'<span class="{"up" if ch >= 0 else "dn"}">{"▲" if ch >= 0 else "▼"} {nm} ₹{p:,.2f} ({ch:+.2f}%)</span>'
        for nm, p, ch in tick)
    components.html(f"""
    <style>
      body{{margin:0}}
      .wrap{{background:#1a237e;color:#fff;overflow:hidden;white-space:nowrap;border-radius:8px;
             font:600 14px system-ui,sans-serif;padding:11px 0}}
      .track{{display:inline-block;padding-left:100%;animation:scroll 150s linear infinite}}
      .wrap:hover .track{{animation-play-state:paused}}
      .track span{{margin-right:34px}}
      .up{{color:#69f0ae}} .dn{{color:#ff8a80}}
      @keyframes scroll{{from{{transform:translateX(0)}}to{{transform:translateX(-100%)}}}}
    </style>
    <div class="wrap"><div class="track">{items}</div></div>
    """, height=52)
    st.caption("Nifty 50 - last market close & previous close-la irundhu % change (5 min-ku oru thadava refresh). Mouse vachaa pause aagum.")
else:
    st.warning("Ticker data kidaikala (yfinance / internet check pannunga).")

# ---- 2. MIDDLE: amount + button ----
st.divider()
c1, c2, c3 = st.columns([2, 1.4, 1])
amount = c1.number_input("💰 Amount (Rs)", min_value=1000, max_value=10_000_000, value=10000, step=1000)
use_lstm = c2.checkbox("LSTM um serkkanum (first time 5-15 min)", value=LSTM_OK, disabled=not LSTM_OK,
                       help="TensorFlow install aagi irundha mattum work aagum. Illa-na LSTM 'N/A' nu varum.")
if not LSTM_OK:
    c2.caption("⚠️ TensorFlow illa - LSTM N/A")
c3.write("")
go = c3.button("🚀 Predict & Suggest", type="primary", use_container_width=True)

if go:
    rows, errs = [], []
    bar = st.progress(0, text="Starting...")
    for i, (sym, (name, sector)) in enumerate(STOCKS.items()):
        bar.progress((i + 1) / len(STOCKS), text=f"{name} ({i + 1}/{len(STOCKS)})")
        r = analyse_symbol(sym, YEARS, bool(use_lstm))
        if "error" in r:
            errs.append(f"{name}: {r['error']}")
        else:
            r = dict(r)
            r.update({"Company": name, "Sector": sector, "Symbol": sym})
            rows.append(r)
    bar.empty()
    st.session_state.update({"budget": int(amount), "res": pd.DataFrame(rows), "errs": errs})

# ---- 3. BOTTOM: result ----
if "res" not in st.session_state:
    st.info("👆 Amount enter panni, 'Predict & Suggest' click pannunga.")
    st.stop()

res, errs, budget = st.session_state["res"], st.session_state["errs"], st.session_state["budget"]
if res.empty:
    st.error("Data / model run aagala. Internet check pannitu thirumba try pannunga.")
    for e in errs[:5]:
        st.write(e)
    st.stop()

ref_cost = (charges(10000, "buy") + charges(10000, "sell")) / 10000 * 100   # Rs 10,000 position-ku round-trip %
ev, base_acc, n_total = evidence_table(res, ref_cost)

# ---- 3a. 4 MODELS ----
st.subheader("🧠 4 Models - Out-of-sample Accuracy (last 20% data, 50 stocks pooled)")
cols = st.columns(4)
for col, (key, label) in zip(cols, MODELS):
    k = res[f"k_{key}"]
    if k.isna().all():
        col.metric(label, "N/A", "LSTM off / TensorFlow illa", delta_color="off")
        continue
    ok = k.notna()
    nn = res.n_test[ok].sum()
    acc, b = k[ok].sum() / nn, res.k_base[ok].sum() / nn
    col.metric(label, f"{acc * 100:.2f}%", f"{(acc - b) * 100:+.2f}% vs baseline ({b * 100:.1f}%)")
st.caption("Naive = 'indru Up na naalaiku Up' (simple rule). Logistic / XGBoost / LSTM = machine learning. "
           "Baseline = 'eppavum adhigam vara direction-a solluradhu'. Final prediction = Logistic + XGBoost + LSTM average (Naive baseline-a vote-la serkkala).")

# ---- 3b. BUY LIST ----
pf = build_portfolio(res, budget)
st.subheader(f"🛒 ₹{budget:,} ku - Entha Stock, Evlo Count")
if not pf:
    st.info("Indha nerathula Up-nu sollura strong stock illa (illa un amount-ku vaanga mudiyala). "
            "Cash-ah vechurunga - NO TRADE dhaan best decision.")
else:
    table = pd.DataFrame([{
        "Company": r["Company"], "Sector": r["Sector"], "Qty": r["qty"],
        "Price": f"₹{r['price']:,.2f}", "Amount": f"₹{r['amount']:,.2f}",
        "Naive Up": pct(r["p_naive"]), "Logistic Up": pct(r["p_lr"]),
        "XGBoost Up": pct(r["p_xgb"]), "LSTM Up": pct(r["p_lstm"]),
        "Final Up": pct(r["prob"]), "Models Up": f"{r['votes_up']}/{r['n_models']}",
        "XAI - Yen Up? (SHAP)": r["why"],
    } for r in pf])
    st.dataframe(table, hide_index=True, use_container_width=True)

    invested = sum(r["amount"] for r in pf)
    buy_ch = sum(charges(r["amount"], "buy") for r in pf)
    cash_left = budget - invested - buy_ch
    exp_sell = sum(r["amount"] * (1 + r["exp_ret"] / 100) for r in pf)
    sell_ch = sum(charges(r["amount"] * (1 + r["exp_ret"] / 100), "sell") for r in pf)
    gross = exp_sell - invested
    net = gross - buy_ch - sell_ch
    breakeven = (buy_ch + sell_ch) / invested * 100

    left, right = st.columns([1.2, 1])
    with left:
        a, b = st.columns(2)
        a.metric("Amount", f"₹{budget:,.2f}")
        b.metric("Invested (shares)", f"₹{invested:,.2f}")
        a.metric("Buy Charges", f"₹{buy_ch:,.2f}")
        b.metric("Cash Left", f"₹{cash_left:,.2f}")
        a.metric("Expected Sell Value (1 day)", f"₹{exp_sell:,.2f}")
        b.metric("Exp. Net P/L (1 day)", f"₹{net:,.2f}", f"{net / invested * 100:+.2f}%")
        sec_split = pd.DataFrame(pf).groupby("Sector")["amount"].sum().sort_values(ascending=False)
        st.caption("Sector split: " + " | ".join(f"{s} ₹{a_:,.0f}" for s, a_ in sec_split.items()))
    with right:
        verdict = (f"Net **positive** (₹{net:,.2f}) - aana idhu chinna number, model accuracy 50-55% dhaan, so guarantee illa."
                   if net > 0 else
                   "Net **negative** - 1 naal trade-la charges profit-ai saapiduchu. Ippadi trade panna loss varra chance adhigam.")
        st.info(
            "**📝 Exp. Net P/L - ena artham?**\n\n"
            f"1. Model-padi {len(pf)} stocks-kum serthu ₹{invested:,.0f} la, naalaiku **gross ₹{gross:,.2f}** varum nu expect "
            "(probability x avg Up/Down move).\n"
            f"2. Aana **buy charges ₹{buy_ch:,.2f} + sell charges ₹{sell_ch:,.2f}** (DP ₹{DP_CHARGE} per stock serthu) kodukkanum.\n"
            f"3. Gross - Charges = **₹{net:,.2f}**. {verdict}\n"
            f"4. Break-even: stocks avg **{breakeven:.2f}%** move aaganum - appo dhaan charges cover aagum. "
            f"Oru stock-oda typical Up-day move ~{np.mean([r['avg_up'] for r in pf]):.1f}% dhaan."
        )

    # ---- 3c. KUTTY NOTES ----
    st.subheader("📒 Kutty Notes - Eppo Buy, Eppo Sell")
    plan = pd.DataFrame([{
        "Company": r["Company"], "Qty": r["qty"],
        "Buy around": f"₹{r['price']:,.2f}",
        "Sell target (typical Up day)": f"₹{r['price'] * (1 + r['avg_up'] / 100):,.2f}",
        "Exit if below (typical Down day)": f"₹{r['price'] * (1 + r['avg_dn'] / 100):,.2f}",
        "Note": ("⚠️ RSI %.0f - overbought, jaagradhai" % r["rsi"]) if r["rsi"] > 70
                else ("RSI %.0f - oversold, bounce chance" % r["rsi"]) if r["rsi"] < 30 else "",
    } for r in pf])
    st.dataframe(plan, hide_index=True, use_container_width=True)

    avoid = res[res.prob < 0.45].sort_values("prob").head(3)
    avoid_txt = ", ".join(f"{r.Company} ({r.prob * 100:.0f}%)" for r in avoid.itertuples()) or "illa"
    st.markdown(
        f"- **BUY eppo?** Indha prediction **{res.date.max()}** close data-la irundhu. Next trading day market open (9:15 AM) aanadhum "
        "udane vaanga vendaam - first 15-20 min price romba hop aagum, settle aana piragu **limit order** podunga.\n"
        "- **SELL eppo?** Model **1-naal** prediction. So vaangina next trading day-oda kadaisi (3:00-3:25 PM) la sell pannuradhu dhaan plan.\n"
        "- **Naduvula profit?** Price 'Sell target' touch aana, adhukku mela wait pannaama profit book pannalaam.\n"
        "- **Naduvula loss?** Price 'Exit if below' keela poitta, model prediction thappunu artham - perusaa loss aagaama sell pannalaam.\n"
        f"- **Vaanga vendaam (Down chance adhigam):** {avoid_txt}.\n"
        "- **Mukkiyam:** Model *close-to-close* predict pannum. Nee market open-la vaangina overnight gap already nadandhirukkum, "
        "adhu model-ku theriyadhu. Same-day sell (intraday) panna charges vera maadhiri irukkum.\n"
        "- Idhu model-oda estimate. Investment advice illa."
    )

# ---- 3d. RESEARCH EVIDENCE ----
st.subheader("🔬 Research Evidence - Model nijamaave work aagudha?")
st.dataframe(ev, hide_index=True, use_container_width=True)
best_net = ev["Net/signal"].replace("-", np.nan).dropna()
st.caption(
    f"Out-of-sample test: {n_total:,} stock-days ({len(res)} stocks x last 20% data). Majority baseline accuracy = {base_acc * 100:.2f}%. "
    f"'Gross/signal' = model 'Up' sonna naal-la next-day avg return. 'Net/signal' = gross - round-trip charges ({ref_cost:.2f}% for ₹10,000 position). "
    "p-value normal approximation; 50 stocks correlated-a irukkradhaala optimistic, 4 models test panradhaala p < 0.0125 dhaan significant (Bonferroni).")
try:
    nets = [float(v.replace("%", "")) for v in best_net]
    if nets and max(nets) <= 0:
        st.error("**Finding:** Edha model-um charges-ai cover panra alavukku net edge kaattala. "
                 "Direction accuracy konjam 50%-ku mela irundhaalum, 1-day trade-la cost-ku appuram profit illa.")
    elif nets:
        st.success("**Finding:** Sila model-ku net positive theriyudhu - aana out-of-sample, correlated stocks, survivorship bias ellam serthu paatha, adhu nijamaana edge-a nu confirm panna innum test venum.")
except Exception:
    pass

with st.expander("📊 Ella 50 companies - 4 model table"):
    full = pd.DataFrame([{
        "Company": r.Company, "Sector": r.Sector, "Last Close": f"₹{r.price:,.2f}",
        "Naive Up": pct(r.p_naive), "Logistic Up": pct(r.p_lr), "XGBoost Up": pct(r.p_xgb),
        "LSTM Up": pct(r.p_lstm), "Final Up": pct(r.prob), "Models Up": f"{r.votes_up}/{r.n_models}",
        "Holdout Acc": f"{r.acc * 100:.1f}%", "Baseline": f"{r.base * 100:.1f}%", "RSI": f"{r.rsi:.0f}",
    } for r in res.sort_values("prob", ascending=False).itertuples()])
    st.dataframe(full, hide_index=True, use_container_width=True)

with st.expander("📑 Method & Limitations (research note)"):
    st.markdown(f"""
- **Data:** yfinance, {YEARS} years daily (adjusted), 50 current Nifty stocks. *Survivorship bias:* ippo index-la irukkura stocks mattum - palaya-la veliya poyi irukkura stocks illa.
- **Features (11):** 1/5/10/20-day returns, MA5/20, MA20/50 ratio, price vs MA20, 10/20-day volatility, RSI(14), volume change.
- **Target:** naalaiku close > indru close.
- **Split:** time-ordered 80/20, train-ku test-ku naduvula {EMBARGO}-day embargo. Scaler train data-la mattum fit. Random shuffle illa.
- **Models:** Naive (momentum rule) | Logistic Regression | XGBoost (150 trees, depth 3) | LSTM (2 x 32, 60-day window, 6 epochs).
  Final signal = Logistic + XGBoost + LSTM probability average. LSTM test-ku train-la irundhu mattum train aagum, prediction-ku refit illa (time save).
- **XAI:** XGBoost SHAP (TreeSHAP) contributions - top 2 features indha prediction-a evlo push pannuchu.
- **Expected return:** P(Up) x historical avg Up-day move + P(Down) x avg Down-day move. Idhu model-oda estimate, real price target illa.
- **Charges:** delivery trade (STT, stamp, exchange, SEBI, GST, DP). Rates maarum - code top-la maathikkalaam.
- **Limitations:** single holdout period (regime-specific), 50 stocks correlated, transaction costs approximate, slippage / bid-ask illa, hyper-parameter tuning illa, intraday gap handle pannala.
""")

if errs:
    with st.expander(f"⚠️ {len(errs)} stocks skip aachu"):
        for e in errs:
            st.write(e)
st.warning("⚠️ Education / research ku mattum. Investment advice illa. Daily direction prediction romba kashtam - loss varalam.")
