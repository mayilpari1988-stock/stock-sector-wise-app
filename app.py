# Nifty-XAI (Fixed Version) - Education purpose only
# Run: streamlit run app.py
#
# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   scikit-learn
#   xgboost
#   tensorflow-cpu   (optional - LSTM venumna mattum)
#
# FIXES vs old code:
#  1. Prediction ippo LATEST day features la irundhu (naalaiku) - dhaan. Munna yesterday features use aagudhu.
#  2. Scaler train data la mattum fit (LSTM leakage fix).
#  3. Fake "Expected Price" (prob*1.2%) remove. Ippo historical avg up/down move vachi expected return.
#  4. Buy + Sell rendu side charges. GST brokerage mela mattum. Other charges (STT etc.) % la.
#  5. Budget thaandi vaanga maatom (qty=0 na skip).
#  6. Accuracy ku baseline (majority class + naive) comparison, edge kaattum.
#  7. LSTM illa na NaN (0 illa). Errors maraikkama kaattum.
#  8. Data cache, 4-model table progress bar, LSTM optional.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Nifty-XAI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.ticker {background-color:#1a237e;color:white;padding:10px;border-radius:8px;font-weight:bold;font-size:14px;overflow-x:auto;white-space:nowrap;}
.ticker span {margin-right:28px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 1. STOCK LIST (Nifty 50 constituents maaralaam - NSE site la verify pannunga)
# ------------------------------------------------
nifty_50_sector = {
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
sectors = sorted(set(v[1] for v in nifty_50_sector.values()))

MODEL_NAMES = ["1. Naive", "2. Logistic Regression", "3. XGBoost", "4. LSTM (Deep Learning)"]

try:
    import tensorflow  # noqa: F401
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ------------------------------------------------
# 2. DATA + FEATURES
# ------------------------------------------------
feature_cols = ['Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
                'MA_Ratio_5_20', 'MA_Ratio_20_50', 'Price_vs_MA20',
                'Vol_10', 'Vol_20', 'RSI', 'Vol_Change']


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, years):
    df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_last_close(symbol):
    try:
        h = yf.Ticker(symbol).history(period="5d")
        if len(h) >= 2:
            curr, prev = float(h['Close'].iloc[-1]), float(h['Close'].iloc[-2])
            return curr, (curr - prev) / prev * 100
    except Exception:
        pass
    return 0.0, 0.0


def make_features(raw):
    """Latest row ah DROP pannala. Latest row = naalaiku prediction ku. Target adhukku NaN."""
    df = raw.copy()
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
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

    df['NextRet'] = df['Close'].pct_change().shift(-1)           # naalaiku return (last row NaN)
    df['Target'] = (df['NextRet'] > 0).astype(float).where(df['NextRet'].notna())

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)                          # Target NaN irundhaalum last row irukkum
    df['Vol_Change'] = df['Vol_Change'].clip(-5, 5)
    return df


# ------------------------------------------------
# 3. MODEL RUNNER (cached)
# ------------------------------------------------
def _run_lstm(Xtr, ytr, Xte, yte, X_latest_hist, n_feat, seq_len=60):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    import tensorflow as tf
    tf.random.set_seed(42)

    scaler = StandardScaler().fit(Xtr)                           # train la mattum fit
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    hist_s = scaler.transform(X_latest_hist)

    def seqs(data, labels, prev=None):
        if prev is not None:
            data = np.vstack([prev, data])
            labels = np.concatenate([np.zeros(len(prev)), labels])
        xs, ys = [], []
        for i in range(seq_len, len(data)):
            xs.append(data[i - seq_len:i]); ys.append(labels[i])
        return np.array(xs), np.array(ys)

    Xs_tr, ys_tr = seqs(Xtr_s, ytr)
    Xs_te, ys_te = seqs(Xte_s, yte, prev=Xtr_s[-seq_len:])       # test ku train kadaisi 60 days context
    model = Sequential([
        Input(shape=(seq_len, n_feat)),
        LSTM(32, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(Xs_tr, ys_tr, epochs=8, batch_size=32, verbose=0, validation_split=0.1)
    acc = float(((model.predict(Xs_te, verbose=0).flatten() > 0.5).astype(int) == ys_te).mean())
    prob = float(model.predict(hist_s[-seq_len:].reshape(1, seq_len, n_feat), verbose=0)[0][0])
    return prob, acc


@st.cache_data(ttl=3600, show_spinner=False)
def run_model(symbol, model_name, years):
    try:
        raw = load_data(symbol, years)
        if raw is None or len(raw) < 250:
            return {"error": "Data kammi / download aagala"}
        df = make_features(raw)
        train_df = df.dropna(subset=['Target'])
        latest = df.iloc[[-1]]                                    # NAALAIKU PREDICTION ROW
        X = train_df[feature_cols].values
        y = train_df['Target'].astype(int).values
        X_latest = latest[feature_cols].values
        split = int(len(X) * 0.8)
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

        # Baselines
        majority = int(ytr.mean() >= 0.5)
        base_acc = float((yte == majority).mean())
        naive_test_pred = (Xte[:, 0] > 0).astype(int)             # Return_1d > 0 => Up
        naive_acc = float((naive_test_pred == yte).mean())

        if model_name == MODEL_NAMES[0]:
            naive_train_pred = (Xtr[:, 0] > 0).astype(int)
            up_mask = naive_train_pred == 1
            p_up_given_up = ytr[up_mask].mean() if up_mask.any() else 0.5
            p_up_given_dn = ytr[~up_mask].mean() if (~up_mask).any() else 0.5
            prob = float(p_up_given_up if X_latest[0, 0] > 0 else p_up_given_dn)
            acc = naive_acc
        elif model_name == MODEL_NAMES[1]:
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
            prob = float(m.predict_proba(sc.transform(X_latest))[0][1])
            acc = float((m.predict(sc.transform(Xte)) == yte).mean())
        elif model_name == MODEL_NAMES[2]:
            m = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric='logloss', random_state=42)
            m.fit(Xtr, ytr)
            prob = float(m.predict_proba(X_latest)[0][1])
            acc = float((m.predict(Xte) == yte).mean())
        else:
            if not LSTM_AVAILABLE:
                return {"error": "TensorFlow install illa"}
            hist = np.vstack([Xtr, Xte])
            prob, acc = _run_lstm(Xtr, ytr, Xte, yte, hist, len(feature_cols))

        # Historical average move (train data) -> expected return (honest estimate)
        nr = train_df['NextRet'].values[:split]
        avg_up = float(nr[nr > 0].mean()) if (nr > 0).any() else 0.0
        avg_dn = float(nr[nr <= 0].mean()) if (nr <= 0).any() else 0.0
        exp_ret = prob * avg_up + (1 - prob) * avg_dn

        return {
            "price": float(raw['Close'].iloc[-1]),
            "date": str(raw.index[-1].date()),
            "prob": prob, "acc": acc,
            "base_acc": base_acc, "naive_acc": naive_acc,
            "exp_ret_pct": exp_ret * 100,
            "rsi": float(latest['RSI'].iloc[0]),
            "n_test": int(len(yte)),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------
# 4. SIDEBAR
# ------------------------------------------------
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Model Selection", MODEL_NAMES, index=2)
budget = st.sidebar.number_input("Amount (Rs)", min_value=1000, max_value=1_000_000, value=10000, step=1000)
brokerage = st.sidebar.number_input("Brokerage per Order (Rs)", min_value=0.0, max_value=500.0, value=20.0, step=5.0,
                                    help="Buy ku oru order, Sell ku oru order - rendu side um charge.")
other_pct = st.sidebar.number_input("Other charges % (STT, stamp, exchange)", min_value=0.0, max_value=1.0,
                                    value=0.12, step=0.01,
                                    help="Approx. Delivery trade la STT 0.1% buy+sell. Un broker contract note paathu maathunga.")
top_n = st.sidebar.slider("Max stocks in portfolio", 1, 10, 5)
only_edge = st.sidebar.checkbox("Baseline ah beat panna models mattum", value=False)
sector_choice = st.sidebar.selectbox("Sector Filter", ["All Sectors"] + sectors)
years = st.sidebar.slider("Data years", 3, 10, 5)
st.sidebar.caption("GST 18% brokerage mela mattum add aagum.")
if not LSTM_AVAILABLE:
    st.sidebar.warning("TensorFlow illa - LSTM work aagadhu.")

# ------------------------------------------------
# 5. HEADER / TICKER
# ------------------------------------------------
st.title("📈 Nifty-XAI - Education Project")
ticker_html = ""
for sym in ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "LT.NS", "TITAN.NS"]:
    p, ch = get_last_close(sym)
    ticker_html += f"<span>{'🟢' if ch >= 0 else '🔴'} {nifty_50_sector[sym][0]} ₹{p:.2f} ({ch:+.2f}%)</span>"
st.markdown(f'<div class="ticker">{ticker_html}</div>', unsafe_allow_html=True)
st.caption("Prices = last close (live illa). Prediction = next trading day Up/Down probability.")

filtered = {k: v for k, v in nifty_50_sector.items() if sector_choice == "All Sectors" or v[1] == sector_choice}

# ------------------------------------------------
# 6. PREDICTION
# ------------------------------------------------
st.subheader(f"📊 Sector Wise Prediction - {model_choice}")
if st.button("🚀 Run Prediction", type="primary"):
    rows, errors = [], []
    bar = st.progress(0, text="Running...")
    for i, (sym, (name, sec)) in enumerate(filtered.items()):
        r = run_model(sym, model_choice, years)
        if "error" in r:
            errors.append((name, r["error"]))
        else:
            rows.append({
                "Company": name, "Sector": sec, "Last Close": r["price"], "Data Date": r["date"],
                "Prediction": "Up 🟢" if r["prob"] > 0.5 else "Down 🔴",
                "Up Probability %": r["prob"] * 100,
                "Test Accuracy %": r["acc"] * 100,
                "Baseline Acc %": max(r["base_acc"], r["naive_acc"]) * 100,
                "Edge %": (r["acc"] - max(r["base_acc"], r["naive_acc"])) * 100,
                "Expected Return %": r["exp_ret_pct"], "RSI": r["rsi"],
            })
        bar.progress((i + 1) / len(filtered), text=f"{name} done")
    bar.empty()
    st.session_state['res'] = pd.DataFrame(rows)
    st.session_state['errors'] = errors

if st.session_state.get('errors'):
    with st.expander(f"⚠️ {len(st.session_state['errors'])} company la error"):
        for n, e in st.session_state['errors']:
            st.write(f"**{n}**: {e}")

if 'res' in st.session_state and len(st.session_state['res']) > 0:
    df_res = st.session_state['res']

    st.subheader("Sector Wise Average")
    sec_avg = (df_res.groupby("Sector")
               .agg(Avg_Up_Prob=("Up Probability %", "mean"), Avg_Edge=("Edge %", "mean"), Companies=("Company", "count"))
               .sort_values("Avg_Up_Prob", ascending=False))
    st.dataframe(sec_avg.style.format({"Avg_Up_Prob": "{:.1f}%", "Avg_Edge": "{:+.1f}%", "Companies": "{:.0f}"}))
    st.bar_chart(sec_avg["Avg_Up_Prob"])
    st.caption("Edge % = model accuracy - best baseline accuracy. Edge ~0 or negative na model-ku value illa.")

    st.subheader("Company List")
    st.dataframe(df_res.sort_values("Up Probability %", ascending=False).style.format({
        "Last Close": "₹{:.2f}", "Up Probability %": "{:.1f}%", "Test Accuracy %": "{:.1f}%",
        "Baseline Acc %": "{:.1f}%", "Edge %": "{:+.1f}%", "Expected Return %": "{:+.3f}%", "RSI": "{:.1f}"}),
        use_container_width=True)

    # ------------------------------------------------
    # 7. PORTFOLIO SIMULATION
    # ------------------------------------------------
    st.divider()
    st.subheader(f"💰 Rs {budget:,} Portfolio Simulation (1-day hold)")
    cand = df_res[(df_res["Up Probability %"] > 50) & (df_res["Expected Return %"] > 0)]
    if only_edge:
        cand = cand[cand["Edge %"] > 0]
    cand = cand.sort_values("Up Probability %", ascending=False).head(top_n)

    def charges(value):
        b = brokerage
        return b + b * 0.18 + value * other_pct / 100

    if cand.empty:
        st.info("Qualify aagura company illa. Cash ah vechurunga.")
    else:
        per_stock = budget / len(cand)
        rows = []
        for _, r in cand.iterrows():
            qty = int(per_stock // r["Last Close"])
            if qty == 0:
                continue                                           # budget thaandi vaanga maatom
            buy_val = qty * r["Last Close"]
            exp_sell_val = buy_val * (1 + r["Expected Return %"] / 100)
            rows.append({"Company": r["Company"], "Qty": qty, "Buy Price": r["Last Close"],
                         "Invested": buy_val, "Exp. Sell Value": exp_sell_val,
                         "Buy Charges": charges(buy_val), "Sell Charges": charges(exp_sell_val),
                         "Gross P/L": exp_sell_val - buy_val})
        if not rows:
            st.warning("Budget romba kammi - oru share kooda vaanga mudiyala.")
        else:
            pf = pd.DataFrame(rows)
            pf["Net P/L"] = pf["Gross P/L"] - pf["Buy Charges"] - pf["Sell Charges"]
            st.dataframe(pf.style.format({c: "₹{:.2f}" for c in pf.columns if c not in ["Company", "Qty"]}),
                         use_container_width=True)

            invested = pf["Invested"].sum()
            buy_ch, sell_ch = pf["Buy Charges"].sum(), pf["Sell Charges"].sum()
            cash_left = budget - invested - buy_ch
            gross, net = pf["Gross P/L"].sum(), pf["Net P/L"].sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invested", f"₹{invested:.2f}")
            c2.metric("Cash Left (after buy charges)", f"₹{cash_left:.2f}")
            c3.metric("Total Charges (Buy+Sell)", f"₹{buy_ch + sell_ch:.2f}")
            c4.metric("Expected Net P/L", f"₹{net:.2f}", f"{net / invested * 100:.2f}%")
            st.caption(f"Gross expected P/L ₹{gross:.2f}. Charges = {len(pf)} buy + {len(pf)} sell orders. "
                       "Expected return = historical avg up/down move × model probability. Idhu guarantee illa.")
            if net <= 0:
                st.warning("Expected profit charges ah cover panna mudiyala - real trading la loss chance adhigam.")
    st.warning("⚠️ Education ku mattum. Investment advice illa. Daily direction accuracy usually ~50-55% dhaan.")
else:
    st.info("👈 Settings maathi 'Run Prediction' click pannunga.")

# ------------------------------------------------
# 8. 4-MODEL COMPARISON TABLE
# ------------------------------------------------
st.divider()
st.subheader("📊 4 Model Comparison - Sector Wise")
include_lstm = st.checkbox("LSTM um serkkanum (romba slow)", value=False, disabled=not LSTM_AVAILABLE)
if st.button("📊 Show 4 Model Comparison Table"):
    names = MODEL_NAMES if include_lstm else MODEL_NAMES[:3]
    rows, errors = [], []
    bar = st.progress(0, text="Running...")
    for i, (sym, (name, sec)) in enumerate(filtered.items()):
        row = {"Company": name, "Sector": sec}
        ok = False
        for mn in names:
            r = run_model(sym, mn, years)
            short = mn.split(".")[0] + "." + mn.split(".")[1].split()[0]
            if "error" in r:
                row[f"{short} Up %"], row[f"{short} Acc %"] = np.nan, np.nan
                errors.append((name, mn, r["error"]))
            else:
                ok = True
                row["Last Close"] = r["price"]
                row[f"{short} Up %"], row[f"{short} Acc %"] = r["prob"] * 100, r["acc"] * 100
        if ok:
            rows.append(row)
        bar.progress((i + 1) / len(filtered), text=f"{name} done")
    bar.empty()
    st.session_state['cmp'] = pd.DataFrame(rows)
    st.session_state['cmp_err'] = errors

if 'cmp' in st.session_state and len(st.session_state['cmp']) > 0:
    cmp_df = st.session_state['cmp']
    st.dataframe(cmp_df.style.format(
        {c: "{:.1f}%" for c in cmp_df.columns if c.endswith("%")} | {"Last Close": "₹{:.2f}"}, na_rep="N/A"),
        use_container_width=True)
    acc_cols = [c for c in cmp_df.columns if c.endswith("Acc %")]
    cols = st.columns(len(acc_cols))
    for col, ac in zip(cols, acc_cols):
        col.metric(ac.replace(" %", " Avg"), f"{cmp_df[ac].mean():.2f}%")
    if st.session_state.get('cmp_err'):
        with st.expander("Errors"):
            for n, m, e in st.session_state['cmp_err']:
                st.write(f"**{n}** ({m}): {e}")
