# Nifty-XAI - 4 Models + XAI + Sector Portfolio + Full Nifty 50 Ticker
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
st.markdown("""<style>.ticker{background-color:#1a237e;color:white;padding:10px;border-radius:8px;font-weight:bold;font-size:14px;overflow:hidden;white-space:nowrap;}.ticker span{margin-right:28px;}</style>""", unsafe_allow_html=True)

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
    import tensorflow
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

feature_cols = ['Return_1d', 'Return_5d', 'Return_10d', 'Return_20d', 'MA_Ratio_5_20', 'MA_Ratio_20_50', 'Price_vs_MA20', 'Vol_10', 'Vol_20', 'RSI', 'Vol_Change']

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
    df = raw.copy()
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    ma5 = df['Close'].rolling(5).mean(); ma20 = df['Close'].rolling(20).mean(); ma50 = df['Close'].rolling(50).mean()
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
    df = df.dropna(subset=feature_cols)
    df['Vol_Change'] = df['Vol_Change'].clip(-5, 5)
    return df

def xai_reason(row):
    reasons = []
    if row['RSI'] > 70:
        reasons.append("RSI Overbought (>70) - Down risk")
    elif row['RSI'] < 30:
        reasons.append("RSI Oversold (<30) - Reversal chance Up")
    elif 45 <= row['RSI'] <= 65:
        reasons.append("RSI Neutral-Strong (45-65)")
    if row['Return_1d'] > 0:
        reasons.append("Yesterday Up - Momentum +")
    else:
        reasons.append("Yesterday Down - Momentum -")
    if row['Return_5d'] > 0.02:
        reasons.append("5-day trend Strong Up")
    elif row['Return_5d'] < -0.02:
        reasons.append("5-day trend Down")
    if row['MA_Ratio_5_20'] > 1:
        reasons.append("MA5 > MA20 - Short trend Up")
    else:
        reasons.append("MA5 < MA20 - Short trend Down")
    if row['Price_vs_MA20'] > 0:
        reasons.append("Price > MA20 - Above average")
    else:
        reasons.append("Price < MA20 - Below average")
    return " | ".join(reasons)

def xai_suggestion(prob, exp_ret, edge, rsi):
    if prob > 0.60 and exp_ret > 0.3 and edge > 0:
        return "✅ STRONG BUY - High confidence"
    if prob > 0.55 and exp_ret > 0:
        return "🟢 BUY - Moderate confidence"
    if prob > 0.50 and exp_ret > 0:
        return "🟡 WATCH / SMALL BUY"
    if prob < 0.45:
        return "🔴 AVOID / SELL - Down chance high"
    return "⚪ HOLD / NO TRADE - Cash hold"

def _run_lstm(Xtr, ytr, Xte, yte, X_latest_hist, n_feat, seq_len=60):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    import tensorflow as tf
    tf.random.set_seed(42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    hist_s = scaler.transform(X_latest_hist)
    def seqs(data, labels, prev=None):
        if prev is not None:
            data = np.vstack([prev, data]); labels = np.concatenate([np.zeros(len(prev)), labels])
        xs, ys = [], []
        for i in range(seq_len, len(data)):
            xs.append(data[i - seq_len:i]); ys.append(labels[i])
        return np.array(xs), np.array(ys)
    Xs_tr, ys_tr = seqs(Xtr_s, ytr)
    Xs_te, ys_te = seqs(Xte_s, yte, prev=Xtr_s[-seq_len:])
    model = Sequential([Input(shape=(seq_len, n_feat)), LSTM(32, return_sequences=True), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(Xs_tr, ys_tr, epochs=8, batch_size=32, verbose=0, validation_split=0.1)
    acc = float(((model.predict(Xs_te, verbose=0).flatten() > 0.5).astype(int) == ys_te).mean())
    prob = float(model.predict(hist_s[-seq_len:].reshape(1, seq_len, n_feat), verbose=0)[0][0])
    return prob, acc

@st.cache_data(ttl=3600, show_spinner=False)
def run_all_models(symbol, years, include_lstm):
    try:
        raw = load_data(symbol, years)
        if raw is None or len(raw) < 250:
            return {"error": "Data kammi"}
        df = make_features(raw)
        train_df = df.dropna(subset=['Target'])
        latest = df.iloc[[-1]]
        X = train_df[feature_cols].values; y = train_df['Target'].astype(int).values
        X_latest = latest[feature_cols].values
        split = int(len(X) * 0.8)
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
        majority = int(ytr.mean() >= 0.5)
        base_acc = float((yte == majority).mean())
        naive_acc = float((((Xte[:, 0] > 0).astype(int)) == yte).mean())
        out = {"price": float(raw['Close'].iloc[-1]), "date": str(raw.index[-1].date()), "base_acc": base_acc, "naive_acc": naive_acc, "rsi": float(latest['RSI'].iloc[0])}
        naive_train = (Xtr[:, 0] > 0).astype(int)
        up_mask = naive_train == 1
        p_up = ytr[up_mask].mean() if up_mask.any() else 0.5
        p_dn = ytr[~up_mask].mean() if (~up_mask).any() else 0.5
        out["Naive_Prob"] = float(p_up if X_latest[0, 0] > 0 else p_dn); out["Naive_Acc"] = naive_acc
        sc = StandardScaler().fit(Xtr)
        m1 = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        out["Logistic_Prob"] = float(m1.predict_proba(sc.transform(X_latest))[0][1]); out["Logistic_Acc"] = float((m1.predict(sc.transform(Xte)) == yte).mean())
        m2 = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
        m2.fit(Xtr, ytr)
        out["XGBoost_Prob"] = float(m2.predict_proba(X_latest)[0][1]); out["XGBoost_Acc"] = float((m2.predict(Xte) == yte).mean())
        imp = m2.feature_importances_
        top_idx = np.argsort(imp)[::-1][:3]
        out["Top_Features"] = ", ".join([f"{feature_cols[i]} ({imp[i]*100:.0f}%)" for i in top_idx])
        if include_lstm:
            if not LSTM_AVAILABLE:
                out["LSTM_Prob"] = np.nan; out["LSTM_Acc"] = np.nan
            else:
                hist = np.vstack([Xtr, Xte, X_latest])
                p, a = _run_lstm(Xtr, ytr, Xte, yte, hist, len(feature_cols))
                out["LSTM_Prob"] = p; out["LSTM_Acc"] = a
        else:
            out["LSTM_Prob"] = np.nan; out["LSTM_Acc"] = np.nan
        probs = [out["Naive_Prob"], out["Logistic_Prob"], out["XGBoost_Prob"]]
        if include_lstm and np.isfinite(out["LSTM_Prob"]):
            probs.append(out["LSTM_Prob"])
        out["Avg_Prob"] = float(np.mean(probs))
        out["Final_Prob"] = float(out["XGBoost_Prob"])
        nr = train_df['NextRet'].values[:split]
        avg_up = float(nr[nr > 0].mean()) if (nr > 0).any() else 0.0
        avg_dn = float(nr[nr <= 0].mean()) if (nr <= 0).any() else 0.0
        out["exp_ret_pct"] = float((out["Final_Prob"] * avg_up + (1 - out["Final_Prob"]) * avg_dn) * 100)
        for c in feature_cols:
            out[c] = float(latest[c].iloc[0])
        out["Return_1d"] = float(latest['Return_1d'].iloc[0]); out["Return_5d"] = float(latest['Return_5d'].iloc[0])
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# SIDEBAR
st.sidebar.title("⚙️ Settings")
budget = st.sidebar.number_input("Amount (Rs)", min_value=1000, max_value=1000000, value=10000, step=1000)
brokerage = st.sidebar.number_input("Brokerage per Order (Rs)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
stt_pct = st.sidebar.number_input("STT %", min_value=0.0, max_value=1.0, value=0.100, step=0.005, format="%.3f")
stamp_pct = st.sidebar.number_input("Stamp duty %", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.3f")
exch_pct = st.sidebar.number_input("Exchange %", min_value=0.0, max_value=1.0, value=0.003, step=0.0005, format="%.4f")
sebi_pct = st.sidebar.number_input("SEBI %", min_value=0.0, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")
dp_charge = st.sidebar.number_input("DP charge (Rs)", min_value=0.0, max_value=100.0, value=15.93, step=0.5)
top_n = st.sidebar.slider("Max stocks in portfolio", 1, 10, 5)
only_edge = st.sidebar.checkbox("Baseline ah beat panna mattum", value=False)
years = st.sidebar.slider("Data years", 3, 10, 5)
include_lstm = st.sidebar.checkbox("LSTM um serkkanum (slow)", value=False, disabled=not LSTM_AVAILABLE)
if LSTM_AVAILABLE:
    st.sidebar.success("✅ LSTM Ready")
else:
    st.sidebar.error("❌ LSTM NOT Ready - Reboot pannunga")

# HEADER - FULL NIFTY 50 SCROLLING TICKER
st.title("📈 Nifty-XAI - 4 Models + XAI")
ticker_html = ""
for sym, (nm, sec) in nifty_50_sector.items():
    p, ch = get_last_close(sym)
    color = "🟢" if ch >= 0 else "🔴"
    ticker_html += f"<span>{color} {nm} ₹{p:.2f} ({ch:+.2f}%) | </span>"
st.markdown(f'<div class="ticker"><marquee behavior="scroll" scrollamount="8" scrolldelay="20">{ticker_html}</marquee></div>', unsafe_allow_html=True)
st.caption("Live Values = Last Close + Yesterday-la irundhu % Change | 50 Companies-um scroll aagum | Top = 4 Model Compare | Middle = XAI | Bottom = Sector Buy List")

# TOP - 4 MODEL COMPARISON + PREDICTION
st.subheader("📊 TOP: 4 Model Comparison - Sector Wise Prediction")
if st.button("🚀 Run 4 Model Prediction", type="primary"):
    rows, errors = [], []
    bar = st.progress(0, text="Running...")
    for i, (sym, (name, sec)) in enumerate(nifty_50_sector.items()):
        r = run_all_models(sym, years, include_lstm)
        if "error" in r:
            errors.append((name, r["error"]))
        else:
            best_acc = max(r["Naive_Acc"], r["Logistic_Acc"], r["XGBoost_Acc"], r["LSTM_Acc"] if np.isfinite(r["LSTM_Acc"]) else 0)
            base = max(r["base_acc"], r["naive_acc"])
            tmp = {"Company": name, "Sector": sec, "Symbol": sym, "Last Close": r["price"], "Data Date": r["date"],
                   "Naive Up %": r["Naive_Prob"]*100, "Naive Acc %": r["Naive_Acc"]*100,
                   "Logistic Up %": r["Logistic_Prob"]*100, "Logistic Acc %": r["Logistic_Acc"]*100,
                   "XGBoost Up %": r["XGBoost_Prob"]*100, "XGBoost Acc %": r["XGBoost_Acc"]*100,
                   "LSTM Up %": r["LSTM_Prob"]*100 if np.isfinite(r["LSTM_Prob"]) else np.nan,
                   "LSTM Acc %": r["LSTM_Acc"]*100 if np.isfinite(r["LSTM_Acc"]) else np.nan,
                   "Avg Up %": r["Avg_Prob"]*100, "Final Up %": r["Final_Prob"]*100,
                   "Best Acc %": best_acc*100, "Baseline Acc %": base*100, "Edge %": (best_acc-base)*100,
                   "Expected Return %": r["exp_ret_pct"], "RSI": r["rsi"],
                   "Return_1d": r["Return_1d"], "Return_5d": r["Return_5d"],
                   "MA_Ratio_5_20": r["MA_Ratio_5_20"], "Price_vs_MA20": r["Price_vs_MA20"],
                   "Top_Features": r["Top_Features"]}
            tmp["Prediction"] = "Up 🟢" if tmp["Final Up %"] > 50 else "Down 🔴"
            tmp["XAI Reason"] = xai_reason(tmp)
            tmp["XAI Suggestion"] = xai_suggestion(tmp["Final Up %"]/100, tmp["Expected Return %"], tmp["Edge %"], tmp["RSI"])
            rows.append(tmp)
        bar.progress((i+1)/len(nifty_50_sector), text=f"{name} done")
    bar.empty()
    st.session_state['res'] = pd.DataFrame(rows)
    st.session_state['errors'] = errors

if st.session_state.get('errors'):
    with st.expander(f"⚠️ {len(st.session_state['errors'])} error"):
        for n, e in st.session_state['errors']:
            st.write(f"**{n}**: {e}")

if 'res' in st.session_state and len(st.session_state['res']) > 0:
    df_res = st.session_state['res']
    st.subheader("🏆 Sector Wise Average - 4 Models")
    sec_avg = df_res.groupby("Sector").agg(Naive=("Naive Up %", "mean"), Logistic=("Logistic Up %", "mean"), XGBoost=("XGBoost Up %", "mean"), LSTM=("LSTM Up %", "mean"), Avg=("Avg Up %", "mean"), BestAcc=("Best Acc %", "mean"), Count=("Company", "count")).sort_values("Avg", ascending=False)
    st.dataframe(sec_avg.style.format({c: "{:.1f}%" for c in sec_avg.columns if c != "Count"}), use_container_width=True)
    st.bar_chart(sec_avg["Avg"])
    st.subheader("📋 Full Company List - 4 Model Comparison")
    show_cols = ["Company", "Sector", "Last Close", "Prediction", "Naive Up %", "Logistic Up %", "XGBoost Up %", "LSTM Up %", "Avg Up %", "XGBoost Acc %", "Edge %", "Expected Return %", "RSI"]
    st.dataframe(df_res.sort_values("Avg Up %", ascending=False)[show_cols].style.format({"Last Close": "₹{:.2f}", "Naive Up %": "{:.1f}%", "Logistic Up %": "{:.1f}%", "XGBoost Up %": "{:.1f}%", "LSTM Up %": "{:.1f}%", "Avg Up %": "{:.1f}%", "XGBoost Acc %": "{:.1f}%", "Edge %": "{:+.1f}%", "Expected Return %": "{:+.3f}%", "RSI": "{:.1f}"}), use_container_width=True)
    st.divider()
    st.subheader("🧠 XAI - Yen Up / Yen Down? + Suggestion")
    st.write("XGBoost Top Features + RSI / Trend / MA la irundhu reason generate aaguthu.")
    xai_cols = ["Company", "Sector", "Prediction", "Final Up %", "Expected Return %", "Top_Features", "XAI Reason", "XAI Suggestion"]
    st.dataframe(df_res.sort_values("Final Up %", ascending=False)[xai_cols].style.format({"Final Up %": "{:.1f}%", "Expected Return %": "{:+.3f}%"}), use_container_width=True)
    sel_comp = st.selectbox("Oru company-a select panni XAI detail paakalam", df_res["Company"].tolist())
    r1 = df_res[df_res["Company"] == sel_comp].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Up %", f"{r1['Final Up %']:.1f}%")
    c2.metric("Expected Return", f"{r1['Expected Return %']:+.3f}%")
    c3.metric("RSI", f"{r1['RSI']:.1f}")
    c4.metric("Suggestion", r1["XAI Suggestion"])
    st.info(f"**{sel_comp} - XAI Reason:** {r1['XAI Reason']}")
    st.info(f"**Top Features (XGBoost):** {r1['Top_Features']}")
    st.divider()
    st.subheader("💰 BOTTOM: Sector Choose Panni Buy List")
    bottom_sector = st.selectbox("Sector Choose Pannunga", ["All Sectors"] + sectors, key="bottom_sector")
    filtered_bottom = df_res if bottom_sector == "All Sectors" else df_res[df_res["Sector"] == bottom_sector]
    st.write(f"**{bottom_sector}** - Total Companies: {len(filtered_bottom)}")
    st.dataframe(filtered_bottom.sort_values("Final Up %", ascending=False)[["Company", "Sector", "Last Close", "Prediction", "Final Up %", "Expected Return %", "RSI", "XAI Suggestion"]].style.format({"Last Close": "₹{:.2f}", "Final Up %": "{:.1f}%", "Expected Return %": "{:+.3f}%", "RSI": "{:.1f}"}), use_container_width=True)
    st.subheader(f"🛒 {bottom_sector} - Entha Stock Vaanganum? Evlo Qty?")
    cand = filtered_bottom[(filtered_bottom["Final Up %"] > 50) & (filtered_bottom["Expected Return %"] > 0) & (filtered_bottom["Last Close"].notna()) & (filtered_bottom["Last Close"] > 0)]
    if only_edge:
        cand = cand[cand["Edge %"] > 0]
    cand = cand.sort_values("Final Up %", ascending=False).head(top_n)
    def charges(value, side):
        exch = value * (exch_pct + sebi_pct) / 100
        return {"Brokerage": brokerage, "STT": value * stt_pct / 100, "Stamp Duty": value * stamp_pct / 100 if side == "buy" else 0.0, "Exchange+SEBI": exch, "GST": 0.18 * (brokerage + exch), "DP Charge": dp_charge if side == "sell" else 0.0}
    if cand.empty:
        st.info("⚠️ Indha sector-la qualify aagura company illa. Cash ah vechurunga - NO TRADE thaan best decision.")
    else:
        per_stock = budget / len(cand)
        rows2, parts = [], []
        for _, r in cand.iterrows():
            price = float(r["Last Close"])
            if not np.isfinite(price) or price <= 0:
                continue
            qty = int(per_stock // price)
            if qty == 0:
                continue
            buy_val = qty * price
            exp_sell_val = buy_val * (1 + r["Expected Return %"] / 100)
            bc, sc = charges(buy_val, "buy"), charges(exp_sell_val, "sell")
            parts.append((bc, sc))
            rows2.append({"Company": r["Company"], "Sector": r["Sector"], "Buy Price": price, "Qty (No of Count)": qty, "Invested": buy_val, "Exp. Sell Value": exp_sell_val, "Buy Charges": sum(bc.values()), "Sell Charges": sum(sc.values()), "Gross P/L": exp_sell_val - buy_val, "XAI Suggestion": r["XAI Suggestion"]})
        pf = pd.DataFrame(rows2)
        pf["Net P/L"] = pf["Gross P/L"] - pf["Buy Charges"] - pf["Sell Charges"]
        st.dataframe(pf.style.format({c: "₹{:.2f}" for c in pf.columns if c not in ["Company", "Sector", "Qty (No of Count)", "XAI Suggestion"]}), use_container_width=True)
        invested = pf["Invested"].sum(); buy_ch = pf["Buy Charges"].sum(); sell_ch = pf["Sell Charges"].sum(); cash_left = budget - invested - buy_ch; net = pf["Net P/L"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invested", f"₹{invested:.2f}")
        c2.metric("Cash Left", f"₹{cash_left:.2f}")
        c3.metric("Total Charges", f"₹{buy_ch + sell_ch:.2f}")
        c4.metric("Expected Net P/L", f"₹{net:.2f}", f"{net/invested*100:.2f}%" if invested > 0 else "0%")
        with st.expander("🧾 Charges Breakdown"):
            bd_dict = {}
            for bc, sc in parts:
                for k in bc:
                    bd_dict[k] = bd_dict.get(k, 0.0) + bc[k] + sc[k]
            bd = pd.DataFrame({"Charge": list(bd_dict.keys()), "Rs": list(bd_dict.values())})
            st.dataframe(bd.style.format({"Rs": "₹{:.2f}"}), hide_index=True)
    st.warning("⚠️ Education ku mattum. Investment advice illa.")
else:
    st.info("👈 'Run 4 Model Prediction' click pannunga.")
st.divider()
st.subheader("🧠 LSTM - Main Model Info")
st.write("Input (60, 11) -> LSTM 32 + Dropout 0.2 -> LSTM 32 + Dropout 0.2 -> Dense 16 -> Sigmoid | Epochs 8, Batch 32, Adam")
if LSTM_AVAILABLE:
    st.success("✅ LSTM Ready")
else:
    st.error("❌ TensorFlow illa - Reboot pannunga")
