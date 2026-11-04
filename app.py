# Nifty-XAI - SIMPLE SIDEBAR VERSION
# Sidebar la mattum: Model Selection + Amount + Transaction Cost per Trade
# Models: 1. Naive 2. Logistic 3. XGBoost 4. LSTM
# Run: streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

st.set_page_config(page_title="Nifty-XAI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main {background-color: #f8f9fa;}
.ticker {background-color: #1a237e; color: white; padding: 10px; border-radius: 8px; font-weight: bold; font-size: 15px;}
.ticker span {margin-right: 30px;}
.portfolio-box {background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 2px solid #2e7d32;}
.charges-box {background-color: #fff3e0; padding: 20px; border-radius: 10px; border: 2px solid #ef6c00;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 1. NIFTY 50 + SECTOR
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
sectors = sorted(list(set(v[1] for v in nifty_50_sector.values())))

# ------------------------------------------------
# 2. FEATURES - UN CODE ADHE
# ------------------------------------------------

def make_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_Ratio_5_20'] = df['MA_5'] / df['MA_20']
    df['MA_Ratio_20_50'] = df['MA_20'] / df['MA_50']
    df['Price_vs_MA20'] = df['Close'] / df['MA_20'] - 1
    df['Vol_10'] = df['Return_1d'].rolling(10).std()
    df['Vol_20'] = df['Return_1d'].rolling(20).std()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].clip(0, 100)
    df['Vol_Change'] = df['Volume'].pct_change(1)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.iloc[:-1]
    return df

feature_cols = ['Return_1d','Return_5d','Return_10d','Return_20d','MA_Ratio_5_20','MA_Ratio_20_50','Price_vs_MA20','Vol_10','Vol_20','RSI','Vol_Change']

@st.cache_data(ttl=300)
def get_live_price(symbol):
    try:
        h = yf.Ticker(symbol).history(period="2d")
        if len(h) >= 2:
            curr = float(h['Close'].iloc[-1]); prev = float(h['Close'].iloc[-2])
            return curr, (curr-prev)/prev*100
        return 0, 0
    except:
        return 0, 0

def predict_one_company(symbol, model_name, years=5):
    try:
        df_raw = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=True)
        if df_raw.empty or len(df_raw) < 200:
            return None
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df = make_features(df_raw)
        X = df[feature_cols].values; y = df['Target'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0); X = np.clip(X, -1e6, 1e6)
        split = int(len(df)*0.8)
        X_train, X_test = X[:split], X[split:]; y_train, y_test = y[:split], y[split:]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
        last_price = float(df_raw['Close'].iloc[-1])
        last_row = X[-1:].reshape(1, -1); last_row_s = scaler.transform(last_row)
        prob_up = 0.5; acc = 0
        if model_name == "1. Naive":
            prob_up = 0.55 if df['Return_1d'].iloc[-1] > 0 else 0.45
            pred = (df['Return_1d'].values[split:] > 0).astype(int); acc = accuracy_score(y_test, pred)
        elif model_name == "2. Logistic Regression":
            m = LogisticRegression(max_iter=1000); m.fit(X_train_s, y_train)
            prob_up = float(m.predict_proba(last_row_s)[0][1]); acc = accuracy_score(y_test, m.predict(X_test_s))
        elif model_name == "3. XGBoost":
            m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
            m.fit(X_train, y_train); prob_up = float(m.predict_proba(last_row)[0][1]); acc = accuracy_score(y_test, m.predict(X_test))
        elif model_name == "4. LSTM (Deep Learning)":
            if not LSTM_AVAILABLE: return None
            SEQ_LEN = 60
            X_all = scaler.fit_transform(X); X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
            def create_seq(data, labels):
                Xs, ys = [], []
                for i in range(SEQ_LEN, len(data)):
                    Xs.append(data[i-SEQ_LEN:i]); ys.append(labels[i])
                return np.array(Xs), np.array(ys)
            X_seq, y_seq = create_seq(X_all, y); s2 = int(len(X_seq)*0.8)
            Xtr, Xte = X_seq[:s2], X_seq[s2:]; ytr, yte = y_seq[:s2], y_seq[s2:]
            lm = Sequential([LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, len(feature_cols))), Dropout(0.2), LSTM(50, return_sequences=False), Dropout(0.2), Dense(25, activation='relu'), Dense(1, activation='sigmoid')])
            lm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            lm.fit(Xtr, ytr, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
            last_seq = X_all[-SEQ_LEN:].reshape(1, SEQ_LEN, len(feature_cols))
            prob_up = float(lm.predict(last_seq, verbose=0)[0][0])
            pred = (lm.predict(Xte, verbose=0) > 0.5).astype(int).flatten(); acc = accuracy_score(yte, pred)
        pred_label = "Up 🟢" if prob_up > 0.5 else "Down 🔴"
        exp_move_pct = (prob_up - 0.5) * 2 * 1.2
        exp_price = last_price * (1 + exp_move_pct/100)
        return {"price": last_price, "prob": prob_up, "pred": pred_label, "acc": acc, "exp_price": exp_price, "exp_move": exp_move_pct, "rsi": float(df['RSI'].iloc[-1])}
    except:
        return None

# ------------------------------------------------
# 3. SIDEBAR - NE KETTA 3 MATTUM DHAN
# ------------------------------------------------

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Idhula mattum maathunga, full website maariyidum.")

model_choice = st.sidebar.selectbox(
    "Model Selection",
    ["1. Naive", "2. Logistic Regression", "3. XGBoost", "4. LSTM (Deep Learning)"],
    index=2
)

budget = st.sidebar.number_input(
    "Amount Enter (Rs)",
    min_value=1000, max_value=1000000, value=10000, step=1000,
    help="Ungaloda total budget. Eg: 10000"
)

trans_cost_per_trade = st.sidebar.number_input(
    "Transaction Cost per Trade (Rs)",
    min_value=0.0, max_value=500.0, value=20.0, step=5.0,
    help="Oru trade ku Brokerage + GST + STT serthu evalo? Eg: 20 Rs"
)

sector_choice = st.sidebar.selectbox("Sector Filter", ["All Sectors"] + sectors)

st.sidebar.divider()
st.sidebar.caption("Model: Naive = Baseline, Logistic/XGBoost = ML, LSTM = Deep Learning")
st.sidebar.caption("Cost: Ne potta cost ah vechu dhan Net Profit calculate aagum.")

# ------------------------------------------------
# 4. MAIN PAGE - TOP TICKER
# ------------------------------------------------

st.title("📈 Nifty-XAI - Research Level")
ticker_text = ""
for sym in ["RELIANCE.NS","HDFCBANK.NS","INFY.NS","TCS.NS","ITC.NS","SBIN.NS","ICICIBANK.NS","LT.NS","TITAN.NS","BHARTIARTL.NS"]:
    p, ch = get_live_price(sym)
    name = nifty_50_sector[sym][0]
    emoji = "🟢" if ch >= 0 else "🔴"
    ticker_text += f"<span>{emoji} {name} ₹{p:.2f} ({ch:+.2f}%)</span>"
st.markdown(f'<div class="ticker"><marquee behavior="scroll" scrollamount="6">{ticker_text}</marquee></div>', unsafe_allow_html=True)
st.caption(f"Selected: Model = {model_choice} | Budget = ₹{budget} | Cost per Trade = ₹{trans_cost_per_trade} | Sector = {sector_choice}")

# ------------------------------------------------
# 5. SECTOR WISE PREDICTION - UN CODE ADHE, ONNUM MAATHALA
# ------------------------------------------------

st.subheader(f"📊 Sector Wise Prediction - {model_choice}")
if st.button("🚀 Run Prediction", type="primary"):
    with st.spinner(f"{model_choice} la predict pannudhu..."):
        results = []
        progress = st.progress(0)
        filtered = {k: v for k, v in nifty_50_sector.items() if sector_choice == "All Sectors" or v[1] == sector_choice}
        for i, (sym, (name, sec)) in enumerate(filtered.items()):
            r = predict_one_company(sym, model_choice, years=5)
            if r:
                results.append({"Company": name, "Sector": sec, "Live Price": r["price"], "Prediction": r["pred"], "Up Probability %": r["prob"]*100, "Accuracy %": r["acc"]*100, "Expected Price Tomorrow": r["exp_price"], "Expected Move %": r["exp_move"], "RSI": r["rsi"]})
            progress.progress((i+1)/len(filtered))
        st.session_state['res'] = pd.DataFrame(results)
    st.success("Mudinjiduchu!")

if 'res' in st.session_state and len(st.session_state['res']) > 0:
    df_res = st.session_state['res']
    st.subheader("Sector Wise Average")
    sector_avg = df_res.groupby("Sector").agg(Avg_Up_Prob=("Up Probability %", "mean"), Companies=("Company", "count")).sort_values("Avg_Up_Prob", ascending=False)
    st.dataframe(sector_avg.style.format("{:.2f}"))
    st.bar_chart(sector_avg["Avg_Up_Prob"])
    st.subheader("Company List with Prediction Value")
    st.dataframe(df_res.sort_values("Up Probability %", ascending=False).style.format({"Live Price": "₹{:.2f}", "Up Probability %": "{:.1f}%", "Accuracy %": "{:.1f}%", "Expected Price Tomorrow": "₹{:.2f}", "Expected Move %": "{:+.2f}%", "RSI": "{:.1f}"}))

    st.divider()
    st.subheader(f"💰 Rs {budget} Portfolio - Real Trading")
    up_df = df_res[df_res["Prediction"].str.contains("Up")].sort_values("Up Probability %", ascending=False).head(5)
    if len(up_df) == 0:
        st.error("Up nu model sollura company illa. Cash ah kaiyila vechurunga.")
    else:
        st.markdown('<div class="portfolio-box">', unsafe_allow_html=True)
        st.write(f"Top {len(up_df)} Up Stocks ah eduthu equal ah divide pannen.")
        per_stock_budget = budget / len(up_df)
        rows = []; total_invested = 0; total_expected = 0
        for _, row in up_df.iterrows():
            price = row["Live Price"]
            qty = int(per_stock_budget // price)
            if qty == 0: qty = 1
            cost = qty * price
            exp_val = qty * row["Expected Price Tomorrow"]
            profit = exp_val - cost
            rows.append({"Company": row["Company"], "Sector": row["Sector"], "Live Price": price, "Qty": qty, "Invested Rs": cost, "Expected Price Tomorrow": row["Expected Price Tomorrow"], "Expected Value": exp_val, "Expected Profit Rs": profit})
            total_invested += cost; total_expected += exp_val
        port_df = pd.DataFrame(rows)
        st.dataframe(port_df.style.format({"Live Price": "₹{:.2f}", "Invested Rs": "₹{:.2f}", "Expected Price Tomorrow": "₹{:.2f}", "Expected Value": "₹{:.2f}", "Expected Profit Rs": "₹{:.2f}"}))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="charges-box">', unsafe_allow_html=True)
        st.subheader("🧾 Transaction Cost + GST + Amount in Hand")
        num_trades = len(port_df)
        total_trans_cost = trans_cost_per_trade * num_trades
        gst_part = total_trans_cost * 18 / 118
        brokerage_part = total_trans_cost - gst_part
        amount_in_hand = budget - total_invested - total_trans_cost
        gross_profit = total_expected - total_invested
        net_profit = gross_profit - total_trans_cost
        net_pct = (net_profit / total_invested * 100) if total_invested > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Invested", f"₹{total_invested:.2f}")
        c1.metric("No. of Trades", f"{num_trades}")
        c2.metric(f"Transaction Cost ({num_trades} x ₹{trans_cost_per_trade})", f"₹{total_trans_cost:.2f}")
        c2.metric("Adhula GST (18%) Approx", f"₹{gst_part:.2f}")
        c3.metric("Brokerage + Others", f"₹{brokerage_part:.2f}")
        c3.metric("Amount in Hand (Cash Left)", f"₹{amount_in_hand:.2f}")
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Budget", f"₹{budget:.2f}")
        col2.metric("💸 Amount in Hand", f"₹{amount_in_hand:.2f}")
        col3.metric("📈 Gross Profit", f"₹{gross_profit:.2f}")
        col4.metric("🏆 Net Profit (After Cost)", f"₹{net_profit:.2f}", f"{net_pct:.2f}%")
        st.caption(f"Formula: Net Profit = Expected Value - Invested - (Transaction Cost per Trade ₹{trans_cost_per_trade} x {num_trades} Trades). GST indha cost kulla included.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.warning("⚠️ Education ku mattum. Model 100% correct illa, loss um varalam.")
else:
    st.info("👈 Left sidebar la Model, Amount, Cost ah maathi, mela Run Prediction ah click pannu.")

# ================================================================
# EXTRA ADD ONLY - KEELA MATTUM - UN CODE AH MODIFY PANNALA
# 4 MODEL ODA PREDICTION AH TABLE FORMAT LA KAAMIKKA
# ================================================================

st.divider()
st.subheader("📊 Extra Table - 4 Model Comparison - Sector Wise")
st.caption(f"Sector: {sector_choice} - Keela button ah click panna, andha sector oda yalla company ku 4 model oda prediction um table la varum. Mela irukura code ah onnum thodala.")

# Extra function - 4 model um orae vaati run pannum - Un predict_one_company ah reuse pannudhu, modify pannala
@st.cache_data(ttl=3600, show_spinner=False)
def get_4model_table(sector_filter):
    filtered_extra = {k: v for k, v in nifty_50_sector.items() if sector_filter == "All Sectors" or v[1] == sector_filter}
    rows_extra = []
    for sym_extra, (name_extra, sec_extra) in filtered_extra.items():
        # Un function ah 4 vaati call pannurom - 4 model ku
        r1 = predict_one_company(sym_extra, "1. Naive", years=2)
        r2 = predict_one_company(sym_extra, "2. Logistic Regression", years=2)
        r3 = predict_one_company(sym_extra, "3. XGBoost", years=2)
        # LSTM slow na, idha mattum konjam fast ah - years=2, epochs already 10 dhan
        r4 = predict_one_company(sym_extra, "4. LSTM (Deep Learning)", years=2)
        if r1 and r2 and r3:
            price_extra = r1["price"]
            # LSTM kedaikala na 0 nu podurom
            lstm_prob_extra = r4["prob"]*100 if r4 else 0
            lstm_acc_extra = r4["acc"]*100 if r4 else 0
            rows_extra.append({
                "Company": name_extra,
                "Sector": sec_extra,
                "Live Price": price_extra,
                "1. Naive - Up %": r1["prob"]*100,
                "2. Logistic - Up %": r2["prob"]*100,
                "3. XGBoost - Up %": r3["prob"]*100,
                "4. LSTM - Up %": lstm_prob_extra,
                "1. Naive Acc %": r1["acc"]*100,
                "2. Logistic Acc %": r2["acc"]*100,
                "3. XGBoost Acc %": r3["acc"]*100,
                "4. LSTM Acc %": lstm_acc_extra,
                "RSI": r1["rsi"]
            })
    return pd.DataFrame(rows_extra)

if st.button("📊 Show 4 Model Comparison Table", type="secondary"):
    with st.spinner(f"{sector_choice} sector ku 4 model um run aagudhu... LSTM na konjam time edukkum"):
        df_4model = get_4model_table(sector_choice)
        st.session_state['df_4model_extra'] = df_4model
    st.success(f"{sector_choice} - {len(df_4model)} company - 4 Model Table Ready!")

if 'df_4model_extra' in st.session_state and len(st.session_state['df_4model_extra']) > 0:
    df_show_extra = st.session_state['df_4model_extra']
    st.write(f"**Showing: {sector_choice} - Total {len(df_show_extra)} Companies - 4 Model Prediction Table**")
    st.dataframe(
        df_show_extra.sort_values("3. XGBoost - Up %", ascending=False).style.format({
            "Live Price": "₹{:.2f}",
            "1. Naive - Up %": "{:.1f}%",
            "2. Logistic - Up %": "{:.1f}%",
            "3. XGBoost - Up %": "{:.1f}%",
            "4. LSTM - Up %": "{:.1f}%",
            "1. Naive Acc %": "{:.1f}%",
            "2. Logistic Acc %": "{:.1f}%",
            "3. XGBoost Acc %": "{:.1f}%",
            "4. LSTM Acc %": "{:.1f}%",
            "RSI": "{:.1f}"
        }),
        use_container_width=True
    )
    # Chinna summary - Average Accuracy
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. Naive Avg Acc", f"{df_show_extra['1. Naive Acc %'].mean():.2f}%")
    c2.metric("2. Logistic Avg Acc", f"{df_show_extra['2. Logistic Acc %'].mean():.2f}%")
    c3.metric("3. XGBoost Avg Acc", f"{df_show_extra['3. XGBoost Acc %'].mean():.2f}%")
    c4.metric("4. LSTM Avg Acc", f"{df_show_extra['4. LSTM Acc %'].mean():.2f}%")
else:
    st.info("Keela irukura 'Show 4 Model Comparison Table' button ah click pannu. Sector ah Banking / IT nu maathi click panna, andha sector oda yalla company + 4 model prediction varum.")