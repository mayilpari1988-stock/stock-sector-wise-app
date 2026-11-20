# Nifty-XAI (Simple) - Education purpose only
# Run: streamlit run app.py
#
# requirements.txt:
#   streamlit
#   yfinance
#   pandas
#   numpy
#   scikit-learn
#   xgboost
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


def new_xgb():
    return xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='logloss', random_state=42)


def analyse(raw):
    f = make_features(raw)
    train = f.dropna(subset=['Target'])
    X, y = train[FEATURES].values, train['Target'].astype(int).values
    x_now = f[FEATURES].iloc[[-1]].values
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    # 1) Holdout check (accuracy vs baseline)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    xg = new_xgb().fit(Xtr, ytr)
    p_te = (lr.predict_proba(sc.transform(Xte))[:, 1] + xg.predict_proba(Xte)[:, 1]) / 2
    acc = float(((p_te > 0.5).astype(int) == yte).mean())
    base = float((yte == int(ytr.mean() >= 0.5)).mean())

    # 2) Final model - ella data-vilum train, naalaiku prediction
    sc2 = StandardScaler().fit(X)
    lr2 = LogisticRegression(max_iter=1000).fit(sc2.transform(X), y)
    xg2 = new_xgb().fit(X, y)
    prob = float((lr2.predict_proba(sc2.transform(x_now))[0][1] + xg2.predict_proba(x_now)[0][1]) / 2)

    # 3) XAI - XGBoost SHAP contributions (last column = bias)
    contrib = xg2.get_booster().predict(xgb.DMatrix(x_now), pred_contribs=True)[0][:-1]
    order = np.argsort(contrib) if prob < 0.5 else np.argsort(-contrib)
    reasons = []
    for i in order[:2]:
        sign = "+" if contrib[i] > 0 else "-"
        reasons.append(f"{describe(FEATURES[i], x_now[0][i])} ({sign})")

    # 4) Expected return = prob x avg up move + (1-prob) x avg down move
    nr = train['NextRet'].values
    avg_up = float(nr[nr > 0].mean()) if (nr > 0).any() else 0.0
    avg_dn = float(nr[nr <= 0].mean()) if (nr <= 0).any() else 0.0
    exp_ret = (prob * avg_up + (1 - prob) * avg_dn) * 100

    return {"prob": prob, "acc": acc, "base": base, "exp_ret": exp_ret,
            "price": float(raw['Close'].iloc[-1]), "date": str(raw.index[-1].date()),
            "why": ", ".join(reasons)}


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(years):
    data = load_all(years)
    rows, errors = [], []
    for sym, (name, sector) in STOCKS.items():
        if sym not in data:
            errors.append(f"{name}: data kidaikala")
            continue
        try:
            r = analyse(data[sym])
            r.update({"Company": name, "Sector": sector, "Symbol": sym})
            rows.append(r)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return pd.DataFrame(rows), errors


# ------------------------------------------------
# PORTFOLIO - amount-ai full-a use panna
# ------------------------------------------------
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
st.set_page_config(page_title="Nifty-XAI", page_icon="📈", layout="wide")
st.title("📈 Nifty-XAI")
st.caption("Nifty 50 stocks-ku naalaiku Up/Down predict panni, un amount-ku edha vaanganum nu sollum. Education project.")

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
c1, c2 = st.columns([2, 1])
amount = c1.number_input("💰 Amount (Rs)", min_value=1000, max_value=10_000_000,
                         value=10000, step=1000)
c2.write("")
c2.write("")
go = c2.button("🚀 Predict & Suggest", type="primary", use_container_width=True)
if go:
    st.session_state["budget"] = int(amount)

# ---- 3. BOTTOM: result ----
if "budget" in st.session_state:
    budget = st.session_state["budget"]
    with st.spinner("50 companies analyse aagudhu... (first time 1-2 min edukkum)"):
        try:
            res, errs = run_analysis(YEARS)
        except Exception as e:
            res, errs = pd.DataFrame(), [f"{type(e).__name__}: {e}"]

    if res.empty:
        st.error("Data / model run aagala. Internet check pannitu thirumba try pannunga.")
        for e in errs[:5]:
            st.write(e)
    else:
        pf = build_portfolio(res, budget)
        st.subheader(f"🛒 ₹{budget:,} ku - Entha Stock, Evlo Count")
        if not pf:
            st.info("Indha nerathula Up-nu sollura strong stock illa (illa un amount-ku vaanga mudiyala). "
                    "Cash-ah vechurunga - NO TRADE dhaan best.")
        else:
            table = pd.DataFrame([{
                "Company": r["Company"], "Sector": r["Sector"], "Qty": r["qty"],
                "Price": f"₹{r['price']:,.2f}", "Amount": f"₹{r['amount']:,.2f}",
                "Up %": f"{r['prob'] * 100:.1f}%",
                "XAI - Yen Up? (SHAP)": r["why"],
            } for r in pf])
            st.dataframe(table, hide_index=True, use_container_width=True)

            invested = sum(r["amount"] for r in pf)
            buy_ch = sum(charges(r["amount"], "buy") for r in pf)
            cash_left = budget - invested - buy_ch
            exp_sell = sum(r["amount"] * (1 + r["exp_ret"] / 100) for r in pf)
            sell_ch = sum(charges(r["amount"] * (1 + r["exp_ret"] / 100), "sell") for r in pf)
            net = exp_sell - invested - buy_ch - sell_ch

            m = st.columns(5)
            m[0].metric("Amount", f"₹{budget:,.2f}")
            m[1].metric("Invested", f"₹{invested:,.2f}")
            m[2].metric("Buy Charges", f"₹{buy_ch:,.2f}")
            m[3].metric("Cash Left", f"₹{cash_left:,.2f}")
            m[4].metric("Exp. Net P/L (1 day)", f"₹{net:,.2f}", f"{net / invested * 100:+.2f}%")

            sec_split = pd.DataFrame(pf).groupby("Sector")["amount"].sum().sort_values(ascending=False)
            st.caption("Sector split: " + " | ".join(f"{s} ₹{a:,.0f}" for s, a in sec_split.items()))
            roundtrip = buy_ch + sell_ch
            st.caption(f"Buy + Sell charges (DP ₹{DP_CHARGE} per stock serthu) ≈ ₹{roundtrip:,.2f} "
                       f"({roundtrip / invested * 100:.2f}% of invested). Naalaiku stock idha vida adhigam move aana dhaan profit.")
            if net <= 0:
                st.warning("Expected profit charges-ai cover pannala - 1 naal hold-la loss varra chance adhigam.")

        avg_acc, avg_base = res.acc.mean() * 100, res.base.mean() * 100
        st.caption(f"Model check (holdout, {len(res)} stocks avg): accuracy {avg_acc:.1f}% vs baseline {avg_base:.1f}% "
                   f"(edge {avg_acc - avg_base:+.1f}%). Data date: {res.date.max()}. Edge ~0 na prediction random-ku kitta.")
        st.warning("⚠️ Education ku mattum. Investment advice illa. Daily direction prediction romba kashtam - loss varalam.")
        if errs:
            with st.expander(f"⚠️ {len(errs)} stocks skip aachu"):
                for e in errs:
                    st.write(e)
else:
    st.info("👆 Amount enter panni, 'Predict & Suggest' click pannunga.")
