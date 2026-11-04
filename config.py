# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PRICE_DIR = BASE_DIR / "data" / "prices"
PRICE_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-01-01"
TRANSACTION_COST = 0.0005
SEEDS = [42, 123, 7]

STOCK_TO_SECTOR = {
    "RELIANCE.NS": "Energy",
    "HDFCBANK.NS": "Bank",
    "ICICIBANK.NS": "Bank",
    "SBIN.NS": "Bank",
    "KOTAKBANK.NS": "Bank",
    "AXISBANK.NS": "Bank",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT",
    "TECHM.NS": "IT",
    "LT.NS": "Infra",
    "TITAN.NS": "Jewellery",
    "BHARTIARTL.NS": "Telecom",
    "ITC.NS": "FMCG",
    "HINDUNILVR.NS": "FMCG",
    "TATAMOTORS.NS": "Auto",
    "MARUTI.NS": "Auto",
    "M&M.NS": "Auto",
    "TATASTEEL.NS": "Metal",
    "JSWSTEEL.NS": "Metal",
    "SUNPHARMA.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "BEL.NS": "Capital Goods",
    "BAJFINANCE.NS": "Finance",
    "NTPC.NS": "Power",
    "POWERGRID.NS": "Power",
    "ONGC.NS": "Energy",
    "COALINDIA.NS": "Energy",
}