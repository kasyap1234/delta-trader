"""
Fetches and caches actual Upstox V3 NSE daily candles for:
  1. Expanded NSE Sector & Factor ETFs (10 additional ETFs on top of our 6 cached ETFs)
  2. 40 liquid Nifty 100 large-cap stocks across all major sectors (for Top-5 Risk-Adjusted Stock Momentum)
  3. Checks actual Binance USD-M historical Open Interest availability (/futures/data/openInterestHist)
"""

import json
import time
from pathlib import Path
import pandas as pd
import requests

from data_loader import load_upstox, resolve_upstox_key

ETF_CACHE = Path(__file__).resolve().parent / "output" / "_cache_upstox_metals"
STOCK_CACHE = Path(__file__).resolve().parent / "output" / "_cache_upstox_nifty100"
OI_CACHE = Path(__file__).resolve().parent / "output" / "_cache_binance_oi"

EXTRA_ETFS = [
    "PHARMABEES",
    "AUTOBEES",
    "INFRABEES",
    "PSUBNKBEES",
    "CPSEETF",
    "JUNIORBEES",
    "CONSUMBEES",
    "MOM100",
    "MOM30IETF",
    "ALPHAETF",
]

# 40 liquid Nifty 100 constituents spanning all major Indian sectors
# (explicitly noted as a fixed current-constituent sample -> survivorship-bias caveat disclosed)
NIFTY100_SAMPLE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "ITC",
    "LT", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "NTPC", "POWERGRID", "ONGC", "COALINDIA", "TATASTEEL", "HINDALCO", "JSWSTEEL", "M&M",
    "BAJAJ-AUTO", "EICHERMOT", "CIPLA", "DRREDDY", "DIVISLAB", "HCLTECH", "TECHM", "WIPRO",
    "BEL", "HAL", "TRENT", "ADANIPORTS", "ADANIENT", "SIEMENS", "ABB", "DLF",
]


def fetch_all():
    ETF_CACHE.mkdir(parents=True, exist_ok=True)
    STOCK_CACHE.mkdir(parents=True, exist_ok=True)
    OI_CACHE.mkdir(parents=True, exist_ok=True)

    print("=== 1. FETCHING EXTRA NSE SECTOR & FACTOR ETFs (Upstox V3 1d) ===")
    for sym in EXTRA_ETFS:
        out = ETF_CACHE / f"NSE_{sym}_1d.csv"
        if out.exists():
            df = pd.read_csv(out, index_col=0, parse_dates=True)
            print(f"CACHED | ETF   {sym:12s} | bars={len(df):4d} | {df.index[0]} -> {df.index[-1]}")
            continue
        try:
            key = resolve_upstox_key(sym, "NSE")
            df = load_upstox(sym, interval="1d", exchange="NSE", start_date="2021-11-24", end_date="2026-11-23", include_warmup=True)
            df.to_csv(out)
            print(f"OK     | ETF   {sym:12s} | key={key:26s} | bars={len(df):4d} | {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
        except Exception as exc:
            print(f"SKIP   | ETF   {sym:12s} | {exc}")

    print("\n=== 2. FETCHING 40 LIQUID NIFTY-100 STOCKS (Upstox V3 1d) ===")
    ok_stocks = 0
    for sym in NIFTY100_SAMPLE:
        out = STOCK_CACHE / f"NSE_{sym}_1d.csv"
        if out.exists():
            ok_stocks += 1
            continue
        try:
            df = load_upstox(sym, interval="1d", exchange="NSE", start_date="2021-11-24", end_date="2026-11-23", include_warmup=True)
            df.to_csv(out)
            ok_stocks += 1
            print(f"OK     | STOCK {sym:12s} | bars={len(df):4d} | {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
        except Exception as exc:
            print(f"SKIP   | STOCK {sym:12s} | {exc}")
    print(f"Total cached Nifty-100 stocks: {ok_stocks}/{len(NIFTY100_SAMPLE)}")

    print("\n=== 3. FETCHING BINANCE USD-M HISTORICAL OPEN INTEREST (6h) ===")
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]:
        try:
            r = requests.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": sym, "period": "6h", "limit": 500},
                timeout=15,
            )
            r.raise_for_status()
            rows = r.json()
            if rows:
                df = pd.DataFrame(rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
                df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
                df = df.set_index("timestamp").sort_index()
                df.to_csv(OI_CACHE / f"{sym}_oi_6h.csv")
                print(f"OK     | OI    {sym:10s} | bars={len(df):4d} | {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"EMPTY  | OI    {sym:10s}")
        except Exception as exc:
            print(f"FAIL   | OI    {sym:10s} | {exc}")
        time.sleep(0.15)


if __name__ == "__main__":
    fetch_all()
