"""
Fetches and caches actual Upstox V3 historical candles (1h -> 6h and 1d) for
Indian Gold & Silver instruments (MCX + NSE) for offline backtesting.

Instruments fetched:
  - MCX GOLDPETAL  (1g Gold Futures, active front-month)
  - MCX SILVERMIC  (1kg Silver Micro Futures, active front-month)
  - NSE GOLDBEES   (Spot Gold ETF, multi-year INR history)
  - NSE SILVERBEES (Spot Silver ETF, multi-year INR history)
  - NSE LIQUIDBEES (Liquid Cash ETF, for idle-cash collateral benchmark)
"""

from pathlib import Path
import pandas as pd
from data_loader import load_upstox, resolve_upstox_key

CACHE_DIR = Path(__file__).resolve().parent / "output" / "_cache_upstox_metals"


def fetch_and_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        # (symbol, exchange, interval, start_date, end_date, include_warmup)
        ("GOLDPETAL", "MCX", "6h", "2024-11-24", "2026-11-23", True),
        ("GOLDPETAL", "MCX", "1d", "2024-11-24", "2026-11-23", True),
        ("SILVERMIC", "MCX", "6h", "2024-11-24", "2026-11-23", True),
        ("SILVERMIC", "MCX", "1d", "2024-11-24", "2026-11-23", True),
        ("GOLDBEES", "NSE", "6h", "2024-11-24", "2026-11-23", True),
        ("GOLDBEES", "NSE", "1d", "2021-11-24", "2026-11-23", True),
        ("SILVERBEES", "NSE", "6h", "2024-11-24", "2026-11-23", True),
        ("SILVERBEES", "NSE", "1d", "2022-02-01", "2026-11-23", True),
        ("LIQUIDBEES", "NSE", "1d", "2021-11-24", "2026-11-23", False),
    ]

    for sym, exch, interval, start_d, end_d, warmup in jobs:
        try:
            key = resolve_upstox_key(sym, exch)
            df = load_upstox(
                sym,
                interval=interval,
                exchange=exch,
                start_date=start_d,
                end_date=end_d,
                include_warmup=warmup,
            )
            out = CACHE_DIR / f"{exch}_{sym}_{interval}.csv"
            df.to_csv(out)
            print(
                f"OK   | {exch:3s} {sym:10s} {interval:2s} | key={key:28s} | "
                f"bars={len(df):4d} | {df.index[0].strftime('%Y-%m-%d %H:%M')} -> {df.index[-1].strftime('%Y-%m-%d %H:%M')} | "
                f"last={df['close'].iloc[-1]:.2f}"
            )
        except Exception as exc:
            print(f"FAIL | {exch:3s} {sym:10s} {interval:2s} | {exc}")


if __name__ == "__main__":
    fetch_and_cache()
