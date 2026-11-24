"""
Fetches and caches actual Upstox V3 historical candles (1d 5Y and 6h 2Y) for
liquid NSE Equity/Sector ETFs: NIFTYBEES, MID150BEES, BANKBEES, ITBEES.
"""

from pathlib import Path
from data_loader import load_upstox, resolve_upstox_key

CACHE_DIR = Path(__file__).resolve().parent / "output" / "_cache_upstox_metals"


def fetch_and_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    syms = ["NIFTYBEES", "MID150BEES", "BANKBEES", "ITBEES"]
    jobs = []
    for sym in syms:
        jobs.append((sym, "NSE", "1d", "2021-11-24", "2026-11-23", True))
        jobs.append((sym, "NSE", "6h", "2024-11-24", "2026-11-23", True))

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
