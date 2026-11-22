# Final Hardened Portfolio Backtester & Live Trading System (v2 — Timeframe Sweep & OOS Ready)

Preserves the exact **Test 2** winning risk metrics (`1%` risk/trade, `3%` max concurrent risk, max `2` crypto positions, `20`-bar Donchian + `2.5x` ATR trail, `1.5%` weekly Nifty credit spread cap) and adds **multi-timeframe plateau testing (`1h`, `2h`, `4h`, `6h`, `12h`, `1d`)** plus **custom `--start` / `--end` out-of-sample date windows**.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Step 1: Timeframe Plateau Check (One Command)
Verifies that `4h` isn't a lucky spike by testing `2h`, `4h`, `6h`, `12h`, and `1d` side-by-side:
```bash
python run_timeframe_sweep.py --intervals 2h 4h 6h 12h 1d
```

## Step 2: Earlier Historical Out-of-Sample Checks (2021–2023 Bear/Chop Regime)

**A. Full Portfolio (ETH + SOL + Gold + Nifty Spreads) on Daily (`1d`) — 2021 to 2023 vs Full 5 Years:**
```bash
# 2021-2023 Out-of-Sample (includes 2022 crypto crash + Gold chop)
python run_hardened_backtest.py --interval 1d --start 2021-01-01 --end 2023-12-31

# Full 5+ Year Portfolio Run (2021-2026)
python run_hardened_backtest.py --interval 1d --start 2021-01-01
```

**B. Crypto Donchian (`ETH` + `SOL`) on `2h`, `4h`, `6h`, `12h`, `1d` — 2021 to 2023 Out-of-Sample:**
*(Uses Binance historical futures data, which goes back to 2020 with no 729-day intraday cap)*
```bash
python run_timeframe_sweep.py --crypto-only --intervals 2h 4h 6h 12h 1d --start 2021-01-01 --end 2023-12-31
```

## Step 3: Single Hardened Backtest (Recent 2 Years)
```bash
# Hardened Test 2 (4H, 1% risk, 200-bar warmup, gap-aware stops, Silver dropped)
python run_hardened_backtest.py --interval 4h --risk 0.01

# Strict Rs 2L Executable Version (Long-Only GOLDIETF.NS + Whole Lots)
python run_hardened_backtest.py --interval 4h --risk 0.01 --gold-etf --whole-lots
```

## Step 4: Live / Paper Execution
```bash
# Engine A: Delta Exchange India 4H Bot (Starts in DELTA_ENV=testnet, DRY_RUN=true)
python live_delta_donchian.py

# Engine B: Weekly Nifty Defined-Risk Credit Spread Advisor
python live_nifty_spread_advisor.py --capital 200000 --lot-size 75
```
