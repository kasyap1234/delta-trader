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

# Whole-lot Donchian simulation with long-only GOLDIETF.NS
python run_hardened_backtest.py --interval 4h --risk 0.01 --gold-etf --whole-lots
```

Warmup candles feed indicators only; they do not extend `--days`. Whole-lot runs exclude the modeled Nifty sleeve because historical listed lot sizes and executable option quotes are not available. Other backtests still model that sleeve; they are not execution or live risk validation.

## Step 4: Live / Paper Execution
```bash
# Engine A: one public data cycle (remove --once for scheduled paper trading)
DELTA_ENV=testnet DRY_RUN=true python live_delta_donchian.py --once

# Engine B: show a Nifty candidate and verify the listed lot size
python live_nifty_spread_advisor.py --capital 200000
```

**Production order submission is disabled.** Before enabling testnet orders, use a dedicated [Delta India demo account](https://demo.delta.exchange) with no existing positions or orders, set ETHUSD/SOLUSD to 5× isolated leverage, and turn off automatic margin top-up and DETO commission. Configure testnet credentials locally and keep `DRY_RUN=true` until explicitly ready to send sandbox orders. `DELTA_ENV=prod DRY_RUN=true` can read public production market data but does not trade.

Run only one bot per testnet account from a persistent working directory. Preserve `bot_state-testnet-orders.json` and its order IDs across restarts; paper state is stored separately. After an unresolved order response, missing protection, or unexpected account state, inspect the actual Delta orders, positions and fills before resuming. Do not delete the state file or retry an uncertain entry just to clear the error. An emergency close halts new entries even after reconciliation. Stop-market orders can fill beyond their trigger; margin and risk sizing are estimates, not guarantees.

The Nifty advisor never sends orders. It requires `UPSTOX_ACCESS_TOKEN` for market status and quotes and `--cost-per-lot` for a conservative round-trip fee/tax allowance before recommending a conditional position. `--lot-size` is an optional cross-check, not an override. The displayed risk applies only if both legs fill as a single spread at or above the quoted minimum net credit and costs are within the allowance; never leg in based on the display.

Run the offline regression checks with `python -m unittest -v`.
