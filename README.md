# Hardened Two-Broker Multi-Asset Trend System (Config C — Max Aggression)

Designed for **`₹2,00,000` total capital** split cleanly across two uncorrelated brokers (zero capital double-counting, whole-lot verified, `< 30%` Max DD across `2021–2026`):

- **Sleeve 1 — Delta Exchange India (`$1,200 ≈ ₹1,02,000`):** `6h` Donchian(20) + EMA(50/200) + **`3.5× ATR` trail** across **`BTCUSD`, `ETHUSD`, `SOLUSD`, `BNBUSD`, `DOGEUSD`** (`2.0%` risk/trade, `8.0%` max open risk, `75%` max margin util, `5×` isolated leverage, **post-only Maker entries with IOC fallback**, **winner pyramiding `+0.5×` @ `+1.5R`**, and **`15%` drawdown brake**).
- **Sleeve 2 — Kotak Neo (`₹98,000`):** `6h` Donchian(20) + EMA(50/200) + **`3.0× ATR` trail** across **`MCX GOLDPETAL`** (1g futures, long + short) and **`NSE SILVERBEES.NS`** (spot ETF, long-only), `2.5%` risk/trade, `8.0%` max open risk, winner pyramiding, and `15%` drawdown brake.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Step 1: Backtest the Exact `₹2,00,000` Two-Broker Portfolio
```bash
# Recent 2 years on real Delta India 6H candles + 6H Gold/Silver whole-lot
python run_hardened_backtest.py --two-broker --interval 6h --days 730 --maker

# Full 5.9-year multi-regime check (2021-2026: 6H Crypto + 1D Gold/Silver)
python run_hardened_backtest.py --two-broker --interval 6h --start 2021-01-01 --maker

# Delta India $1,200 5-coin sleeve only
python run_hardened_backtest.py --usd --interval 6h --days 730
```

## Step 2: Timeframe Plateau & OOS Sweep
```bash
# 2021-2023 Out-of-Sample Crypto Sweep (Binance futures)
python run_timeframe_sweep.py --crypto-only --whole-lots --capital 1200 --usd-inr 1 --maker --intervals 4h 6h 12h 1d --start 2021-01-01 --end 2023-12-31

# Recent 2-Year Real Delta India Sweep
python run_timeframe_sweep.py --usd --intervals 4h 6h 12h 1d
```

## Step 3: Live / Paper Execution
```bash
# Sleeve 1 (Delta Exchange India): one public data cycle (remove --once for 6H daemon)
DELTA_ENV=testnet DRY_RUN=true python live_delta_donchian.py --once

# Sleeve 2 (Kotak Neo MCX GOLDPETAL + NSE SILVERBEES.NS): generate order tickets & alerts
python live_kotak_mcx_advisor.py --capital 98000
# Record a confirmed Kotak Neo fill / pyramid / exit:
python live_kotak_mcx_advisor.py --record-entry GOLDPETAL LONG 8 9600 9240
python live_kotak_mcx_advisor.py --record-pyramid GOLDPETAL 4 10150
python live_kotak_mcx_advisor.py --record-exit GOLDPETAL 10400 --fees-inr 60
```

**Production order submission is disabled.** Before enabling testnet orders, use a dedicated [Delta India demo account](https://demo.delta.exchange) with no existing positions or orders, set the account to isolated margin, set all 5 symbols (`BTCUSD`, `ETHUSD`, `SOLUSD`, `BNBUSD`, `DOGEUSD`) to `5×` isolated leverage, and turn off automatic margin top-up and DETO commission. Configure testnet credentials locally (`BOT_DATA_DIR` for persistent state/logs) and keep `DRY_RUN=true` until ready for sandbox burn-in.

Run only one bot per testnet account from a persistent `BOT_DATA_DIR`. Preserve `bot_state-testnet-orders.json` across restarts; paper state is stored separately. After an unresolved order response, missing protection, or unexpected account state, inspect actual Delta orders, positions, and fills before running `python live_delta_donchian.py --resume` once verified flat.

Run the offline regression test suite with `python -m unittest -v`.
