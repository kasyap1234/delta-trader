# Config C trading system

Config C models ₹2,00,000 split between a Delta India crypto sleeve ($1,200 at ₹85/USD) and a Kotak metals sleeve (₹98,000). The deployed workflow runs Delta on testnet and Kotak in paper mode. The NIFTY tools are separate from Config C.

- **Delta:** 6h Donchian(20), EMA(50/200), 3.5× ATR trail, 2% risk per trade / 8% open risk, up to 75% margin utilization at 5× isolated leverage, maker entry with IOC fallback, winner pyramiding, and a 15% drawdown brake. The five-symbol research basket includes BTC, ETH, SOL, DOGE, and BNB; set `DELTA_SYMBOLS` to the symbols available in the target environment.
- **Kotak:** 6h Donchian(20), EMA(50/200), 3× ATR trail, 2.5% risk per trade / 8% open risk, winner pyramiding, and a 15% drawdown brake. GOLDPETAL can go long or short; SILVERBEES is long-only.

## Where to look

| Path | Purpose |
| --- | --- |
| [`live_delta_donchian.py`](live_delta_donchian.py) | Delta signal evaluation, testnet order execution, and exchange reconciliation. |
| [`live_kotak_mcx_advisor.py`](live_kotak_mcx_advisor.py) | Kotak metals order tickets, manual fill recording, and paper trading. |
| [`config.py`](config.py), [`data_loader.py`](data_loader.py), [`portfolio_engine.py`](portfolio_engine.py) | Settings, market data loaders, and shared indicators/backtest engines, respectively. |
| [`research/`](research/) | Portfolio backtest, timeframe comparisons, NIFTY option replay and manual tickets. |
| [`research/snapshots/`](research/snapshots/README.md) | Preserved historical CSVs from different runs; not current bot state. |
| [`tests/`](tests/) and [`deploy/`](deploy/) | Offline regression suite and VM bootstrap/systemd units. |
| `data/` | Local bot state, journals, and logs. Git-ignored; preserve it across deploys. |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Research commands

Run these modules from the repository root. Commands that write CSVs print a unique folder under `research/output/` (not tracked by Git). The two-broker mode writes an equity curve only; other portfolio modes write the equity curve and trades together.

```bash
# Recent 6h Delta candles + 6h native INR Upstox MCX GOLDPETAL / NSE SILVERBEES candles
python -m research.portfolio_backtest --two-broker --interval 6h --days 730 --maker

# Legacy proxy comparison: Delta 6h crypto + 6h Yahoo COMEX gold/silver futures proxies
python -m research.portfolio_backtest --two-broker --interval 6h --days 730 --kotak-source yfinance --maker

# Longer proxy run: Binance 6h crypto + daily Yahoo gold/silver futures
python -m research.portfolio_backtest --two-broker --interval 6h --start 2021-01-01 --end 2026-11-22 --kotak-source yfinance --maker

# Delta India $1,200 5-coin sleeve only
python -m research.portfolio_backtest --usd --interval 6h --days 730

# Compare timeframes on a fixed window of Binance futures data
python -m research.timeframe_sweep --crypto-only --whole-lots --capital 1200 --usd-inr 1 --maker --intervals 4h 6h 12h 1d --start 2021-01-01 --end 2023-12-31

# Compare timeframes on recent Delta India candles
python -m research.timeframe_sweep --usd --intervals 4h 6h 12h 1d
```

Separate NIFTY tools: `python -m research.nifty_options_replay --help` describes the Upstox expired-option candle replay with simulated fills. `python -m research.nifty_spread_tickets --help` describes the manual spread advisor; it does not submit orders.

## Live and paper execution
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

**Production order submission is disabled.** Before enabling testnet orders, use a dedicated [Delta India demo account](https://demo.delta.exchange) with no existing positions or orders. Set the account to isolated margin and each configured symbol to `5×` isolated leverage. Turn off automatic margin top-up and DETO commission. Configure testnet credentials locally (`BOT_DATA_DIR` for persistent state/logs) and keep `DRY_RUN=true` until ready for sandbox burn-in.

Run only one bot per testnet account from a persistent `BOT_DATA_DIR`. Preserve `bot_state-testnet-orders.json` across restarts; paper state is stored separately. After an unresolved order response, missing protection, or unexpected account state, inspect actual Delta orders, positions, and fills before running `python live_delta_donchian.py --resume` once verified flat.

Run the offline regression test suite from the repository root with `python -m unittest -v`. The VM bootstrap uses the same command.
