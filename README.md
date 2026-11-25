# Trading system

The Delta service places testnet orders only. The Kotak service runs an independent ETF rotation in paper mode.

- **Delta Four-Pod Balanced** (`four_pod_balanced`): 6-hour Donchian, Supertrend + ADX, time-series momentum, and funding + volume squeeze. The old `3pod` name is a compatibility alias. The bot nets by symbol and uses collared IOC entries. Legacy Config C is a separate Donchian mode.
- **Four-Pod offline replay** (`python -m research.replay_four_pod`): uses the live decision engine and simulates one net position and stop per symbol from local 6h candles. Quotes, fills and costs are modeled; it is not a broker replay. The old five-coin Binance USD-M benchmark is **legacy, not decision parity**.
- **Kotak ETF rotation:** daily paper trading. The legacy MCX metals advisor and NIFTY tools are separate workflows.

## Where to look

| Path | Purpose |
| --- | --- |
| [`live_delta_donchian.py`](live_delta_donchian.py) | Delta signal evaluation, testnet order execution, and exchange reconciliation. |
| [`four_pod_engine.py`](four_pod_engine.py), [`research/replay_four_pod.py`](research/replay_four_pod.py) | Shared Four-Pod decisions and the offline replay adapter. |
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

Run these modules from the repository root. The Four-Pod replay takes an explicit new output directory; other research commands that write CSVs use `research/output/` (not tracked by Git). The two-broker mode writes an equity curve only; other portfolio modes write the equity curve and trades together.

```bash
# Recent 6h Delta candles + 6h native INR Upstox MCX GOLDPETAL / NSE SILVERBEES candles
python -m research.portfolio_backtest --two-broker --interval 6h --days 730 --maker

# Legacy proxy comparison: Delta 6h crypto + 6h Yahoo COMEX gold/silver futures proxies
python -m research.portfolio_backtest --two-broker --interval 6h --days 730 --kotak-source yfinance --maker

# Longer proxy run: Binance 6h crypto + daily Yahoo gold/silver futures
python -m research.portfolio_backtest --two-broker --interval 6h --start 2021-01-01 --end 2026-11-22 --kotak-source yfinance --maker

# Canonical Four-Pod replay smoke run (synthetic, no network or credentials)
python -m research.replay_four_pod --demo --squeeze-funding 0.00005 --output "$(mktemp -d)/replay"

# Legacy comparison only: Binance proxy with different entries and independent pod stops
python -m research.validate_4pod_crypto
```

For local CSV inputs, the manifest format, coverage requirements and the limits of parity, see the [Four-Pod replay decision trail](research/four_pod_replay.md).

Retired Delta strategy sweeps and three-pod comparisons remain available at Git checkpoint `74b061d`. The older portfolio command above is a research reference, not the strategy running on the VM.

Separate NIFTY tools: `python -m research.nifty_options_replay --help` describes the Upstox expired-option candle replay with simulated fills. `python -m research.nifty_spread_tickets --help` describes the manual spread advisor; it does not submit orders.

## Live and paper execution
```bash
# Sleeve 1 (Delta Exchange India): one public data cycle (remove --once for 6H daemon)
DELTA_ENV=testnet DELTA_SYMBOLS= DRY_RUN=true python live_delta_donchian.py --strategy four_pod_balanced --once

# Sleeve 2 (Kotak Neo MCX GOLDPETAL + NSE SILVERBEES.NS): generate order tickets & alerts
python live_kotak_mcx_advisor.py --capital 98000
# Record a confirmed Kotak Neo fill / pyramid / exit:
python live_kotak_mcx_advisor.py --record-entry GOLDPETAL LONG 8 9600 9240
python live_kotak_mcx_advisor.py --record-pyramid GOLDPETAL 4 10150
python live_kotak_mcx_advisor.py --record-exit GOLDPETAL 10400 --fees-inr 60
```

Leave `DELTA_SYMBOLS` blank to use the venue defaults. Testnet uses BTCUSD, ETHUSD, SOLUSD and DOGEUSD. `DELTA_ENV=prod` adds BNBUSD for production-data dry runs. A nonempty `DELTA_SYMBOLS` overrides either basket.

**Production order submission is disabled.** Before enabling testnet orders, use a dedicated [Delta India demo account](https://demo.delta.exchange) with no existing positions or orders. Set the account to isolated margin and each configured symbol to `5×` isolated leverage. Turn off automatic margin top-up and DETO commission. Configure testnet credentials locally (`BOT_DATA_DIR` for persistent state/logs) and keep `DRY_RUN=true` until ready for sandbox burn-in.

Run only one bot per testnet account from a persistent `BOT_DATA_DIR`. Preserve `bot_state-3pod-testnet-orders.json` for Four-Pod Balanced across restarts. The filename keeps its old prefix for compatibility. Legacy Config C and paper mode use separate state files. After an unresolved order response, missing protection, or unexpected account state, inspect actual Delta orders, positions, and fills before running `python live_delta_donchian.py --strategy four_pod_balanced --resume` once verified flat.

Run the offline regression test suite from the repository root with `python -m unittest -v`. The VM bootstrap uses the same command.
