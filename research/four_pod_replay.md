# Four-Pod replay decision trail

## What is shared

[`four_pod_engine.py`](../four_pod_engine.py) prepares the latest closed UTC candle and the prior usable feature row, then computes a new virtual book, integer signed net target, and tick-rounded stop from explicit inputs. It performs no I/O and does not mutate the input book. It retains live's Donchian → Supertrend → TSMOM → Squeeze order, gross shared virtual margin, UTC bar-open slots, same-bar exit behavior, and tightest aligned stop (or ATR fallback). Funding is requested only at the old Squeeze lookup point; a missing answer still gates that entry. The live funding endpoint, window, and averaging implementation are unchanged.

[`live_delta_donchian.py`](../live_delta_donchian.py) still owns quotes, paper stops and funding, reconciliation, wallet/brake state, duplicate suppression, and orders. It installs the returned virtual book and consumed candle before persisting an order or stop-edit intent. The `3pod-` state prefix, Config C path, production-order gate and venue symbol defaults are unchanged. The old [`validate_4pod_crypto.py`](validate_4pod_crypto.py) remains a **legacy Binance proxy benchmark**, not a replay of these decisions.

## Run locally

```bash
# Deterministic synthetic smoke scenario; no local market cache needed.
.venv/bin/python -m research.replay_four_pod --demo --squeeze-funding 0.00005 \
  --output "$(mktemp -d)/replay"

# Archived local data (paths in market.json are relative to the manifest unless absolute).
.venv/bin/python -m research.replay_four_pod --manifest /path/to/market.json \
  --start 2025-06-01T00:00Z --end 2025-07-01T00:00Z --output "$(mktemp -d)/replay"
```

`market.json` is an ordered list (same order as live `symbols` for decision comparisons). Example item with **scenario** contract metadata:

```json
[
  {
    "symbol": "ETHUSD",
    "csv": "ETHUSD_6h.csv",
    "venue": "Delta Exchange India production",
    "kind": "venue",
    "source": "local archive of /v2/history/candles for ETHUSD",
    "metadata_source": "scenario values, not a historical product snapshot",
    "contract_value": 0.01,
    "tick_size": "0.05"
  }
]
```

Use `kind: venue`, `proxy`, or `synthetic` accurately. CSV columns: `timestamp,open,high,low,close,volume`; timestamps are timezone-qualified **candle opens**. Start and end are inclusive **decision closes** on UTC 6h boundaries. The CLI requires the full contiguous window for all symbols from `start − 165 days + 6h` through `end − 6h` and rejects gaps, duplicates, and invalid rows in that range. Each event recomputes features only from its trailing window and warmup. This models Delta's 80-day request plus 85-day loader padding shortly after a 6h close; identical raw input is needed for exact EMA/Supertrend parity. No fallback data, config/`.env`, network, or account access.

### Execution assumptions and outputs

- Starts flat with one USD wallet ($1,200 by default); sizing uses cash capped at starting capital, not unrealized gains. Symbol decisions share virtual margin and are processed in manifest order; the simulator separately checks new/increased net exposure against the same wallet at 5× leverage. Neither check simulates exchange liquidation or all exchange limits.
- Mark is the just-closed candle close, not a historical mark tick. Default modeled half-spread is 4 bps, extra adverse slippage 5 bps, taker fee 5 bps per fill, and IOC collar 15 bps rounded like live. An IOC fills in full or is rejected. Like live and paper, replay gates new or increased net exposure at execution, even when an existing virtual book requests it. Reductions can proceed, and a flip rechecks the gate after its close before reopening. No maker fills. A held net stop is checked against the completed candle's trade OHLC proxy, fills from the stop or worse gap open with costs, and clears the book. Same-size stop edits only tighten; a resize can set a different stop. On contiguous events, a stop already crossed at the open is modeled at that open; other stops are recognized at close.
- Signed 8h cash funding defaults to 1 bp (longs pay, shorts receive at positive rates). It uses the last available mark at settlement. A contiguous gap-open exit precedes funding strictly after the open; funding exactly at the open is already settled by the prior close. Unknown intrabar crossings (or crossings after skipped events outside the strict CLI) retain the funding-before-stop convention. OHLC cannot establish the true intrabar order. Replay rolls UTC day/week loss baselines after any prior-period gap exits and before funding and close-recognized stops at the new boundary. `--squeeze-funding` supplies a separate constant modeled signal observation. Without it, Squeeze funding is unavailable, not assumed zero. Neither input is observed historical funding.
- `intents.csv` records quotes, virtual book/target/stop and simulated outcome; `fills.csv` records signed net entry, resize, exit, stop and flip fills; `funding.csv` records settlements; `equity.csv` records wallet, unrealized P&L and net positions; `run.json` records scenario settings, input coverage and hashes, code hashes and results. Terminal positions are marked, not force-liquidated.

This is decision parity for identical inputs, not prediction of testnet or real fills. The simulator omits partial fills, order-book depth, tick-level mark stops, wallet discrepancies, outages, latency, and exchange restart/reconciliation behavior. No new authenticated tests or deployment were performed. The replay does not compare performance to Config C and is not a holdout or expected return estimate.

### Testnet execution and recovery limits

Four-Pod net IOCs persist the signed change, side-specific limit and client order ID with the consumed candle before submitting. A reduction does not cancel its existing reduce-only stop before reconciliation. Recovery checks a terminal order lookup against its identity, any echoed terms and the fill-price limit, then checks the position and stop. It never retries the IOC or opens the second leg of an interrupted flip. It records a verified partial fill only after confirming protection at the actual remaining size. A confirmed flat position may need cleanup of its own stop.

A same-size trail persists the desired trigger with the consumed candle and keeps the old stop snapshot until read-back. After an uncertain or rejected edit, reconciliation confirms an already-applied trigger or retries the edit against the same verified stop at its old trigger and size. An already-filled stop goes through normal exit reconciliation instead. It does not re-evaluate the consumed candle.

The Four-Pod exchange adapter defers same-side increases until protected adds are verified on Delta testnet. The shared signal engine is unchanged; paper and replay can still model these adds, so this is **not execution parity**. The separate Config C pyramid path is unchanged.

For a new reduction, the adapter persists the stop-edit attempt before issuing the resize. Restart reads the broker before committing; it never reissues a spent resize. Older pending intents can be confirmed from broker state but are not eligible for a new resize. If a legacy add is already underprotected, or the reduction stop cannot be confirmed, the adapter latches an entry halt and retains the pending intent. It does not automatically replace the stop, flatten, or repeat an ambiguous IOC. If the stop is absent with a nonzero remainder, **the entire remainder may be unprotected**. The existing non-IOC missing-stop path may attempt one reduce-only emergency close; that attempt is not a guarantee of execution.

For unresolved protection, verify the actual signed position, terminal IOC/fills, and working reduce-only stop at Delta before any recovery action. If an operator changes the exposure, reconcile the preserved state against that action before resuming; a flat broker alone does not clear an unmatched pending intent. Resolve persisted `pending_net` or `pending_trail` before rolling back. Release still requires a dedicated Delta India testnet account with no concurrent bot, and authenticated evidence for a partial fill, resize failure, and restart read-back. The local REST fake does not establish those venue behaviors.

### Testnet release checklist

On 2026-11-25, 04:05–04:06 UTC, an operator-approved maintenance window temporarily isolated the existing GCP bot's Delta India testnet account. The bot was stopped, the account was verified flat with no working orders, and the probe used fresh local state via the VM's IP-whitelisted connection. The BTCUSD two-contract entry IOC and one-contract reduce-only IOC both filled completely. Delta reduced the working stop to one contract without changing its trigger. Delta returned HTTP 400 for a deliberately malformed edit on that stop; restart issued GETs only and latched the entry halt. The final reduce-only cleanup filled, the account was verified flat with no working orders, and the unchanged service resumed. See the local evidence `research/output/stop-recovery/testnet-probe.log`. **The genuine partial IOC fill gate remains open; a fully filled reduction that leaves a position is not a partial IOC fill.**

1. Prefer a disposable **Delta India testnet** account. If using the deployed bot's testnet account, require an approved maintenance window and stop the deployed service before using that account. From a local workstation, use a fresh `BOT_DATA_DIR` with one symbol, isolated margin, and the smallest supported quantity that can demonstrate a partial fill. Verify the account is flat with no working orders and no concurrent writers before submitting anything. Do not use production credentials or the deployed state file.
2. Establish a protected position and exercise a net reduction. For a true partial-fill result, require the authenticated terminal IOC record to show `0 < unfilled_size < size`. Compare the signed remainder, working stop ID/side/size/trigger/reduce-only flag, client ID, and persisted contracts and P&L before and after restart. If Delta fills the IOC completely or not at all, record that and leave the partial-fill gate open.
3. In a separate case, verify a rejected stop edit and restart: record the actual broker rejection, preserve the original stop, and verify the persisted attempt is spent and no duplicate IOC or edit is sent. A client-side injected timeout or rejection tests adapter behavior but does not count as a broker rejection. If protection becomes uncertain, stop the validation and reconcile immediately; do not induce missing coverage by removing protection.
4. Capture UTC timestamps and redacted order/position/stop read-backs, cleanup results, and the code revision under test. Verify the disposable account returns to flat with no working orders. Commit, push and redeploy the shared-engine refactor only after the release checks pass; do not enable real-money orders or promote an unqualified strategy.

## Local verification

Working tree based on `74b061d`, with the pre-existing naming/venue edits and staged research deletions preserved. `.venv/bin/python -m unittest discover -v`: original baseline 44 tests, 8 failures/2 errors with local environment defaults; 44 passed before extraction with `DELTA_STRATEGY_MODE=config_c`; 52 passed after extraction and test-local strategy pinning. [`tests/test_four_pod_parity.py`](../tests/test_four_pod_parity.py) compares exact live-adapter and replay intents/books including stale and duplicate bars, lazy funding, UTC TSMOM flip, margin/brake/blockers and a gap-through net stop. Broker execution remains covered by mocked tests, not real orders. After the recovery changes, `.venv/bin/python -m unittest discover -s tests` passed 81 tests, including cold-start reconciliation for excluded symbols, fail-closed handling when product metadata is unavailable, and the cycle and restart regressions in [`tests/test_four_pod_execution.py`](../tests/test_four_pod_execution.py).

Two practical offline smoke runs (no Config C comparison):

| 6h scenario, both starting at USD 1,200 | UTC decision closes | Events / intents / fills | End cash / marked equity |
| --- | --- | ---: | ---: |
| Built-in synthetic DEMO-A/B, constant Squeeze input per command above | 2024-07-01 00:00 → 2024-08-01 00:00 | 125 / 250 / 15 | $1,197.46 / $1,218.29 |
| Local Delta India BTCUSD, ETHUSD, SOLUSD, DOGEUSD candle cache; no Squeeze observation | 2025-06-01 00:00 → 2025-07-01 00:00 | 121 / 484 / 22 | $1,186.86 / $1,187.23 |

The second run used local, untracked `research/output/_cache_delta_2y/{SYMBOL}_6h_730d.csv` (provenance: `load_cached_delta_klines` at `74b061d:research/validate_3pod_crypto.py`, which writes public Delta India production candles; cache not independently re-fetched). Used candle opens per symbol: 2024-12-18 06:00 to 2025-06-30 18:00 UTC, 779 rows, no fallback. For the manifest use the symbol order in the table, with `(contract_value, tick_size)` respectively `(.001, "0.5")`, `(.01, "0.05")`, `(1, "0.001")`, `(100, "0.000001")` (scenario values, not as-of metadata). All modeled costs otherwise use the defaults above. Rebuilt final cash and marked equity from the fill, funding and intent CSVs for both runs.
