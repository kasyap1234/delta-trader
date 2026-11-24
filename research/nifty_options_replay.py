"""
UPSTOX EXPIRED OPTIONS REPLAY — ENGINE B (NIFTY WEEKLY CREDIT SPREADS)

Uses historical Upstox market data with simulated fills:
  1. Upstox V3 Daily Spot candles for Nifty 50 (EMA20/EMA50 regime + 20-day realized vol).
  2. Upstox V2 Expired Instruments API for real expired weekly Nifty option contracts & lot sizes.
  3. Upstox V2 Expired Historical Candles for real daily OHLC option prices of both spread legs.
  4. Runs the synthetic BSM model over the exact same date window side-by-side for direct comparison.

Requires: UPSTOX_ACCESS_TOKEN in .env (Upstox Plus plan for Expired Instruments API).
"""

import argparse
import math
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import (
    load_upstox,
    load_upstox_expired_candles,
    load_upstox_expired_expiries,
    load_upstox_expired_option_contracts,
)
from portfolio_engine import run_weekly_credit_spreads
from research.outputs import new_run_directory


def _round_to_strike(px: float, step: int = 50) -> int:
    return int(round(px / step) * step)


def _find_contract(contracts: list, strike: int, opt_type: str) -> Optional[dict]:
    for c in contracts:
        if c.get("instrument_type") == opt_type and int(round(float(c.get("strike_price", -1)))) == strike:
            return c
    return None


def run_authentic_upstox_nifty_spreads(
    capital: float = 200_000.0,
    cost_per_lot: float = 75.0,
    slippage_pts_per_leg: float = 0.5,
    max_expiries: int = 26,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Replays weekly Nifty credit spreads on Upstox expired option candles.
    Returns (trades_df, realized_equity_series, spot_df).
    """
    cfg = PortfolioConfig(initial_capital=capital)

    print("1. Fetching available expired Nifty 50 expiries from Upstox V2...")
    expiries = load_upstox_expired_expiries("NSE_INDEX|Nifty 50")
    if not expiries:
        raise RuntimeError("No expired Nifty expiries returned. Verify Upstox Plus plan and UPSTOX_ACCESS_TOKEN.")

    expiries = expiries[-max_expiries:]
    first_exp_dt = datetime.strptime(expiries[0], "%Y-%m-%d")
    spot_start = (first_exp_dt - timedelta(days=120)).strftime("%Y-%m-%d")
    spot_end = expiries[-1]

    print(f"2. Fetching authentic Nifty 50 daily spot from Upstox V3 ({spot_start} -> {spot_end})...")
    spot_df = load_upstox("NIFTY", interval="1d", start_date=spot_start, end_date=spot_end, exchange="NSE", include_warmup=False)
    spot_df["ret"] = np.log(spot_df["close"] / spot_df["close"].shift(1))
    spot_df["rv20"] = spot_df["ret"].rolling(20).std() * np.sqrt(252)
    spot_df["ema20"] = spot_df["close"].ewm(span=20, adjust=False).mean()
    spot_df["ema50"] = spot_df["close"].ewm(span=50, adjust=False).mean()
    spot_df = spot_df.dropna()

    cur_capital = capital
    previous_exit = None
    records: List[Dict] = []
    skipped = Counter()
    cheapest_rejected_lot = None

    print(f"3. Replaying {len(expiries)} real expired weekly Nifty expiries...")
    for exp_str in expiries:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        # Select strikes at the signal session's close, then enter at the next open.
        eligible_days = spot_df[spot_df.index.date < exp_date]
        if len(eligible_days) < 5:
            skipped["insufficient Nifty spot history"] += 1
            continue
        signal = eligible_days.iloc[-5]
        entry_date = eligible_days.index[-4].date()
        if previous_exit is not None and entry_date <= previous_exit:
            skipped["previous spread still open at entry"] += 1
            continue
        sessions = spot_df.loc[(spot_df.index.date >= entry_date) & (spot_df.index.date <= exp_date), "close"]
        if sessions.index[-1].date() != exp_date:
            skipped["missing Nifty expiry-day close"] += 1
            continue

        S0 = float(signal["close"])
        iv_est = max(float(signal["rv20"]) * cfg.vrp_iv_multiplier, 0.12)
        weekly_sd = S0 * iv_est * np.sqrt(7.0 / 365.0)
        is_bull = signal["ema20"] >= signal["ema50"]

        if is_bull:
            opt_type = "PE"
            short_k = _round_to_strike(S0 - cfg.short_strike_sd * weekly_sd, 50)
            long_k = _round_to_strike(S0 - cfg.long_strike_sd * weekly_sd, 50)
            side = "BULL_PUT_SPREAD"
        else:
            opt_type = "CE"
            short_k = _round_to_strike(S0 + cfg.short_strike_sd * weekly_sd, 50)
            long_k = _round_to_strike(S0 + cfg.long_strike_sd * weekly_sd, 50)
            side = "BEAR_CALL_SPREAD"

        width = abs(short_k - long_k)
        if width < 50:
            skipped["strike width below 50 points"] += 1
            continue

        contracts = load_upstox_expired_option_contracts(exp_str, "NSE_INDEX|Nifty 50")
        short_c = _find_contract(contracts, short_k, opt_type)
        long_c = _find_contract(contracts, long_k, opt_type)
        if not short_c or not long_c:
            skipped["one or both selected contracts not returned"] += 1
            continue

        lot_size = int(short_c.get("lot_size", 0))
        if lot_size <= 0 or int(long_c.get("lot_size", 0)) != lot_size:
            skipped["invalid or mismatched lot sizes"] += 1
            continue
        from_str = entry_date.strftime("%Y-%m-%d")

        short_key = short_c.get("expired_instrument_key") or short_c["instrument_key"]
        long_key = long_c.get("expired_instrument_key") or long_c["instrument_key"]
        short_df = load_upstox_expired_candles(short_key, exp_str, from_str, interval="day")
        long_df = load_upstox_expired_candles(long_key, exp_str, from_str, interval="day")
        time.sleep(0.15)

        if short_df.empty or long_df.empty:
            skipped["one or both option candle series empty"] += 1
            continue

        short_prices = short_df.set_index(short_df.index.date)[["open", "close"]]
        long_prices = long_df.set_index(long_df.index.date)[["open", "close"]]
        spread = (short_prices - long_prices).dropna(how="all")
        if entry_date not in spread.index or not np.isfinite(spread.at[entry_date, "open"]):
            skipped["missing option open on entry date"] += 1
            continue

        raw_credit = float(spread.at[entry_date, "open"])
        net_credit = raw_credit - (2.0 * slippage_pts_per_leg)
        if not (0 < net_credit <= raw_credit < width):
            skipped["entry credit non-positive after slippage or outside spread bounds"] += 1
            continue

        max_loss_per_lot = max((width - net_credit) * lot_size + cost_per_lot, 1.0)
        risk_budget = cur_capital * cfg.spread_risk_per_week_pct
        lots = math.floor(risk_budget / max_loss_per_lot)
        if lots < 1:
            skipped["one lot exceeds risk budget"] += 1
            cheapest_rejected_lot = min(cheapest_rejected_lot or max_loss_per_lot, max_loss_per_lot)
            continue

        exit_reason = "EXPIRY"
        for timestamp, spot_close in sessions.items():
            session = timestamp.date()
            if session == exp_date:
                if exit_reason != "EXPIRY" and session in spread.index and np.isfinite(spread.at[session, "open"]):
                    val = float(spread.at[session, "open"])
                    if 0 <= val <= width:
                        exit_val = val
                        break
                # Nifty index options are cash settled at intrinsic value on expiry.
                exit_reason = "EXPIRY"
                direction = 1 if opt_type == "CE" else -1
                exit_val = max(direction * (spot_close - short_k), 0) - max(direction * (spot_close - long_k), 0)
                break

            # Once a close triggers an exit, price that exit at the next session's open.
            price = "close" if exit_reason == "EXPIRY" else "open"
            value = float(spread.at[session, price]) if session in spread.index else float("nan")
            if not 0 <= value <= width:
                raise RuntimeError(f"{exp_str}: missing or invalid option spread {price} on {session} after entry")
            if price == "open":
                exit_val = value
                break
            if value <= net_credit * 0.35:
                exit_reason = "TP_65PCT"
            elif value >= net_credit * 2.5:
                exit_reason = "SL_2.5X"

        exit_slip = 0.0 if exit_reason == "EXPIRY" else 2.0 * slippage_pts_per_leg
        net_pts = net_credit - (exit_val + exit_slip)
        net_pts = max(net_pts, -(width - net_credit))  # Capped by protective wing
        net_pnl = (net_pts * lot_size - cost_per_lot) * lots
        cur_capital += net_pnl
        previous_exit = session

        records.append({
            "expiry": exp_str,
            "entry_date": str(entry_date),
            "exit_date": str(session),
            "side": side,
            "short_strike": short_k,
            "long_strike": long_k,
            "lot_size": lot_size,
            "lots": lots,
            "entry_credit_pts": round(net_credit, 2),
            "exit_val_pts": round(exit_val, 2),
            "net_pnl_rs": round(net_pnl, 2),
            "equity_rs": round(cur_capital, 2),
            "exit_reason": exit_reason,
        })

    print(f"Replay: {len(records)} trade(s), {sum(skipped.values())} skipped out of {len(expiries)} expiries.")
    for reason, count in skipped.most_common():
        print(f"  {count}: {reason}")
    if cheapest_rejected_lot is not None:
        risk_rate = cfg.spread_risk_per_week_pct
        print(f"  Initial {risk_rate:.1%} risk budget: ₹{capital * risk_rate:,.2f}; "
              f"lowest rejected one-lot max loss: ₹{cheapest_rejected_lot:,.2f} "
              f"(requires ₹{cheapest_rejected_lot / risk_rate:,.2f} equity at that cap).")

    trades_df = pd.DataFrame(records)
    if trades_df.empty:
        return trades_df, pd.Series(dtype=float), spot_df

    eq = pd.Series(trades_df["equity_rs"].values, index=pd.to_datetime(trades_df["exit_date"]))
    return trades_df, eq, spot_df


def _summarize(name: str, pnls: np.ndarray, initial_capital: float) -> dict:
    if len(pnls) == 0:
        return {"Model": name, "Trades": 0, "Win %": 0.0, "PF": 0.0, "Net Return %": 0.0, "Max DD %": 0.0}
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else 99.0
    eq = initial_capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(np.insert(eq, 0, initial_capital))[1:]
    max_dd = ((peak - eq) / peak).max() * 100.0
    return {
        "Model": name,
        "Trades": len(pnls),
        "Win %": round(len(wins) / len(pnls) * 100.0, 1),
        "PF": round(float(pf), 2),
        "Net Return %": round((eq[-1] - initial_capital) / initial_capital * 100.0, 2),
        "Max DD %": round(float(max_dd), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Nifty Weekly Credit Spreads on Authentic Upstox Expired Option Candles")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Capital in Rs (default: 200000)")
    parser.add_argument("--cost-per-lot", type=float, default=75.0, help="Round-trip brokerage+taxes per lot in Rs (default: 75)")
    parser.add_argument("--slippage-pts", type=float, default=0.5, help="Slippage per option leg in points (default: 0.5)")
    parser.add_argument("--max-expiries", type=int, default=26, help="Max historical weekly expiries to replay (default: 26)")
    args = parser.parse_args()

    real_df, real_eq, spot_df = run_authentic_upstox_nifty_spreads(
        capital=args.capital,
        cost_per_lot=args.cost_per_lot,
        slippage_pts_per_leg=args.slippage_pts,
        max_expiries=args.max_expiries,
    )

    if real_df.empty:
        print("No spreads qualified; no trade CSV was written.")
        return

    # Compare against Synthetic BSM over the exact same date window
    cfg = PortfolioConfig(initial_capital=args.capital)
    eval_start = pd.Timestamp(real_df["entry_date"].iloc[0], tz="Asia/Kolkata")
    eval_end = pd.Timestamp(real_df["exit_date"].iloc[-1], tz="Asia/Kolkata")
    bsm_trades, _, _ = run_weekly_credit_spreads(spot_df, "NIFTY", cfg, eval_start=eval_start, eval_end=eval_end)

    summary = pd.DataFrame([
        _summarize("1. AUTHENTIC UPSTOX EXPIRED OPTIONS (Whole Lots)", real_df["net_pnl_rs"].to_numpy(), args.capital),
        _summarize("2. SYNTHETIC BSM MODEL (Same Window)", np.array([t.net_pnl for t in bsm_trades]), args.capital),
    ])

    print(f"\n================ REAL UPSTOX OPTIONS VS. BSM ({real_df['entry_date'].iloc[0]} -> {real_df['exit_date'].iloc[-1]}) ================\n")
    print(summary.to_string(index=False))
    trade_path = new_run_directory("nifty-options-replay") / "spread_trades.csv"
    real_df.to_csv(trade_path, index=False)
    print(f"\nSaved trade log: {trade_path}")


if __name__ == "__main__":
    main()
