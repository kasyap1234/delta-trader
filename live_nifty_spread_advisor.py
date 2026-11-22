"""
ENGINE B ADVISOR — COLLATERAL-BACKED NIFTY WEEKLY CREDIT SPREADS
Data Source: Upstox V3 (Free with Upstox Pro) or Yahoo Finance fallback.

Computes the exact weekly defined-risk Bull Put or Bear Call Spread:
  - Trend Filter: Daily EMA(20) vs EMA(50)
  - Volatility: 20-day Realized Vol * 1.15 VRP markup
  - Strikes: Short ~0.80 SD OTM, Protective Wing ~1.60 SD OTM (rounded to nearest 50-pt Nifty strike)
  - Sizing: Strictly caps max loss at 1.5% of Rs 2,00,000 collateral pool
  - Exit Rules: Take profit at 65% of credit collected, or cut at 2.5x credit stop-loss
"""

import argparse
import gzip
import json
import math
import os
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import requests

from config import PortfolioConfig
from data_loader import UPSTOX_MASTERS, load_upstox, load_yfinance


def round_to_strike(px: float, step: int = 50) -> int:
    return int(round(px / step) * step)


def listed_spread(short_k: int, long_k: int, opt: str, now: datetime):
    response = requests.get(UPSTOX_MASTERS["NSE"], timeout=20)
    response.raise_for_status()
    listed = {}
    for inst in json.loads(gzip.decompress(response.content)):
        if inst.get("underlying_key") != "NSE_INDEX|Nifty 50" or inst.get("instrument_type") != opt:
            continue
        expiry = datetime.fromtimestamp(inst["expiry"] / 1000, now.tzinfo).date()
        if now.date() <= expiry <= (now + timedelta(days=7)).date():
            listed.setdefault(expiry, {})[inst["strike_price"]] = inst
    if not listed:
        raise ValueError("No Nifty expiry listed in the next seven days")
    expiry = min(listed)
    legs = listed[expiry]
    if short_k not in legs or long_k not in legs:
        raise ValueError(f"Both strikes are not listed for {expiry}")
    short, long = legs[short_k], legs[long_k]
    if short["lot_size"] != long["lot_size"] or int(short["lot_size"]) <= 0:
        raise ValueError("Legs do not share a valid lot size")
    return expiry, short, long


def spread_quote(token, expiry, short, long, opt):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    base = "https://api.upstox.com/v2"
    status = requests.get(f"{base}/market/status/NSE", headers=headers, timeout=15, allow_redirects=False)
    status.raise_for_status()
    if status.json()["data"]["status"] != "NORMAL_OPEN":
        raise ValueError("NSE is not in its normal trading session")
    response = requests.get(f"{base}/option/chain", headers=headers, timeout=15, allow_redirects=False,
                            params={"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": expiry.isoformat()})
    response.raise_for_status()
    quotes = {}
    for row in response.json()["data"]:
        leg = row.get("put_options" if opt == "PE" else "call_options")
        if leg:
            quotes[leg["instrument_key"]] = leg["market_data"]
    bid, ask = quotes[short["instrument_key"]], quotes[long["instrument_key"]]
    credit = Decimal(str(bid["bid_price"])) - Decimal(str(ask["ask_price"]))
    width = abs(short["strike_price"] - long["strike_price"])
    if not credit.is_finite() or not 0 < credit < width or not 0 < float(ask["ask_price"]):
        raise ValueError("No valid positive credit from the short bid and protective ask")
    return credit, min(int(bid["bid_qty"]), int(ask["ask_qty"]))


def main():
    parser = argparse.ArgumentParser(description="Nifty Weekly Defined-Risk Credit Spread Advisor")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Total pledged collateral capital in Rs")
    parser.add_argument("--lot-size", type=int, help="Optional expected lot size, verified against the listed contracts")
    parser.add_argument("--cost-per-lot", type=float, help="Worst-case round-trip fees and taxes per spread lot, in Rs; required for a recommendation")
    parser.add_argument("--use-yf", action="store_true", help="Use Yahoo Finance ^NSEI instead of Upstox V3")
    args = parser.parse_args()
    if not math.isfinite(args.capital) or args.capital <= 0 or (args.lot_size is not None and args.lot_size <= 0):
        parser.error("capital and any expected lot size must be positive")
    if args.cost_per_lot is not None and (not math.isfinite(args.cost_per_lot) or args.cost_per_lot <= 0):
        parser.error("cost-per-lot must be positive")

    cfg = PortfolioConfig(initial_capital=args.capital)
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    if args.use_yf or not token:
        print("Fetching Nifty 50 daily data via Yahoo Finance (^NSEI)...")
        df = load_yfinance("^NSEI", interval="1d", days=120)
    else:
        print("Fetching Nifty 50 daily data via Upstox V3...")
        df = load_upstox("NIFTY", interval="1d", days=120, exchange="NSE")

    # Use completed daily data, not a partial bar from the current trading session.
    df = df.loc[df.index.date < now.date()].copy()
    if len(df) < 50 or (now.date() - df.index[-1].date()).days > 4 or not (np.isfinite(df["close"]) & (df["close"] > 0)).all():
        print("NO TRADE: insufficient, stale, or invalid daily Nifty history")
        return
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    latest = df.dropna().iloc[-1]

    spot = float(latest["close"])
    rv20 = float(latest["rv20"])
    iv_est = max(rv20 * cfg.vrp_iv_multiplier, 0.12)
    weekly_sd = spot * iv_est * np.sqrt(7.0 / 365.0)
    is_bull = latest["ema20"] >= latest["ema50"]

    if is_bull:
        strategy = "BULL PUT SPREAD (Uptrend: EMA20 >= EMA50)"
        short_k = round_to_strike(spot - cfg.short_strike_sd * weekly_sd, 50)
        long_k = round_to_strike(spot - cfg.long_strike_sd * weekly_sd, 50)
        opt = "PE"
    else:
        strategy = "BEAR CALL SPREAD (Downtrend: EMA20 < EMA50)"
        short_k = round_to_strike(spot + cfg.short_strike_sd * weekly_sd, 50)
        long_k = round_to_strike(spot + cfg.long_strike_sd * weekly_sd, 50)
        opt = "CE"

    width = abs(short_k - long_k)
    max_risk_rs = Decimal(str(args.capital)) * Decimal(str(cfg.spread_risk_per_week_pct))
    print(f"Nifty candidate (not an order): {strategy}; {short_k}/{long_k} {opt}, budget Rs {max_risk_rs:,.2f}")
    try:
        if width == 0:
            raise ValueError("Rounded strikes coincide")
        expiry, short, long = listed_spread(short_k, long_k, opt, now)
        lot_size = int(short["lot_size"])
        if args.lot_size is not None and args.lot_size != lot_size:
            raise ValueError(f"Expected lot size {args.lot_size} differs from listed {lot_size}")
        print(f"Listed expiry {expiry}, lot size {lot_size}, one-lot loss before credit and costs Rs {width * lot_size:,.2f}")
        if not token or args.cost_per_lot is None:
            raise ValueError("An Upstox token for live quotes and --cost-per-lot are required")
        credit, quote_qty = spread_quote(token, expiry, short, long, opt)
        max_loss_per_lot = (Decimal(width) - credit) * lot_size + Decimal(str(args.cost_per_lot))
        lots = min(int(max_risk_rs // max_loss_per_lot), quote_qty // lot_size)
        if lots < 1:
            raise ValueError(f"One quoted lot requires Rs {max_loss_per_lot:,.2f} of risk and {lot_size} units of quoted depth")
    except requests.RequestException:
        print("NO TRADE: could not verify NSE contracts, market status or live quotes")
        return
    except (ValueError, KeyError, TypeError, ArithmeticError) as error:
        print(f"NO TRADE: {error}")
        return

    report = f"""
================ NIFTY WEEKLY CREDIT SPREAD SETUP ================
Date           : {df.index[-1].date()}
Nifty Spot     : {spot:,.2f}
EMA(20) / (50) : {latest['ema20']:,.2f} / {latest['ema50']:,.2f}
20d RealizedVol: {rv20*100:.1f}%  | Est. Weekly 1-SD Move: +/- {weekly_sd:.0f} pts
Regime         : {strategy}

CONDITIONAL SPREAD ONLY — {expiry}:
  1. SELL : {short['trading_symbol']} x {lots} lot(s) ({lots * lot_size} qty)
  2. BUY  : {long['trading_symbol']} x {lots} lot(s) ({lots * lot_size} qty) [Protective Wing]
  Use one spread limit order at a minimum net credit of {credit} points. Do not leg in.
  Quotes and displayed depth are a snapshot, not a fill guarantee.

RISK & MANAGEMENT RULES:
  - Spread Width      : {width} pts
  - Max Risk Budget   : Rs {max_risk_rs:,.0f} (1.5% of Rs {args.capital:,.0f})
  - Max Loss at Limit : Rs {lots * max_loss_per_lot:,.2f} (includes the supplied cost allowance)
  - Take Profit       : Close spread when value drops to 35% of credit (65% max profit)
  - Stop Loss         : Close spread if spread value expands to 2.5x credit received
==================================================================
"""
    print(report)

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat, "text": report},
                timeout=10,
            )
        except requests.RequestException:
            print("Telegram send failed")


if __name__ == "__main__":
    main()
