"""
ENGINE B ADVISOR — COLLATERAL-BACKED NIFTY WEEKLY CREDIT SPREADS
Data Source: Upstox V3 (Free with Upstox Pro) or Yahoo Finance fallback.

Computes the exact weekly defined-risk Bull Put or Bear Call Spread:
  - Trend Filter: Daily EMA(20) vs EMA(50)
  - Volatility: 20-day Realized Vol * 1.15 VRP markup
  - Strikes: Short ~0.80 SD OTM, Protective Wing ~1.60 SD OTM (rounded to nearest 50-pt Nifty strike)
  - Sizing: Strictly caps max loss at 1.5% of pledged collateral pool
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


def listed_expiry_contracts(opt: str, now: datetime, min_dte: int = 4, max_dte: int = 10):
    response = requests.get(UPSTOX_MASTERS["NSE"], timeout=20)
    response.raise_for_status()
    listed = {}
    today = now.date()
    min_date = today + timedelta(days=min_dte)
    max_date = today + timedelta(days=max_dte)
    for inst in json.loads(gzip.decompress(response.content)):
        if inst.get("underlying_key") != "NSE_INDEX|Nifty 50" or inst.get("instrument_type") != opt:
            continue
        expiry = datetime.fromtimestamp(inst["expiry"] / 1000, now.tzinfo).date()
        if min_date <= expiry <= max_date:
            listed.setdefault(expiry, {})[int(round(float(inst["strike_price"])))] = inst
    if not listed:
        raise ValueError(f"No Nifty expiry listed between {min_dte} and {max_dte} DTE")
    expiry = min(listed)
    return expiry, listed[expiry]


def fetch_option_chain(token: str, expiry):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    base = "https://api.upstox.com/v2"
    status = requests.get(f"{base}/market/status/NFO", headers=headers, timeout=15, allow_redirects=False)
    status.raise_for_status()
    if status.json()["data"]["status"] != "NORMAL_OPEN":
        raise ValueError("NFO is not in its normal trading session")
    response = requests.get(
        f"{base}/option/chain",
        headers=headers,
        timeout=15,
        allow_redirects=False,
        params={"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": expiry.isoformat()},
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        raise ValueError("Empty option chain returned by Upstox")
    return rows


def spread_quote_from_chain(rows: list, short: dict, long: dict, opt: str):
    quotes = {}
    for row in rows:
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
    parser.add_argument("--min-dte", type=int, default=4, help="Minimum days to expiry for weekly spread selection (default: 4)")
    parser.add_argument("--use-yf", action="store_true", help="Use Yahoo Finance ^NSEI instead of Upstox V3")
    args = parser.parse_args()
    if not math.isfinite(args.capital) or args.capital <= 0 or (args.lot_size is not None and args.lot_size <= 0):
        parser.error("capital and any expected lot size must be positive")
    if args.cost_per_lot is not None and (not math.isfinite(args.cost_per_lot) or args.cost_per_lot <= 0):
        parser.error("cost-per-lot must be positive")
    if args.min_dte < 0 or args.min_dte > 10:
        parser.error("min-dte must be between 0 and 10")

    cfg = PortfolioConfig(initial_capital=args.capital)
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    if args.use_yf or not token:
        print("Fetching Nifty 50 daily data via Yahoo Finance (^NSEI)...")
        df = load_yfinance("^NSEI", interval="1d", days=120)
    else:
        print("Fetching Nifty 50 daily data via Upstox V3...")
        df = load_upstox("NIFTY", interval="1d", days=120, exchange="NSE")

    # Use completed daily data for EMA/RV regime, never a partial session bar.
    df = df.loc[df.index.date < now.date()].copy()
    if len(df) < 50 or (now.date() - df.index[-1].date()).days > 4 or not (np.isfinite(df["close"]) & (df["close"] > 0)).all():
        print("NO TRADE: insufficient, stale, or invalid daily Nifty history")
        return
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    latest = df.dropna().iloc[-1]

    rv20 = float(latest["rv20"])
    iv_est = max(rv20 * cfg.vrp_iv_multiplier, 0.12)
    is_bull = latest["ema20"] >= latest["ema50"]
    opt = "PE" if is_bull else "CE"
    strategy = "BULL PUT SPREAD (Uptrend: EMA20 >= EMA50)" if is_bull else "BEAR CALL SPREAD (Downtrend: EMA20 < EMA50)"
    max_risk_rs = Decimal(str(args.capital)) * Decimal(str(cfg.spread_risk_per_week_pct))

    try:
        expiry, legs = listed_expiry_contracts(opt, now, min_dte=args.min_dte, max_dte=10)
        dte = max((expiry - now.date()).days, 1)
        if not token or args.cost_per_lot is None:
            raise ValueError("An Upstox token for live quotes and --cost-per-lot are required")
        chain_rows = fetch_option_chain(token, expiry)
        live_spots = [
            float(r["underlying_spot_price"])
            for r in chain_rows
            if r.get("underlying_spot_price") is not None and math.isfinite(float(r["underlying_spot_price"])) and float(r["underlying_spot_price"]) > 0
        ]
        spot = live_spots[0] if live_spots else float(latest["close"])
        weekly_sd = spot * iv_est * np.sqrt(dte / 365.0)
        if is_bull:
            short_k = round_to_strike(spot - cfg.short_strike_sd * weekly_sd, 50)
            long_k = round_to_strike(spot - cfg.long_strike_sd * weekly_sd, 50)
        else:
            short_k = round_to_strike(spot + cfg.short_strike_sd * weekly_sd, 50)
            long_k = round_to_strike(spot + cfg.long_strike_sd * weekly_sd, 50)

        width = abs(short_k - long_k)
        print(f"Nifty candidate (not an order): {strategy}; {short_k}/{long_k} {opt} ({dte} DTE), budget Rs {max_risk_rs:,.2f}")
        if width == 0:
            raise ValueError("Rounded strikes coincide")
        if short_k not in legs or long_k not in legs:
            raise ValueError(f"Both strikes ({short_k}/{long_k} {opt}) are not listed for {expiry}")
        short, long = legs[short_k], legs[long_k]
        if short["lot_size"] != long["lot_size"] or int(short["lot_size"]) <= 0:
            raise ValueError("Legs do not share a valid lot size")
        lot_size = int(short["lot_size"])
        if args.lot_size is not None and args.lot_size != lot_size:
            raise ValueError(f"Expected lot size {args.lot_size} differs from listed {lot_size}")
        print(f"Listed expiry {expiry}, lot size {lot_size}, one-lot loss before credit and costs Rs {width * lot_size:,.2f}")

        credit, quote_qty = spread_quote_from_chain(chain_rows, short, long, opt)
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
Date           : {now.date()} (Regime from {df.index[-1].date()})
Nifty Spot     : {spot:,.2f}
EMA(20) / (50) : {latest['ema20']:,.2f} / {latest['ema50']:,.2f}
20d RealizedVol: {rv20*100:.1f}%  | Est. {dte}-Day 1-SD Move: +/- {weekly_sd:.0f} pts
Regime         : {strategy}

CONDITIONAL SPREAD ONLY — {expiry} ({dte} DTE):
  1. SELL : {short['trading_symbol']} x {lots} lot(s) ({lots * lot_size} qty)
  2. BUY  : {long['trading_symbol']} x {lots} lot(s) ({lots * lot_size} qty) [Protective Wing]
  Use a combo/spread limit order at >= {credit} pts net credit if your broker supports it.
  If executing via basket orders, legs fill individually (buy wing first to avoid naked margin spike).
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
            r = requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat, "text": report},
                timeout=10,
            )
            if not r.ok or not r.json().get("ok"):
                print(f"Telegram rejected alert (HTTP {r.status_code})")
        except (requests.RequestException, ValueError):
            print("Telegram send failed")


if __name__ == "__main__":
    main()
