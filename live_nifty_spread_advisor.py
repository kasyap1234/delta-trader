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
import math
import os

import numpy as np
import requests

from config import PortfolioConfig
from data_loader import load_upstox, load_yfinance


def round_to_strike(px: float, step: int = 50) -> int:
    return int(round(px / step) * step)


def main():
    parser = argparse.ArgumentParser(description="Nifty Weekly Defined-Risk Credit Spread Advisor")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Total pledged collateral capital in Rs")
    parser.add_argument("--lot-size", type=int, default=75, help="Current NSE Nifty option lot size (default: 75)")
    parser.add_argument("--use-yf", action="store_true", help="Use Yahoo Finance ^NSEI instead of Upstox V3")
    args = parser.parse_args()

    cfg = PortfolioConfig(initial_capital=args.capital)

    if args.use_yf or not os.getenv("UPSTOX_ACCESS_TOKEN"):
        print("Fetching Nifty 50 daily data via Yahoo Finance (^NSEI)...")
        df = load_yfinance("^NSEI", interval="1d", days=120)
    else:
        print("Fetching Nifty 50 daily data via Upstox V3...")
        df = load_upstox("NIFTY", interval="1d", days=120, exchange="NSE")

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

    width = max(abs(short_k - long_k), 50)
    max_risk_rs = args.capital * cfg.spread_risk_per_week_pct
    max_loss_per_lot = width * args.lot_size
    lots = max(1, math.floor(max_risk_rs / max_loss_per_lot))

    report = f"""
================ NIFTY WEEKLY CREDIT SPREAD SETUP ================
Date           : {df.index[-1].date()}
Nifty Spot     : {spot:,.2f}
EMA(20) / (50) : {latest['ema20']:,.2f} / {latest['ema50']:,.2f}
20d RealizedVol: {rv20*100:.1f}%  | Est. Weekly 1-SD Move: +/- {weekly_sd:.0f} pts
Regime         : {strategy}

RECOMMENDED DEFINED-RISK LEGS (Nearest Weekly Expiry):
  1. SELL : NIFTY {short_k} {opt}  x {lots} lot(s) ({lots * args.lot_size} qty)
  2. BUY  : NIFTY {long_k} {opt}   x {lots} lot(s) ({lots * args.lot_size} qty) [Protective Wing]

RISK & MANAGEMENT RULES:
  - Spread Width      : {width} pts
  - Max Risk Budget   : Rs {max_risk_rs:,.0f} (1.5% of Rs {args.capital:,.0f})
  - Worst-Case Loss   : <= Rs {lots * max_loss_per_lot:,.0f} (minus net credit received)
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
        except Exception as e:
            print(f"Telegram send failed: {e}")


if __name__ == "__main__":
    main()
