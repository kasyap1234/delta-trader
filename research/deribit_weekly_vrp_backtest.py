"""
Real-Data Weekly Defined-Risk Crypto Option Selling Backtest (Deribit Public History API)
=========================================================================================
ZERO BSM SIMULATION:
  - Entry prices: Actual executed Deribit option trades from
    https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time
    on each Friday (08:30 - 14:30 UTC) for the exact 7-DTE Friday expiry.
  - Expiry settlement: Official Deribit Friday 08:00 UTC delivery prices from
    https://www.deribit.com/api/v2/public/get_delivery_prices.
  - Underlying regime features (ADX14, EMA20/50): Cached actual Binance USD-M daily closes.
  - Sizing: $1,200 USD account, Delta India contract steps (0.001 BTC, 0.01 ETH),
    2.0% max defined loss per spread, Deribit/Delta fee schedule (0.03% underlying capped at 12.5% premium)
    + explicit 5% (base) and 15% (stress) bid-ask slippage haircut on collected credit.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from portfolio_engine import PortfolioTrade
from research.crypto_idle_margin_study import _adx, load_cached_binance_klines
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

CACHE_DIR = Path(__file__).resolve().parent / "output" / "_cache_deribit_vrp"
DERIBIT_HIST_URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
DERIBIT_DELIV_URL = "https://www.deribit.com/api/v2/public/get_delivery_prices"

CAPITAL_USD = 1200.0
LOT_STEP = {"BTC": 0.001, "ETH": 0.01}


def deribit_expiry_code(dt: pd.Timestamp) -> str:
    """Formats a Friday expiry date as Deribit symbol code, e.g. 9OCT26 or 20NOV26."""
    return f"{dt.day}{dt.strftime('%b').upper()}{dt.strftime('%y')}"


def fetch_delivery_prices(currency: str) -> Dict[str, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{currency}_delivery_prices.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    idx_name = f"{currency.lower()}_usd"
    out: Dict[str, float] = {}
    offset = 0
    while offset < 900:
        r = requests.get(DERIBIT_DELIV_URL, params={"index_name": idx_name, "count": 100, "offset": offset}, timeout=15)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
        if not data:
            break
        for row in data:
            out[row["date"]] = float(row["delivery_price"])
        if len(data) < 100:
            break
        offset += 100
        time.sleep(0.1)
    cache_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def fetch_friday_chain(currency: str, entry_friday: pd.Timestamp) -> pd.DataFrame:
    """Fetches actual executed Deribit option trades on entry_friday (08:30-14:30 UTC) for the 7-DTE expiry."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expiry_dt = entry_friday + pd.Timedelta(days=7)
    exp_code = deribit_expiry_code(expiry_dt)
    cache_path = CACHE_DIR / f"{currency}_{entry_friday.strftime('%Y-%m-%d')}_{exp_code}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    t1 = int((entry_friday + pd.Timedelta(hours=8, minutes=30)).timestamp() * 1000)
    t2 = int((entry_friday + pd.Timedelta(hours=14, minutes=30)).timestamp() * 1000)
    prefix = f"{currency}-{exp_code}-"

    all_rows = []
    cur_end = t2
    # Up to 3 pages of 1000 trades per Friday window
    for _ in range(3):
        r = requests.get(
            DERIBIT_HIST_URL,
            params={
                "currency": currency,
                "kind": "option",
                "start_timestamp": t1,
                "end_timestamp": cur_end,
                "count": 1000,
            },
            timeout=20,
        )
        r.raise_for_status()
        res = r.json().get("result", {})
        trades = res.get("trades", [])
        if not trades:
            break
        for tr in trades:
            inst = tr.get("instrument_name", "")
            if not inst.startswith(prefix):
                continue
            parts = inst.split("-")
            if len(parts) != 4:
                continue
            strike = float(parts[2])
            opt_type = parts[3]  # 'C' or 'P'
            idx_px = float(tr["index_price"])
            px_coin = float(tr["price"])
            all_rows.append({
                "timestamp": tr["timestamp"],
                "instrument": inst,
                "strike": strike,
                "opt_type": opt_type,
                "px_coin": px_coin,
                "index_price": idx_px,
                "px_usd": px_coin * idx_px,
                "iv": float(tr.get("iv", np.nan)),
                "amount": float(tr.get("amount", 0.0)),
            })
        if not res.get("has_more"):
            break
        min_ts = min(tr["timestamp"] for tr in trades)
        if min_ts <= t1:
            break
        cur_end = min_ts - 1
        time.sleep(0.12)

    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=["timestamp", "instrument", "strike", "opt_type", "px_coin", "index_price", "px_usd", "iv", "amount"])
    df.to_csv(cache_path, index=False)
    time.sleep(0.08)
    return df


def summarize_chain(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """Aggregates raw executed option trades into per-(opt_type, strike) median USD prices."""
    if df.empty:
        return np.nan, pd.DataFrame()
    spot = float(df["index_price"].median())
    agg = (
        df.groupby(["opt_type", "strike"], as_index=False)
        .agg(px_usd=("px_usd", "median"), iv=("iv", "median"), trades=("px_usd", "count"))
    )
    return spot, agg


def pick_spread(
    chain: pd.DataFrame,
    spot: float,
    opt_type: str,
    short_otm_pct: float,
    wing_otm_pct: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Finds actually traded short & protective long strikes closest to target OTM %.
    Returns (k_short, k_long, px_short_usd, px_long_usd) or None if unavailable.
    """
    sub = chain[chain["opt_type"] == opt_type]
    if len(sub) < 2:
        return None
    if opt_type == "P":
        target_s = spot * (1.0 - short_otm_pct)
        target_l = spot * (1.0 - wing_otm_pct)
        s_pool = sub[sub["strike"] < spot * 0.99]
        if s_pool.empty:
            return None
        s_row = s_pool.iloc[(s_pool["strike"] - target_s).abs().argsort()[:1]].iloc[0]
        k_s, px_s = float(s_row["strike"]), float(s_row["px_usd"])
        l_pool = sub[sub["strike"] < k_s]
        if l_pool.empty:
            return None
        l_row = l_pool.iloc[(l_pool["strike"] - target_l).abs().argsort()[:1]].iloc[0]
        k_l, px_l = float(l_row["strike"]), float(l_row["px_usd"])
    else:
        target_s = spot * (1.0 + short_otm_pct)
        target_l = spot * (1.0 + wing_otm_pct)
        s_pool = sub[sub["strike"] > spot * 1.01]
        if s_pool.empty:
            return None
        s_row = s_pool.iloc[(s_pool["strike"] - target_s).abs().argsort()[:1]].iloc[0]
        k_s, px_s = float(s_row["strike"]), float(s_row["px_usd"])
        l_pool = sub[sub["strike"] > k_s]
        if l_pool.empty:
            return None
        l_row = l_pool.iloc[(l_pool["strike"] - target_l).abs().argsort()[:1]].iloc[0]
        k_l, px_l = float(l_row["strike"]), float(l_row["px_usd"])

    # Require sensible positive credit and wing width <= 20% of spot
    width = abs(k_s - k_l)
    if width <= 0 or width > spot * 0.25 or px_s <= px_l:
        return None
    return k_s, k_l, px_s, px_l


def run_vrp_strategy(
    chains_by_ccy: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, float, float, pd.DataFrame]]],
    regime_by_ccy: Dict[str, pd.DataFrame],
    mode: str = "iron_condor",   # "iron_condor", "chop_ic", "trend_spread"
    short_otm: float = 0.08,
    wing_otm: float = 0.14,
    risk_per_spread_pct: float = 0.02,
    credit_haircut: float = 0.05,
    adx_max: float = 25.0,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, int]]:
    # Collect all expiry dates for equity timeline
    all_dates = sorted({exp for rows in chains_by_ccy.values() for _, exp, _, _, _ in rows})
    if not all_dates:
        return [], pd.Series(dtype=float), {}

    capital = CAPITAL_USD
    pnl_by_date = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
    trades: List[PortfolioTrade] = []
    diag = {"weeks_checked": 0, "skipped_no_quotes": 0, "skipped_regime": 0, "skipped_lot_size": 0}

    # Iterate chronologically across all weekly cycles
    by_friday: Dict[pd.Timestamp, List[Tuple[str, pd.Timestamp, float, float, pd.DataFrame]]] = {}
    for ccy, rows in chains_by_ccy.items():
        for fri, exp, spot, settle, chain in rows:
            by_friday.setdefault(fri, []).append((ccy, exp, spot, settle, chain))

    for fri in sorted(by_friday.keys()):
        for ccy, exp, spot, settle, chain in by_friday[fri]:
            diag["weeks_checked"] += 1
            reg_df = regime_by_ccy[ccy]
            prior_reg = reg_df.loc[reg_df.index < fri]
            if prior_reg.empty:
                diag["skipped_regime"] += 1
                continue
            r_row = prior_reg.iloc[-1]
            adx_val = float(r_row["adx14"])
            is_bull = float(r_row["ema20"]) >= float(r_row["ema50"])

            if mode == "chop_ic" and adx_val >= adx_max:
                diag["skipped_regime"] += 1
                continue

            sides_to_sell: List[str] = []
            if mode in ("iron_condor", "chop_ic"):
                sides_to_sell = ["P", "C"]
            elif mode == "trend_spread":
                sides_to_sell = ["P"] if is_bull else ["C"]

            for opt_type in sides_to_sell:
                picked = pick_spread(chain, spot, opt_type, short_otm, wing_otm)
                if picked is None:
                    diag["skipped_no_quotes"] += 1
                    continue
                k_s, k_l, px_s, px_l = picked
                gross_credit = px_s - px_l
                # Apply bid-ask haircut + Deribit/Delta option fee (0.03% of spot per leg capped at 12.5% of option px)
                fee_s = min(0.0003 * spot, 0.125 * px_s)
                fee_l = min(0.0003 * spot, 0.125 * px_l)
                net_credit = gross_credit * (1.0 - credit_haircut) - (fee_s + fee_l)
                if net_credit <= 0:
                    diag["skipped_no_quotes"] += 1
                    continue

                width = abs(k_s - k_l)
                max_loss_unit = max(width - net_credit, 1e-6)
                risk_usd = capital * risk_per_spread_pct
                step = LOT_STEP[ccy]
                raw_qty = risk_usd / max_loss_unit
                lots = math.floor(raw_qty / step)
                if lots < 1:
                    diag["skipped_lot_size"] += 1
                    continue
                qty = lots * step

                # Official Deribit expiry intrinsic settlement
                if opt_type == "P":
                    intrinsic_payout = max(k_s - settle, 0.0) - max(k_l - settle, 0.0)
                    label = "BULL_PUT_SPREAD"
                else:
                    intrinsic_payout = max(settle - k_s, 0.0) - max(settle - k_l, 0.0)
                    label = "BEAR_CALL_SPREAD"

                # Settlement delivery fee (0.015% of spot capped at 12.5% of intrinsic if ITM)
                deliv_fee = min(0.00015 * spot, 0.125 * intrinsic_payout) if intrinsic_payout > 0 else 0.0
                net_pnl = (net_credit - intrinsic_payout - deliv_fee) * qty
                capital += net_pnl
                pnl_by_date.loc[exp] += net_pnl
                trades.append(PortfolioTrade(
                    engine="DERIBIT_VRP",
                    asset=f"{ccy}USD",
                    side=label,
                    entry_time=fri,
                    exit_time=exp,
                    entry_price=spot,
                    exit_price=settle,
                    qty=qty,
                    gross_pnl=(gross_credit - intrinsic_payout) * qty,
                    funding_cost=(fee_s + fee_l + deliv_fee) * qty,
                    net_pnl=net_pnl,
                    r_multiple=net_pnl / max(max_loss_unit * qty, 1e-6),
                    exit_reason="ITM_LOSS" if intrinsic_payout > 0 else "OTM_WIN",
                ))

    eq = (CAPITAL_USD + pnl_by_date.cumsum()).resample("1D").last().ffill()
    return trades, eq, diag


def main():
    out_dir = new_run_directory("deribit-weekly-vrp")

    # 2-year weekly Friday schedule matching our OOS window: 2024-11-22 -> 2026-11-13 (104 weekly expiries)
    fridays = pd.date_range("2024-11-22", "2026-11-13", freq="W-FRI", tz="UTC")
    print(f"Fetching/loading real Deribit weekly option trades + official delivery prices for {len(fridays)} Fridays (BTC + ETH)...")

    chains_by_ccy: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, float, float, pd.DataFrame]]] = {}
    regime_by_ccy: Dict[str, pd.DataFrame] = {}

    for ccy in ["BTC", "ETH"]:
        deliv = fetch_delivery_prices(ccy)
        # Load cached Binance 6h candles for daily ADX14 / EMA20 / EMA50 regime filter
        b_df = load_cached_binance_klines(f"{ccy}USDT", "6h", "2021-11-24", "2026-11-23")
        d = b_df[["open", "high", "low", "close", "volume"]].tz_convert("UTC").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
        d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
        d["adx14"] = _adx(d, 14)
        regime_by_ccy[ccy] = d.dropna()

        rows = []
        missing_deliv = 0
        empty_chains = 0
        for fri in fridays:
            exp = fri + pd.Timedelta(days=7)
            exp_key = exp.strftime("%Y-%m-%d")
            if exp_key not in deliv:
                missing_deliv += 1
                continue
            raw_trades = fetch_friday_chain(ccy, fri)
            spot, agg = summarize_chain(raw_trades)
            if agg.empty or not math.isfinite(spot):
                empty_chains += 1
                continue
            rows.append((fri, exp, spot, deliv[exp_key], agg))
        chains_by_ccy[ccy] = rows
        print(f"  {ccy}: {len(rows)}/{len(fridays)} weekly expiries with real executed trades & official settlement (missing_deliv={missing_deliv}, empty_chain={empty_chains})")

    variants = [
        ("1. Always-On Iron Condor (8%/14% OTM, 5% slip)", "iron_condor", 0.08, 0.14, 0.02, 0.05),
        ("2. Always-On Iron Condor (8%/14% OTM, 15% stress)", "iron_condor", 0.08, 0.14, 0.02, 0.15),
        ("3. Chop-Gated IC (ADX<25, 8%/14% OTM, 5% slip)", "chop_ic", 0.08, 0.14, 0.02, 0.05),
        ("4. Chop-Gated IC (ADX<25, 8%/14% OTM, 15% stress)", "chop_ic", 0.08, 0.14, 0.02, 0.15),
        ("5. Chop-Gated IC Wider (ADX<25, 10%/16% OTM, 5%)", "chop_ic", 0.10, 0.16, 0.02, 0.05),
        ("6. Trend-Aligned Spread (EMA20/50, 8%/14% OTM, 5%)", "trend_spread", 0.08, 0.14, 0.02, 0.05),
        ("7. Trend-Aligned Spread (EMA20/50, 6%/12% OTM, 5%)", "trend_spread", 0.06, 0.12, 0.02, 0.05),
        ("8. Trend-Aligned Spread (EMA20/50, 6%/12%, 15% str)", "trend_spread", 0.06, 0.12, 0.02, 0.15),
    ]

    summary_rows = []
    curves = {}
    for name, mode, s_otm, w_otm, r_pct, hc in variants:
        tr, eq, diag = run_vrp_strategy(
            chains_by_ccy, regime_by_ccy, mode=mode, short_otm=s_otm, wing_otm=w_otm,
            risk_per_spread_pct=r_pct, credit_haircut=hc
        )
        st = calc_curve_stats(name, eq, tr, CAPITAL_USD)
        st["SkipNoQ"] = diag.get("skipped_no_quotes", 0)
        st["SkipLot"] = diag.get("skipped_lot_size", 0)
        summary_rows.append(st)
        curves[name] = eq
        if "3. Chop-Gated IC (ADX<25, 8%/14% OTM, 5% slip)" == name and tr:
            pd.DataFrame([t.__dict__ for t in tr]).to_csv(out_dir / "chop_ic_trades.csv", index=False)
        if "7. Trend-Aligned Spread (EMA20/50, 6%/12% OTM, 5%)" == name and tr:
            pd.DataFrame([t.__dict__ for t in tr]).to_csv(out_dir / "trend_spread_trades.csv", index=False)

    df_sum = pd.DataFrame(summary_rows)
    print("\n================ REAL DERIBIT 2-YEAR WEEKLY OPTION SELLING ($1,200, 2024-11 -> 2026-11) ================")
    print(df_sum.to_string(index=False))

    df_sum.to_csv(out_dir / "deribit_vrp_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "deribit_vrp_equity_curves.csv")
    print(f"\nSaved all real-data Deribit VRP tables and ledgers to {out_dir}/")


if __name__ == "__main__":
    main()
