"""
5-Year Top-5 Liquid Crypto Idle-Margin Companion Study (2021-11 -> 2026-11)
===========================================================================
Strict protocol:
  - Universe: Top 5 liquid majors only (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, DOGEUSDT)
  - Data: Actual Binance USD-M Futures 6h candles + actual historical 8h funding rates
  - Split: In-Sample (2021-11-24 -> 2024-11-23) vs Untouched Out-of-Sample (2024-11-24 -> 2026-11-23)
  - Capital & Constraints: $1,200 USD, whole-lot rounding, 5x leverage, shared 75% max margin cap
  - Baseline: Unchanged Config C (6h Donchian(20) + EMA(50/200) + 3.5x ATR + Pyramid + 15% DD Brake)
  - Idle-Margin Companions (strictly secondary to Config C; zero Config C entries blocked/delayed/resized):
      A. FUNDING_CARRY_REV : Actual historical 8h funding crowding/carry reversal
      B. PAIRS_RV          : Market-neutral hedged spread z-score mean reversion (ETH/BTC, SOL/ETH, BNB/BTC)
      C. CHOP_RANGE        : ADX(<20)-gated Bollinger Band fade to SMA(20)
      D. TREND_PULLBACK    : EMA(20) pullback swing with fixed 1.5x ATR TP / 1.5x ATR SL
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from config import PortfolioConfig
from data_loader import load_binance, resolve_date_window
from portfolio_engine import (
    PortfolioTrade,
    add_ema_atr_features,
    run_shared_donchian_portfolio,
)
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

TOP5_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
CACHE_DIR = Path(__file__).resolve().parent / "output" / "_cache_binance_5y"


def load_binance_funding(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Fetches actual historical 8h funding rates from Binance USD-M Futures (cached locally)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_funding_{start_date}_{end_date}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        idx = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        return pd.Series(df["fundingRate"].values, index=idx, name="fundingRate").sort_index()

    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = requests.get(
            BINANCE_FUNDING_URL,
            params={"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000},
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_t = int(batch[-1]["fundingTime"])
        if last_t <= cur:
            break
        cur = last_t + 1
        time.sleep(0.08)

    if not rows:
        return pd.Series(dtype=float, name="fundingRate")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "fundingRate"]].drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df.to_csv(cache_file)
    return df["fundingRate"]


def load_cached_binance_klines(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.index.name = "timestamp"
        return df.sort_index()
    df = load_binance(symbol, interval=interval, start_date=start_date, end_date=end_date, include_warmup=True)
    df.to_csv(cache_file)
    return df


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def prepare_features(
    asset_dfs: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in asset_dfs.items():
        f = add_ema_atr_features(df)
        f["ema5"] = f["close"].ewm(span=5, adjust=False).mean()
        f["ema20"] = f["close"].ewm(span=20, adjust=False).mean()
        f["sma20"] = f["close"].rolling(20).mean()
        f["std20"] = f["close"].rolling(20).std()
        f["bb_up"] = f["sma20"] + 2.0 * f["std20"]
        f["bb_lo"] = f["sma20"] - 2.0 * f["std20"]
        f["adx14"] = _adx(f, 14)
        f["rsi6"] = _rsi(f["close"], 6)

        # Align actual 8h funding rates onto 6h bars (latest known funding <= bar close, no lookahead)
        fr = funding_map.get(sym, pd.Series(dtype=float))
        if not fr.empty:
            aligned_fr = fr.reindex(f.index, method="ffill").fillna(0.0)
            f["funding_rate"] = aligned_fr
            f["funding_avg3"] = aligned_fr.rolling(4).mean()  # ~24h average funding
        else:
            f["funding_rate"] = 0.0
            f["funding_avg3"] = 0.0

        f = f.iloc[cfg.ema_warmup_bars:].dropna()
        f = f[(f.index >= eval_start) & (f.index <= eval_end)]
        if not f.empty:
            prepared[sym] = f
    return prepared


def _round_lot(sym: str, raw_qty: float, cfg: PortfolioConfig) -> float:
    if raw_qty <= 0 or not math.isfinite(raw_qty):
        return 0.0
    if not cfg.whole_lot_rounding:
        return raw_qty
    step = cfg.lot_multipliers.get(sym, 1.0)
    lots = math.floor(raw_qty / step)
    return lots * step if lots >= 1 else 0.0


def build_base_timeline(
    base_trades: List[PortfolioTrade],
    all_ts: List[pd.Timestamp],
    cfg: PortfolioConfig,
) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], float]:
    """Maps each timestamp to Config C's open {symbol: margin_used} and computes Config C idle %."""
    base_open: Dict[pd.Timestamp, Dict[str, float]] = {ts: {} for ts in all_ts}
    lev = cfg.max_leverage or 5.0
    for t in base_trades:
        margin = (t.qty * t.entry_price) / lev
        for ts in all_ts:
            if t.entry_time <= ts < t.exit_time:
                base_open[ts][t.asset] = margin
    idle_bars = sum(1 for ts in all_ts if not base_open[ts])
    return base_open, round(idle_bars / max(len(all_ts), 1) * 100.0, 1)


def _actual_funding_cashflow(side: str, notional: float, rate: float, periods: int) -> float:
    """
    Returns net funding COST (positive = paid, negative = received) using actual Binance 8h funding rate.
    When fundingRate > 0: longs pay shorts. When fundingRate < 0: shorts pay longs.
    """
    if side == "LONG":
        return notional * rate * periods
    return -notional * rate * periods


def run_idle_companion(
    mode: str,
    prepared: Dict[str, pd.DataFrame],
    base_eq: pd.Series,
    base_open: Dict[pd.Timestamp, Dict[str, float]],
    cfg: PortfolioConfig,
    risk_per_trade: float = 0.0075,   # 0.75% small consistent risk per companion trade
    max_companion_margin_frac: float = 0.40,  # cap companion at 40% of total equity, always inside shared 75%
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Runs a single idle-margin companion strategy strictly on symbols & margin NOT used by Config C.
    Returns (companion_trades, companion_daily_pnl_series, diagnostics).
    """
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    lev = cfg.max_leverage or 5.0
    base_eq_by_ts = base_eq.reindex(pd.DatetimeIndex(all_ts).normalize()).ffill().bfill()

    comp_realized = 0.0
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    pnl_records: List[Tuple[pd.Timestamp, float]] = []
    active_bars = 0

    # Precompute pair spreads for PAIRS_RV
    pair_defs = [("ETHUSDT", "BTCUSDT"), ("SOLUSDT", "ETHUSDT"), ("BNBUSDT", "BTCUSDT")]
    pair_z: Dict[Tuple[str, str], pd.Series] = {}
    if mode == "PAIRS_RV":
        for a, b in pair_defs:
            if a in prepared and b in prepared:
                idx = prepared[a].index.intersection(prepared[b].index)
                lr = np.log(prepared[a].loc[idx, "close"] / prepared[b].loc[idx, "close"])
                z = (lr - lr.rolling(40).mean()) / lr.rolling(40).std().replace(0, np.nan)
                pair_z[(a, b)] = z

    pair_pos: Dict[Tuple[str, str], Optional[Dict]] = {p: None for p in pair_defs}

    def close_single(sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal comp_realized
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_cost"]
        comp_realized += net
        trades.append(
            PortfolioTrade(
                engine=mode, asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_cost"], net_pnl=net, r_multiple=net / max(pos["risk_rs"], 1e-6),
                exit_reason=reason,
            )
        )
        pos_map[sym] = None

    def close_pair(pair: Tuple[str, str], ts: pd.Timestamp, present: Dict[str, pd.Series], reason: str, slip: float) -> None:
        nonlocal comp_realized
        pp = pair_pos[pair]
        if pp is None:
            return
        a, b = pair
        if a not in present or b not in present:
            return
        for leg_sym, leg in ((a, pp["leg_a"]), (b, pp["leg_b"])):
            is_long = leg["side"] == "LONG"
            raw_px = float(present[leg_sym]["open"]) if reason != "END_OF_TEST" else float(present[leg_sym]["close"])
            exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
            gross = (exit_px - leg["entry"]) * leg["qty"] if is_long else (leg["entry"] - exit_px) * leg["qty"]
            net = gross - leg["funding_cost"]
            comp_realized += net
            trades.append(
                PortfolioTrade(
                    engine=mode, asset=leg_sym, side=leg["side"], entry_time=pp["entry_time"], exit_time=ts,
                    entry_price=leg["entry"], exit_price=exit_px, qty=leg["qty"], gross_pnl=gross,
                    funding_cost=leg["funding_cost"], net_pnl=net, r_multiple=net / max(pp["risk_rs"] / 2.0, 1e-6),
                    exit_reason=reason,
                )
            )
        pair_pos[pair] = None

    for bar_idx, ts in enumerate(all_ts):
        present = {}
        for sym, df in prepared.items():
            if ts in df.index:
                row = df.loc[ts]
                present[sym] = row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        base_held = base_open[ts]
        base_eq_now = float(base_eq_by_ts.iloc[bar_idx]) if bar_idx < len(base_eq_by_ts) else cfg.initial_capital
        total_eq = max(base_eq_now + comp_realized, 0.0)
        base_margin = sum(base_held.values())

        if mode == "PAIRS_RV":
            # Yield immediately if Config C enters either leg
            for pair in pair_defs:
                pp = pair_pos[pair]
                if pp is not None and (pair[0] in base_held or pair[1] in base_held):
                    close_pair(pair, ts, present, "YIELD_TO_CONFIG_C", entry_slip)

            # Check pair exits & entries at open using prior bar z-score
            for pair in pair_defs:
                a, b = pair
                if pair not in pair_z or a not in present or b not in present:
                    continue
                prev_a, prev_b = last_bar.get(a), last_bar.get(b)
                if prev_a is None or prev_b is None:
                    continue
                prev_ts = prev_a.name
                if prev_ts not in pair_z[pair].index:
                    continue
                z_val = float(pair_z[pair].loc[prev_ts])
                if not math.isfinite(z_val):
                    continue

                pp = pair_pos[pair]
                if pp is not None:
                    # Exit on reversion, stop z-score, or 28-bar (7-day) time stop
                    reverted = (pp["dir"] == "LONG_SPREAD" and z_val >= -0.3) or (pp["dir"] == "SHORT_SPREAD" and z_val <= 0.3)
                    stopped = (pp["dir"] == "LONG_SPREAD" and z_val <= -3.5) or (pp["dir"] == "SHORT_SPREAD" and z_val >= 3.5)
                    if reverted or stopped or pp["bars"] >= 28:
                        reason = "Z_REVERT" if reverted else ("Z_STOP" if stopped else "TIME_STOP")
                        close_pair(pair, ts, present, reason, cfg.slippage_pct if stopped else entry_slip)
                elif a not in base_held and b not in base_held and total_eq > 0:
                    # Avoid double-using same coin across two pairs
                    used_coins = {s for p, v in pair_pos.items() if v is not None for s in p}
                    if a in used_coins or b in used_coins:
                        continue
                    direction = "LONG_SPREAD" if z_val <= -2.0 else ("SHORT_SPREAD" if z_val >= 2.0 else None)
                    if direction is not None:
                        comp_margin = sum(v["margin_rs"] for v in pair_pos.values() if v is not None)
                        free_margin = min(
                            max(total_eq * cfg.max_margin_utilization - base_margin - comp_margin, 0.0),
                            max(total_eq * max_companion_margin_frac - comp_margin, 0.0),
                        )
                        leg_notional = min(total_eq * 0.35, (free_margin * lev) / 2.0)
                        side_a = "LONG" if direction == "LONG_SPREAD" else "SHORT"
                        side_b = "SHORT" if direction == "LONG_SPREAD" else "LONG"
                        ea = float(present[a]["open"]) * (1.0 + entry_slip if side_a == "LONG" else 1.0 - entry_slip)
                        eb = float(present[b]["open"]) * (1.0 + entry_slip if side_b == "LONG" else 1.0 - entry_slip)
                        qa = _round_lot(a, leg_notional / ea, cfg)
                        qb = _round_lot(b, leg_notional / eb, cfg)
                        if qa > 0 and qb > 0:
                            margin_used = (qa * ea + qb * eb) / lev
                            pair_pos[pair] = {
                                "dir": direction, "entry_time": ts, "bars": 0,
                                "margin_rs": margin_used, "risk_rs": total_eq * 0.015,
                                "leg_a": {"side": side_a, "entry": ea, "qty": qa, "funding_cost": 0.0, "last_f": ts},
                                "leg_b": {"side": side_b, "entry": eb, "qty": qb, "funding_cost": 0.0, "last_f": ts},
                            }

            for sym, row in present.items():
                last_bar[sym] = row

            # Funding & mark-to-market for open pairs
            unr = 0.0
            any_open = False
            for pair, pp in pair_pos.items():
                if pp is None:
                    continue
                any_open = True
                pp["bars"] += 1
                for leg_sym, leg in ((pair[0], pp["leg_a"]), (pair[1], pp["leg_b"])):
                    row = present.get(leg_sym)
                    if row is not None:
                        periods = (ts - leg["last_f"]) // pd.Timedelta(hours=8)
                        if periods:
                            rate = float(row["funding_rate"])
                            leg["funding_cost"] += _actual_funding_cashflow(leg["side"], float(row["close"]) * leg["qty"], rate, periods)
                            leg["last_f"] += periods * pd.Timedelta(hours=8)
                    if leg_sym in last_bar:
                        px = float(last_bar[leg_sym]["close"])
                        unr += ((px - leg["entry"]) * leg["qty"] if leg["side"] == "LONG" else (leg["entry"] - px) * leg["qty"]) - leg["funding_cost"]
            if any_open:
                active_bars += 1
            pnl_records.append((ts, comp_realized + unr))
            continue

        # Single-asset companions: FUNDING_CARRY_REV, CHOP_RANGE, TREND_PULLBACK
        for sym in list(pos_map.keys()):
            if pos_map[sym] is not None and sym in base_held and sym in present:
                close_single(sym, ts, float(present[sym]["open"]), "YIELD_TO_CONFIG_C", entry_slip)

        for sym, row in present.items():
            prev = last_bar.get(sym)
            pos = pos_map[sym]
            if pos is not None and prev is not None:
                is_long = pos["side"] == "LONG"
                exit_now, reason = False, ""
                if mode == "FUNDING_CARRY_REV":
                    fr_avg = float(prev["funding_avg3"])
                    if (is_long and fr_avg >= 0.0) or ((not is_long) and fr_avg <= 0.0001):
                        exit_now, reason = True, "FUNDING_NORM"
                    elif pos["bars"] >= 12:
                        exit_now, reason = True, "TIME_STOP"
                elif mode == "CHOP_RANGE":
                    if (is_long and prev["close"] >= prev["sma20"]) or ((not is_long) and prev["close"] <= prev["sma20"]):
                        exit_now, reason = True, "BB_MID_TP"
                    elif prev["adx14"] >= 25.0:
                        exit_now, reason = True, "REGIME_EXIT"
                    elif pos["bars"] >= 12:
                        exit_now, reason = True, "TIME_STOP"
                elif mode == "TREND_PULLBACK":
                    if pos["bars"] >= 12:
                        exit_now, reason = True, "TIME_STOP"
                if exit_now:
                    close_single(sym, ts, float(row["open"]), reason, entry_slip)
                    continue
            elif pos is None and prev is not None and sym not in base_held and prev["atr14"] > 0 and total_eq > 0:
                side = None
                tp_mult, sl_mult = 1.5, 1.5
                if mode == "FUNDING_CARRY_REV":
                    fr_avg = float(prev["funding_avg3"])
                    if fr_avg <= -0.00010 and prev["rsi6"] < 40:
                        side = "LONG"
                    elif fr_avg >= 0.00035 and prev["close"] < prev["ema5"]:
                        side = "SHORT"
                    tp_mult, sl_mult = 2.0, 1.5
                elif mode == "CHOP_RANGE":
                    if prev["adx14"] < 20.0:
                        if prev["close"] < prev["bb_lo"]:
                            side = "LONG"
                        elif prev["close"] > prev["bb_up"]:
                            side = "SHORT"
                    tp_mult, sl_mult = 2.5, 1.5
                elif mode == "TREND_PULLBACK":
                    if prev["ema50"] > prev["ema200"] and prev["low"] <= prev["ema20"] and prev["rsi6"] < 35:
                        side = "LONG"
                    elif prev["ema50"] < prev["ema200"] and prev["high"] >= prev["ema20"] and prev["rsi6"] > 65:
                        side = "SHORT"
                    tp_mult, sl_mult = 1.5, 1.5

                if side is not None:
                    comp_margin = sum(p["margin_rs"] for p in pos_map.values() if p is not None)
                    free_margin = min(
                        max(total_eq * cfg.max_margin_utilization - base_margin - comp_margin, 0.0),
                        max(total_eq * max_companion_margin_frac - comp_margin, 0.0),
                    )
                    if free_margin > 0:
                        open_px = float(row["open"])
                        entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = sl_mult * float(prev["atr14"])
                        tp_dist = tp_mult * float(prev["atr14"])
                        raw_qty = min((total_eq * risk_per_trade) / sl_dist, (free_margin * lev) / entry_px)
                        qty = _round_lot(sym, raw_qty, cfg)
                        if qty > 0:
                            pos_map[sym] = {
                                "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                                "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                                "tp": entry_px + tp_dist if side == "LONG" else entry_px - tp_dist,
                                "risk_rs": qty * sl_dist, "margin_rs": (qty * entry_px) / lev,
                                "funding_cost": 0.0, "last_f": ts, "bars": 0,
                            }

        for sym, row in present.items():
            last_bar[sym] = row

        # Check intrabar SL/TP and actual 8h funding
        any_open = False
        unr = 0.0
        for sym, pos in list(pos_map.items()):
            if pos is None:
                continue
            row = present.get(sym)
            if row is not None:
                pos["bars"] += 1
                periods = (ts - pos["last_f"]) // pd.Timedelta(hours=8)
                if periods:
                    rate = float(row["funding_rate"])
                    pos["funding_cost"] += _actual_funding_cashflow(pos["side"], float(row["close"]) * pos["qty"], rate, periods)
                    pos["last_f"] += periods * pd.Timedelta(hours=8)
                is_long = pos["side"] == "LONG"
                sl_hit = (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"])
                tp_hit = (row["high"] >= pos["tp"]) if is_long else (row["low"] <= pos["tp"])
                if sl_hit:
                    gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                    raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                    close_single(sym, ts, raw_exit, "GAP_STOP" if gapped else "HARD_SL", cfg.slippage_pct)
                    continue
                if tp_hit:
                    close_single(sym, ts, pos["tp"], "FIXED_TP", entry_slip)
                    continue
            any_open = True
            if sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_cost"]
        if any_open:
            active_bars += 1
        pnl_records.append((ts, comp_realized + unr))

    final_ts = all_ts[-1]
    if mode == "PAIRS_RV":
        for pair in pair_defs:
            if pair_pos[pair] is not None:
                close_pair(pair, final_ts, last_bar, "END_OF_TEST", cfg.slippage_pct)
    else:
        for sym in list(pos_map.keys()):
            if pos_map[sym] is not None and sym in last_bar:
                close_single(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    pnl_s = pd.Series(
        [v for _, v in pnl_records[:-1]] + [comp_realized],
        index=pd.DatetimeIndex([t for t, _ in pnl_records]),
    ).resample("1D").last().ffill()
    return trades, pnl_s, {"active_bar_pct": round(active_bars / max(len(all_ts), 1) * 100.0, 1)}


def evaluate_window(
    label: str,
    asset_dfs: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    start_str: str,
    end_str: str,
) -> pd.DataFrame:
    _, _, eval_start, eval_end = resolve_date_window("6h", start_date=start_str, end_date=end_str, include_warmup=True)
    prepared = prepare_features(asset_dfs, funding_map, cfg, eval_start, eval_end)
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))

    base_trades, base_eq, _ = run_shared_donchian_portfolio(
        asset_dfs, set(asset_dfs.keys()), cfg, eval_start=eval_start, eval_end=eval_end
    )
    base_open, base_idle_pct = build_base_timeline(base_trades, all_ts, cfg)

    rows = []
    base_row = calc_curve_stats("0. Baseline: Unchanged Config C", base_eq, base_trades, cfg.initial_capital)
    base_row["Active %"] = round(100.0 - base_idle_pct, 1)
    rows.append(base_row)

    comp_results = {}
    for mode, title in [
        ("FUNDING_CARRY_REV", "A. + Funding Carry/Crowding Rev"),
        ("PAIRS_RV", "B. + Market-Neutral Pairs RV"),
        ("CHOP_RANGE", "C. + ADX<20 Chop BB Fade"),
        ("TREND_PULLBACK", "D. + Trend EMA(20) Pullback"),
    ]:
        c_trades, c_pnl, c_diag = run_idle_companion(mode, prepared, base_eq, base_open, cfg)
        comb_eq = (base_eq + c_pnl.reindex(base_eq.index).ffill().fillna(0.0)).dropna()
        standalone_eq = (cfg.initial_capital + c_pnl).dropna()
        comp_results[mode] = (c_trades, c_pnl, c_diag)

        st_s = calc_curve_stats(f"   [Standalone] {mode}", standalone_eq, c_trades, cfg.initial_capital)
        st_s["Active %"] = c_diag["active_bar_pct"]
        rows.append(st_s)

        st_c = calc_curve_stats(title, comb_eq, base_trades + c_trades, cfg.initial_capital)
        st_c["Active %"] = round(min(100.0, (100.0 - base_idle_pct) + c_diag["active_bar_pct"]), 1)
        rows.append(st_c)

    print(f"\n================ {label} ({start_str} -> {end_str}) ================")
    print(f"Config C Idle Time (zero positions open): {base_idle_pct:.1f}% of 6h bars | Capital: ${cfg.initial_capital:,.0f} USD\n")
    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))
    return res_df


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Year Top-5 Liquid Crypto Idle-Margin Companion Study")
    parser.add_argument("--capital", type=float, default=1200.0)
    parser.add_argument("--taker-only", action="store_true", help="Stress test with 0.15%% entry slippage")
    args = parser.parse_args()

    entry_slip = 0.0015 if args.taker_only else 0.0004
    cfg = PortfolioConfig(
        initial_capital=args.capital, usd_inr_rate=1.0,
        donchian_risk_per_trade_pct=0.02, max_concurrent_risk_pct=0.08,
        max_crypto_concurrent=5, atr_trail_mult=3.5,
        slippage_pct=0.0015, entry_slippage_pct=entry_slip,
        pyramid_add_fraction=0.5, dd_brake_threshold=0.15,
        whole_lot_rounding=True, enable_credit_spreads=False,
        crypto_assets=TOP5_COINS, commodity_assets=[],
    )

    print("Loading 5-year Binance USD-M 6h candles + actual historical 8h funding rates for Top 5 majors...")
    asset_dfs: Dict[str, pd.DataFrame] = {}
    funding_map: Dict[str, pd.Series] = {}
    for sym in TOP5_COINS:
        df = load_cached_binance_klines(sym, "6h", "2021-11-24", "2026-11-23")
        fr = load_binance_funding(sym, "2021-08-01", "2026-11-23")
        asset_dfs[sym] = df
        funding_map[sym] = fr
        print(f"  {sym}: {len(df)} 6h bars ({df.index[0].date()} -> {df.index[-1].date()}) | {len(fr)} actual 8h funding prints")

    is_df = evaluate_window("IN-SAMPLE (3Y: Bear + Chop + Bull)", asset_dfs, funding_map, cfg, "2021-11-24", "2024-11-23")
    oos_df = evaluate_window("OUT-OF-SAMPLE HOLDOUT (2Y)", asset_dfs, funding_map, cfg, "2024-11-24", "2026-11-23")
    full_df = evaluate_window("FULL 5-YEAR WINDOW", asset_dfs, funding_map, cfg, "2021-11-24", "2026-11-23")

    out_dir = new_run_directory("crypto-idle-margin-5y")
    is_df.to_csv(out_dir / "in_sample_2021_2024.csv", index=False)
    oos_df.to_csv(out_dir / "oos_2024_2026.csv", index=False)
    full_df.to_csv(out_dir / "full_5y_2021_2026.csv", index=False)
    print(f"\nSaved study tables to {out_dir}/")


if __name__ == "__main__":
    main()
