"""
5-Year Broad Crypto Strategy Lab (Top 5 Liquid Majors, 2021-11 -> 2026-11)
==========================================================================
Tests 6 structurally different crypto perp strategy families against unchanged Config C
on cached 5-year Binance USD-M 6h candles + actual historical 8h funding rates ($1,200, whole lots):

  0. Baseline          : Unchanged Config C (6h Donchian(20) + EMA(50/200) + 3.5x ATR)
  1. KELTNER_SQUEEZE   : BB(20,2) inside KC(20,1.5) volatility squeeze breakout + 3.0x ATR trail
  2. SUPERTREND_ADX    : Supertrend(10,3.0) flip + ADX(14)>20 (no EMA200 lag)
  3. TSMOM_VOL_TARGET  : Multi-horizon continuous trend (7d/21d/60d) inverse-vol targeted (CTA classic)
  4. DUAL_MOM_ROTATION : Absolute + Relative Momentum rotation (Long top if >0, Short bottom if <0)
  5. FUNDING_CARRY     : Persistent 7d actual funding-rate carry tilt aligned with EMA(20)
  6. NR7_BREAKOUT      : Narrow-Range-7 volatility contraction breakout + 2.5x ATR trail

Then combines any IS-passing (2021-2024) strategies with unchanged Config C on shared margin
and evaluates on the untouched 2024-2026 OOS holdout + full 5-year window + calendar years.
"""

import argparse
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import resolve_date_window
from portfolio_engine import PortfolioTrade, add_ema_atr_features, run_shared_donchian_portfolio
from research.crypto_idle_margin_study import (
    TOP5_COINS,
    _actual_funding_cashflow,
    _adx,
    _round_lot,
    load_binance_funding,
    load_cached_binance_klines,
)
from research.crypto_trend_csmom_5y import donchian_margin_series, make_cfg
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    n = len(df)
    st_dir = np.ones(n, dtype=int)
    final_u = upper.to_numpy().copy()
    final_l = lower.to_numpy().copy()
    close = df["close"].to_numpy()

    for i in range(1, n):
        if not (math.isfinite(final_u[i]) and math.isfinite(final_l[i])):
            st_dir[i] = st_dir[i - 1]
            continue
        if close[i - 1] <= final_u[i - 1]:
            final_u[i] = min(final_u[i], final_u[i - 1])
        if close[i - 1] >= final_l[i - 1]:
            final_l[i] = max(final_l[i], final_l[i - 1])
        if st_dir[i - 1] == 1:
            st_dir[i] = -1 if close[i] < final_l[i] else 1
        else:
            st_dir[i] = 1 if close[i] > final_u[i] else -1

    line = np.where(st_dir == 1, final_l, final_u)
    return pd.Series(st_dir, index=df.index), pd.Series(line, index=df.index)


def prepare_lab_frames(
    dfs_6h: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in dfs_6h.items():
        f = add_ema_atr_features(df)
        f["ema20"] = f["close"].ewm(span=20, adjust=False).mean()
        f["sma20"] = f["close"].rolling(20).mean()
        f["std20"] = f["close"].rolling(20).std()
        f["bb_up"] = f["sma20"] + 2.0 * f["std20"]
        f["bb_lo"] = f["sma20"] - 2.0 * f["std20"]
        f["kc_up"] = f["ema20"] + 1.5 * f["atr14"]
        f["kc_lo"] = f["ema20"] - 1.5 * f["atr14"]
        f["in_squeeze"] = (f["bb_up"] < f["kc_up"]) & (f["bb_lo"] > f["kc_lo"])
        f["recent_squeeze"] = f["in_squeeze"].rolling(4).max() > 0
        f["adx14"] = _adx(f, 14)
        st_dir, st_line = _supertrend(f, 10, 3.0)
        f["st_dir"] = st_dir
        f["st_line"] = st_line

        f["ret1"] = f["close"].pct_change()
        f["vol28"] = f["ret1"].rolling(28).std() * np.sqrt(4.0 * 365.0)  # annualized vol on 6h bars
        f["ret_7d"] = f["close"].pct_change(28)
        f["ret_21d"] = f["close"].pct_change(84)
        f["ret_60d"] = f["close"].pct_change(240)

        rng = f["high"] - f["low"]
        f["is_nr7"] = rng <= rng.rolling(7).min()
        f["prev_hi"] = f["high"].shift(1)
        f["prev_lo"] = f["low"].shift(1)

        fr = funding_map.get(sym, pd.Series(dtype=float))
        if not fr.empty:
            aligned = fr.reindex(f.index, method="ffill").fillna(0.0)
            f["funding_rate"] = aligned
            f["funding_7d"] = aligned.rolling(28).mean()
        else:
            f["funding_rate"] = 0.0
            f["funding_7d"] = 0.0

        f = f.iloc[250:].dropna()
        f = f[(f.index >= eval_start) & (f.index <= eval_end)]
        if not f.empty:
            out[sym] = f
    return out


def run_event_strategy(
    mode: str,
    prepared: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    risk_pct: float = 0.02,
    atr_trail: float = 3.0,
) -> Tuple[List[PortfolioTrade], pd.Series, pd.Series]:
    """Runs KELTNER_SQUEEZE, SUPERTREND_ADX, FUNDING_CARRY, or NR7_BREAKOUT with actual 8h funding."""
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    lev = cfg.max_leverage or 5.0

    capital = cfg.initial_capital
    peak_eq = capital
    brake = False
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    prev_bar2: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    margin_records: List[Tuple[pd.Timestamp, float]] = []

    def close_pos(sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal capital
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_cost"]
        capital += net
        trades.append(
            PortfolioTrade(
                engine=mode, asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_cost"], net_pnl=net, r_multiple=net / max(pos["risk_rs"], 1e-6),
                exit_reason=reason,
            )
        )
        pos_map[sym] = None

    for ts in all_ts:
        # Update DD brake
        unr_now = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr_now += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_cost"]
        cur_eq = capital + unr_now
        peak_eq = max(peak_eq, cur_eq)
        dd = (peak_eq - cur_eq) / peak_eq if peak_eq > 0 else 0.0
        if cfg.dd_brake_threshold:
            if dd >= cfg.dd_brake_threshold:
                brake = True
            elif dd <= cfg.dd_brake_threshold * 0.5:
                brake = False
        scale = cfg.dd_brake_scale if brake else 1.0
        eff_risk = risk_pct * scale
        eff_max_risk = cfg.max_concurrent_risk_pct * scale

        present = {}
        for sym, df in prepared.items():
            if ts in df.index:
                row = df.loc[ts]
                present[sym] = row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        # Exits & Entries at open using only prior closed bar(s)
        for sym, row in present.items():
            prev = last_bar.get(sym)
            prev2 = prev_bar2.get(sym)
            pos = pos_map[sym]

            if pos is not None and prev is not None:
                is_long = pos["side"] == "LONG"
                if mode == "SUPERTREND_ADX" and ((is_long and prev["st_dir"] == -1) or ((not is_long) and prev["st_dir"] == 1)):
                    close_pos(sym, ts, float(row["open"]), "ST_FLIP", entry_slip)
                    continue
                if mode == "FUNDING_CARRY":
                    f7 = float(prev["funding_7d"])
                    if (is_long and (f7 > 0.00005 or prev["close"] < prev["ema20"])) or ((not is_long) and (f7 < 0.00005 or prev["close"] > prev["ema20"])):
                        close_pos(sym, ts, float(row["open"]), "CARRY_EXIT", entry_slip)
                        continue

            elif pos is None and prev is not None and prev["atr14"] > 0 and capital > 0:
                side = None
                if mode == "KELTNER_SQUEEZE":
                    if prev["recent_squeeze"] and not prev["in_squeeze"]:
                        if prev["close"] > prev["kc_up"] and prev["close"] > prev["ema50"]:
                            side = "LONG"
                        elif prev["close"] < prev["kc_lo"] and prev["close"] < prev["ema50"]:
                            side = "SHORT"
                elif mode == "SUPERTREND_ADX" and prev2 is not None:
                    if prev["st_dir"] == 1 and prev2["st_dir"] == -1 and prev["adx14"] >= 20.0:
                        side = "LONG"
                    elif prev["st_dir"] == -1 and prev2["st_dir"] == 1 and prev["adx14"] >= 20.0:
                        side = "SHORT"
                elif mode == "FUNDING_CARRY":
                    f7 = float(prev["funding_7d"])
                    if f7 <= -0.00005 and prev["close"] > prev["ema20"]:
                        side = "LONG"
                    elif f7 >= 0.00020 and prev["close"] < prev["ema20"]:
                        side = "SHORT"
                elif mode == "NR7_BREAKOUT" and prev2 is not None:
                    if prev2["is_nr7"]:
                        if prev["close"] > prev2["high"] and prev["ema50"] > prev["ema200"]:
                            side = "LONG"
                        elif prev["close"] < prev2["low"] and prev["ema50"] < prev["ema200"]:
                            side = "SHORT"

                if side is not None:
                    active = [p for p in pos_map.values() if p is not None]
                    open_risk = sum(p["risk_pct"] for p in active)
                    if open_risk + eff_risk <= eff_max_risk + 1e-9:
                        open_px = float(row["open"])
                        entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = atr_trail * float(prev["atr14"])
                        alloc_m = sum(p["margin"] for p in active)
                        m_budget = max(capital * cfg.max_margin_utilization - alloc_m, 0.0)
                        raw_qty = min((capital * eff_risk) / sl_dist, (m_budget * lev) / entry_px)
                        qty = _round_lot(sym, raw_qty, cfg)
                        if qty > 0:
                            pos_map[sym] = {
                                "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                                "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                                "risk_rs": qty * sl_dist, "risk_pct": (qty * sl_dist) / capital,
                                "margin": (qty * entry_px) / lev, "funding_cost": 0.0, "last_f": ts,
                            }

        for sym, row in present.items():
            if sym in last_bar:
                prev_bar2[sym] = last_bar[sym]
            last_bar[sym] = row

        # Intrabar stops, trailing, funding
        unr = 0.0
        m_used = 0.0
        for sym, pos in list(pos_map.items()):
            if pos is None:
                continue
            row = present.get(sym)
            if row is not None:
                periods = (ts - pos["last_f"]) // pd.Timedelta(hours=8)
                if periods:
                    pos["funding_cost"] += _actual_funding_cashflow(pos["side"], float(row["close"]) * pos["qty"], float(row["funding_rate"]), periods)
                    pos["last_f"] += periods * pd.Timedelta(hours=8)
                is_long = pos["side"] == "LONG"
                if (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"]):
                    gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                    raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                    close_pos(sym, ts, raw_exit, "GAP_STOP" if gapped else "ATR_TRAIL", cfg.slippage_pct)
                    continue
                trail = atr_trail * float(row["atr14"])
                c_px = float(row["close"])
                pos["sl"] = max(pos["sl"], c_px - trail) if is_long else min(pos["sl"], c_px + trail)
            if sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_cost"]
                m_used += pos["margin"]
        eq_records.append((ts, capital + unr))
        margin_records.append((ts, m_used))

    final_ts = all_ts[-1]
    for sym in list(pos_map.keys()):
        if pos_map[sym] is not None and sym in last_bar:
            close_pos(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    return trades, eq.resample("1D").last().ffill(), pd.Series([v for _, v in margin_records], index=pd.DatetimeIndex([t for t, _ in margin_records]))


def run_tsmom_vol_target(
    prepared: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    target_ann_vol: float = 0.35,
    rebalance_bars: int = 8,  # every 48h
) -> Tuple[List[PortfolioTrade], pd.Series, pd.Series]:
    """
    CTA Multi-Horizon Time-Series Momentum (7d + 21d + 60d signal) sized by inverse volatility.
    Continuous allocation across Top 5 majors with deadband to avoid churn.
    """
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    lev = cfg.max_leverage or 5.0
    capital = cfg.initial_capital
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    margin_records: List[Tuple[pd.Timestamp, float]] = []
    n_coins = len(prepared)

    def close_pos(sym: str, ts: pd.Timestamp, raw_px: float, reason: str) -> None:
        nonlocal capital
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - entry_slip if is_long else 1.0 + entry_slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_cost"]
        capital += net
        trades.append(
            PortfolioTrade(
                engine="TSMOM_VOL", asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_cost"], net_pnl=net, r_multiple=net / max(capital * 0.02, 1.0),
                exit_reason=reason,
            )
        )
        pos_map[sym] = None

    for bar_idx, ts in enumerate(all_ts):
        present = {}
        for sym, df in prepared.items():
            if ts in df.index:
                row = df.loc[ts]
                present[sym] = row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        if bar_idx > 0 and bar_idx % rebalance_bars == 0 and capital > 0:
            # Compute target weight per coin
            targets: Dict[str, Tuple[Optional[str], float]] = {}
            total_notional = 0.0
            for sym in present:
                prev = last_bar.get(sym)
                if prev is None or prev["vol28"] <= 0.05:
                    targets[sym] = (None, 0.0)
                    continue
                sig = (np.sign(prev["ret_7d"]) + np.sign(prev["ret_21d"]) + np.sign(prev["ret_60d"])) / 3.0
                if abs(sig) < 0.34:  # require at least 2 of 3 horizons to agree
                    targets[sym] = (None, 0.0)
                    continue
                side = "LONG" if sig > 0 else "SHORT"
                weight = min((target_ann_vol / float(prev["vol28"])) * abs(sig) / n_coins, 0.60)
                notional = capital * weight
                targets[sym] = (side, notional)
                total_notional += notional

            max_notional = capital * lev * cfg.max_margin_utilization
            scale_down = min(1.0, max_notional / total_notional) if total_notional > 0 else 1.0

            for sym, (side, notional) in targets.items():
                pos = pos_map[sym]
                notional *= scale_down
                if side is None or notional <= 0:
                    if pos is not None:
                        close_pos(sym, ts, float(present[sym]["open"]), "SIGNAL_FLAT")
                    continue
                if pos is not None and pos["side"] != side:
                    close_pos(sym, ts, float(present[sym]["open"]), "SIGNAL_FLIP")
                    pos = None
                if pos is None:
                    open_px = float(present[sym]["open"])
                    entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                    qty = _round_lot(sym, notional / entry_px, cfg)
                    if qty > 0:
                        pos_map[sym] = {
                            "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                            "margin": (qty * entry_px) / lev, "funding_cost": 0.0, "last_f": ts,
                        }

        for sym, row in present.items():
            last_bar[sym] = row

        unr = 0.0
        m_used = 0.0
        for sym, pos in pos_map.items():
            if pos is None:
                continue
            row = present.get(sym)
            if row is not None:
                periods = (ts - pos["last_f"]) // pd.Timedelta(hours=8)
                if periods:
                    pos["funding_cost"] += _actual_funding_cashflow(pos["side"], float(row["close"]) * pos["qty"], float(row["funding_rate"]), periods)
                    pos["last_f"] += periods * pd.Timedelta(hours=8)
            if sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_cost"]
                m_used += pos["margin"]
        eq_records.append((ts, capital + unr))
        margin_records.append((ts, m_used))

    final_ts = all_ts[-1]
    for sym in list(pos_map.keys()):
        if pos_map[sym] is not None and sym in last_bar:
            close_pos(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST")

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    return trades, eq.resample("1D").last().ffill(), pd.Series([v for _, v in margin_records], index=pd.DatetimeIndex([t for t, _ in margin_records]))


def run_dual_momentum_rotation(
    prepared: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    rebalance_bars: int = 16,  # 4 days
    gross_leverage: float = 1.5,
) -> Tuple[List[PortfolioTrade], pd.Series, pd.Series]:
    """
    Absolute + Relative Dual Momentum Rotation across Top 5 majors:
      - Long Top-2 ONLY if 21d return > 0 AND Close > EMA(50)
      - Short Bottom-2 ONLY if 21d return < 0 AND Close < EMA(50)
      - Otherwise cash. Fixes the 2022 bear-market flaw of pure CS-Mom!
    """
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    lev = cfg.max_leverage or 5.0
    capital = cfg.initial_capital
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    margin_records: List[Tuple[pd.Timestamp, float]] = []

    def close_pos(sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal capital
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_cost"]
        capital += net
        trades.append(
            PortfolioTrade(
                engine="DUAL_MOM", asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_cost"], net_pnl=net, r_multiple=net / max(pos["risk_rs"], 1e-6),
                exit_reason=reason,
            )
        )
        pos_map[sym] = None

    for bar_idx, ts in enumerate(all_ts):
        present = {}
        for sym, df in prepared.items():
            if ts in df.index:
                row = df.loc[ts]
                present[sym] = row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        if bar_idx > 0 and bar_idx % rebalance_bars == 0 and capital > 0:
            scores = []
            for sym in present:
                prev = last_bar.get(sym)
                if prev is not None and prev["vol28"] > 0.05 and math.isfinite(prev["ret_21d"]):
                    scores.append((float(prev["ret_21d"]) / float(prev["vol28"]), sym, prev))
            scores.sort(key=lambda x: x[0])
            desired: Dict[str, str] = {}
            for _, sym, prev in scores[-2:]:
                if prev["ret_21d"] > 0 and prev["close"] > prev["ema50"]:
                    desired[sym] = "LONG"
            for _, sym, prev in scores[:2]:
                if sym not in desired and prev["ret_21d"] < 0 and prev["close"] < prev["ema50"]:
                    desired[sym] = "SHORT"

            for sym, pos in list(pos_map.items()):
                if pos is not None and desired.get(sym) != pos["side"] and sym in present:
                    close_pos(sym, ts, float(present[sym]["open"]), "REBALANCE", entry_slip)

            if desired:
                max_gross = min(gross_leverage, lev * cfg.max_margin_utilization)
                leg_notional = (capital * max_gross) / max(len(desired), 2)
                for sym, side in desired.items():
                    if pos_map[sym] is not None or sym not in present:
                        continue
                    open_px = float(present[sym]["open"])
                    prev = last_bar[sym]
                    entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                    qty = _round_lot(sym, leg_notional / entry_px, cfg)
                    if qty > 0:
                        sl_dist = 3.5 * float(prev["atr14"])
                        pos_map[sym] = {
                            "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                            "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                            "risk_rs": qty * sl_dist, "margin": (qty * entry_px) / lev,
                            "funding_cost": 0.0, "last_f": ts,
                        }

        for sym, row in present.items():
            last_bar[sym] = row

        unr = 0.0
        m_used = 0.0
        for sym, pos in list(pos_map.items()):
            if pos is None:
                continue
            row = present.get(sym)
            if row is not None:
                periods = (ts - pos["last_f"]) // pd.Timedelta(hours=8)
                if periods:
                    pos["funding_cost"] += _actual_funding_cashflow(pos["side"], float(row["close"]) * pos["qty"], float(row["funding_rate"]), periods)
                    pos["last_f"] += periods * pd.Timedelta(hours=8)
                is_long = pos["side"] == "LONG"
                if (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"]):
                    gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                    raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                    close_pos(sym, ts, raw_exit, "GAP_STOP" if gapped else "HARD_STOP", cfg.slippage_pct)
                    continue
            if sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_cost"]
                m_used += pos["margin"]
        eq_records.append((ts, capital + unr))
        margin_records.append((ts, m_used))

    final_ts = all_ts[-1]
    for sym in list(pos_map.keys()):
        if pos_map[sym] is not None and sym in last_bar:
            close_pos(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    return trades, eq.resample("1D").last().ffill(), pd.Series([v for _, v in margin_records], index=pd.DatetimeIndex([t for t, _ in margin_records]))


def evaluate_all(
    label: str,
    dfs_6h: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    start_str: str,
    end_str: str,
    capital: float,
    entry_slip: float,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series], List[PortfolioTrade]]:
    _, _, ev_s, ev_e = resolve_date_window("6h", start_date=start_str, end_date=end_str, include_warmup=True)
    cfg = make_cfg(capital, entry_slip)
    prepared = prepare_lab_frames(dfs_6h, funding_map, cfg, ev_s, ev_e)

    base_t, base_eq, _ = run_shared_donchian_portfolio(dfs_6h, set(TOP5_COINS), cfg, eval_start=ev_s, eval_end=ev_e)
    sq_t, sq_eq, sq_m = run_event_strategy("KELTNER_SQUEEZE", prepared, cfg, risk_pct=0.02, atr_trail=3.0)
    st_t, st_eq, st_m = run_event_strategy("SUPERTREND_ADX", prepared, cfg, risk_pct=0.02, atr_trail=3.5)
    fc_t, fc_eq, fc_m = run_event_strategy("FUNDING_CARRY", prepared, cfg, risk_pct=0.015, atr_trail=3.0)
    nr_t, nr_eq, nr_m = run_event_strategy("NR7_BREAKOUT", prepared, cfg, risk_pct=0.02, atr_trail=2.5)
    ts_t, ts_eq, ts_m = run_tsmom_vol_target(prepared, cfg, target_ann_vol=0.35, rebalance_bars=8)
    dm_t, dm_eq, dm_m = run_dual_momentum_rotation(prepared, cfg, rebalance_bars=16, gross_leverage=1.5)

    common = base_eq.index
    def al(s: pd.Series) -> pd.Series:
        return s.reindex(common).ffill().bfill()

    rows = [
        calc_curve_stats("0. Baseline: Unchanged Config C (6h Donchian)", al(base_eq), base_t, capital),
        calc_curve_stats("1. Keltner/BB Squeeze Breakout (3.0x ATR)", al(sq_eq), sq_t, capital),
        calc_curve_stats("2. Supertrend(10,3) + ADX>20 (no EMA200)", al(st_eq), st_t, capital),
        calc_curve_stats("3. CTA TSMOM Vol-Target (7d/21d/60d)", al(ts_eq), ts_t, capital),
        calc_curve_stats("4. Dual-Momentum Rotation (Abs+Rel 21d)", al(dm_eq), dm_t, capital),
        calc_curve_stats("5. Funding-Rate Carry Tilt (7d actual FR)", al(fc_eq), fc_t, capital),
        calc_curve_stats("6. NR7 Volatility Contraction Breakout", al(nr_eq), nr_t, capital),
    ]
    df_out = pd.DataFrame(rows)
    print(f"\n================ {label} ({start_str} -> {end_str}) ================")
    print(df_out.to_string(index=False))

    curves = {
        "ConfigC": al(base_eq), "Squeeze": al(sq_eq), "Supertrend": al(st_eq),
        "TSMOM": al(ts_eq), "DualMom": al(dm_eq), "FundingCarry": al(fc_eq), "NR7": al(nr_eq),
    }
    margins = {
        "Squeeze": sq_m, "Supertrend": st_m, "TSMOM": ts_m, "DualMom": dm_m, "FundingCarry": fc_m, "NR7": nr_m,
    }
    return df_out, curves, margins, base_t


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Year Broad Crypto Strategy Lab (Top 5 Majors)")
    parser.add_argument("--capital", type=float, default=1200.0)
    parser.add_argument("--taker-only", action="store_true")
    args = parser.parse_args()
    entry_slip = 0.0015 if args.taker_only else 0.0004

    dfs_6h = {sym: load_cached_binance_klines(sym, "6h", "2021-11-24", "2026-11-23") for sym in TOP5_COINS}
    funding_map = {sym: load_binance_funding(sym, "2021-08-01", "2026-11-23") for sym in TOP5_COINS}

    is_df, is_curves, _, _ = evaluate_all("IN-SAMPLE (3Y)", dfs_6h, funding_map, "2021-11-24", "2024-11-23", args.capital, entry_slip)
    oos_df, oos_curves, _, _ = evaluate_all("OUT-OF-SAMPLE HOLDOUT (2Y)", dfs_6h, funding_map, "2024-11-24", "2026-11-23", args.capital, entry_slip)
    full_df, full_curves, full_margins, base_t = evaluate_all("FULL 5-YEAR WINDOW", dfs_6h, funding_map, "2021-11-24", "2026-11-23", args.capital, entry_slip)

    print("\n================ 5-YEAR DAILY RETURN CORRELATIONS ================")
    print(pd.DataFrame(full_curves).pct_change().dropna().corr().round(2).to_string())

    # Test spare-margin stacks of Config C + top IS-positive strategies (0.5x scale on spare margin, auditing peak margin <= 75%)
    print("\n================ SPARE-MARGIN MULTI-STRATEGY STACKS (FULL 5Y & OOS) ================")
    stack_rows = []
    for key in ("Squeeze", "Supertrend", "TSMOM", "DualMom", "FundingCarry", "NR7"):
        is_ret = (is_curves[key].iloc[-1] / args.capital - 1.0) * 100.0
        if is_ret <= 5.0:
            continue
        # 1.0x Config C + 0.5x companion (using spare margin)
        comb_5y = full_curves["ConfigC"] + 0.5 * (full_curves[key] - args.capital)
        comb_oos = oos_curves["ConfigC"] + 0.5 * (oos_curves[key] - args.capital)
        m_comp = full_margins[key] * 0.5
        m_base = donchian_margin_series(base_t, m_comp.index, lev=5.0)
        tot_m = m_base + m_comp
        eq_bar = comb_5y.reindex(tot_m.index.normalize()).ffill().bfill()
        pk_mgn = float((tot_m / eq_bar.values).max() * 100.0) if len(tot_m) else 0.0
        st5 = calc_curve_stats(f"5Y: Config C + 0.5x {key} (PkMgn {pk_mgn:.0f}%)", comb_5y, [], args.capital)
        sto = calc_curve_stats(f"OOS: Config C + 0.5x {key}", comb_oos, [], args.capital)
        stack_rows.extend([st5, sto])

    if stack_rows:
        print(pd.DataFrame(stack_rows).to_string(index=False))
    else:
        print("No candidate exceeded +5% in-sample.")

    print("\n================ CALENDAR YEAR RETURNS (%) ================")
    yr_rows = {}
    for name, s in full_curves.items():
        ye = s.resample("YE").last()
        ret = ye.pct_change()
        ret.iloc[0] = ye.iloc[0] / args.capital - 1.0
        yr_rows[name] = (ret * 100.0).round(1)
    yr_df = pd.DataFrame(yr_rows).T
    yr_df.columns = [str(c.year) for c in yr_df.columns]
    print(yr_df.to_string())

    out_dir = new_run_directory("crypto-broad-lab-5y")
    is_df.to_csv(out_dir / "in_sample_2021_2024.csv", index=False)
    oos_df.to_csv(out_dir / "oos_2024_2026.csv", index=False)
    full_df.to_csv(out_dir / "full_5y_2021_2026.csv", index=False)
    pd.DataFrame(full_curves).to_csv(out_dir / "equity_curves_5y.csv")
    print(f"\nSaved lab results to {out_dir}/")


if __name__ == "__main__":
    main()
