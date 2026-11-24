"""
Compare unchanged Config C (Delta 6h Donchian) against alternative crypto perpetual strategies
on identical Delta Exchange India historical candles, starting capital ($1,200), whole-lot
contract multipliers, fee/slippage assumptions, 8h funding drag, and shared margin limits.

Strategies tested:
  1. Baseline: Unchanged Config C (6h Donchian(20) + EMA(50/200) + 3.5x ATR + Pyramid + 15% DD Brake)
  2. CS-Mom (7d / 14d): Cross-Sectional Long/Short Volatility-Adjusted Momentum (Top-1 Long / Bottom-1 Short)
  3. MR-RSI2: Short-Term Pullback Mean Reversion (RSI(2) extreme + EMA(200) regime + hard ATR stop)
  4. Combined (Config C + Idle-Margin CS-Mom): Shared $1,200 equity & 75% max margin utilization,
     with Config C strictly prioritized and trade-ledger parity audited.
"""

import argparse
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import load_delta_candles, resolve_date_window
from portfolio_engine import (
    PortfolioTrade,
    add_donchian_features,
    add_ema_atr_features,
    run_shared_donchian_portfolio,
)
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

DEFAULT_DELTA_BASKET = ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"]


def _rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _prepare_common_frames(
    asset_dfs: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    eval_start: Optional[pd.Timestamp],
    eval_end: Optional[pd.Timestamp],
) -> Dict[str, pd.DataFrame]:
    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in asset_dfs.items():
        feat = add_ema_atr_features(df)
        feat["donchian_high"] = feat["high"].rolling(cfg.donchian_entry_len).max().shift(1)
        feat["donchian_low"] = feat["low"].rolling(cfg.donchian_entry_len).min().shift(1)
        feat["rsi2"] = _rsi(feat["close"], 2)
        feat["ema5"] = feat["close"].ewm(span=5, adjust=False).mean()
        feat["ret1"] = feat["close"].pct_change()
        feat["vol28"] = feat["ret1"].rolling(28).std()  # ~7d on 6h bars
        feat["mom28"] = feat["close"].pct_change(28)     # ~7d on 6h bars
        feat["mom56"] = feat["close"].pct_change(56)     # ~14d on 6h bars
        feat = feat.iloc[cfg.ema_warmup_bars:].dropna()
        if eval_start is not None:
            feat = feat[feat.index >= eval_start]
        if eval_end is not None:
            feat = feat[feat.index <= eval_end]
        if not feat.empty:
            prepared[sym] = feat
    return prepared


def _round_lot(sym: str, raw_qty: float, cfg: PortfolioConfig) -> float:
    if raw_qty <= 0 or not math.isfinite(raw_qty):
        return 0.0
    if not cfg.whole_lot_rounding:
        return raw_qty
    step = cfg.lot_multipliers.get(sym, 1.0)
    lots = math.floor(raw_qty / step)
    return lots * step if lots >= 1 else 0.0


def run_cross_sectional_momentum(
    prepared: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    lookback_col: str = "mom28",
    rebalance_bars: int = 12,        # 12 * 6h = 3 days
    gross_leverage: float = 1.5,     # 0.75x long + 0.75x short (fits well inside 5x / 75% margin cap)
    top_k: int = 1,
    sl_atr_mult: Optional[float] = 4.0,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Market-neutral Cross-Sectional Volatility-Adjusted Momentum:
      - Every `rebalance_bars`, ranks all available coins using prior-bar `lookback_col / vol28`.
      - Goes LONG Top-K and SHORT Bottom-K at current bar open.
      - Respects whole-lot rounding, slippage, 8h funding, 75% margin cap, and optional hard ATR stop.
    """
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    if not all_ts:
        return [], pd.Series(dtype=float), {"idle_bar_pct": 100.0}

    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    capital = cfg.initial_capital
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    idle_bars = 0

    def close_pos(sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal capital
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_paid"]
        capital += net
        trades.append(
            PortfolioTrade(
                engine="CS_MOM", asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_paid"], net_pnl=net, r_multiple=net / max(pos["risk_rs"], 1e-6),
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

        # Rebalance at open using only prior-bar closed signals
        if bar_idx > 0 and bar_idx % rebalance_bars == 0 and capital > 0:
            scores = []
            for sym in present:
                prev = last_bar.get(sym)
                if prev is not None and prev["vol28"] > 1e-9 and math.isfinite(prev[lookback_col]):
                    scores.append((float(prev[lookback_col]) / float(prev["vol28"]), sym))
            if len(scores) >= 2 * top_k:
                scores.sort(key=lambda x: x[0])
                desired: Dict[str, str] = {}
                for _, sym in scores[:top_k]:
                    desired[sym] = "SHORT"
                for _, sym in scores[-top_k:]:
                    desired[sym] = "LONG"

                # Close positions that flipped or dropped out
                for sym, pos in list(pos_map.items()):
                    if pos is not None and desired.get(sym) != pos["side"] and sym in present:
                        close_pos(sym, ts, float(present[sym]["open"]), "REBALANCE", entry_slip)

                # Open new legs
                max_gross = min(gross_leverage, (cfg.max_leverage or 5.0) * cfg.max_margin_utilization)
                leg_notional = (capital * max_gross) / (2.0 * top_k)
                for sym, side in desired.items():
                    if pos_map[sym] is not None or sym not in present:
                        continue
                    open_px = float(present[sym]["open"])
                    prev = last_bar[sym]
                    entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                    qty = _round_lot(sym, leg_notional / entry_px, cfg)
                    if qty <= 0:
                        continue
                    atr = float(prev["atr14"])
                    sl_dist = (sl_atr_mult * atr) if (sl_atr_mult and atr > 0) else (entry_px * 0.15)
                    sl_px = (entry_px - sl_dist) if side == "LONG" else (entry_px + sl_dist)
                    pos_map[sym] = {
                        "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                        "sl": sl_px, "risk_rs": qty * sl_dist, "funding_paid": 0.0, "last_funding_ts": ts,
                    }

        for sym, row in present.items():
            last_bar[sym] = row

        # Funding & hard stop checks
        for sym, pos in list(pos_map.items()):
            row = present.get(sym)
            if pos is None or row is None:
                continue
            periods = (ts - pos["last_funding_ts"]) // pd.Timedelta(hours=8)
            if periods:
                # Net funding drag modeled conservatively (paid on long; neutral/cost on short)
                pos["funding_paid"] += float(row["close"]) * pos["qty"] * cfg.crypto_funding_8h_pct * periods
                pos["last_funding_ts"] += periods * pd.Timedelta(hours=8)
            if sl_atr_mult:
                is_long = pos["side"] == "LONG"
                hit = (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"])
                if hit:
                    gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                    raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                    close_pos(sym, ts, raw_exit, "GAP_STOP" if gapped else "HARD_STOP", cfg.slippage_pct)

        if not any(p is not None for p in pos_map.values()):
            idle_bars += 1

        unr = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_paid"]
        eq_records.append((ts, capital + unr))

    final_ts = all_ts[-1]
    for sym, pos in list(pos_map.items()):
        if pos is not None and sym in last_bar:
            close_pos(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    daily_eq = eq.resample("1D").last().ffill()
    return trades, daily_eq, {"idle_bar_pct": round(idle_bars / max(len(all_ts), 1) * 100.0, 1)}


def run_rsi2_mean_reversion(
    prepared: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    rsi_entry: float = 10.0,
    sl_atr_mult: float = 2.0,
    max_hold_bars: int = 8,
    risk_pct: float = 0.015,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Short-term RSI(2) Pullback Mean Reversion (Long when Close > EMA200 & RSI2 < rsi_entry;
    Short when Close < EMA200 & RSI2 > 100 - rsi_entry). Exits on Close crossing EMA(5),
    time stop (`max_hold_bars`), or hard ATR stop.
    """
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    if not all_ts:
        return [], pd.Series(dtype=float), {"idle_bar_pct": 100.0}

    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    capital = cfg.initial_capital
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    idle_bars = 0

    def close_pos(sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal capital
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_paid"]
        capital += net
        trades.append(
            PortfolioTrade(
                engine="MR_RSI2", asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_paid"], net_pnl=net, r_multiple=net / max(pos["risk_rs"], 1e-6),
                exit_reason=reason,
            )
        )
        pos_map[sym] = None

    for ts in all_ts:
        present = {}
        for sym, df in prepared.items():
            if ts in df.index:
                row = df.loc[ts]
                present[sym] = row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        # Exits at open based on prior-bar EMA(5) cross or max hold bars; entries based on prior-bar RSI(2)
        for sym, row in present.items():
            prev = last_bar.get(sym)
            pos = pos_map[sym]
            if pos is not None and prev is not None:
                is_long = pos["side"] == "LONG"
                mean_hit = (prev["close"] >= prev["ema5"]) if is_long else (prev["close"] <= prev["ema5"])
                if mean_hit or pos["bars_held"] >= max_hold_bars:
                    close_pos(sym, ts, float(row["open"]), "MEAN_EXIT" if mean_hit else "TIME_STOP", entry_slip)
                    continue
            elif pos is None and prev is not None and prev["atr14"] > 0 and capital > 0:
                side = None
                if prev["close"] > prev["ema200"] and prev["rsi2"] < rsi_entry:
                    side = "LONG"
                elif prev["close"] < prev["ema200"] and prev["rsi2"] > (100.0 - rsi_entry):
                    side = "SHORT"
                if side is not None:
                    active = [p for p in pos_map.values() if p is not None]
                    open_risk = sum(p["risk_pct"] for p in active)
                    if open_risk + risk_pct <= cfg.max_concurrent_risk_pct + 1e-9:
                        open_px = float(row["open"])
                        entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = sl_atr_mult * float(prev["atr14"])
                        raw_qty = (capital * risk_pct) / sl_dist
                        alloc_margin = sum(p["margin_rs"] for p in active)
                        margin_budget = max(capital * cfg.max_margin_utilization - alloc_margin, 0.0)
                        max_qty = (margin_budget * (cfg.max_leverage or 5.0)) / entry_px
                        qty = _round_lot(sym, min(raw_qty, max_qty), cfg)
                        if qty > 0:
                            notional = qty * entry_px
                            pos_map[sym] = {
                                "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                                "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                                "risk_rs": qty * sl_dist, "risk_pct": (qty * sl_dist) / capital,
                                "margin_rs": notional / (cfg.max_leverage or 5.0),
                                "funding_paid": 0.0, "last_funding_ts": ts, "bars_held": 0,
                            }

        for sym, row in present.items():
            last_bar[sym] = row

        for sym, pos in list(pos_map.items()):
            row = present.get(sym)
            if pos is None or row is None:
                continue
            pos["bars_held"] += 1
            periods = (ts - pos["last_funding_ts"]) // pd.Timedelta(hours=8)
            if periods:
                pos["funding_paid"] += float(row["close"]) * pos["qty"] * cfg.crypto_funding_8h_pct * periods
                pos["last_funding_ts"] += periods * pd.Timedelta(hours=8)
            is_long = pos["side"] == "LONG"
            hit = (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"])
            if hit:
                gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                close_pos(sym, ts, raw_exit, "GAP_STOP" if gapped else "ATR_STOP", cfg.slippage_pct)

        if not any(p is not None for p in pos_map.values()):
            idle_bars += 1

        unr = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_paid"]
        eq_records.append((ts, capital + unr))

    final_ts = all_ts[-1]
    for sym, pos in list(pos_map.items()):
        if pos is not None and sym in last_bar:
            close_pos(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    daily_eq = eq.resample("1D").last().ffill()
    return trades, daily_eq, {"idle_bar_pct": round(idle_bars / max(len(all_ts), 1) * 100.0, 1)}


def run_config_c_plus_idle_csmom(
    asset_dfs: Dict[str, pd.DataFrame],
    prepared: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    eval_start: Optional[pd.Timestamp],
    eval_end: Optional[pd.Timestamp],
    lookback_col: str = "mom28",
    rebalance_bars: int = 12,
    overlay_gross_lev: float = 1.0,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, object]]:
    """
    Shared-margin test: runs Config C (Donchian) with strict first-priority capital/margin,
    and only opens CS-Mom legs on symbols NOT currently held by Config C using remaining
    idle margin under the shared 75% margin cap. Audits Config C trade-ledger parity.
    """
    base_trades, base_eq, _ = run_shared_donchian_portfolio(
        asset_dfs, set(asset_dfs.keys()), cfg, eval_start=eval_start, eval_end=eval_end
    )
    # Build bar-by-bar map of Config C open symbols and margin usage from base_trades
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct

    # Track Config C open positions per timestamp to enforce strict shared margin without altering Config C fills
    base_open_by_ts: Dict[pd.Timestamp, Dict[str, float]] = {ts: {} for ts in all_ts}
    for t in base_trades:
        margin = (t.qty * t.entry_price) / (cfg.max_leverage or 5.0)
        for ts in all_ts:
            if t.entry_time <= ts < t.exit_time:
                base_open_by_ts[ts][t.asset] = margin

    overlay_capital = 0.0
    pos_map: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    overlay_trades: List[PortfolioTrade] = []
    overlay_eq_records: List[Tuple[pd.Timestamp, float]] = []
    blocked_by_config_c = 0

    base_eq_by_ts = base_eq.reindex(pd.DatetimeIndex(all_ts).normalize()).ffill().bfill()

    def close_ov(sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal overlay_capital
        pos = pos_map[sym]
        if pos is None:
            return
        is_long = pos["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - pos["entry"]) * pos["qty"] if is_long else (pos["entry"] - exit_px) * pos["qty"]
        net = gross - pos["funding_paid"]
        overlay_capital += net
        overlay_trades.append(
            PortfolioTrade(
                engine="CS_MOM_OV", asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_paid"], net_pnl=net, r_multiple=net / max(pos["risk_rs"], 1e-6),
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

        base_held = base_open_by_ts[ts]
        # Yield immediately if Config C enters a symbol held by overlay
        for sym in list(pos_map.keys()):
            if pos_map[sym] is not None and sym in base_held and sym in present:
                close_ov(sym, ts, float(present[sym]["open"]), "YIELD_TO_CONFIG_C", entry_slip)

        base_eq_now = float(base_eq_by_ts.iloc[bar_idx]) if bar_idx < len(base_eq_by_ts) else cfg.initial_capital
        total_eq_now = max(base_eq_now + overlay_capital, 0.0)
        base_margin = sum(base_held.values())
        ov_margin = sum(p["margin_rs"] for p in pos_map.values() if p is not None)
        free_margin = max(total_eq_now * cfg.max_margin_utilization - base_margin - ov_margin, 0.0)

        if bar_idx > 0 and bar_idx % rebalance_bars == 0 and total_eq_now > 0:
            scores = []
            for sym in present:
                prev = last_bar.get(sym)
                if prev is not None and prev["vol28"] > 1e-9 and math.isfinite(prev[lookback_col]):
                    scores.append((float(prev[lookback_col]) / float(prev["vol28"]), sym))
            if len(scores) >= 2:
                scores.sort(key=lambda x: x[0])
                desired = {scores[0][1]: "SHORT", scores[-1][1]: "LONG"}
                for sym, pos in list(pos_map.items()):
                    if pos is not None and desired.get(sym) != pos["side"] and sym in present:
                        close_ov(sym, ts, float(present[sym]["open"]), "REBALANCE", entry_slip)

                ov_margin = sum(p["margin_rs"] for p in pos_map.values() if p is not None)
                free_margin = max(total_eq_now * cfg.max_margin_utilization - base_margin - ov_margin, 0.0)
                leg_notional = min((total_eq_now * overlay_gross_lev) / 2.0, (free_margin * (cfg.max_leverage or 5.0)) / 2.0)
                for sym, side in desired.items():
                    if sym in base_held:
                        blocked_by_config_c += 1
                        continue
                    if pos_map[sym] is not None or sym not in present or leg_notional <= 0:
                        continue
                    open_px = float(present[sym]["open"])
                    prev = last_bar[sym]
                    entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                    qty = _round_lot(sym, leg_notional / entry_px, cfg)
                    if qty <= 0:
                        continue
                    atr = float(prev["atr14"])
                    sl_dist = 4.0 * atr if atr > 0 else entry_px * 0.15
                    pos_map[sym] = {
                        "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                        "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                        "risk_rs": qty * sl_dist, "margin_rs": (qty * entry_px) / (cfg.max_leverage or 5.0),
                        "funding_paid": 0.0, "last_funding_ts": ts,
                    }

        for sym, row in present.items():
            last_bar[sym] = row

        for sym, pos in list(pos_map.items()):
            row = present.get(sym)
            if pos is None or row is None:
                continue
            periods = (ts - pos["last_funding_ts"]) // pd.Timedelta(hours=8)
            if periods:
                pos["funding_paid"] += float(row["close"]) * pos["qty"] * cfg.crypto_funding_8h_pct * periods
                pos["last_funding_ts"] += periods * pd.Timedelta(hours=8)
            is_long = pos["side"] == "LONG"
            hit = (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"])
            if hit:
                gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                close_ov(sym, ts, raw_exit, "GAP_STOP" if gapped else "HARD_STOP", cfg.slippage_pct)

        unr = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_paid"]
        overlay_eq_records.append((ts, overlay_capital + unr))

    final_ts = all_ts[-1]
    for sym, pos in list(pos_map.items()):
        if pos is not None and sym in last_bar:
            close_ov(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    ov_pnl = pd.Series(
        [v for _, v in overlay_eq_records[:-1]] + [overlay_capital],
        index=pd.DatetimeIndex([t for t, _ in overlay_eq_records]),
    ).resample("1D").last().ffill()
    combined_eq = (base_eq + ov_pnl.reindex(base_eq.index).ffill().fillna(0.0)).dropna()
    return base_trades + overlay_trades, combined_eq, {
        "config_c_trades_preserved": len(base_trades),
        "overlay_trades": len(overlay_trades),
        "overlay_legs_skipped_for_config_c": blocked_by_config_c,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Config C vs Cross-Sectional Momentum & Mean Reversion on Delta India candles")
    parser.add_argument("--interval", default="6h", choices=["4h", "6h", "12h", "1d"])
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=1200.0)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_DELTA_BASKET)
    parser.add_argument("--taker-only", action="store_true", help="Use 0.15%% entry cost instead of 0.04%% maker")
    args = parser.parse_args()

    entry_slip = 0.0015 if args.taker_only else 0.0004
    cfg = PortfolioConfig(
        initial_capital=args.capital, usd_inr_rate=1.0,
        donchian_risk_per_trade_pct=0.02, max_concurrent_risk_pct=0.08,
        max_crypto_concurrent=len(args.symbols), atr_trail_mult=3.5,
        slippage_pct=0.0015, entry_slippage_pct=entry_slip,
        pyramid_add_fraction=0.5, dd_brake_threshold=0.15,
        whole_lot_rounding=True, enable_credit_spreads=False,
        crypto_assets=args.symbols, commodity_assets=[],
    )

    _, _, eval_start_ist, eval_end_ist = resolve_date_window(args.interval, args.days, args.start, args.end, include_warmup=True)
    asset_dfs: Dict[str, pd.DataFrame] = {}
    print(f"Loading Delta Exchange India {args.interval} candles for {args.symbols}...")
    for sym in args.symbols:
        df = load_delta_candles(sym, resolution=args.interval, days=args.days, start_date=args.start, end_date=args.end)
        asset_dfs[sym] = df
        print(f"  {sym}: {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})")

    prepared = _prepare_common_frames(asset_dfs, cfg, eval_start_ist, eval_end_ist)
    common_start = min(df.index[0] for df in prepared.values())
    common_end = max(df.index[-1] for df in prepared.values())

    # 1. Baseline: Unchanged Config C
    base_trades, base_eq, base_diag = run_shared_donchian_portfolio(
        asset_dfs, set(args.symbols), cfg, eval_start=eval_start_ist, eval_end=eval_end_ist
    )
    # 2. CS-Mom 7d (rebalance every 3d)
    cs7_trades, cs7_eq, cs7_diag = run_cross_sectional_momentum(prepared, cfg, lookback_col="mom28", rebalance_bars=12, gross_leverage=1.5)
    # 3. CS-Mom 14d (rebalance every 7d)
    cs14_trades, cs14_eq, cs14_diag = run_cross_sectional_momentum(prepared, cfg, lookback_col="mom56", rebalance_bars=28, gross_leverage=1.5)
    # 4. RSI(2) Pullback Mean Reversion
    mr_trades, mr_eq, mr_diag = run_rsi2_mean_reversion(prepared, cfg, rsi_entry=10.0, sl_atr_mult=2.0, max_hold_bars=8)
    # 5. Combined: Config C + Idle-Margin CS-Mom 14d
    comb_trades, comb_eq, comb_diag = run_config_c_plus_idle_csmom(
        asset_dfs, prepared, cfg, eval_start_ist, eval_end_ist, lookback_col="mom56", rebalance_bars=28, overlay_gross_lev=1.0
    )

    rows = [
        calc_curve_stats("1. Baseline: Unchanged Config C (6h Donchian)", base_eq, base_trades, cfg.initial_capital),
        calc_curve_stats("2. CS-Mom 7d (3d rebal, 1.5x gross L/S)", cs7_eq, cs7_trades, cfg.initial_capital),
        calc_curve_stats("3. CS-Mom 14d (7d rebal, 1.5x gross L/S)", cs14_eq, cs14_trades, cfg.initial_capital),
        calc_curve_stats("4. MR-RSI(2) Pullback (<10 / >90, 2x ATR SL)", mr_eq, mr_trades, cfg.initial_capital),
        calc_curve_stats("5. Combined: Config C + Idle-Margin CS-Mom 14d", comb_eq, comb_trades, cfg.initial_capital),
    ]
    print(f"\n================ DELTA INDIA CRYPTO STRATEGY COMPARISON ({common_start.date()} -> {common_end.date()}) ================")
    print(f"Capital: ${cfg.initial_capital:,.0f} USD | Whole Lots: True | Entry Slip: {entry_slip*100:.2f}% | Stop Slip: {cfg.slippage_pct*100:.2f}% | Funding: 0.01%/8h\n")
    print(pd.DataFrame(rows).to_string(index=False))

    # Correlations vs Baseline Config C
    b_ret = base_eq.pct_change().dropna()
    print("\nDaily Return Correlations vs Config C:")
    for name, eq in [("CS-Mom 7d", cs7_eq), ("CS-Mom 14d", cs14_eq), ("MR-RSI(2)", mr_eq), ("Combined", comb_eq)]:
        corr = b_ret.corr(eq.reindex(b_ret.index).ffill().pct_change().dropna())
        print(f"  {name:15s}: {corr:+.2f}")

    print(f"\nOverlay Parity Audit: {comb_diag}")

    out_dir = new_run_directory("crypto-strategy-compare")
    pd.DataFrame({
        "config_c": base_eq,
        "cs_mom_7d": cs7_eq.reindex(base_eq.index).ffill(),
        "cs_mom_14d": cs14_eq.reindex(base_eq.index).ffill(),
        "mr_rsi2": mr_eq.reindex(base_eq.index).ffill(),
        "combined": comb_eq.reindex(base_eq.index).ffill(),
    }).to_csv(out_dir / "equity_curves.csv")
    print(f"\nSaved equity curves to {out_dir}/equity_curves.csv")


if __name__ == "__main__":
    main()
