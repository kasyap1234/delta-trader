"""
Head-to-Head 4H vs 6H Comparison for Config C and 3-Pod Balanced
================================================================
Compares 4H vs 6H on identical dates, $1,200 USD capital, whole-lot multipliers,
shared 75% margin cap, and both Maker (0.04%) and Taker (0.15%) entry costs:
  - 5-Year Binance USD-M (2021-11-24 -> 2026-11-23) with actual historical 8h funding rates
  - 2-Year Delta Exchange India (2024-11-24 -> 2026-11-23) with actual Delta candles

TSMOM lookbacks & vol annualization are scaled to identical calendar horizons (7d/21d/60d, 48h rebal)
on both 4H (6 bars/day) and 6H (4 bars/day), while Donchian(20) and Supertrend(10,3)+ADX(14) run natively
on the bar interval.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import load_binance, load_delta_candles, resolve_date_window
from portfolio_engine import PortfolioTrade, add_ema_atr_features, run_shared_donchian_portfolio
from research.crypto_broad_strategy_lab_5y import _supertrend
from research.crypto_idle_margin_study import (
    TOP5_COINS,
    _actual_funding_cashflow,
    _adx,
    _round_lot,
    load_binance_funding,
    load_cached_binance_klines,
)
from research.crypto_trend_csmom_5y import make_cfg
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats
from research.validate_3pod_crypto import DELTA_TOP5, load_cached_delta_klines

CACHE_4H = Path(__file__).resolve().parent / "output" / "_cache_4h"


def load_cached_4h_binance(sym: str) -> pd.DataFrame:
    CACHE_4H.mkdir(parents=True, exist_ok=True)
    f = CACHE_4H / f"bin_{sym}_4h_2021-11-24_2026-11-23.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.index.name = "timestamp"
        return df.sort_index()
    # Need >= 360 bars (~65 days) warmup before 2021-11-24 -> start 2021-09-01
    df = load_binance(sym, interval="4h", start_date="2021-09-01", end_date="2026-11-23", include_warmup=False)
    df.to_csv(f)
    return df


def load_cached_4h_delta(sym: str) -> pd.DataFrame:
    CACHE_4H.mkdir(parents=True, exist_ok=True)
    f = CACHE_4H / f"delta_{sym}_4h_730d.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.index.name = "timestamp"
        return df.sort_index()
    df = load_delta_candles(sym, resolution="4h", days=760, include_warmup=True)
    df.to_csv(f)
    return df


def run_3pod_interval(
    dfs: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    interval: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    w_dc: float = 0.40,
    w_st: float = 0.35,
    w_ts: float = 0.35,
    use_actual_funding: bool = True,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    bars_per_day = 6 if interval == "4h" else 4
    rebal_bars = 2 * bars_per_day       # 48h
    b_7d = 7 * bars_per_day
    b_21d = 21 * bars_per_day
    b_60d = 60 * bars_per_day
    warmup = max(cfg.ema_warmup_bars, b_60d + 10)

    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in dfs.items():
        f = add_ema_atr_features(df)
        f["donchian_high"] = f["high"].rolling(cfg.donchian_entry_len).max().shift(1)
        f["donchian_low"] = f["low"].rolling(cfg.donchian_entry_len).min().shift(1)
        f["adx14"] = _adx(f, 14)
        st_dir, _ = _supertrend(f, 10, 3.0)
        f["st_dir"] = st_dir
        f["ret1"] = f["close"].pct_change()
        f["vol7d"] = f["ret1"].rolling(b_7d).std() * np.sqrt(bars_per_day * 365.0)
        f["ret_7d"] = f["close"].pct_change(b_7d)
        f["ret_21d"] = f["close"].pct_change(b_21d)
        f["ret_60d"] = f["close"].pct_change(b_60d)
        fr = funding_map.get(sym, pd.Series(dtype=float))
        f["funding_rate"] = fr.reindex(f.index, method="ffill").fillna(0.0) if not fr.empty else 0.0
        f = f.iloc[warmup:].dropna()
        f = f[(f.index >= eval_start) & (f.index <= eval_end)]
        if not f.empty:
            prepared[sym] = f

    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    if not all_ts:
        return [], pd.Series(dtype=float), {"peak_margin_pct": 0.0}

    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    lev = cfg.max_leverage or 5.0
    n_coins = len(prepared)
    capital = cfg.initial_capital
    peak_eq = capital
    brake = False

    pos_dc: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    pos_st: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    pos_ts: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    prev_bar2: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    peak_mgn = 0.0

    def alloc_m() -> float:
        return sum(p["margin"] for b in (pos_dc, pos_st, pos_ts) for p in b.values() if p is not None)

    def tot_unr() -> float:
        u = 0.0
        for b in (pos_dc, pos_st, pos_ts):
            for sym, p in b.items():
                if p is not None and sym in last_bar:
                    px = float(last_bar[sym]["close"])
                    u += ((px - p["entry"]) * p["qty"] if p["side"] == "LONG" else (p["entry"] - px) * p["qty"]) - p["funding"]
        return u

    def close_leg(book, eng, sym, ts, raw_px, reason, slip):
        nonlocal capital
        p = book[sym]
        if p is None:
            return
        is_long = p["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - p["entry"]) * p["qty"] if is_long else (p["entry"] - exit_px) * p["qty"]
        net = gross - p["funding"]
        capital += net
        trades.append(PortfolioTrade(
            engine=eng, asset=sym, side=p["side"], entry_time=p["entry_time"], exit_time=ts,
            entry_price=p["entry"], exit_price=exit_px, qty=p["qty"], gross_pnl=gross,
            funding_cost=p["funding"], net_pnl=net, r_multiple=net / max(p["risk_rs"], 1e-6), exit_reason=reason,
        ))
        book[sym] = None

    for bar_idx, ts in enumerate(all_ts):
        cur_eq = capital + tot_unr()
        peak_eq = max(peak_eq, cur_eq)
        dd = (peak_eq - cur_eq) / peak_eq if peak_eq > 0 else 0.0
        if cfg.dd_brake_threshold:
            if dd >= cfg.dd_brake_threshold:
                brake = True
            elif dd <= cfg.dd_brake_threshold * 0.5:
                brake = False
        dd_scale = cfg.dd_brake_scale if brake else 1.0

        present = {sym: (df.loc[ts].iloc[-1] if isinstance(df.loc[ts], pd.DataFrame) else df.loc[ts])
                   for sym, df in prepared.items() if ts in df.index}

        # Pod 1: Donchian
        if w_dc > 0 and capital > 0:
            dc_risk = cfg.donchian_risk_per_trade_pct * w_dc * dd_scale
            dc_max = cfg.max_concurrent_risk_pct * w_dc * dd_scale
            for sym, row in present.items():
                prev = last_bar.get(sym)
                p = pos_dc[sym]
                if p is None and prev is not None and prev["atr14"] > 0:
                    side = "LONG" if (prev["ema50"] > prev["ema200"] and prev["close"] > prev["donchian_high"]) else (
                        "SHORT" if (prev["ema50"] < prev["ema200"] and prev["close"] < prev["donchian_low"]) else None
                    )
                    if side and sum(x["risk_pct"] for x in pos_dc.values() if x) + dc_risk <= dc_max + 1e-9:
                        epx = float(row["open"]) * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sld = cfg.atr_trail_mult * float(prev["atr14"])
                        fm = max(capital * cfg.max_margin_utilization - alloc_m(), 0.0)
                        qty = _round_lot(sym, min((capital * dc_risk) / sld, (fm * lev) / epx), cfg)
                        if qty > 0:
                            pos_dc[sym] = {
                                "side": side, "entry_time": ts, "entry": epx, "init_entry": epx, "init_sl_dist": sld,
                                "qty": qty, "margin": (qty * epx) / lev, "risk_rs": qty * sld, "risk_pct": (qty * sld) / capital,
                                "sl": epx - sld if side == "LONG" else epx + sld, "funding": 0.0, "last_f": ts, "pyramided": False,
                            }
                elif p is not None and cfg.pyramid_add_fraction > 0 and not p["pyramided"] and prev is not None:
                    is_l = p["side"] == "LONG"
                    opx = float(row["open"])
                    if ((opx - p["init_entry"]) if is_l else (p["init_entry"] - opx)) >= cfg.pyramid_trigger_r * p["init_sl_dist"] and (
                        (p["sl"] >= p["init_entry"]) if is_l else (p["sl"] <= p["init_entry"])
                    ):
                        aepx = opx * (1.0 + entry_slip if is_l else 1.0 - entry_slip)
                        fm = max(capital * cfg.max_margin_utilization - alloc_m(), 0.0)
                        aq = _round_lot(sym, min(p["qty"] * cfg.pyramid_add_fraction, (fm * lev) / aepx), cfg)
                        if aq > 0:
                            nq = p["qty"] + aq
                            p["entry"] = (p["entry"] * p["qty"] + aepx * aq) / nq
                            p["qty"] = nq
                            p["margin"] += (aq * aepx) / lev
                            p["pyramided"] = True

        # Pod 2: Supertrend + ADX
        if w_st > 0 and capital > 0:
            st_risk = 0.02 * w_st * dd_scale
            st_max = 0.08 * w_st * dd_scale
            for sym, row in present.items():
                prev, prev2 = last_bar.get(sym), prev_bar2.get(sym)
                p = pos_st[sym]
                if p is not None and prev is not None:
                    is_l = p["side"] == "LONG"
                    if (is_l and prev["st_dir"] == -1) or ((not is_l) and prev["st_dir"] == 1):
                        close_leg(pos_st, "SUPERTREND", sym, ts, float(row["open"]), "ST_FLIP", entry_slip)
                        continue
                elif p is None and prev is not None and prev2 is not None and prev["atr14"] > 0:
                    side = "LONG" if (prev["st_dir"] == 1 and prev2["st_dir"] == -1 and prev["adx14"] >= 20.0) else (
                        "SHORT" if (prev["st_dir"] == -1 and prev2["st_dir"] == 1 and prev["adx14"] >= 20.0) else None
                    )
                    if side and sum(x["risk_pct"] for x in pos_st.values() if x) + st_risk <= st_max + 1e-9:
                        epx = float(row["open"]) * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sld = 3.5 * float(prev["atr14"])
                        fm = max(capital * cfg.max_margin_utilization - alloc_m(), 0.0)
                        qty = _round_lot(sym, min((capital * st_risk) / sld, (fm * lev) / epx), cfg)
                        if qty > 0:
                            pos_st[sym] = {
                                "side": side, "entry_time": ts, "entry": epx, "qty": qty, "margin": (qty * epx) / lev,
                                "risk_rs": qty * sld, "risk_pct": (qty * sld) / capital,
                                "sl": epx - sld if side == "LONG" else epx + sld, "funding": 0.0, "last_f": ts,
                            }

        # Pod 3: CTA TSMOM
        if w_ts > 0 and bar_idx > 0 and bar_idx % rebal_bars == 0 and capital > 0:
            tvol = 0.35 * w_ts * dd_scale
            for sym, row in present.items():
                prev = last_bar.get(sym)
                p = pos_ts[sym]
                if prev is None or prev["vol7d"] <= 0.05:
                    if p is not None:
                        close_leg(pos_ts, "TSMOM", sym, ts, float(row["open"]), "SIGNAL_FLAT", entry_slip)
                    continue
                sig = (np.sign(prev["ret_7d"]) + np.sign(prev["ret_21d"]) + np.sign(prev["ret_60d"])) / 3.0
                if abs(sig) < 0.34:
                    if p is not None:
                        close_leg(pos_ts, "TSMOM", sym, ts, float(row["open"]), "SIGNAL_FLAT", entry_slip)
                    continue
                side = "LONG" if sig > 0 else "SHORT"
                if p is not None and p["side"] != side:
                    close_leg(pos_ts, "TSMOM", sym, ts, float(row["open"]), "SIGNAL_FLIP", entry_slip)
                    p = None
                if p is None:
                    w = min((tvol / float(prev["vol7d"])) * abs(sig) / n_coins, 0.40 * w_ts)
                    fm = max(capital * cfg.max_margin_utilization - alloc_m(), 0.0)
                    epx = float(row["open"]) * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                    qty = _round_lot(sym, min(capital * w, fm * lev) / epx, cfg)
                    if qty > 0:
                        pos_ts[sym] = {
                            "side": side, "entry_time": ts, "entry": epx, "qty": qty, "margin": (qty * epx) / lev,
                            "risk_rs": max(capital * 0.01, 1.0), "sl": None, "funding": 0.0, "last_f": ts,
                        }

        for sym, row in present.items():
            if sym in last_bar:
                prev_bar2[sym] = last_bar[sym]
            last_bar[sym] = row

        for book, eng, tr_m in ((pos_dc, "DONCHIAN", cfg.atr_trail_mult), (pos_st, "SUPERTREND", 3.5), (pos_ts, "TSMOM", None)):
            for sym, p in list(book.items()):
                if p is None or sym not in present:
                    continue
                row = present[sym]
                periods = (ts - p["last_f"]) // pd.Timedelta(hours=8)
                if periods:
                    notional = float(row["close"]) * p["qty"]
                    p["funding"] += (
                        _actual_funding_cashflow(p["side"], notional, float(row["funding_rate"]), periods)
                        if use_actual_funding else notional * cfg.crypto_funding_8h_pct * periods
                    )
                    p["last_f"] += periods * pd.Timedelta(hours=8)
                if tr_m is not None and p["sl"] is not None:
                    is_l = p["side"] == "LONG"
                    if (row["low"] <= p["sl"]) if is_l else (row["high"] >= p["sl"]):
                        gap = (row["open"] < p["sl"]) if is_l else (row["open"] > p["sl"])
                        close_leg(book, eng, sym, ts, float(row["open"]) if (cfg.gap_aware_stops and gap) else p["sl"],
                                  "GAP_STOP" if gap else "ATR_TRAIL", cfg.slippage_pct)
                        continue
                    cpx = float(row["close"])
                    tr = tr_m * float(row["atr14"])
                    p["sl"] = max(p["sl"], cpx - tr) if is_l else min(p["sl"], cpx + tr)

        eq_now = capital + tot_unr()
        if eq_now > 0:
            peak_mgn = max(peak_mgn, alloc_m() / eq_now * 100.0)
        eq_records.append((ts, eq_now))

    final_ts = all_ts[-1]
    for book, eng in ((pos_dc, "DONCHIAN"), (pos_st, "SUPERTREND"), (pos_ts, "TSMOM")):
        for sym in list(book.keys()):
            if book[sym] is not None and sym in last_bar:
                close_leg(book, eng, sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    return trades, eq.resample("1D").last().ffill(), {"peak_margin_pct": round(peak_mgn, 1)}


def run_comparison(title, dfs_4h, dfs_6h, fr_map, start_str, end_str, use_actual_fr):
    _, _, ev_s, ev_e = resolve_date_window("6h", start_date=start_str, end_date=end_str, include_warmup=True)
    rows = []
    for slip_lbl, eslip in (("Maker 0.04%", 0.0004), ("Taker 0.15%", 0.0015)):
        for tf, dfs in (("6H", dfs_6h), ("4H", dfs_4h)):
            cfg = make_cfg(1200.0, eslip)
            cfg.crypto_assets = list(dfs.keys())
            bt, beq, _ = run_shared_donchian_portfolio(dfs, set(dfs.keys()), cfg, eval_start=ev_s, eval_end=ev_e)
            pt, peq, pdg = run_3pod_interval(dfs, fr_map, cfg, tf.lower(), ev_s, ev_e, 0.40, 0.35, 0.35, use_actual_fr)
            rows.append(calc_curve_stats(f"[{slip_lbl}] {tf} Config C (100%)", beq, bt, 1200.0))
            rows.append(calc_curve_stats(f"[{slip_lbl}] {tf} 3-Pod Balanced (PkMgn {pdg['peak_margin_pct']}%)", peq, pt, 1200.0))
    df = pd.DataFrame(rows)
    print(f"\n================ {title} ({start_str} -> {end_str}) ================")
    print(df.to_string(index=False))
    return df


def main():
    print("Loading 4H + 6H Binance 5Y and Delta 2Y datasets...")
    bin_6h = {s: load_cached_binance_klines(s, "6h", "2021-11-24", "2026-11-23") for s in TOP5_COINS}
    bin_4h = {s: load_cached_4h_binance(s) for s in TOP5_COINS}
    fr_bin = {s: load_binance_funding(s, "2021-08-01", "2026-11-23") for s in TOP5_COINS}

    del_6h = {s: load_cached_delta_klines(s, "6h", 730) for s in DELTA_TOP5}
    del_4h = {s: load_cached_4h_delta(s) for s in DELTA_TOP5}

    d5 = run_comparison("BINANCE 5-YEAR FULL (Actual 8h Funding)", bin_4h, bin_6h, fr_bin, "2021-11-24", "2026-11-23", True)
    doos = run_comparison("BINANCE 2-YEAR OOS (Actual 8h Funding)", bin_4h, bin_6h, fr_bin, "2024-11-24", "2026-11-23", True)
    ddel = run_comparison("DELTA EXCHANGE INDIA 2-YEAR (Flat 0.01%/8h)", del_4h, del_6h, {}, "2024-11-24", "2026-11-23", False)

    out = new_run_directory("compare-4h-vs-6h-3pod")
    d5.to_csv(out / "binance_5y.csv", index=False)
    doos.to_csv(out / "binance_oos_2y.csv", index=False)
    ddel.to_csv(out / "delta_2y.csv", index=False)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
