"""
Single-Account Shared-Margin Validation: 3-Pod Balanced vs. 4-Pod (+ Funding & Volume Squeeze)
==============================================================================================
Tests inside ONE shared $1,200 USD account (whole lots, shared 75% max margin cap,
portfolio-level 15% drawdown brake, actual 8h Binance funding rates, 0.04% maker & 0.15% taker):
  - Baseline      : Unchanged Config C (100%)
  - 3-Pod Balanced: 0.40 Donchian + 0.35 Supertrend+ADX + 0.35 CTA TSMOM
  - 4-Pod Balanced: 0.30 Donchian + 0.25 Supertrend+ADX + 0.25 CTA TSMOM + 0.35 Funding+Vol Squeeze
  - 4-Pod Growth  : 0.40 Donchian + 0.35 Supertrend+ADX + 0.35 CTA TSMOM + 0.35 Funding+Vol Squeeze
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from portfolio_engine import PortfolioTrade, add_3pod_features
from research.crypto_idle_margin_study import (
    TOP5_COINS,
    _round_lot,
    load_binance_funding,
    load_cached_binance_klines,
)
from research.crypto_trend_csmom_5y import make_cfg
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats
from portfolio_engine import run_shared_donchian_portfolio


def prepare_4pod_frames(
    dfs: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    start_str: str,
    end_str: str,
) -> Dict[str, pd.DataFrame]:
    ev_s = pd.Timestamp(start_str, tz="Asia/Kolkata")
    ev_e = pd.Timestamp(end_str, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    out = {}
    for sym, raw in dfs.items():
        f = add_3pod_features(raw, entry_len=20, warmup_bars=240)
        f["vol_ma20"] = raw["volume"].rolling(20).mean().shift(1).reindex(f.index)
        fr = funding_map.get(sym, pd.Series(dtype=float))
        if not fr.empty:
            fr_3d = fr.rolling(9, min_periods=1).mean()
            f["fr_3d"] = fr_3d.reindex(f.index, method="ffill").fillna(0.0001)
        else:
            f["fr_3d"] = 0.0001
        f = f[(f.index >= ev_s) & (f.index <= ev_e)].dropna()
        if not f.empty:
            out[sym] = f
    return out


def run_shared_4pod(
    prepped: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    weights: Tuple[float, float, float, float],  # (w_dc, w_st, w_tsmom, w_squeeze)
    capital_init: float = 1200.0,
    entry_slip: float = 0.0004,
    exit_slip: float = 0.0015,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    w_dc, w_st, w_ts, w_sq = weights
    cfg = PortfolioConfig(initial_capital=capital_init, whole_lot_rounding=True)
    all_ts = sorted(set().union(*[df.index for df in prepped.values()]))

    # Virtual pods per symbol
    pods: Dict[str, Dict[str, Optional[dict]]] = {
        s: {"dc": None, "st": None, "ts": None, "sq": None} for s in prepped
    }
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    peak_eq = capital_init
    brake = False
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []
    peak_mgn_pct = 0.0

    def total_margin() -> float:
        return sum(
            p["margin"] for sp in pods.values() for p in sp.values() if p is not None
        )

    def open_risk_pod(pod_name: str) -> float:
        return sum(
            sp[pod_name]["risk_pct"] for sp in pods.values() if sp[pod_name] is not None
        )

    def close_pod(sym: str, pod_name: str, raw_exit: float, ts: pd.Timestamp, reason: str):
        nonlocal capital
        p = pods[sym][pod_name]
        if p is None:
            return
        is_long = p["side"] == "LONG"
        ex_px = raw_exit * (1.0 - exit_slip if is_long else 1.0 + exit_slip)
        gross = ((ex_px - p["entry"]) if is_long else (p["entry"] - ex_px)) * p["qty"]
        net = gross - p["funding"]
        capital += net
        trades.append(PortfolioTrade(
            pod_name.upper(), sym, p["side"], p["entry_time"], ts,
            p["entry"], ex_px, p["qty"], gross, p["funding"], net, net / max(p["risk_usd"], 1e-6), reason
        ))
        pods[sym][pod_name] = None

    for bar_i, ts in enumerate(all_ts):
        # Portfolio-level MTM DD brake
        unr = 0.0
        for s, sp in pods.items():
            if s not in last_bar:
                continue
            c = float(last_bar[s]["close"])
            for p in sp.values():
                if p is not None:
                    unr += (((c - p["entry"]) if p["side"] == "LONG" else (p["entry"] - c)) * p["qty"] - p["funding"])
        cur_eq = capital + unr
        peak_eq = max(peak_eq, cur_eq)
        dd = (peak_eq - cur_eq) / peak_eq if peak_eq > 0 else 0.0
        if dd >= 0.15:
            brake = True
        elif dd <= 0.075:
            brake = False
        dd_scale = 0.5 if brake else 1.0

        # Entries & signal exits at bar open using prior-bar indicators
        for sym, df in prepped.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            prev = last_bar.get(sym)
            if prev is None or prev["atr14"] <= 0:
                last_bar[sym] = row
                continue

            open_px = float(row["open"])
            atr = float(prev["atr14"])

            # --- Pod 1: Donchian (Config C) ---
            if w_dc > 0 and pods[sym]["dc"] is None and capital > 0:
                side = None
                if prev["ema50"] > prev["ema200"] and prev["close"] > prev["donchian_high"]:
                    side = "LONG"
                elif prev["ema50"] < prev["ema200"] and prev["close"] < prev["donchian_low"]:
                    side = "SHORT"
                if side is not None:
                    eff_r = 0.02 * w_dc * dd_scale
                    eff_mr = 0.08 * w_dc * dd_scale
                    if open_risk_pod("dc") + eff_r <= eff_mr + 1e-9:
                        entry = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = 3.5 * atr
                        raw_qty = (capital * eff_r) / sl_dist
                        mgn_avail = max(capital * 0.75 - total_margin(), 0.0)
                        raw_qty = min(raw_qty, (mgn_avail * 5.0) / entry)
                        qty = _round_lot(sym, raw_qty, cfg)
                        if qty > 0:
                            r_usd = max(qty * sl_dist, 1e-6)
                            pods[sym]["dc"] = {
                                "side": side, "entry_time": ts, "entry": entry, "init_entry": entry,
                                "init_sl": sl_dist, "qty": qty, "sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
                                "margin": qty * entry / 5.0, "risk_usd": r_usd, "risk_pct": r_usd / capital,
                                "funding": 0.0, "last_f_ts": ts, "pyramided": False, "trail_mult": 3.5,
                            }
            elif w_dc > 0 and pods[sym]["dc"] is not None and not pods[sym]["dc"]["pyramided"] and capital > 0:
                p = pods[sym]["dc"]
                is_l = p["side"] == "LONG"
                move = (open_px - p["init_entry"]) if is_l else (p["init_entry"] - open_px)
                be_ok = (p["sl"] >= p["init_entry"]) if is_l else (p["sl"] <= p["init_entry"])
                if move >= 1.5 * p["init_sl"] and be_ok:
                    add_entry = open_px * (1.0 + entry_slip if is_l else 1.0 - entry_slip)
                    mgn_avail = max(capital * 0.75 - total_margin(), 0.0)
                    raw_add = min(p["qty"] * 0.5, (mgn_avail * 5.0) / add_entry)
                    add_qty = _round_lot(sym, raw_add, cfg)
                    if add_qty > 0:
                        nq = p["qty"] + add_qty
                        p["entry"] = (p["entry"] * p["qty"] + add_entry * add_qty) / nq
                        p["qty"] = nq
                        p["margin"] += add_qty * add_entry / 5.0
                        p["pyramided"] = True

            # --- Pod 2: Supertrend(10,3) + ADX>20 ---
            if w_st > 0:
                p_st = pods[sym]["st"]
                if p_st is not None:
                    is_l = p_st["side"] == "LONG"
                    if (is_l and prev["st_dir"] == -1) or ((not is_l) and prev["st_dir"] == 1):
                        close_pod(sym, "st", open_px, ts, "ST_FLIP")
                if pods[sym]["st"] is None and prev["adx14"] >= 20.0 and capital > 0:
                    side = "LONG" if prev["st_dir"] == 1 else "SHORT"
                    eff_r = 0.02 * w_st * dd_scale
                    eff_mr = 0.08 * w_st * dd_scale
                    if open_risk_pod("st") + eff_r <= eff_mr + 1e-9:
                        entry = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = 3.5 * atr
                        raw_qty = (capital * eff_r) / sl_dist
                        mgn_avail = max(capital * 0.75 - total_margin(), 0.0)
                        raw_qty = min(raw_qty, (mgn_avail * 5.0) / entry)
                        qty = _round_lot(sym, raw_qty, cfg)
                        if qty > 0:
                            r_usd = max(qty * sl_dist, 1e-6)
                            pods[sym]["st"] = {
                                "side": side, "entry_time": ts, "entry": entry, "qty": qty,
                                "sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
                                "margin": qty * entry / 5.0, "risk_usd": r_usd, "risk_pct": r_usd / capital,
                                "funding": 0.0, "last_f_ts": ts, "trail_mult": 3.5,
                            }

            # --- Pod 3: CTA TSMOM Vol-Target (48h = 8 bars rebalance) ---
            if w_ts > 0 and bar_i % 8 == 0:
                sig = (
                    (1 if prev["ret_7d"] > 0 else -1)
                    + (1 if prev["ret_21d"] > 0 else -1)
                    + (1 if prev["ret_60d"] > 0 else -1)
                ) / 3.0
                if abs(sig) < 0.3:
                    sig = 0.0
                target_side = "LONG" if sig > 0 else ("SHORT" if sig < 0 else None)
                p_ts = pods[sym]["ts"]
                if p_ts is not None and target_side != p_ts["side"]:
                    close_pod(sym, "ts", open_px, ts, "TSMOM_REBAL")
                if pods[sym]["ts"] is None and target_side is not None and capital > 0:
                    vol = float(prev["vol28"])
                    if not math.isfinite(vol) or vol < 0.15:
                        vol = 0.60
                    notional = capital * min(abs(sig) * (0.35 / vol) / len(prepped), 0.6) * w_ts * dd_scale
                    entry = open_px * (1.0 + entry_slip if target_side == "LONG" else 1.0 - entry_slip)
                    mgn_avail = max(capital * 0.75 - total_margin(), 0.0)
                    raw_qty = min(notional / entry, (mgn_avail * 5.0) / entry)
                    qty = _round_lot(sym, raw_qty, cfg)
                    if qty > 0:
                        sl_dist = 4.5 * atr
                        r_usd = max(qty * sl_dist, 1e-6)
                        pods[sym]["ts"] = {
                            "side": target_side, "entry_time": ts, "entry": entry, "qty": qty,
                            "sl": entry - sl_dist if target_side == "LONG" else entry + sl_dist,
                            "margin": qty * entry / 5.0, "risk_usd": r_usd, "risk_pct": r_usd / capital,
                            "funding": 0.0, "last_f_ts": ts, "trail_mult": 4.5,
                        }

            # --- Pod 4: Funding + Volume Squeeze Composite ---
            if w_sq > 0 and pods[sym]["sq"] is None and capital > 0:
                vol_surge = float(prev["volume"]) > 1.5 * float(prev["vol_ma20"])
                side = None
                if vol_surge and float(prev["close"]) > float(prev["donchian_high"]) and float(prev["fr_3d"]) <= 0.00005:
                    side = "LONG"
                elif vol_surge and float(prev["close"]) < float(prev["donchian_low"]) and float(prev["fr_3d"]) >= 0.00015:
                    side = "SHORT"
                if side is not None:
                    eff_r = 0.02 * w_sq * dd_scale
                    eff_mr = 0.08 * w_sq * dd_scale
                    if open_risk_pod("sq") + eff_r <= eff_mr + 1e-9:
                        entry = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = 3.0 * atr
                        raw_qty = (capital * eff_r) / sl_dist
                        mgn_avail = max(capital * 0.75 - total_margin(), 0.0)
                        raw_qty = min(raw_qty, (mgn_avail * 5.0) / entry)
                        qty = _round_lot(sym, raw_qty, cfg)
                        if qty > 0:
                            r_usd = max(qty * sl_dist, 1e-6)
                            pods[sym]["sq"] = {
                                "side": side, "entry_time": ts, "entry": entry, "qty": qty,
                                "sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
                                "margin": qty * entry / 5.0, "risk_usd": r_usd, "risk_pct": r_usd / capital,
                                "funding": 0.0, "last_f_ts": ts, "trail_mult": 3.0,
                            }

            last_bar[sym] = row

        # Funding accrual & intrabar trailing stop check
        for sym, sp in pods.items():
            row = last_bar.get(sym)
            if row is None or row.name != ts:
                continue
            fr_s = funding_map.get(sym, pd.Series(dtype=float))
            for pod_name, p in list(sp.items()):
                if p is None:
                    continue
                if not fr_s.empty:
                    r_sum = float(fr_s.loc[(fr_s.index > p["last_f_ts"]) & (fr_s.index <= ts)].sum())
                    p["funding"] += r_sum * p["qty"] * float(row["close"]) * (1.0 if p["side"] == "LONG" else -1.0)
                p["last_f_ts"] = ts

                is_l = p["side"] == "LONG"
                stop = p["sl"]
                hit = row["low"] <= stop if is_l else row["high"] >= stop
                if hit:
                    gapped = row["open"] < stop if is_l else row["open"] > stop
                    raw_ex = float(row["open"]) if gapped else stop
                    close_pod(sym, pod_name, raw_ex, ts, "GAP_STOP" if gapped else "ATR_TRAIL")
                else:
                    trail = p["trail_mult"] * float(row["atr14"])
                    c = float(row["close"])
                    p["sl"] = max(stop, c - trail) if is_l else min(stop, c + trail)

        unr = 0.0
        for s, sp in pods.items():
            if s not in last_bar:
                continue
            c = float(last_bar[s]["close"])
            for p in sp.values():
                if p is not None:
                    unr += (((c - p["entry"]) if p["side"] == "LONG" else (p["entry"] - c)) * p["qty"] - p["funding"])
        eq_now = capital + unr
        peak_mgn_pct = max(peak_mgn_pct, total_margin() / max(eq_now, 1.0) * 100.0)
        eq_recs.append((ts, eq_now))

    final_ts = all_ts[-1]
    for sym, sp in pods.items():
        if sym not in last_bar:
            continue
        for pod_name in list(sp.keys()):
            if sp[pod_name] is not None:
                close_pod(sym, pod_name, float(last_bar[sym]["close"]), final_ts, "END_OF_TEST")

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"peak_margin_pct": round(peak_mgn_pct, 1)}


def main():
    out_dir = new_run_directory("validate-4pod-crypto")
    bin_dfs = {sym: load_cached_binance_klines(sym, "6h", "2021-11-24", "2026-11-23") for sym in TOP5_COINS}
    fund_map = {sym: load_binance_funding(sym, "2021-08-01", "2026-11-23") for sym in TOP5_COINS}

    for title, s_d, e_d in [
        ("BINANCE 5Y FULL (2021-11-24 -> 2026-11-23)", "2021-11-24", "2026-11-23"),
        ("BINANCE IS 3Y (2021-11-24 -> 2024-11-23)", "2021-11-24", "2024-11-23"),
        ("BINANCE OOS 2Y (2024-11-24 -> 2026-11-23)", "2024-11-24", "2026-11-23"),
    ]:
        prepped = prepare_4pod_frames(bin_dfs, fund_map, s_d, e_d)
        rows = []
        for slip_lbl, eslip in [("Maker 0.04%", 0.0004), ("Taker 0.15%", 0.0015)]:
            for name, w in [
                ("3-Pod Balanced (0.40/0.35/0.35/0.00)", (0.40, 0.35, 0.35, 0.00)),
                ("4-Pod Balanced (0.30/0.25/0.25/0.35)", (0.30, 0.25, 0.25, 0.35)),
                ("4-Pod Growth   (0.40/0.35/0.35/0.35)", (0.40, 0.35, 0.35, 0.35)),
                ("4-Pod Squeeze+ (0.40/0.35/0.35/0.50)", (0.40, 0.35, 0.35, 0.50)),
            ]:
                tr, eq, diag = run_shared_4pod(prepped, fund_map, w, entry_slip=eslip, exit_slip=0.0015)
                st = calc_curve_stats(f"[{slip_lbl}] {name}", eq, tr, 1200.0)
                st["PkMgn%"] = diag["peak_margin_pct"]
                rows.append(st)
        print(f"\n================ {title} ================")
        df_r = pd.DataFrame(rows)
        print(df_r.to_string(index=False))
        df_r.to_csv(out_dir / f"{title.split()[1].lower()}_results.csv", index=False)
    print(f"\nSaved 4-Pod validation tables to {out_dir}/")


if __name__ == "__main__":
    main()
