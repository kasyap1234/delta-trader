"""
Single-Account Shared-Margin Validation of the 3-Pod Crypto Engine
==================================================================
Pods (all on Top 5 liquid majors: BTC, ETH, SOL, BNB, DOGE):
  - Pod 1: Config C (6h Donchian(20) + EMA(50/200) + 3.5x ATR trail + +0.5x pyramid @ +1.5R)
  - Pod 2: Supertrend(10,3) + ADX(14)>20 (no EMA200 lag, 3.5x ATR trail + ST flip exit)
  - Pod 3: CTA TSMOM Vol-Target (7d/21d/60d multi-horizon trend, inverse-vol weighted, 48h rebal)

Validates inside ONE shared $1,200 USD account with:
  - Real whole-lot contract multipliers (BTC=0.001, ETH=0.01, SOL=1, BNB=0.1, DOGE=100)
  - Shared 75% max margin ceiling (strictly enforced per bar across all 3 pods)
  - Portfolio-level 15% drawdown brake
  - Both 0.04% maker entry and 0.15% worst-case taker entry stress tests
  - 5-Year Binance USD-M (2021-11 -> 2026-11, actual 8h funding rates) + IS/OOS split
  - 2-Year Delta Exchange India cross-check (2024-11 -> 2026-11, actual Delta 6h candles)
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import load_delta_candles, resolve_date_window
from portfolio_engine import PortfolioTrade, run_shared_donchian_portfolio
from research.crypto_broad_strategy_lab_5y import prepare_lab_frames
from research.crypto_idle_margin_study import (
    TOP5_COINS,
    _actual_funding_cashflow,
    _round_lot,
    load_binance_funding,
    load_cached_binance_klines,
)
from research.crypto_trend_csmom_5y import make_cfg
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

DELTA_TOP5 = ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"]
DELTA_CACHE_DIR = Path(__file__).resolve().parent / "output" / "_cache_delta_2y"


def load_cached_delta_klines(symbol: str, interval: str = "6h", days: int = 730) -> pd.DataFrame:
    DELTA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DELTA_CACHE_DIR / f"{symbol}_{interval}_{days}d.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.index.name = "timestamp"
        return df.sort_index()
    df = load_delta_candles(symbol, resolution=interval, days=days, include_warmup=True)
    df.to_csv(cache_file)
    return df


def run_single_account_3pod(
    dfs_6h: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    w_donchian: float = 1.0,
    w_supertrend: float = 0.35,
    w_tsmom: float = 0.35,
    use_actual_funding: bool = True,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Runs all 3 pods inside ONE shared $1,200 account with real whole-lot rounding,
    shared 75% margin cap, and portfolio-level 15% drawdown brake.
    """
    prepared = prepare_lab_frames(dfs_6h, funding_map, cfg, eval_start, eval_end)
    # Also attach shifted Donchian High/Low
    for sym, df in prepared.items():
        raw = dfs_6h[sym]
        dh = raw["high"].rolling(cfg.donchian_entry_len).max().shift(1)
        dl = raw["low"].rolling(cfg.donchian_entry_len).min().shift(1)
        df["donchian_high"] = dh.reindex(df.index)
        df["donchian_low"] = dl.reindex(df.index)
        prepared[sym] = df.dropna()

    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    if not all_ts:
        return [], pd.Series(dtype=float), {"peak_margin_pct": 0.0, "skipped_margin": 0}

    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    lev = cfg.max_leverage or 5.0
    n_coins = len(prepared)

    capital = cfg.initial_capital
    peak_eq = capital
    brake_active = False

    # Independent virtual sub-books sharing ONE capital pool and ONE 75% margin ceiling
    pos_dc: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    pos_st: Dict[str, Optional[Dict]] = {s: None for s in prepared}
    pos_ts: Dict[str, Optional[Dict]] = {s: None for s in prepared}

    last_bar: Dict[str, pd.Series] = {}
    prev_bar2: Dict[str, pd.Series] = {}
    trades: List[PortfolioTrade] = []
    eq_records: List[Tuple[pd.Timestamp, float]] = []
    peak_margin_pct = 0.0
    skipped_margin = 0

    def total_allocated_margin() -> float:
        m = 0.0
        for book in (pos_dc, pos_st, pos_ts):
            for p in book.values():
                if p is not None:
                    m += p["margin"]
        return m

    def total_unrealized() -> float:
        u = 0.0
        for book in (pos_dc, pos_st, pos_ts):
            for sym, p in book.items():
                if p is not None and sym in last_bar:
                    px = float(last_bar[sym]["close"])
                    u += ((px - p["entry"]) * p["qty"] if p["side"] == "LONG" else (p["entry"] - px) * p["qty"]) - p["funding"]
        return u

    def close_leg(book: Dict[str, Optional[Dict]], engine: str, sym: str, ts: pd.Timestamp, raw_px: float, reason: str, slip: float) -> None:
        nonlocal capital
        p = book[sym]
        if p is None:
            return
        is_long = p["side"] == "LONG"
        exit_px = raw_px * (1.0 - slip if is_long else 1.0 + slip)
        gross = (exit_px - p["entry"]) * p["qty"] if is_long else (p["entry"] - exit_px) * p["qty"]
        net = gross - p["funding"]
        capital += net
        trades.append(
            PortfolioTrade(
                engine=engine, asset=sym, side=p["side"], entry_time=p["entry_time"], exit_time=ts,
                entry_price=p["entry"], exit_price=exit_px, qty=p["qty"], gross_pnl=gross,
                funding_cost=p["funding"], net_pnl=net, r_multiple=net / max(p["risk_rs"], 1e-6),
                exit_reason=reason,
            )
        )
        book[sym] = None

    for bar_idx, ts in enumerate(all_ts):
        # 1. Portfolio-level mark-to-market drawdown brake
        cur_eq = capital + total_unrealized()
        peak_eq = max(peak_eq, cur_eq)
        dd = (peak_eq - cur_eq) / peak_eq if peak_eq > 0 else 0.0
        if cfg.dd_brake_threshold:
            if dd >= cfg.dd_brake_threshold:
                brake_active = True
            elif dd <= cfg.dd_brake_threshold * 0.5:
                brake_active = False
        dd_scale = cfg.dd_brake_scale if brake_active else 1.0

        present = {}
        for sym, df in prepared.items():
            if ts in df.index:
                row = df.loc[ts]
                present[sym] = row.iloc[-1] if isinstance(row, pd.DataFrame) else row

        # 2. POD 1: Config C Donchian (strict first priority on shared margin)
        if w_donchian > 0 and capital > 0:
            dc_risk = cfg.donchian_risk_per_trade_pct * w_donchian * dd_scale
            dc_max_risk = cfg.max_concurrent_risk_pct * w_donchian * dd_scale
            for sym, row in present.items():
                prev = last_bar.get(sym)
                p = pos_dc[sym]
                if p is None and prev is not None and prev["atr14"] > 0:
                    side = None
                    if prev["ema50"] > prev["ema200"] and prev["close"] > prev["donchian_high"]:
                        side = "LONG"
                    elif prev["ema50"] < prev["ema200"] and prev["close"] < prev["donchian_low"]:
                        side = "SHORT"
                    if side is not None:
                        open_dc_risk = sum(x["risk_pct"] for x in pos_dc.values() if x is not None)
                        if open_dc_risk + dc_risk <= dc_max_risk + 1e-9:
                            open_px = float(row["open"])
                            entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                            sl_dist = cfg.atr_trail_mult * float(prev["atr14"])
                            free_m = max(capital * cfg.max_margin_utilization - total_allocated_margin(), 0.0)
                            raw_qty = min((capital * dc_risk) / sl_dist, (free_m * lev) / entry_px)
                            qty = _round_lot(sym, raw_qty, cfg)
                            if qty > 0:
                                pos_dc[sym] = {
                                    "side": side, "entry_time": ts, "entry": entry_px, "init_entry": entry_px,
                                    "init_sl_dist": sl_dist, "qty": qty, "margin": (qty * entry_px) / lev,
                                    "risk_rs": qty * sl_dist, "risk_pct": (qty * sl_dist) / capital,
                                    "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                                    "funding": 0.0, "last_f": ts, "pyramided": False,
                                }
                            else:
                                skipped_margin += 1
                elif p is not None and cfg.pyramid_add_fraction > 0 and not p["pyramided"] and prev is not None:
                    is_long = p["side"] == "LONG"
                    open_px = float(row["open"])
                    move = (open_px - p["init_entry"]) if is_long else (p["init_entry"] - open_px)
                    be_ok = (p["sl"] >= p["init_entry"]) if is_long else (p["sl"] <= p["init_entry"])
                    if move >= cfg.pyramid_trigger_r * p["init_sl_dist"] and be_ok:
                        add_entry = open_px * (1.0 + entry_slip if is_long else 1.0 - entry_slip)
                        free_m = max(capital * cfg.max_margin_utilization - total_allocated_margin(), 0.0)
                        raw_add = min(p["qty"] * cfg.pyramid_add_fraction, (free_m * lev) / add_entry)
                        add_qty = _round_lot(sym, raw_add, cfg)
                        if add_qty > 0:
                            new_qty = p["qty"] + add_qty
                            p["entry"] = (p["entry"] * p["qty"] + add_entry * add_qty) / new_qty
                            p["qty"] = new_qty
                            p["margin"] += (add_qty * add_entry) / lev
                            p["pyramided"] = True

        # 3. POD 2: Supertrend(10,3) + ADX>20 (uses remaining free margin)
        if w_supertrend > 0 and capital > 0:
            st_risk = 0.02 * w_supertrend * dd_scale
            st_max_risk = 0.08 * w_supertrend * dd_scale
            for sym, row in present.items():
                prev = last_bar.get(sym)
                prev2 = prev_bar2.get(sym)
                p = pos_st[sym]
                if p is not None and prev is not None:
                    is_long = p["side"] == "LONG"
                    if (is_long and prev["st_dir"] == -1) or ((not is_long) and prev["st_dir"] == 1):
                        close_leg(pos_st, "SUPERTREND", sym, ts, float(row["open"]), "ST_FLIP", entry_slip)
                        continue
                elif p is None and prev is not None and prev2 is not None and prev["atr14"] > 0:
                    side = None
                    if prev["st_dir"] == 1 and prev2["st_dir"] == -1 and prev["adx14"] >= 20.0:
                        side = "LONG"
                    elif prev["st_dir"] == -1 and prev2["st_dir"] == 1 and prev["adx14"] >= 20.0:
                        side = "SHORT"
                    if side is not None:
                        open_st_risk = sum(x["risk_pct"] for x in pos_st.values() if x is not None)
                        if open_st_risk + st_risk <= st_max_risk + 1e-9:
                            open_px = float(row["open"])
                            entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                            sl_dist = 3.5 * float(prev["atr14"])
                            free_m = max(capital * cfg.max_margin_utilization - total_allocated_margin(), 0.0)
                            raw_qty = min((capital * st_risk) / sl_dist, (free_m * lev) / entry_px)
                            qty = _round_lot(sym, raw_qty, cfg)
                            if qty > 0:
                                pos_st[sym] = {
                                    "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                                    "margin": (qty * entry_px) / lev, "risk_rs": qty * sl_dist,
                                    "risk_pct": (qty * sl_dist) / capital,
                                    "sl": entry_px - sl_dist if side == "LONG" else entry_px + sl_dist,
                                    "funding": 0.0, "last_f": ts,
                                }
                            else:
                                skipped_margin += 1

        # 4. POD 3: CTA TSMOM Vol-Target (48h rebalance, uses remaining free margin)
        if w_tsmom > 0 and bar_idx > 0 and bar_idx % 8 == 0 and capital > 0:
            target_vol = 0.35 * w_tsmom * dd_scale
            for sym, row in present.items():
                prev = last_bar.get(sym)
                p = pos_ts[sym]
                if prev is None or prev["vol28"] <= 0.05:
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
                    weight = min((target_vol / float(prev["vol28"])) * abs(sig) / n_coins, 0.40 * w_tsmom)
                    free_m = max(capital * cfg.max_margin_utilization - total_allocated_margin(), 0.0)
                    notional = min(capital * weight, free_m * lev)
                    open_px = float(row["open"])
                    entry_px = open_px * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                    qty = _round_lot(sym, notional / entry_px, cfg)
                    if qty > 0:
                        pos_ts[sym] = {
                            "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                            "margin": (qty * entry_px) / lev, "risk_rs": max(capital * 0.01, 1.0),
                            "sl": None, "funding": 0.0, "last_f": ts,
                        }

        for sym, row in present.items():
            if sym in last_bar:
                prev_bar2[sym] = last_bar[sym]
            last_bar[sym] = row

        # 5. Intrabar funding & stops across all 3 pods
        for book, eng_name, trail_mult in ((pos_dc, "DONCHIAN", cfg.atr_trail_mult), (pos_st, "SUPERTREND", 3.5), (pos_ts, "TSMOM", None)):
            for sym, p in list(book.items()):
                if p is None:
                    continue
                row = present.get(sym)
                if row is None:
                    continue
                periods = (ts - p["last_f"]) // pd.Timedelta(hours=8)
                if periods:
                    notional = float(row["close"]) * p["qty"]
                    if use_actual_funding:
                        p["funding"] += _actual_funding_cashflow(p["side"], notional, float(row["funding_rate"]), periods)
                    else:
                        p["funding"] += notional * cfg.crypto_funding_8h_pct * periods
                    p["last_f"] += periods * pd.Timedelta(hours=8)
                if trail_mult is not None and p["sl"] is not None:
                    is_long = p["side"] == "LONG"
                    if (row["low"] <= p["sl"]) if is_long else (row["high"] >= p["sl"]):
                        gapped = (row["open"] < p["sl"]) if is_long else (row["open"] > p["sl"])
                        raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else p["sl"]
                        close_leg(book, eng_name, sym, ts, raw_exit, "GAP_STOP" if gapped else "ATR_TRAIL", cfg.slippage_pct)
                        continue
                    c_px = float(row["close"])
                    tr = trail_mult * float(row["atr14"])
                    p["sl"] = max(p["sl"], c_px - tr) if is_long else min(p["sl"], c_px + tr)

        eq_now = capital + total_unrealized()
        if eq_now > 0:
            peak_margin_pct = max(peak_margin_pct, (total_allocated_margin() / eq_now) * 100.0)
        eq_records.append((ts, eq_now))

    final_ts = all_ts[-1]
    for book, eng_name in ((pos_dc, "DONCHIAN"), (pos_st, "SUPERTREND"), (pos_ts, "TSMOM")):
        for sym in list(book.keys()):
            if book[sym] is not None and sym in last_bar:
                close_leg(book, eng_name, sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    return trades, eq.resample("1D").last().ffill(), {
        "peak_margin_pct": round(peak_margin_pct, 1),
        "skipped_margin": skipped_margin,
    }


def print_block(title: str, dfs_6h: Dict[str, pd.DataFrame], funding_map: Dict[str, pd.Series], start_str: str, end_str: str, capital: float, use_actual_funding: bool) -> pd.DataFrame:
    _, _, ev_s, ev_e = resolve_date_window("6h", start_date=start_str, end_date=end_str, include_warmup=True)
    rows = []
    curves = {}
    for cost_label, entry_slip in (("Maker 0.04%", 0.0004), ("Taker 0.15%", 0.0015)):
        cfg = make_cfg(capital, entry_slip)
        cfg.crypto_assets = list(dfs_6h.keys())
        base_t, base_eq, _ = run_shared_donchian_portfolio(dfs_6h, set(dfs_6h.keys()), cfg, eval_start=ev_s, eval_end=ev_e)
        bal_t, bal_eq, bal_d = run_single_account_3pod(dfs_6h, funding_map, cfg, ev_s, ev_e, 0.40, 0.35, 0.35, use_actual_funding)
        agg_t, agg_eq, agg_d = run_single_account_3pod(dfs_6h, funding_map, cfg, ev_s, ev_e, 1.00, 0.35, 0.35, use_actual_funding)

        for name, eq, tr, extra in [
            (f"[{cost_label}] Baseline Config C (100%)", base_eq, base_t, ""),
            (f"[{cost_label}] 3-Pod Balanced (0.40/0.35/0.35) PkMgn {bal_d['peak_margin_pct']}%", bal_eq, bal_t, ""),
            (f"[{cost_label}] 3-Pod Spare-Mgn (1.0/0.35/0.35) PkMgn {agg_d['peak_margin_pct']}%", agg_eq, agg_t, ""),
        ]:
            rows.append(calc_curve_stats(name, eq, tr, capital))
        if cost_label == "Maker 0.04%":
            curves = {"ConfigC": base_eq, "Balanced": bal_eq, "SpareMgn": agg_eq}
            dc_in_agg = [t for t in agg_t if t.engine == "DONCHIAN"]
            parity_note = f"Parity (Maker): Standalone Config C={len(base_t)} trades vs Config C inside Spare-Mgn 3-Pod={len(dc_in_agg)} trades (skipped_margin={agg_d['skipped_margin']})"

    df_out = pd.DataFrame(rows)
    print(f"\n================ {title} ({start_str} -> {end_str}) ================")
    print(df_out.to_string(index=False))
    print(parity_note)

    if len(curves["ConfigC"]) > 365:
        print("\nCalendar Year Returns (Maker 0.04%, %):")
        yr_rows = {}
        for k, s in curves.items():
            ye = s.resample("YE").last()
            r = ye.pct_change()
            r.iloc[0] = ye.iloc[0] / capital - 1.0
            yr_rows[k] = (r * 100.0).round(1)
        yr_df = pd.DataFrame(yr_rows).T
        yr_df.columns = [str(c.year) for c in yr_df.columns]
        print(yr_df.to_string())
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-Account 3-Pod Crypto Validation + Taker Stress + Delta Cross-Check")
    parser.add_argument("--capital", type=float, default=1200.0)
    parser.add_argument("--skip-delta", action="store_true")
    args = parser.parse_args()

    dfs_bin = {sym: load_cached_binance_klines(sym, "6h", "2021-11-24", "2026-11-23") for sym in TOP5_COINS}
    fr_bin = {sym: load_binance_funding(sym, "2021-08-01", "2026-11-23") for sym in TOP5_COINS}

    df_5y = print_block("BINANCE USD-M 5-YEAR FULL (Actual 8h Funding)", dfs_bin, fr_bin, "2021-11-24", "2026-11-23", args.capital, True)
    df_is = print_block("BINANCE USD-M IN-SAMPLE 3Y (Actual 8h Funding)", dfs_bin, fr_bin, "2021-11-24", "2024-11-23", args.capital, True)
    df_oos = print_block("BINANCE USD-M OOS HOLDOUT 2Y (Actual 8h Funding)", dfs_bin, fr_bin, "2024-11-24", "2026-11-23", args.capital, True)

    out_dir = new_run_directory("validate-3pod-crypto")
    df_5y.to_csv(out_dir / "binance_5y.csv", index=False)
    df_is.to_csv(out_dir / "binance_is_3y.csv", index=False)
    df_oos.to_csv(out_dir / "binance_oos_2y.csv", index=False)

    if not args.skip_delta:
        print("\nLoading actual Delta Exchange India 6h candles for 2-year venue cross-check...")
        dfs_delta = {sym: load_cached_delta_klines(sym, "6h", 730) for sym in DELTA_TOP5}
        df_delta = print_block("DELTA EXCHANGE INDIA 2Y CROSS-CHECK (Flat 0.01%/8h Funding)", dfs_delta, {}, "2024-11-24", "2026-11-23", args.capital, False)
        df_delta.to_csv(out_dir / "delta_2y.csv", index=False)

    print(f"\nSaved validation tables to {out_dir}/")


if __name__ == "__main__":
    main()
