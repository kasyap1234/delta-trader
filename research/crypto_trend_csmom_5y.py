"""
5-Year Top-5 Liquid Crypto Multi-Timeframe Trend & Cross-Sectional Momentum Study (2021-11 -> 2026-11)
Uses locally cached 5-year Binance USD-M 6h candles + actual historical 8h funding rates.
Zero double-counting: every multi-sleeve split sizes each sleeve on its exact USD slice with whole-lot rounding,
and the spare-margin CS-Mom overlay tracks exact bar-by-bar peak shared margin utilization (<= 75%).
"""

import argparse
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import _resample_ohlcv, resolve_date_window
from portfolio_engine import PortfolioTrade, add_ema_atr_features, run_shared_donchian_portfolio
from research.crypto_idle_margin_study import (
    TOP5_COINS,
    _actual_funding_cashflow,
    _round_lot,
    load_binance_funding,
    load_cached_binance_klines,
)
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats


def make_cfg(capital: float, entry_slip: float = 0.0004) -> PortfolioConfig:
    return PortfolioConfig(
        initial_capital=capital, usd_inr_rate=1.0,
        donchian_risk_per_trade_pct=0.02, max_concurrent_risk_pct=0.08,
        max_crypto_concurrent=5, atr_trail_mult=3.5,
        slippage_pct=0.0015, entry_slippage_pct=entry_slip,
        pyramid_add_fraction=0.5, dd_brake_threshold=0.15,
        whole_lot_rounding=True, enable_credit_spreads=False,
        crypto_assets=TOP5_COINS, commodity_assets=[],
    )


def resample_dict(dfs_6h: Dict[str, pd.DataFrame], interval: str) -> Dict[str, pd.DataFrame]:
    if interval == "6h":
        return dfs_6h
    out = {}
    for sym, df in dfs_6h.items():
        utc = df.tz_convert("UTC")
        res = _resample_ohlcv(utc, interval).tz_convert("Asia/Kolkata")
        out[sym] = res
    return out


def run_csmom_actual_funding(
    dfs_6h: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    lookback_bars: int = 56,     # 56 * 6h = 14 days
    rebalance_bars: int = 28,    # 28 * 6h = 7 days
    gross_leverage: float = 1.5, # 0.75x Long + 0.75x Short
    top_k: int = 1,
    sl_atr_mult: float = 4.0,
) -> Tuple[List[PortfolioTrade], pd.Series, pd.Series]:
    """
    Intact market-neutral Cross-Sectional L/S Volatility-Adjusted Momentum using actual historical 8h funding rates.
    Returns (trades, daily_equity_series, bar_margin_used_series).
    """
    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in dfs_6h.items():
        f = add_ema_atr_features(df)
        f["ret1"] = f["close"].pct_change()
        f["vol28"] = f["ret1"].rolling(28).std()
        f["mom"] = f["close"].pct_change(lookback_bars)
        fr = funding_map.get(sym, pd.Series(dtype=float))
        f["funding_rate"] = fr.reindex(f.index, method="ffill").fillna(0.0) if not fr.empty else 0.0
        f = f.iloc[cfg.ema_warmup_bars:].dropna()
        f = f[(f.index >= eval_start) & (f.index <= eval_end)]
        if not f.empty:
            prepared[sym] = f

    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    if not all_ts:
        return [], pd.Series(dtype=float), pd.Series(dtype=float)

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
                engine=f"CS_MOM_{lookback_bars}", asset=sym, side=pos["side"],
                entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"],
                gross_pnl=gross, funding_cost=pos["funding_cost"], net_pnl=net,
                r_multiple=net / max(pos["risk_rs"], 1e-6), exit_reason=reason,
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
                if prev is not None and prev["vol28"] > 1e-9 and math.isfinite(prev["mom"]):
                    scores.append((float(prev["mom"]) / float(prev["vol28"]), sym))
            if len(scores) >= 2 * top_k:
                scores.sort(key=lambda x: x[0])
                desired: Dict[str, str] = {}
                for _, sym in scores[:top_k]:
                    desired[sym] = "SHORT"
                for _, sym in scores[-top_k:]:
                    desired[sym] = "LONG"

                for sym, pos in list(pos_map.items()):
                    if pos is not None and desired.get(sym) != pos["side"] and sym in present:
                        close_pos(sym, ts, float(present[sym]["open"]), "REBALANCE", entry_slip)

                max_gross = min(gross_leverage, lev * cfg.max_margin_utilization)
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
                    pos_map[sym] = {
                        "side": side, "entry_time": ts, "entry": entry_px, "qty": qty,
                        "sl": (entry_px - sl_dist) if side == "LONG" else (entry_px + sl_dist),
                        "risk_rs": qty * sl_dist, "margin": (qty * entry_px) / lev,
                        "funding_cost": 0.0, "last_f": ts,
                    }

        for sym, row in present.items():
            last_bar[sym] = row

        for sym, pos in list(pos_map.items()):
            row = present.get(sym)
            if pos is None or row is None:
                continue
            periods = (ts - pos["last_f"]) // pd.Timedelta(hours=8)
            if periods:
                rate = float(row["funding_rate"])
                pos["funding_cost"] += _actual_funding_cashflow(pos["side"], float(row["close"]) * pos["qty"], rate, periods)
                pos["last_f"] += periods * pd.Timedelta(hours=8)
            if sl_atr_mult:
                is_long = pos["side"] == "LONG"
                if (row["low"] <= pos["sl"]) if is_long else (row["high"] >= pos["sl"]):
                    gapped = (row["open"] < pos["sl"]) if is_long else (row["open"] > pos["sl"])
                    raw_exit = float(row["open"]) if (cfg.gap_aware_stops and gapped) else pos["sl"]
                    close_pos(sym, ts, raw_exit, "GAP_STOP" if gapped else "HARD_STOP", cfg.slippage_pct)

        unr = 0.0
        m_used = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar:
                px = float(last_bar[sym]["close"])
                unr += ((px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - px) * pos["qty"]) - pos["funding_cost"]
                m_used += pos["margin"]
        eq_records.append((ts, capital + unr))
        margin_records.append((ts, m_used))

    final_ts = all_ts[-1]
    for sym, pos in list(pos_map.items()):
        if pos is not None and sym in last_bar:
            close_pos(sym, final_ts, float(last_bar[sym]["close"]), "END_OF_TEST", cfg.slippage_pct)

    eq = pd.Series([v for _, v in eq_records[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_records]))
    daily_eq = eq.resample("1D").last().ffill()
    m_series = pd.Series([v for _, v in margin_records], index=pd.DatetimeIndex([t for t, _ in margin_records]))
    return trades, daily_eq, m_series


def donchian_margin_series(trades: List[PortfolioTrade], idx: pd.DatetimeIndex, lev: float = 5.0) -> pd.Series:
    m = pd.Series(0.0, index=idx)
    for t in trades:
        margin = (t.qty * t.entry_price) / lev
        m.loc[(m.index >= t.entry_time) & (m.index < t.exit_time)] += margin
    return m


def evaluate_period(
    label: str,
    dfs_6h: Dict[str, pd.DataFrame],
    dfs_12h: Dict[str, pd.DataFrame],
    dfs_1d: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    start_str: str,
    end_str: str,
    capital: float,
    entry_slip: float,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    _, _, ev_s, ev_e = resolve_date_window("6h", start_date=start_str, end_date=end_str, include_warmup=True)
    cfg_full = make_cfg(capital, entry_slip)
    crypto_set = set(TOP5_COINS)

    # Standalone sleeves ($1,200 each)
    t_6h, eq_6h, _ = run_shared_donchian_portfolio(dfs_6h, crypto_set, cfg_full, eval_start=ev_s, eval_end=ev_e)
    t_12h, eq_12h, _ = run_shared_donchian_portfolio(dfs_12h, crypto_set, cfg_full, eval_start=ev_s, eval_end=ev_e)
    t_1d, eq_1d, _ = run_shared_donchian_portfolio(dfs_1d, crypto_set, cfg_full, eval_start=ev_s, eval_end=ev_e)
    t_cs14, eq_cs14, m_cs14 = run_csmom_actual_funding(dfs_6h, funding_map, cfg_full, ev_s, ev_e, lookback_bars=56, rebalance_bars=28, gross_leverage=1.5)
    t_cs30, eq_cs30, _ = run_csmom_actual_funding(dfs_6h, funding_map, cfg_full, ev_s, ev_e, lookback_bars=120, rebalance_bars=56, gross_leverage=1.5)

    # True whole-lot capital-split portfolios (sum of sleeve capitals == $1,200, ZERO double-counting)
    cfg_half = make_cfg(capital * 0.5, entry_slip)
    cfg_third = make_cfg(capital / 3.0, entry_slip)
    t_6h_h, eq_6h_h, _ = run_shared_donchian_portfolio(dfs_6h, crypto_set, cfg_half, eval_start=ev_s, eval_end=ev_e)
    t_12h_h, eq_12h_h, _ = run_shared_donchian_portfolio(dfs_12h, crypto_set, cfg_half, eval_start=ev_s, eval_end=ev_e)
    t_cs14_h, eq_cs14_h, _ = run_csmom_actual_funding(dfs_6h, funding_map, cfg_half, ev_s, ev_e, lookback_bars=56, rebalance_bars=28, gross_leverage=1.5)

    t_6h_3, eq_6h_3, _ = run_shared_donchian_portfolio(dfs_6h, crypto_set, cfg_third, eval_start=ev_s, eval_end=ev_e)
    t_12h_3, eq_12h_3, _ = run_shared_donchian_portfolio(dfs_12h, crypto_set, cfg_third, eval_start=ev_s, eval_end=ev_e)
    t_1d_3, eq_1d_3, _ = run_shared_donchian_portfolio(dfs_1d, crypto_set, cfg_third, eval_start=ev_s, eval_end=ev_e)

    # Spare-margin stack: Unchanged Config C ($1,200) + 0.75x gross CS-Mom 14d (intact L/S) on same $1,200 account
    t_cs14_ov, eq_cs14_ov, m_cs14_ov = run_csmom_actual_funding(
        dfs_6h, funding_map, cfg_full, ev_s, ev_e, lookback_bars=56, rebalance_bars=28, gross_leverage=0.75
    )

    common = eq_6h.index
    def align(s: pd.Series) -> pd.Series:
        return s.reindex(common).ffill().bfill()

    eq_blend_6_12 = align(eq_6h_h) + align(eq_12h_h)
    eq_blend_mtf3 = align(eq_6h_3) + align(eq_12h_3) + align(eq_1d_3)
    eq_blend_12_cs = align(eq_12h_h) + align(eq_cs14_h)
    eq_stack_c_cs = align(eq_6h) + (align(eq_cs14_ov) - capital)

    # Audit peak shared margin utilization for the spare-margin stack
    m_base = donchian_margin_series(t_6h, m_cs14_ov.index, lev=5.0)
    tot_margin = m_base + m_cs14_ov
    daily_eq_bar = eq_stack_c_cs.reindex(tot_margin.index.normalize()).ffill().bfill()
    peak_util_pct = float((tot_margin / daily_eq_bar.values).max() * 100.0) if len(tot_margin) else 0.0

    rows = [
        calc_curve_stats("1. Baseline: Unchanged Config C (6h Donchian)", align(eq_6h), t_6h, capital),
        calc_curve_stats("2. Standalone: 12h Donchian (Config C rules)", align(eq_12h), t_12h, capital),
        calc_curve_stats("3. Standalone: 1d Donchian (Config C rules)", align(eq_1d), t_1d, capital),
        calc_curve_stats("4. Standalone: CS-Mom 14d (7d rebal, 1.5x L/S)", align(eq_cs14), t_cs14, capital),
        calc_curve_stats("5. Standalone: CS-Mom 30d (14d rebal, 1.5x L/S)", align(eq_cs30), t_cs30, capital),
        calc_curve_stats("6. Split 50/50: 6h + 12h Donchian ($600+$600)", eq_blend_6_12, t_6h_h + t_12h_h, capital),
        calc_curve_stats("7. Split 3-Speed MTF: 6h + 12h + 1d ($400 ea)", eq_blend_mtf3, t_6h_3 + t_12h_3 + t_1d_3, capital),
        calc_curve_stats("8. Split 50/50: 12h Donchian + CS-Mom 14d", eq_blend_12_cs, t_12h_h + t_cs14_h, capital),
        calc_curve_stats(f"9. Spare-Margin: Config C + 0.75x CS-Mom 14d (PkMgn {peak_util_pct:.0f}%)", eq_stack_c_cs, t_6h + t_cs14_ov, capital),
    ]
    df_out = pd.DataFrame(rows)
    print(f"\n================ {label} ({start_str} -> {end_str}) ================")
    print(df_out.to_string(index=False))

    curves = {
        "ConfigC_6h": align(eq_6h),
        "Donchian_12h": align(eq_12h),
        "Donchian_1d": align(eq_1d),
        "CS_Mom_14d": align(eq_cs14),
        "CS_Mom_30d": align(eq_cs30),
        "Split_6h_12h": eq_blend_6_12,
        "Split_MTF3": eq_blend_mtf3,
        "Split_12h_CS14": eq_blend_12_cs,
        "Stack_C_CS14": eq_stack_c_cs,
    }
    return df_out, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Year Top-5 Crypto Multi-Timeframe Trend & CS-Mom Study")
    parser.add_argument("--capital", type=float, default=1200.0)
    parser.add_argument("--taker-only", action="store_true")
    args = parser.parse_args()
    entry_slip = 0.0015 if args.taker_only else 0.0004

    dfs_6h = {sym: load_cached_binance_klines(sym, "6h", "2021-11-24", "2026-11-23") for sym in TOP5_COINS}
    funding_map = {sym: load_binance_funding(sym, "2021-08-01", "2026-11-23") for sym in TOP5_COINS}
    dfs_12h = resample_dict(dfs_6h, "12h")
    dfs_1d = resample_dict(dfs_6h, "1d")

    is_df, _ = evaluate_period("IN-SAMPLE (3Y)", dfs_6h, dfs_12h, dfs_1d, funding_map, "2021-11-24", "2024-11-23", args.capital, entry_slip)
    oos_df, _ = evaluate_period("OUT-OF-SAMPLE HOLDOUT (2Y)", dfs_6h, dfs_12h, dfs_1d, funding_map, "2024-11-24", "2026-11-23", args.capital, entry_slip)
    full_df, full_curves = evaluate_period("FULL 5-YEAR WINDOW", dfs_6h, dfs_12h, dfs_1d, funding_map, "2021-11-24", "2026-11-23", args.capital, entry_slip)

    # Correlation matrix of standalone sleeves
    standalone = pd.DataFrame({k: full_curves[k] for k in ("ConfigC_6h", "Donchian_12h", "Donchian_1d", "CS_Mom_14d", "CS_Mom_30d")})
    print("\n================ 5-YEAR DAILY RETURN CORRELATIONS ================")
    print(standalone.pct_change().dropna().corr().round(2).to_string())

    # Calendar year-by-year returns
    print("\n================ CALENDAR YEAR RETURNS (%) ================")
    yr_rows = {}
    for name in ("ConfigC_6h", "Donchian_12h", "CS_Mom_14d", "Split_6h_12h", "Split_MTF3", "Split_12h_CS14", "Stack_C_CS14"):
        s = full_curves[name]
        ye = s.resample("YE").last()
        ret = ye.pct_change()
        ret.iloc[0] = ye.iloc[0] / args.capital - 1.0
        yr_rows[name] = (ret * 100.0).round(1)
    yr_df = pd.DataFrame(yr_rows).T
    yr_df.columns = [str(c.year) for c in yr_df.columns]
    print(yr_df.to_string())

    out_dir = new_run_directory("crypto-trend-csmom-5y")
    is_df.to_csv(out_dir / "in_sample_2021_2024.csv", index=False)
    oos_df.to_csv(out_dir / "oos_2024_2026.csv", index=False)
    full_df.to_csv(out_dir / "full_5y_2021_2026.csv", index=False)
    pd.DataFrame(full_curves).to_csv(out_dir / "equity_curves_5y.csv")
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
