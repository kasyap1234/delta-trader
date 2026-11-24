"""
Kotak Neo Multi-Asset NSE ETF + Metals Backtest Study (Actual Upstox V3 Data)
=============================================================================
Universe (all actual Upstox V3 NSE_EQ INR candles):
  - Metals (2): GOLDBEES, SILVERBEES
  - Equities (4): NIFTYBEES, MID150BEES, BANKBEES, ITBEES
  - Cash Yield Benchmark: Modeled 6.5% p.a. LIQUIDBEES / overnight liquid fund on idle cash

Evaluates on Rs 98,000 capital, whole-lot rounding, zero capital double-counting,
0.05%/side base cost + 0.15%/side stress test:
  - 4-Year Daily (2022-12-01 -> 2026-11-23), split into:
      * In-Sample (2Y): 2022-12-01 -> 2024-11-23
      * Out-of-Sample (2Y): 2024-11-24 -> 2026-11-23
  - 5-Year Daily (2021-11-24 -> 2026-11-23) on 5 ETFs (ex-SILVERBEES which listed Feb 2022)
  - 2-Year 6H (2024-11-24 -> 2026-11-23) on all 6 ETFs
  - Single-Account Shared-Capital Multi-Pod validation (Metals Config C + Equity Engine)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio_engine import PortfolioTrade, run_shared_donchian_portfolio
from research.kotak_metals_upstox_study import (
    CAPITAL_INR,
    LIQUID_YIELD_APR,
    buy_and_hold_curve,
    load_cached,
    make_kotak_cfg,
    prepare_metal_features,
    run_config_c_with_cash_yield,
    run_supertrend_adx,
    run_tsmom_vol_target,
    yearly_returns,
)
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

METALS = ["GOLDBEES", "SILVERBEES"]
EQUITIES = ["NIFTYBEES", "MID150BEES", "BANKBEES", "ITBEES"]
ALL6 = METALS + EQUITIES


def run_top_k_vol_parity_rotation(
    prepared: Dict[str, pd.DataFrame],
    interval: str,
    top_k: int = 2,
    rebal_days: int = 10,
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    idle_yield_apr: float = LIQUID_YIELD_APR,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Long-Only Dual-Momentum Inverse-Vol Top-K ETF Rotation:
      - Rebalance every `rebal_days` trading days.
      - Qualify ETFs with positive blended momentum (0.5*ret_21d + 0.5*ret_60d > 0) AND Close > EMA50.
      - Hold up to `top_k` highest-momentum qualifiers weighted by inverse volatility (1/vol_ann).
      - Unfilled slots stay in LIQUIDBEES (modeled 6.5% p.a. yield).
    """
    bars_per_day = 2 if interval == "6h" else 1
    rebal_every = rebal_days * bars_per_day
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []
    idle_pcts: List[float] = []
    last_ts_yield: Optional[pd.Timestamp] = None

    for bar_i, ts in enumerate(all_ts):
        if idle_yield_apr > 0 and last_ts_yield is not None:
            dt_days = max((ts - last_ts_yield).total_seconds() / 86400.0, 0.0)
            if dt_days > 0:
                alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                capital += max(capital - alloc, 0.0) * ((1.0 + idle_yield_apr) ** (dt_days / 365.0) - 1.0)
        last_ts_yield = ts

        if bar_i % rebal_every == 0 and len(last_bar) >= 2:
            cands = []
            for sym, prev in last_bar.items():
                if ts not in prepared[sym].index:
                    continue
                mom = 0.5 * float(prev["ret_med"]) + 0.5 * float(prev["ret_slow"])
                if mom > 0 and float(prev["close"]) > float(prev["ema50"]):
                    vol = float(prev["vol_ann"])
                    if not math.isfinite(vol) or vol < 0.05:
                        vol = 0.15
                    cands.append((sym, mom, vol))
            cands.sort(key=lambda x: x[1], reverse=True)
            chosen = cands[:top_k]
            chosen_syms = {c[0] for c in chosen}

            # Close positions not in chosen set
            for sym, pos in list(pos_map.items()):
                if pos is not None and sym not in chosen_syms and ts in prepared[sym].index:
                    ex_px = float(prepared[sym].loc[ts]["open"]) * (1.0 - slippage)
                    pnl = (ex_px - pos["entry"]) * pos["qty"]
                    capital += pnl
                    trades.append(PortfolioTrade(
                        "ROTATION", sym, "LONG", pos["entry_time"], ts,
                        pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.025, 1.0), "ROTATE_OUT"
                    ))
                    pos_map[sym] = None

            # Compute inverse-vol target weights (scaled by len(chosen)/top_k so empty slots stay in cash!)
            if chosen:
                inv_vols = {s: 1.0 / v for s, _, v in chosen}
                iv_sum = sum(inv_vols.values())
                invest_frac = len(chosen) / float(top_k)
                # Mark-to-market total equity for clean rebalancing
                mtm_eq = capital + sum(
                    (float(prepared[s].loc[ts]["open"]) - p["entry"]) * p["qty"]
                    for s, p in pos_map.items()
                    if p is not None and ts in prepared[s].index
                )
                for sym, _, _ in chosen:
                    if pos_map[sym] is not None:
                        continue  # Keep existing winner to reduce turnover
                    w = invest_frac * (inv_vols[sym] / iv_sum)
                    open_px = float(prepared[sym].loc[ts]["open"])
                    entry = open_px * (1.0 + slippage)
                    alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                    budget = min(mtm_eq * w, max(capital - alloc, 0.0))
                    qty = math.floor(budget / entry)
                    if qty >= 1:
                        pos_map[sym] = {
                            "entry_time": ts, "entry": entry, "qty": qty, "margin": qty * entry,
                        }

        for sym, df in prepared.items():
            if ts in df.index:
                last_bar[sym] = df.loc[ts]

        alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
        unr = sum(
            (float(last_bar[s]["close"]) - p["entry"]) * p["qty"]
            for s, p in pos_map.items()
            if p is not None and s in last_bar
        )
        eq_now = capital + unr
        idle_pcts.append(max(capital - alloc, 0.0) / max(eq_now, 1.0) * 100.0)
        eq_recs.append((ts, eq_now))

    final_ts = all_ts[-1]
    for sym, pos in pos_map.items():
        if pos is not None and sym in last_bar:
            ex_px = float(last_bar[sym]["close"]) * (1.0 - slippage)
            pnl = (ex_px - pos["entry"]) * pos["qty"]
            capital += pnl
            trades.append(PortfolioTrade(
                "ROTATION", sym, "LONG", pos["entry_time"], final_ts,
                pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.025, 1.0), "END_OF_TEST"
            ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"avg_idle_cash_pct": round(float(np.mean(idle_pcts)), 1)}


def run_ibs_rsi2_equity_pullback(
    prepared: Dict[str, pd.DataFrame],
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    idle_yield_apr: float = LIQUID_YIELD_APR,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Classic Indian Equity ETF Mean-Reversion Pullback (Long-Only):
      - Entry: Close > EMA(200) AND RSI(2) < 15 AND IBS = (C-L)/(H-L) < 0.25
      - Exit: Close > prev High OR Close > SMA(5) OR 3.0x ATR hard stop
      - Sizing: up to 50% cash per ETF (max 2 concurrent), idle cash in LIQUIDBEES (6.5% p.a.)
    """
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []
    idle_pcts: List[float] = []
    last_ts_yield: Optional[pd.Timestamp] = None

    for ts in all_ts:
        if idle_yield_apr > 0 and last_ts_yield is not None:
            dt_days = max((ts - last_ts_yield).total_seconds() / 86400.0, 0.0)
            if dt_days > 0:
                alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                capital += max(capital - alloc, 0.0) * ((1.0 + idle_yield_apr) ** (dt_days / 365.0) - 1.0)
        last_ts_yield = ts

        for sym, df in prepared.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            prev = last_bar.get(sym)
            pos = pos_map[sym]

            if pos is not None and prev is not None:
                if float(prev["close"]) > float(prev["sma5"]) or float(prev["close"]) < pos["sl"]:
                    ex_px = float(row["open"]) * (1.0 - slippage)
                    pnl = (ex_px - pos["entry"]) * pos["qty"]
                    capital += pnl
                    trades.append(PortfolioTrade(
                        "IBS_RSI2", sym, "LONG", pos["entry_time"], ts,
                        pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.02, 1.0), "SMA5_OR_SL"
                    ))
                    pos_map[sym] = None
            elif pos is None and prev is not None and capital > 0:
                rng = float(prev["high"]) - float(prev["low"])
                ibs = (float(prev["close"]) - float(prev["low"])) / rng if rng > 1e-9 else 0.5
                if float(prev["close"]) > float(prev["ema200"]) and float(prev["rsi2"]) < 15.0 and ibs < 0.25:
                    alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                    entry = float(row["open"]) * (1.0 + slippage)
                    budget = min(capital * 0.5, max(capital - alloc, 0.0))
                    qty = math.floor(budget / entry)
                    if qty >= 1:
                        pos_map[sym] = {
                            "entry_time": ts, "entry": entry, "qty": qty,
                            "margin": qty * entry, "sl": entry - 3.0 * float(prev["atr14"]),
                        }
            last_bar[sym] = row

        alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
        unr = sum(
            (float(last_bar[s]["close"]) - p["entry"]) * p["qty"]
            for s, p in pos_map.items()
            if p is not None and s in last_bar
        )
        eq_now = capital + unr
        idle_pcts.append(max(capital - alloc, 0.0) / max(eq_now, 1.0) * 100.0)
        eq_recs.append((ts, eq_now))

    final_ts = all_ts[-1]
    for sym, pos in pos_map.items():
        if pos is not None and sym in last_bar:
            ex_px = float(last_bar[sym]["close"]) * (1.0 - slippage)
            pnl = (ex_px - pos["entry"]) * pos["qty"]
            capital += pnl
            trades.append(PortfolioTrade(
                "IBS_RSI2", sym, "LONG", pos["entry_time"], final_ts,
                pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.02, 1.0), "END_OF_TEST"
            ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"avg_idle_cash_pct": round(float(np.mean(idle_pcts)), 1)}


def run_suite(
    title: str,
    raw_all: Dict[str, pd.DataFrame],
    interval: str,
    start_str: str,
    end_str: str,
    slippage: float = 0.0005,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    ev_s = pd.Timestamp(start_str, tz="Asia/Kolkata")
    ev_e = pd.Timestamp(end_str, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)

    metals_raw = {k: raw_all[k] for k in METALS if k in raw_all}
    eq_raw = {k: raw_all[k] for k in EQUITIES if k in raw_all}
    prep_all = prepare_metal_features(raw_all, interval, ev_s, ev_e, warmup_bars=200)
    prep_metals = {k: prep_all[k] for k in metals_raw if k in prep_all}
    prep_eq = {k: prep_all[k] for k in eq_raw if k in prep_all}

    rows = []
    curves: Dict[str, pd.Series] = {}

    def rec(name: str, eq: pd.Series, trades: List[PortfolioTrade], note: str = ""):
        st = calc_curve_stats(name, eq, trades, CAPITAL_INR)
        st["Notes"] = note
        rows.append(st)
        curves[name] = eq

    # 0. Baseline: Unchanged Config C Metals Only (MCX 8x Gold Scen + 6.5% Liquid)
    cfg_m_mcx = make_kotak_cfg(slippage=slippage, long_only_assets={"SILVERBEES"}, max_leverage=8.0)
    t_bm, eq_bm, d_bm = run_config_c_with_cash_yield(metals_raw, cfg_m_mcx, ev_s, ev_e, idle_yield_apr=0.0)
    rec("0a. Baseline: Metals Config C (MCX 8x)", eq_bm, t_bm, f"Pyr={d_bm.get('pyramids',0)}")
    t_bml, eq_bml, d_bml = run_config_c_with_cash_yield(metals_raw, cfg_m_mcx, ev_s, ev_e, idle_yield_apr=LIQUID_YIELD_APR)
    rec("0b. Baseline: Metals Config C + Liquid", eq_bml, t_bml, f"Idle={d_bml.get('avg_idle_cash_pct',0)}%")

    # 1. Config C on 4 Equity ETFs only (+ Liquid)
    cfg_eq = make_kotak_cfg(slippage=slippage, long_only_assets=set(eq_raw.keys()), max_leverage=1.0)
    t_eqc, eq_eqc, d_eqc = run_config_c_with_cash_yield(eq_raw, cfg_eq, ev_s, ev_e, idle_yield_apr=LIQUID_YIELD_APR)
    rec("1. Equities Only: Config C + Liquid", eq_eqc, t_eqc, f"Idle={d_eqc.get('avg_idle_cash_pct',0)}%")

    # 2. Equities Only: Supertrend+ADX (+ Liquid)
    t_eqst, eq_eqst, d_eqst = run_supertrend_adx(prep_eq, set(prep_eq.keys()), slippage=slippage, max_leverage=1.0, idle_yield_apr=LIQUID_YIELD_APR)
    rec("2. Equities Only: Supertrend+ADX + Liq", eq_eqst, t_eqst, f"Idle={d_eqst.get('avg_idle_cash_pct',0)}%")

    # 3. Equities Only: IBS+RSI(2) Pullback (+ Liquid)
    t_eqmr, eq_eqmr, d_eqmr = run_ibs_rsi2_equity_pullback(prep_eq, slippage=slippage, idle_yield_apr=LIQUID_YIELD_APR)
    rec("3. Equities Only: IBS+RSI(2) + Liquid", eq_eqmr, t_eqmr, f"Idle={d_eqmr.get('avg_idle_cash_pct',0)}%")

    # 4. Equities Only: Dual-Mom Top-2 Vol-Parity Rotation (+ Liquid)
    t_eqrot, eq_eqrot, d_eqrot = run_top_k_vol_parity_rotation(prep_eq, interval, top_k=2, rebal_days=10, slippage=slippage, idle_yield_apr=LIQUID_YIELD_APR)
    rec("4. Equities Only: Top-2 VolParity + Liq", eq_eqrot, t_eqrot, f"Idle={d_eqrot.get('avg_idle_cash_pct',0)}%")

    # 5. ALL 6 ETFs: Config C Expanded (MCX 8x Gold Scen + 5 Spot ETFs + Liquid)
    lo_all6 = set(raw_all.keys()) - {"GOLDBEES"}
    cfg_a6 = make_kotak_cfg(slippage=slippage, long_only_assets=lo_all6, max_leverage=8.0)
    t_a6, eq_a6, d_a6 = run_config_c_with_cash_yield(raw_all, cfg_a6, ev_s, ev_e, idle_yield_apr=LIQUID_YIELD_APR)
    rec("5. ALL 6: Config C (MCX Gold) + Liquid", eq_a6, t_a6, f"Idle={d_a6.get('avg_idle_cash_pct',0)}%")

    # 6. ALL 6 ETFs: Top-2 Dual-Mom Vol-Parity Rotation (100% Spot Cash + Liquid)
    t_r2, eq_r2, d_r2 = run_top_k_vol_parity_rotation(prep_all, interval, top_k=2, rebal_days=10, slippage=slippage, idle_yield_apr=LIQUID_YIELD_APR)
    rec("6. ALL 6: Top-2 VolParity Rot + Liquid", eq_r2, t_r2, f"Idle={d_r2.get('avg_idle_cash_pct',0)}%")

    # 7. ALL 6 ETFs: Top-3 Dual-Mom Vol-Parity Rotation (100% Spot Cash + Liquid)
    t_r3, eq_r3, d_r3 = run_top_k_vol_parity_rotation(prep_all, interval, top_k=3, rebal_days=10, slippage=slippage, idle_yield_apr=LIQUID_YIELD_APR)
    rec("7. ALL 6: Top-3 VolParity Rot + Liquid", eq_r3, t_r3, f"Idle={d_r3.get('avg_idle_cash_pct',0)}%")

    # 8. ALL 6 ETFs: CTA TSMOM Vol-Target (MCX Gold 8x + 5 Spot + Liquid)
    t_ts, eq_ts, d_ts = run_tsmom_vol_target(prep_all, lo_all6, interval, slippage=slippage, max_leverage=8.0, idle_yield_apr=LIQUID_YIELD_APR)
    rec("8. ALL 6: CTA TSMOM (MCX Gold) + Liq", eq_ts, t_ts, f"Idle={d_ts.get('avg_idle_cash_pct',0)}%")

    # 9. Clean 50/50 Capital Split: Metals Config C (Rs 49k) + Equities Top-2 Rotation (Rs 49k)
    common = eq_bml.index.intersection(eq_eqrot.index)
    eq_split_rot = 0.5 * eq_bml.reindex(common).ffill() + 0.5 * eq_eqrot.reindex(common).ffill()
    rec("9. 50/50: Metals Config C + Eq Top-2", eq_split_rot, [], "Zero double-count")

    # 10. Clean 50/50 Capital Split: Metals Config C (Rs 49k) + Equities IBS+RSI2 (Rs 49k)
    common2 = eq_bml.index.intersection(eq_eqmr.index)
    eq_split_mr = 0.5 * eq_bml.reindex(common2).ffill() + 0.5 * eq_eqmr.reindex(common2).ffill()
    rec("10. 50/50: Metals Config C + Eq IBS/RSI2", eq_split_mr, [], "Zero double-count")

    # 11. Benchmarks
    eq_nifty = buy_and_hold_curve(prep_eq, {"NIFTYBEES": 1.0})
    rec("11. Benchmark: Buy & Hold NIFTYBEES", eq_nifty, [], "Passive INR")
    eq_bh4 = buy_and_hold_curve(prep_eq, {s: 1.0 / len(prep_eq) for s in prep_eq})
    rec("12. Benchmark: B&H Equal-Wt 4 Equities", eq_bh4, [], "Passive INR")
    eq_bh6 = buy_and_hold_curve(prep_all, {s: 1.0 / len(prep_all) for s in prep_all})
    rec("13. Benchmark: B&H Equal-Wt All 6 ETFs", eq_bh6, [], "Passive INR")

    df_res = pd.DataFrame(rows)
    print(f"\n================ {title} ({start_str} -> {end_str}, slip={slippage*100:.2f}%) ================")
    print(df_res.to_string(index=False))
    return df_res, curves


def main():
    out_dir = new_run_directory("kotak-nse-etfs-upstox")

    raw_1d = {sym: load_cached("NSE", sym, "1d") for sym in ALL6}
    raw_6h = {sym: load_cached("NSE", sym, "6h") for sym in ALL6}

    # 1. In-Sample 2Y Daily (2022-12-01 -> 2024-11-23)
    df_is, _ = run_suite("IN-SAMPLE 2Y DAILY (All 6 ETFs)", raw_1d, "1d", "2022-12-01", "2024-11-23", slippage=0.0005)

    # 2. Out-of-Sample 2Y Daily (2024-11-24 -> 2026-11-23)
    df_oos_1d, _ = run_suite("OUT-OF-SAMPLE 2Y DAILY (All 6 ETFs)", raw_1d, "1d", "2024-11-24", "2026-11-23", slippage=0.0005)

    # 3. Full ~4Y Daily (2022-12-01 -> 2026-11-23)
    df_4y, curves_4y = run_suite("FULL ~4-YEAR DAILY (All 6 ETFs)", raw_1d, "1d", "2022-12-01", "2026-11-23", slippage=0.0005)

    # 4. Stress Test 0.15%/side Full ~4Y Daily
    df_stress, _ = run_suite("STRESS TEST 0.15%/SIDE ~4Y DAILY (All 6 ETFs)", raw_1d, "1d", "2022-12-01", "2026-11-23", slippage=0.0015)

    # 5. 2-Year 6H (2024-11-24 -> 2026-11-23)
    df_2y_6h, _ = run_suite("2-YEAR 6H (All 6 ETFs)", raw_6h, "6h", "2024-11-24", "2026-11-23", slippage=0.0005)

    # 6. Full 5-Year Daily (2021-11-24 -> 2026-11-23) on 5 ETFs (ex-SILVERBEES)
    raw_5y = {k: v for k, v in raw_1d.items() if k != "SILVERBEES"}
    df_5y, _ = run_suite("FULL 5-YEAR DAILY (5 ETFs: GOLDBEES + 4 Equities)", raw_5y, "1d", "2021-11-24", "2026-11-23", slippage=0.0005)

    print("\n================ CALENDAR YEAR RETURNS (~4Y DAILY, 0.05% SLIP) ================")
    for name, eq in curves_4y.items():
        print(f"{name:40s} | {yearly_returns(eq)}")

    # Correlation between Metals Config C and Equity strategies
    corr_df = pd.DataFrame({
        "Metals_CfgC": curves_4y["0b. Baseline: Metals Config C + Liquid"].pct_change(),
        "Eq_CfgC": curves_4y["1. Equities Only: Config C + Liquid"].pct_change(),
        "Eq_ST_ADX": curves_4y["2. Equities Only: Supertrend+ADX + Liq"].pct_change(),
        "Eq_IBS_RSI2": curves_4y["3. Equities Only: IBS+RSI(2) + Liquid"].pct_change(),
        "Eq_Top2_Rot": curves_4y["4. Equities Only: Top-2 VolParity + Liq"].pct_change(),
        "All6_Top2_Rot": curves_4y["6. ALL 6: Top-2 VolParity Rot + Liquid"].pct_change(),
        "All6_TSMOM": curves_4y["8. ALL 6: CTA TSMOM (MCX Gold) + Liq"].pct_change(),
    }).dropna().corr().round(2)
    print("\n================ DAILY RETURN CORRELATION MATRIX (~4Y) ================")
    print(corr_df.to_string())

    df_is.to_csv(out_dir / "is_2y_1d.csv", index=False)
    df_oos_1d.to_csv(out_dir / "oos_2y_1d.csv", index=False)
    df_4y.to_csv(out_dir / "full_4y_1d.csv", index=False)
    df_stress.to_csv(out_dir / "stress_4y_1d.csv", index=False)
    df_2y_6h.to_csv(out_dir / "oos_2y_6h.csv", index=False)
    df_5y.to_csv(out_dir / "full_5y_5etfs_1d.csv", index=False)
    pd.DataFrame(curves_4y).to_csv(out_dir / "equity_curves_4y_1d.csv")
    print(f"\nSaved all tables and curves to {out_dir}/")


if __name__ == "__main__":
    main()
