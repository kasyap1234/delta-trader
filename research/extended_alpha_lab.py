"""
Extended Alpha Lab: 4 New Strategy Studies (Actual Upstox V3 + Binance 5Y Data)
===============================================================================
1. Expanded 12+ NSE Sector & Factor ETFs Rotation vs Current 6-ETF Baseline (Rs 98k, 4Y IS/OOS + Stress)
2. Nifty-100 Sample (40 liquid large-caps) Top-5 Risk-Adjusted Stock Momentum (Rs 98k, 4Y & 5Y IS/OOS + Stress)
   *Explicitly discloses survivorship bias (fixed current-constituent sample).*
3. Crypto Perp Open-Interest + Funding Squeeze Composite (checks actual Binance OI window + 5Y actual Funding+Volume Squeeze)
4. Delta BTC/ETH Defined-Risk Weekly Option Selling (VRP in Chop) — EXPLICITLY LABELED MODELED BSM SCENARIO
   (No historical Delta option-chain quotes exist; models weekly defined-risk Iron Condors on actual 5Y BTC/ETH candles.)
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio_engine import PortfolioTrade, _bsm_price, add_ema_atr_features
from research.crypto_idle_margin_study import (
    TOP5_COINS,
    _actual_funding_cashflow,
    _adx,
    _round_lot,
    load_binance_funding,
    load_cached_binance_klines,
)
from research.kotak_metals_upstox_study import (
    CAPITAL_INR,
    LIQUID_YIELD_APR,
    buy_and_hold_curve,
    load_cached,
    prepare_metal_features,
    yearly_returns,
)
from research.kotak_nse_etf_upstox_study import ALL6, run_top_k_vol_parity_rotation
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

ETF_CACHE = Path(__file__).resolve().parent / "output" / "_cache_upstox_metals"
STOCK_CACHE = Path(__file__).resolve().parent / "output" / "_cache_upstox_nifty100"
OI_CACHE = Path(__file__).resolve().parent / "output" / "_cache_binance_oi"


def load_csv_ist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    df.index.name = "timestamp"
    return df.sort_index()


def run_risk_adj_rotation(
    prepared: Dict[str, pd.DataFrame],
    top_k: int = 3,
    rebal_days: int = 10,
    rank_by_sharpe: bool = False,
    use_6m_skip5: bool = False,
    regime_df: Optional[pd.DataFrame] = None,
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    idle_yield_apr: float = LIQUID_YIELD_APR,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Generalized Long-Only Inverse-Vol Momentum Rotation (for ETFs or Cash Stocks):
      - Optional skip-5d 6M momentum (`use_6m_skip5=True`) or blended 21d/60d momentum
      - Optional Sharpe ranking (`mom / vol_ann`)
      - Optional Nifty > SMA(200) regime gate (`regime_df`)
      - Unfilled slots earn modeled 6.5% p.a. LIQUIDBEES yield.
    """
    # Precompute 6M skip-5d return if needed
    prepped = {}
    for sym, df in prepared.items():
        f = df.copy()
        if use_6m_skip5:
            f["ret_6m_s5"] = f["close"].shift(5) / f["close"].shift(126) - 1.0
            f["vol_60"] = f["close"].pct_change().rolling(60).std() * np.sqrt(252.0)
        prepped[sym] = f

    regime_ok_map: Optional[pd.Series] = None
    if regime_df is not None:
        r_close = regime_df["close"]
        regime_ok_map = (r_close > r_close.rolling(200).mean())

    from config import PortfolioConfig
    cfg = PortfolioConfig(whole_lot_rounding=True)
    all_ts = sorted(set().union(*[df.index for df in prepped.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepped}
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

        if bar_i % rebal_days == 0 and len(last_bar) >= 2:
            mkt_ok = True
            if regime_ok_map is not None:
                prior_r = regime_ok_map.loc[regime_ok_map.index < ts]
                if not prior_r.empty and not bool(prior_r.iloc[-1]):
                    mkt_ok = False

            cands = []
            if mkt_ok:
                for sym, prev in last_bar.items():
                    if ts not in prepped[sym].index:
                        continue
                    if use_6m_skip5:
                        mom = float(prev.get("ret_6m_s5", np.nan))
                        vol = float(prev.get("vol_60", np.nan))
                    else:
                        mom = 0.5 * float(prev["ret_med"]) + 0.5 * float(prev["ret_slow"])
                        vol = float(prev["vol_ann"])
                    if not math.isfinite(mom) or mom <= 0 or float(prev["close"]) <= float(prev["ema50"]):
                        continue
                    if not math.isfinite(vol) or vol < 0.05:
                        vol = 0.15
                    score = (mom / vol) if rank_by_sharpe else mom
                    cands.append((sym, score, vol))
                cands.sort(key=lambda x: x[1], reverse=True)
            chosen = cands[:top_k]
            chosen_syms = {c[0] for c in chosen}

            for sym, pos in list(pos_map.items()):
                if pos is not None and sym not in chosen_syms and ts in prepped[sym].index:
                    ex_px = float(prepped[sym].loc[ts]["open"]) * (1.0 - slippage)
                    pnl = (ex_px - pos["entry"]) * pos["qty"]
                    capital += pnl
                    trades.append(PortfolioTrade(
                        "ROT_EXT", sym, "LONG", pos["entry_time"], ts,
                        pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.025, 1.0), "ROTATE_OUT"
                    ))
                    pos_map[sym] = None

            if chosen:
                inv_vols = {s: 1.0 / v for s, _, v in chosen}
                iv_sum = sum(inv_vols.values())
                invest_frac = len(chosen) / float(top_k)
                mtm_eq = capital + sum(
                    (float(prepped[s].loc[ts]["open"]) - p["entry"]) * p["qty"]
                    for s, p in pos_map.items()
                    if p is not None and ts in prepped[s].index
                )
                for sym, _, _ in chosen:
                    if pos_map[sym] is not None:
                        continue
                    w = invest_frac * (inv_vols[sym] / iv_sum)
                    open_px = float(prepped[sym].loc[ts]["open"])
                    entry = open_px * (1.0 + slippage)
                    alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                    budget = min(mtm_eq * w, max(capital - alloc, 0.0))
                    qty = math.floor(budget / entry)
                    if qty >= 1:
                        pos_map[sym] = {"entry_time": ts, "entry": entry, "qty": qty, "margin": qty * entry}

        for sym, df in prepped.items():
            if ts in df.index:
                last_bar[sym] = df.loc[ts]

        alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
        unr = sum((float(last_bar[s]["close"]) - p["entry"]) * p["qty"] for s, p in pos_map.items() if p is not None and s in last_bar)
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
                "ROT_EXT", sym, "LONG", pos["entry_time"], final_ts,
                pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.025, 1.0), "END_OF_TEST"
            ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"avg_idle_cash_pct": round(float(np.mean(idle_pcts)), 1)}


def run_crypto_funding_vol_squeeze_5y(
    dfs: Dict[str, pd.DataFrame],
    funding_map: Dict[str, pd.Series],
    start_str: str,
    end_str: str,
    capital_init: float = 1200.0,
    entry_slip: float = 0.0004,
    exit_slip: float = 0.0015,
) -> Tuple[List[PortfolioTrade], pd.Series]:
    """
    5-Year Crypto Funding + Volume Squeeze Composite on Top-5 Binance USD-M 6h candles
    (with actual historical 8h funding rates):
      - LONG squeeze: 20-bar breakout up + volume > 1.5x 20-bar avg + 3d avg funding <= 0.005% (non-crowded / short squeeze)
      - SHORT squeeze: 20-bar breakdown + volume > 1.5x 20-bar avg + 3d avg funding >= 0.015% (crowded longs unwinding)
      - Exit: 3.0x ATR trailing stop, 2.0% risk/trade, 8.0% max risk, whole lots.
    """
    ev_s = pd.Timestamp(start_str, tz="Asia/Kolkata")
    ev_e = pd.Timestamp(end_str, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)

    prepped = {}
    for sym, raw in dfs.items():
        f = add_ema_atr_features(raw)
        f["dc_hi"] = f["high"].rolling(20).max().shift(1)
        f["dc_lo"] = f["low"].rolling(20).min().shift(1)
        f["vol_ma"] = f["volume"].rolling(20).mean().shift(1)
        # Align 3-day rolling average actual 8h funding rate strictly prior to bar
        fr = funding_map.get(sym, pd.Series(dtype=float))
        if not fr.empty:
            fr_3d = fr.rolling(9, min_periods=1).mean()
            f["fr_3d"] = fr_3d.reindex(f.index, method="ffill").fillna(0.0001)
        else:
            f["fr_3d"] = 0.0001
        f = f.iloc[200:].dropna()
        f = f[(f.index >= ev_s) & (f.index <= ev_e)]
        if not f.empty:
            prepped[sym] = f

    from config import PortfolioConfig
    cfg = PortfolioConfig(whole_lot_rounding=True)
    all_ts = sorted(set().union(*[df.index for df in prepped.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepped}
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []

    for ts in all_ts:
        for sym, df in prepped.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            prev = last_bar.get(sym)
            if pos_map[sym] is None and prev is not None and prev["atr14"] > 0:
                vol_surge = float(prev["volume"]) > 1.5 * float(prev["vol_ma"])
                side = None
                if vol_surge and float(prev["close"]) > float(prev["dc_hi"]) and float(prev["fr_3d"]) <= 0.00005:
                    side = "LONG"
                elif vol_surge and float(prev["close"]) < float(prev["dc_lo"]) and float(prev["fr_3d"]) >= 0.00015:
                    side = "SHORT"
                if side is not None and capital > 0:
                    active = [p for p in pos_map.values() if p is not None]
                    if sum(p["risk_pct"] for p in active) + 0.02 <= 0.08 + 1e-9:
                        entry = float(row["open"]) * (1.0 + entry_slip if side == "LONG" else 1.0 - entry_slip)
                        sl_dist = 3.0 * float(prev["atr14"])
                        raw_qty = (capital * 0.02) / sl_dist
                        alloc = sum(p["margin"] for p in active)
                        budget = max(capital * 0.75 - alloc, 0.0)
                        raw_qty = min(raw_qty, (budget * 5.0) / entry)
                        qty = _round_lot(sym, raw_qty, cfg)
                        if qty > 0:
                            risk_rs = max(qty * sl_dist, 1e-6)
                            pos_map[sym] = {
                                "side": side, "entry_time": ts, "entry": entry, "qty": qty,
                                "sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
                                "margin": qty * entry / 5.0, "risk_rs": risk_rs, "risk_pct": risk_rs / capital,
                                "funding_paid": 0.0, "last_f_ts": ts,
                            }
            last_bar[sym] = row

        for sym, pos in pos_map.items():
            row = last_bar.get(sym)
            if pos is None or row is None or row.name != ts:
                continue
            fr_s = funding_map.get(sym, pd.Series(dtype=float))
            if not fr_s.empty:
                r_sum = float(fr_s.loc[(fr_s.index > pos["last_f_ts"]) & (fr_s.index <= ts)].sum())
                sign_f = 1.0 if pos["side"] == "LONG" else -1.0
                pos["funding_paid"] += r_sum * pos["qty"] * float(row["close"]) * sign_f
            pos["last_f_ts"] = ts
            is_long = pos["side"] == "LONG"
            stop = pos["sl"]
            hit = row["low"] <= stop if is_long else row["high"] >= stop
            if hit:
                gapped = row["open"] < stop if is_long else row["open"] > stop
                raw_ex = float(row["open"]) if gapped else stop
                ex_px = raw_ex * (1.0 - exit_slip if is_long else 1.0 + exit_slip)
                gross = ((ex_px - pos["entry"]) if is_long else (pos["entry"] - ex_px)) * pos["qty"]
                net = gross - pos["funding_paid"]
                capital += net
                trades.append(PortfolioTrade("SQUEEZE", sym, pos["side"], pos["entry_time"], ts, pos["entry"], ex_px, pos["qty"], gross, pos["funding_paid"], net, net / pos["risk_rs"], "TRAIL"))
                pos_map[sym] = None
            else:
                trail = 3.0 * float(row["atr14"])
                c = float(row["close"])
                pos["sl"] = max(stop, c - trail) if is_long else min(stop, c + trail)

        unr = sum(
            (((float(last_bar[s]["close"]) - p["entry"]) if p["side"] == "LONG" else (p["entry"] - float(last_bar[s]["close"]))) * p["qty"] - p["funding_paid"])
            for s, p in pos_map.items()
            if p is not None and s in last_bar
        )
        eq_recs.append((ts, capital + unr))

    eq_s = pd.Series([v for _, v in eq_recs], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill()


def run_modeled_crypto_vrp_iron_condor(
    btc_6h: pd.DataFrame,
    eth_6h: pd.DataFrame,
    start_str: str,
    end_str: str,
    capital_init: float = 1200.0,
    iv_mult: float = 1.15,
    slip_frac: float = 0.06,
    stress_loss_pen: float = 0.0,
    chop_only: bool = True,
) -> Tuple[List[PortfolioTrade], pd.Series]:
    """
    MODELED BSM SCENARIO ONLY (NOT OBSERVED OPTION QUOTES):
      - Simulates weekly 7-DTE defined-risk OTM Iron Condors on BTC + ETH ($1,200 account)
      - Short strikes at +/- 1.0 SD, protective wings at +/- 1.8 SD
      - Risk capped at 2.0% of equity per weekly condor per coin (1.0% per side)
      - Optional chop gate (`chop_only=True`): enters only when ADX(14) < 25 (sideways regime)
      - Take profit at 55% of credit collected; stop-loss at 2.0x credit or expiry settlement.
    """
    ev_s = pd.Timestamp(start_str, tz="Asia/Kolkata")
    ev_e = pd.Timestamp(end_str, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)

    daily_map = {}
    for sym, raw in [("BTCUSDT", btc_6h), ("ETHUSDT", eth_6h)]:
        d = raw[["open", "high", "low", "close", "volume"]].resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        d["ret"] = np.log(d["close"] / d["close"].shift(1))
        d["rv20"] = d["ret"].rolling(20).std() * np.sqrt(365.0)
        d["adx14"] = _adx(d, 14)
        d = d.dropna()
        d = d[(d.index >= ev_s) & (d.index <= ev_e)]
        daily_map[sym] = d

    common = daily_map["BTCUSDT"].index.intersection(daily_map["ETHUSDT"].index)
    capital = capital_init
    trades: List[PortfolioTrade] = []
    pnl_by_day = pd.Series(0.0, index=common)

    for sym in ["BTCUSDT", "ETHUSDT"]:
        d = daily_map[sym].reindex(common)
        i = 0
        while i < len(d) - 7:
            row = d.iloc[i]
            if chop_only and float(row["adx14"]) >= 25.0:
                i += 1
                continue
            S0 = float(row["close"])
            iv = max(float(row["rv20"]) * iv_mult, 0.25)
            T0 = 7.0 / 365.0
            w_sd = S0 * iv * math.sqrt(T0)

            k_sp, k_lp = S0 - 1.0 * w_sd, S0 - 1.8 * w_sd
            k_sc, k_lc = S0 + 1.0 * w_sd, S0 + 1.8 * w_sd
            put_cr = _bsm_price(S0, k_sp, T0, iv, "P", r=0.0) - _bsm_price(S0, k_lp, T0, iv, "P", r=0.0)
            call_cr = _bsm_price(S0, k_sc, T0, iv, "C", r=0.0) - _bsm_price(S0, k_lc, T0, iv, "C", r=0.0)
            credit = (put_cr + call_cr) * (1.0 - slip_frac)
            wing_w = 0.8 * w_sd
            max_loss_u = max(wing_w - credit, 1e-6)
            if credit <= 0:
                i += 1
                continue

            risk_usd = capital * 0.02
            qty = risk_usd / max_loss_u
            exit_idx = min(i + 7, len(d) - 1)
            exit_reason = "EXPIRY"
            exit_val = 0.0
            for day_off in range(1, 8):
                if i + day_off >= len(d):
                    break
                Sc = float(d.iloc[i + day_off]["close"])
                Trem = max((7.0 - day_off) / 365.0, 0.0)
                ivc = max(float(d.iloc[i + day_off]["rv20"]) * iv_mult, 0.25)
                pv = _bsm_price(Sc, k_sp, Trem, ivc, "P", r=0.0) - _bsm_price(Sc, k_lp, Trem, ivc, "P", r=0.0)
                cv = _bsm_price(Sc, k_sc, Trem, ivc, "C", r=0.0) - _bsm_price(Sc, k_lc, Trem, ivc, "C", r=0.0)
                val = pv + cv
                if val <= credit * 0.45:
                    exit_idx, exit_val, exit_reason = i + day_off, val, "TP_55PCT"
                    break
                if val >= credit * 2.2:
                    exit_idx, exit_val, exit_reason = i + day_off, val, "SL_2.2X"
                    break
                exit_idx, exit_val = i + day_off, val

            net_u = credit - exit_val * (1.0 + slip_frac)
            if net_u < 0 and stress_loss_pen > 0:
                net_u *= (1.0 + stress_loss_pen)
            net_u = max(net_u, -max_loss_u)
            net_pnl = net_u * qty
            capital += net_pnl
            pnl_by_day.iloc[exit_idx] += net_pnl
            trades.append(PortfolioTrade(
                "VRP_MOD", sym, "IRON_CONDOR", d.index[i], d.index[exit_idx],
                S0, float(d.iloc[exit_idx]["close"]), qty, net_pnl, 0.0, net_pnl, net_pnl / risk_usd, exit_reason
            ))
            i = exit_idx + 1

    eq = capital_init + pnl_by_day.cumsum()
    return trades, eq


def main():
    out_dir = new_run_directory("extended-alpha-lab")

    # =========================================================================
    # EXPERIMENT 1: EXPANDED 12+ NSE SECTOR & FACTOR ETFs vs CURRENT 6-ETF BASELINE
    # =========================================================================
    etf_files = sorted(ETF_CACHE.glob("NSE_*_1d.csv"))
    all_etfs: Dict[str, pd.DataFrame] = {}
    for f in etf_files:
        sym = f.stem.replace("NSE_", "").replace("_1d", "")
        if sym == "LIQUIDBEES":
            continue
        df = load_csv_ist(f)
        # Keep ETFs with history starting on or before 2022-03-01 (so they have >=200 warmup bars by 2022-12-01)
        if len(df) >= 1100 and df.index[0] <= pd.Timestamp("2022-03-01", tz="Asia/Kolkata"):
            all_etfs[sym] = df

    print(f"\n================ EXPERIMENT 1: EXPANDED NSE ETFs ({len(all_etfs)} ETFs) vs CURRENT 6-ETF BASELINE ================")
    print(f"Eligible ETFs (>=4Y history): {sorted(all_etfs.keys())}")

    for label, s_d, e_d, slip in [
        ("IS 2Y (2022-12 -> 2024-11, 0.05%)", "2022-12-01", "2024-11-23", 0.0005),
        ("OOS 2Y (2024-11 -> 2026-11, 0.05%)", "2024-11-24", "2026-11-23", 0.0005),
        ("FULL 4Y (2022-12 -> 2026-11, 0.05%)", "2022-12-01", "2026-11-23", 0.0005),
        ("STRESS 4Y (2022-12 -> 2026-11, 0.15%)", "2022-12-01", "2026-11-23", 0.0015),
    ]:
        ev_s = pd.Timestamp(s_d, tz="Asia/Kolkata")
        ev_e = pd.Timestamp(e_d, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        prep_all = prepare_metal_features(all_etfs, "1d", ev_s, ev_e, warmup_bars=200)
        prep_6 = {k: prep_all[k] for k in ALL6 if k in prep_all}
        rows = []
        t0, eq0, d0 = run_top_k_vol_parity_rotation(prep_6, "1d", top_k=2, rebal_days=10, slippage=slip)
        rows.append(calc_curve_stats("0. Current 6-ETF Top-2 VolParity", eq0, t0, CAPITAL_INR))
        t1, eq1, d1 = run_risk_adj_rotation(prep_all, top_k=2, rebal_days=10, rank_by_sharpe=False, slippage=slip)
        rows.append(calc_curve_stats(f"1. {len(prep_all)}-ETF Top-2 RawMom", eq1, t1, CAPITAL_INR))
        t2, eq2, d2 = run_risk_adj_rotation(prep_all, top_k=3, rebal_days=10, rank_by_sharpe=False, slippage=slip)
        rows.append(calc_curve_stats(f"2. {len(prep_all)}-ETF Top-3 RawMom", eq2, t2, CAPITAL_INR))
        t3, eq3, d3 = run_risk_adj_rotation(prep_all, top_k=2, rebal_days=10, rank_by_sharpe=True, slippage=slip)
        rows.append(calc_curve_stats(f"3. {len(prep_all)}-ETF Top-2 SharpeMom", eq3, t3, CAPITAL_INR))
        t4, eq4, d4 = run_risk_adj_rotation(prep_all, top_k=3, rebal_days=10, rank_by_sharpe=True, slippage=slip)
        rows.append(calc_curve_stats(f"4. {len(prep_all)}-ETF Top-3 SharpeMom", eq4, t4, CAPITAL_INR))
        print(f"\n--- {label} ---")
        print(pd.DataFrame(rows).to_string(index=False))
        if "FULL 4Y" in label:
            print("Calendar Years (Full 4Y):")
            for nm, c in [("6-ETF Top-2", eq0), (f"{len(prep_all)}-ETF Top-2 Raw", eq1), (f"{len(prep_all)}-ETF Top-3 Raw", eq2), (f"{len(prep_all)}-ETF Top-2 Sharpe", eq3)]:
                print(f"  {nm:24s} | {yearly_returns(c)}")
            eq_etf_best = eq1

    # =========================================================================
    # EXPERIMENT 2: NIFTY-100 SAMPLE (40 STOCKS) TOP-5 RISK-ADJUSTED MOMENTUM
    # =========================================================================
    stock_files = sorted(STOCK_CACHE.glob("NSE_*_1d.csv"))
    stocks: Dict[str, pd.DataFrame] = {}
    for f in stock_files:
        sym = f.stem.replace("NSE_", "").replace("_1d", "")
        df = load_csv_ist(f)
        if len(df) >= 1200:
            stocks[sym] = df
    nifty_df = all_etfs["NIFTYBEES"]

    print(f"\n================ EXPERIMENT 2: NIFTY-100 ({len(stocks)} LIQUID STOCKS) TOP-5 RISK-ADJ MOMENTUM ================")
    print("CAVEAT: Fixed current-constituent sample (contains survivorship bias; treat as upper-bound research screen).")
    for label, s_d, e_d, slip in [
        ("IS 2Y (2022-12 -> 2024-11, 0.05%)", "2022-12-01", "2024-11-23", 0.0005),
        ("OOS 2Y (2024-11 -> 2026-11, 0.05%)", "2024-11-24", "2026-11-23", 0.0005),
        ("FULL 4Y (2022-12 -> 2026-11, 0.05%)", "2022-12-01", "2026-11-23", 0.0005),
        ("STRESS 4Y (2022-12 -> 2026-11, 0.15%)", "2022-12-01", "2026-11-23", 0.0015),
        ("FULL 5Y (2021-11 -> 2026-11, 0.05%)", "2021-11-24", "2026-11-23", 0.0005),
    ]:
        ev_s = pd.Timestamp(s_d, tz="Asia/Kolkata")
        ev_e = pd.Timestamp(e_d, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        prep_st = prepare_metal_features(stocks, "1d", ev_s, ev_e, warmup_bars=200)
        rows = []
        # 1. Top-5 6M skip-5d Sharpe Mom (Monthly 20d, Nifty>SMA200 gate)
        ts1, eqs1, _ = run_risk_adj_rotation(prep_st, top_k=5, rebal_days=20, rank_by_sharpe=True, use_6m_skip5=True, regime_df=nifty_df, slippage=slip)
        rows.append(calc_curve_stats("1. Top-5 6M-Sharpe (20d, Nifty>200)", eqs1, ts1, CAPITAL_INR))
        # 2. Top-5 6M skip-5d Sharpe Mom (Bi-Weekly 10d, no Nifty gate)
        ts2, eqs2, _ = run_risk_adj_rotation(prep_st, top_k=5, rebal_days=10, rank_by_sharpe=True, use_6m_skip5=True, regime_df=None, slippage=slip)
        rows.append(calc_curve_stats("2. Top-5 6M-Sharpe (10d, Unfiltered)", eqs2, ts2, CAPITAL_INR))
        # 3. Top-5 21d/60d Sharpe Mom (10d, Nifty>200 gate)
        ts3, eqs3, _ = run_risk_adj_rotation(prep_st, top_k=5, rebal_days=10, rank_by_sharpe=True, use_6m_skip5=False, regime_df=nifty_df, slippage=slip)
        rows.append(calc_curve_stats("3. Top-5 21/60d-Sharpe (10d, >200)", eqs3, ts3, CAPITAL_INR))
        print(f"\n--- {label} ---")
        print(pd.DataFrame(rows).to_string(index=False))
        if "FULL 4Y" in label:
            print("Calendar Years (Full 4Y):")
            for nm, c in [("Top-5 6M-Sharpe (20d, >200)", eqs1), ("Top-5 6M-Sharpe (10d)", eqs2)]:
                print(f"  {nm:30s} | {yearly_returns(c)}")
            common_4y = eq_etf_best.index.intersection(eqs1.index)
            blend_50 = 0.5 * eq_etf_best.reindex(common_4y).ffill() + 0.5 * eqs1.reindex(common_4y).ffill()
            st_b = calc_curve_stats("50/50 Blend: Expanded ETFs + Top-5 Stocks", blend_50, [], CAPITAL_INR)
            print(f"  50/50 Blend (4Y): Net={st_b['Net Return %']:+.1f}% | CAGR={st_b['CAGR %']:+.2f}% | MaxDD={st_b['Max DD %']:.2f}% | Calmar={st_b['Calmar']:.2f} | Sharpe={st_b['Sharpe']:.2f}")

    # =========================================================================
    # EXPERIMENT 3: CRYPTO OI + FUNDING SQUEEZE COMPOSITE
    # =========================================================================
    print("\n================ EXPERIMENT 3: CRYPTO OI + FUNDING SQUEEZE COMPOSITE ================")
    oi_files = sorted(OI_CACHE.glob("*_oi_6h.csv"))
    for f in oi_files:
        df_oi = pd.read_csv(f, index_col=0, parse_dates=True)
        print(f"  Actual Binance OI history ({f.stem}): {len(df_oi)} 6h bars ({df_oi.index[0].strftime('%Y-%m-%d')} -> {df_oi.index[-1].strftime('%Y-%m-%d')})")
    print("  NOTE: Binance public openInterestHist only serves ~30 days — too short for multi-year OI backtesting.")
    print("  Testing 5-Year Funding + Volume Squeeze Composite (actual 8h funding + 6h volume surge, 2021-11 -> 2026-11):")

    bin_dfs = {sym: load_cached_binance_klines(sym, "6h", "2021-11-24", "2026-11-23") for sym in TOP5_COINS}
    fund_map = {sym: load_binance_funding(sym, "2021-08-01", "2026-11-23") for sym in TOP5_COINS}
    sq_rows = []
    for lbl, s_d, e_d in [
        ("IS 3Y (2021-11 -> 2024-11)", "2021-11-24", "2024-11-23"),
        ("OOS 2Y (2024-11 -> 2026-11)", "2024-11-24", "2026-11-23"),
        ("FULL 5Y (2021-11 -> 2026-11)", "2021-11-24", "2026-11-23"),
    ]:
        tsq, eqsq = run_crypto_funding_vol_squeeze_5y(bin_dfs, fund_map, s_d, e_d)
        sq_rows.append(calc_curve_stats(f"Funding+Vol Squeeze | {lbl}", eqsq, tsq, 1200.0))
    print(pd.DataFrame(sq_rows).to_string(index=False))

    # =========================================================================
    # EXPERIMENT 4: DELTA BTC/ETH WEEKLY VRP OPTIONS — NO MODELED SIMULATION
    # =========================================================================
    print("\n================ EXPERIMENT 4: DELTA BTC/ETH WEEKLY VRP OPTIONS ================")
    print("SKIPPED (NO SIMULATION): Historical Delta Exchange India option-chain quotes are not available in this cache.")
    print(f"\nRun folder: {out_dir}")


if __name__ == "__main__":
    main()
