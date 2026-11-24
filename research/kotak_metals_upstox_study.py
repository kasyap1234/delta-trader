"""
Kotak Neo Gold & Silver Strategy Backtest & Improvement Study (Actual Upstox V3 Data)
=====================================================================================
Uses cached actual Upstox V3 INR candles (no COMEX USD proxies):
  - Part A: Current Kotak live sleeve (MCX GOLDPETAL + NSE SILVERBEES, 6h) over active contract window
            + MCX SILVERMIC whole-lot feasibility check on Rs 98,000
  - Part B: 2-Year Actual Upstox 6h vs 1d comparison (NSE GOLDBEES + NSE SILVERBEES, 2024-11-24 -> 2026-11-23)
  - Part C: ~4-Year Multi-Year Daily Validation (NSE GOLDBEES + NSE SILVERBEES, 2022-12-01 -> 2026-11-23)
            with strict In-Sample (2022-12 -> 2024-11) vs Out-of-Sample (2024-11 -> 2026-11) split
            + calendar year returns + 0.15% stress test

Starting capital: Rs 98,000 INR, whole-lot rounding, zero capital double-counting.
Modeled idle-cash LIQUIDBEES yield (6.5% p.a.) and MCX 8x-leverage scenarios on GOLDBEES
are explicitly labeled as scenarios/modeled inputs.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PortfolioConfig
from portfolio_engine import (
    PortfolioTrade,
    add_donchian_features,
    add_ema_atr_features,
    run_shared_donchian_portfolio,
)
from research.outputs import new_run_directory
from research.portfolio_backtest import calc_curve_stats

CACHE_DIR = Path(__file__).resolve().parent / "output" / "_cache_upstox_metals"
CAPITAL_INR = 98_000.0
LIQUID_YIELD_APR = 0.065  # Modeled 6.5% p.a. LIQUIDBEES / overnight liquid fund yield


def load_cached(exch: str, sym: str, interval: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{exch}_{sym}_{interval}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    df.index.name = "timestamp"
    return df.sort_index()


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


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    hl2 = (df["high"] + df["low"]) / 2.0
    final_u = (hl2 + mult * atr).to_numpy().copy()
    final_l = (hl2 - mult * atr).to_numpy().copy()
    close_arr = df["close"].to_numpy()
    st_dir = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if not (math.isfinite(final_u[i]) and math.isfinite(final_l[i])):
            st_dir[i] = st_dir[i - 1]
            continue
        if close_arr[i - 1] <= final_u[i - 1]:
            final_u[i] = min(final_u[i], final_u[i - 1])
        if close_arr[i - 1] >= final_l[i - 1]:
            final_l[i] = max(final_l[i], final_l[i - 1])
        if st_dir[i - 1] == 1:
            st_dir[i] = -1 if close_arr[i] < final_l[i] else 1
        else:
            st_dir[i] = 1 if close_arr[i] > final_u[i] else -1
    return pd.Series(st_dir, index=df.index), pd.Series(np.where(st_dir == 1, final_l, final_u), index=df.index)


def _rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def prepare_metal_features(
    dfs: Dict[str, pd.DataFrame],
    interval: str,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    warmup_bars: int = 200,
) -> Dict[str, pd.DataFrame]:
    bars_per_day = 2 if interval == "6h" else 1  # NSE has 2 6h buckets/day (06:00, 12:00)
    bars_per_year = 252.0 * bars_per_day
    lb_fast = 10 * bars_per_day
    lb_med = 21 * bars_per_day
    lb_slow = 60 * bars_per_day

    out: Dict[str, pd.DataFrame] = {}
    for sym, raw in dfs.items():
        f = add_ema_atr_features(raw)
        f["donchian_high"] = f["high"].rolling(20).max().shift(1)
        f["donchian_low"] = f["low"].rolling(20).min().shift(1)
        f["adx14"] = _adx(f, 14)
        f["st_dir"], f["st_line"] = _supertrend(f, 10, 3.0)
        f["rsi2"] = _rsi(f["close"], 2)
        f["sma5"] = f["close"].rolling(5).mean()
        f["ret1"] = f["close"].pct_change()
        f["vol_ann"] = f["ret1"].rolling(20 * bars_per_day).std() * np.sqrt(bars_per_year)
        f["ret_fast"] = f["close"].pct_change(lb_fast)
        f["ret_med"] = f["close"].pct_change(lb_med)
        f["ret_slow"] = f["close"].pct_change(lb_slow)
        f = f.iloc[warmup_bars:].dropna()
        f = f[(f.index >= eval_start) & (f.index <= eval_end)]
        if not f.empty:
            out[sym] = f
    return out


def make_kotak_cfg(
    capital: float = CAPITAL_INR,
    slippage: float = 0.0005,
    long_only_assets: Optional[set] = None,
    max_leverage: float = 8.0,
    risk: float = 0.025,
    max_risk: float = 0.08,
    atr_mult: float = 3.0,
    pyramid: float = 0.5,
    dd_brake: Optional[float] = 0.15,
) -> PortfolioConfig:
    return PortfolioConfig(
        initial_capital=capital,
        usd_inr_rate=1.0,
        donchian_risk_per_trade_pct=risk,
        max_concurrent_risk_pct=max_risk,
        atr_trail_mult=atr_mult,
        slippage_pct=slippage,
        entry_slippage_pct=slippage,
        max_leverage=max_leverage,
        max_margin_utilization=0.75,
        pyramid_add_fraction=pyramid,
        dd_brake_threshold=dd_brake,
        whole_lot_rounding=True,
        enable_credit_spreads=False,
        long_only_assets=set(long_only_assets or set()),
    )


def run_config_c_with_cash_yield(
    dfs: Dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    idle_yield_apr: float = 0.0,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Runs unchanged Config C via run_shared_donchian_portfolio, and optionally adds
    modeled daily LIQUIDBEES yield on unallocated cash (strictly zero double-counting).
    """
    trades, eq, diag = run_shared_donchian_portfolio(dfs, set(), cfg, eval_start=eval_start, eval_end=eval_end)
    if eq.empty or idle_yield_apr <= 0.0:
        return trades, eq, diag

    # Compute daily allocated capital from trade intervals (conservative: assumes peak notional while open)
    daily_alloc = pd.Series(0.0, index=eq.index)
    for t in trades:
        is_lo = t.asset in cfg.long_only_assets
        margin = t.entry_price * t.qty if is_lo else (t.entry_price * t.qty / (cfg.max_leverage or 1.0))
        mask = (daily_alloc.index >= t.entry_time.normalize()) & (daily_alloc.index <= t.exit_time.normalize())
        daily_alloc.loc[mask] += margin

    daily_rate = (1.0 + idle_yield_apr) ** (1.0 / 365.0) - 1.0
    idle_cash = (eq - daily_alloc).clip(lower=0.0)
    avg_idle_pct = float((idle_cash / eq.replace(0, np.nan)).mean() * 100.0)
    cum_yield = (idle_cash * daily_rate).cumsum()
    diag = dict(diag)
    diag["avg_idle_cash_pct"] = round(avg_idle_pct, 1)
    diag["yield_inr"] = round(float(cum_yield.iloc[-1]), 2)
    return trades, eq + cum_yield, diag


def run_supertrend_adx(
    prepared: Dict[str, pd.DataFrame],
    long_only_assets: set,
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    max_leverage: float = 8.0,
    risk_pct: float = 0.025,
    max_risk_pct: float = 0.08,
    atr_mult: float = 3.0,
    idle_yield_apr: float = 0.0,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    peak_eq = capital_init
    brake = False
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []
    idle_pcts: List[float] = []
    last_ts_yield: Optional[pd.Timestamp] = None

    for ts in all_ts:
        # Accrue modeled idle cash yield on elapsed calendar days
        if idle_yield_apr > 0 and last_ts_yield is not None:
            dt_days = max((ts - last_ts_yield).total_seconds() / 86400.0, 0.0)
            if dt_days > 0:
                alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                idle = max(capital - alloc, 0.0)
                capital += idle * ((1.0 + idle_yield_apr) ** (dt_days / 365.0) - 1.0)
        last_ts_yield = ts

        unr = sum(
            ((float(last_bar[s]["close"]) - p["entry"]) if p["side"] == "LONG" else (p["entry"] - float(last_bar[s]["close"]))) * p["qty"]
            for s, p in pos_map.items()
            if p is not None and s in last_bar
        )
        cur_eq = capital + unr
        peak_eq = max(peak_eq, cur_eq)
        dd = (peak_eq - cur_eq) / peak_eq if peak_eq > 0 else 0.0
        if dd >= 0.15:
            brake = True
        elif dd <= 0.075:
            brake = False
        scale = 0.5 if brake else 1.0
        eff_r = risk_pct * scale
        eff_mr = max_risk_pct * scale

        for sym, df in prepared.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            prev = last_bar.get(sym)
            pos = pos_map[sym]
            is_lo = sym in long_only_assets

            # Exit on Supertrend flip at next open
            if pos is not None and prev is not None:
                is_long = pos["side"] == "LONG"
                flipped = (is_long and prev["st_dir"] == -1) or ((not is_long) and prev["st_dir"] == 1)
                if flipped:
                    exit_px = float(row["open"]) * (1.0 - slippage if is_long else 1.0 + slippage)
                    pnl = ((exit_px - pos["entry"]) if is_long else (pos["entry"] - exit_px)) * pos["qty"]
                    capital += pnl
                    trades.append(PortfolioTrade(
                        "SUPERTREND", sym, pos["side"], pos["entry_time"], ts,
                        pos["entry"], exit_px, pos["qty"], pnl, 0.0, pnl, pnl / pos["risk_rs"], "ST_FLIP"
                    ))
                    pos_map[sym] = None

            # Entry on Supertrend direction + ADX >= 20
            if pos_map[sym] is None and prev is not None and prev["atr14"] > 0 and prev["adx14"] >= 20.0:
                side = None
                if prev["st_dir"] == 1:
                    side = "LONG"
                elif (not is_lo) and prev["st_dir"] == -1:
                    side = "SHORT"
                if side is not None and capital > 0:
                    active = [p for p in pos_map.values() if p is not None]
                    open_r = sum(p["risk_pct"] for p in active)
                    if open_r + eff_r <= eff_mr + 1e-9:
                        entry = float(row["open"]) * (1.0 + slippage if side == "LONG" else 1.0 - slippage)
                        sl_dist = atr_mult * float(prev["atr14"])
                        raw_qty = (capital * eff_r) / sl_dist
                        alloc = sum(p["margin"] for p in active)
                        if is_lo:
                            budget = max(capital - alloc, 0.0)
                            raw_qty = min(raw_qty, budget / entry)
                        else:
                            budget = max(capital * 0.75 - alloc, 0.0)
                            raw_qty = min(raw_qty, (budget * max_leverage) / entry)
                        qty = math.floor(raw_qty)
                        if qty >= 1:
                            margin = qty * entry if is_lo else (qty * entry / max_leverage)
                            risk_rs = max(qty * sl_dist, 1e-6)
                            pos_map[sym] = {
                                "side": side, "entry_time": ts, "entry": entry, "qty": qty,
                                "sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
                                "margin": margin, "risk_rs": risk_rs, "risk_pct": risk_rs / capital,
                            }
            last_bar[sym] = row

        # Intrabar trailing stop check
        for sym, pos in pos_map.items():
            row = last_bar.get(sym)
            if pos is None or row is None or row.name != ts:
                continue
            is_long = pos["side"] == "LONG"
            stop = pos["sl"]
            hit = row["low"] <= stop if is_long else row["high"] >= stop
            if hit:
                gapped = row["open"] < stop if is_long else row["open"] > stop
                raw_ex = float(row["open"]) if gapped else stop
                exit_px = raw_ex * (1.0 - slippage if is_long else 1.0 + slippage)
                pnl = ((exit_px - pos["entry"]) if is_long else (pos["entry"] - exit_px)) * pos["qty"]
                capital += pnl
                trades.append(PortfolioTrade(
                    "SUPERTREND", sym, pos["side"], pos["entry_time"], ts,
                    pos["entry"], exit_px, pos["qty"], pnl, 0.0, pnl, pnl / pos["risk_rs"],
                    "GAP_STOP" if gapped else "ATR_TRAIL"
                ))
                pos_map[sym] = None
            else:
                trail = atr_mult * float(row["atr14"])
                c = float(row["close"])
                pos["sl"] = max(stop, c - trail) if is_long else min(stop, c + trail)

        alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
        unr = sum(
            ((float(last_bar[s]["close"]) - p["entry"]) if p["side"] == "LONG" else (p["entry"] - float(last_bar[s]["close"]))) * p["qty"]
            for s, p in pos_map.items()
            if p is not None and s in last_bar
        )
        eq_now = capital + unr
        idle_pcts.append(max(capital - alloc, 0.0) / max(eq_now, 1.0) * 100.0)
        eq_recs.append((ts, eq_now))

    final_ts = all_ts[-1]
    for sym, pos in pos_map.items():
        if pos is not None and sym in last_bar:
            is_long = pos["side"] == "LONG"
            exit_px = float(last_bar[sym]["close"]) * (1.0 - slippage if is_long else 1.0 + slippage)
            pnl = ((exit_px - pos["entry"]) if is_long else (pos["entry"] - exit_px)) * pos["qty"]
            capital += pnl
            trades.append(PortfolioTrade(
                "SUPERTREND", sym, pos["side"], pos["entry_time"], final_ts,
                pos["entry"], exit_px, pos["qty"], pnl, 0.0, pnl, pnl / pos["risk_rs"], "END_OF_TEST"
            ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"avg_idle_cash_pct": round(float(np.mean(idle_pcts)), 1)}


def run_tsmom_vol_target(
    prepared: Dict[str, pd.DataFrame],
    long_only_assets: set,
    interval: str,
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    max_leverage: float = 8.0,
    target_vol: float = 0.25,
    idle_yield_apr: float = 0.0,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    CTA Time-Series Momentum (fast/med/slow = 10d/21d/60d), inverse-vol weighted,
    rebalanced weekly (every 5 trading days).
    """
    bars_per_day = 2 if interval == "6h" else 1
    rebal_every = 5 * bars_per_day
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []
    idle_pcts: List[float] = []
    last_ts_yield: Optional[pd.Timestamp] = None
    n_assets = max(len(prepared), 1)

    for bar_i, ts in enumerate(all_ts):
        if idle_yield_apr > 0 and last_ts_yield is not None:
            dt_days = max((ts - last_ts_yield).total_seconds() / 86400.0, 0.0)
            if dt_days > 0:
                alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
                idle = max(capital - alloc, 0.0)
                capital += idle * ((1.0 + idle_yield_apr) ** (dt_days / 365.0) - 1.0)
        last_ts_yield = ts

        if bar_i % rebal_every == 0:
            # Close or resize at open using prior bar signals
            for sym, df in prepared.items():
                if ts not in df.index or sym not in last_bar:
                    continue
                row = df.loc[ts]
                prev = last_bar[sym]
                is_lo = sym in long_only_assets
                sig = (
                    (1 if prev["ret_fast"] > 0 else -1)
                    + (1 if prev["ret_med"] > 0 else -1)
                    + (1 if prev["ret_slow"] > 0 else -1)
                ) / 3.0
                if is_lo and sig <= 0:
                    sig = 0.0
                if abs(sig) < 0.3:
                    sig = 0.0

                vol = float(prev["vol_ann"])
                if not math.isfinite(vol) or vol < 0.05:
                    vol = 0.15
                # Target notional per asset
                lev_cap = 1.0 if is_lo else min(max_leverage * 0.5, 3.0)
                w = min(abs(sig) * (target_vol / vol) / n_assets, lev_cap / n_assets)
                target_side = "LONG" if sig > 0 else ("SHORT" if sig < 0 else None)
                open_px = float(row["open"])
                target_qty = math.floor((capital * w) / open_px) if target_side else 0

                pos = pos_map[sym]
                if pos is not None and (target_side != pos["side"] or target_qty == 0):
                    is_long = pos["side"] == "LONG"
                    ex_px = open_px * (1.0 - slippage if is_long else 1.0 + slippage)
                    pnl = ((ex_px - pos["entry"]) if is_long else (pos["entry"] - ex_px)) * pos["qty"]
                    capital += pnl
                    trades.append(PortfolioTrade(
                        "TSMOM", sym, pos["side"], pos["entry_time"], ts,
                        pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.02, 1.0), "REBAL"
                    ))
                    pos_map[sym] = None

                if pos_map[sym] is None and target_side is not None and target_qty >= 1 and capital > 0:
                    active = [p for p in pos_map.values() if p is not None]
                    alloc = sum(p["margin"] for p in active)
                    entry = open_px * (1.0 + slippage if target_side == "LONG" else 1.0 - slippage)
                    if is_lo:
                        budget = max(capital - alloc, 0.0)
                        qty = min(target_qty, math.floor(budget / entry))
                    else:
                        budget = max(capital * 0.75 - alloc, 0.0)
                        qty = min(target_qty, math.floor((budget * max_leverage) / entry))
                    if qty >= 1:
                        margin = qty * entry if is_lo else (qty * entry / max_leverage)
                        pos_map[sym] = {
                            "side": target_side, "entry_time": ts, "entry": entry, "qty": qty, "margin": margin,
                        }

        for sym, df in prepared.items():
            if ts in df.index:
                last_bar[sym] = df.loc[ts]

        alloc = sum(p["margin"] for p in pos_map.values() if p is not None)
        unr = sum(
            ((float(last_bar[s]["close"]) - p["entry"]) if p["side"] == "LONG" else (p["entry"] - float(last_bar[s]["close"]))) * p["qty"]
            for s, p in pos_map.items()
            if p is not None and s in last_bar
        )
        eq_now = capital + unr
        idle_pcts.append(max(capital - alloc, 0.0) / max(eq_now, 1.0) * 100.0)
        eq_recs.append((ts, eq_now))

    final_ts = all_ts[-1]
    for sym, pos in pos_map.items():
        if pos is not None and sym in last_bar:
            is_long = pos["side"] == "LONG"
            ex_px = float(last_bar[sym]["close"]) * (1.0 - slippage if is_long else 1.0 + slippage)
            pnl = ((ex_px - pos["entry"]) if is_long else (pos["entry"] - ex_px)) * pos["qty"]
            capital += pnl
            trades.append(PortfolioTrade(
                "TSMOM", sym, pos["side"], pos["entry_time"], final_ts,
                pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.02, 1.0), "END_OF_TEST"
            ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"avg_idle_cash_pct": round(float(np.mean(idle_pcts)), 1)}


def run_dual_momentum_rotation(
    prepared: Dict[str, pd.DataFrame],
    interval: str,
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    idle_yield_apr: float = LIQUID_YIELD_APR,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """
    Long-Only Dual-Momentum Metals Rotation (100% cash ETFs, zero leverage):
      - Every 5 trading days, rank GOLDBEES and SILVERBEES by 21d return.
      - Hold the stronger metal with 100% cash if its 21d return > 0 AND Close > EMA50.
      - If neither qualifies, sit 100% in LIQUIDBEES (modeled 6.5% p.a. yield).
    """
    bars_per_day = 2 if interval == "6h" else 1
    rebal_every = 5 * bars_per_day
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    pos: Optional[dict] = None
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
                alloc = pos["margin"] if pos is not None else 0.0
                idle = max(capital - alloc, 0.0)
                capital += idle * ((1.0 + idle_yield_apr) ** (dt_days / 365.0) - 1.0)
        last_ts_yield = ts

        if bar_i % rebal_every == 0 and len(last_bar) == len(prepared):
            best_sym = None
            best_score = -1e9
            for sym, prev in last_bar.items():
                if ts not in prepared[sym].index:
                    continue
                if prev["ret_med"] > 0 and prev["close"] > prev["ema50"]:
                    score = float(prev["ret_med"])
                    if score > best_score:
                        best_score = score
                        best_sym = sym

            if pos is not None and pos["sym"] != best_sym and pos["sym"] in prepared and ts in prepared[pos["sym"]].index:
                ex_px = float(prepared[pos["sym"]].loc[ts]["open"]) * (1.0 - slippage)
                pnl = (ex_px - pos["entry"]) * pos["qty"]
                capital += pnl
                trades.append(PortfolioTrade(
                    "DUAL_MOM", pos["sym"], "LONG", pos["entry_time"], ts,
                    pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.025, 1.0), "ROTATE"
                ))
                pos = None

            if pos is None and best_sym is not None and ts in prepared[best_sym].index and capital > 0:
                entry = float(prepared[best_sym].loc[ts]["open"]) * (1.0 + slippage)
                qty = math.floor(capital / entry)
                if qty >= 1:
                    pos = {"sym": best_sym, "entry_time": ts, "entry": entry, "qty": qty, "margin": qty * entry}

        for sym, df in prepared.items():
            if ts in df.index:
                last_bar[sym] = df.loc[ts]

        alloc = pos["margin"] if pos is not None else 0.0
        unr = ((float(last_bar[pos["sym"]]["close"]) - pos["entry"]) * pos["qty"]) if (pos is not None and pos["sym"] in last_bar) else 0.0
        eq_now = capital + unr
        idle_pcts.append(max(capital - alloc, 0.0) / max(eq_now, 1.0) * 100.0)
        eq_recs.append((ts, eq_now))

    final_ts = all_ts[-1]
    if pos is not None and pos["sym"] in last_bar:
        ex_px = float(last_bar[pos["sym"]]["close"]) * (1.0 - slippage)
        pnl = (ex_px - pos["entry"]) * pos["qty"]
        capital += pnl
        trades.append(PortfolioTrade(
            "DUAL_MOM", pos["sym"], "LONG", pos["entry_time"], final_ts,
            pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.025, 1.0), "END_OF_TEST"
        ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {"avg_idle_cash_pct": round(float(np.mean(idle_pcts)), 1)}


def run_rsi2_pullback(
    prepared: Dict[str, pd.DataFrame],
    capital_init: float = CAPITAL_INR,
    slippage: float = 0.0005,
    idle_yield_apr: float = LIQUID_YIELD_APR,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, float]]:
    """Long-Only RSI(2)<15 pullback above EMA(200), exit Close > SMA(5), 50% capital per metal."""
    all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
    pos_map: Dict[str, Optional[dict]] = {s: None for s in prepared}
    last_bar: Dict[str, pd.Series] = {}
    capital = capital_init
    trades: List[PortfolioTrade] = []
    eq_recs: List[Tuple[pd.Timestamp, float]] = []
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
                if prev["close"] > prev["sma5"] or prev["close"] < pos["sl"]:
                    ex_px = float(row["open"]) * (1.0 - slippage)
                    pnl = (ex_px - pos["entry"]) * pos["qty"]
                    capital += pnl
                    trades.append(PortfolioTrade(
                        "RSI2_PB", sym, "LONG", pos["entry_time"], ts,
                        pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.02, 1.0), "SMA5_OR_SL"
                    ))
                    pos_map[sym] = None
            elif pos is None and prev is not None and prev["close"] > prev["ema200"] and prev["rsi2"] < 15.0 and capital > 0:
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

        unr = sum((float(last_bar[s]["close"]) - p["entry"]) * p["qty"] for s, p in pos_map.items() if p is not None and s in last_bar)
        eq_recs.append((ts, capital + unr))

    final_ts = all_ts[-1]
    for sym, pos in pos_map.items():
        if pos is not None and sym in last_bar:
            ex_px = float(last_bar[sym]["close"]) * (1.0 - slippage)
            pnl = (ex_px - pos["entry"]) * pos["qty"]
            capital += pnl
            trades.append(PortfolioTrade(
                "RSI2_PB", sym, "LONG", pos["entry_time"], final_ts,
                pos["entry"], ex_px, pos["qty"], pnl, 0.0, pnl, pnl / max(capital * 0.02, 1.0), "END_OF_TEST"
            ))

    eq_s = pd.Series([v for _, v in eq_recs[:-1]] + [capital], index=pd.DatetimeIndex([t for t, _ in eq_recs]))
    return trades, eq_s.resample("1D").last().ffill(), {}


def buy_and_hold_curve(prepared: Dict[str, pd.DataFrame], weights: Dict[str, float], capital: float = CAPITAL_INR) -> pd.Series:
    idx = sorted(set().union(*[df.index for df in prepared.values()]))
    parts = []
    for sym, w in weights.items():
        s = prepared[sym]["close"].reindex(idx).ffill().bfill()
        parts.append((capital * w) * (s / float(s.iloc[0])))
    total = sum(parts)
    return total.resample("1D").last().ffill()


def yearly_returns(eq: pd.Series, capital: float = CAPITAL_INR) -> str:
    ye = eq.resample("YE").last()
    yr = ye.pct_change()
    yr.iloc[0] = ye.iloc[0] / capital - 1.0
    return " | ".join(f"{d.year}: {v*100:+.1f}%" for d, v in yr.items())


def evaluate_suite(
    title: str,
    raw_dfs: Dict[str, pd.DataFrame],
    interval: str,
    eval_start_str: str,
    eval_end_str: str,
    slippage: float = 0.0005,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    eval_start = pd.Timestamp(eval_start_str, tz="Asia/Kolkata")
    eval_end = pd.Timestamp(eval_end_str, tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)

    prepared = prepare_metal_features(raw_dfs, interval, eval_start, eval_end, warmup_bars=200)
    syms = list(prepared.keys())
    long_only_all = set(syms)
    # Scenario where GOLDBEES acts as 8x L/S MCX GOLDPETAL proxy (explicitly labeled as Scenario)
    long_only_silver = {s for s in syms if "SILVER" in s}

    rows = []
    curves: Dict[str, pd.Series] = {}

    def record(name: str, eq: pd.Series, trades: List[PortfolioTrade], extra: str = ""):
        st = calc_curve_stats(name, eq, trades, CAPITAL_INR)
        st["Notes"] = extra
        rows.append(st)
        curves[name] = eq

    # 1. Baseline Config C (Spot 1x Long-Only ETFs)
    cfg_spot = make_kotak_cfg(slippage=slippage, long_only_assets=long_only_all, max_leverage=1.0)
    t1, eq1, d1 = run_config_c_with_cash_yield(raw_dfs, cfg_spot, eval_start, eval_end, idle_yield_apr=0.0)
    record("1. Config C (Spot 1x L-Only)", eq1, t1, f"Pyr={d1.get('pyramids',0)}")

    # 2. Config C (Spot 1x) + Modeled 6.5% LIQUIDBEES Idle Cash Yield
    t1y, eq1y, d1y = run_config_c_with_cash_yield(raw_dfs, cfg_spot, eval_start, eval_end, idle_yield_apr=LIQUID_YIELD_APR)
    record("2. Config C (Spot 1x) + 6.5% Liquid", eq1y, t1y, f"Idle={d1y.get('avg_idle_cash_pct',0)}%")

    # 3. Config C (MCX 8x Gold L/S Scenario + Silver 1x L-Only) — mirrors live GOLDPETAL+SILVERBEES
    cfg_mcx = make_kotak_cfg(slippage=slippage, long_only_assets=long_only_silver, max_leverage=8.0)
    t2, eq2, d2 = run_config_c_with_cash_yield(raw_dfs, cfg_mcx, eval_start, eval_end, idle_yield_apr=0.0)
    record("3. Config C (MCX 8x Gold Scen)", eq2, t2, f"Pyr={d2.get('pyramids',0)}")

    # 4. Config C (MCX 8x Gold Scen) + 6.5% Pledged LIQUIDBEES Yield
    t2y, eq2y, d2y = run_config_c_with_cash_yield(raw_dfs, cfg_mcx, eval_start, eval_end, idle_yield_apr=LIQUID_YIELD_APR)
    record("4. Config C (MCX 8x) + 6.5% Liquid", eq2y, t2y, f"Idle={d2y.get('avg_idle_cash_pct',0)}%")

    # 5. Supertrend(10,3)+ADX>20 (Spot 1x + 6.5% Liquid)
    t3, eq3, d3 = run_supertrend_adx(prepared, long_only_all, slippage=slippage, max_leverage=1.0, idle_yield_apr=LIQUID_YIELD_APR)
    record("5. Supertrend+ADX (Spot + Liquid)", eq3, t3, f"Idle={d3.get('avg_idle_cash_pct',0)}%")

    # 6. Supertrend(10,3)+ADX>20 (MCX 8x Gold Scen + Liquid)
    t3m, eq3m, d3m = run_supertrend_adx(prepared, long_only_silver, slippage=slippage, max_leverage=8.0, idle_yield_apr=LIQUID_YIELD_APR)
    record("6. Supertrend+ADX (MCX 8x + Liquid)", eq3m, t3m, f"Idle={d3m.get('avg_idle_cash_pct',0)}%")

    # 7. CTA TSMOM Vol-Target (MCX 8x Gold Scen + Liquid)
    t4, eq4, d4 = run_tsmom_vol_target(prepared, long_only_silver, interval, slippage=slippage, max_leverage=8.0, idle_yield_apr=LIQUID_YIELD_APR)
    record("7. CTA TSMOM (MCX 8x + Liquid)", eq4, t4, f"Idle={d4.get('avg_idle_cash_pct',0)}%")

    # 8. 3-Pod Blend (0.40 Config C + 0.35 Supertrend + 0.35 TSMOM, MCX 8x + Liquid)
    common = eq2y.index.intersection(eq3m.index).intersection(eq4.index)
    eq_3pod = (
        0.40 * eq2y.reindex(common).ffill()
        + 0.35 * eq3m.reindex(common).ffill()
        + 0.25 * eq4.reindex(common).ffill()
    )
    record("8. 3-Pod Blend (MCX 8x + Liquid)", eq_3pod, [], "40/35/25 blend")

    # 9. Dual-Momentum Metal Rotation (100% Spot Cash + 6.5% Liquid)
    t5, eq5, d5 = run_dual_momentum_rotation(prepared, interval, slippage=slippage, idle_yield_apr=LIQUID_YIELD_APR)
    record("9. Dual-Mom Rotation (Spot+Liquid)", eq5, t5, f"Idle={d5.get('avg_idle_cash_pct',0)}%")

    # 10. RSI(2) Pullback (Spot + Liquid)
    t6, eq6, _ = run_rsi2_pullback(prepared, slippage=slippage, idle_yield_apr=LIQUID_YIELD_APR)
    record("10. RSI(2) Pullback (Spot+Liquid)", eq6, t6, "Above EMA200")

    # 11. Buy & Hold 50/50 Benchmark
    eq_bh = buy_and_hold_curve(prepared, {s: 1.0 / len(syms) for s in syms})
    record("11. Buy & Hold 50/50 (Benchmark)", eq_bh, [], "Passive INR")

    df_res = pd.DataFrame(rows)
    print(f"\n================ {title} ({eval_start_str} -> {eval_end_str}, slip={slippage*100:.2f}%) ================")
    print(df_res.to_string(index=False))
    return df_res, curves


def main():
    out_dir = new_run_directory("kotak-metals-upstox")

    # Load all cached actual Upstox V3 series
    gp_6h = load_cached("MCX", "GOLDPETAL", "6h")
    sm_6h = load_cached("MCX", "SILVERMIC", "6h")
    gb_6h = load_cached("NSE", "GOLDBEES", "6h")
    sb_6h = load_cached("NSE", "SILVERBEES", "6h")
    gb_1d = load_cached("NSE", "GOLDBEES", "1d")
    sb_1d = load_cached("NSE", "SILVERBEES", "1d")

    # -----------------------------------------------------------------------
    # PART A: EXACT CURRENT KOTAK LIVE SLEEVE (MCX GOLDPETAL + NSE SILVERBEES, 6h)
    # -----------------------------------------------------------------------
    gp_feat = add_donchian_features(gp_6h, 20, 200)
    mcx_start = gp_feat.index[0]
    mcx_end = gp_feat.index[-1]
    cfg_live = make_kotak_cfg(slippage=0.0005, long_only_assets={"SILVERBEES"}, max_leverage=8.0)
    t_live, eq_live, diag_live = run_shared_donchian_portfolio(
        {"GOLDPETAL": gp_6h, "SILVERBEES": sb_6h}, set(), cfg_live, eval_start=mcx_start, eval_end=mcx_end
    )
    st_live = calc_curve_stats("Current Live Sleeve (MCX GOLDPETAL + NSE SILVERBEES, 6h)", eq_live, t_live, CAPITAL_INR)
    print("\n================ PART A: EXACT CURRENT KOTAK LIVE SLEEVE (ACTUAL UPSTOX 6H) ================")
    print(f"Active GOLDPETAL contract window after 200-bar warmup: {mcx_start.strftime('%Y-%m-%d')} -> {mcx_end.strftime('%Y-%m-%d')} ({len(gp_feat)} bars)")
    print(pd.DataFrame([st_live]).to_string(index=False))
    print(f"Diagnostics: {diag_live}")
    if t_live:
        print(pd.DataFrame([t.__dict__ for t in t_live])[["asset", "side", "entry_time", "exit_time", "entry_price", "exit_price", "qty", "net_pnl", "exit_reason"]].to_string(index=False))

    # Check MCX SILVERMIC feasibility on Rs 98,000
    t_sm, eq_sm, diag_sm = run_shared_donchian_portfolio(
        {"GOLDPETAL": gp_6h, "SILVERMIC": sm_6h}, set(),
        make_kotak_cfg(slippage=0.0005, long_only_assets=set(), max_leverage=8.0),
        eval_start=mcx_start, eval_end=mcx_end,
    )
    sm_trades = [t for t in t_sm if t.asset == "SILVERMIC"]
    print(f"MCX SILVERMIC (1kg, ~Rs {sm_6h['close'].iloc[-1]:,.0f}/lot) feasibility on Rs 98k (2.5% risk = Rs 2,450): trades={len(sm_trades)}, skipped_lot_size={diag_sm['skipped_lot_size']}")

    # -----------------------------------------------------------------------
    # PART B: 2-YEAR ACTUAL UPSTOX 6H vs 1D (NSE GOLDBEES + NSE SILVERBEES)
    # -----------------------------------------------------------------------
    df_2y_6h, _ = evaluate_suite(
        "PART B1: 2-YEAR ACTUAL UPSTOX 6H (NSE GOLDBEES + SILVERBEES)",
        {"GOLDBEES": gb_6h, "SILVERBEES": sb_6h},
        "6h", "2024-11-24", "2026-11-23", slippage=0.0005,
    )
    df_2y_1d, _ = evaluate_suite(
        "PART B2: 2-YEAR ACTUAL UPSTOX 1D (NSE GOLDBEES + SILVERBEES)",
        {"GOLDBEES": gb_1d, "SILVERBEES": sb_1d},
        "1d", "2024-11-24", "2026-11-23", slippage=0.0005,
    )

    # -----------------------------------------------------------------------
    # PART C: ~4-YEAR DAILY VALIDATION (IS 2022-12 -> 2024-11 vs OOS 2024-11 -> 2026-11)
    # -----------------------------------------------------------------------
    df_is, _ = evaluate_suite(
        "PART C1: IN-SAMPLE 2Y DAILY (2022-12-01 -> 2024-11-23)",
        {"GOLDBEES": gb_1d, "SILVERBEES": sb_1d},
        "1d", "2022-12-01", "2024-11-23", slippage=0.0005,
    )
    df_full, curves_full = evaluate_suite(
        "PART C2: FULL ~4-YEAR DAILY (2022-12-01 -> 2026-11-23)",
        {"GOLDBEES": gb_1d, "SILVERBEES": sb_1d},
        "1d", "2022-12-01", "2026-11-23", slippage=0.0005,
    )
    df_stress, _ = evaluate_suite(
        "PART C3: STRESS TEST 0.15%/SIDE ~4-YEAR DAILY (2022-12-01 -> 2026-11-23)",
        {"GOLDBEES": gb_1d, "SILVERBEES": sb_1d},
        "1d", "2022-12-01", "2026-11-23", slippage=0.0015,
    )

    print("\n================ CALENDAR YEAR RETURNS (FULL ~4Y DAILY, 0.05% SLIP) ================")
    for name, eq in curves_full.items():
        print(f"{name:38s} | {yearly_returns(eq)}")

    df_2y_6h.to_csv(out_dir / "part_b1_2y_6h.csv", index=False)
    df_2y_1d.to_csv(out_dir / "part_b2_2y_1d.csv", index=False)
    df_is.to_csv(out_dir / "part_c1_is_1d.csv", index=False)
    df_full.to_csv(out_dir / "part_c2_full_4y_1d.csv", index=False)
    df_stress.to_csv(out_dir / "part_c3_stress_4y_1d.csv", index=False)
    pd.DataFrame(curves_full).to_csv(out_dir / "equity_curves_4y_1d.csv")
    print(f"\nSaved all tables and equity curves to {out_dir}/")


if __name__ == "__main__":
    main()
