import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import PortfolioConfig


@dataclass
class PortfolioTrade:
    engine: str          # "DONCHIAN" or "CREDIT_SPREAD"
    asset: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    funding_cost: float
    net_pnl: float
    r_multiple: float
    exit_reason: str


def add_ema_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes EMA(50), EMA(200), and ATR(14) without dropping rows for Donchian windows."""
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def add_donchian_features(
    df: pd.DataFrame,
    entry_len: int = 20,
    warmup_bars: int = 200,
    eval_start: Optional[pd.Timestamp] = None,
    eval_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Computes EMA(50), EMA(200), ATR(14), and shifted Donchian High/Low,
    strictly drops the first `warmup_bars` rows, and optionally slices to [eval_start, eval_end].
    """
    df = add_ema_atr_features(df)
    df["donchian_high"] = df["high"].rolling(entry_len).max().shift(1)
    df["donchian_low"] = df["low"].rolling(entry_len).min().shift(1)

    df = df.iloc[warmup_bars:].dropna()

    if eval_start is not None:
        df = df[df.index >= eval_start]
    if eval_end is not None:
        df = df[df.index <= eval_end]
    return df


def _pnl(side: str, entry: float, exit_px: float, qty: float) -> float:
    return (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty


# ---------------------------------------------------------------------------
# ENGINE A: HARDENED SHARED-CAPITAL MULTI-ASSET DONCHIAN PORTFOLIO
# ---------------------------------------------------------------------------
def run_shared_donchian_portfolio(
    asset_dfs: Dict[str, pd.DataFrame],
    crypto_set: set,
    cfg: PortfolioConfig,
    eval_start: Optional[pd.Timestamp] = None,
    eval_end: Optional[pd.Timestamp] = None,
) -> Tuple[List[PortfolioTrade], pd.Series, Dict[str, int]]:
    """
    Hardened Multi-Asset Donchian Engine:
      1. Strictly enforces 200-bar EMA warmup before first signal.
      2. Gap-aware stop fills: if bar opens past SL, fills at bar.open (not optimistic SL).
      3. Enforces 1% risk/trade, 3% max concurrent risk, and max 2 concurrent crypto positions.
      4. Automatically enforces long-only for spot ETFs (e.g., GOLDIETF.NS).
      5. Optional whole-lot rounding (skips trade if 1 lot exceeds risk cap).
      6. Deducts 0.05% per-side slippage/fees + 0.01% per 8h crypto funding drag.
      7. Force-closes all remaining open positions at the final bar.
    """
    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in asset_dfs.items():
        feat = add_donchian_features(df, cfg.donchian_entry_len, cfg.ema_warmup_bars, eval_start, eval_end)
        if not feat.empty:
            prepared[sym] = feat

    if not prepared:
        return [], pd.Series(dtype=float), {"gap_stops": 0, "skipped_lot_size": 0, "pyramids": 0, "dd_brake_bars": 0}

    all_timestamps = sorted(set().union(*[df.index for df in prepared.values()]))
    if not all_timestamps:
        return [], pd.Series(dtype=float), {"gap_stops": 0, "skipped_lot_size": 0, "pyramids": 0, "dd_brake_bars": 0}

    pos_map: Dict[str, Optional[Dict]] = {sym: None for sym in prepared}
    last_bar_map: Dict[str, pd.Series] = {}

    entry_slip = cfg.slippage_pct if cfg.entry_slippage_pct is None else cfg.entry_slippage_pct
    capital = cfg.initial_capital
    peak_equity = cfg.initial_capital
    brake_active = False
    trades: List[PortfolioTrade] = []
    equity_records: List[Tuple[pd.Timestamp, float]] = []
    diagnostics = {"gap_stops": 0, "skipped_lot_size": 0, "pyramids": 0, "dd_brake_bars": 0}

    for ts in all_timestamps:
        # Update drawdown brake from mark-to-market equity before evaluating new entries.
        unr_now = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar_map:
                cur_px = float(last_bar_map[sym]["close"])
                unr_now += _pnl(pos["side"], pos["entry"], cur_px, pos["qty"]) * pos["fx"] - pos["funding_paid"]
        cur_eq = capital + unr_now
        peak_equity = max(peak_equity, cur_eq)
        cur_dd = (peak_equity - cur_eq) / peak_equity if peak_equity > 0 else 0.0
        if cfg.dd_brake_threshold is not None and cfg.dd_brake_threshold > 0:
            if cur_dd >= cfg.dd_brake_threshold:
                brake_active = True
            elif cur_dd <= cfg.dd_brake_threshold * 0.5:
                brake_active = False
        if brake_active:
            diagnostics["dd_brake_bars"] += 1
        risk_scale = cfg.dd_brake_scale if brake_active else 1.0
        eff_risk_pct = cfg.donchian_risk_per_trade_pct * risk_scale
        eff_max_risk_pct = cfg.max_concurrent_risk_pct * risk_scale

        for sym, df in prepared.items():
            if ts not in df.index:
                continue
            row = df.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            prev = last_bar_map.get(sym)
            # Enter at the open using only prior-bar information.
            if pos_map[sym] is None and prev is not None and prev["atr14"] > 0:
                is_long_only = sym in cfg.long_only_assets
                side = None
                if prev["ema50"] > prev["ema200"] and prev["close"] > prev["donchian_high"]:
                    side = "LONG"
                elif (not is_long_only) and prev["ema50"] < prev["ema200"] and prev["close"] < prev["donchian_low"]:
                    side = "SHORT"

                if side is not None:
                    active_positions = [p for p in pos_map.values() if p is not None]
                    current_open_risk_pct = sum(p["risk_pct"] for p in active_positions)
                    open_crypto_count = sum(1 for s, p in pos_map.items() if p is not None and s in crypto_set)

                    can_enter_risk = (current_open_risk_pct + eff_risk_pct) <= (eff_max_risk_pct + 1e-9)
                    can_enter_crypto = (sym not in crypto_set) or (open_crypto_count < cfg.max_crypto_concurrent)

                    if can_enter_risk and can_enter_crypto and capital > 0:
                        fx = cfg.usd_inr_rate if sym in cfg.usd_quoted_assets else 1.0
                        entry = float(row["open"]) * (1 + entry_slip if side == "LONG" else 1 - entry_slip)
                        sl_dist = cfg.atr_trail_mult * float(prev["atr14"])
                        target_risk_rs = capital * eff_risk_pct
                        raw_qty = (target_risk_rs / fx) / sl_dist

                        if is_long_only:
                            allocated_spot_rs = sum(p["margin_rs"] for p in active_positions)
                            spot_budget_rs = max(capital - allocated_spot_rs, 0.0)
                            max_qty = (spot_budget_rs / fx) / entry
                            raw_qty = min(raw_qty, max_qty)
                        elif cfg.max_leverage is not None and cfg.max_leverage > 0:
                            allocated_margin_rs = sum(p["margin_rs"] for p in active_positions)
                            margin_budget_rs = max(capital * cfg.max_margin_utilization - allocated_margin_rs, 0.0)
                            max_qty = ((margin_budget_rs * cfg.max_leverage) / fx) / entry
                            raw_qty = min(raw_qty, max_qty)

                        if cfg.whole_lot_rounding:
                            step = cfg.lot_multipliers.get(sym, 1.0)
                            lots = math.floor(raw_qty / step)
                            if lots < 1:
                                diagnostics["skipped_lot_size"] += 1
                                last_bar_map[sym] = row
                                continue
                            qty = lots * step
                        else:
                            qty = raw_qty

                        actual_risk_rs = max(qty * sl_dist * fx, 1e-6)
                        notional_rs = qty * entry * fx
                        margin_rs = notional_rs if is_long_only else (notional_rs / cfg.max_leverage if cfg.max_leverage else 0.0)

                        if qty > 0:
                            pos_map[sym] = {
                                "side": side,
                                "entry_time": ts,
                                "entry": entry,
                                "init_entry": entry,
                                "init_sl_dist": sl_dist,
                                "qty": qty,
                                "fx": fx,
                                "margin_rs": margin_rs,
                                "risk_rs": actual_risk_rs,
                                "risk_pct": actual_risk_rs / capital,
                                "sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
                                "funding_paid": 0.0,
                                "last_funding_ts": ts,
                                "pyramided": False,
                            }

            # Winner pyramiding: add once price moves +pyramid_trigger_r * init_sl_dist AND stop >= breakeven.
            elif (
                pos_map[sym] is not None
                and cfg.pyramid_add_fraction > 0
                and not pos_map[sym]["pyramided"]
                and prev is not None
                and capital > 0
            ):
                pos = pos_map[sym]
                is_long = pos["side"] == "LONG"
                is_long_only = sym in cfg.long_only_assets
                open_px = float(row["open"])
                move = (open_px - pos["init_entry"]) if is_long else (pos["init_entry"] - open_px)
                be_ok = (pos["sl"] >= pos["init_entry"]) if is_long else (pos["sl"] <= pos["init_entry"])
                if move >= cfg.pyramid_trigger_r * pos["init_sl_dist"] and be_ok:
                    fx = pos["fx"]
                    add_entry = open_px * (1 + entry_slip if is_long else 1 - entry_slip)
                    raw_add = pos["qty"] * cfg.pyramid_add_fraction
                    active_positions = [p for p in pos_map.values() if p is not None]
                    allocated_margin_rs = sum(p["margin_rs"] for p in active_positions)
                    if is_long_only:
                        spot_budget_rs = max(capital - allocated_margin_rs, 0.0)
                        raw_add = min(raw_add, (spot_budget_rs / fx) / add_entry)
                    elif cfg.max_leverage is not None and cfg.max_leverage > 0:
                        margin_budget_rs = max(capital * cfg.max_margin_utilization - allocated_margin_rs, 0.0)
                        raw_add = min(raw_add, ((margin_budget_rs * cfg.max_leverage) / fx) / add_entry)

                    if cfg.whole_lot_rounding:
                        step = cfg.lot_multipliers.get(sym, 1.0)
                        add_qty = math.floor(raw_add / step) * step
                    else:
                        add_qty = raw_add

                    if add_qty > 0:
                        new_qty = pos["qty"] + add_qty
                        pos["entry"] = (pos["entry"] * pos["qty"] + add_entry * add_qty) / new_qty
                        pos["qty"] = new_qty
                        add_notional = add_qty * add_entry * fx
                        pos["margin_rs"] += add_notional if is_long_only else (
                            add_notional / cfg.max_leverage if cfg.max_leverage else 0.0
                        )
                        pos["pyramided"] = True
                        diagnostics["pyramids"] += 1

            last_bar_map[sym] = row

        # Check stops only after evaluating every entry at this open. Intrabar exits
        # cannot free capital or capacity for another entry at the same open.
        for sym, pos in pos_map.items():
            row = last_bar_map.get(sym)
            if pos is None or row is None or row.name != ts:
                continue
            if sym in crypto_set:
                periods = (ts - pos["last_funding_ts"]) // pd.Timedelta(hours=8)
                if periods:
                    pos["funding_paid"] += float(row["close"]) * pos["qty"] * pos["fx"] * cfg.crypto_funding_8h_pct * periods
                    pos["last_funding_ts"] += periods * pd.Timedelta(hours=8)

            is_long = pos["side"] == "LONG"
            stop = pos["sl"]
            sl_hit = row["low"] <= stop if is_long else row["high"] >= stop
            if not sl_hit:
                trail = cfg.atr_trail_mult * float(row["atr14"])
                close = float(row["close"])
                pos["sl"] = max(stop, close - trail) if is_long else min(stop, close + trail)
                continue

            gapped = row["open"] < stop if is_long else row["open"] > stop
            gap_fill = cfg.gap_aware_stops and gapped
            diagnostics["gap_stops"] += int(gap_fill)
            raw_exit = float(row["open"]) if gap_fill else stop
            exit_px = raw_exit * (1 - cfg.slippage_pct if is_long else 1 + cfg.slippage_pct)
            gross = _pnl(pos["side"], pos["entry"], exit_px, pos["qty"]) * pos["fx"]
            net = gross - pos["funding_paid"]
            capital += net
            trades.append(PortfolioTrade(
                engine="DONCHIAN", asset=sym, side=pos["side"], entry_time=pos["entry_time"], exit_time=ts,
                entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"], gross_pnl=gross,
                funding_cost=pos["funding_paid"], net_pnl=net, r_multiple=net / pos["risk_rs"],
                exit_reason="GAP_STOP" if gap_fill else "ATR_TRAIL",
            ))
            pos_map[sym] = None

        unrealized = 0.0
        for sym, pos in pos_map.items():
            if pos is not None and sym in last_bar_map:
                cur_px = float(last_bar_map[sym]["close"])
                unrealized += _pnl(pos["side"], pos["entry"], cur_px, pos["qty"]) * pos["fx"] - pos["funding_paid"]
        equity_records.append((ts, capital + unrealized))

    # Step 3: Force-close remaining open positions at final bar
    final_ts = all_timestamps[-1]
    for sym, pos in pos_map.items():
        if pos is not None and sym in last_bar_map:
            is_long = pos["side"] == "LONG"
            exit_px = float(last_bar_map[sym]["close"]) * (1 - cfg.slippage_pct if is_long else 1 + cfg.slippage_pct)
            gross = _pnl(pos["side"], pos["entry"], exit_px, pos["qty"]) * pos["fx"]
            net = gross - pos["funding_paid"]
            capital += net
            trades.append(
                PortfolioTrade(
                    engine="DONCHIAN", asset=sym, side=pos["side"],
                    entry_time=pos["entry_time"], exit_time=final_ts,
                    entry_price=pos["entry"], exit_price=exit_px, qty=pos["qty"],
                    gross_pnl=gross, funding_cost=pos["funding_paid"],
                    net_pnl=net, r_multiple=net / pos["risk_rs"],
                    exit_reason="END_OF_TEST",
                )
            )

    eq_series = pd.Series(
        [v for _, v in equity_records[:-1]] + [capital],
        index=pd.DatetimeIndex([t for t, _ in equity_records]),
        name="donchian_equity",
    )
    daily_eq = eq_series.resample("1D").last().ffill()
    return trades, daily_eq, diagnostics


# ---------------------------------------------------------------------------
# ENGINE B: COLLATERAL-BACKED WEEKLY CREDIT SPREADS (VRP)
# ---------------------------------------------------------------------------
def _bsm_price(S: float, K: float, T: float, sigma: float, opt_type: str, r: float = 0.06) -> float:
    if T <= 1e-6:
        return max(0.0, S - K) if opt_type == "C" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "C":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def run_weekly_credit_spreads(
    df_underlying: pd.DataFrame,
    asset_name: str,
    cfg: PortfolioConfig,
    eval_start: Optional[pd.Timestamp] = None,
    eval_end: Optional[pd.Timestamp] = None,
) -> Tuple[List[PortfolioTrade], pd.Series, pd.Series]:
    daily = df_underlying[["open", "high", "low", "close", "volume"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    if len(daily) < 60:
        empty = pd.Series(dtype=float)
        return [], empty, empty

    daily["ret"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["rv20"] = daily["ret"].rolling(20).std() * np.sqrt(252)
    daily["ema20"] = daily["close"].ewm(span=20, adjust=False).mean()
    daily["ema50"] = daily["close"].ewm(span=50, adjust=False).mean()
    daily = daily.dropna()

    if eval_start is not None:
        daily = daily[daily.index >= eval_start]
    if eval_end is not None:
        daily = daily[daily.index <= eval_end]

    if len(daily) < 10:
        empty = pd.Series(dtype=float)
        return [], empty, empty

    trades: List[PortfolioTrade] = []
    capital = cfg.initial_capital
    pnl_by_day = pd.Series(0.0, index=daily.index)
    i = 0

    while i < len(daily) - 6:
        row = daily.iloc[i]
        entry_time = daily.index[i]
        S0 = float(row["close"])
        iv = max(float(row["rv20"]) * cfg.vrp_iv_multiplier, 0.12)
        T0 = 7.0 / 365.0
        weekly_sd = S0 * iv * np.sqrt(T0)

        is_bull = row["ema20"] >= row["ema50"]
        if is_bull:
            opt_type = "P"
            k_short = S0 - cfg.short_strike_sd * weekly_sd
            k_long = S0 - cfg.long_strike_sd * weekly_sd
            side_label = "BULL_PUT_SPREAD"
        else:
            opt_type = "C"
            k_short = S0 + cfg.short_strike_sd * weekly_sd
            k_long = S0 + cfg.long_strike_sd * weekly_sd
            side_label = "BEAR_CALL_SPREAD"

        credit = (_bsm_price(S0, k_short, T0, iv, opt_type) - _bsm_price(S0, k_long, T0, iv, opt_type)) * 0.96
        spread_width = abs(k_short - k_long)
        max_loss_unit = max(spread_width - credit, 1e-6)

        if credit <= 0:
            i += 1
            continue

        risk_rs = capital * cfg.spread_risk_per_week_pct
        qty = risk_rs / max_loss_unit

        exit_idx = min(i + 5, len(daily) - 1)
        exit_reason = "EXPIRY"
        exit_val = 0.0

        for d in range(1, 6):
            if i + d >= len(daily):
                break
            S_cur = float(daily.iloc[i + d]["close"])
            T_rem = max((7.0 - d * 1.4) / 365.0, 0.0)
            cur_iv = max(float(daily.iloc[i + d]["rv20"]) * cfg.vrp_iv_multiplier, 0.12)
            cur_val = _bsm_price(S_cur, k_short, T_rem, cur_iv, opt_type) - _bsm_price(S_cur, k_long, T_rem, cur_iv, opt_type)

            if cur_val <= credit * 0.35:
                exit_idx, exit_val, exit_reason = i + d, cur_val, "TP_65PCT"
                break
            if cur_val >= credit * 2.5:
                exit_idx, exit_val, exit_reason = i + d, cur_val, "SL_2.5X"
                break
            exit_idx, exit_val = i + d, cur_val

        net_per_unit = credit - (exit_val * 1.04)
        if net_per_unit < 0 and cfg.spread_stress_penalty > 0:
            net_per_unit *= (1.0 + cfg.spread_stress_penalty)
        net_per_unit = max(net_per_unit, -max_loss_unit)

        net_pnl = net_per_unit * qty
        capital += net_pnl
        pnl_by_day.iloc[exit_idx] += net_pnl

        trades.append(
            PortfolioTrade(
                engine="CREDIT_SPREAD", asset=asset_name, side=side_label,
                entry_time=entry_time, exit_time=daily.index[exit_idx],
                entry_price=S0, exit_price=float(daily.iloc[exit_idx]["close"]),
                qty=qty, gross_pnl=net_pnl, funding_cost=0.0, net_pnl=net_pnl,
                r_multiple=net_pnl / risk_rs, exit_reason=exit_reason,
            )
        )
        i = exit_idx + 1

    cum_spread_pnl = pnl_by_day.cumsum()
    days_elapsed = (daily.index - daily.index[0]).days
    collateral_cum_pnl = pd.Series(
        cfg.initial_capital * ((1.0 + cfg.collateral_yield_apr) ** (days_elapsed / 365.0) - 1.0),
        index=daily.index,
    )
    return trades, cum_spread_pnl, collateral_cum_pnl
