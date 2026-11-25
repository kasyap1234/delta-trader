"""Four-Pod decision rules. Broker policy and clocks belong to the adapters."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, fields
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Mapping, TypedDict

import numpy as np
import pandas as pd

from portfolio_engine import add_3pod_features

if TYPE_CHECKING:
    from config import DeltaLiveConfig


CANDLE_DAYS = 80  # Live Delta loader adds its interval-specific warmup padding.
FEATURE_WARMUP = 240
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DecisionRules:
    # Offline defaults mirror DeltaLiveConfig; live takes the actual configured values.
    donchian_entry_len: int = 20
    risk_per_trade_pct: float = 0.02
    leverage: int = 5
    max_margin_utilization: float = 0.75
    atr_trail_mult: float = 3.5
    pyramid_add_fraction: float = 0.5
    pyramid_trigger_r: float = 1.5
    pod_weight_donchian: float = 0.30
    pod_weight_supertrend: float = 0.25
    pod_weight_tsmom: float = 0.25
    pod_weight_squeeze: float = 0.35
    tsmom_target_vol: float = 0.35
    tsmom_rebalance_bars: int = 8
    squeeze_atr_trail_mult: float = 3.0
    squeeze_vol_mult: float = 1.5
    squeeze_long_max_funding: float = 0.00005
    squeeze_short_min_funding: float = 0.00015
    catastrophe_stop_atr_mult: float = 4.5

    @classmethod
    def from_live_config(cls, cfg: DeltaLiveConfig) -> DecisionRules:
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})


class Pod(TypedDict):
    """Persisted legacy shape. Donchian additionally carries its pyramid state."""
    side: Literal["LONG", "SHORT"]
    contracts: int
    entry_price: float
    sl_price: float | None


class DonchianPod(Pod):
    init_entry: float
    init_sl_dist: float
    pyramided: bool


PodBook = dict[str, Pod | DonchianPod]


@dataclass(frozen=True)
class Features:
    close: float
    high: float
    low: float
    volume: float
    atr14: float
    ema50: float
    ema200: float
    donchian_high: float
    donchian_low: float
    st_dir: int
    adx14: float
    vol28: float
    ret_7d: float
    ret_21d: float
    ret_60d: float
    vol_ma20: float

    @classmethod
    def from_row(cls, row: pd.Series) -> Features:
        return cls(**{f.name: (int if f.name == "st_dir" else float)(row[f.name]) for f in fields(cls)})


@dataclass(frozen=True)
class ClosedBar:
    open_at: pd.Timestamp
    period: pd.Timedelta
    closed: Features
    prior: Features

    @property
    def key(self) -> str:
        return self.open_at.isoformat()


@dataclass(frozen=True)
class Market:
    mark: float
    contract_value: float
    tick_size: Decimal


@dataclass(frozen=True)
class DecisionContext:
    allocation: float
    dd_scale: float
    can_open: bool
    symbol_count: int


@dataclass(frozen=True)
class PreparedBar:
    history: pd.DataFrame
    bar: ClosedBar | None = None
    issue: str | None = None
    stale: bool = False


class InvalidClosedBar(ValueError):
    pass


def prepare_bar(raw: pd.DataFrame, as_of: pd.Timestamp, period: pd.Timedelta, entry_len: int) -> PreparedBar:
    """Use the final continuous, finite suffix of the adapter's candle window."""
    expected = as_of.tz_convert("UTC").floor(period) - period
    history = raw.loc[raw.index <= expected]
    if history.empty or history.index[-1] != expected:
        return PreparedBar(history, issue="latest closed candle is unavailable", stale=True)
    history = history.loc[np.isfinite(history[OHLCV]).all(axis=1)]
    if expected not in history.index:
        raise InvalidClosedBar("invalid latest closed candle")
    gaps = history.index.to_series().diff().ne(period)
    contiguous = history.loc[gaps[gaps].index[-1]:]
    if len(contiguous) <= FEATURE_WARMUP + 1:
        return PreparedBar(history, issue=f"insufficient continuous candles for 3-pod features ({len(contiguous)} <= 241)")
    features = add_3pod_features(contiguous, entry_len=entry_len, warmup_bars=FEATURE_WARMUP)
    if expected not in features.index or len(features) < 2:
        return PreparedBar(history, issue="insufficient 3-pod feature rows")
    return PreparedBar(history, ClosedBar(expected, period, Features.from_row(features.loc[expected]),
                                         Features.from_row(features.iloc[-2])))


def round_price(price: float, tick: Decimal, sign: int) -> str:
    ticks = (Decimal(str(price)) / tick).to_integral_value(rounding=ROUND_CEILING if sign > 0 else ROUND_FLOOR)
    return format(ticks * tick, "f")


class FundingRequest(Enum):
    NEEDED = auto()


@dataclass(frozen=True)
class NetIntent:
    book: PodBook
    target: int
    stop: str | None


def decide_four_pod(
    symbol: str, bar: ClosedBar, market: Market, context: DecisionContext, rules: DecisionRules,
    books: Mapping[str, PodBook], contract_values: Mapping[str, float],
    funding: float | None | FundingRequest = FundingRequest.NEEDED,
) -> NetIntent | FundingRequest:
    """Return a new book and intent without mutating inputs or doing I/O.

    A funding request is not an intent. Retry with the same snapshot and the
    as-of funding observation (None means unavailable, which gates entry).
    The gross virtual margin pass uses portfolio/pod insertion order, including
    effects of earlier pods within the same symbol decision.
    """
    sym_pods = deepcopy(books.get(symbol, {}))
    portfolio = {**books, symbol: sym_pods}
    closed, prev_closed = bar.closed, bar.prior
    atr, closed_px = closed.atr14, closed.close
    bar_high, bar_low = closed.high, closed.low
    mark_px, cval = market.mark, market.contract_value
    allocation, dd_scale, can_open = context.allocation, context.dd_scale, context.can_open

    def free_margin_budget() -> float:
        used = 0.0
        for sym, book in portfolio.items():
            value = cval if sym == symbol else contract_values.get(sym)
            if value is None or not isinstance(book, dict):
                continue
            for pod in book.values():
                if pod:
                    used += abs(int(pod["contracts"])) * value * float(pod["entry_price"]) / rules.leverage
        return max(rules.max_margin_utilization * allocation - used, 0.0)

    # Pod 1: Donchian
    dc = sym_pods.get("DONCHIAN")
    if dc is not None:
        s_dc = 1 if dc["side"] == "LONG" else -1
        if (bar_low <= float(dc["sl_price"])) if s_dc > 0 else (bar_high >= float(dc["sl_price"])):
            sym_pods.pop("DONCHIAN", None)
            dc = None
        else:
            new_sl = closed_px - s_dc * rules.atr_trail_mult * atr
            if (new_sl - float(dc["sl_price"])) * s_dc > 0:
                dc["sl_price"] = new_sl
            # Pyramid check (+1.5R and SL >= breakeven)
            if can_open and rules.pyramid_add_fraction > 0 and not dc.get("pyramided"):
                be_ok = (float(dc["sl_price"]) - float(dc["init_entry"])) * s_dc >= 0
                move = (mark_px - float(dc["init_entry"])) * s_dc
                if be_ok and move >= rules.pyramid_trigger_r * float(dc["init_sl_dist"]):
                    add_c = min(
                        math.floor(int(dc["contracts"]) * rules.pyramid_add_fraction),
                        math.floor(free_margin_budget() * rules.leverage / (mark_px * cval)),
                    )
                    if add_c >= 1:
                        dc["contracts"] = int(dc["contracts"]) + add_c
                        dc["pyramided"] = True
    elif can_open and rules.pod_weight_donchian > 0 and atr > 0:
        dc_side = None
        if closed.ema50 > closed.ema200 and closed_px > float(closed.donchian_high):
            dc_side = "LONG"
        elif closed.ema50 < closed.ema200 and closed_px < float(closed.donchian_low):
            dc_side = "SHORT"
        if dc_side is not None:
            dc_risk = rules.risk_per_trade_pct * rules.pod_weight_donchian * dd_scale
            sl_dist = rules.atr_trail_mult * atr
            c_qty = math.floor(min(
                allocation * dc_risk / (sl_dist * cval),
                free_margin_budget() * rules.leverage / (mark_px * cval),
            ))
            if c_qty >= 1:
                s_dc = 1 if dc_side == "LONG" else -1
                sym_pods["DONCHIAN"] = {
                    "side": dc_side, "contracts": c_qty, "entry_price": mark_px,
                    "init_entry": mark_px, "init_sl_dist": sl_dist,
                    "sl_price": mark_px - s_dc * sl_dist, "pyramided": False,
                }

    # Pod 2: Supertrend(10,3) + ADX(14) >= 20
    st = sym_pods.get("SUPERTREND")
    st_dir = int(closed.st_dir)
    prev_st_dir = int(prev_closed.st_dir)
    if st is not None:
        s_st = 1 if st["side"] == "LONG" else -1
        flipped = (s_st > 0 and st_dir == -1) or (s_st < 0 and st_dir == 1)
        sl_hit = (bar_low <= float(st["sl_price"])) if s_st > 0 else (bar_high >= float(st["sl_price"]))
        if flipped or sl_hit:
            sym_pods.pop("SUPERTREND", None)
            st = None
        else:
            new_sl = closed_px - s_st * 3.5 * atr
            if (new_sl - float(st["sl_price"])) * s_st > 0:
                st["sl_price"] = new_sl
    elif can_open and rules.pod_weight_supertrend > 0 and atr > 0:
        st_side = None
        if st_dir == 1 and prev_st_dir == -1 and float(closed.adx14) >= 20.0:
            st_side = "LONG"
        elif st_dir == -1 and prev_st_dir == 1 and float(closed.adx14) >= 20.0:
            st_side = "SHORT"
        if st_side is not None:
            st_risk = rules.risk_per_trade_pct * rules.pod_weight_supertrend * dd_scale
            sl_dist = 3.5 * atr
            c_qty = math.floor(min(
                allocation * st_risk / (sl_dist * cval),
                free_margin_budget() * rules.leverage / (mark_px * cval),
            ))
            if c_qty >= 1:
                s_st = 1 if st_side == "LONG" else -1
                sym_pods["SUPERTREND"] = {
                    "side": st_side, "contracts": c_qty, "entry_price": mark_px,
                    "sl_price": mark_px - s_st * sl_dist,
                }

    # Pod 3: TSMOM volatility targeting on UTC epoch slots
    bar_slot = int(bar.open_at.timestamp() // int(bar.period.total_seconds()))
    if rules.pod_weight_tsmom > 0 and (bar_slot % rules.tsmom_rebalance_bars == 0):
        vol28 = float(closed.vol28)
        sig = (np.sign(float(closed.ret_7d)) + np.sign(float(closed.ret_21d)) + np.sign(float(closed.ret_60d))) / 3.0
        if vol28 <= 0.05 or abs(sig) < 0.34 or not can_open:
            sym_pods.pop("TSMOM", None)
        else:
            ts_side = "LONG" if sig > 0 else "SHORT"
            cur_ts = sym_pods.get("TSMOM")
            if cur_ts is not None and cur_ts["side"] != ts_side:
                sym_pods.pop("TSMOM", None)
                cur_ts = None
            if cur_ts is None:
                t_vol = rules.tsmom_target_vol * rules.pod_weight_tsmom * dd_scale
                weight = min((t_vol / vol28) * abs(sig) / max(context.symbol_count, 1), 0.40 * rules.pod_weight_tsmom)
                notional = min(allocation * weight, free_margin_budget() * rules.leverage)
                c_qty = math.floor(notional / (mark_px * cval))
                if c_qty >= 1:
                    sym_pods["TSMOM"] = {
                        "side": ts_side, "contracts": c_qty, "entry_price": mark_px,
                        "sl_price": None,
                    }

    # Pod 4: Funding + volume squeeze
    sq = sym_pods.get("SQUEEZE")
    if sq is not None:
        s_sq = 1 if sq["side"] == "LONG" else -1
        if (bar_low <= float(sq["sl_price"])) if s_sq > 0 else (bar_high >= float(sq["sl_price"])):
            sym_pods.pop("SQUEEZE", None)
            sq = None
        else:
            new_sl = closed_px - s_sq * rules.squeeze_atr_trail_mult * atr
            if (new_sl - float(sq["sl_price"])) * s_sq > 0:
                sq["sl_price"] = new_sl
    elif can_open and rules.pod_weight_squeeze > 0 and atr > 0:
        vol_ma = float(closed.vol_ma20)
        vol_surge = vol_ma > 0 and float(closed.volume) > rules.squeeze_vol_mult * vol_ma
        if vol_surge and (closed_px > float(closed.donchian_high) or closed_px < float(closed.donchian_low)):
            if funding is FundingRequest.NEEDED:
                return FundingRequest.NEEDED
            fr_3d = funding
            sq_side = None
            if fr_3d is not None:
                if closed_px > float(closed.donchian_high) and fr_3d <= rules.squeeze_long_max_funding:
                    sq_side = "LONG"
                elif closed_px < float(closed.donchian_low) and fr_3d >= rules.squeeze_short_min_funding:
                    sq_side = "SHORT"
            if sq_side is not None:
                sq_risk = rules.risk_per_trade_pct * rules.pod_weight_squeeze * dd_scale
                sl_dist = rules.squeeze_atr_trail_mult * atr
                c_qty = math.floor(min(
                    allocation * sq_risk / (sl_dist * cval),
                    free_margin_budget() * rules.leverage / (mark_px * cval),
                ))
                if c_qty >= 1:
                    s_sq = 1 if sq_side == "LONG" else -1
                    sym_pods["SQUEEZE"] = {
                        "side": sq_side, "contracts": c_qty, "entry_price": mark_px,
                        "sl_price": mark_px - s_sq * sl_dist,
                    }

    # Compute signed net target contracts and tightest aligned protective stop.
    net_signed = 0
    for p in sym_pods.values():
        if p:
            net_signed += int(p["contracts"]) * (1 if p["side"] == "LONG" else -1)

    net_sl_str = None
    if net_signed != 0:
        net_sign = 1 if net_signed > 0 else -1
        aligned_stops = [
            float(p["sl_price"]) for p in sym_pods.values()
            if p and p.get("sl_price") is not None and (1 if p["side"] == "LONG" else -1) == net_sign
        ]
        if aligned_stops:
            raw_sl = max(aligned_stops) if net_sign > 0 else min(aligned_stops)
        else:
            raw_sl = mark_px - net_sign * rules.catastrophe_stop_atr_mult * atr
        # Ensure stop is strictly on the protective side of mark_px
        if (mark_px - raw_sl) * net_sign <= 0:
            raw_sl = mark_px - net_sign * rules.atr_trail_mult * atr
        net_sl_str = round_price(raw_sl, market.tick_size, net_sign)

    return NetIntent(sym_pods, net_signed, net_sl_str)
