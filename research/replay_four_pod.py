"""Offline 6h Four-Pod decision replay; execution is an explicitly modeled scenario."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from four_pod_engine import (
    CANDLE_DAYS, OHLCV, DecisionContext, DecisionRules, FundingRequest, Market, NetIntent,
    PodBook, decide_four_pod, prepare_bar, round_price,
)

PERIOD = pd.Timedelta(hours=6)
FUNDING_PERIOD = pd.Timedelta(hours=8)
# Live load_delta_candles(days=80) adds 85 calendar days for 6h and is scheduled after close.
LOOKBACK = pd.Timedelta(days=CANDLE_DAYS + 85)


@dataclass(frozen=True)
class Instrument:
    contract_value: float
    tick_size: Decimal


@dataclass(frozen=True)
class Scenario:
    capital: float = 1200.0
    half_spread: float = 0.0004
    slippage: float = 0.0005  # additional adverse price move beyond modeled touch; also used for stops
    fee_rate: float = 0.0005  # taker, all fills
    ioc_collar: float = 0.0015
    funding_rate: float = 0.0001  # signed 8h cash transfer: positive means longs pay
    squeeze_funding: float | None = None  # separate signal observation; unknown by default
    dd_threshold: float = 0.15
    dd_scale: float = 0.5
    daily_limit: float = 0.06
    weekly_limit: float = 0.12


@dataclass(frozen=True)
class Position:
    signed: int
    entry_price: float
    stop: str


@dataclass
class Replay:
    instruments: dict[str, Instrument]  # processing order matters for shared virtual margin
    scenario: Scenario
    rules: DecisionRules = field(default_factory=DecisionRules)
    books: dict[str, PodBook] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    last_bar: dict[str, str] = field(default_factory=dict)
    fills: list[dict] = field(default_factory=list)
    funding: list[dict] = field(default_factory=list)
    intents: list[dict] = field(default_factory=list)
    equity: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.peak = self.scenario.capital
        self.brake = False
        self.contract_values = {s: m.contract_value for s, m in self.instruments.items()}
        self.marks: dict[str, float] = {}
        self._last_event: pd.Timestamp | None = None
        self._stop_checked: dict[str, pd.Timestamp] = {}
        self._baselines: dict[str, tuple[str, float]] = {}

    def _fill(self, at: pd.Timestamp, sym: str, signed: int, price: float, reason: str, stop: str | None) -> None:
        old = self.positions.get(sym)
        prev = old.signed if old else 0
        after = prev + signed
        cval = self.contract_values[sym]
        gross = 0.0
        if old and prev * signed < 0:
            gross = (price - old.entry_price) * min(abs(signed), abs(prev)) * cval * (1 if prev > 0 else -1)
        fee = abs(signed) * price * cval * self.scenario.fee_rate
        self.cash += gross - fee
        if after:
            assert stop is not None and (prev == 0 or prev * after > 0)  # flips close and open separately
            if old and prev * signed > 0:
                entry = (abs(prev) * old.entry_price + abs(signed) * price) / abs(after)
            else:
                entry = old.entry_price if old else price
            self.positions[sym] = Position(after, entry, stop)
        else:
            self.positions.pop(sym)
        self.fills.append(dict(at=at, symbol=sym, reason=reason, signed_fill=signed, price=price,
                               before=prev, after=after, stop_after=stop if after else None,
                               gross_realized_usd=gross, fee_usd=fee, cash_usd=self.cash))

    def _roll_baselines(self, at: pd.Timestamp) -> None:
        for period, key in (("day", at.strftime("%Y-%m-%d")), ("week", at.strftime("%G-W%V"))):
            if self._baselines.get(period, (None,))[0] != key:
                self._baselines[period] = (key, self.cash)

    def _entries_blocked(self, blocked: bool) -> bool:
        return blocked or self.cash <= 0 or any(
            self.cash - baseline <= -min(baseline, self.scenario.capital) * limit + 1e-9
            for baseline, limit in ((self._baselines["day"][1], self.scenario.daily_limit),
                                    (self._baselines["week"][1], self.scenario.weekly_limit))
        )

    def _fund(self, at: pd.Timestamp, next_marks: dict[str, float]) -> None:
        if self._last_event is None:
            return
        funding_at = self._last_event.floor(FUNDING_PERIOD) + FUNDING_PERIOD
        while funding_at <= at:
            # UTC day/week boundaries coincide with a settlement, even across skipped closes.
            self._roll_baselines(funding_at)
            for sym, pos in self.positions.items():
                # An interior settlement can only see the previous closed mark.
                mark = next_marks.get(sym, self.marks[sym]) if funding_at == at else self.marks[sym]
                paid = pos.signed * self.contract_values[sym] * mark * self.scenario.funding_rate
                self.cash -= paid
                self.funding.append(dict(at=funding_at, symbol=sym, signed=pos.signed, mark=mark,
                                         paid_usd=paid, cash_usd=self.cash))
            funding_at += FUNDING_PERIOD

    def _sync(self, at: pd.Timestamp, sym: str, intent: NetIntent, market: Market, *, blocked: bool) -> str:
        old = self.positions.get(sym)
        cur = old.signed if old else 0
        target, stop = intent.target, intent.stop
        if cur == target:
            if old and stop is not None:
                sign = 1 if cur > 0 else -1
                if (float(stop) - float(old.stop)) * sign > 0 and (market.mark - float(stop)) * sign > 0:
                    self.positions[sym] = replace(old, stop=stop)
            return "unchanged_size"

        def ioc(size: int, after: int, reason: str) -> str:
            sign = 1 if size > 0 else -1
            price = market.mark * (1 + sign * (self.scenario.half_spread + self.scenario.slippage))
            limit = float(round_price(market.mark * (1 + sign * self.scenario.ioc_collar), market.tick_size, -sign))
            if (price - limit) * sign > 0:
                return "outside_ioc_collar"
            if after and (stop is None or float(stop) <= 0 or (market.mark - float(stop)) * after <= 0):
                return "invalid_stop"
            if abs(after) > abs(cur):
                # Model a shared wallet admission check, separate from virtual pod sizing.
                # No extra allocation/risk bucket and no pledge of unrealized gains.
                margin = sum(abs(p.signed) * self.contract_values[s] * self.marks[s]
                             for s, p in self.positions.items() if s != sym)
                margin += abs(after) * market.contract_value * max(market.mark, price)
                fee = abs(size) * market.contract_value * price * self.scenario.fee_rate
                if margin / self.rules.leverage > self.cash - fee:
                    return "wallet_margin"
            self._fill(at, sym, size, price, reason, stop)
            return "filled"

        flipped = cur * target < 0
        if cur and (target == 0 or flipped):
            result = ioc(-cur, 0, "flip_close" if flipped else "target_close")
            if target == 0 or result != "filled":
                return result
            cur = 0
        # Recheck after a successful flip close, which may have tripped a loss limit.
        if abs(target) > abs(cur) and self._entries_blocked(blocked):
            result = "entries_blocked"
        else:
            result = ioc(target - cur, target, "flip_open" if flipped else ("resize" if cur else "entry"))
        return f"flip_closed_{result}" if flipped and result != "filled" else result

    def step(self, at: pd.Timestamp, frames: dict[str, pd.DataFrame], *, blocked: bool = False,
             strict: bool = False) -> dict[str, NetIntent | str]:
        """Process a UTC close event. Frames are ordered by instrument, not by data arrival."""
        at = at.tz_convert("UTC")
        if at != at.floor(PERIOD) or (self._last_event is not None and at < self._last_event):
            raise ValueError("events must be ordered UTC 6h closes")
        prepared = {
            sym: prepare_bar(frames[sym].loc[frames[sym].index > at - LOOKBACK], at, PERIOD,
                             self.rules.donchian_entry_len)
            for sym in self.instruments
        }
        if strict:
            for sym, p in prepared.items():
                if p.bar is None:
                    raise ValueError(f"{sym} at {at}: {p.issue}")

        def stop_out(when: pd.Timestamp, sym: str, pos: Position, trigger: float) -> None:
            sign = 1 if pos.signed > 0 else -1
            price = trigger * (1 - sign * (self.scenario.half_spread + self.scenario.slippage))
            self._fill(when, sym, -pos.signed, price, "net_stop", None)
            self.books[sym] = {}

        next_marks = {sym: p.bar.closed.close for sym, p in prepared.items() if p.bar is not None}
        # On contiguous events, an opening gap crossed the stop before later settlements.
        # Funding exactly at the open has already settled in the preceding close event.
        for sym, p in prepared.items():
            pos = self.positions.get(sym)
            if pos and p.bar is not None and self._last_event == p.bar.open_at and self._stop_checked.get(sym) != at:
                opening = float(p.history.iloc[-1].open)
                if (opening <= float(pos.stop) if pos.signed > 0 else opening >= float(pos.stop)):
                    stop_out(p.bar.open_at, sym, pos, opening)
        self._fund(at, next_marks)
        self._roll_baselines(at)  # Also initializes a replay with no previous event/settlement.
        self.marks.update(next_marks)
        # Other stops are recognized at close: OHLC does not timestamp an intrabar crossing.
        for sym, p in prepared.items():
            if p.bar is None or self._stop_checked.get(sym) == at:
                continue
            self._stop_checked[sym] = at
            pos = self.positions.get(sym)
            if pos:
                stop = float(pos.stop)
                sign = 1 if pos.signed > 0 else -1
                if (p.bar.closed.low <= stop if sign > 0 else p.bar.closed.high >= stop):
                    opening = float(p.history.iloc[-1].open)
                    trigger = min(stop, opening) if sign > 0 else max(stop, opening)
                    stop_out(at, sym, pos, trigger)

        # The order-mode live cycle discards virtual books found flat at reconciliation.
        for sym in self.books:
            if sym not in self.positions:
                self.books[sym] = {}
        self.peak = max(self.peak, self.cash)
        dd = (self.peak - self.cash) / self.peak
        if not self.scenario.dd_threshold or dd <= self.scenario.dd_threshold * 0.5:
            self.brake = False
        elif dd >= self.scenario.dd_threshold:
            self.brake = True
        result: dict[str, NetIntent | str] = {}
        for sym, p in prepared.items():
            bar = p.bar
            if bar is None:
                result[sym] = "stale" if p.stale else "warmup"
                continue
            if self.last_bar.get(sym) == bar.key:
                result[sym] = "duplicate"
                continue
            self.books.setdefault(sym, {})
            instrument = self.instruments[sym]
            market = Market(self.marks[sym], instrument.contract_value, instrument.tick_size)
            context = DecisionContext(min(self.cash, self.scenario.capital), self.scenario.dd_scale if self.brake else 1.,
                                      not self._entries_blocked(blocked), len(self.instruments))
            args = (sym, bar, market, context, self.rules, self.books, self.contract_values)
            intent = decide_four_pod(*args)
            funding_requested = intent is FundingRequest.NEEDED
            if funding_requested:
                intent = decide_four_pod(*args, funding=self.scenario.squeeze_funding)
            assert isinstance(intent, NetIntent)
            self.books[sym] = intent.book
            execution = self._sync(at, sym, intent, market, blocked=blocked)
            self.last_bar[sym] = bar.key
            pos = self.positions.get(sym)
            self.intents.append(dict(at=at, bar_open=bar.key, symbol=sym, mark=market.mark,
                                     bid=market.mark * (1 - self.scenario.half_spread),
                                     ask=market.mark * (1 + self.scenario.half_spread),
                                     allocation_usd=context.allocation, can_open=context.can_open, dd_scale=context.dd_scale,
                                     funding_requested=funding_requested, squeeze_funding=self.scenario.squeeze_funding
                                     if funding_requested else None, target=intent.target, stop=intent.stop,
                                     book=json.dumps(intent.book, allow_nan=False), execution=execution,
                                     filled_target=pos.signed if pos else 0, active_stop=pos.stop if pos else None))
            result[sym] = intent
        mtm = sum((self.marks[sym] - p.entry_price) * p.signed * self.contract_values[sym]
                  for sym, p in self.positions.items())
        row = dict(at=at, cash_usd=self.cash, unrealized_usd=mtm, equity_usd=self.cash + mtm,
                   positions=json.dumps({sym: asdict(p) for sym, p in self.positions.items()}))
        if self._last_event == at:
            self.equity[-1] = row
        else:
            self.equity.append(row)
        self._last_event = at
        return result


def read_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={name: str for name in OHLCV})
    index = frame.pop("timestamp").map(pd.Timestamp)
    if any(pd.isna(t) or t.tzinfo is None for t in index):
        raise ValueError(f"{path}: candle timestamps must include a timezone")
    frame.index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return frame[OHLCV].astype(float)


def validate_coverage(sym: str, frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = pd.date_range(start - LOOKBACK + PERIOD, end - PERIOD, freq=PERIOD)
    used = frame.loc[(frame.index >= required[0]) & (frame.index <= required[-1])]
    if not used.index.equals(required):
        missing = required.difference(used.index)
        more = f"; first missing={missing[0]}" if len(missing) else ""
        raise ValueError(f"{sym}: need ordered unique 6h candles for {required[0]} to {required[-1]}{more}")
    if (not np.isfinite(used[OHLCV]).all(axis=None) or (used[OHLCV[:-1]] <= 0).any(axis=None)
            or (used.volume < 0).any() or (used.high < used[["open", "low", "close"]].max(axis=1)).any()
            or (used.low > used[["open", "close"]].min(axis=1)).any()):
        raise ValueError(f"{sym}: invalid OHLCV in required window")
    return used


def demo_frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-01", "2024-09-01", freq=PERIOD, tz="UTC")
    i = np.arange(len(index))
    frames = {}
    for symbol, phase in (("DEMO-A", 0.), ("DEMO-B", 2.)):
        close = 100 + i / 12 + 25 * np.sin(i / 28 + phase) + 4 * np.sin(i / 2)
        opening = np.r_[close[0], close[:-1]]
        frames[symbol] = pd.DataFrame(dict(open=opening, high=np.maximum(opening, close) + 1,
                                           low=np.minimum(opening, close) - 1, close=close,
                                           volume=np.where(i % 11 == 0, 3000., 1000.)), index=index)
    return frames


def load_manifest(path: Path) -> tuple[dict, dict, dict]:
    # A list, rather than key-sorted JSON, explicitly defines portfolio order.
    instruments, frames, sources = {}, {}, {}
    for item in json.loads(path.read_text()):
        sym = item["symbol"]
        if not sym or sym in instruments:
            raise ValueError(f"empty or duplicate symbol: {sym!r}")
        source = {k: item[k] for k in ("venue", "kind", "source", "metadata_source")}
        if source["kind"] not in {"venue", "proxy", "synthetic"} or not all(source.values()):
            raise ValueError(f"{sym}: declare venue, kind (venue/proxy/synthetic), source, metadata_source")
        csv_path = (path.parent / item["csv"]).resolve()
        instrument = Instrument(float(item["contract_value"]), Decimal(str(item["tick_size"])))
        if (not math.isfinite(instrument.contract_value) or instrument.contract_value <= 0
                or not instrument.tick_size.is_finite() or instrument.tick_size <= 0):
            raise ValueError(f"{sym}: contract value and tick must be positive and finite")
        frames[sym] = read_candles(csv_path)
        instruments[sym] = instrument
        sources[sym] = dict(source, csv=str(csv_path), sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                            interval="6h", contract_value=instrument.contract_value, tick_size=str(instrument.tick_size))
    if not instruments:
        raise ValueError("manifest needs at least one instrument")
    return instruments, frames, sources


def close_event(value: str) -> pd.Timestamp:
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None or at.tz_convert("UTC").floor(PERIOD) != at:
        raise ValueError("start/end need timezone-qualified 6h close timestamps")
    return at.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="ordered JSON list of local candle files and instrument metadata")
    group.add_argument("--demo", action="store_true", help="use built-in synthetic 6h events")
    parser.add_argument("--start", default="2024-07-01T00:00Z", help="first decision close (inclusive)")
    parser.add_argument("--end", default="2024-08-01T00:00Z", help="last decision close (inclusive)")
    parser.add_argument("--output", required=True, type=Path, help="new local output directory")
    defaults = Scenario()
    for name in ("capital", "half_spread", "slippage", "fee_rate", "ioc_collar", "funding_rate", "squeeze_funding"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    args = parser.parse_args()
    scenario = replace(defaults, **{k: getattr(args, k) for k in (
        "capital", "half_spread", "slippage", "fee_rate", "ioc_collar", "funding_rate", "squeeze_funding",
    )})
    try:
        start, end = close_event(args.start), close_event(args.end)
        if end < start or args.output.exists():
            raise ValueError("require start <= end and a new output directory")
        if (not all(math.isfinite(v) for v in asdict(scenario).values() if v is not None)
                or scenario.capital <= 0 or not 0 <= scenario.half_spread + scenario.slippage < 1
                or min(scenario.half_spread, scenario.slippage, scenario.fee_rate, scenario.ioc_collar) < 0
                or scenario.ioc_collar >= 1):
            raise ValueError("invalid scenario costs or capital")
        if args.demo:
            frames = demo_frames()
            instruments = {sym: Instrument(0.1, Decimal("0.01")) for sym in frames}
            sources = {sym: dict(venue="none", kind="synthetic", source="built-in demo_frames",
                                 interval="6h", contract_value=0.1, tick_size="0.01", metadata_source="synthetic")
                       for sym in frames}
        else:
            instruments, frames, sources = load_manifest(args.manifest)
        for sym, frame in frames.items():
            frames[sym] = validate_coverage(sym, frame, start, end)
            sources[sym].update(first_bar_open=str(frames[sym].index[0]), last_bar_open=str(frames[sym].index[-1]),
                                bars=len(frames[sym]))
        replay = Replay(instruments, scenario)
        for at in pd.date_range(start, end, freq=PERIOD):
            replay.step(at, frames, strict=True)
        gross = math.fsum(row["gross_realized_usd"] for row in replay.fills)
        fees = math.fsum(row["fee_usd"] for row in replay.fills)
        funding = math.fsum(row["paid_usd"] for row in replay.funding)
        final = replay.equity[-1]
        if not math.isclose(scenario.capital + gross - fees - funding, final["cash_usd"], abs_tol=1e-8):
            raise ValueError("fill/funding ledger does not reconcile to cash")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    fill_cols = "at symbol reason signed_fill price before after stop_after gross_realized_usd fee_usd cash_usd".split()
    pd.DataFrame(replay.fills, columns=fill_cols).to_csv(args.output / "fills.csv", index=False)
    fund_cols = "at symbol signed mark paid_usd cash_usd".split()
    pd.DataFrame(replay.funding, columns=fund_cols).to_csv(args.output / "funding.csv", index=False)
    pd.DataFrame(replay.intents).to_csv(args.output / "intents.csv", index=False)
    pd.DataFrame(replay.equity).to_csv(args.output / "equity.csv", index=False)
    root = Path(__file__).resolve().parents[1]
    code = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in ("four_pod_engine.py", "portfolio_engine.py", "research/replay_four_pod.py")}
    summary = dict(strategy="four_pod_balanced", execution="modeled, not historical fills", currency="USD",
                   interval="6h", mark_source="last closed candle close", code_sha256=code,
                   start=str(start), end=str(end), events=len(replay.equity), intents=len(replay.intents),
                   fills=len(replay.fills), gross_realized_usd=gross, fees_usd=fees, funding_paid_usd=funding,
                   final=final, scenario=asdict(scenario), decision_rules=asdict(replay.rules), instruments=sources)
    text = json.dumps(summary, default=str, allow_nan=False, indent=2)
    (args.output / "run.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
