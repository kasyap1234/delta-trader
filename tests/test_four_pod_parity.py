"""Exact same-input comparison through the live and offline Four-Pod adapters."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import four_pod_engine as engine
import live_delta_donchian as delta
from config import DeltaLiveConfig
from research.replay_four_pod import Instrument, PERIOD, Position, Replay, Scenario, demo_frames, validate_coverage


AT = pd.Timestamp("2024-07-01T00:00Z")
SYMBOL = "ETHUSD"
MARK = 100.03


def candles(at, mark=MARK):
    # Includes the unclosed next candle. Both adapters must end at at - PERIOD.
    index = pd.date_range(end=at, periods=301, freq=PERIOD).tz_convert("Asia/Kolkata")
    return pd.DataFrame(dict(open=mark, close=mark, high=mark + 1, low=mark - 1, volume=300.), index=index)


def indicator_fixture(st_dir=1, trend=1, breakout=True, surge=True):
    """Script only indicators; OHLC for stop checks still comes from the adapter input."""
    def features(frame, *, entry_len, warmup_bars):
        assert warmup_bars == 240
        out = frame.iloc[warmup_bars:].assign(
            atr14=2., ema50=105. if breakout else 100., ema200=100.,
            donchian_high=99. if breakout else 110., donchian_low=90., st_dir=-st_dir,
            adx14=20., vol28=.5, ret_7d=trend, ret_21d=trend, ret_60d=trend,
            vol_ma20=100. if surge else 300.,
        )
        out.iat[-1, out.columns.get_loc("st_dir")] = st_dir
        return out
    return features


class AdapterParityTest(unittest.TestCase):
    def adapters(self, *, funding=None, **cfg_args):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cfg = DeltaLiveConfig(
            env="testnet", dry_run=True, strategy_mode="four_pod_balanced", resolution="6h", symbols=[SYMBOL],
            capital_usd=1200., api_key="", api_secret="", data_dir=directory.name, telegram_token="",
            telegram_chat_id="", use_maker_entry=False, maker_timeout_sec=0., entry_slippage_pct=0., **cfg_args,
        )
        live = delta.LiveDonchianEngine(cfg)
        instrument = Instrument(.1, Decimal("0.1"))
        live.client.products_cache[SYMBOL] = dict(contract_value=instrument.contract_value, tick_size=instrument.tick_size,
                                                   fee_rate=0.)
        replay = Replay({SYMBOL: instrument}, replace(Scenario(), half_spread=0., slippage=0., fee_rate=0.,
                                                     funding_rate=0., squeeze_funding=funding),
                        engine.DecisionRules.from_live_config(cfg))
        return live, replay

    def both(self, adapters, at, frame, *, funding=None, blocked=False):
        live, replay = adapters
        expected = at - PERIOD
        mark = frame.at[expected, "close"] if expected in frame.index else MARK
        with ExitStack() as stack:
            stack.enter_context(patch.object(delta.pd.Timestamp, "now", return_value=at.tz_convert("UTC")))
            stack.enter_context(patch.object(delta, "load_delta_candles", return_value=frame))
            quotes = stack.enter_context(patch.object(live.client, "get_ticker_quotes", return_value=(mark, mark, mark)))
            fr = stack.enter_context(patch.object(delta, "load_delta_funding_avg", return_value=funding))
            sync = stack.enter_context(patch.object(live, "_sync_net_position", wraps=live._sync_net_position))
            stack.enter_context(patch.object(live, "_save_state"))
            stack.enter_context(patch.object(live, "_log_journal"))
            retry = live._process_3pod_symbol(SYMBOL, ["blocked"] if blocked else [], available=None)
        intent = replay.step(at, {SYMBOL: frame}, blocked=blocked)[SYMBOL]
        self.assertEqual(retry, intent == "stale")
        if isinstance(intent, engine.NetIntent):
            sync.assert_called_once()
            args = sync.call_args.args
            self.assertEqual((args[2], args[3]), (intent.target, intent.stop))
            self.assertEqual((args[7], args[8]), (expected, PERIOD))
            self.assertEqual(live.state["pods"][SYMBOL], intent.book)
            self.assertEqual(live.state["last_entry_bar"], replay.last_bar)
            quotes.assert_called_once_with(SYMBOL)
            if replay.intents[-1]["funding_requested"]:
                fr.assert_called_once_with(SYMBOL, days=4, base_url=delta.DELTA_INDIA_BASE)
            else:
                fr.assert_not_called()
        else:
            sync.assert_not_called()
            fr.assert_not_called()
            if intent == "stale":
                quotes.assert_not_called()
        return intent

    def seed(self, adapters, book, pos, checked):
        live, replay = adapters
        live.state["pods"][SYMBOL] = deepcopy(book)
        replay.books[SYMBOL] = deepcopy(book)
        live.state["positions"][SYMBOL] = dict(
            side="LONG" if pos.signed > 0 else "SHORT", contracts=abs(pos.signed), entry_price=pos.entry_price,
            contract_value=.1, sl_price=pos.stop, paper_checked_at=checked.isoformat(),
        )
        replay.positions[SYMBOL] = pos

    def test_staged_squeeze_and_exact_net_stop_with_duplicate_and_stale(self):
        adapters = self.adapters(funding=.00005)
        frame = candles(AT)
        frame.iloc[-1] = [100., 1e9, 1., 1e9, 1e9]  # future candle
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()):
            intent = self.both(adapters, AT, frame, funding=.00005)
            self.assertEqual({k: p["contracts"] for k, p in intent.book.items()},
                             {"DONCHIAN": 10, "SUPERTREND": 8, "SQUEEZE": 13})
            self.assertEqual((intent.target, intent.stop), (31, "94.1"))
            self.assertEqual(self.both(adapters, AT, frame), "duplicate")
            self.assertEqual(self.both(adapters, AT + 2 * PERIOD, frame), "stale")
        self.assertEqual(len(adapters[1].fills), 1)
        self.assertEqual(len(adapters[1].intents), 1)

    def test_utc_momentum_open_slot_flip_closes_then_opens_and_catastrophe_stop(self):
        adapters = self.adapters(pod_weight_donchian=0., pod_weight_supertrend=0., pod_weight_squeeze=0.)
        rebalance = pd.Timestamp("2024-07-02T06:00Z").tz_convert("Asia/Kolkata")
        self.seed(adapters, {"TSMOM": dict(side="LONG", contracts=11, entry_price=MARK, sl_price=None)},
                  Position(11, MARK, "91.1"), rebalance - 2 * PERIOD)
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture(trend=-1, breakout=False)):
            prior = self.both(adapters, rebalance - PERIOD, candles(rebalance - PERIOD))
            self.assertEqual((prior.target, prior.stop), (11, "91.1"))
            intent = self.both(adapters, rebalance, candles(rebalance))
        self.assertEqual((intent.target, intent.stop), (-11, "109.0"))
        self.assertEqual(intent.book["TSMOM"], dict(side="SHORT", contracts=11, entry_price=MARK, sl_price=None))
        self.assertEqual([f["reason"] for f in adapters[1].fills], ["flip_close", "flip_open"])

    def test_sequential_virtual_margin_caps_later_pods_and_funding_request_remains_lazy(self):
        adapters = self.adapters(risk_per_trade_pct=.5, max_margin_utilization=.1, dd_brake_threshold=.15)
        live, replay = adapters
        live.state.update(equity_usd=1000., dd_brake_active=True)
        replay.cash = 1000.
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()):
            intent = self.both(adapters, AT, candles(AT))
        self.assertEqual((intent.target, intent.stop), (49, "93.1"))
        self.assertEqual(set(intent.book), {"DONCHIAN"})
        self.assertTrue(replay.intents[-1]["funding_requested"])  # old code requests even when margin is exhausted
        self.assertEqual(replay.intents[-1]["dd_scale"], .5)
        # Entry blockers suppress the Squeeze request, but may not suppress intent from an existing book.
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()):
            self.assertEqual(self.both(adapters, AT + PERIOD, candles(AT + PERIOD), blocked=True).target, 49)
        self.assertFalse(replay.intents[-1]["funding_requested"])

    def test_net_stop_gap_clears_book_before_intent_even_if_pod_stop_not_hit(self):
        adapters = self.adapters(pod_weight_tsmom=0.)
        self.seed(adapters, {"SUPERTREND": dict(side="LONG", contracts=5, sl_price=70., entry_price=100.)},
                  Position(5, 100., "95.0"), AT - PERIOD)
        frame = candles(AT, mark=100.)
        frame.loc[AT - PERIOD, ["open", "low"]] = [93., 92.]
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture(breakout=False, surge=False)):
            intent = self.both(adapters, AT, frame, blocked=True)
        self.assertEqual(intent, engine.NetIntent({}, 0, None))
        live, sim = adapters
        self.assertEqual((live.state["positions"], sim.positions), ({}, {}))
        self.assertEqual(live.state["equity_usd"], sim.cash)
        fill, = sim.fills
        self.assertEqual((fill["reason"], fill["signed_fill"], fill["price"]), ("net_stop", -5, 93.))

    def test_real_feature_window_and_offline_coverage_guard(self):
        adapters = self.adapters()
        frame = demo_frames()["DEMO-A"]
        frame = frame.loc[frame.index > AT - pd.Timedelta(days=165)].copy()
        required = validate_coverage(SYMBOL, frame, AT, AT)
        intent = self.both(adapters, AT, frame)
        self.assertEqual(intent, self.both(self.adapters(), AT, required))
        self.assertEqual(engine.prepare_bar(frame, AT, PERIOD, 20).bar,
                         engine.prepare_bar(required, AT, PERIOD, 20).bar)
        with self.assertRaisesRegex(ValueError, "first missing="):
            validate_coverage(SYMBOL, frame.drop(required.index[20]), AT, AT)
        with self.assertRaisesRegex(ValueError, "ordered unique"):
            validate_coverage(SYMBOL, pd.concat([required, required.iloc[-1:]]), AT, AT)
        with self.assertRaisesRegex(ValueError, "insufficient continuous"):
            self.adapters()[1].step(AT, {SYMBOL: required.iloc[-241:]}, strict=True)

    def test_replay_collar_and_signed_funding_use_available_marks(self):
        instrument = Instrument(.1, Decimal("0.01"))
        rules = replace(engine.DecisionRules(), pod_weight_tsmom=0., pod_weight_squeeze=0.)
        sim = Replay({SYMBOL: instrument}, Scenario(), rules)
        rejected = Replay({SYMBOL: instrument}, replace(Scenario(), half_spread=.01), rules)
        at = AT + PERIOD  # 06:00 UTC; funding settles at 08:00 before the next decision.
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()):
            self.assertEqual(sim.step(at, {SYMBOL: candles(at)}), rejected.step(at, {SYMBOL: candles(at)}))
            sim.step(at + PERIOD, {SYMBOL: candles(at + PERIOD, mark=MARK + 1)})
        self.assertEqual(rejected.intents[-1]["execution"], "outside_ioc_collar")
        self.assertEqual(rejected.fills, [])
        fill, = sim.fills
        funding, = sim.funding
        self.assertEqual(funding["at"], AT + pd.Timedelta(hours=8))
        self.assertEqual(funding["mark"], MARK)  # not the 12:00 close
        self.assertAlmostEqual(funding["paid_usd"], fill["signed_fill"] * .1 * MARK * .0001)
        self.assertAlmostEqual(sim.cash, 1200. - fill["fee_usd"] - funding["paid_usd"])
        self.assertAlmostEqual(sim.equity[-1]["equity_usd"], sim.cash + sim.equity[-1]["unrealized_usd"])

    def test_replay_gap_open_stop_precedes_interior_funding_but_intrabar_stop_does_not(self):
        rules = replace(engine.DecisionRules(), pod_weight_tsmom=0., pod_weight_squeeze=0.)
        instrument = Instrument(.1, Decimal("0.01"))
        gap, intrabar = (Replay({SYMBOL: instrument}, Scenario(), rules) for _ in range(2))
        at = AT + PERIOD  # entry at 06:00; next candle contains the 08:00 funding boundary
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()):
            for sim in (gap, intrabar):
                sim.step(at, {SYMBOL: candles(at)})
            frame = candles(at + PERIOD)
            frame.loc[at, ["open", "low"]] = [80., 79.]
            gap.step(at + PERIOD, {SYMBOL: frame}, blocked=True)
            frame.loc[at, "open"] = MARK
            intrabar.step(at + PERIOD, {SYMBOL: frame}, blocked=True)
        self.assertEqual(gap.funding, [])
        self.assertEqual(gap.fills[-1]["at"], at)
        self.assertEqual(gap.fills[-1]["reason"], "net_stop")
        self.assertEqual(intrabar.funding[0]["at"], AT + pd.Timedelta(hours=8))
        self.assertEqual(intrabar.fills[-1]["at"], at + PERIOD)

    def test_sunday_open_gap_exit_precedes_monday_baselines_and_funding(self):
        at = AT - PERIOD  # Sunday 18:00 UTC, followed by the day/week boundary.
        other = "OTHER"
        instrument = Instrument(.1, Decimal("0.01"))
        scenario = replace(Scenario(), half_spread=0., slippage=0., fee_rate=0., funding_rate=.001,
                           daily_limit=.01, weekly_limit=.01)
        sim = Replay({SYMBOL: instrument, other: instrument}, scenario)
        sim.positions.update({SYMBOL: Position(10, 100., "95"), other: Position(4, 100., "90")})
        intents = (engine.NetIntent({}, 10, "95"), engine.NetIntent({}, 4, "90"),
                   engine.NetIntent({}, 1, "90"), engine.NetIntent({}, 4, "90"))
        with (patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()),
              patch("research.replay_four_pod.decide_four_pod", side_effect=intents)):
            sim.step(at, {SYMBOL: candles(at, 100.), other: candles(at, 100.)})
            gap = candles(AT, 100.)
            gap.loc[at, ["open", "low"]] = [80., 79.]
            sim.step(AT, {SYMBOL: gap, other: candles(AT, 110.)})
        exit_fill = sim.fills[0]
        self.assertEqual((exit_fill["at"], exit_fill["price"], exit_fill["cash_usd"]), (at, 80., 1180.))
        self.assertEqual(sim._baselines, {"day": ("2024-07-01", 1180.), "week": ("2024-W27", 1180.)})
        settlement, = sim.funding
        self.assertEqual((settlement["at"], settlement["symbol"], settlement["mark"]), (AT, other, 110.))
        self.assertAlmostEqual(settlement["cash_usd"], 1180. - 4 * .1 * 110. * .001)
        self.assertTrue(sim.intents[-2]["can_open"])
        _, new_fill = sim.fills
        self.assertEqual((new_fill["at"], new_fill["signed_fill"], new_fill["symbol"]), (AT, 1, SYMBOL))
        self.assertEqual(new_fill["cash_usd"], settlement["cash_usd"])

    def test_blocked_paper_and_replay_gate_net_increases_but_allow_risk_reduction(self):
        for current, target, stop, filled, execution in (
            (0, 4, "95", 0, "entries_blocked"),
            (4, 8, "95", 4, "entries_blocked"),
            (4, 2, "95", 2, "filled"),
            (4, -2, "110", 0, "flip_closed_entries_blocked"),
        ):
            with self.subTest(current=current, target=target):
                live, sim = self.adapters()
                if current:
                    self.seed((live, sim), {}, Position(current, 100., "95"), AT - PERIOD)
                intent = engine.NetIntent({}, target, stop)
                with (patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()),
                      patch.object(delta, "decide_four_pod", return_value=intent),
                      patch("research.replay_four_pod.decide_four_pod", return_value=intent)):
                    self.both((live, sim), AT, candles(AT), blocked=True)
                self.assertEqual((live.state["positions"].get(SYMBOL, {}).get("contracts", 0),
                                  sim.intents[-1]["filled_target"]), (filled, filled))
                self.assertEqual(sim.intents[-1]["execution"], execution)
                self.assertFalse(sim.intents[-1]["can_open"])

    def test_paper_and_replay_recheck_loss_limit_between_flip_legs(self):
        live, sim = self.adapters(daily_loss_limit_pct=.01)
        sim.scenario = replace(sim.scenario, daily_limit=.01)
        self.seed((live, sim), {}, Position(6, 125., "95"), AT - PERIOD)
        intent = engine.NetIntent({}, -2, "110")
        with (patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()),
              patch.object(delta, "decide_four_pod", return_value=intent),
              patch("research.replay_four_pod.decide_four_pod", return_value=intent)):
            self.both((live, sim), AT, candles(AT))
        self.assertTrue(sim.intents[-1]["can_open"])  # True at decision time, false after realizing the loss.
        self.assertEqual((live.state["positions"], sim.positions), ({}, {}))
        self.assertEqual(sim.intents[-1]["execution"], "flip_closed_entries_blocked")
        self.assertEqual([f["reason"] for f in sim.fills], ["flip_close"])

    def test_default_decision_rules_match_live(self):
        live, _ = self.adapters()
        self.assertEqual(engine.DecisionRules(), engine.DecisionRules.from_live_config(live.cfg))
        costs = Scenario()
        self.assertEqual((live.cfg.dd_brake_threshold, live.cfg.dd_brake_scale,
                          live.cfg.daily_loss_limit_pct, live.cfg.weekly_loss_limit_pct),
                         (costs.dd_threshold, costs.dd_scale, costs.daily_limit, costs.weekly_limit))

    def test_decision_does_not_mutate_existing_books(self):
        frame = candles(AT)
        with patch.object(engine, "add_3pod_features", side_effect=indicator_fixture()):
            bar = engine.prepare_bar(frame, AT, PERIOD, 20).bar
        market = engine.Market(MARK, .1, Decimal("0.1"))
        old = {SYMBOL: {"SUPERTREND": dict(side="SHORT", contracts=5, entry_price=110., sl_price=120.)}}
        snapshot = deepcopy(old)
        args = (SYMBOL, bar, market, engine.DecisionContext(1200., 1., True, 1), engine.DecisionRules(), old, {})
        self.assertIs(engine.decide_four_pod(*args), engine.FundingRequest.NEEDED)
        self.assertEqual(old, snapshot)
        intent = engine.decide_four_pod(*args, funding=None)
        self.assertNotIn("SUPERTREND", intent.book)  # flip exits, not a same-bar reverse entry
        self.assertEqual(old, snapshot)
