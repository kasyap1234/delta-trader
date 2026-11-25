"""Exercise Four-Pod writes and restart recovery via a deterministic Delta REST fake."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import requests

import four_pod_engine as decision
import live_delta_donchian as delta
from config import DeltaLiveConfig
from tests.test_four_pod_parity import AT, PERIOD, SYMBOL, candles, indicator_fixture
from tests.test_preflight import DeltaFixture


class NetBroker(DeltaFixture):
    def __init__(self, state_file):
        super().__init__()
        self.mark = 100.03
        self.trade_price = Decimal("100.00")  # A tick-aligned trade price, not the mark.
        self.state_file = Path(state_file)
        self.at_submit = []
        self.fill = None
        self.net_timeout = False
        self.keep_old_stop = []
        self.position_offset = 0
        self.ioc_report_overrides = {}
        self.at_edit = []

    def fill_stop(self):
        self.orders["8"] = dict(self.stop, state="closed", unfilled_size=0, average_fill_price=self.stop["stop_price"],
                                paid_commission="0.25", updated_at="2024-07-01T00:00:01Z")
        self.stop = None
        self.position = 0

    def request(self, method, url, *, data, **kwargs):
        path = urlsplit(url).path
        if path != "/v2/orders" or method == "GET":
            return super().request(method, url, data=data, **kwargs)
        payload = json.loads(data)
        cid = payload.get("client_order_id")
        if method == "POST":
            self.at_submit.append(json.loads(self.state_file.read_text()) if self.state_file.exists() else {})
        if method == "PUT":
            self.at_edit.append(json.loads(self.state_file.read_text()))
            if self.phase == "timeout-before-edit":
                self.writes.append((method, payload))
                raise requests.Timeout("Edit did not arrive")
            if self.phase == "stop-fills-during-edit":
                self.writes.append((method, payload))
                self.stop["stop_price"] = payload["stop_price"]
                self.fill_stop()
                return self.response(200, {})
        if method == "DELETE" and self.stop and payload["id"] == self.stop["id"]:
            self.writes.append((method, payload))
            self.stop = None
            return self.response(200, {})
        if method != "POST" or "bracket_stop_loss_price" in payload:
            return super().request(method, url, data=data, **kwargs)
        self.writes.append((method, payload))
        self.keep_old_stop.append(deepcopy(self.stop))
        if self.phase == "missing-ioc":
            raise requests.Timeout("Request disposition unknown")
        if payload["order_type"] != "limit_order" or payload["time_in_force"] != "ioc":
            raise AssertionError("unexpected non-IOC net transaction")
        quantity = payload["size"]
        filled = quantity if self.fill is None else self.fill
        sign = 1 if payload["side"] == "buy" else -1
        if payload.get("reduce_only") and (sign * self.position >= 0 or filled > abs(self.position)):
            raise AssertionError("IOC is not reducing")
        self.position += sign * filled + self.position_offset
        limit = Decimal(payload["limit_price"])
        price = min(self.trade_price, limit) if sign > 0 else max(self.trade_price, limit)
        order = dict(state="closed" if filled == quantity else "cancelled", product_id=1,
                     client_order_id=cid, side=payload["side"], size=quantity, unfilled_size=quantity - filled,
                     average_fill_price=str(price), paid_commission="0.20", updated_at="2024-07-01T00:00:01Z",
                     limit_price=payload["limit_price"], order_type=payload["order_type"],
                     time_in_force=payload["time_in_force"], reduce_only=payload.get("reduce_only", False))
        order.update(self.ioc_report_overrides)
        self.orders[cid] = order
        if self.net_timeout:
            raise requests.Timeout("Response lost after IOC")
        return self.response(200, order)


class FourPodExecutionTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.cfg = DeltaLiveConfig(env="testnet", dry_run=False, strategy_mode="four_pod_balanced", resolution="6h",
                                   symbols=[SYMBOL], capital_usd=1200., api_key="", api_secret="", data_dir=directory,
                                   telegram_token="", telegram_chat_id="", use_maker_entry=False)
        self.broker = NetBroker(self.cfg.state_file)
        self.enterContext(patch.object(delta.requests, "request", side_effect=self.broker.request))
        self.engine = self.restart()
        self.frame = candles(AT)
        self.enterContext(patch.object(delta.pd.Timestamp, "now", return_value=AT))
        self.enterContext(patch.object(delta, "load_delta_candles", return_value=self.frame))
        self.enterContext(patch.object(delta, "load_delta_funding_avg", return_value=None))
        self.enterContext(patch.object(decision, "add_3pod_features", side_effect=indicator_fixture()))

    def restart(self):
        engine = delta.LiveDonchianEngine(self.cfg)
        engine.client.products_cache[SYMBOL] = dict(id=1, contract_value=.1, tick_size=Decimal("0.05"), fee_rate=0.)
        return engine

    def seed(self, signed=6, entry_price=99.):
        self.broker.position = signed
        quantity = abs(signed)
        self.broker.stop = dict(id=8, state="pending", size=quantity, unfilled_size=quantity,
                                side="sell" if signed > 0 else "buy", stop_order_type="stop_loss_order",
                                order_type="market_order", reduce_only=True,
                                stop_price="98" if signed > 0 else "102", stop_trigger_method="mark_price")
        self.engine.state["positions"][SYMBOL] = dict(side="LONG" if signed > 0 else "SHORT", contracts=quantity,
            contract_value=.1, entry_price=entry_price, entry_fee=.3, sl_price=self.broker.stop["stop_price"], sl_order_id=8,
            risk_pct=.02, client_order_id="previous")
        self.engine._save_state()

    def sync(self, target, stop):
        price = self.broker.mark
        self.engine._sync_net_position(SYMBOL, self.engine.client.products_cache[SYMBOL], target, stop,
                                       price, price, price, AT - PERIOD, PERIOD)

    def submit_legacy_add(self):
        """Seed an IOC sent by an older build, bypassing the current adapter gate."""
        pos = self.engine.state["positions"][SYMBOL]
        pos["pending_net"] = dict(client_order_id="legacy-add", signed_delta=4, limit_price="100.15", stop_price="97")
        self.engine._save_state()
        self.engine.client.submit_net_order(1, "buy", 4, "100.15", "legacy-add", reduce_only=False)

    def reconcile(self, engine):
        engine._reconcile(SYMBOL, engine.client.products_cache[SYMBOL], wait_for_bracket=False)

    def check_protected(self, engine, signed, stop):
        pos = engine.state["positions"][SYMBOL]
        self.assertEqual(self.broker.position, signed)
        self.assertEqual((pos["contracts"], self.broker.stop["unfilled_size"]), (abs(signed), abs(signed)))
        self.assertEqual((pos["sl_price"], self.broker.stop["stop_price"]), (stop, stop))
        self.assertNotIn("pending_net", pos)
        self.assertNotIn("pending_trail", pos)

    def test_accepted_resize_can_loosen_old_stop(self):
        self.seed()
        self.sync(2, "90")
        self.check_protected(self.restart(), 2, "90")
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "PUT"])

    def test_resize_timeout_partial_fill_recovers_only_actual_remainder(self):
        self.seed()
        self.broker.fill, self.broker.net_timeout = 1, True
        with self.assertRaises(requests.Timeout):
            self.sync(2, "90")
        restarted = self.restart()
        self.reconcile(restarted)
        self.check_protected(self.restart(), 5, "90")
        self.reconcile(self.restart())
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "PUT"])
        order = self.broker.writes[0][1]
        self.assertTrue(order["reduce_only"])
        self.assertEqual(self.broker.at_submit[0]["positions"][SYMBOL]["pending_net"]["client_order_id"],
                         order["client_order_id"])
        state = self.restart().state
        self.assertAlmostEqual(state["equity_usd"], 1199.85)
        self.assertAlmostEqual(state["daily_pnl_usd"], -.15)
        self.assertAlmostEqual(state["weekly_pnl_usd"], -.15)
        self.assertAlmostEqual(state["positions"][SYMBOL]["entry_fee"], .25)

    def test_increase_timeout_partial_fill_reconciles_weighted_entry_and_stop(self):
        self.seed()
        self.broker.fill, self.broker.net_timeout = 2, True
        with self.assertRaises(requests.Timeout):
            self.submit_legacy_add()
        # The older build's resize was accepted; restart confirms it without another write.
        self.broker.stop.update(size=8, unfilled_size=8, stop_price="97")
        self.reconcile(self.restart())
        restarted = self.restart()
        self.check_protected(restarted, 8, "97")
        pos = restarted.state["positions"][SYMBOL]
        self.assertAlmostEqual(pos["entry_price"], (99. * 6 + 100. * 2) / 8)
        self.assertAlmostEqual(pos["entry_fee"], .5)
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertNotIn("reduce_only", self.broker.writes[0][1])
        self.assertEqual(self.broker.keep_old_stop[0]["size"], 6)

    def test_zero_fill_does_not_change_stop_or_retry_ioc(self):
        self.seed()
        self.broker.fill = 0
        self.sync(2, "90")
        self.reconcile(self.restart())
        self.check_protected(self.restart(), 6, "98")
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])

    def test_close_timeout_partial_fill_keeps_stop_and_never_flips(self):
        self.seed(signed=-6)
        self.broker.fill, self.broker.net_timeout = 4, True
        with self.assertRaises(requests.Timeout):
            self.sync(2, "90")
        self.reconcile(self.restart())
        self.check_protected(self.restart(), -2, "102")
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "PUT"])
        self.assertEqual(self.broker.keep_old_stop[0]["id"], 8)
        self.assertTrue(self.broker.writes[0][1]["reduce_only"])

    def test_close_timeout_full_fill_cleans_only_own_stop_after_flat(self):
        self.seed()
        self.broker.net_timeout = True
        with self.assertRaises(requests.Timeout):
            self.sync(0, None)
        self.reconcile(self.restart())
        self.reconcile(self.restart())
        self.assertEqual(self.restart().state["positions"], {})
        self.assertIsNone(self.broker.stop)
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "DELETE"])
        self.assertEqual(len(Path(self.cfg.journal_file).read_text().splitlines()), 2)

    def test_missing_ioc_stays_unresolved_even_if_original_position_unchanged(self):
        self.seed()
        self.broker.phase = "missing-ioc"
        with self.assertRaises(requests.Timeout):
            self.sync(0, None)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                self.reconcile(self.restart())
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertEqual(self.broker.stop["size"], 6)
        self.engine = self.restart()
        with self.assertRaisesRegex(RuntimeError, "pending net IOC"):
            self.sync(0, None)
        self.assertIn("pending_net", self.engine.state["positions"][SYMBOL])
        self.assertEqual(len(self.broker.writes), 1)

    def test_position_mismatch_does_not_edit_stop_or_submit_again(self):
        self.seed()
        self.broker.position_offset = -1
        with self.assertRaises(RuntimeError):
            self.sync(2, "90")
        with self.assertRaises(RuntimeError):
            self.reconcile(self.restart())
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertIn("pending_net", self.restart().state["positions"][SYMBOL])

    def test_accepted_stop_edit_timeout_recovers_without_reediting(self):
        self.seed()
        self.broker.phase = "timeout-edit"
        with self.assertRaises(requests.Timeout):
            self.sync(2, "90")
        self.reconcile(self.restart())
        self.check_protected(self.restart(), 2, "90")
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "PUT"])

    def test_removed_symbol_is_reconciled_but_never_sent_to_strategy(self):
        self.seed()
        self.cfg.symbols = ["SOLUSD"]
        engine = self.restart()
        engine.client.products_cache.clear()  # Fresh process loads metadata from the broker.
        with patch.object(engine, "_process_3pod_symbol", return_value=False) as strategy:
            with self.assertRaisesRegex(RuntimeError, "outside configured basket"):
                engine.run_cycle()
            self.assertTrue(strategy.call_args.args[1])  # Existing SOL positions may be managed, but no entries.
            self.broker.fill_stop()
            engine.run_cycle()
            self.assertEqual([call.args[0] for call in strategy.call_args_list], ["SOLUSD", "SOLUSD"])
        self.assertEqual(self.restart().state["positions"], {})
        self.assertEqual(self.broker.writes, [])

    def test_removed_symbol_without_product_metadata_blocks_resume(self):
        self.cfg.symbols = ["SOLUSD"]
        self.engine.state["positions"]["DELISTED"] = {"contracts": 6}
        self.engine._save_state()
        engine = self.restart()
        engine.client.products_cache.clear()
        with self.assertLogs(delta.log, "ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "Resume requires successful reconciliation"):
                engine.run_cycle(resume=True)
        self.assertIn("DELISTED: product metadata unavailable", " ".join(logs.output))
        self.assertIn("DELISTED", self.restart().state["positions"])
        self.assertEqual(self.broker.writes, [])

    def test_cycle_defers_unprotected_same_side_increase_before_and_after_restart(self):
        for signed, target, stop in ((6, 10, "97"), (-6, -10, "103")):
            with self.subTest(signed=signed):
                self.seed(signed=signed)
                self.broker.phase = "reject-edit"
                for _ in range(2):
                    with patch.object(delta, "decide_four_pod", return_value=decision.NetIntent({}, target, stop)):
                        self.engine.run_cycle()
                    self.engine = self.restart()
                    self.check_protected(self.engine, signed, "98" if signed > 0 else "102")
                self.assertEqual(self.broker.writes, [])

    def test_legacy_add_with_short_stop_coverage_halts_without_writes(self):
        self.seed()
        self.submit_legacy_add()
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "resize unconfirmed"):
                self.restart().run_cycle()
        self.assertEqual(self.broker.position, 10)
        self.assertEqual(self.broker.stop["unfilled_size"], 6)
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertTrue(self.restart().state["halt_new_entries"])

    def test_crash_after_persisting_resize_attempt_does_not_send_on_restart(self):
        self.seed()
        # SystemExit bypasses ordinary reconciliation error handling, as a lost process would.
        with patch.object(self.engine.client, "edit_stop", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                self.sync(2, "90")
        pending = self.restart().state["positions"][SYMBOL]["pending_net"]
        self.assertIs(pending["stop_edit_attempted"], True)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "resize unconfirmed"):
                self.restart().run_cycle()
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertEqual(self.broker.position, 2)
        self.assertEqual(self.broker.stop["unfilled_size"], 6)
        self.assertTrue(self.restart().state["halt_new_entries"])

    def test_rejected_resize_is_not_repeated_on_restart(self):
        self.seed()
        self.broker.phase = "reject-edit"
        with self.assertRaises(delta.DeltaAPIError):
            self.sync(2, "90")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "resize unconfirmed"):
                self.restart().run_cycle()
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "PUT"])
        self.assertTrue(self.restart().state["halt_new_entries"])

    def test_halt_blocks_same_side_increase_but_allows_reduction(self):
        self.seed()
        self.engine.state["halt_new_entries"] = True
        self.sync(9, "98")
        self.assertEqual(self.broker.writes, [])
        self.sync(2, "98")
        self.check_protected(self.restart(), 2, "98")
        self.assertTrue(self.broker.writes[0][1]["reduce_only"])

    def test_blockers_at_adapter_boundary_allow_flip_close_only(self):
        self.seed()
        with patch.object(delta, "decide_four_pod", return_value=decision.NetIntent({}, -4, "110")):
            self.engine._process_3pod_symbol(SYMBOL, ["wallet unavailable"], available=None)
        writes = [payload for method, payload in self.broker.writes if method == "POST"]
        self.assertEqual(len(writes), 1)
        self.assertTrue(writes[0]["reduce_only"])
        self.assertEqual(self.broker.position, 0)

    def test_flip_rechecks_loss_limit_after_confirmed_close(self):
        self.seed(entry_price=125.)
        self.cfg.daily_loss_limit_pct = .01
        self.sync(-4, "110")
        self.assertEqual(self.broker.position, 0)
        self.assertEqual([method for method, _ in self.broker.writes], ["POST", "DELETE"])
        order, = self.broker.orders.values()
        self.assertGreaterEqual(Decimal(order["average_fill_price"]), Decimal(order["limit_price"]))
        self.assertTrue(self.engine._circuit_breaker_tripped())

    def test_net_ioc_outside_limit_stays_unresolved_on_either_side(self):
        for target, reported_price in ((2, "99.899999999999999999"), (10, "100.150000000000000001")):
            with self.subTest(target=target):
                self.seed()
                self.broker.writes.clear()
                self.broker.ioc_report_overrides = {"average_fill_price": reported_price}
                with self.assertRaisesRegex(RuntimeError, "outside persisted limit"):
                    if target > 6:
                        self.submit_legacy_add()
                        self.reconcile(self.restart())
                    else:
                        self.sync(target, "97")
                restarted = self.restart()
                with self.assertRaisesRegex(RuntimeError, "outside persisted limit"):
                    self.reconcile(restarted)
                self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
                self.assertIn("pending_net", restarted.state["positions"][SYMBOL])
                self.assertEqual(restarted.state["positions"][SYMBOL]["contracts"], 6)
                self.assertEqual(restarted.state["equity_usd"], 1200.)

    def test_net_ioc_lookup_checks_reported_identity_and_terms(self):
        for field, value in (("client_order_id", "wrong"), ("product_id", 2), ("side", "buy"),
                             ("limit_price", "99.95"), ("order_type", "market_order"),
                             ("time_in_force", "gtc"), ("reduce_only", False)):
            with self.subTest(field=field):
                self.seed()
                self.broker.writes.clear()
                self.broker.ioc_report_overrides = {field: value}
                with self.assertRaisesRegex(RuntimeError, "net IOC not reconciled"):
                    self.sync(2, "97")
                self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
                self.assertIn("pending_net", self.restart().state["positions"][SYMBOL])

    def test_missing_stop_after_net_fill_stays_unresolved_and_does_not_submit(self):
        self.seed()
        self.broker.net_timeout = True
        with self.assertRaises(requests.Timeout):
            self.submit_legacy_add()
        self.broker.stop = None
        with self.assertRaisesRegex(RuntimeError, "protective stop missing after net IOC"):
            self.reconcile(self.restart())
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertEqual(self.broker.position, 10)  # Exposure may be completely unprotected.
        self.assertIn("pending_net", self.restart().state["positions"][SYMBOL])

    def test_trail_restart_confirms_or_retries_without_redeciding(self):
        book = {"DONCHIAN": {"contracts": 6}}
        for phase, error, retries in (("timeout-edit", requests.Timeout, 0),
                                      ("reject-edit", delta.DeltaAPIError, 1),
                                      ("timeout-before-edit", requests.Timeout, 1)):
            with self.subTest(phase=phase):
                self.seed()
                self.engine.state["last_entry_bar"].clear()
                self.broker.at_edit.clear()
                self.broker.writes.clear()
                self.broker.phase = phase
                with patch.object(delta, "decide_four_pod", return_value=decision.NetIntent(book, 6, "99.50")):
                    with self.assertRaises(error):
                        self.engine.run_cycle()
                restarted = self.restart()
                self.broker.phase = None
                with patch.object(delta, "decide_four_pod") as decide:
                    restarted.run_cycle()
                    restarted.run_cycle()
                decide.assert_not_called()
                self.check_protected(self.restart(), 6, "99.50")
                self.assertEqual([method for method, _ in self.broker.writes], ["PUT"] * (1 + retries))
                saved = self.broker.at_edit[0]
                self.assertEqual(saved["last_entry_bar"][SYMBOL], (AT - PERIOD).isoformat())
                self.assertEqual(saved["pods"][SYMBOL], book)
                self.assertEqual(saved["positions"][SYMBOL]["sl_price"], "98")
                self.assertEqual(saved["positions"][SYMBOL].get("pending_trail"), {"stop_price": "99.50"})

    def test_pending_trail_uses_stop_fill_reconciliation_instead_of_retrying(self):
        self.seed()
        self.broker.phase = "timeout-edit"
        with patch.object(delta, "decide_four_pod", return_value=decision.NetIntent({}, 6, "99.50")):
            with self.assertRaises(requests.Timeout):
                self.engine.run_cycle()
        self.assertIn("pending_trail", self.restart().state["positions"][SYMBOL])
        self.broker.fill_stop()
        with patch.object(delta, "decide_four_pod") as decide:
            self.restart().run_cycle()
            self.restart().run_cycle()
        decide.assert_not_called()
        self.assertEqual(self.restart().state["positions"], {})
        self.assertEqual([method for method, _ in self.broker.writes], ["PUT"])
        row, = Path(self.cfg.journal_file).read_text().splitlines()[1:]
        self.assertIn(",99.5,", row)

    def test_stop_fill_during_trail_confirmation_is_booked_without_second_put(self):
        self.seed()
        self.broker.phase = "stop-fills-during-edit"
        with patch.object(delta, "decide_four_pod", return_value=decision.NetIntent({}, 6, "99.50")):
            self.engine.run_cycle()
        self.assertEqual(self.restart().state["positions"], {})
        self.assertEqual([method for method, _ in self.broker.writes], ["PUT"])
        self.assertEqual(len(Path(self.cfg.journal_file).read_text().splitlines()), 2)

    def test_entry_crash_after_fill_never_reuses_signal_candle_low(self):
        self.frame.loc[AT - PERIOD, "low"] = 80.  # older than the bracket being entered now
        self.broker.phase = "timeout-entry"
        with self.assertRaises(requests.Timeout):
            self.engine.run_cycle()
        restarted = self.restart()
        self.broker.phase = None
        with patch.object(delta, "decide_four_pod", wraps=decision.decide_four_pod) as decide:
            restarted.run_cycle()
        decide.assert_not_called()
        self.assertEqual([method for method, _ in self.broker.writes], ["POST"])
        self.assertEqual(restarted.state["last_entry_bar"][SYMBOL], (AT - PERIOD).isoformat())
        self.assertEqual(self.broker.at_submit[0]["last_entry_bar"], restarted.state["last_entry_bar"])
        self.assertGreater(self.broker.position, 0)
