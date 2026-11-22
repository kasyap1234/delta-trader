import csv
import gzip
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import live_delta_donchian as delta
import live_kotak_mcx_advisor as kotak
import live_nifty_spread_advisor as nifty
import validate_real_nifty_spreads as vns
from config import DeltaLiveConfig, KotakMcxConfig, PortfolioConfig
from data_loader import load_delta_candles, resolve_date_window
from portfolio_engine import run_shared_donchian_portfolio


if not hasattr(unittest.TestCase, "enterContext"):
    def _enterContext(self, cm):
        res = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return res
    unittest.TestCase.enterContext = _enterContext


def bars(index, start, slope):
    prices = pd.Series([start + slope * i for i in range(len(index))], index=index)
    return pd.DataFrame(dict(open=prices, high=prices + .5, low=prices - .5, close=prices, volume=10))


class DeltaFixture:
    def __init__(self):
        self.phase = None
        self.mark = 2240.07
        self.position = 0
        self.orders = {}
        self.stop = None
        self.writes = []
        self.margin_mode = "isolated"
        self.default_auto_topup = False
        self.deto_for_commission = False
        self.leverage_by_product = {}
        self.updated_at_iso = False
        self.wallet_equity = 1200.0
        self.next_order_id = 100

    def request(self, method, url, *, data, **kwargs):
        path = urlsplit(url).path
        order = json.loads(data) if data else {}
        status = 200
        if method != "GET":
            self.writes.append((method, order))
        if path == "/v2/products":
            result = [dict(symbol=s, id=i, contract_value=v, tick_size="0.05", notional_type="vanilla",
                           quoting_asset={"symbol": "USD"}, settling_asset={"symbol": "USD"},
                           maintenance_margin="0.5", maintenance_margin_scaling_factor="0.000002",
                           max_leverage_notional=("0" if s in {"ETHUSD", "DOGEUSD"} else "100000"),
                           taker_commission_rate="0.0005", maker_commission_rate="0.0002")
                      for s, i, v in [("ETHUSD", 1, "0.01"), ("SOLUSD", 2, "1"), ("BTCUSD", 3, "0.001"),
                                      ("BNBUSD", 4, "0.1"), ("DOGEUSD", 5, "100")]]
        elif path == "/v2/profile":
            result = dict(id=101, margin_mode=self.margin_mode)
        elif path.startswith("/v2/tickers/"):
            result = dict(mark_price=self.mark, quotes=dict(best_bid=str(self.mark - 0.05), best_ask=str(self.mark + 0.05)))
        elif path == "/v2/positions":
            result = dict(size=self.position if url.endswith("=1") else 0)
        elif path == "/v2/users/trading_preferences" and method == "GET":
            result = dict(user_id=101, default_auto_topup=self.default_auto_topup,
                          deto_for_commission=self.deto_for_commission)
        elif path == "/v2/users/trading_preferences" and method == "PUT":
            self.default_auto_topup = order["default_auto_topup"]
            self.deto_for_commission = order["deto_for_commission"]
            result = dict(default_auto_topup=self.default_auto_topup, deto_for_commission=self.deto_for_commission)
        elif path == "/v2/users/margin_mode" and method == "PUT":
            self.margin_mode = order["margin_mode"]
            result = dict(margin_mode=self.margin_mode)
        elif path == "/v2/sub_accounts":
            result = [dict(id="101", margin_mode=self.margin_mode)]
        elif path.endswith("/orders/leverage") and method == "GET":
            pid = int(path.split("/")[3])
            result = dict(leverage=self.leverage_by_product.get(pid, 5))
        elif path.endswith("/orders/leverage") and method == "POST":
            pid = int(path.split("/")[3])
            self.leverage_by_product[pid] = int(order["leverage"])
            result = dict(leverage=str(self.leverage_by_product[pid]))
        elif path == "/v2/wallet/balances":
            eq = str(self.wallet_equity)
            return SimpleNamespace(
                ok=True,
                status_code=200,
                json=lambda: dict(
                    success=True,
                    meta=dict(net_equity="41871.35"),
                    result=[
                        dict(asset_symbol="BTC", balance="0.25", available_balance="0.25"),
                        dict(asset_symbol="USD", balance=eq, available_balance=eq),
                    ],
                ),
            )
        elif path.startswith("/v2/orders/"):
            result = self.orders.get(path.rsplit("/", 1)[-1])
            status = 200 if result else 404
        elif path == "/v2/orders" and method == "GET":
            result = [self.stop] if self.stop and "product_ids=1" in url else []
        elif path == "/v2/orders" and method == "DELETE":
            oid = order["id"]
            for rec in self.orders.values():
                if rec.get("id") == oid and rec.get("state") == "open":
                    rec["state"] = "cancelled"
            result = dict(id=oid, state="cancelled")
        elif path == "/v2/orders" and method == "POST":
            cid = order["client_order_id"]
            size = order["size"]
            if self.phase == "reject-entry":
                return self.response(400, None, error=dict(code="server_rejected_unknown"))
            if self.phase == "reject-insufficient-margin":
                return self.response(400, None, error=dict(code="insufficient_margin", context={"required": "500"}))
            if order.get("reduce_only"):
                assert self.position == size and order["side"] == "sell"
                self.position = 0
                self.orders[cid] = dict(state="closed", size=size, unfilled_size=0, average_fill_price="2240",
                                        paid_commission="0.3", updated_at=int(time.time() * 1e6))
                return self.response(200, {})
            if order.get("post_only"):
                assert order["order_type"] == "limit_order" and order["time_in_force"] == "gtc"
                if self.phase == "maker-cross":
                    return self.response(400, None, error=dict(code="post_only_would_cross"))
                oid = self.next_order_id
                self.next_order_id += 1
                if self.phase == "maker-unfilled":
                    self.orders[cid] = dict(id=oid, state="open", product_id=1, side=order["side"],
                                            size=size, unfilled_size=size, average_fill_price="0", paid_commission="0")
                    return self.response(200, self.orders[cid])
                self.position = size
                self.orders[cid] = dict(id=oid, state="closed", product_id=1, side=order["side"],
                                        size=size, unfilled_size=0, average_fill_price=order["limit_price"],
                                        paid_commission="0.16")
                self.stop = dict(id=8, state="pending", size=size, unfilled_size=size, side="sell",
                                 stop_order_type="stop_loss_order", order_type="market_order", reduce_only=True,
                                 stop_price=order["bracket_stop_loss_price"], stop_trigger_method="mark_price")
                return self.response(200, self.orders[cid])
            if "bracket_stop_loss_price" not in order:
                # Winner pyramid IOC add order (no duplicate bracket stop).
                assert order["order_type"] == "limit_order" and order["time_in_force"] == "ioc"
                self.position += size
                self.orders[cid] = dict(state="closed", product_id=1, side=order["side"],
                                        size=size, unfilled_size=0, average_fill_price=str(self.mark),
                                        paid_commission="0.20")
                if self.phase == "timeout-pyramid":
                    raise requests.Timeout("Response lost after pyramid fill")
                return self.response(200, self.orders[cid])
            assert order["order_type"] == "limit_order" and order["time_in_force"] == "ioc"
            assert Decimal(order["limit_price"]) % Decimal("0.05") == 0
            assert order["bracket_stop_trigger_method"] == "mark_price"
            assert Decimal(order["bracket_stop_loss_price"]) % Decimal("0.05") == 0
            filled = size // 2 if self.phase == "timeout-entry" else size
            self.position = filled
            self.orders[cid] = dict(state="closed" if filled == size else "cancelled", product_id=1, side="buy",
                                    size=size, unfilled_size=size-filled, average_fill_price=str(self.mark + .23),
                                    paid_commission="0.40")
            if self.phase != "no-stop":
                self.stop = dict(id=8, state="pending", size=filled, unfilled_size=filled, side="sell",
                                 stop_order_type="stop_loss_order", order_type="market_order", reduce_only=True,
                                 stop_price=order["bracket_stop_loss_price"], stop_trigger_method="mark_price")
            if self.phase == "timeout-entry":
                raise requests.Timeout("Response lost after partial fill")
            result = self.orders[cid]
        elif path == "/v2/orders" and method == "PUT":
            assert "size" in order and isinstance(order["size"], int) and order["size"] > 0
            if self.phase == "reject-edit":
                return self.response(400, None)
            self.stop.update(stop_price=order["stop_price"], size=order["size"], unfilled_size=order["size"])
            if self.phase == "timeout-edit":
                raise requests.Timeout("Response lost after edit")
            result = self.stop
        else:
            raise AssertionError((method, url))
        return self.response(status, result)

    def close_at_stop(self):
        ts = "2026-11-22T12:00:00.123456Z" if self.updated_at_iso else int(time.time() * 1e6)
        self.orders["8"] = dict(self.stop, state="closed", unfilled_size=0, average_fill_price="2244.9",
                                paid_commission="0.60", updated_at=ts)
        self.stop = None
        self.position = 0

    @staticmethod
    def response(status, result, error=None):
        payload = dict(success=status == 200, result=result)
        if error is not None:
            payload["error"] = error
        return SimpleNamespace(ok=status == 200, status_code=status, json=lambda: payload)


class DeltaSafetyTest(unittest.TestCase):
    def setUp(self):
        location = self.enterContext(tempfile.TemporaryDirectory())
        self.cfg = DeltaLiveConfig(
            env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
            resolution="4h", risk_per_trade_pct=0.01, max_concurrent_risk_pct=0.03,
            max_crypto_concurrent=2, atr_trail_mult=2.5, max_margin_utilization=0.60,
            use_maker_entry=False, pyramid_add_fraction=0.0, dd_brake_threshold=0.0,
            daily_loss_limit_pct=0.03, weekly_loss_limit_pct=0.07,
            telegram_token="", telegram_chat_id="", state_file=f"{location}/bot.json",
            journal_file=f"{location}/journal.csv",
        )
        self.exchange = DeltaFixture()
        self.end = pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=4)
        self.start = 2000
        self.length = 241
        self.fail_sol = False
        self.enterContext(patch.object(delta.requests, "request", self.exchange.request))
        self.enterContext(patch.object(delta, "load_delta_candles", self.candles))

    def candles(self, symbol, **kwargs):
        if symbol == "SOLUSD" and self.fail_sol:
            raise requests.Timeout("SOL candles unavailable")
        return bars(pd.date_range(end=self.end, freq="4h", periods=self.length), self.start, 1)

    def cycle(self):
        delta.LiveDonchianEngine(self.cfg).run_cycle()

    def test_rejected_entry_cannot_resubmit_on_restart(self):
        self.exchange.phase = "reject-entry"
        with self.assertRaises(RuntimeError):
            self.cycle()
        with self.assertRaises(RuntimeError):
            self.cycle()
        self.assertEqual(len(self.exchange.writes), 1)
        self.assertEqual(self.exchange.position, 0)

    def test_accepted_entry_timeout_reconciles_partial_fill_no_duplicate(self):
        self.exchange.phase = "timeout-entry"
        with self.assertRaises(requests.Timeout):
            self.cycle()
        sent = self.exchange.writes[0][1]["size"]
        self.assertEqual(sent, 91)
        self.cycle()
        self.assertEqual(len(self.exchange.writes), 1)
        self.assertEqual(self.exchange.position, 45)
        self.assertEqual(self.exchange.stop["size"], 45)

    def test_unprotected_verified_position_is_closed_reduce_only_and_halted(self):
        self.exchange.phase = "no-stop"
        with self.assertRaisesRegex(RuntimeError, "emergency close submitted"):
            self.cycle()
        self.assertEqual(self.exchange.position, 0)
        self.assertTrue(self.exchange.writes[1][1]["reduce_only"])
        self.end += pd.Timedelta(hours=4)
        with self.assertLogs(delta.log, level="INFO"):
            self.cycle()
        self.assertEqual(len(self.exchange.writes), 2)

    def test_stop_edit_rejection_and_timeout_preserve_protection_and_use_real_fill(self):
        self.cycle()
        original = self.exchange.stop["stop_price"]
        self.start += 10
        self.exchange.mark = 2250.07
        self.exchange.phase = "reject-edit"
        with self.assertRaises(RuntimeError):
            self.cycle()
        self.assertEqual(self.exchange.stop["stop_price"], original)
        self.exchange.phase = "timeout-edit"
        with self.assertRaises(requests.Timeout):
            self.cycle()
        self.assertGreater(float(self.exchange.stop["stop_price"]), float(original))
        self.cycle()
        self.exchange.phase = None
        self.start += 1
        self.cycle()
        self.exchange.close_at_stop()
        self.cycle()
        self.assertEqual([method for method, _ in self.exchange.writes], ["POST", "PUT", "PUT", "PUT"])
        with open(self.cfg.journal_file) as f:
            row, = csv.DictReader(f)
        self.assertEqual(float(row["exit_price"]), 2244.9)
        self.assertEqual(float(row["pnl_usd"]), 3.19)  # (2244.9 - 2240.3) × 0.91 − 1.00

    def test_iso_timestamp_rejected_margin_cleanup_and_isolated_check(self):
        self.exchange.margin_mode = "cross"
        with self.assertRaisesRegex(RuntimeError, "isolated margin"):
            self.cycle()
        self.exchange.margin_mode = "isolated"
        self.exchange.phase = "reject-insufficient-margin"
        with self.assertRaisesRegex(delta.DeltaAPIError, "insufficient_margin"):
            self.cycle()
        self.assertEqual(delta.LiveDonchianEngine(self.cfg).state["positions"], {})
        self.exchange.phase = None
        self.exchange.updated_at_iso = True
        self.end += pd.Timedelta(hours=4)
        fake_now = self.end + pd.Timedelta(hours=4, seconds=15)
        with patch.object(delta.pd.Timestamp, "now", return_value=fake_now):
            delta.LiveDonchianEngine(self.cfg).run_cycle()
            self.exchange.close_at_stop()
            delta.LiveDonchianEngine(self.cfg).run_cycle()
        with open(self.cfg.journal_file) as f:
            row, = csv.DictReader(f)
        self.assertTrue(row["timestamp"].startswith("2026-11-22T12:00:00.123456"))

    def test_dry_run_checks_first_bar_after_entry(self):
        location = self.enterContext(tempfile.TemporaryDirectory())
        paper_cfg = DeltaLiveConfig(env="testnet", dry_run=True, symbols=["ETHUSD"], capital_usd=1200,
                                    resolution="4h", risk_per_trade_pct=0.01, atr_trail_mult=2.5,
                                    use_maker_entry=False, pyramid_add_fraction=0.0, dd_brake_threshold=0.0,
                                    telegram_token="", telegram_chat_id="", data_dir=location)
        self.assertTrue(paper_cfg.state_file.startswith(location))
        engine = delta.LiveDonchianEngine(paper_cfg)
        engine.run_cycle()
        self.assertIn("ETHUSD", engine.state["positions"])
        next_bar = self.end + pd.Timedelta(hours=4)
        self.assertEqual(engine.state["positions"]["ETHUSD"]["paper_checked_at"], next_bar.isoformat())
        self.end = next_bar
        hit_frame = bars(pd.date_range(end=self.end, freq="4h", periods=self.length), self.start, 1)
        hit_frame.iloc[-1, hit_frame.columns.get_loc("low")] = 1000.0
        fake_now = self.end + pd.Timedelta(hours=4, seconds=15)
        with (patch.object(delta, "load_delta_candles", return_value=hit_frame),
              patch.object(delta.pd.Timestamp, "now", return_value=fake_now)):
            engine.run_cycle()
        self.assertEqual(engine.state["positions"], {})

    def test_sol_timeout_does_not_lose_confirmed_eth_trade(self):
        self.cfg.symbols.append("SOLUSD")
        self.fail_sol = True
        for _ in range(2):
            with self.assertRaises(requests.Timeout):
                self.cycle()
        self.assertEqual(len(self.exchange.writes), 1)
        self.assertEqual(self.exchange.stop["size"], self.exchange.position)

    def test_unknown_position_stale_candles_and_warmup_fail_closed(self):
        self.exchange.position = 5
        with self.assertRaisesRegex(RuntimeError, "unexpected exchange position"):
            self.cycle()
        self.exchange.position = 0
        self.length = 80
        self.cycle()
        self.length = 241
        self.end -= pd.Timedelta(hours=4)
        self.cycle()
        self.assertEqual(self.exchange.writes, [])

    def test_bad_switch_and_production_orders_are_rejected(self):
        with patch.dict(os.environ, {"DELTA_ENV": "prod", "DRY_RUN": "tru"}):
            with self.assertRaisesRegex(ValueError, "DRY_RUN"):
                DeltaLiveConfig()
        with self.assertRaisesRegex(ValueError, "Production orders"):
            DeltaLiveConfig(env="prod", dry_run=False)

    def test_exact_weekly_loss_blocks_entries(self):
        bot = delta.LiveDonchianEngine(self.cfg)
        bot.state["weekly_pnl_usd"] = -84
        bot.run_cycle()
        self.assertEqual(self.exchange.writes, [])


class NiftySafetyTest(unittest.TestCase):
    def advisor(self, *, credit, quantity=130, status="NORMAL_OPEN", capital=200000, expected_lot=None, dte=7, spot=25990.0):
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        recent = bars(pd.date_range(end=now.date() - timedelta(days=1), periods=100, freq="1D", tz=now.tzinfo), 25000, 10)
        expiry = int((now + timedelta(days=dte)).timestamp() * 1000)
        short, long = 25650, 25300
        master = gzip.compress(json.dumps([
            dict(underlying_key="NSE_INDEX|Nifty 50", instrument_type="PE", expiry=expiry, strike_price=strike,
                 lot_size=65, instrument_key=str(strike), trading_symbol=f"NIFTY {strike} PE") for strike in (short, long)
        ]).encode())

        def get(url, **kwargs):
            if url == nifty.UPSTOX_MASTERS["NSE"]:
                return SimpleNamespace(raise_for_status=lambda: None, content=master)
            if url.endswith("/market/status/NFO"):
                result = {"status": status}
            elif url.endswith("/option/chain"):
                result = [dict(underlying_spot_price=spot, put_options=dict(instrument_key=str(strike), market_data=book)) for strike, book in (
                    (short, dict(bid_price=credit + 10, bid_qty=quantity)), (long, dict(ask_price=10, ask_qty=195)))]
            else:
                raise AssertionError(url)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(data=result))

        argv = ["advisor", "--use-yf", "--capital", str(capital), "--cost-per-lot", "75"]
        if expected_lot is not None:
            argv += ["--lot-size", str(expected_lot)]
        output = io.StringIO()
        with (patch.object(nifty, "load_yfinance", return_value=recent), patch.object(nifty.requests, "get", get),
              patch.object(sys, "argv", argv), patch.dict(os.environ, {"UPSTOX_ACCESS_TOKEN": "fixture", "TELEGRAM_BOT_TOKEN": ""}),
              redirect_stdout(output)):
            nifty.main()
        return output.getvalue()

    def test_credit_depth_and_lot_size_must_fit_without_rounding_up(self):
        safe = self.advisor(credit=305, quantity=65, capital=400000)
        self.assertIn("Listed expiry", safe)
        self.assertIn("x 1 lot(s) (65 qty)", safe)
        self.assertIn("Rs 3,000.00 (includes", safe)  # (350 - 305) × 65 + 75
        self.assertIn("if your broker supports it", safe)
        for output in (self.advisor(credit=304.99), self.advisor(credit=305, quantity=64),
                       self.advisor(credit=305, status="NORMAL_CLOSE"), self.advisor(credit=305, expected_lot=75),
                       self.advisor(credit=305, quantity=65, capital=400000, dte=1)):
            self.assertIn("NO TRADE:", output)
            self.assertNotIn("CONDITIONAL SPREAD", output)


class BacktestSafetyTest(unittest.TestCase):
    def test_stop_on_entry_bar_and_terminal_equity_reconcile(self):
        index = pd.date_range("2024-01-01", periods=240, freq="4h", tz="UTC")
        history = bars(index, 100, .1)
        history.loc[index[-4], ["high", "close"]] = [130, 129]
        history.loc[index[-3], ["open", "high", "low", "close"]] = [129, 129.5, 80, 128]
        history.loc[index[-2]:, ["open", "high", "low", "close"]] = [128, 128.5, 127.5, 128]
        cfg = PortfolioConfig()
        for low, reason, exit_timestamp in [(80, "ATR_TRAIL", index[-3]), (127, "END_OF_TEST", index[-1])]:
            history.loc[index[-3], "low"] = low
            trades, equity, _ = run_shared_donchian_portfolio({"ETHUSDT": history}, {"ETHUSDT"}, cfg)
            trade, = trades
            self.assertEqual(trade.entry_time, index[-3])
            self.assertEqual((trade.exit_reason, trade.exit_time), (reason, exit_timestamp))
            self.assertAlmostEqual(equity.iloc[-1] - cfg.initial_capital, trade.net_pnl)

    def test_lookback_padding_is_not_in_evaluation_window(self):
        fetch_start, end, eval_start, eval_end = resolve_date_window("4h", days=14, end_date="2024-03-15")
        self.assertEqual(eval_end - eval_start, pd.Timedelta(days=14))
        self.assertEqual(eval_start - fetch_start, pd.Timedelta(days=60))
        self.assertEqual(eval_end, end)


class DeltaConfigurationTest(unittest.TestCase):
    def test_6h_resolution_schedule(self):
        with patch.dict(os.environ, {"DELTA_RESOLUTION": "6h", "DELTA_ENV": "testnet", "DRY_RUN": "true"}):
            cfg = DeltaLiveConfig()
            self.assertEqual(cfg.resolution, "6h")

        t = datetime(2026, 11, 22, 5, 30, 0, tzinfo=delta.timezone.utc)
        self.assertEqual(delta.seconds_until_next_bar_close("6h", now=t), 1815.0)

    def test_legacy_override_cannot_enable_production_writes(self):
        with patch.dict(os.environ, {"DELTA_ENV": "prod", "DRY_RUN": "false", "ALLOW_PROD_ORDERS": "true"}):
            with self.assertRaisesRegex(ValueError, "Production orders are disabled"):
                DeltaLiveConfig()
            cfg = DeltaLiveConfig(dry_run=True)
        client = delta.DeltaExchangeClient(cfg)
        cfg.dry_run = False
        with patch.object(delta.requests, "request") as transport:
            with self.assertRaisesRegex(RuntimeError, "restricted to testnet"):
                client._request("POST", "/v2/orders", body={})
            transport.assert_not_called()

    def test_12h_candles_align_with_utc_and_drop_incomplete_pairs(self):
        timestamps = pd.date_range("2026-11-20 18:00", periods=8, freq="6h", tz="UTC").delete(4)
        rows = [dict(time=int(t.timestamp()), open=i, high=i + 2, low=i - 1, close=i + 1, volume=i + 1)
                for i, t in enumerate(timestamps)]
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"result": rows})
        with patch("data_loader.requests.get", return_value=response):
            frame = load_delta_candles("ETHUSD", "12h", days=2, include_warmup=False)
        self.assertEqual(list(frame.index.tz_convert("UTC")), list(pd.to_datetime(["2026-11-21", "2026-11-22"], utc=True)))
        self.assertEqual(frame.iloc[0].tolist(), [1, 4, 0, 3, 5])


class NiftyHistoricalReplayTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-06-01", "2026-11-13", tz="Asia/Kolkata")
        self.spot = bars(self.dates, 24_000, 0)
        self.expiries = ["2026-11-13"]
        self.short_open = {}
        self.short_close = {}
        self.missing_candle = None

    def contracts(self, expiry, instrument_key):
        suffix = pd.Timestamp(expiry).strftime("%d-%m-%Y")
        return [{"instrument_type": "PE", "strike_price": strike, "lot_size": 65,
                 "instrument_key": f"NSE_FO|{token}|{suffix}"}
                for strike, token in ((23_350, 47983), (23_700, 47984))]

    def candles(self, key, to_date, from_date, interval):
        suffix = pd.Timestamp(to_date).strftime("%d-%m-%Y")
        self.assertIn(key, (f"NSE_FO|47983|{suffix}", f"NSE_FO|47984|{suffix}"))
        # No expiry-day option quote: rely on the index settlement for held positions.
        days = self.spot.index[(self.spot.index >= from_date) & (self.spot.index < to_date)]
        frame = pd.DataFrame({"open": 20.0, "close": 25.0}, index=days)
        if key == f"NSE_FO|47984|{suffix}":
            frame["open"] = [self.short_open.get(str(t.date()), 120.0) for t in days]
            frame["close"] = [self.short_close.get(str(t.date()), 105.0) for t in days]
            if self.missing_candle:
                frame = frame.drop(pd.Timestamp(self.missing_candle, tz=days.tz))
        return frame

    def replay(self, *, capital=2_000_000.0, output=None):
        with (
            patch.object(vns, "load_upstox_expired_expiries", return_value=self.expiries),
            patch.object(vns, "load_upstox", return_value=self.spot.copy()),
            patch.object(vns, "load_upstox_expired_option_contracts", side_effect=self.contracts),
            patch.object(vns, "load_upstox_expired_candles", side_effect=self.candles),
            patch.object(vns.time, "sleep"),
            redirect_stdout(output if output is not None else io.StringIO()),
        ):
            trades, _, _ = vns.run_authentic_upstox_nifty_spreads(capital=capital, cost_per_lot=75.0)
            return trades

    def test_no_trades_reports_one_lot_risk_rejection(self):
        output = io.StringIO()
        self.assertTrue(self.replay(capital=200_000, output=output).empty)
        report = output.getvalue()
        self.assertIn("1: one lot exceeds risk budget", report)
        self.assertIn("risk budget: ₹3,000.00", report)
        self.assertIn("one-lot max loss: ₹16,390.00", report)

    def test_signal_and_exit_use_next_session_opens(self):
        self.short_close["2026-11-10"] = 45.0
        self.short_open["2026-11-11"] = 70.0
        trade, = self.replay().itertuples()
        self.assertEqual((trade.entry_date, trade.exit_date, trade.exit_reason),
                         ("2026-11-09", "2026-11-11", "TP_65PCT"))
        self.assertEqual((trade.entry_credit_pts, trade.exit_val_pts, trade.net_pnl_rs), (99, 50, 3045))

    def test_expiry_settles_at_intrinsic_not_last_option_close(self):
        expiry = self.expiries[0]
        self.spot.loc[expiry, "close"] = 23_620
        trade, = self.replay().itertuples()
        self.assertEqual((trade.exit_date, trade.exit_reason, trade.exit_val_pts), (expiry, "EXPIRY", 80))
        self.spot = self.spot.iloc[:-1]
        self.assertTrue(self.replay().empty)

    def test_open_spread_blocks_overlapping_candidate(self):
        self.expiries.insert(0, "2026-11-06")
        self.spot = self.spot.drop(pd.to_datetime(["2026-11-10", "2026-11-11"]).tz_localize(self.dates.tz))
        self.assertEqual(self.replay()["expiry"].tolist(), ["2026-11-06"])

    def test_missing_exit_open_fails_replay_instead_of_erasing_trade(self):
        self.short_close["2026-11-10"] = 45.0
        self.missing_candle = "2026-11-11"
        with self.assertRaisesRegex(RuntimeError, "2026-11-11 after entry"):
            self.replay()

    def test_tp_day_before_expiry_without_expiry_candle_falls_back_to_intrinsic(self):
        self.short_close["2026-11-12"] = 45.0
        trade, = self.replay().itertuples()
        self.assertEqual((trade.exit_date, trade.exit_reason, trade.exit_val_pts), ("2026-11-13", "EXPIRY", 0.0))


class HardeningRegressionTest(unittest.TestCase):
    def test_resume_requires_flat_account_and_clears_halt(self):
        location = self.enterContext(tempfile.TemporaryDirectory())
        cfg = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                              telegram_token="", telegram_chat_id="", data_dir=location)
        exchange = DeltaFixture()
        with patch.object(delta.requests, "request", exchange.request):
            engine = delta.LiveDonchianEngine(cfg)
            engine.state["halt_new_entries"] = True
            engine.state["positions"]["ETHUSD"] = {"side": "LONG", "contracts": 1, "sl_price": "2000"}
            engine._save_state()
            with self.assertRaisesRegex(RuntimeError, "flat account"):
                engine.run_cycle(resume=True)
            engine.state["positions"].clear()
            engine._save_state()
            engine.run_cycle(resume=True)
            self.assertFalse(engine.state["halt_new_entries"])

    def test_backtest_usd_inr_and_whole_lot_sizing(self):
        index = pd.date_range("2024-01-01", periods=240, freq="4h", tz="UTC")
        history = bars(index, 100, .1)
        history.loc[index[-4], ["high", "close"]] = [130, 129]
        history.loc[index[-3], ["open", "high", "low", "close"]] = [129, 129.5, 128, 128.5]
        cfg = PortfolioConfig(
            initial_capital=200_000.0, usd_inr_rate=85.0, whole_lot_rounding=True,
            donchian_risk_per_trade_pct=0.01, max_concurrent_risk_pct=0.03,
            atr_trail_mult=2.5, slippage_pct=0.0005, pyramid_add_fraction=0.0, dd_brake_threshold=None,
        )
        trades, equity, diag = run_shared_donchian_portfolio({"SOLUSDT": history}, {"SOLUSDT"}, cfg)
        self.assertEqual(diag["skipped_lot_size"], 0)
        trade, = trades
        # Risk = Rs 2000 / 85 = $23.53; SL dist = 2.5 * 1.5 = 3.75 -> 6.27 SOL -> 6.0 SOL.
        self.assertEqual(trade.qty, 6.0)
        self.assertAlmostEqual(equity.iloc[-1] - cfg.initial_capital, trade.net_pnl)

    def test_upstox_loader_raises_on_401(self):
        import data_loader as dl
        resp = SimpleNamespace(status_code=401, ok=False, text="Unauthorized")
        with (patch.dict(os.environ, {"UPSTOX_ACCESS_TOKEN": "expired"}),
              patch.object(dl.requests, "get", return_value=resp)):
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                dl.load_upstox("NIFTY", interval="1d", days=5)


class ConfigCMaxAggressionTest(unittest.TestCase):
    def setUp(self):
        self.location = self.enterContext(tempfile.TemporaryDirectory())
        self.exchange = DeltaFixture()
        self.end = pd.Timestamp.now(tz="UTC").floor("6h") - pd.Timedelta(hours=6)
        self.start = 2000.0
        self.enterContext(patch.object(delta.requests, "request", self.exchange.request))
        self.enterContext(patch.object(delta, "load_delta_candles", self._candles))
        self.enterContext(patch.object(delta.time, "sleep"))

    def _candles(self, symbol, **kwargs):
        return bars(pd.date_range(end=self.end, freq="6h", periods=241), self.start, 1.0)

    def test_config_c_defaults(self):
        with patch.dict(os.environ, {"DELTA_ENV": "testnet", "DRY_RUN": "true", "DELTA_SYMBOLS": ""}, clear=False):
            cfg = DeltaLiveConfig(data_dir=self.location)
        self.assertEqual(cfg.symbols, ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"])
        self.assertEqual(cfg.resolution, "6h")
        self.assertEqual(cfg.risk_per_trade_pct, 0.02)
        self.assertEqual(cfg.max_concurrent_risk_pct, 0.08)
        self.assertEqual(cfg.atr_trail_mult, 3.5)
        self.assertEqual(cfg.pyramid_add_fraction, 0.5)
        self.assertEqual(cfg.dd_brake_threshold, 0.15)
        self.assertTrue(cfg.use_maker_entry)

    def test_maker_entry_fill_timeout_and_cross_fallback(self):
        # 1) Maker fills directly with bracket stop.
        cfg1 = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                               maker_timeout_sec=0.0, telegram_token="", telegram_chat_id="",
                               state_file=f"{self.location}/m1.json", journal_file=f"{self.location}/m1.csv")
        delta.LiveDonchianEngine(cfg1).run_cycle()
        self.assertEqual(self.exchange.writes[0][1]["post_only"], True)
        self.assertGreater(self.exchange.position, 0)
        self.assertEqual(self.exchange.stop["size"], self.exchange.position)

        # 2) Maker stays unfilled -> cancelled -> falls back cleanly to collared IOC.
        self.exchange = DeltaFixture()
        self.exchange.phase = "maker-unfilled"
        self.enterContext(patch.object(delta.requests, "request", self.exchange.request))
        cfg2 = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                               maker_timeout_sec=0.0, telegram_token="", telegram_chat_id="",
                               state_file=f"{self.location}/m2.json", journal_file=f"{self.location}/m2.csv")
        delta.LiveDonchianEngine(cfg2).run_cycle()
        methods = [m for m, _ in self.exchange.writes]
        self.assertEqual(methods, ["POST", "DELETE", "POST"])
        self.assertEqual(self.exchange.writes[2][1]["time_in_force"], "ioc")
        self.assertEqual(self.exchange.stop["size"], self.exchange.position)

        # 3) Maker rejected as post_only_would_cross -> immediate IOC fallback.
        self.exchange = DeltaFixture()
        self.exchange.phase = "maker-cross"
        self.enterContext(patch.object(delta.requests, "request", self.exchange.request))
        cfg3 = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                               maker_timeout_sec=0.0, telegram_token="", telegram_chat_id="",
                               state_file=f"{self.location}/m3.json", journal_file=f"{self.location}/m3.csv")
        delta.LiveDonchianEngine(cfg3).run_cycle()
        self.assertEqual([m for m, _ in self.exchange.writes], ["POST", "POST"])
        self.assertEqual(self.exchange.writes[1][1]["time_in_force"], "ioc")
        self.assertGreater(self.exchange.position, 0)

    def test_winner_pyramid_resizes_stop_and_recovers_from_timeout(self):
        cfg = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                              use_maker_entry=False, telegram_token="", telegram_chat_id="",
                              state_file=f"{self.location}/pyr.json", journal_file=f"{self.location}/pyr.csv")
        delta.LiveDonchianEngine(cfg).run_cycle()
        init_pos = self.exchange.position
        self.assertGreater(init_pos, 0)

        # Advance price strongly (> +1.5R and trail stop strictly above breakeven) and simulate timeout on pyramid POST.
        self.start += 25.0
        self.exchange.mark = 2265.0
        self.exchange.phase = "timeout-pyramid"
        with self.assertRaises(requests.Timeout):
            delta.LiveDonchianEngine(cfg).run_cycle()

        # On next cycle, reconciliation resolves pending_pyramid and widens the existing exchange stop to full size.
        self.exchange.phase = None
        eng = delta.LiveDonchianEngine(cfg)
        eng.run_cycle()
        self.assertGreater(self.exchange.position, init_pos)
        self.assertLessEqual(self.exchange.position, init_pos + (init_pos // 2))
        self.assertEqual(self.exchange.stop["size"], self.exchange.position)
        self.assertTrue(eng.state["positions"]["ETHUSD"]["pyramided"])
        self.assertNotIn("pending_pyramid", eng.state["positions"]["ETHUSD"])

    def test_drawdown_brake_halves_risk_and_clears_on_recovery(self):
        cfg = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                              use_maker_entry=False, pyramid_add_fraction=0.0, telegram_token="", telegram_chat_id="",
                              state_file=f"{self.location}/dd.json", journal_file=f"{self.location}/dd.csv")
        eng = delta.LiveDonchianEngine(cfg)
        eng.state["peak_equity_usd"] = 1200.0
        eng.state["day_equity_open"] = 1000.0
        eng.state["week_equity_open"] = 1000.0
        eng._save_state()

        # Simulate multi-week 16.7% drawdown -> brake activates and halves effective risk limits (1% / 4%).
        self.exchange.wallet_equity = 1000.0
        eng = delta.LiveDonchianEngine(cfg)
        eng.run_cycle()
        self.assertTrue(eng.state["dd_brake_active"])
        self.assertEqual(eng._effective_risk_limits(), (0.01, 0.04))
        self.assertGreater(self.exchange.position, 0)
        self.assertLessEqual(eng.state["positions"]["ETHUSD"]["risk_pct"], 0.01)

        # Recover to 5% DD (<= 7.5% release threshold) -> brake clears cleanly.
        self.exchange.close_at_stop()
        self.exchange.wallet_equity = 1140.0
        eng = delta.LiveDonchianEngine(cfg)
        eng.run_cycle()
        self.assertFalse(eng.state["dd_brake_active"])
        self.assertEqual(eng._effective_risk_limits(), (0.02, 0.08))

    def test_portfolio_engine_pyramiding_and_dd_brake(self):
        index = pd.date_range("2024-01-01", periods=280, freq="6h", tz="UTC")
        history = bars(index, 100, 0.5)
        # Create a clean breakout and steady rally so trailing stop moves past breakeven and triggers pyramid.
        history.iloc[220:, history.columns.get_loc("open")] += [i * 1.2 for i in range(60)]
        history.iloc[220:, history.columns.get_loc("high")] += [i * 1.2 + 0.5 for i in range(60)]
        history.iloc[220:, history.columns.get_loc("low")] += [i * 1.2 - 0.5 for i in range(60)]
        history.iloc[220:, history.columns.get_loc("close")] += [i * 1.2 for i in range(60)]
        cfg = PortfolioConfig(
            initial_capital=1200.0, usd_inr_rate=1.0, whole_lot_rounding=True,
            donchian_risk_per_trade_pct=0.02, max_concurrent_risk_pct=0.08,
            atr_trail_mult=3.5, pyramid_add_fraction=0.5, pyramid_trigger_r=1.5,
            dd_brake_threshold=0.15, entry_slippage_pct=0.0004, slippage_pct=0.0015,
        )
        trades, equity, diag = run_shared_donchian_portfolio({"ETHUSD": history}, {"ETHUSD"}, cfg)
        self.assertGreaterEqual(diag["pyramids"], 1)
        self.assertGreater(equity.iloc[-1], 1200.0)

    def test_kotak_mcx_advisor_sizing_long_only_and_state(self):
        cfg = KotakMcxConfig(capital_inr=98_000.0, data_dir=self.location, telegram_token="", telegram_chat_id="")
        adv = kotak.KotakMcxAdvisor(cfg)
        # GOLDPETAL at Rs 9,600/g and ATR Rs 120 -> SL dist Rs 360 -> 2.5% risk (Rs 2,450) -> 6 lots.
        sized = adv.size_order("GOLDPETAL", price_inr=9600.0, atr_inr=120.0, risk_pct=0.025, max_risk_pct=0.08)
        self.assertEqual(sized["qty"], 6)
        with self.assertRaisesRegex(ValueError, "long-only"):
            adv.record_entry("SILVERBEES.NS", "SHORT", 10, 100.0, 110.0)
        adv.record_entry("GOLDPETAL", "LONG", 6, 9600.0, 9240.0)
        adv.record_pyramid("GOLDPETAL", 3, 10200.0)
        self.assertEqual(adv.state["positions"]["GOLDPETAL"]["qty"], 9)
        self.assertAlmostEqual(adv.state["positions"]["GOLDPETAL"]["entry_inr"], 9800.0)
        pnl = adv.record_exit("GOLDPETAL", 10500.0, fees_inr=50.0)
        self.assertAlmostEqual(pnl, 6250.0)
        self.assertAlmostEqual(adv.state["equity_inr"], 104_250.0)
        self.assertTrue(os.path.exists(f"{self.location}/kotak_mcx_journal.csv"))

    def test_setup_account_configures_isolated_prefs_and_all_5_symbols(self):
        self.exchange.margin_mode = "cross"
        self.exchange.default_auto_topup = True
        self.exchange.deto_for_commission = True
        for pid in (1, 2, 3, 4, 5):
            self.exchange.leverage_by_product[pid] = 20
        cfg = DeltaLiveConfig(env="testnet", dry_run=False, capital_usd=1200,
                              symbols=["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"],
                              telegram_token="", telegram_chat_id="",
                              state_file=f"{self.location}/setup.json", journal_file=f"{self.location}/setup.csv")
        # Fail-closed if an open position exists.
        self.exchange.position = 10
        with self.assertRaisesRegex(RuntimeError, "close all open positions"):
            delta.LiveDonchianEngine(cfg).setup_account()
        self.exchange.position = 0

        summary = delta.LiveDonchianEngine(cfg).setup_account()
        self.assertEqual(self.exchange.margin_mode, "isolated")
        self.assertFalse(self.exchange.default_auto_topup)
        self.assertFalse(self.exchange.deto_for_commission)
        self.assertEqual(summary["symbols"], ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"])
        for pid in (1, 2, 3, 4, 5):
            self.assertEqual(self.exchange.leverage_by_product[pid], 5)
