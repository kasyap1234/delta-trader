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
import live_nifty_spread_advisor as nifty
from config import DeltaLiveConfig, PortfolioConfig
from data_loader import resolve_date_window
from portfolio_engine import run_shared_donchian_portfolio


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

    def request(self, method, url, *, data, **kwargs):
        path = urlsplit(url).path
        order = json.loads(data) if data else {}
        status = 200
        if method != "GET":
            self.writes.append((method, order))
        if path == "/v2/products":
            result = [dict(symbol=s, id=i, contract_value=v, tick_size="0.05", notional_type="vanilla",
                           quoting_asset={"symbol": "USD"}, settling_asset={"symbol": "USD"})
                      for s, i, v in [("ETHUSD", 1, "0.01"), ("SOLUSD", 2, "1")]]
        elif path.startswith("/v2/tickers/"):
            result = dict(mark_price=self.mark)
        elif path == "/v2/positions":
            result = dict(size=self.position if url.endswith("=1") else 0)
        elif path == "/v2/users/trading_preferences":
            result = dict(default_auto_topup=False, deto_for_commission=False)
        elif path.endswith("/orders/leverage"):
            result = dict(leverage=5)
        elif path == "/v2/wallet/balances":
            result = [dict(asset_symbol="USD", balance="1200", available_balance="1200")]
        elif path.startswith("/v2/orders/"):
            result = self.orders.get(path.rsplit("/", 1)[-1])
            status = 200 if result else 404
        elif path == "/v2/orders" and method == "GET":
            result = [self.stop] if self.stop and "product_ids=1" in url else []
        elif path == "/v2/orders" and method == "POST":
            cid = order["client_order_id"]
            size = order["size"]
            if self.phase == "reject-entry":
                return self.response(400, None)
            if order.get("reduce_only"):
                assert self.position == size and order["side"] == "sell"
                self.position = 0
                self.orders[cid] = dict(state="closed", size=size, unfilled_size=0, average_fill_price="2240",
                                        paid_commission="0.3", updated_at=int(time.time() * 1e6))
                return self.response(200, {})
            assert order["order_type"] == "market_order" and order["time_in_force"] == "ioc"
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
            if self.phase == "reject-edit":
                return self.response(400, None)
            self.stop.update(stop_price=order["stop_price"])
            if self.phase == "timeout-edit":
                raise requests.Timeout("Response lost after edit")
            result = self.stop
        else:
            raise AssertionError((method, url))
        return self.response(status, result)

    def close_at_stop(self):
        self.orders["8"] = dict(self.stop, state="closed", unfilled_size=0, average_fill_price="2244.9",
                                paid_commission="0.60", updated_at=int(time.time() * 1e6))
        self.stop = None
        self.position = 0

    @staticmethod
    def response(status, result):
        return SimpleNamespace(ok=status == 200, status_code=status, json=lambda: dict(success=status == 200, result=result))


class DeltaSafetyTest(unittest.TestCase):
    def setUp(self):
        location = self.enterContext(tempfile.TemporaryDirectory())
        self.cfg = DeltaLiveConfig(env="testnet", dry_run=False, symbols=["ETHUSD"], capital_usd=1200,
                                   telegram_token="", telegram_chat_id="", state_file=f"{location}/bot.json",
                                   journal_file=f"{location}/journal.csv")
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
        self.assertEqual(sent, 160)  # $720 available for 5x margin; ATR alone would permit 320.
        self.cycle()
        self.assertEqual(len(self.exchange.writes), 1)
        self.assertEqual(self.exchange.position, 80)
        self.assertEqual(self.exchange.stop["size"], 80)

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
        self.assertEqual(float(row["pnl_usd"]), 6.36)  # (2244.9 - 2240.3) × 1.6 − 1.00

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
    def advisor(self, *, credit, quantity=130, status="NORMAL_OPEN", capital=200000, expected_lot=None):
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        recent = bars(pd.date_range(end=now.date() - timedelta(days=1), periods=100, freq="1D", tz=now.tzinfo), 25000, 10)
        expiry = int((now + timedelta(days=2)).timestamp() * 1000)
        short, long = 25650, 25300
        master = gzip.compress(json.dumps([
            dict(underlying_key="NSE_INDEX|Nifty 50", instrument_type="PE", expiry=expiry, strike_price=strike,
                 lot_size=65, instrument_key=str(strike), trading_symbol=f"NIFTY {strike} PE") for strike in (short, long)
        ]).encode())

        def get(url, **kwargs):
            if url == nifty.UPSTOX_MASTERS["NSE"]:
                return SimpleNamespace(raise_for_status=lambda: None, content=master)
            if url.endswith("/market/status/NSE"):
                result = {"status": status}
            elif url.endswith("/option/chain"):
                result = [dict(put_options=dict(instrument_key=str(strike), market_data=book)) for strike, book in (
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
        self.assertIn("minimum net credit of 305 points", safe)
        for output in (self.advisor(credit=304.99), self.advisor(credit=305, quantity=64),
                       self.advisor(credit=305, status="NORMAL_CLOSE"), self.advisor(credit=305, expected_lot=75)):
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
