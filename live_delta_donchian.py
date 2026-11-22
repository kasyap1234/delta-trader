"""
REAL LIVE / PAPER EXECUTION BOT — ENGINE A (6H Multi-Coin Donchian Trend + 3.5x ATR Trail)
Broker: Delta Exchange India (testnet orders, public production data only)

Config C (Max Aggression, <30% Target Max DD):
  - Timeframe: 6H (evaluated on every closed 6H candle)
  - Assets: BTCUSD, ETHUSD, SOLUSD, BNBUSD, DOGEUSD perpetual futures
  - Risk per Trade: 2.0% of account equity (halved to 1.0% when 15% drawdown brake is active)
  - Max Concurrent Open Risk: 8.0% (Max 5 simultaneous crypto positions, 75% max margin util)
  - Leverage: 5x Isolated (keeps >= 25% free margin buffer)
  - Entry: EMA(50) > EMA(200) & Close > 20-bar Donchian High (Long); inverse for Short.
           Post-only Maker limit entry at touch with short timeout fallback to collared IOC limit.
  - Exit: Immediate Reduce-Only Stop-Market at 3.5x ATR(14), trailed every 6H bar (never widened).
  - Pyramiding: Adds +0.5x size once up +1.5R with stop >= breakeven, resizing the exchange stop.
  - Safety: DRY_RUN=true by default, Testnet default, Daily (-6%) & Weekly (-12%) circuit breakers,
            15% Drawdown Brake, crash-safe JSON state persistence, and Telegram notifications.
"""

import argparse
import csv
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import hmac
import json
import logging
import math
import signal
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from logging.handlers import RotatingFileHandler
from threading import Event
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from config import DeltaLiveConfig
from data_loader import load_delta_candles
from portfolio_engine import add_donchian_features, add_ema_atr_features

log = logging.getLogger("DeltaDonchianBot")


def send_telegram(cfg: DeltaLiveConfig, text: str) -> None:
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage",
            json={"chat_id": cfg.telegram_chat_id, "text": text},
            timeout=10,
        )
        if not response.ok or not response.json().get("ok"):
            log.warning("Telegram rejected alert (HTTP %s)", response.status_code)
    except (requests.RequestException, ValueError):
        log.warning("Telegram alert failed")


class DeltaAPIError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, error: dict):
        self.status = status
        self.code = error.get("code")
        self.context = error.get("context")
        super().__init__(f"Delta API error ({status}) on {method} {path}: {self.code or 'invalid or unsuccessful response'}; context={self.context!r}")


def exchange_timestamp(value) -> datetime:
    try:
        epoch = int(value)
    except (TypeError, ValueError):
        return pd.to_datetime(value, utc=True, errors="raise").to_pydatetime()
    unit = "us" if abs(epoch) >= 10**14 else "ms" if abs(epoch) >= 10**11 else "s"
    return pd.to_datetime(epoch, unit=unit, utc=True).to_pydatetime()


class DeltaExchangeClient:
    """REST v2 client with writes restricted to the India testnet."""

    def __init__(self, cfg: DeltaLiveConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url
        self.products_cache: Dict[str, Dict[str, Any]] = {}

    def _sign(self, method: str, path: str, query_string: str = "", payload: str = "") -> Dict[str, str]:
        ts = str(int(time.time()))
        sig_data = method.upper() + ts + path + query_string + payload
        signature = hmac.new(
            self.cfg.api_secret.encode("utf-8"),
            sig_data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "api-key": self.cfg.api_key,
            "timestamp": ts,
            "signature": signature,
            "User-Agent": "delta-donchian-4h-bot/1.0",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None, auth: bool = True) -> dict:
        method = method.upper()
        if method != "GET" and (self.cfg.dry_run or self.cfg.env != "testnet"):
            raise RuntimeError("Delta writes are restricted to testnet order mode")
        query_string = "?" + urlencode(params, safe=",") if params else ""
        payload = json.dumps(body, separators=(",", ":")) if body is not None else ""
        url = f"{self.base_url}{path}{query_string}"
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            headers = self._sign(method, path, query_string, payload) if auth else {"User-Agent": "delta-donchian-bot/1.0"}
            try:
                response = requests.request(method, url, headers=headers, data=payload if body is not None else None,
                                            timeout=10, allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == attempts - 1:
                    raise
            else:
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                if not isinstance(data, dict):
                    data = {}
                if response.ok and data.get("success") is True:
                    return data
                if attempt == attempts - 1 or response.status_code not in {429, 500, 502, 503, 504}:
                    error = data.get("error")
                    raise DeltaAPIError(response.status_code, method, path, error if isinstance(error, dict) else {})
            time.sleep(.5 * 2**attempt)

    def load_products(self) -> None:
        resp = self._request("GET", "/v2/products", auth=False)
        products = {}
        for p in resp.get("result", []):
            sym = p.get("symbol")
            if sym in self.cfg.symbols:
                if p["notional_type"] != "vanilla" or any(p[k]["symbol"] != "USD" for k in ("quoting_asset", "settling_asset")):
                    raise RuntimeError(f"Unsupported contract denomination for {sym}")
                max_lev_notional = float(p.get("max_leverage_notional") or 0)
                products[sym] = {
                    "id": int(p["id"]),
                    "contract_value": float(p["contract_value"]),
                    "tick_size": Decimal(p["tick_size"]),
                    "maintenance_rate": float(p["maintenance_margin"]) / 100,
                    "maintenance_scaling": float(p["maintenance_margin_scaling_factor"]),
                    "max_leverage_notional": max_lev_notional if max_lev_notional > 0 else math.inf,
                    "fee_rate": float(p["taker_commission_rate"]) * 1.18,
                    "maker_fee_rate": float(p.get("maker_commission_rate", p["taker_commission_rate"])) * 1.18,
                }
        if set(products) != set(self.cfg.symbols):
            raise RuntimeError("Delta did not return all configured products")
        self.products_cache = products
        log.info(f"Loaded Delta products ({self.cfg.env}): {self.products_cache}")

    def get_ticker_quotes(self, symbol: str) -> tuple[float, float, float]:
        resp = self._request("GET", f"/v2/tickers/{symbol}", auth=False)
        res = resp["result"]
        mark = float(res["mark_price"])
        if not math.isfinite(mark) or mark <= 0:
            raise RuntimeError(f"Invalid mark price for {symbol}")
        quotes = res.get("quotes") if isinstance(res.get("quotes"), dict) else {}
        bid = float(quotes.get("best_bid") or 0)
        ask = float(quotes.get("best_ask") or 0)
        bid = bid if math.isfinite(bid) and bid > 0 else mark
        ask = ask if math.isfinite(ask) and ask > 0 else mark
        return mark, bid, ask

    def get_ticker_mark_price(self, symbol: str) -> float:
        mark, _, _ = self.get_ticker_quotes(symbol)
        return mark

    def position(self, product_id: int) -> dict:
        return self._request("GET", "/v2/positions", params={"product_id": product_id})["result"]

    def open_orders(self, product_id: int) -> list:
        resp = self._request("GET", "/v2/orders", params={"product_ids": product_id, "states": "open,pending"})
        if resp.get("meta", {}).get("after"):
            raise RuntimeError(f"Unexpected open order count for product {product_id}")
        return resp["result"]

    def get_margin_mode(self, prefs: Optional[dict] = None) -> Optional[str]:
        profile = self._request("GET", "/v2/profile")["result"]
        mode = profile.get("margin_mode") if isinstance(profile, dict) else None
        if mode is not None:
            return mode
        if prefs is None:
            prefs = self._request("GET", "/v2/users/trading_preferences")["result"]
        mode = prefs.get("margin_mode")
        if mode is None:
            accounts = self._request("GET", "/v2/sub_accounts")["result"]
            account = next((a for a in accounts if str(a["id"]) == str(prefs["user_id"])), {})
            mode = account.get("margin_mode")
        return mode

    def check_entry_settings(self, product_id: int) -> None:
        prefs = self._request("GET", "/v2/users/trading_preferences")["result"]
        if prefs["default_auto_topup"] is not False or prefs["deto_for_commission"] is not False:
            raise RuntimeError("Disable auto top-up and DETO commission on the dedicated testnet account")
        mode = self.get_margin_mode(prefs)
        leverage = self._request("GET", f"/v2/products/{product_id}/orders/leverage")["result"]["leverage"]
        if mode != "isolated" or float(leverage) != self.cfg.leverage:
            raise RuntimeError(f"Verify isolated margin and set product {product_id} to {self.cfg.leverage}x leverage")

    def set_margin_mode(self, mode: str = "isolated") -> None:
        self._request("PUT", "/v2/users/margin_mode", body={"margin_mode": mode})

    def set_trading_preferences(self, default_auto_topup: bool = False, deto_for_commission: bool = False) -> None:
        self._request("PUT", "/v2/users/trading_preferences", body={
            "default_auto_topup": default_auto_topup,
            "deto_for_commission": deto_for_commission,
        })

    def set_order_leverage(self, product_id: int, leverage: int) -> None:
        self._request("POST", f"/v2/products/{product_id}/orders/leverage", body={"leverage": str(leverage)})

    def wallet(self) -> tuple[float, float]:
        response = self._request("GET", "/v2/wallet/balances")
        wallet = next((w for w in response["result"] if w["asset_symbol"] == "USD"), None)
        if wallet is None:
            raise RuntimeError("Delta account has no USD wallet")
        # Use USD wallet balance rather than meta.net_equity so unrelated BTC/ETH spot balances don't skew sizing/DD.
        equity = float(wallet["balance"])
        available = float(wallet["available_balance"])
        if not all(math.isfinite(x) for x in (equity, available)):
            raise RuntimeError("Invalid Delta wallet equity or available balance")
        return equity, available

    def submit_entry(self, product_id: int, pos: dict) -> None:
        self._request("POST", "/v2/orders", body={
            "product_id": product_id, "size": pos["contracts"], "side": "buy" if pos["side"] == "LONG" else "sell",
            "order_type": "limit_order", "limit_price": pos["entry_limit"], "time_in_force": "ioc",
            "client_order_id": pos["client_order_id"],
            "bracket_stop_trigger_method": "mark_price", "bracket_stop_loss_price": pos["sl_price"],
        })

    def submit_maker_entry(self, product_id: int, pos: dict) -> dict:
        return self._request("POST", "/v2/orders", body={
            "product_id": product_id, "size": pos["contracts"], "side": "buy" if pos["side"] == "LONG" else "sell",
            "order_type": "limit_order", "limit_price": pos["maker_limit"], "time_in_force": "gtc",
            "post_only": True, "client_order_id": pos["client_order_id"],
            "bracket_stop_trigger_method": "mark_price", "bracket_stop_loss_price": pos["sl_price"],
        })["result"]

    def submit_pyramid_add(self, product_id: int, side: str, contracts: int, limit_price: str, client_order_id: str) -> None:
        self._request("POST", "/v2/orders", body={
            "product_id": product_id, "size": contracts, "side": "buy" if side == "LONG" else "sell",
            "order_type": "limit_order", "limit_price": limit_price, "time_in_force": "ioc",
            "client_order_id": client_order_id,
        })

    def cancel_order(self, product_id: int, order_id: int) -> None:
        self._request("DELETE", "/v2/orders", body={"id": order_id, "product_id": product_id})

    def edit_stop(self, product_id: int, order_id: int, size: int, stop_price: str) -> None:
        self._request("PUT", "/v2/orders", body={"id": order_id, "product_id": product_id, "size": size, "stop_price": stop_price})

    def emergency_close(self, product_id: int, size: int, client_id: str) -> None:
        self._request("POST", "/v2/orders", body={
            "product_id": product_id, "size": abs(size), "side": "sell" if size > 0 else "buy",
            "order_type": "market_order", "time_in_force": "ioc", "reduce_only": True, "client_order_id": client_id,
        })


class LiveDonchianEngine:
    def __init__(self, cfg: DeltaLiveConfig):
        self.cfg = cfg
        self.client = DeltaExchangeClient(cfg)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.cfg.state_file):
            with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            state.setdefault("peak_equity_usd", max(float(state.get("equity_usd", self.cfg.capital_usd)), self.cfg.capital_usd))
            state.setdefault("dd_brake_active", False)
            return state
        return {
            "equity_usd": self.cfg.capital_usd,
            "peak_equity_usd": self.cfg.capital_usd,
            "dd_brake_active": False,
            "positions": {},
            "last_entry_bar": {},
            "daily_pnl_usd": 0.0,
            "weekly_pnl_usd": 0.0,
            "last_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "last_week": datetime.now(timezone.utc).strftime("%G-W%V"),
        }

    def _save_state(self) -> None:
        path = Path(self.cfg.state_file)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _log_journal(self, row: Dict[str, Any]) -> None:
        exists = os.path.exists(self.cfg.journal_file)
        with open(self.cfg.journal_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _reset_periodic_counters(self) -> None:
        now = datetime.now(timezone.utc)
        for period, key in (("day", now.strftime("%Y-%m-%d")), ("week", now.strftime("%G-W%V"))):
            baseline = f"{period}_equity_open"
            if self.state[f"last_{period}"] != key:
                self.state[f"last_{period}"] = key
                self.state["daily_pnl_usd" if period == "day" else "weekly_pnl_usd"] = 0.0
                self.state[baseline] = self.state["equity_usd"]
            self.state.setdefault(baseline, self.state["equity_usd"])

    def _update_drawdown_brake(self) -> None:
        eq = float(self.state["equity_usd"])
        peak = max(float(self.state.get("peak_equity_usd", eq)), eq)
        self.state["peak_equity_usd"] = peak
        if not self.cfg.dd_brake_threshold or peak <= 0:
            self.state["dd_brake_active"] = False
            return
        dd = (peak - eq) / peak
        was_active = bool(self.state.get("dd_brake_active", False))
        if dd >= self.cfg.dd_brake_threshold:
            self.state["dd_brake_active"] = True
        elif dd <= self.cfg.dd_brake_threshold * 0.5:
            self.state["dd_brake_active"] = False
        now_active = bool(self.state["dd_brake_active"])
        if now_active != was_active:
            msg = (f"DRAWDOWN BRAKE {'ACTIVATED' if now_active else 'CLEARED'} | "
                   f"Equity=${eq:.2f} | Peak=${peak:.2f} | DD={dd*100:.1f}% | "
                   f"Risk scale={'%.2fx' % self.cfg.dd_brake_scale if now_active else '1.00x'}")
            log.warning(msg)
            send_telegram(self.cfg, msg)

    def _effective_risk_limits(self) -> tuple[float, float]:
        scale = self.cfg.dd_brake_scale if (self.cfg.dd_brake_threshold > 0 and self.state.get("dd_brake_active")) else 1.0
        return self.cfg.risk_per_trade_pct * scale, self.cfg.max_concurrent_risk_pct * scale

    def _circuit_breaker_tripped(self) -> bool:
        equity = self.state["equity_usd"]
        if equity <= 0:
            return True
        for period, counter, limit in (("day", "daily_pnl_usd", self.cfg.daily_loss_limit_pct),
                                       ("week", "weekly_pnl_usd", self.cfg.weekly_loss_limit_pct)):
            baseline = self.state.get(f"{period}_equity_open", equity)
            budget = min(baseline, self.cfg.capital_usd) * limit
            pnl = min(self.state[counter], equity - baseline)
            if pnl <= -budget + 1e-9:
                log.warning("%s loss limit reached. Pausing new entries.", period)
                return True
        return False

    def _reconcile(self, sym: str, meta: dict, wait_for_bracket: bool = True) -> None:
        positions = self.state["positions"]
        pos = positions.get(sym)
        product = meta["id"]
        remote = self.client.position(product)
        size = int(remote["size"])
        orders = self.client.open_orders(product)
        if pos is None:
            if size or orders:
                raise RuntimeError(f"{sym}: unexpected exchange position or open orders; reconcile manually")
            return

        is_long = pos["side"] == "LONG"
        sign = 1 if is_long else -1
        if "emergency_close_id" in pos and (size or orders):
            raise RuntimeError(f"{sym}: emergency close not confirmed; check the testnet account now")
        if "sl_order_id" not in pos and "emergency_close_id" not in pos:
            entry = self.client._request("GET", f"/v2/orders/client_order_id/{pos['client_order_id']}")["result"]
            if entry["state"] == "open" and pos.get("entry_mode") == "maker":
                self.client.cancel_order(product, int(entry["id"]))
                entry = self.client._request("GET", f"/v2/orders/client_order_id/{pos['client_order_id']}")["result"]
            filled = int(entry["size"]) - int(entry["unfilled_size"])
            if (entry["state"] not in {"closed", "cancelled"} or int(entry["product_id"]) != product
                    or entry["side"] != ("buy" if is_long else "sell") or not 0 <= filled <= pos["contracts"]):
                raise RuntimeError(f"{sym}: entry not reconciled ({pos['client_order_id']})")
            if not filled and not size and not orders:
                del positions[sym]
                self._save_state()
                return
            if not filled or size != sign * filled:
                raise RuntimeError(f"{sym}: position does not match entry fill; reconcile manually")
            fill_price = float(entry["average_fill_price"])
            entry_fee = float(entry["paid_commission"])
            if not math.isfinite(fill_price) or fill_price <= 0 or not math.isfinite(entry_fee):
                raise RuntimeError(f"{sym}: invalid entry fill")
        elif "pending_pyramid" in pos:
            pending = pos["pending_pyramid"]
            try:
                pyr = self.client._request("GET", f"/v2/orders/client_order_id/{pending['client_order_id']}")["result"]
            except DeltaAPIError as err:
                if err.status == 404 and size == sign * pos["contracts"]:
                    del pos["pending_pyramid"]
                    self._save_state()
                    return self._reconcile(sym, meta, wait_for_bracket=wait_for_bracket)
                raise
            pyr_filled = int(pyr["size"]) - int(pyr["unfilled_size"])
            if (pyr["state"] not in {"closed", "cancelled"} or int(pyr["product_id"]) != product
                    or pyr["side"] != ("buy" if is_long else "sell") or not 0 <= pyr_filled <= int(pending["contracts"])):
                raise RuntimeError(f"{sym}: pyramid order not reconciled ({pending['client_order_id']})")
            if not pyr_filled:
                del pos["pending_pyramid"]
                self._save_state()
                if size != sign * pos["contracts"]:
                    raise RuntimeError(f"{sym}: exchange size changed after zero-fill pyramid; reconcile manually")
            else:
                expected_total = pos["contracts"] + pyr_filled
                if size != sign * expected_total:
                    raise RuntimeError(f"{sym}: position does not match pyramid fill; reconcile manually")
                pyr_price = float(pyr["average_fill_price"])
                pyr_fee = float(pyr["paid_commission"])
                if not math.isfinite(pyr_price) or pyr_price <= 0 or not math.isfinite(pyr_fee):
                    raise RuntimeError(f"{sym}: invalid pyramid fill")
                pos["entry_price"] = (pos["entry_price"] * pos["contracts"] + pyr_price * pyr_filled) / expected_total
                pos["contracts"] = expected_total
                pos["entry_fee"] = pos.get("entry_fee", 0.0) + pyr_fee
                pos["pyramided"] = True
                del pos["pending_pyramid"]
                self._save_state()
                # Immediately resize the existing protective stop to cover the enlarged position.
                stop_order = next((o for o in orders if int(o["id"]) == pos["sl_order_id"]), None)
                if stop_order is None:
                    self._emergency_close(sym, product, pos, size, "protective stop missing after pyramid")
                if int(stop_order["unfilled_size"]) != expected_total:
                    self.client.edit_stop(product, pos["sl_order_id"], expected_total, pos["sl_price"])
                    orders = self.client.open_orders(product)
        elif not size and not orders:
            order_path = (f"client_order_id/{pos['emergency_close_id']}" if "emergency_close_id" in pos
                          else str(pos["sl_order_id"]))
            stop_fill = self.client._request("GET", f"/v2/orders/{order_path}")["result"]
            exit_px = float(stop_fill["average_fill_price"] or 0)
            exit_fee = float(stop_fill["paid_commission"])
            if (stop_fill["state"] != "closed" or int(stop_fill["size"]) - int(stop_fill["unfilled_size"]) != pos["contracts"]
                    or not math.isfinite(exit_px) or exit_px <= 0 or not math.isfinite(exit_fee)):
                raise RuntimeError(f"{sym}: cannot verify closing fill; reconcile manually")
            at = exchange_timestamp(stop_fill["updated_at"])
            self._record_exit(sym, pos, exit_px, pos["entry_fee"] + exit_fee, at)
            return
        elif size != sign * pos["contracts"]:
            raise RuntimeError(f"{sym}: exchange size changed unexpectedly; reconcile manually")

        if not orders and wait_for_bracket:
            for _ in range(4):
                time.sleep(.5)
                refreshed = int(self.client.position(product)["size"])
                orders = self.client.open_orders(product)
                if refreshed != size or orders:
                    return self._reconcile(sym, meta, wait_for_bracket=False)

        if not orders:
            if int(self.client.position(product)["size"]) != size:
                raise RuntimeError(f"{sym}: position changed during reconciliation")
            if "sl_order_id" not in pos:
                pos.update(contracts=abs(size), entry_price=fill_price, entry_fee=entry_fee)
            self._emergency_close(sym, product, pos, size, "no protective stop visible")

        if len(orders) != 1:
            raise RuntimeError(f"{sym}: expected exactly one protective stop; check the testnet account now")
        stop = orders[0]
        stop_px = Decimal(stop["stop_price"] or "0")
        previous = Decimal(pos["sl_price"])
        if (stop["stop_order_type"] != "stop_loss_order" or stop["order_type"] != "market_order"
                or stop["stop_trigger_method"] != "mark_price" or stop["reduce_only"] not in (True, "true")
                or stop["side"] != ("sell" if is_long else "buy") or int(stop["unfilled_size"]) != abs(size)
                or not stop_px.is_finite() or stop_px <= 0 or (stop_px - previous) * sign < 0
                or ("sl_order_id" in pos and int(stop["id"]) != pos["sl_order_id"])):
            raise RuntimeError(f"{sym}: exchange stop does not match position; check the testnet account now")
        if "sl_order_id" not in pos:
            risk_usd = max((fill_price - float(stop_px)) * sign, 0) * abs(size) * meta["contract_value"] + entry_fee
            init_sl_dist = max((fill_price - float(stop_px)) * sign, float(pos.get("init_sl_dist", 0.0)))
            pos.update(contracts=abs(size), entry_price=fill_price, init_entry_price=fill_price,
                       init_sl_dist=init_sl_dist, entry_fee=entry_fee, sl_order_id=int(stop["id"]),
                       risk_usd=risk_usd, risk_pct=risk_usd / self.state["equity_usd"])
            log.info("%s: reconciled %s contracts with verified stop %s", sym, size, stop["id"])
        pos["sl_price"] = format(stop_px, "f")
        self._save_state()

    def _emergency_close(self, sym, product, pos, size, reason):
        pos["emergency_close_id"] = uuid4().hex
        self.state["halt_new_entries"] = True
        self._save_state()
        log.error("%s: %s. Submitting one reduce-only emergency close.", sym, reason)
        self.client.emergency_close(product, size, pos["emergency_close_id"])
        raise RuntimeError(f"{sym}: emergency close submitted ({reason}); verify the testnet account")

    def setup_account(self) -> Dict[str, Any]:
        """Programmatically configures isolated margin, disables auto top-up/DETO, and sets leverage on all symbols."""
        if self.cfg.dry_run or self.cfg.env != "testnet":
            raise RuntimeError("--setup-account requires DELTA_ENV=testnet and DRY_RUN=false")
        if not self.client.products_cache:
            self.client.load_products()

        for sym in self.cfg.symbols:
            pid = self.client.products_cache[sym]["id"]
            pos = self.client.position(pid)
            orders = self.client.open_orders(pid)
            if int(pos["size"]) != 0 or orders:
                raise RuntimeError(f"{sym}: close all open positions and cancel open orders before --setup-account")

        # Set account-level isolated margin if not already isolated.
        prefs = self.client._request("GET", "/v2/users/trading_preferences")["result"]
        mode = self.client.get_margin_mode(prefs)
        if mode != "isolated":
            self.client.set_margin_mode("isolated")
            log.info("Set Delta testnet account margin_mode=isolated")

        if prefs.get("default_auto_topup") is not False or prefs.get("deto_for_commission") is not False:
            self.client.set_trading_preferences(default_auto_topup=False, deto_for_commission=False)
            log.info("Disabled default_auto_topup and deto_for_commission")

        configured = []
        for sym in self.cfg.symbols:
            pid = self.client.products_cache[sym]["id"]
            self.client.set_order_leverage(pid, self.cfg.leverage)
            self.client.check_entry_settings(pid)
            configured.append(sym)
            log.info("%s (product_id=%s): verified isolated margin + %sx leverage", sym, pid, self.cfg.leverage)

        equity, available = self.client.wallet()
        self.state["equity_usd"] = equity
        self.state["peak_equity_usd"] = max(float(self.state.get("peak_equity_usd", equity)), equity)
        self._save_state()
        return {"symbols": configured, "leverage": self.cfg.leverage, "equity_usd": equity, "available_usd": available}

    def run_cycle(self, *, resume: bool = False) -> bool:
        self._reset_periodic_counters()
        if not self.client.products_cache:
            self.client.load_products()
        errors = []
        entry_blockers = []
        reconciled = []
        for sym in self.cfg.symbols:
            try:
                if not self.cfg.dry_run:
                    self._reconcile(sym, self.client.products_cache[sym])
                reconciled.append(sym)
            except Exception as error:
                log.error("%s reconciliation failed: %s", sym, error)
                errors.append(error)
                entry_blockers.append(error)

        available = None
        if not self.cfg.dry_run:
            try:
                self.state["equity_usd"], available = self.client.wallet()
            except Exception as error:
                errors.append(error)
                entry_blockers.append(error)
        self._update_drawdown_brake()

        if resume and (errors or self.state["positions"]):
            raise RuntimeError("Resume requires successful reconciliation and a flat account")
        if resume:
            self.state["halt_new_entries"] = False
            self._save_state()
            return False

        # Manage verified positions before considering any new orders.
        candidates = []
        retry = False
        for sym in reconciled:
            try:
                candidate, stale = self._process_symbol(sym)
                retry |= stale
                if candidate:
                    candidates.append(candidate)
            except Exception as error:
                log.error("%s processing failed: %s", sym, error)
                errors.append(error)
                if sym in self.state["positions"]:
                    entry_blockers.append(error)

        for candidate in candidates:
            if entry_blockers:
                break
            try:
                self._consider_entry(*candidate, available)
            except Exception as error:
                log.error("%s entry failed: %s", candidate[0], error)
                errors.append(error)
                entry_blockers.append(error)

        self._save_state()
        if errors:
            raise errors[0]
        return retry

    def _process_symbol(self, sym):
        meta = self.client.products_cache[sym]
        raw_df = load_delta_candles(sym, resolution=self.cfg.resolution, days=60, base_url=self.cfg.base_url)
        period = pd.Timedelta("1D" if self.cfg.resolution == "1d" else self.cfg.resolution)
        expected = pd.Timestamp.now(tz="UTC").floor(period) - period
        history = raw_df.loc[raw_df.index <= expected]
        if history.empty or history.index[-1] != expected:
            log.warning("%s: latest closed candle is unavailable", sym)
            return None, True
        valid = np.isfinite(history[["open", "high", "low", "close", "volume"]]).all(axis=1)
        history = history.loc[valid]
        if expected not in history.index:
            raise RuntimeError(f"{sym}: invalid latest closed candle")
        gaps = history.index.to_series().diff().ne(period)
        contiguous = history.loc[gaps[gaps].index[-1]:]
        continuous = len(contiguous) > self.cfg.ema_slow
        pos = self.state["positions"].get(sym)
        if not continuous and pos is None:
            log.warning("%s: insufficient continuous candles for new entries", sym)
            return None, False
        if continuous:
            features = add_donchian_features(contiguous, entry_len=self.cfg.donchian_entry_len,
                                             warmup_bars=self.cfg.ema_slow)
        else:
            features = add_ema_atr_features(history)
        if expected not in features.index or not np.isfinite(features.at[expected, "atr14"]):
            raise RuntimeError(f"{sym}: insufficient ATR history to manage position")
        closed = features.loc[expected]
        mark_px, best_bid, best_ask = self.client.get_ticker_quotes(sym)
        atr = float(closed["atr14"])
        if pos is not None:
            self._manage_position(sym, meta, pos, mark_px, float(closed["close"]), atr, history, period)
            return None, False
        return (sym, meta, expected, period, closed, mark_px, best_bid, best_ask, atr), False

    def _consider_entry(self, sym, meta, expected, period, closed, mark_px, best_bid, best_ask, atr, available):
        if (self.state.get("halt_new_entries") or self._circuit_breaker_tripped()
                or self.state["last_entry_bar"].get(sym) == expected.isoformat()):
            return

        open_positions = self.state["positions"]
        if len(open_positions) >= self.cfg.max_crypto_concurrent:
            log.info(f"{sym}: Max concurrent crypto positions ({self.cfg.max_crypto_concurrent}) reached.")
            return

        eff_risk_pct, eff_max_risk_pct = self._effective_risk_limits()
        open_risk_pct = sum(p["risk_pct"] for p in open_positions.values())
        if (open_risk_pct + eff_risk_pct) > (eff_max_risk_pct + 1e-9):
            log.info(f"{sym}: Max portfolio risk cap ({eff_max_risk_pct*100:.1f}%) reached.")
            return

        side = None
        if closed["ema50"] > closed["ema200"] and closed["close"] > closed["donchian_high"]:
            side = "LONG"
        elif closed["ema50"] < closed["ema200"] and closed["close"] < closed["donchian_low"]:
            side = "SHORT"

        if side:
            passive_px = best_bid if side == "LONG" else best_ask
            self._enter_position(
                sym, meta, side, mark_px, atr, expected.isoformat(), available,
                next_bar_at=(expected + period).isoformat(), passive_px=passive_px,
            )
        else:
            log.info(f"{sym} {self.cfg.resolution.upper()} | Close={closed['close']:.2f} | EMA50={closed['ema50']:.2f} | EMA200={closed['ema200']:.2f} | No signal.")

    def _enter_position(
        self, sym: str, meta: dict, side: str, entry_px: float, atr: float, bar: str,
        available=None, next_bar_at: Optional[str] = None, passive_px: Optional[float] = None,
    ) -> None:
        sl_dist = self.cfg.atr_trail_mult * atr
        if not math.isfinite(sl_dist) or sl_dist <= 0:
            return
        sign = 1 if side == "LONG" else -1
        sl_price = self._stop_price(entry_px - sign * sl_dist, meta["tick_size"], sign)
        if not (float(sl_price) > 0 and (entry_px - float(sl_price)) * sign > 0):
            return

        contract_val = meta["contract_value"]
        eff_risk_pct, _ = self._effective_risk_limits()
        allocation = self.state["equity_usd"]
        if not self.cfg.dry_run:
            self.client.check_entry_settings(meta["id"])
            _, available = self.client.wallet()
        allocation = min(allocation, self.cfg.capital_usd)
        allocated_margin = sum(p["contracts"] * p["contract_value"] * p["entry_price"] / self.cfg.leverage
                               for p in self.state["positions"].values())
        max_util = self.cfg.max_margin_utilization
        free_buf = 1.0 - max_util
        margin_budget = max_util * allocation - allocated_margin
        if not self.cfg.dry_run:
            margin_budget = min(margin_budget, available - free_buf * allocation)
        if margin_budget <= 0:
            log.warning("%s: Insufficient free margin budget", sym)
            return
        fee_rate = meta["fee_rate"]
        maker_fee_rate = meta.get("maker_fee_rate", fee_rate)
        slip = self.cfg.entry_slippage_pct
        entry_limit = self._stop_price(entry_px * (1 + sign * slip), meta["tick_size"], -sign)
        touch_px = passive_px if (passive_px is not None and math.isfinite(passive_px) and passive_px > 0) else entry_px
        # Passive maker price rounds toward passive side (down for LONG, up for SHORT) so it doesn't cross.
        maker_limit = self._stop_price(touch_px, meta["tick_size"], -sign)
        limit = float(entry_limit)
        stop = float(sl_price)
        expected_exit = stop * (1 - sign * slip)
        risk_per_unit = (limit - expected_exit) * sign + fee_rate * (limit + expected_exit)
        margin_price = max(entry_px * (1 + slip), limit)
        if risk_per_unit <= 0 or margin_price <= 0:
            return
        contracts = math.floor(min(allocation * eff_risk_pct / (risk_per_unit * contract_val),
                                   margin_budget * self.cfg.leverage / (margin_price * contract_val),
                                   meta["max_leverage_notional"] / (margin_price * contract_val)))
        maintenance = meta["maintenance_rate"] + meta["maintenance_scaling"] * contracts
        liquidation_buffer = 1 / self.cfg.leverage - maintenance - .02
        if (entry_px - stop) * sign / entry_px >= liquidation_buffer:
            log.warning("%s: stop exceeds estimated liquidation buffer", sym)
            return
        if contracts < 1:
            log.warning("%s: No contracts fit the risk and margin limits", sym)
            return

        paper_fill = float(maker_limit) if self.cfg.use_maker_entry else limit
        paper_fee = paper_fill * contracts * contract_val * (maker_fee_rate if self.cfg.use_maker_entry else fee_rate)
        pos = self.state["positions"][sym] = {
            "side": side,
            "contracts": contracts,
            "contract_value": contract_val,
            "entry_price": paper_fill if self.cfg.dry_run else entry_px,
            "init_entry_price": paper_fill if self.cfg.dry_run else entry_px,
            "init_sl_dist": sl_dist,
            "entry_limit": entry_limit,
            "maker_limit": maker_limit,
            "entry_mode": "maker" if self.cfg.use_maker_entry else "ioc",
            "entry_fee": paper_fee if self.cfg.dry_run else 0,
            "paper_checked_at": next_bar_at or datetime.now(timezone.utc).isoformat(),
            "sl_price": sl_price,
            "risk_pct": eff_risk_pct,
            "pyramided": False,
            "client_order_id": uuid4().hex,
        }
        self.state["last_entry_bar"][sym] = bar
        # Persist the intent before the request. A missing or ambiguous response never authorizes a resubmission.
        self._save_state()
        if not self.cfg.dry_run:
            reject_codes = {
                "insufficient_margin", "insufficient_commission", "order_size_exceed_available",
                "order_size_not_available_in_orderbook", "risk_limits_breached", "immediate_liquidation",
                "invalid_contract", "trading_not_allowed_on_current_margin_mode", "less_than_order_min_size",
                "invalid_order", "invalid_stop_price", "product_not_operational",
            }
            need_ioc_fallback = not self.cfg.use_maker_entry
            if self.cfg.use_maker_entry:
                try:
                    maker_res = self.client.submit_maker_entry(meta["id"], pos)
                except DeltaAPIError as error:
                    if error.status == 400 and error.code in {
                        "post_only_would_cross", "immediate_execution_stop_order", "order_would_immediately_match",
                    }:
                        log.info("%s: post-only maker would cross (%s); falling back to collared IOC", sym, error.code)
                        need_ioc_fallback = True
                    else:
                        if error.status == 400 and error.code in reject_codes:
                            del self.state["positions"][sym]
                            self._save_state()
                        raise
                else:
                    deadline = time.monotonic() + self.cfg.maker_timeout_sec
                    order_info = maker_res
                    while order_info.get("state") == "open" and time.monotonic() < deadline:
                        time.sleep(min(1.5, max(deadline - time.monotonic(), 0.05)))
                        order_info = self.client._request("GET", f"/v2/orders/client_order_id/{pos['client_order_id']}")["result"]
                    if order_info.get("state") == "open":
                        try:
                            self.client.cancel_order(meta["id"], int(order_info["id"]))
                        except DeltaAPIError as cancel_err:
                            if cancel_err.status not in {400, 404}:
                                raise
                        order_info = self.client._request("GET", f"/v2/orders/client_order_id/{pos['client_order_id']}")["result"]
                    maker_filled = int(order_info["size"]) - int(order_info["unfilled_size"])
                    if maker_filled == 0 and order_info.get("state") in {"cancelled", "closed"}:
                        log.info("%s: maker limit unfilled after %.1fs; falling back to collared IOC", sym, self.cfg.maker_timeout_sec)
                        need_ioc_fallback = True

            if need_ioc_fallback:
                if pos.get("entry_mode") == "maker":
                    pos["entry_mode"] = "ioc"
                    pos["client_order_id"] = uuid4().hex
                    self._save_state()
                try:
                    self.client.submit_entry(meta["id"], pos)
                except DeltaAPIError as error:
                    # Only recognized, explicit business rejections prove no order was accepted.
                    if error.status == 400 and error.code in reject_codes:
                        del self.state["positions"][sym]
                        self._save_state()
                    raise
            self._reconcile(sym, meta)
            if "sl_order_id" not in pos:
                return  # Verified zero-fill IOC.
        msg = (f"ENTRY {sym} {side} ({pos.get('entry_mode', 'ioc').upper()}) | Contracts={pos['contracts']} | "
               f"Fill={pos['entry_price']:.2f} | SL={pos['sl_price']} (DRY_RUN={self.cfg.dry_run})")
        log.info(msg)
        send_telegram(self.cfg, msg)

    def _record_exit(self, sym: str, pos: dict, exit_px: float, fees: float, at: datetime) -> None:
        sign = 1 if pos["side"] == "LONG" else -1
        pnl_usd = (exit_px - pos["entry_price"]) * sign * pos["contracts"] * pos["contract_value"] - fees
        self.state["equity_usd"] += pnl_usd
        for period, key, counter in (("day", at.strftime("%Y-%m-%d"), "daily_pnl_usd"),
                                     ("week", at.strftime("%G-W%V"), "weekly_pnl_usd")):
            if key == self.state[f"last_{period}"] or pnl_usd < 0:
                self.state[counter] += pnl_usd
        del self.state["positions"][sym]
        self._update_drawdown_brake()
        self._save_state()
        self._log_journal({
            "timestamp": at.isoformat(), "symbol": sym, "side": pos["side"], "contracts": pos["contracts"],
            "entry_price": pos["entry_price"], "exit_price": exit_px, "fees_usd": fees,
            "pnl_usd": round(pnl_usd, 2), "equity_usd": round(self.state["equity_usd"], 2),
        })
        pnl_note = "net of fees & est. funding" if self.cfg.dry_run else "net of fees"
        msg = f"EXIT {sym} {pos['side']} | Fill={exit_px:.2f} | PnL=${pnl_usd:+.2f} ({pnl_note})"
        log.info(msg)
        send_telegram(self.cfg, msg)

    @staticmethod
    def _stop_price(price: float, tick: Decimal, sign: int) -> str:
        ticks = (Decimal(str(price)) / tick).to_integral_value(rounding=ROUND_CEILING if sign > 0 else ROUND_FLOOR)
        return format(ticks * tick, "f")

    def _manage_position(self, sym: str, meta: dict, pos: dict, mark_px: float, closed_px: float, atr: float, history, period) -> None:
        sign = 1 if pos["side"] == "LONG" else -1
        old_sl = float(pos["sl_price"])
        if self.cfg.dry_run:
            now = pd.Timestamp.now(tz="UTC")
            checked = pd.Timestamp(pos.get("paper_checked_at", now))
            bars = history.loc[history.index >= checked]
            hit = bars["low"].le(old_sl) if sign > 0 else bars["high"].ge(old_sl)
            if hit.any() or (mark_px - old_sl) * sign <= 0:
                if hit.any():
                    bar = bars.loc[hit].iloc[0]
                    trigger = min(old_sl, bar.open) if sign > 0 else max(old_sl, bar.open)
                    exit_at = bar.name + period
                else:
                    trigger, exit_at = mark_px, now
                exit_px = trigger * (1 - sign * self.cfg.entry_slippage_pct)
                qty = pos["contracts"] * pos["contract_value"]
                fees = pos.get("entry_fee", 0) + exit_px * qty * meta["fee_rate"] + pos.get("paper_funding", 0)
                self._record_exit(sym, pos, exit_px, fees, exit_at.to_pydatetime())
                return
            if not bars.empty:
                pos["paper_checked_at"] = (bars.index[-1] + period).isoformat()
            last_funding = pd.Timestamp(pos.setdefault("paper_funding_at", now.isoformat())).floor("8h")
            funding_at = now.floor("8h")
            if funding_at > last_funding:
                periods = (funding_at - last_funding) // pd.Timedelta(hours=8)
                pos["paper_funding"] = pos.get("paper_funding", 0) + periods * mark_px * pos["contracts"] * pos["contract_value"] * .0001
                pos["paper_funding_at"] = funding_at.isoformat()

        new_sl = self._stop_price(closed_px - sign * self.cfg.atr_trail_mult * atr, meta["tick_size"], sign)
        if float(new_sl) > 0 and (float(new_sl) - old_sl) * sign > 0 and (mark_px - float(new_sl)) * sign > 0:
            if self.cfg.dry_run:
                pos["sl_price"] = new_sl
                self._save_state()
            else:
                self.client.edit_stop(meta["id"], pos["sl_order_id"], pos["contracts"], new_sl)
                self._reconcile(sym, meta)
                if sym not in self.state["positions"]:
                    return  # Stop filled while the edit was in flight.
                if (Decimal(pos["sl_price"]) - Decimal(new_sl)) * sign < 0:
                    raise RuntimeError(f"{sym}: stop edit not yet reflected at the exchange")
            msg = f"TRAIL SL {sym} {pos['side']} | {old_sl} -> {pos['sl_price']} (Mark={mark_px:.2f})"
            log.info(msg)
            send_telegram(self.cfg, msg)
        else:
            log.info(f"HOLD {sym} {pos['side']} | Mark={mark_px:.2f} | SL={old_sl} | Unchanged.")

        self._maybe_pyramid(sym, meta, pos, mark_px)

    def _maybe_pyramid(self, sym: str, meta: dict, pos: dict, mark_px: float) -> None:
        if (
            self.cfg.pyramid_add_fraction <= 0
            or pos.get("pyramided")
            or "pending_pyramid" in pos
            or self.state.get("halt_new_entries")
            or self._circuit_breaker_tripped()
        ):
            return
        init_entry = float(pos.get("init_entry_price", pos["entry_price"]))
        init_sl_dist = float(pos.get("init_sl_dist", 0.0))
        if init_sl_dist <= 0:
            return
        sign = 1 if pos["side"] == "LONG" else -1
        cur_sl = float(pos["sl_price"])
        be_ok = (cur_sl - init_entry) * sign >= 0
        move = (mark_px - init_entry) * sign
        if not be_ok or move < self.cfg.pyramid_trigger_r * init_sl_dist:
            return

        allocation = min(self.state["equity_usd"], self.cfg.capital_usd)
        allocated_margin = sum(p["contracts"] * p["contract_value"] * p["entry_price"] / self.cfg.leverage
                               for p in self.state["positions"].values())
        max_util = self.cfg.max_margin_utilization
        margin_budget = max_util * allocation - allocated_margin
        if not self.cfg.dry_run:
            _, available = self.client.wallet()
            margin_budget = min(margin_budget, available - (1.0 - max_util) * allocation)
        if margin_budget <= 0:
            return

        slip = self.cfg.entry_slippage_pct
        add_limit = self._stop_price(mark_px * (1 + sign * slip), meta["tick_size"], -sign)
        margin_price = max(mark_px * (1 + slip), float(add_limit))
        contract_val = meta["contract_value"]
        max_by_margin = math.floor(margin_budget * self.cfg.leverage / (margin_price * contract_val))
        notional_cap = meta["max_leverage_notional"]
        if math.isfinite(notional_cap):
            max_by_notional = math.floor(notional_cap / (margin_price * contract_val)) - pos["contracts"]
            add_contracts = min(math.floor(pos["contracts"] * self.cfg.pyramid_add_fraction), max_by_margin, max_by_notional)
        else:
            add_contracts = min(math.floor(pos["contracts"] * self.cfg.pyramid_add_fraction), max_by_margin)
        if add_contracts < 1:
            return

        if self.cfg.dry_run:
            fill_px = float(add_limit)
            new_total = pos["contracts"] + add_contracts
            pos["entry_price"] = (pos["entry_price"] * pos["contracts"] + fill_px * add_contracts) / new_total
            pos["contracts"] = new_total
            pos["entry_fee"] = pos.get("entry_fee", 0.0) + fill_px * add_contracts * contract_val * meta["fee_rate"]
            pos["pyramided"] = True
            self._save_state()
        else:
            pyr_cid = uuid4().hex
            pos["pending_pyramid"] = {
                "client_order_id": pyr_cid,
                "contracts": add_contracts,
                "limit_price": add_limit,
            }
            self._save_state()
            try:
                self.client.submit_pyramid_add(meta["id"], pos["side"], add_contracts, add_limit, pyr_cid)
            except DeltaAPIError as error:
                if error.status == 400 and error.code in {
                    "insufficient_margin", "insufficient_commission", "order_size_exceed_available",
                    "order_size_not_available_in_orderbook", "risk_limits_breached", "immediate_liquidation",
                    "less_than_order_min_size", "invalid_order",
                }:
                    del pos["pending_pyramid"]
                    self._save_state()
                    log.warning("%s: pyramid add rejected (%s); keeping original position protected", sym, error.code)
                    return
                raise
            self._reconcile(sym, meta)
            if not pos.get("pyramided"):
                return

        msg = (f"PYRAMID {sym} {pos['side']} | +{add_contracts} -> {pos['contracts']} contracts | "
               f"AvgEntry={pos['entry_price']:.2f} | SL={pos['sl_price']} (DRY_RUN={self.cfg.dry_run})")
        log.info(msg)
        send_telegram(self.cfg, msg)


def seconds_until_next_bar_close(resolution: str = "4h", now: Optional[datetime] = None) -> float:
    """Returns seconds until the next UTC candle close (+15s buffer) for 1h/2h/4h/6h/12h/1d."""
    now = now or datetime.now(timezone.utc)
    step_hours = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}.get(resolution.lower())
    if step_hours is None:
        raise ValueError(f"Unsupported resolution for scheduling: {resolution}")
    next_hour = ((now.hour // step_hours) + 1) * step_hours
    if next_hour >= 24:
        next_dt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=15, microsecond=0)
    else:
        next_dt = now.replace(hour=next_hour, minute=0, second=15, microsecond=0)
    return max((next_dt - now).total_seconds(), 15.0)

def seconds_until_next_4h_close() -> float:
    return seconds_until_next_bar_close("4h")


def main():
    parser = argparse.ArgumentParser(description="Delta Donchian Bot")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--resume", action="store_true", help="Verify flat reconciled account and clear halt_new_entries")
    parser.add_argument("--setup-account", action="store_true", help="Configure isolated margin, disable auto top-up/DETO, and set 5x leverage on all symbols (testnet only)")
    args = parser.parse_args()
    cfg = DeltaLiveConfig()
    if not cfg.dry_run and (not cfg.api_key or not cfg.api_secret):
        raise ValueError("DELTA_API_KEY and DELTA_API_SECRET required when DRY_RUN=false")

    Path(cfg.state_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            RotatingFileHandler(cfg.log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    stop_event = Event()

    def _handle_stop(signum, _frame):
        log.info("Received signal %s; shutting down gracefully...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    with open(cfg.state_file + ".lock", "a", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = LiveDonchianEngine(cfg)
        start_msg = f"Starting Delta {cfg.resolution.upper()} Donchian Bot | ENV={cfg.env} | DRY_RUN={cfg.dry_run} | Symbols={cfg.symbols}"
        log.info(start_msg)
        if args.setup_account:
            summary = engine.setup_account()
            log.info("Delta testnet account setup complete: %s", summary)
            return
        if args.resume:
            engine.run_cycle(resume=True)
            log.info("Account verified flat and reconciled; halt_new_entries cleared.")
            return
        if args.once:
            engine.run_cycle()
            return

        send_telegram(cfg, start_msg)
        last_error = None
        err_count = 0
        stale_retries = 0
        while not stop_event.is_set():
            try:
                retry = engine.run_cycle()
                last_error = None
                err_count = 0
                if retry and stale_retries < 4:
                    stale_retries += 1
                    sleep_s = 20.0
                else:
                    if retry:
                        send_telegram(cfg, f"BOT WARNING: latest {cfg.resolution} candle still missing after {stale_retries} retries")
                    stale_retries = 0
                    sleep_s = seconds_until_next_bar_close(cfg.resolution)
            except Exception as e:
                log.exception("Cycle failed")
                err_count += 1
                stale_retries = 0
                if str(e) != last_error:
                    send_telegram(cfg, f"BOT ERROR: {e}")
                last_error = str(e)
                sleep_s = min(15.0 * (2 ** min(err_count - 1, 5)), 300.0)
            stop_event.wait(sleep_s)
        send_telegram(cfg, f"Stopped Delta {cfg.resolution.upper()} Donchian Bot cleanly")


if __name__ == "__main__":
    main()
