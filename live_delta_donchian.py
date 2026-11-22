"""
REAL LIVE / PAPER EXECUTION BOT — ENGINE A (4H Donchian Trend + 2.5x ATR Trail)
Broker: Delta Exchange India (testnet orders, public production data only)

Risk Metrics (Locked to Test 2 Winner):
  - Timeframe: 4H (evaluated on every closed 4H candle)
  - Assets: ETHUSD, SOLUSD perpetual futures
  - Risk per Trade: 1.0% of account equity
  - Max Concurrent Open Risk: 3.0% (Max 2 simultaneous crypto positions)
  - Leverage: 5x Isolated (keeps >= 40% free margin buffer)
  - Entry: EMA(50) > EMA(200) & Close > 20-bar Donchian High (Long); inverse for Short
  - Exit: Immediate Reduce-Only Stop-Market at 2.5x ATR(14), trailed every 4H bar (never widened)
  - Safety: DRY_RUN=true by default, Testnet default, Daily (-3%) & Weekly (-7%) circuit breakers,
            crash-safe JSON state persistence, and Telegram notifications.
"""

import argparse
import csv
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from config import DeltaLiveConfig
from data_loader import load_delta_candles
from portfolio_engine import add_donchian_features

log = logging.getLogger("DeltaDonchianBot")


def send_telegram(cfg: DeltaLiveConfig, text: str) -> None:
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage",
            json={"chat_id": cfg.telegram_chat_id, "text": text},
            timeout=10,
        )
    except requests.RequestException:
        log.warning("Telegram alert failed")


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
        if method != "GET" and (self.cfg.dry_run or (self.cfg.env != "testnet" and not self.cfg.allow_prod_orders)):
            raise RuntimeError("Delta writes are restricted to testnet order mode unless ALLOW_PROD_ORDERS=true")
        query_string = "?" + urlencode(params, safe=",") if params else ""
        payload = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers = self._sign(method, path, query_string, payload) if auth else {"User-Agent": "delta-donchian-4h-bot/1.0"}

        url = f"{self.base_url}{path}{query_string}"
        r = requests.request(method, url, headers=headers, data=payload if body is not None else None,
                             timeout=15, allow_redirects=False)
        data = r.json()
        if not r.ok or not data.get("success", False):
            raise RuntimeError(f"Delta API error ({r.status_code}) on {method} {path}")
        return data

    def load_products(self) -> None:
        resp = self._request("GET", "/v2/products", auth=False)
        products = {}
        for p in resp.get("result", []):
            sym = p.get("symbol")
            if sym in self.cfg.symbols:
                if p["notional_type"] != "vanilla" or any(p[k]["symbol"] != "USD" for k in ("quoting_asset", "settling_asset")):
                    raise RuntimeError(f"Unsupported contract denomination for {sym}")
                products[sym] = {
                    "id": int(p["id"]),
                    "contract_value": float(p["contract_value"]),
                    "tick_size": Decimal(p["tick_size"]),
                }
        if set(products) != set(self.cfg.symbols):
            raise RuntimeError("Delta did not return all configured products")
        self.products_cache = products
        log.info(f"Loaded Delta products ({self.cfg.env}): {self.products_cache}")

    def get_ticker_mark_price(self, symbol: str) -> float:
        resp = self._request("GET", f"/v2/tickers/{symbol}", auth=False)
        price = float(resp["result"]["mark_price"])
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError(f"Invalid mark price for {symbol}")
        return price

    def position(self, product_id: int) -> dict:
        return self._request("GET", "/v2/positions", params={"product_id": product_id})["result"]

    def open_orders(self, product_id: int) -> list:
        resp = self._request("GET", "/v2/orders", params={"product_ids": product_id, "states": "open,pending"})
        if resp.get("meta", {}).get("after"):
            raise RuntimeError(f"Unexpected open order count for product {product_id}")
        return resp["result"]

    def wallet_for_entry(self, product_id: int) -> tuple[float, float]:
        prefs = self._request("GET", "/v2/users/trading_preferences")["result"]
        if prefs["default_auto_topup"] or prefs["deto_for_commission"]:
            raise RuntimeError("Disable auto top-up and DETO commission on the dedicated testnet account")
        leverage = self._request("GET", f"/v2/products/{product_id}/orders/leverage")["result"]["leverage"]
        if int(leverage) != self.cfg.leverage:
            raise RuntimeError(f"Set testnet product {product_id} to {self.cfg.leverage}x isolated leverage before starting")
        wallets = self._request("GET", "/v2/wallet/balances")["result"]
        wallet = next(w for w in wallets if w["asset_symbol"] == "USD")
        balance, available = float(wallet["balance"]), float(wallet["available_balance"])
        if not all(math.isfinite(x) for x in (balance, available)):
            raise RuntimeError("Invalid USD wallet balance")
        return balance, available

    def submit_entry(self, product_id: int, pos: dict) -> None:
        self._request("POST", "/v2/orders", body={
            "product_id": product_id, "size": pos["contracts"], "side": "buy" if pos["side"] == "LONG" else "sell",
            "order_type": "market_order", "time_in_force": "ioc", "client_order_id": pos["client_order_id"],
            "bracket_stop_trigger_method": "mark_price", "bracket_stop_loss_price": pos["sl_price"],
        })

    def edit_stop(self, product_id: int, order_id: int, stop_price: str) -> None:
        self._request("PUT", "/v2/orders", body={"id": order_id, "product_id": product_id, "stop_price": stop_price})

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
                return json.load(f)
        return {
            "equity_usd": self.cfg.capital_usd,
            "positions": {},
            "last_entry_bar": {},
            "daily_pnl_usd": 0.0,
            "weekly_pnl_usd": 0.0,
            "last_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "last_week": datetime.now(timezone.utc).strftime("%Y-W%W"),
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
        cur_day = now.strftime("%Y-%m-%d")
        cur_week = now.strftime("%Y-W%W")
        if self.state.get("last_day") != cur_day:
            self.state["daily_pnl_usd"] = 0.0
            self.state["last_day"] = cur_day
        if self.state.get("last_week") != cur_week:
            self.state["weekly_pnl_usd"] = 0.0
            self.state["last_week"] = cur_week

    def _circuit_breaker_tripped(self) -> bool:
        eq = self.state["equity_usd"]
        if eq <= 0:
            return True
        if self.state["daily_pnl_usd"] <= -eq * self.cfg.daily_loss_limit_pct + 1e-9:
            log.warning("DAILY CIRCUIT BREAKER TRIPPED (-3%). Pausing new entries until next UTC day.")
            return True
        if self.state["weekly_pnl_usd"] <= -eq * self.cfg.weekly_loss_limit_pct + 1e-9:
            log.warning("WEEKLY CIRCUIT BREAKER TRIPPED (-7%). Pausing new entries until next UTC week.")
            return True
        return False

    def _reconcile(self, sym: str, meta: dict) -> None:
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
        elif not size and not orders:
            order_path = (f"client_order_id/{pos['emergency_close_id']}" if "emergency_close_id" in pos
                          else str(pos["sl_order_id"]))
            stop_fill = self.client._request("GET", f"/v2/orders/{order_path}")["result"]
            exit_px = float(stop_fill["average_fill_price"] or 0)
            exit_fee = float(stop_fill["paid_commission"])
            if (stop_fill["state"] != "closed" or int(stop_fill["size"]) - int(stop_fill["unfilled_size"]) != pos["contracts"]
                    or not math.isfinite(exit_px) or exit_px <= 0 or not math.isfinite(exit_fee)):
                raise RuntimeError(f"{sym}: cannot verify closing fill; reconcile manually")
            at = datetime.fromtimestamp(int(stop_fill["updated_at"]) / 1_000_000, timezone.utc)
            self._record_exit(sym, pos, exit_px, pos["entry_fee"] + exit_fee, at)
            return
        elif size != sign * pos["contracts"]:
            raise RuntimeError(f"{sym}: exchange size changed unexpectedly; reconcile manually")

        if not orders:
            if int(self.client.position(product)["size"]) != size:
                raise RuntimeError(f"{sym}: position changed during reconciliation; retrying without an order")
            if "sl_order_id" not in pos:
                pos.update(contracts=abs(size), entry_price=fill_price, entry_fee=entry_fee)
            # Only close a position whose side and size were reconciled above. Never touch an unknown position.
            pos["emergency_close_id"] = uuid4().hex
            self.state["halt_new_entries"] = True
            self._save_state()
            log.error("%s: no protective stop visible; submitting one reduce-only emergency close", sym)
            self.client.emergency_close(product, size, pos["emergency_close_id"])
            raise RuntimeError(f"{sym}: emergency close submitted; verify the fill and account before resuming")

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
            pos.update(contracts=abs(size), entry_price=fill_price, entry_fee=entry_fee, sl_order_id=int(stop["id"]),
                       risk_usd=risk_usd, risk_pct=risk_usd / self.state["equity_usd"])
            log.info("%s: reconciled %s contracts with verified stop %s", sym, size, stop["id"])
        pos["sl_price"] = format(stop_px, "f")
        self._save_state()

    def run_cycle(self) -> None:
        self._reset_periodic_counters()
        if not self.client.products_cache:
            self.client.load_products()
        if not self.cfg.dry_run:
            for sym in self.cfg.symbols:
                self._reconcile(sym, self.client.products_cache[sym])

        for sym in self.cfg.symbols:
            meta = self.client.products_cache[sym]
            raw_df = load_delta_candles(sym, resolution=self.cfg.resolution, days=60, base_url=self.cfg.base_url)
            period = pd.Timedelta(self.cfg.resolution)
            expected = pd.Timestamp.now(tz="UTC").floor(self.cfg.resolution) - period
            history = raw_df.loc[raw_df.index <= expected]
            required = self.cfg.ema_slow + 1
            recent = history.tail(required)
            if (len(recent) < required or recent.index[-1] != expected
                    or not (recent.index.to_series().diff().iloc[1:] == period).all()
                    or not np.isfinite(recent[["open", "high", "low", "close", "volume"]]).to_numpy().all()):
                log.warning("%s: stale, missing or invalid closed candles; skipping", sym)
                continue

            features = add_donchian_features(history, entry_len=self.cfg.donchian_entry_len, warmup_bars=self.cfg.ema_slow)
            closed = features.loc[expected]
            mark_px = self.client.get_ticker_mark_price(sym)
            atr = float(closed["atr14"])

            pos = self.state["positions"].get(sym)

            # 1. Manage existing position
            if pos is not None:
                self._manage_position(sym, meta, pos, mark_px, float(closed["close"]), atr)
                continue

            # 2. Check fresh entry if circuit breaker is clear
            if (self.state.get("halt_new_entries") or self._circuit_breaker_tripped()
                    or self.state["last_entry_bar"].get(sym) == expected.isoformat()):
                continue

            open_positions = self.state["positions"]
            if len(open_positions) >= self.cfg.max_crypto_concurrent:
                log.info(f"{sym}: Max concurrent crypto positions ({self.cfg.max_crypto_concurrent}) reached.")
                continue

            open_risk_pct = sum(p["risk_pct"] for p in open_positions.values())
            if (open_risk_pct + self.cfg.risk_per_trade_pct) > (self.cfg.max_concurrent_risk_pct + 1e-9):
                log.info(f"{sym}: Max portfolio risk cap ({self.cfg.max_concurrent_risk_pct*100:.1f}%) reached.")
                continue

            side = None
            if closed["ema50"] > closed["ema200"] and closed["close"] > closed["donchian_high"]:
                side = "LONG"
            elif closed["ema50"] < closed["ema200"] and closed["close"] < closed["donchian_low"]:
                side = "SHORT"

            if side:
                self._enter_position(sym, meta, side, mark_px, atr, expected.isoformat())
            else:
                log.info(f"{sym} {self.cfg.resolution.upper()} | Close={closed['close']:.2f} | EMA50={closed['ema50']:.2f} | EMA200={closed['ema200']:.2f} | No signal.")

        self._save_state()

    def _enter_position(self, sym: str, meta: dict, side: str, entry_px: float, atr: float, bar: str) -> None:
        sl_dist = self.cfg.atr_trail_mult * atr
        if not math.isfinite(sl_dist) or sl_dist <= 0:
            return
        sign = 1 if side == "LONG" else -1
        sl_price = self._stop_price(entry_px - sign * sl_dist, meta["tick_size"], sign)
        if not (float(sl_price) > 0 and (entry_px - float(sl_price)) * sign > 0):
            return

        contract_val = meta["contract_value"]
        allocation = self.state["equity_usd"]
        if not self.cfg.dry_run:
            balance, available = self.client.wallet_for_entry(meta["id"])
            allocation = min(allocation, balance)
        allocated_margin = sum(p["contracts"] * p["contract_value"] * p["entry_price"] / self.cfg.leverage
                               for p in self.state["positions"].values())
        margin_budget = .6 * allocation - allocated_margin
        if not self.cfg.dry_run:
            margin_budget = min(margin_budget, available - .4 * allocation)
        risk_usd = allocation * self.cfg.risk_per_trade_pct
        contracts = math.floor(min(risk_usd / (sl_dist * contract_val),
                                   margin_budget * self.cfg.leverage / (entry_px * contract_val)))
        if contracts < 1:
            log.warning("%s: No contracts fit the risk and margin limits", sym)
            return

        pos = self.state["positions"][sym] = {
            "side": side,
            "contracts": contracts,
            "contract_value": contract_val,
            "entry_price": entry_px,
            "sl_price": sl_price,
            "risk_pct": self.cfg.risk_per_trade_pct,
            "client_order_id": uuid4().hex,
        }
        self.state["last_entry_bar"][sym] = bar
        # Persist the intent before the request. A missing or ambiguous response never authorizes a resubmission.
        self._save_state()
        if not self.cfg.dry_run:
            self.client.submit_entry(meta["id"], pos)
            self._reconcile(sym, meta)
            if "sl_order_id" not in pos:
                return  # Verified zero-fill IOC.
        msg = f"ENTRY {sym} {side} | Contracts={pos['contracts']} | Fill={pos['entry_price']:.2f} | SL={pos['sl_price']} (DRY_RUN={self.cfg.dry_run})"
        log.info(msg)
        send_telegram(self.cfg, msg)

    def _record_exit(self, sym: str, pos: dict, exit_px: float, fees: float, at: datetime) -> None:
        sign = 1 if pos["side"] == "LONG" else -1
        pnl_usd = (exit_px - pos["entry_price"]) * sign * pos["contracts"] * pos["contract_value"] - fees
        self.state["equity_usd"] += pnl_usd
        if at.strftime("%Y-%m-%d") == self.state["last_day"]:
            self.state["daily_pnl_usd"] += pnl_usd
        if at.strftime("%Y-W%W") == self.state["last_week"]:
            self.state["weekly_pnl_usd"] += pnl_usd
        del self.state["positions"][sym]
        self._save_state()
        self._log_journal({
            "timestamp": at.isoformat(), "symbol": sym, "side": pos["side"], "contracts": pos["contracts"],
            "entry_price": pos["entry_price"], "exit_price": exit_px, "fees_usd": fees,
            "pnl_usd": round(pnl_usd, 2), "equity_usd": round(self.state["equity_usd"], 2),
        })
        msg = f"EXIT {sym} {pos['side']} | Fill={exit_px:.2f} | PnL=${pnl_usd:+.2f} before funding"
        log.info(msg)
        send_telegram(self.cfg, msg)

    @staticmethod
    def _stop_price(price: float, tick: Decimal, sign: int) -> str:
        ticks = (Decimal(str(price)) / tick).to_integral_value(rounding=ROUND_CEILING if sign > 0 else ROUND_FLOOR)
        return format(ticks * tick, "f")

    def _manage_position(self, sym: str, meta: dict, pos: dict, mark_px: float, closed_px: float, atr: float) -> None:
        sign = 1 if pos["side"] == "LONG" else -1
        old_sl = float(pos["sl_price"])
        if self.cfg.dry_run and (mark_px - old_sl) * sign <= 0:
            self._record_exit(sym, pos, mark_px, 0.0, datetime.now(timezone.utc))
            return

        new_sl = self._stop_price(closed_px - sign * self.cfg.atr_trail_mult * atr, meta["tick_size"], sign)
        if float(new_sl) > 0 and (float(new_sl) - old_sl) * sign > 0 and (mark_px - float(new_sl)) * sign > 0:
            if self.cfg.dry_run:
                pos["sl_price"] = new_sl
                self._save_state()
            else:
                self.client.edit_stop(meta["id"], pos["sl_order_id"], new_sl)
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
    parser = argparse.ArgumentParser(description="Delta 4H Donchian Bot")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    args = parser.parse_args()
    cfg = DeltaLiveConfig()
    if not cfg.dry_run and (not cfg.api_key or not cfg.api_secret):
        raise ValueError("DELTA_API_KEY and DELTA_API_SECRET required when DRY_RUN=false")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler("delta_live_bot.log"), logging.StreamHandler()],
    )
    with open(cfg.state_file + ".lock", "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = LiveDonchianEngine(cfg)
        log.info(f"Starting Delta {cfg.resolution.upper()} Donchian Bot | ENV={cfg.env} | DRY_RUN={cfg.dry_run} | Symbols={cfg.symbols}")
        if args.once:
            engine.run_cycle()
            return
        last_error = None
        while True:
            try:
                engine.run_cycle()
                last_error = None
                sleep_s = seconds_until_next_bar_close(cfg.resolution)
            except Exception as e:
                log.exception("Cycle failed")
                if str(e) != last_error:
                    send_telegram(cfg, f"BOT ERROR: {e}")
                last_error = str(e)
                sleep_s = 15
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
