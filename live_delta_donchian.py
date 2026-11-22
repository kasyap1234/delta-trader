"""
REAL LIVE / PAPER EXECUTION BOT — ENGINE A (4H Donchian Trend + 2.5x ATR Trail)
Broker: Delta Exchange India (Testnet & Production REST API v2)

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

import csv
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
import requests

from config import DeltaLiveConfig
from data_loader import load_delta_candles
from portfolio_engine import add_donchian_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("delta_live_bot.log"), logging.StreamHandler()],
)
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
    except Exception as e:
        log.warning(f"Telegram alert failed: {e}")


class DeltaExchangeClient:
    """Lightweight, zero-dependency REST v2 client for Delta Exchange India (Prod & Testnet)."""

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
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        payload = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers = self._sign(method, path, query_string, payload) if auth else {"User-Agent": "delta-donchian-4h-bot/1.0"}

        url = f"{self.base_url}{path}{query_string}"
        r = requests.request(method, url, headers=headers, data=payload if body is not None else None, timeout=15)
        data = r.json()
        if not r.ok or not data.get("success", False):
            raise RuntimeError(f"Delta API error ({r.status_code}) on {path}: {data}")
        return data

    def load_products(self) -> None:
        resp = self._request("GET", "/v2/products", auth=False)
        for p in resp.get("result", []):
            sym = p.get("symbol")
            if sym in self.cfg.symbols:
                self.products_cache[sym] = {
                    "id": int(p["id"]),
                    "contract_value": float(p.get("contract_value", 1.0)),
                    "tick_size": float(p.get("tick_size", 0.01)),
                }
        log.info(f"Loaded Delta products ({self.cfg.env}): {self.products_cache}")

    def get_ticker_mark_price(self, symbol: str) -> float:
        resp = self._request("GET", f"/v2/tickers/{symbol}", auth=False)
        res = resp["result"]
        return float(res.get("mark_price") or res.get("close"))

    def set_leverage(self, product_id: int, leverage: int) -> None:
        if self.cfg.dry_run:
            log.info(f"[DRY RUN] Set leverage product_id={product_id} -> {leverage}x")
            return
        self._request("POST", f"/v2/products/{product_id}/orders/leverage", body={"leverage": str(leverage)})

    def get_open_position_size(self, product_id: int) -> int:
        if self.cfg.dry_run:
            return 0
        resp = self._request("GET", "/v2/positions", params={"product_id": product_id})
        res = resp.get("result") or {}
        return int(res.get("size", 0))

    def place_market_order(self, product_id: int, side: str, size: int, reduce_only: bool = False) -> str:
        if self.cfg.dry_run:
            oid = f"DRY_MKT_{int(time.time())}"
            log.info(f"[DRY RUN] MKT {side.upper()} {size} contracts (product_id={product_id}, reduce_only={reduce_only}) -> {oid}")
            return oid
        body = {
            "product_id": product_id,
            "size": size,
            "side": side.lower(),
            "order_type": "market_order",
            "reduce_only": reduce_only,
        }
        resp = self._request("POST", "/v2/orders", body=body)
        return str(resp["result"]["id"])

    def place_stop_market_order(self, product_id: int, side: str, size: int, stop_price: float) -> str:
        if self.cfg.dry_run:
            oid = f"DRY_SL_{int(time.time())}"
            log.info(f"[DRY RUN] STOP-MKT {side.upper()} {size} contracts @ {stop_price:.4f} (product_id={product_id}) -> {oid}")
            return oid
        body = {
            "product_id": product_id,
            "size": size,
            "side": side.lower(),
            "order_type": "market_order",
            "stop_order_type": "stop_loss_order",
            "stop_price": str(round(stop_price, 4)),
            "reduce_only": True,
        }
        resp = self._request("POST", "/v2/orders", body=body)
        return str(resp["result"]["id"])

    def cancel_order(self, product_id: int, order_id: str) -> None:
        if self.cfg.dry_run:
            log.info(f"[DRY RUN] Cancel order {order_id} (product_id={product_id})")
            return
        try:
            self._request("DELETE", "/v2/orders", body={"id": int(order_id), "product_id": product_id})
        except Exception as e:
            log.warning(f"Cancel order {order_id} warning: {e}")


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
            "daily_pnl_usd": 0.0,
            "weekly_pnl_usd": 0.0,
            "last_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "last_week": datetime.now(timezone.utc).strftime("%Y-W%W"),
        }

    def _save_state(self) -> None:
        with open(self.cfg.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

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
        if self.state["daily_pnl_usd"] <= -(eq * self.cfg.daily_loss_limit_pct):
            log.warning("DAILY CIRCUIT BREAKER TRIPPED (-3%). Pausing new entries until next UTC day.")
            return True
        if self.state["weekly_pnl_usd"] <= -(eq * self.cfg.weekly_loss_limit_pct):
            log.warning("WEEKLY CIRCUIT BREAKER TRIPPED (-7%). Pausing new entries until next UTC week.")
            return True
        return False

    def run_cycle(self) -> None:
        self._reset_periodic_counters()
        if not self.client.products_cache:
            self.client.load_products()
            for sym, meta in self.client.products_cache.items():
                self.client.set_leverage(meta["id"], self.cfg.leverage)

        for sym in self.cfg.symbols:
            meta = self.client.products_cache.get(sym)
            if not meta:
                log.error(f"Product metadata not found for {sym}")
                continue

            # Fetch ~60 days of 4H candles (~360 bars) -> guarantees 200-bar EMA warmup
            raw_df = load_delta_candles(sym, resolution=self.cfg.resolution, days=60, base_url=self.cfg.base_url)
            feat_df = add_donchian_features(raw_df, entry_len=self.cfg.donchian_entry_len, warmup_bars=self.cfg.ema_slow)
            if len(feat_df) < 3:
                log.warning(f"Insufficient warmed-up 4H bars for {sym} ({len(feat_df)}). Skipping.")
                continue

            # Use the last CLOSED 4H bar (-2) to avoid repainting
            closed = feat_df.iloc[-2]
            mark_px = self.client.get_ticker_mark_price(sym)
            atr = float(closed["atr14"])

            pos = self.state["positions"].get(sym)

            # 1. Manage existing position
            if pos is not None:
                self._manage_position(sym, meta, pos, mark_px, float(closed["close"]), atr)
                continue

            # 2. Check fresh entry if circuit breaker is clear
            if self._circuit_breaker_tripped():
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
                self._enter_position(sym, meta, side, mark_px, atr)
            else:
                log.info(f"{sym} 4H | Close={closed['close']:.2f} | EMA50={closed['ema50']:.2f} | EMA200={closed['ema200']:.2f} | No signal.")

        self._save_state()

    def _enter_position(self, sym: str, meta: dict, side: str, entry_px: float, atr: float) -> None:
        sl_dist = self.cfg.atr_trail_mult * atr
        risk_usd = self.state["equity_usd"] * self.cfg.risk_per_trade_pct
        contract_val = meta["contract_value"]

        contracts = math.floor(risk_usd / (sl_dist * contract_val))
        if contracts < 1:
            log.warning(f"{sym}: Calculated contracts < 1 (risk=${risk_usd:.2f}, sl_dist={sl_dist:.2f}). Skipping.")
            return

        sl_price = (entry_px - sl_dist) if side == "LONG" else (entry_px + sl_dist)
        entry_side = "buy" if side == "LONG" else "sell"
        exit_side = "sell" if side == "LONG" else "buy"

        self.client.place_market_order(meta["id"], entry_side, contracts, reduce_only=False)
        sl_oid = self.client.place_stop_market_order(meta["id"], exit_side, contracts, sl_price)

        self.state["positions"][sym] = {
            "side": side,
            "contracts": contracts,
            "contract_value": contract_val,
            "entry_price": entry_px,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "sl_price": sl_price,
            "sl_order_id": sl_oid,
            "risk_usd": contracts * sl_dist * contract_val,
            "risk_pct": self.cfg.risk_per_trade_pct,
        }
        msg = f"ENTRY {sym} {side} | Contracts={contracts} | Entry~{entry_px:.2f} | SL={sl_price:.2f} | ATR={atr:.2f} (DRY_RUN={self.cfg.dry_run})"
        log.info(msg)
        send_telegram(self.cfg, msg)

    def _manage_position(self, sym: str, meta: dict, pos: dict, mark_px: float, closed_px: float, atr: float) -> None:
        is_long = pos["side"] == "LONG"
        exit_side = "sell" if is_long else "buy"

        # Check if stop was hit (simulated in DRY_RUN, or verified on exchange in live mode)
        stopped_out = False
        if self.cfg.dry_run:
            stopped_out = (is_long and mark_px <= pos["sl_price"]) or ((not is_long) and mark_px >= pos["sl_price"])
        else:
            live_size = self.client.get_open_position_size(meta["id"])
            if live_size == 0:
                stopped_out = True

        if stopped_out:
            exit_px = mark_px
            pnl_usd = ((exit_px - pos["entry_price"]) if is_long else (pos["entry_price"] - exit_px)) * pos["contracts"] * pos["contract_value"]
            self.state["equity_usd"] += pnl_usd
            self.state["daily_pnl_usd"] += pnl_usd
            self.state["weekly_pnl_usd"] += pnl_usd
            self._log_journal({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": sym,
                "side": pos["side"],
                "contracts": pos["contracts"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_px,
                "pnl_usd": round(pnl_usd, 2),
                "equity_usd": round(self.state["equity_usd"], 2),
            })
            del self.state["positions"][sym]
            msg = f"EXIT {sym} {pos['side']} | Exit~{exit_px:.2f} | PnL=${pnl_usd:+.2f} | Equity=${self.state['equity_usd']:.2f}"
            log.info(msg)
            send_telegram(self.cfg, msg)
            return

        # Ratchet 2.5x ATR trailing stop (NEVER widens)
        trail_dist = self.cfg.atr_trail_mult * atr
        new_sl = (closed_px - trail_dist) if is_long else (closed_px + trail_dist)
        should_ratchet = (new_sl > pos["sl_price"]) if is_long else (new_sl < pos["sl_price"])

        if should_ratchet:
            old_sl = pos["sl_price"]
            self.client.cancel_order(meta["id"], pos["sl_order_id"])
            new_oid = self.client.place_stop_market_order(meta["id"], exit_side, pos["contracts"], new_sl)
            pos["sl_price"] = new_sl
            pos["sl_order_id"] = new_oid
            msg = f"TRAIL SL {sym} {pos['side']} | {old_sl:.2f} -> {new_sl:.2f} (Mark={mark_px:.2f})"
            log.info(msg)
            send_telegram(self.cfg, msg)
        else:
            log.info(f"HOLD {sym} {pos['side']} | Mark={mark_px:.2f} | SL={pos['sl_price']:.2f} | Unchanged.")


def seconds_until_next_4h_close() -> float:
    now = datetime.now(timezone.utc)
    next_hour = ((now.hour // 4) + 1) * 4
    if next_hour >= 24:
        next_dt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=15, microsecond=0)
    else:
        next_dt = now.replace(hour=next_hour, minute=0, second=15, microsecond=0)
    return max((next_dt - now).total_seconds(), 15.0)


def main():
    cfg = DeltaLiveConfig()
    if not cfg.dry_run and (not cfg.api_key or not cfg.api_secret):
        raise ValueError("DELTA_API_KEY and DELTA_API_SECRET required when DRY_RUN=false")

    engine = LiveDonchianEngine(cfg)
    log.info(f"Starting Delta 4H Donchian Bot | ENV={cfg.env} | DRY_RUN={cfg.dry_run} | Symbols={cfg.symbols}")

    # Run once immediately on startup, then sync strictly to every 4H UTC candle close (+15s)
    while True:
        try:
            engine.run_cycle()
        except Exception as e:
            log.exception(f"Cycle error: {e}")
            send_telegram(cfg, f"BOT ERROR: {e}")

        sleep_s = seconds_until_next_4h_close()
        log.info(f"Sleeping {sleep_s/60:.1f} minutes until next 4H bar close...")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
