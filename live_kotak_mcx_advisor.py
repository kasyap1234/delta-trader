"""
KOTAK NEO MCX + NSE METALS DONCHIAN ADVISOR (Sleeve 2 — Rs 98,000)
Matches Config C (Max Aggression) for the INR commodity sleeve:
  - MCX GOLDPETAL (1g Gold Futures, Long + Short): 6H Donchian(20) + EMA(50/200) + 3.0x ATR(14) trail
  - NSE SILVERBEES.NS (Spot Silver ETF, Long-Only): 6H/1D Donchian(20) + EMA(50/200) + 3.0x ATR(14) trail
  - Risk: 2.5% per trade, 8.0% max concurrent risk, Winner Pyramiding (+0.5x @ +1.5R), 15% Drawdown Brake.

Generates exact whole-lot Kotak Neo order tickets + Telegram alerts + monthly MCX expiry rollover reminders.
Tracks manually confirmed fills in persistent JSON state (kotak_mcx_state.json).
"""

import argparse
import calendar
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

from config import KotakMcxConfig
from data_loader import load_yfinance
from portfolio_engine import add_donchian_features

log = logging.getLogger("KotakMcxAdvisor")

GRAMS_PER_TROY_OZ = 31.1035

ASSETS = {
    "GOLDPETAL": {
        "proxy": "GC=F",
        "broker_segment": "MCX",
        "long_only": False,
        "unit_label": "lot(s) (1g each)",
        "leverage": 8.0,
        "usd_oz_to_unit": 1.0 / GRAMS_PER_TROY_OZ,
    },
    "SILVERBEES.NS": {
        "proxy": "SI=F",
        "broker_segment": "NSE_EQ",
        "long_only": True,
        "unit_label": "share(s) (~1g silver)",
        "leverage": 1.0,
        "usd_oz_to_unit": 1.0 / GRAMS_PER_TROY_OZ,
    },
}


def send_telegram(cfg: KotakMcxConfig, text: str) -> None:
    if not cfg.telegram_token or not cfg.telegram_chat_id:
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage",
            json={"chat_id": cfg.telegram_chat_id, "text": text},
            timeout=10,
        )
        if not r.ok or not r.json().get("ok"):
            log.warning("Telegram rejected Kotak advisor alert (HTTP %s)", r.status_code)
    except (requests.RequestException, ValueError):
        log.warning("Telegram alert failed")


def fetch_usd_inr(fallback: float = 85.0) -> float:
    try:
        df = load_yfinance("INR=X", interval="1d", days=10, include_warmup=False)
        px = float(df["close"].iloc[-1])
        if math.isfinite(px) and 50.0 <= px <= 150.0:
            return px
    except Exception:
        pass
    return fallback


def mcx_rollover_warning(today: Optional[date] = None) -> Optional[str]:
    today = today or datetime.now(timezone.utc).date()
    last_day = calendar.monthrange(today.year, today.month)[1]
    days_left = last_day - today.day
    if days_left <= 4:
        return f"MCX ROLLOVER ALERT: {days_left} calendar day(s) left in month — roll open GOLDPETAL to next month contract."
    return None


class KotakMcxAdvisor:
    def __init__(self, cfg: KotakMcxConfig):
        self.cfg = cfg
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.cfg.state_file):
            with open(self.cfg.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            state.setdefault("peak_equity_inr", max(state.get("equity_inr", self.cfg.capital_inr), self.cfg.capital_inr))
            state.setdefault("dd_brake_active", False)
            state.setdefault("positions", {})
            return state
        return {
            "equity_inr": self.cfg.capital_inr,
            "peak_equity_inr": self.cfg.capital_inr,
            "dd_brake_active": False,
            "positions": {},
        }

    def save_state(self) -> None:
        path = Path(self.cfg.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def _update_dd_brake(self) -> tuple[float, float, float]:
        eq = float(self.state["equity_inr"])
        peak = max(float(self.state.get("peak_equity_inr", eq)), eq)
        self.state["peak_equity_inr"] = peak
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if self.cfg.dd_brake_threshold > 0:
            if dd >= self.cfg.dd_brake_threshold:
                self.state["dd_brake_active"] = True
            elif dd <= self.cfg.dd_brake_threshold * 0.5:
                self.state["dd_brake_active"] = False
        scale = self.cfg.dd_brake_scale if self.state.get("dd_brake_active") else 1.0
        return dd, self.cfg.risk_per_trade_pct * scale, self.cfg.max_concurrent_risk_pct * scale

    def size_order(self, sym: str, price_inr: float, atr_inr: float, risk_pct: float, max_risk_pct: float) -> Dict[str, Any]:
        spec = ASSETS[sym]
        eq = float(self.state["equity_inr"])
        open_positions = self.state["positions"]
        open_risk = sum(float(p["risk_pct"]) for p in open_positions.values())
        if (open_risk + risk_pct) > (max_risk_pct + 1e-9):
            return {"qty": 0, "reason": f"max concurrent risk ({max_risk_pct*100:.1f}%) reached"}

        sl_dist_inr = self.cfg.atr_trail_mult * atr_inr
        if not math.isfinite(sl_dist_inr) or sl_dist_inr <= 0 or price_inr <= 0:
            return {"qty": 0, "reason": "invalid price/ATR"}

        risk_inr = eq * risk_pct
        raw_qty = risk_inr / sl_dist_inr

        allocated_margin = sum(
            float(p["qty"]) * float(p["entry_inr"]) / ASSETS[s]["leverage"]
            for s, p in open_positions.items()
            if s in ASSETS
        )
        if spec["long_only"]:
            budget = max(eq - allocated_margin, 0.0)
            raw_qty = min(raw_qty, budget / price_inr)
        else:
            budget = max(eq * self.cfg.max_margin_utilization - allocated_margin, 0.0)
            raw_qty = min(raw_qty, (budget * spec["leverage"]) / price_inr)

        qty = math.floor(raw_qty)
        if qty < 1:
            return {"qty": 0, "reason": f"1 unit risk Rs {sl_dist_inr:,.0f} exceeds Rs {risk_inr:,.0f} risk/margin cap"}

        notional = qty * price_inr
        est_margin = notional / spec["leverage"]
        actual_risk = qty * sl_dist_inr
        return {
            "qty": qty,
            "sl_dist_inr": sl_dist_inr,
            "notional_inr": notional,
            "margin_inr": est_margin,
            "risk_inr": actual_risk,
            "risk_pct": actual_risk / eq,
        }

    def _log_journal(self, row: Dict[str, Any]) -> None:
        journal = Path(self.cfg.data_dir) / "kotak_mcx_journal.csv"
        journal.parent.mkdir(parents=True, exist_ok=True)
        exists = journal.exists()
        with journal.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def evaluate_symbol(self, sym: str, df_proxy: pd.DataFrame, fx_rate: float, paper_auto: bool = False) -> str:
        spec = ASSETS[sym]
        feat = add_donchian_features(df_proxy, entry_len=self.cfg.donchian_entry_len, warmup_bars=self.cfg.ema_slow)
        if feat.empty:
            return f"{sym}: insufficient history after {self.cfg.ema_slow}-bar warmup"

        closed = feat.iloc[-1]
        bar_ts = feat.index[-1].isoformat()
        conv = spec["usd_oz_to_unit"] * fx_rate
        close_inr = float(closed["close"]) * conv
        high_inr = float(closed["high"]) * conv
        low_inr = float(closed["low"]) * conv
        open_inr = float(closed["open"]) * conv
        atr_inr = float(closed["atr14"]) * conv
        ema50 = float(closed["ema50"])
        ema200 = float(closed["ema200"])
        dc_hi = float(closed["donchian_high"])
        dc_lo = float(closed["donchian_low"])

        dd, eff_risk, eff_max_risk = self._update_dd_brake()
        pos = self.state["positions"].get(sym)

        if pos is not None:
            sign = 1 if pos["side"] == "LONG" else -1
            old_sl = float(pos["sl_inr"])
            new_sl = round(close_inr - sign * self.cfg.atr_trail_mult * atr_inr, 2)
            sl_stopped = (low_inr <= old_sl if sign > 0 else high_inr >= old_sl) or ((close_inr - old_sl) * sign <= 0)
            lines = [f"[{spec['broker_segment']}] {sym} OPEN {pos['side']} x{pos['qty']} | Px=Rs {close_inr:,.2f} | SL=Rs {old_sl:,.2f}"]
            if sl_stopped:
                if paper_auto:
                    gapped = (open_inr < old_sl) if sign > 0 else (open_inr > old_sl)
                    raw_exit = open_inr if gapped else old_sl
                    exit_px = round(raw_exit * (1 - 0.0005 if sign > 0 else 1 + 0.0005), 2)
                    pnl = self.record_exit(sym, exit_px, fees_inr=round(exit_px * int(pos["qty"]) * 0.0005, 2))
                    lines.append(f"  PAPER EXIT (STOP HIT): Closed @ Rs {exit_px:,.2f} | Net PnL: Rs {pnl:+,.2f} | Equity: Rs {self.state['equity_inr']:,.2f}")
                    return "\n".join(lines)
                lines.append(f"  ACTION: STOP HIT -> Close {pos['qty']} {spec['unit_label']} on Kotak Neo and run --record-exit {sym} <fill_px>")
            elif (new_sl - old_sl) * sign > 0 and (close_inr - new_sl) * sign > 0:
                pos["sl_inr"] = new_sl
                self.save_state()
                lines.append(f"  ACTION: TRAIL SL -> Move Kotak Neo SL from Rs {old_sl:,.2f} to Rs {new_sl:,.2f}")
            else:
                lines.append("  ACTION: HOLD (SL unchanged)")

            # Check Winner Pyramid trigger (+1.5R and stop >= breakeven)
            init_entry = float(pos.get("init_entry_inr", pos["entry_inr"]))
            init_sl_dist = float(pos.get("init_sl_dist_inr", self.cfg.atr_trail_mult * atr_inr))
            cur_sl = float(pos["sl_inr"])
            be_ok = (cur_sl - init_entry) * sign >= 0
            move = (close_inr - init_entry) * sign
            if (
                self.cfg.pyramid_add_fraction > 0
                and not pos.get("pyramided")
                and be_ok
                and move >= self.cfg.pyramid_trigger_r * init_sl_dist
            ):
                add_qty = math.floor(int(pos["qty"]) * self.cfg.pyramid_add_fraction)
                if add_qty >= 1:
                    if paper_auto:
                        fill_px = round(close_inr * (1 + 0.0005 if sign > 0 else 1 - 0.0005), 2)
                        self.record_pyramid(sym, add_qty, fill_px)
                        lines.append(
                            f"  PAPER PYRAMID (+{move/init_sl_dist:.2f}R, SL>=BE): Added +{add_qty} @ Rs {fill_px:,.2f} -> total x{pos['qty']}"
                        )
                    else:
                        lines.append(
                            f"  PYRAMID TRIGGER (+{move/init_sl_dist:.2f}R, SL>=BE): Add +{add_qty} {spec['unit_label']} "
                            f"at ~Rs {close_inr:,.2f}, update total SL qty to {int(pos['qty']) + add_qty} @ Rs {cur_sl:,.2f}, "
                            f"then run --record-pyramid {sym} {add_qty} <fill_px>"
                        )
            return "\n".join(lines)

        last_bars = self.state.setdefault("last_entry_bar", {})
        if last_bars.get(sym) == bar_ts:
            return f"[{spec['broker_segment']}] {sym} | Px=Rs {close_inr:,.2f} | Already evaluated bar {bar_ts}"

        side = None
        if ema50 > ema200 and float(closed["close"]) > dc_hi:
            side = "LONG"
        elif (not spec["long_only"]) and ema50 < ema200 and float(closed["close"]) < dc_lo:
            side = "SHORT"

        if not side:
            regime = "BULL" if ema50 > ema200 else "BEAR"
            return (
                f"[{spec['broker_segment']}] {sym} | Px=Rs {close_inr:,.2f} | Regime={regime} "
                f"| No breakout signal (DD={dd*100:.1f}%, Brake={'ON' if self.state['dd_brake_active'] else 'OFF'})"
            )

        sized = self.size_order(sym, close_inr, atr_inr, eff_risk, eff_max_risk)
        if sized["qty"] < 1:
            return f"[{spec['broker_segment']}] {sym} {side} SIGNAL SKIPPED: {sized['reason']}"

        sign = 1 if side == "LONG" else -1
        fill_inr = round(close_inr * (1 + 0.0005 if sign > 0 else 1 - 0.0005), 2)
        sl_inr = round(fill_inr - sign * sized["sl_dist_inr"], 2)
        if paper_auto:
            self.record_entry(sym, side, sized["qty"], fill_inr, sl_inr)
            last_bars[sym] = bar_ts
            self.save_state()
            return "\n".join([
                f"[{spec['broker_segment']}] PAPER ENTRY: {side} {sym} x {sized['qty']} {spec['unit_label']} @ Rs {fill_inr:,.2f}",
                f"  Initial SL: Rs {sl_inr:,.2f} ({self.cfg.atr_trail_mult:.1f}x ATR) | Risk: Rs {sized['risk_inr']:,.0f} ({sized['risk_pct']*100:.2f}%)",
            ])
        return "\n".join([
            f"[{spec['broker_segment']}] ENTRY SIGNAL: {side} {sym} x {sized['qty']} {spec['unit_label']}",
            f"  Est Price : Rs {close_inr:,.2f} (FX={fx_rate:.2f}) | Initial SL: Rs {sl_inr:,.2f} ({self.cfg.atr_trail_mult:.1f}x ATR)",
            f"  Notional  : Rs {sized['notional_inr']:,.0f} | Est Margin: Rs {sized['margin_inr']:,.0f} | Risk: Rs {sized['risk_inr']:,.0f} ({sized['risk_pct']*100:.2f}%)",
            f"  Confirm   : python live_kotak_mcx_advisor.py --record-entry {sym} {side} {sized['qty']} <fill_px> {sl_inr}",
        ])

    def run_once(self, fx_override: Optional[float] = None, paper_auto: bool = False) -> str:
        fx = fx_override if fx_override is not None else fetch_usd_inr(self.cfg.usd_inr_rate)
        dd, eff_risk, eff_max_risk = self._update_dd_brake()
        mode_tag = "PAPER-AUTO" if paper_auto else "ADVISOR"
        header = (
            f"KOTAK NEO MCX/ETF {mode_tag} ({self.cfg.resolution.upper()}) | "
            f"Equity=Rs {self.state['equity_inr']:,.0f} | Peak=Rs {self.state['peak_equity_inr']:,.0f} | "
            f"DD={dd*100:.1f}% | Risk={eff_risk*100:.2f}%/{eff_max_risk*100:.1f}% | USDINR={fx:.2f}"
        )
        blocks = [header]
        roll = mcx_rollover_warning()
        if roll:
            blocks.append(roll)
        for sym, spec in ASSETS.items():
            try:
                df = load_yfinance(spec["proxy"], interval=self.cfg.resolution, days=180, include_warmup=True)
                blocks.append(self.evaluate_symbol(sym, df, fx, paper_auto=paper_auto))
            except Exception as exc:
                blocks.append(f"[{spec['broker_segment']}] {sym} ERROR: {exc}")
        self.save_state()
        report = "\n\n".join(blocks)
        if any(k in report for k in ("ENTRY SIGNAL", "PAPER ENTRY", "PAPER EXIT", "PAPER PYRAMID", "ACTION: TRAIL", "ACTION: STOP HIT", "PYRAMID TRIGGER", "ROLLOVER ALERT")):
            send_telegram(self.cfg, report)
        return report

    def record_entry(self, sym: str, side: str, qty: int, fill_inr: float, sl_inr: float) -> None:
        if sym not in ASSETS:
            raise ValueError(f"Unsupported symbol: {sym}")
        side = side.upper()
        if ASSETS[sym]["long_only"] and side != "LONG":
            raise ValueError(f"{sym} is long-only")
        if qty < 1 or fill_inr <= 0 or sl_inr <= 0:
            raise ValueError("qty, fill_inr, and sl_inr must be positive")
        sl_dist = abs(fill_inr - sl_inr)
        risk_inr = qty * sl_dist
        self.state["positions"][sym] = {
            "side": side,
            "qty": qty,
            "entry_inr": fill_inr,
            "init_entry_inr": fill_inr,
            "init_sl_dist_inr": sl_dist,
            "sl_inr": sl_inr,
            "risk_pct": risk_inr / max(float(self.state["equity_inr"]), 1.0),
            "pyramided": False,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_state()

    def record_pyramid(self, sym: str, add_qty: int, fill_inr: float) -> None:
        pos = self.state["positions"].get(sym)
        if not pos:
            raise ValueError(f"No open position for {sym}")
        new_qty = int(pos["qty"]) + add_qty
        pos["entry_inr"] = (float(pos["entry_inr"]) * int(pos["qty"]) + fill_inr * add_qty) / new_qty
        pos["qty"] = new_qty
        pos["pyramided"] = True
        self.save_state()

    def record_exit(self, sym: str, exit_inr: float, fees_inr: float = 0.0) -> float:
        pos = self.state["positions"].get(sym)
        if not pos:
            raise ValueError(f"No open position for {sym}")
        sign = 1 if pos["side"] == "LONG" else -1
        pnl = (exit_inr - float(pos["entry_inr"])) * sign * int(pos["qty"]) - fees_inr
        self.state["equity_inr"] = float(self.state["equity_inr"]) + pnl
        del self.state["positions"][sym]
        self._update_dd_brake()
        self.save_state()
        self._log_journal({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "side": pos["side"],
            "qty": int(pos["qty"]),
            "entry_inr": round(float(pos["entry_inr"]), 2),
            "exit_inr": round(exit_inr, 2),
            "fees_inr": round(fees_inr, 2),
            "pnl_inr": round(pnl, 2),
            "equity_inr": round(float(self.state["equity_inr"]), 2),
        })
        return pnl


def main() -> None:
    parser = argparse.ArgumentParser(description="Kotak Neo MCX GOLDPETAL + NSE SILVERBEES Advisor (Config C)")
    parser.add_argument("--capital", type=float, default=None, help="Override Kotak INR capital (default: 98000)")
    parser.add_argument("--usd-inr", type=float, default=None, help="Override USD/INR exchange rate")
    parser.add_argument("--resolution", choices=["4h", "6h", "12h", "1d"], default=None)
    parser.add_argument("--paper-auto", action="store_true", help="Automatically track paper entries, trails, pyramids, and exits into kotak_mcx_journal.csv")
    parser.add_argument("--status", action="store_true", help="Print stored Kotak Neo advisor state")
    parser.add_argument("--record-entry", nargs=5, metavar=("SYM", "SIDE", "QTY", "FILL_INR", "SL_INR"))
    parser.add_argument("--record-pyramid", nargs=3, metavar=("SYM", "ADD_QTY", "FILL_INR"))
    parser.add_argument("--record-exit", nargs=2, metavar=("SYM", "EXIT_INR"))
    parser.add_argument("--fees-inr", type=float, default=0.0, help="Optional fees/STT/CTT for --record-exit")
    args = parser.parse_args()

    cfg = KotakMcxConfig(
        **({} if args.capital is None else {"capital_inr": args.capital}),
        **({} if args.resolution is None else {"resolution": args.resolution}),
    )
    log_path = Path(cfg.data_dir) / "kotak_mcx_advisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")],
    )
    advisor = KotakMcxAdvisor(cfg)

    if args.record_entry:
        sym, side, qty, fill_px, sl_px = args.record_entry
        advisor.record_entry(sym, side, int(qty), float(fill_px), float(sl_px))
        print(f"Recorded {side} {sym} x{qty} @ Rs {float(fill_px):,.2f} (SL Rs {float(sl_px):,.2f})")
        return
    if args.record_pyramid:
        sym, add_qty, fill_px = args.record_pyramid
        advisor.record_pyramid(sym, int(add_qty), float(fill_px))
        print(f"Recorded pyramid +{add_qty} {sym} @ Rs {float(fill_px):,.2f}")
        return
    if args.record_exit:
        sym, exit_px = args.record_exit
        pnl = advisor.record_exit(sym, float(exit_px), fees_inr=args.fees_inr)
        print(f"Recorded exit {sym} @ Rs {float(exit_px):,.2f} | Net PnL: Rs {pnl:+,.2f} | Equity: Rs {advisor.state['equity_inr']:,.2f}")
        return
    if args.status:
        print(json.dumps(advisor.state, indent=2))
        return

    report = advisor.run_once(fx_override=args.usd_inr, paper_auto=args.paper_auto)
    log.info("\n%s", report)
    print(report)


if __name__ == "__main__":
    main()
