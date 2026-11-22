import os
from math import isfinite
from dataclasses import dataclass, field
from typing import Dict, List, Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class PortfolioConfig:
    """
    Test 2 Winning Risk Metrics (Preserved Exactly):
      - 1.0% risk per Donchian trade
      - 3.0% max concurrent open risk across portfolio
      - Max 2 simultaneous crypto positions (correlation cap)
      - 20-bar Donchian breakout + 2.5x ATR(14) trailing stop
      - 1.5% capped weekly max loss on Nifty Credit Spreads (0.80 SD / 1.60 SD)
    """
    initial_capital: float = 200_000.0

    # Engine A: Multi-Asset Donchian Trend (Test 2 metrics preserved)
    donchian_risk_per_trade_pct: float = 0.01     # 1.0% risk per trade
    max_concurrent_risk_pct: float = 0.03         # 3.0% max total open risk
    max_crypto_concurrent: int = 2                # Max 2 crypto positions at once
    donchian_entry_len: int = 20                  # 20-bar breakout
    atr_trail_mult: float = 2.5                   # 2.5x ATR trailing stop
    slippage_pct: float = 0.0005                  # 0.05% per side (slippage + fees)
    crypto_funding_8h_pct: float = 0.0001         # 0.01% per 8h on open crypto perps

    # Hardening Controls
    ema_warmup_bars: int = 200                    # Strictly drop first 200 bars before trading
    gap_aware_stops: bool = True                  # Fill at bar open if price gaps past stop-loss
    whole_lot_rounding: bool = False              # Optional whole-lot rounding for strict Rs 2L realism
    spread_stress_penalty: float = 0.0            # Optional extra loss penalty on spread SL (e.g. 0.15 = 15% worse)

    # Engine B: Collateral-Backed Weekly Credit Spreads (VRP)
    enable_credit_spreads: bool = True
    spread_risk_per_week_pct: float = 0.015       # 1.5% capped max loss per weekly spread
    short_strike_sd: float = 0.80                 # ~0.80 SD (~20-22 Delta) short strike
    long_strike_sd: float = 1.60                  # Protective wing at 1.60 SD
    vrp_iv_multiplier: float = 1.15               # IV = 1.15x 20-day realized vol
    collateral_yield_apr: float = 0.07            # 7.0% p.a. from pledged Liquid ETFs

    # Default Hardened Asset Mix (Silver SI=F dropped due to negative PF)
    crypto_assets: List[str] = field(default_factory=lambda: ["ETHUSDT", "SOLUSDT"])
    commodity_assets: List[str] = field(default_factory=lambda: ["GC=F"])
    spread_asset: str = "^NSEI"

    # Spot ETFs are automatically restricted to LONG-ONLY (no naked shorting)
    long_only_assets: Set[str] = field(
        default_factory=lambda: {"GOLDIETF.NS", "GOLDCASE.NS", "GOLDBEES.NS", "SILVERBEES.NS"}
    )

    # Minimum tradable lot/unit multiplier when whole_lot_rounding=True
    lot_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "ETHUSDT": 0.01,        # Delta Exchange ETHUSD contract = 0.01 ETH
            "SOLUSDT": 1.0,         # Delta Exchange SOLUSD contract = 1 SOL
            "BTCUSDT": 0.001,       # Delta Exchange BTCUSD contract = 0.001 BTC
            "GOLDIETF.NS": 1.0,     # 1 ETF share
            "GOLDCASE.NS": 1.0,     # 1 ETF share
            "GOLDBEES.NS": 1.0,     # 1 ETF share
            "GOLDM": 100.0,         # MCX Gold Mini = 100g
            "SILVERM": 5.0,         # MCX Silver Mini = 5kg
            "GC=F": 1.0,            # Continuous proxy unit
        }
    )


@dataclass
class DeltaLiveConfig:
    env: str = field(default_factory=lambda: os.getenv("DELTA_ENV", "testnet").strip().lower())
    api_key: str = field(default_factory=lambda: os.getenv("DELTA_API_KEY", ""), repr=False)
    api_secret: str = field(default_factory=lambda: os.getenv("DELTA_API_SECRET", ""), repr=False)
    dry_run: bool = field(default_factory=lambda: {"true": True, "false": False}.get(os.getenv("DRY_RUN", "true").strip().lower()))

    # Capital & Risk (Exact Test 2 Parameters)
    capital_usd: float = field(default_factory=lambda: float(os.getenv("DELTA_CAPITAL_USD", "1200.0")))
    risk_per_trade_pct: float = 0.01              # 1.0% risk per trade
    max_concurrent_risk_pct: float = 0.03         # 3.0% max total open risk
    max_crypto_concurrent: int = 2                # Max 2 open crypto positions
    leverage: int = 5                             # 5x isolated leverage (keeps >=40% free margin)

    # Strategy Parameters (Exact Test 2 Parameters)
    symbols: List[str] = field(default_factory=lambda: ["ETHUSD", "SOLUSD"])
    resolution: str = "4h"
    donchian_entry_len: int = 20
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_trail_mult: float = 2.5

    # Circuit Breakers
    daily_loss_limit_pct: float = 0.03            # Stop new entries at -3% daily realized loss
    weekly_loss_limit_pct: float = 0.07           # Stop new entries at -7% weekly realized loss

    # Persistence & Alerts
    state_file: str = ""
    journal_file: str = ""
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""), repr=False)
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    def __post_init__(self):
        if self.dry_run is not True and self.dry_run is not False:
            raise ValueError("DRY_RUN must be true or false")
        if self.env not in {"prod", "testnet"}:
            raise ValueError("DELTA_ENV must be testnet or prod")
        if not self.dry_run and self.env != "testnet":
            raise ValueError("Production orders are disabled pending authenticated testnet validation")
        if not isfinite(self.capital_usd) or self.capital_usd <= 0:
            raise ValueError("DELTA_CAPITAL_USD must be positive and finite")
        mode = f"{self.env}-{'paper' if self.dry_run else 'orders'}"
        self.state_file = self.state_file or f"bot_state-{mode}.json"
        self.journal_file = self.journal_file or f"trade_journal-{mode}.csv"

    @property
    def base_url(self) -> str:
        if self.env == "prod":
            return "https://api.india.delta.exchange"
        return "https://cdn-ind.testnet.deltaex.org"
