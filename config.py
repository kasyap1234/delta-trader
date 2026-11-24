import os
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class PortfolioConfig:
    """
    Max-Aggression Multi-Asset Donchian Portfolio Config (Config C):
      - Delta Crypto (6H): 2.0% risk/trade, 8.0% max open risk, max 5 crypto positions, 3.5x ATR trail
      - Kotak Neo MCX/ETF (6H): 2.5% risk/trade, 8.0% max open risk, 3.0x ATR trail (GOLDPETAL + SILVERBEES)
      - Winner Pyramiding (+0.5x at +1.5R once stop >= breakeven) + 15% Drawdown Brake (0.5x risk)
    """
    initial_capital: float = 200_000.0
    usd_inr_rate: float = 85.0                    # Convert USD-quoted assets to INR capital (set 1.0 for USD runs)

    # Engine A: Multi-Asset Donchian Trend
    donchian_risk_per_trade_pct: float = 0.02     # 2.0% risk per trade
    max_concurrent_risk_pct: float = 0.08         # 8.0% max total open risk
    max_crypto_concurrent: int = 5                # Max 5 crypto positions at once
    donchian_entry_len: int = 20                  # 20-bar breakout
    atr_trail_mult: float = 3.5                   # 3.5x ATR trailing stop
    slippage_pct: float = 0.0015                  # 0.15% per side default (or stop exit cost when entry_slippage_pct is set)
    entry_slippage_pct: Optional[float] = None    # Optional maker entry cost (e.g. 0.0004); defaults to slippage_pct
    crypto_funding_8h_pct: float = 0.0001         # 0.01% per 8h on open crypto perps
    max_leverage: Optional[float] = 5.0           # Enforce max leverage on non-spot assets
    max_margin_utilization: float = 0.75          # Keep >=25% free margin buffer

    # Winner Pyramiding & Drawdown Brake
    pyramid_add_fraction: float = 0.5             # Add +0.5x initial size on winners (0.0 disables)
    pyramid_trigger_r: float = 1.5                # Trigger pyramid at +1.5R once trailing stop >= breakeven
    dd_brake_threshold: Optional[float] = 0.15    # Halve risk when drawdown >= 15% (None disables)
    dd_brake_scale: float = 0.5                   # Risk multiplier while drawdown brake is active

    # Hardening Controls
    ema_warmup_bars: int = 200                    # Strictly drop first 200 bars before trading
    gap_aware_stops: bool = True                  # Fill at bar open if price gaps past stop-loss
    whole_lot_rounding: bool = False              # Optional whole-lot rounding for strict realism
    spread_stress_penalty: float = 0.0            # Optional extra loss penalty on spread SL (e.g. 0.15 = 15% worse)

    # Engine B: Collateral-Backed Weekly Credit Spreads (VRP — disabled by default; requires Rs 11L+ for 1 Nifty lot)
    enable_credit_spreads: bool = False
    spread_risk_per_week_pct: float = 0.015       # 1.5% capped max loss per weekly spread
    short_strike_sd: float = 0.80                 # ~0.80 SD (~20-22 Delta) short strike
    long_strike_sd: float = 1.60                  # Protective wing at 1.60 SD
    vrp_iv_multiplier: float = 1.15               # IV = 1.15x 20-day realized vol
    collateral_yield_apr: float = 0.07            # 7.0% p.a. from pledged Liquid ETFs

    # Default Asset Mix (5 consistent liquid Delta majors + Gold proxy)
    crypto_assets: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"])
    commodity_assets: List[str] = field(default_factory=lambda: ["GC=F"])
    spread_asset: str = "^NSEI"

    # Spot ETFs are automatically restricted to LONG-ONLY (no naked shorting)
    long_only_assets: Set[str] = field(
        default_factory=lambda: {
            "GOLDIETF.NS", "GOLDCASE.NS", "GOLDBEES.NS", "SILVERBEES.NS",
            "GOLDIETF", "GOLDCASE", "GOLDBEES", "SILVERBEES",
        }
    )

    # Assets whose prices/ticks are denominated in USD rather than INR
    usd_quoted_assets: Set[str] = field(
        default_factory=lambda: {
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "XRPUSDT",
            "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD", "XRPUSD",
            "XAUTUSD", "PAXGUSD", "GC=F", "SI=F", "CL=F",
        }
    )

    # Minimum tradable lot/unit multiplier when whole_lot_rounding=True
    lot_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "BTCUSDT": 0.001,       # Delta Exchange BTCUSD contract = 0.001 BTC
            "ETHUSDT": 0.01,        # Delta Exchange ETHUSD contract = 0.01 ETH
            "SOLUSDT": 1.0,         # Delta Exchange SOLUSD contract = 1 SOL
            "BNBUSDT": 0.1,         # Delta Exchange BNBUSD contract = 0.1 BNB
            "DOGEUSDT": 100.0,      # Delta Exchange DOGEUSD contract = 100 DOGE
            "XRPUSDT": 1.0,         # Delta Exchange XRPUSD contract = 1 XRP
            "BTCUSD": 0.001,
            "ETHUSD": 0.01,
            "SOLUSD": 1.0,
            "BNBUSD": 0.1,
            "DOGEUSD": 100.0,
            "XRPUSD": 1.0,
            "XAUTUSD": 0.001,       # Delta Exchange XAUTUSD contract = 0.001 oz gold
            "PAXGUSD": 0.001,       # Delta Exchange PAXGUSD contract = 0.001 oz gold
            "GOLDIETF.NS": 1.0,     # 1 ETF share
            "GOLDCASE.NS": 1.0,     # 1 ETF share
            "GOLDBEES.NS": 1.0,     # 1 ETF share
            "SILVERBEES.NS": 1.0,   # 1 ETF share (~1g silver)
            "GOLDIETF": 1.0,        # Upstox NSE_EQ: 1 ETF share (INR)
            "GOLDCASE": 1.0,        # Upstox NSE_EQ: 1 ETF share (INR)
            "GOLDBEES": 1.0,        # Upstox NSE_EQ: 1 ETF share (INR)
            "SILVERBEES": 1.0,      # Upstox NSE_EQ: 1 ETF share (INR)
            "GOLDPETAL": 1.0,       # Upstox MCX GOLDPETAL: 1 lot = 1g (quoted per 1g in INR)
            "GOLDM": 10.0,          # Upstox MCX GOLDM: 1 lot = 100g (quoted per 10g in INR)
            "SILVERMIC": 1.0,       # Upstox MCX SILVERMIC: 1 lot = 1kg (quoted per 1kg in INR)
            "GC=F": 1.0 / 31.1035,       # Proxy mapped to MCX GOLDPETAL (1g)
            "SI=F": 1.0 / 31.1035,       # Proxy mapped to SILVERBEES (~1g silver)
        }
    )


def _env_bool(name: str, default: str) -> Optional[bool]:
    return {"true": True, "false": False}.get(os.getenv(name, default).strip().lower())


def _env_symbols(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass
class DeltaLiveConfig:
    env: str = field(default_factory=lambda: os.getenv("DELTA_ENV", "testnet").strip().lower())
    api_key: str = field(default_factory=lambda: os.getenv("DELTA_API_KEY", ""), repr=False)
    api_secret: str = field(default_factory=lambda: os.getenv("DELTA_API_SECRET", ""), repr=False)
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", "true"))

    # Capital & Risk (Config C: Max Aggression, <30% Target Max DD)
    capital_usd: float = field(default_factory=lambda: float(os.getenv("DELTA_CAPITAL_USD", "1200.0")))
    risk_per_trade_pct: float = 0.02              # 2.0% risk per trade
    max_concurrent_risk_pct: float = 0.08         # 8.0% max total open risk
    max_crypto_concurrent: int = 5                # Max 5 open crypto positions
    leverage: int = 5                             # 5x isolated leverage
    max_margin_utilization: float = 0.75          # Keep >=25% free margin buffer
    entry_slippage_pct: float = 0.0015            # 0.15% max IOC slippage collar (fallback / stop estimate)
    use_maker_entry: bool = field(default_factory=lambda: _env_bool("DELTA_MAKER_ENTRY", "true"))
    maker_timeout_sec: float = field(default_factory=lambda: float(os.getenv("DELTA_MAKER_TIMEOUT_SEC", "15.0")))

    # Strategy Parameters (Config C or 4-Pod Balanced: Donchian + Supertrend+ADX + CTA TSMOM + Funding+Vol Squeeze)
    strategy_mode: str = field(default_factory=lambda: os.getenv("DELTA_STRATEGY_MODE", "config_c").strip().lower())
    pod_weight_donchian: float = 0.30             # 4-Pod Balanced Pod 1 weight
    pod_weight_supertrend: float = 0.25           # 4-Pod Balanced Pod 2 weight (Supertrend(10,3)+ADX>20)
    pod_weight_tsmom: float = 0.25                # 4-Pod Balanced Pod 3 weight (CTA TSMOM Vol-Target)
    pod_weight_squeeze: float = 0.35              # 4-Pod Balanced Pod 4 weight (Funding + Volume Squeeze)
    squeeze_atr_trail_mult: float = 3.0           # 3.0x ATR trailing stop for Pod 4 Squeeze
    squeeze_vol_mult: float = 1.5                 # Volume surge threshold (> 1.5x 20-bar avg)
    squeeze_long_max_funding: float = 0.00005     # <= 0.005% 3d avg funding for LONG squeeze
    squeeze_short_min_funding: float = 0.00015    # >= 0.015% 3d avg funding for SHORT squeeze
    tsmom_rebalance_bars: int = 8                 # Rebalance TSMOM every 8 bars (48h on 6h)
    tsmom_target_vol: float = 0.35                # Annualized vol target before pod weight & DD scale
    catastrophe_stop_atr_mult: float = 4.5        # Protective exchange stop when only TSMOM is open
    symbols: List[str] = field(
        default_factory=lambda: _env_symbols("DELTA_SYMBOLS", ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"])
    )
    resolution: str = field(default_factory=lambda: os.getenv("DELTA_RESOLUTION", "6h").strip().lower())
    donchian_entry_len: int = 20
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_trail_mult: float = 3.5
    pyramid_add_fraction: float = 0.5             # Add +0.5x initial size on winners (0.0 disables)
    pyramid_trigger_r: float = 1.5                # Trigger pyramid at +1.5R once stop >= breakeven
    dd_brake_threshold: float = 0.15              # Halve risk when equity drawdown >= 15% (0.0 disables)
    dd_brake_scale: float = 0.5                   # Risk multiplier while drawdown brake is active

    # Circuit Breakers (scaled for 2% risk / 5-coin basket)
    daily_loss_limit_pct: float = 0.06            # Stop new entries at -6% daily loss
    weekly_loss_limit_pct: float = 0.12           # Stop new entries at -12% weekly loss

    # Persistence & Alerts
    data_dir: str = field(default_factory=lambda: os.getenv("BOT_DATA_DIR", "."))
    state_file: str = ""
    journal_file: str = ""
    log_file: str = ""
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""), repr=False)
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    def __post_init__(self):
        if self.dry_run is not True and self.dry_run is not False:
            raise ValueError("DRY_RUN must be true or false")
        if self.use_maker_entry is not True and self.use_maker_entry is not False:
            raise ValueError("DELTA_MAKER_ENTRY must be true or false")
        if self.env not in {"prod", "testnet"}:
            raise ValueError("DELTA_ENV must be testnet or prod")
        if self.strategy_mode not in {"config_c", "3pod"}:
            raise ValueError("DELTA_STRATEGY_MODE must be config_c or 3pod")
        if self.resolution not in {"1h", "2h", "4h", "6h", "12h", "1d"}:
            raise ValueError("DELTA_RESOLUTION must be one of 1h, 2h, 4h, 6h, 12h, 1d")
        if not self.dry_run and self.env != "testnet":
            raise ValueError("Production orders are disabled pending authenticated testnet validation")
        if not isfinite(self.capital_usd) or self.capital_usd <= 0:
            raise ValueError("DELTA_CAPITAL_USD must be positive and finite")
        if not isfinite(self.entry_slippage_pct) or not (0.0 <= self.entry_slippage_pct < 0.05):
            raise ValueError("entry_slippage_pct must be between 0.0 and 0.05")
        if not isfinite(self.maker_timeout_sec) or self.maker_timeout_sec < 0:
            raise ValueError("maker_timeout_sec must be non-negative and finite")
        if not isfinite(self.max_margin_utilization) or not (0.1 <= self.max_margin_utilization <= 0.90):
            raise ValueError("max_margin_utilization must be between 0.1 and 0.90")
        if not isfinite(self.pyramid_add_fraction) or not (0.0 <= self.pyramid_add_fraction <= 1.0):
            raise ValueError("pyramid_add_fraction must be between 0.0 and 1.0")
        if not isfinite(self.dd_brake_threshold) or not (0.0 <= self.dd_brake_threshold < 0.5):
            raise ValueError("dd_brake_threshold must be between 0.0 and 0.5")
        mode_prefix = "3pod-" if self.strategy_mode == "3pod" else ""
        mode = f"{mode_prefix}{self.env}-{'paper' if self.dry_run else 'orders'}"
        base_dir = Path(self.data_dir).expanduser().resolve()
        self.data_dir = str(base_dir)
        self.state_file = self.state_file or str(base_dir / f"bot_state-{mode}.json")
        self.journal_file = self.journal_file or str(base_dir / f"trade_journal-{mode}.csv")
        self.log_file = self.log_file or str(base_dir / f"delta_live_bot-{mode}.log")

    @property
    def base_url(self) -> str:
        if self.env == "prod":
            return "https://api.india.delta.exchange"
        return "https://cdn-ind.testnet.deltaex.org"


@dataclass
class KotakMcxConfig:
    """Sleeve 2: Kotak Neo Advisor — 'rotation' (All-6 Top-2 Vol-Parity ETFs + LIQUIDBEES) or 'config_c' (MCX GOLDPETAL + SILVERBEES)."""
    capital_inr: float = field(default_factory=lambda: float(os.getenv("KOTAK_CAPITAL_INR", "98000.0")))
    strategy_mode: str = field(default_factory=lambda: os.getenv("KOTAK_STRATEGY_MODE", "config_c").strip().lower())
    data_source: str = field(default_factory=lambda: os.getenv("KOTAK_DATA_SOURCE", "upstox").strip().lower())
    usd_inr_rate: float = field(default_factory=lambda: float(os.getenv("USD_INR_RATE", "85.0")))
    rotation_symbols: List[str] = field(
        default_factory=lambda: _env_symbols(
            "KOTAK_ROTATION_SYMBOLS",
            ["GOLDBEES", "SILVERBEES", "NIFTYBEES", "MID150BEES", "BANKBEES", "ITBEES"],
        )
    )
    rotation_top_k: int = 2
    rotation_rebal_days: int = 10
    rotation_cash_symbol: str = "LIQUIDBEES"
    risk_per_trade_pct: float = 0.025             # 2.5% risk per trade
    max_concurrent_risk_pct: float = 0.08         # 8.0% max total open risk
    max_leverage_mcx: float = 8.0                 # ~12.5% MCX margin assumption
    max_margin_utilization: float = 0.75          # Keep >=25% free margin buffer
    resolution: str = field(default_factory=lambda: os.getenv("KOTAK_RESOLUTION", "6h").strip().lower())
    donchian_entry_len: int = 20
    ema_fast: int = 50
    ema_slow: int = 200
    atr_trail_mult: float = 3.0                   # 3.0x ATR trail for Gold + Silver
    pyramid_add_fraction: float = 0.5             # +0.5x on winners at +1.5R once stop >= breakeven
    pyramid_trigger_r: float = 1.5
    dd_brake_threshold: float = 0.15              # Halve risk when equity drawdown >= 15%
    dd_brake_scale: float = 0.5
    data_dir: str = field(default_factory=lambda: os.getenv("BOT_DATA_DIR", "."))
    state_file: str = ""
    journal_file: str = ""
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""), repr=False)
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    def __post_init__(self):
        if self.strategy_mode not in {"config_c", "rotation"}:
            raise ValueError("KOTAK_STRATEGY_MODE must be rotation or config_c")
        if self.data_source not in {"upstox", "yfinance"}:
            raise ValueError("KOTAK_DATA_SOURCE must be upstox or yfinance")
        if self.resolution not in {"1h", "2h", "4h", "6h", "12h", "1d"}:
            raise ValueError("KOTAK_RESOLUTION must be one of 1h, 2h, 4h, 6h, 12h, 1d")
        if not isfinite(self.capital_inr) or self.capital_inr <= 0:
            raise ValueError("KOTAK_CAPITAL_INR must be positive and finite")
        base_dir = Path(self.data_dir).expanduser().resolve()
        self.data_dir = str(base_dir)
        if self.strategy_mode == "rotation":
            self.state_file = self.state_file or str(base_dir / "kotak_rotation_state.json")
            self.journal_file = self.journal_file or str(base_dir / "kotak_rotation_journal.csv")
        else:
            self.state_file = self.state_file or str(base_dir / "kotak_mcx_state.json")
            self.journal_file = self.journal_file or str(base_dir / "kotak_mcx_journal.csv")
