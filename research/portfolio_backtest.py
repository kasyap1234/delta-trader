"""Backtest Config C, individual sleeves, or the legacy synthetic portfolio."""

import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import PortfolioConfig
from data_loader import (
    SUPPORTED_INTERVALS,
    load_binance,
    load_delta_candles,
    load_upstox,
    load_yfinance,
    resolve_date_window,
)
from portfolio_engine import (
    PortfolioTrade,
    run_shared_donchian_portfolio,
    run_weekly_credit_spreads,
)
from research.outputs import new_run_directory


def calc_curve_stats(name: str, equity_curve: pd.Series, trades: List[PortfolioTrade], initial_capital: float, rf_apr: float = 0.0) -> Dict[str, Any]:
    if equity_curve.empty:
        return {
            "Portfolio / Sleeve": name, "Trades": 0, "Win %": 0.0, "PF": 0.0,
            "Net Return %": 0.0, "CAGR %": 0.0, "Max DD %": 0.0, "Calmar": 0.0, "Sharpe": 0.0,
        }

    eq = equity_curve.dropna()
    peak = eq.cummax()
    dd = ((peak - eq) / peak) * 100.0
    max_dd = float(dd.max())

    net_ret = ((float(eq.iloc[-1]) - initial_capital) / initial_capital) * 100.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = (((float(eq.iloc[-1]) / initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if eq.iloc[-1] > 0 else -100.0
    calmar = cagr / max_dd if max_dd > 0.01 else 0.0

    daily_rets = eq.pct_change().dropna()
    excess_rets = daily_rets - (rf_apr / 365.0)
    sharpe = (excess_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 1e-9 else 0.0

    pnls = np.array([t.net_pnl for t in trades]) if trades else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = (len(wins) / len(pnls) * 100.0) if len(pnls) else 0.0
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else (99.0 if len(wins) else 0.0)

    return {
        "Portfolio / Sleeve": name,
        "Trades": len(trades),
        "Win %": round(win_rate, 1),
        "PF": round(float(pf), 2),
        "Net Return %": round(net_ret, 2),
        "CAGR %": round(cagr, 2),
        "Max DD %": round(max_dd, 2),
        "Calmar": round(calmar, 2),
        "Sharpe": round(float(sharpe), 2),
    }


def summarize_by_asset(trades: List[PortfolioTrade], initial_capital: float, currency: str = "Rs") -> pd.DataFrame:
    rows = []
    for asset in sorted(set(t.asset for t in trades)):
        sub = [t for t in trades if t.asset == asset]
        pnls = np.array([t.net_pnl for t in sub])
        rs = np.array([t.r_multiple for t in sub])
        funding = sum(t.funding_cost for t in sub)
        gap_stops = sum(1 for t in sub if t.exit_reason == "GAP_STOP")
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else 99.0
        rows.append({
            "Asset": asset,
            "Engine": sub[0].engine,
            "Trades": len(sub),
            "Win %": round(len(wins) / len(pnls) * 100.0, 1),
            "PF": round(float(pf), 2),
            "Avg R": round(float(rs.mean()), 2),
            "Gap Stops": gap_stops,
            f"Funding ({currency})": round(funding, 2 if currency == "$" else 0),
            f"Net PnL ({currency})": round(float(pnls.sum()), 2 if currency == "$" else 0),
            "Contrib %": round(float(pnls.sum()) / initial_capital * 100.0, 2),
        })
    return pd.DataFrame(rows)


def run_single_backtest(
    interval: str = "6h",
    days: int = 730,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    capital: float = 200_000.0,
    usd_inr_rate: float = 85.0,
    risk: float = 0.02,
    max_risk: float = 0.08,
    max_crypto_concurrent: int = 5,
    atr_mult: float = 3.5,
    slippage: float = 0.0015,
    entry_slippage: Optional[float] = None,
    pyramid_add: float = 0.5,
    pyramid_r: float = 1.5,
    dd_brake: Optional[float] = 0.15,
    spread_share: float = 0.5,
    crypto_source: str = "binance",
    commodity_source: str = "yfinance",
    crypto_syms: Optional[List[str]] = None,
    commodities: Optional[List[str]] = None,
    gold_etf: bool = False,
    crypto_only: bool = False,
    enable_credit_spreads: bool = False,
    spread_asset: str = "^NSEI",
    use_upstox_nifty: bool = False,
    whole_lots: bool = False,
    spread_stress: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    crypto_syms = list(crypto_syms or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"])
    if crypto_source == "delta":
        crypto_syms = [s.replace("USDT", "USD") if s.endswith("USDT") else s for s in crypto_syms]

    if crypto_only:
        commodities = []
    elif gold_etf:
        commodities = ["GOLDIETF.NS"]
    else:
        commodities = list(commodities) if commodities is not None else ["GC=F"]

    cfg = PortfolioConfig(
        initial_capital=capital,
        usd_inr_rate=usd_inr_rate,
        donchian_risk_per_trade_pct=risk,
        max_concurrent_risk_pct=max_risk,
        max_crypto_concurrent=max_crypto_concurrent,
        atr_trail_mult=atr_mult,
        slippage_pct=slippage,
        entry_slippage_pct=entry_slippage,
        pyramid_add_fraction=pyramid_add,
        pyramid_trigger_r=pyramid_r,
        dd_brake_threshold=dd_brake,
        crypto_assets=crypto_syms,
        commodity_assets=commodities,
        spread_asset="NIFTY" if use_upstox_nifty else spread_asset,
        enable_credit_spreads=enable_credit_spreads and not crypto_only and not whole_lots,
        whole_lot_rounding=whole_lots,
        spread_stress_penalty=spread_stress,
    )

    _, _, eval_start_ist, eval_end_ist = resolve_date_window(interval, days, start_date, end_date, include_warmup=True)

    asset_dfs: Dict[str, pd.DataFrame] = {}
    crypto_set = set(crypto_syms)

    if verbose:
        window_desc = f"{start_date or f'last {days}d'} -> {end_date or 'today'}"
        print(f"\nLoading {interval} hardened dataset ({window_desc}, 200-bar warmup + gap-aware stops)...")
        if whole_lots and not crypto_only:
            print("  Nifty options excluded in whole-lot mode: no historical contract lot and quote data.")

    if crypto_source == "delta":
        for sym in crypto_syms:
            if sym + "T" in cfg.lot_multipliers:
                cfg.lot_multipliers[sym] = cfg.lot_multipliers[sym + "T"]

    for sym in crypto_syms:
        if verbose:
            print(f"  [{crypto_source.upper()}] {sym}...")
        if crypto_source == "delta":
            asset_dfs[sym] = load_delta_candles(sym, resolution=interval, days=days, start_date=start_date, end_date=end_date)
        else:
            asset_dfs[sym] = load_binance(sym, interval=interval, days=days, start_date=start_date, end_date=end_date)

    for sym in commodities:
        if commodity_source == "upstox":
            sym_u = sym.upper().replace(".NS", "")
            exch = "MCX" if sym_u.startswith(("GOLD", "SILVERM", "SILVERMIC", "CRUDE")) and not sym_u.endswith(("BEES", "IETF", "CASE")) else "NSE"
            if verbose:
                print(f"  [Upstox V3 {exch}] {sym}...")
            asset_dfs[sym] = load_upstox(sym, interval=interval, days=days, exchange=exch, start_date=start_date, end_date=end_date)
        else:
            if verbose:
                print(f"  [Yahoo Finance] {sym}...")
            asset_dfs[sym] = load_yfinance(sym, interval=interval, days=days, start_date=start_date, end_date=end_date)

    donchian_trades, donchian_eq, diag = run_shared_donchian_portfolio(
        asset_dfs, crypto_set, cfg, eval_start=eval_start_ist, eval_end=eval_end_ist
    )

    if cfg.enable_credit_spreads:
        if use_upstox_nifty:
            if verbose:
                print("  [Upstox V3] NIFTY (Daily for Credit Spreads)...")
            spread_df = load_upstox("NIFTY", interval="1d", days=days, exchange="NSE", start_date=start_date, end_date=end_date)
        else:
            if verbose:
                print(f"  [Yahoo Finance] {cfg.spread_asset} (Daily for Credit Spreads)...")
            spread_df = load_yfinance(cfg.spread_asset, interval="1d", days=days, start_date=start_date, end_date=end_date)

        spread_trades, spread_cum_pnl, collateral_cum_pnl = run_weekly_credit_spreads(
            spread_df, cfg.spread_asset, cfg, eval_start=eval_start_ist, eval_end=eval_end_ist
        )
    else:
        spread_trades, spread_cum_pnl, collateral_cum_pnl = [], pd.Series(dtype=float), pd.Series(dtype=float)

    if not donchian_eq.empty and not spread_cum_pnl.empty:
        common_idx = donchian_eq.index.union(spread_cum_pnl.index).sort_values()
    elif not donchian_eq.empty:
        common_idx = donchian_eq.index
    else:
        common_idx = spread_cum_pnl.index

    donchian_pnl = (donchian_eq - cfg.initial_capital).reindex(common_idx).ffill().fillna(0.0)
    spread_pnl = spread_cum_pnl.reindex(common_idx).ffill().fillna(0.0)
    collat_pnl = collateral_cum_pnl.reindex(common_idx).ffill().fillna(0.0)

    donchian_w = (1.0 - spread_share) if cfg.enable_credit_spreads else 1.0
    spread_w = spread_share if cfg.enable_credit_spreads else 0.0

    eq_donchian = cfg.initial_capital + donchian_pnl
    eq_spreads_pure = cfg.initial_capital + spread_pnl
    eq_spreads_stacked = cfg.initial_capital + spread_pnl + collat_pnl
    eq_combined_pure = cfg.initial_capital + donchian_w * donchian_pnl + spread_w * spread_pnl
    eq_combined_total = cfg.initial_capital + donchian_w * donchian_pnl + spread_w * (spread_pnl + collat_pnl)

    all_trades = donchian_trades + spread_trades

    stats = {
        "donchian": calc_curve_stats("Engine A: Shared Donchian (Hardened)", eq_donchian, donchian_trades, cfg.initial_capital),
        "spreads_pure": calc_curve_stats("Engine B: Weekly Credit Spreads (Pure)", eq_spreads_pure, spread_trades, cfg.initial_capital),
        "spreads_stacked": calc_curve_stats("Engine B: Spreads + 7% Collateral Yield", eq_spreads_stacked, spread_trades, cfg.initial_capital, rf_apr=cfg.collateral_yield_apr),
        "combined_pure": calc_curve_stats(f"COMBINED ({donchian_w:.0%} A + {spread_w:.0%} B, Pure)", eq_combined_pure, all_trades, cfg.initial_capital),
        "combined_total": calc_curve_stats(f"COMBINED TOTAL ({donchian_w:.0%} A + {spread_w:.0%} B + Yield)", eq_combined_total, all_trades, cfg.initial_capital, rf_apr=cfg.collateral_yield_apr * spread_w),
    }

    return {
        "cfg": cfg,
        "interval": interval,
        "common_idx": common_idx,
        "diag": diag,
        "stats": stats,
        "donchian_trades": donchian_trades,
        "spread_trades": spread_trades,
        "all_trades": all_trades,
        "equity_df": pd.DataFrame({
            "donchian_equity": eq_donchian,
            "spreads_pure_equity": eq_spreads_pure,
            "combined_pure_equity": eq_combined_pure,
            "combined_total_equity": eq_combined_total,
        }),
    }


def run_two_broker_backtest(
    interval: str = "6h",
    days: int = 730,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    delta_capital_usd: float = 1200.0,
    kotak_capital_inr: float = 98_000.0,
    usd_inr_rate: float = 85.0,
    crypto_source: str = "delta",
    kotak_source: str = "upstox",
    maker_entry: bool = True,
    pyramid_add: float = 0.5,
    dd_brake: Optional[float] = 0.15,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Simulates the exact Rs 2,00,000 two-broker Config C portfolio with zero capital double-counting."""
    delta_syms = ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"] if crypto_source == "delta" else [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"
    ]
    entry_slip = 0.0004 if maker_entry else 0.0015
    delta_res = run_single_backtest(
        interval=interval, days=days, start_date=start_date, end_date=end_date,
        capital=delta_capital_usd, usd_inr_rate=1.0, risk=0.02, max_risk=0.08,
        max_crypto_concurrent=5, atr_mult=3.5, slippage=0.0015, entry_slippage=entry_slip,
        pyramid_add=pyramid_add, dd_brake=dd_brake, crypto_source=crypto_source,
        crypto_syms=delta_syms, crypto_only=True, whole_lots=True, verbose=verbose,
    )

    if kotak_source == "upstox":
        mcx_interval = interval
        _, _, eval_start_ist, eval_end_ist = resolve_date_window(mcx_interval, days, start_date, end_date, include_warmup=True)
        if verbose:
            print("  [Upstox V3 MCX] GOLDPETAL (native INR)...")
            print("  [Upstox V3 NSE] SILVERBEES (native INR)...")
        mcx_dfs = {
            "GOLDPETAL": load_upstox("GOLDPETAL", interval=mcx_interval, days=days, exchange="MCX", start_date=start_date, end_date=end_date),
            "SILVERBEES": load_upstox("SILVERBEES", interval=mcx_interval, days=days, exchange="NSE", start_date=start_date, end_date=end_date),
        }
        mcx_cfg = PortfolioConfig(
            initial_capital=kotak_capital_inr, usd_inr_rate=1.0,
            donchian_risk_per_trade_pct=0.025, max_concurrent_risk_pct=0.08,
            atr_trail_mult=3.0, slippage_pct=0.0005, entry_slippage_pct=0.0005,
            max_leverage=8.0, max_margin_utilization=0.75,
            pyramid_add_fraction=pyramid_add, dd_brake_threshold=dd_brake,
            whole_lot_rounding=True, enable_credit_spreads=False,
            long_only_assets={"SILVERBEES"},
        )
        kotak_label = "Sleeve 2: Kotak Neo (Upstox MCX GOLDPETAL + NSE SILVERBEES)"
    else:
        mcx_interval = "1d" if start_date else interval
        _, _, eval_start_ist, eval_end_ist = resolve_date_window(mcx_interval, days, start_date, end_date, include_warmup=True)
        mcx_dfs = {
            "GC=F": load_yfinance("GC=F", interval=mcx_interval, days=days, start_date=start_date, end_date=end_date),
            "SI=F": load_yfinance("SI=F", interval=mcx_interval, days=days, start_date=start_date, end_date=end_date),
        }
        mcx_cfg = PortfolioConfig(
            initial_capital=kotak_capital_inr, usd_inr_rate=usd_inr_rate,
            donchian_risk_per_trade_pct=0.025, max_concurrent_risk_pct=0.08,
            atr_trail_mult=3.0, slippage_pct=0.0005, entry_slippage_pct=0.0005,
            max_leverage=8.0, max_margin_utilization=0.75,
            pyramid_add_fraction=pyramid_add, dd_brake_threshold=dd_brake,
            whole_lot_rounding=True, enable_credit_spreads=False,
            long_only_assets={"SI=F"},
        )
        kotak_label = "Sleeve 2: Kotak Neo (Proxy GC=F + SI=F)"
    mcx_trades, mcx_eq, mcx_diag = run_shared_donchian_portfolio(
        mcx_dfs, set(), mcx_cfg, eval_start=eval_start_ist, eval_end=eval_end_ist
    )

    delta_eq_inr = delta_res["equity_df"]["donchian_equity"] * usd_inr_rate
    common_idx = delta_eq_inr.index.union(mcx_eq.index).sort_values()
    d_aligned = delta_eq_inr.reindex(common_idx).ffill().bfill()
    m_aligned = mcx_eq.reindex(common_idx).ffill().bfill()
    combined_eq = d_aligned + m_aligned
    total_init_inr = delta_capital_usd * usd_inr_rate + kotak_capital_inr
    corr = d_aligned.pct_change().dropna().corr(m_aligned.pct_change().dropna())

    return {
        "common_idx": common_idx,
        "total_init_inr": total_init_inr,
        "corr": float(corr) if pd.notna(corr) else 0.0,
        "kotak_source": kotak_source,
        "delta_res": delta_res,
        "mcx_trades": mcx_trades,
        "mcx_diag": mcx_diag,
        "stats": {
            "delta": delta_res["stats"]["donchian"],
            "kotak": calc_curve_stats(kotak_label, m_aligned, mcx_trades, kotak_capital_inr),
            "combined": calc_curve_stats("COMBINED (Rs 2L Two-Broker, Zero Double-Count)", combined_eq, delta_res["donchian_trades"] + mcx_trades, total_init_inr),
        },
        "equity_df": pd.DataFrame({
            "delta_equity_inr": d_aligned,
            "kotak_equity_inr": m_aligned,
            "combined_equity_inr": combined_eq,
        }),
    }


def main():
    parser = argparse.ArgumentParser(description="Hardened Portfolio Backtester (Config C Max Aggression + Two-Broker)")
    parser.add_argument("--interval", choices=list(SUPPORTED_INTERVALS), default="6h", help="Donchian timeframe (default: 6h)")
    parser.add_argument("--days", type=int, default=730, help="Lookback days if --start is not set (default: 730)")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD (e.g. 2021-01-01)")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD (e.g. 2023-12-31)")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Initial capital (default: 200000 INR)")
    parser.add_argument("--usd-inr", type=float, default=85.0, help="INR per USD for USD-quoted assets (default: 85.0)")
    parser.add_argument("--usd", action="store_true", help="Simulate realistic Delta India USD account ($1200, 5 coins, whole lots, Config C)")
    parser.add_argument("--two-broker", action="store_true", help="Simulate Rs 2L Two-Broker Config C ($1200 Delta + Rs 98k Kotak Neo)")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per Donchian trade (default: 0.02 = 2%%)")
    parser.add_argument("--max-risk", type=float, default=0.08, help="Max concurrent open risk (default: 0.08 = 8%%)")
    parser.add_argument("--atr-mult", type=float, default=3.5, help="ATR trailing stop multiple (default: 3.5)")
    parser.add_argument("--slippage", type=float, default=0.0015, help="Per-side stop/exit fee+slippage rate (default: 0.0015)")
    parser.add_argument("--entry-slippage", type=float, default=None, help="Optional entry cost rate (e.g. 0.0004 for maker)")
    parser.add_argument("--maker", action="store_true", help="Use realistic Delta maker entry cost (0.04%% entry / 0.15%% stop)")
    parser.add_argument("--pyramid", type=float, default=0.5, help="Winner pyramid add fraction (default: 0.5; 0 disables)")
    parser.add_argument("--dd-brake", type=float, default=0.15, help="Drawdown brake threshold (default: 0.15; 0 disables)")
    parser.add_argument("--spread-share", type=float, default=0.5, help="Capital weight for Engine B in combined stats (default: 0.5)")
    parser.add_argument("--enable-spreads", action="store_true", help="Include legacy synthetic Engine B Nifty spreads")
    parser.add_argument("--crypto-source", choices=["binance", "delta"], default="binance", help="Crypto candle source")
    parser.add_argument("--kotak-source", choices=["upstox", "yfinance"], default="upstox", help="Sleeve 2 candle source in --two-broker mode (default: upstox native INR)")
    parser.add_argument("--commodity-source", choices=["yfinance", "upstox"], default="yfinance", help="Commodity candle source in single-portfolio mode")
    parser.add_argument("--crypto", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"], help="Crypto symbols")
    parser.add_argument("--commodities", nargs="*", default=["GC=F"], help="Commodity/ETF symbols (default: GC=F)")
    parser.add_argument("--gold-etf", action="store_true", help="Replace GC=F with long-only GOLDIETF.NS")
    parser.add_argument("--crypto-only", action="store_true", help="Run Crypto Donchian only (great for 2021-2023 OOS)")
    parser.add_argument("--spread-asset", default="^NSEI", help="Asset for Weekly Credit Spreads (default: ^NSEI)")
    parser.add_argument("--use-upstox-nifty", action="store_true", help="Fetch Nifty daily data from Upstox V3")
    parser.add_argument("--whole-lots", action="store_true", help="Round Donchian lots and exclude unverified historical option lots")
    parser.add_argument("--spread-stress", type=float, default=0.0, help="Extra loss penalty on spread stop-outs (e.g. 0.15)")
    args = parser.parse_args()

    dd_brake = args.dd_brake if args.dd_brake > 0 else None

    if args.two_broker:
        src = "binance" if args.start else ("delta" if args.crypto_source == "binance" else args.crypto_source)
        tb = run_two_broker_backtest(
            interval=args.interval, days=args.days, start_date=args.start, end_date=args.end,
            usd_inr_rate=args.usd_inr, crypto_source=src, kotak_source=args.kotak_source,
            maker_entry=args.maker or args.entry_slippage is None,
            pyramid_add=args.pyramid, dd_brake=dd_brake, verbose=True,
        )
        idx = tb["common_idx"]
        print(f"\n================ TWO-BROKER CONFIG C SUMMARY ({idx[0].date()} -> {idx[-1].date()}) ================")
        print(f"Total Capital: Rs {tb['total_init_inr']:,.0f} ($1,200 Delta + Rs 98,000 Kotak Neo) | Kotak Source: {tb['kotak_source'].upper()} | Sleeve Corr: {tb['corr']:+.2f}\n")
        print(pd.DataFrame([tb["stats"]["delta"], tb["stats"]["kotak"], tb["stats"]["combined"]]).to_string(index=False))
        yr_end = tb["equity_df"]["combined_equity_inr"].resample("YE").last()
        yr_ret = yr_end.pct_change()
        yr_ret.iloc[0] = yr_end.iloc[0] / tb["total_init_inr"] - 1.0
        print("\nAnnual Combined Return: " + " | ".join(f"{d.year}: {v*100:+.1f}%" for d, v in yr_ret.items()))
        equity_path = new_run_directory("config-c") / "equity_curve.csv"
        tb["equity_df"].to_csv(equity_path)
        print(f"\nSaved: {equity_path}")
        return

    entry_slip = 0.0004 if args.maker else args.entry_slippage
    if args.usd:
        if args.capital == 200_000.0:
            args.capital = 1200.0
        args.usd_inr = 1.0
        args.crypto_source = "delta"
        if args.crypto == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]:
            args.crypto = ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "DOGEUSD"]
        args.crypto_only = True
        args.whole_lots = True
        if entry_slip is None:
            entry_slip = 0.0004

    res = run_single_backtest(
        interval=args.interval,
        days=args.days,
        start_date=args.start,
        end_date=args.end,
        capital=args.capital,
        usd_inr_rate=args.usd_inr,
        risk=args.risk,
        max_risk=args.max_risk,
        atr_mult=args.atr_mult,
        slippage=args.slippage,
        entry_slippage=entry_slip,
        pyramid_add=args.pyramid,
        dd_brake=dd_brake,
        spread_share=args.spread_share,
        crypto_source=args.crypto_source,
        commodity_source=args.commodity_source,
        crypto_syms=args.crypto,
        commodities=args.commodities,
        gold_etf=args.gold_etf,
        crypto_only=args.crypto_only,
        enable_credit_spreads=args.enable_spreads,
        spread_asset=args.spread_asset,
        use_upstox_nifty=args.use_upstox_nifty,
        whole_lots=args.whole_lots,
        spread_stress=args.spread_stress,
        verbose=True,
    )

    cfg = res["cfg"]
    common_idx = res["common_idx"]
    diag = res["diag"]
    all_trades = res["all_trades"]
    donchian_trades = res["donchian_trades"]
    if common_idx.empty:
        parser.exit(1, "No bars remain in the evaluation window after warmup.\n")

    ccy = "$" if args.usd else "Rs"
    print(f"\n================ HARDENED PORTFOLIO SUMMARY ({common_idx[0].date()} -> {common_idx[-1].date()}) ================")
    print(
        f"Capital: {ccy} {cfg.initial_capital:,.0f} | Risk/Trade: {cfg.donchian_risk_per_trade_pct*100:.1f}% | "
        f"Max Risk: {cfg.max_concurrent_risk_pct*100:.1f}% | ATR: {cfg.atr_trail_mult:.1f}x | TF: {args.interval} | "
        f"Pyramids: {diag.get('pyramids', 0)} | DD-Brake Bars: {diag.get('dd_brake_bars', 0)} | "
        f"Gap-Stops: {diag['gap_stops']} | Skipped (Lot Size): {diag['skipped_lot_size']}\n"
    )

    rows = [res["stats"]["donchian"]]
    if cfg.enable_credit_spreads:
        rows.extend([
            res["stats"]["spreads_pure"],
            res["stats"]["spreads_stacked"],
            res["stats"]["combined_pure"],
            res["stats"]["combined_total"],
        ])
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n================ PER-ASSET BREAKDOWN ================\n")
    print(summarize_by_asset(all_trades, cfg.initial_capital, currency=ccy).to_string(index=False))

    if donchian_trades:
        t_df = pd.DataFrame([t.__dict__ for t in all_trades])
        t_df["year"] = pd.to_datetime(t_df["exit_time"]).dt.year
        print(f"\n================ ANNUAL REALIZED PnL ({ccy}) ================\n")
        print(t_df.groupby(["year", "engine"])["net_pnl"].sum().unstack(fill_value=0).round(2 if args.usd else 0).to_string())

        d_pnls = sorted([t.net_pnl for t in donchian_trades], reverse=True)
        tot_d = sum(d_pnls)
        top3 = sum(d_pnls[:3])
        print(f"\nDonchian Top-3 Trades Share: {ccy} {top3:,.2f} / {ccy} {tot_d:,.2f} ({(top3/tot_d*100) if tot_d else 0:.1f}%) | Ex-Top-3: {ccy} {tot_d - top3:,.2f}")

    output_dir = new_run_directory("portfolio-backtest")
    pd.DataFrame([t.__dict__ for t in all_trades]).to_csv(output_dir / "trades.csv", index=False)
    res["equity_df"].to_csv(output_dir / "equity_curve.csv")
    print(f"\nSaved trades and equity curve to {output_dir}/")


if __name__ == "__main__":
    main()
