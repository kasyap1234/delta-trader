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


def calc_curve_stats(name: str, equity_curve: pd.Series, trades: List[PortfolioTrade], initial_capital: float) -> Dict[str, Any]:
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
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(365)) if daily_rets.std() > 1e-9 else 0.0

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


def summarize_by_asset(trades: List[PortfolioTrade], initial_capital: float) -> pd.DataFrame:
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
            "Funding (Rs)": round(funding, 0),
            "Net PnL (Rs)": round(float(pnls.sum()), 0),
            "Contrib %": round(float(pnls.sum()) / initial_capital * 100.0, 2),
        })
    return pd.DataFrame(rows)


def run_single_backtest(
    interval: str = "4h",
    days: int = 730,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk: float = 0.01,
    max_risk: float = 0.03,
    crypto_source: str = "binance",
    crypto_syms: Optional[List[str]] = None,
    commodities: Optional[List[str]] = None,
    gold_etf: bool = False,
    crypto_only: bool = False,
    spread_asset: str = "^NSEI",
    use_upstox_nifty: bool = False,
    whole_lots: bool = False,
    spread_stress: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    crypto_syms = list(crypto_syms or ["ETHUSDT", "SOLUSDT"])
    if crypto_source == "delta":
        crypto_syms = [s.replace("USDT", "USD") for s in crypto_syms]

    if crypto_only:
        commodities = []
    elif gold_etf:
        commodities = ["GOLDIETF.NS"]
    else:
        commodities = list(commodities) if commodities is not None else ["GC=F"]

    cfg = PortfolioConfig(
        donchian_risk_per_trade_pct=risk,
        max_concurrent_risk_pct=max_risk,
        crypto_assets=crypto_syms,
        commodity_assets=commodities,
        spread_asset="NIFTY" if use_upstox_nifty else spread_asset,
        enable_credit_spreads=not crypto_only,
        whole_lot_rounding=whole_lots,
        spread_stress_penalty=spread_stress,
    )

    _, _, eval_start_ist, eval_end_ist = resolve_date_window(interval, days, start_date, end_date, include_warmup=True)

    asset_dfs: Dict[str, pd.DataFrame] = {}
    crypto_set = set(crypto_syms)

    if verbose:
        window_desc = f"{start_date or f'last {days}d'} -> {end_date or 'today'}"
        print(f"\nLoading {interval} hardened dataset ({window_desc}, 200-bar warmup + gap-aware stops)...")

    for sym in crypto_syms:
        if verbose:
            print(f"  [{crypto_source.upper()}] {sym}...")
        if crypto_source == "delta":
            asset_dfs[sym] = load_delta_candles(sym, resolution=interval, days=days, start_date=start_date, end_date=end_date)
        else:
            asset_dfs[sym] = load_binance(sym, interval=interval, days=days, start_date=start_date, end_date=end_date)

    for sym in commodities:
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
        start_dt = max(donchian_eq.index[0], spread_cum_pnl.index[0])
        common_idx = donchian_eq.index.union(spread_cum_pnl.index).sort_values()
        common_idx = common_idx[common_idx >= start_dt]
    elif not donchian_eq.empty:
        common_idx = donchian_eq.index
    else:
        common_idx = spread_cum_pnl.index

    donchian_pnl = (donchian_eq.reindex(common_idx).ffill().bfill() - cfg.initial_capital) if not donchian_eq.empty else pd.Series(0.0, index=common_idx)
    if not spread_cum_pnl.empty:
        spread_pnl = spread_cum_pnl.reindex(common_idx).ffill().fillna(0.0)
        spread_pnl = spread_pnl - spread_pnl.iloc[0]
        collat_pnl = collateral_cum_pnl.reindex(common_idx).ffill().fillna(0.0)
        collat_pnl = collat_pnl - collat_pnl.iloc[0]
    else:
        spread_pnl = pd.Series(0.0, index=common_idx)
        collat_pnl = pd.Series(0.0, index=common_idx)

    eq_donchian = cfg.initial_capital + donchian_pnl
    eq_spreads_pure = cfg.initial_capital + spread_pnl
    eq_spreads_stacked = cfg.initial_capital + spread_pnl + collat_pnl
    eq_combined_pure = cfg.initial_capital + donchian_pnl + spread_pnl
    eq_combined_total = cfg.initial_capital + donchian_pnl + spread_pnl + collat_pnl

    all_trades = donchian_trades + spread_trades

    stats = {
        "donchian": calc_curve_stats("Engine A: Shared Donchian (Hardened)", eq_donchian, donchian_trades, cfg.initial_capital),
        "spreads_pure": calc_curve_stats("Engine B: Weekly Credit Spreads (Pure)", eq_spreads_pure, spread_trades, cfg.initial_capital),
        "spreads_stacked": calc_curve_stats("Engine B: Spreads + 7% Collateral Yield", eq_spreads_stacked, spread_trades, cfg.initial_capital),
        "combined_pure": calc_curve_stats("COMBINED (Engine A + B, Pure Trading)", eq_combined_pure, all_trades, cfg.initial_capital),
        "combined_total": calc_curve_stats("COMBINED TOTAL (A + B + 7% Collateral)", eq_combined_total, all_trades, cfg.initial_capital),
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


def main():
    parser = argparse.ArgumentParser(description="Hardened Test-2 Portfolio Backtester (Multi-Timeframe + Date Ranges)")
    parser.add_argument("--interval", choices=list(SUPPORTED_INTERVALS), default="4h", help="Donchian timeframe (1h, 2h, 4h, 6h, 12h, 1d)")
    parser.add_argument("--days", type=int, default=730, help="Lookback days if --start is not set (default: 730)")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD (e.g. 2021-01-01)")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD (e.g. 2023-12-31)")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per Donchian trade (default: 0.01 = 1%%)")
    parser.add_argument("--max-risk", type=float, default=0.03, help="Max concurrent open risk (default: 0.03 = 3%%)")
    parser.add_argument("--crypto-source", choices=["binance", "delta"], default="binance", help="Crypto candle source")
    parser.add_argument("--crypto", nargs="+", default=["ETHUSDT", "SOLUSDT"], help="Crypto symbols")
    parser.add_argument("--commodities", nargs="*", default=["GC=F"], help="Commodity/ETF symbols (default: GC=F)")
    parser.add_argument("--gold-etf", action="store_true", help="Replace GC=F with long-only GOLDIETF.NS")
    parser.add_argument("--crypto-only", action="store_true", help="Run Crypto Donchian only (great for 2021-2023 4H OOS)")
    parser.add_argument("--spread-asset", default="^NSEI", help="Asset for Weekly Credit Spreads (default: ^NSEI)")
    parser.add_argument("--use-upstox-nifty", action="store_true", help="Fetch Nifty daily data from Upstox V3")
    parser.add_argument("--whole-lots", action="store_true", help="Enforce strict whole-lot/contract rounding")
    parser.add_argument("--spread-stress", type=float, default=0.0, help="Extra loss penalty on spread stop-outs (e.g. 0.15)")
    args = parser.parse_args()

    res = run_single_backtest(
        interval=args.interval,
        days=args.days,
        start_date=args.start,
        end_date=args.end,
        risk=args.risk,
        max_risk=args.max_risk,
        crypto_source=args.crypto_source,
        crypto_syms=args.crypto,
        commodities=args.commodities,
        gold_etf=args.gold_etf,
        crypto_only=args.crypto_only,
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

    print(f"\n================ HARDENED PORTFOLIO SUMMARY ({common_idx[0].date()} -> {common_idx[-1].date()}) ================")
    print(
        f"Capital: Rs {cfg.initial_capital:,.0f} | Risk/Trade: {cfg.donchian_risk_per_trade_pct*100:.1f}% | "
        f"Max Risk: {cfg.max_concurrent_risk_pct*100:.1f}% | TF: {args.interval} | "
        f"Warmup: {cfg.ema_warmup_bars} bars | Gap-Stops: {diag['gap_stops']} | Skipped (Lot Size): {diag['skipped_lot_size']}\n"
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
    print(summarize_by_asset(all_trades, cfg.initial_capital).to_string(index=False))

    if donchian_trades:
        t_df = pd.DataFrame([t.__dict__ for t in all_trades])
        t_df["year"] = pd.to_datetime(t_df["exit_time"]).dt.year
        print("\n================ ANNUAL REALIZED PnL (Rs) ================\n")
        print(t_df.groupby(["year", "engine"])["net_pnl"].sum().unstack(fill_value=0).round(0).to_string())

        d_pnls = sorted([t.net_pnl for t in donchian_trades], reverse=True)
        tot_d = sum(d_pnls)
        top3 = sum(d_pnls[:3])
        print(f"\nDonchian Top-3 Trades Share: Rs {top3:,.0f} / Rs {tot_d:,.0f} ({(top3/tot_d*100) if tot_d else 0:.1f}%) | Ex-Top-3: Rs {tot_d - top3:,.0f}")

    pd.DataFrame([t.__dict__ for t in all_trades]).to_csv("hardened_trades.csv", index=False)
    res["equity_df"].to_csv("hardened_equity_curve.csv")
    print("\nSaved: hardened_trades.csv and hardened_equity_curve.csv")


if __name__ == "__main__":
    main()
