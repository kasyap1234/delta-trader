"""
ONE-COMMAND TIMEFRAME PLATEAU & OUT-OF-SAMPLE SWEEP RUNNER

Runs the exact same hardened portfolio across multiple candle timeframes
(default: 2h, 4h, 6h, 12h, 1d) and prints a side-by-side comparison table
to verify that the edge is a broad plateau rather than a lucky 4h spike.
"""

import argparse
from typing import List

import pandas as pd

from data_loader import SUPPORTED_INTERVALS
from run_hardened_backtest import run_single_backtest


def main():
    parser = argparse.ArgumentParser(description="Side-by-Side Timeframe Plateau & OOS Sweep")
    parser.add_argument(
        "--intervals",
        nargs="+",
        choices=list(SUPPORTED_INTERVALS),
        default=["2h", "4h", "6h", "12h", "1d"],
        help="Timeframes to compare side-by-side (default: 2h 4h 6h 12h 1d)",
    )
    parser.add_argument("--days", type=int, default=730, help="Lookback days if --start not set (default: 730)")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD (e.g. 2021-01-01)")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD (e.g. 2023-12-31)")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per Donchian trade (default: 0.01 = 1%%)")
    parser.add_argument("--max-risk", type=float, default=0.03, help="Max concurrent open risk (default: 0.03 = 3%%)")
    parser.add_argument("--crypto-source", choices=["binance", "delta"], default="binance")
    parser.add_argument("--crypto", nargs="+", default=["ETHUSDT", "SOLUSDT"])
    parser.add_argument("--commodities", nargs="*", default=["GC=F"])
    parser.add_argument("--gold-etf", action="store_true", help="Use long-only GOLDIETF.NS instead of GC=F")
    parser.add_argument("--crypto-only", action="store_true", help="Crypto Donchian only (unlocks 2021-2023 intraday OOS)")
    parser.add_argument("--whole-lots", action="store_true", help="Enforce strict whole-lot rounding")
    args = parser.parse_args()

    donchian_rows = []
    combined_rows = []

    print(f"\nRunning Timeframe Sweep: {args.intervals} | Window: {args.start or f'last {args.days}d'} -> {args.end or 'today'}")
    print(f"Risk/Trade: {args.risk*100:.1f}% | Max Concurrent Risk: {args.max_risk*100:.1f}% | Crypto-Only: {args.crypto_only}\n")

    for tf in args.intervals:
        print(f">>> Backtesting timeframe: {tf} ...")
        try:
            res = run_single_backtest(
                interval=tf,
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
                whole_lots=args.whole_lots,
                verbose=False,
            )
            d = res["stats"]["donchian"]
            c_pure = res["stats"]["combined_pure"]
            c_tot = res["stats"]["combined_total"]
            idx = res["common_idx"]
            period_str = f"{idx[0].date()}..{idx[-1].date()}" if len(idx) else "N/A"

            donchian_rows.append({
                "TF": tf,
                "Period": period_str,
                "Trades": d["Trades"],
                "Win %": d["Win %"],
                "PF": d["PF"],
                "Net Ret %": d["Net Return %"],
                "CAGR %": d["CAGR %"],
                "Max DD %": d["Max DD %"],
                "Calmar": d["Calmar"],
                "Sharpe": d["Sharpe"],
                "Gap Stops": res["diag"]["gap_stops"],
            })

            if not args.crypto_only:
                combined_rows.append({
                    "TF": tf,
                    "Period": period_str,
                    "Trades": c_tot["Trades"],
                    "PF": c_tot["PF"],
                    "Pure Ret %": c_pure["Net Return %"],
                    "Total Ret %": c_tot["Net Return %"],
                    "Total CAGR %": c_tot["CAGR %"],
                    "Max DD %": c_tot["Max DD %"],
                    "Calmar": c_tot["Calmar"],
                    "Sharpe": c_tot["Sharpe"],
                })
        except Exception as e:
            print(f"  [ERROR on {tf}]: {e}")

    print("\n================ ENGINE A: DONCHIAN TIMEFRAME PLATEAU ================\n")
    df_d = pd.DataFrame(donchian_rows)
    print(df_d.to_string(index=False))

    if combined_rows:
        print("\n================ COMBINED PORTFOLIO (DONCHIAN + SPREADS + 7% YIELD) ================\n")
        df_c = pd.DataFrame(combined_rows)
        print(df_c.to_string(index=False))
        df_c.to_csv("timeframe_sweep_combined.csv", index=False)

    df_d.to_csv("timeframe_sweep_donchian.csv", index=False)
    print("\nSaved: timeframe_sweep_donchian.csv" + (" and timeframe_sweep_combined.csv" if combined_rows else ""))


if __name__ == "__main__":
    main()
