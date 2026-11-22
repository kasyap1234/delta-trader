import gzip
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UPSTOX_BASE = "https://api.upstox.com/v3"
UPSTOX_MASTERS = {
    "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "MCX": "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz",
}
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"
DELTA_INDIA_BASE = "https://api.india.delta.exchange"

SUPPORTED_INTERVALS = ("1h", "2h", "4h", "6h", "12h", "1d")

# Calendar days needed to guarantee >=200 warmup bars (accounting for weekends on equities/metals)
WARMUP_CALENDAR_DAYS = {
    "1h": 20,
    "2h": 35,
    "4h": 60,
    "6h": 85,
    "12h": 160,
    "1d": 320,
}


def _normalize_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly normalizes all timestamps to Asia/Kolkata (pandas 3.x safe)."""
    idx = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df.index = idx.tz_convert("Asia/Kolkata")
    df.index.name = "timestamp"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = "1D" if interval == "1d" else interval
    return (
        df[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def resolve_date_window(
    interval: str,
    days: int = 730,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_warmup: bool = True,
) -> Tuple[datetime, datetime, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Returns (fetch_start_utc, fetch_end_utc, eval_start_ist, eval_end_ist).
    Automatically pads fetch_start_utc backward by WARMUP_CALENDAR_DAYS[interval]
    so 200-bar EMA warmup finishes right before start_date.
    """
    now_utc = datetime.now(timezone.utc)
    if end_date:
        eval_end_ist = pd.Timestamp(datetime.strptime(end_date, "%Y-%m-%d"), tz="Asia/Kolkata") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        end_dt = eval_end_ist.tz_convert("UTC").to_pydatetime()
    else:
        end_dt = now_utc
        eval_end_ist = None

    if start_date:
        eval_start_ist = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"), tz="Asia/Kolkata")
        base_start = eval_start_ist.tz_convert("UTC").to_pydatetime()
    else:
        base_start = end_dt - timedelta(days=days)
        eval_start_ist = pd.Timestamp(base_start).tz_convert("Asia/Kolkata")

    if base_start >= end_dt:
        raise ValueError("Backtest start must precede its end")
    pad_days = WARMUP_CALENDAR_DAYS.get(interval, 60) if include_warmup else 0
    fetch_start = base_start - timedelta(days=pad_days)
    return fetch_start, end_dt, eval_start_ist, eval_end_ist


def load_binance(
    symbol: str,
    interval: str = "4h",
    days: int = 730,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_warmup: bool = True,
) -> pd.DataFrame:
    """
    Free Binance USD-M Futures klines (supports 1h, 2h, 4h, 6h, 12h, 1d natively back to 2020).
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval}. Choose from {SUPPORTED_INTERVALS}")

    fetch_start, fetch_end, _, _ = resolve_date_window(interval, days, start_date, end_date, include_warmup)
    start_ms = int(fetch_start.timestamp() * 1000)
    end_ms = int(fetch_end.timestamp() * 1000)

    all_rows = []
    cur_start = start_ms
    while cur_start < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": 1500,
        }
        r = requests.get(BINANCE_FAPI, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        if last_open_time <= cur_start:
            break
        cur_start = last_open_time + 1
        time.sleep(0.12)

    if not all_rows:
        raise RuntimeError(f"No Binance data returned for {symbol}")

    df = pd.DataFrame(
        all_rows,
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")
    return _normalize_ist_index(df)


def load_delta_candles(
    symbol: str,
    resolution: str = "4h",
    days: int = 730,
    base_url: str = DELTA_INDIA_BASE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_warmup: bool = True,
) -> pd.DataFrame:
    """
    Free public Delta Exchange India historical candles (/v2/history/candles, no API key needed).
    Native support: 1h, 2h, 4h, 6h, 1d. For 12h, fetches 6h and resamples cleanly.
    """
    if resolution not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported resolution {resolution}. Choose from {SUPPORTED_INTERVALS}")

    api_res = "6h" if resolution == "12h" else resolution
    fetch_start, fetch_end, _, _ = resolve_date_window(resolution, days, start_date, end_date, include_warmup)
    start_s = int(fetch_start.timestamp())
    end_s = int(fetch_end.timestamp())

    step_map = {"1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400}
    chunk_sec = step_map[api_res] * 1800

    all_rows = []
    cur_start = start_s
    while cur_start < end_s:
        cur_end = min(end_s, cur_start + chunk_sec)
        url = f"{base_url}/v2/history/candles"
        params = {"resolution": api_res, "symbol": symbol.upper(), "start": cur_start, "end": cur_end}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json().get("result", [])
        if rows:
            all_rows.extend(rows)
        cur_start = cur_end + 1
        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError(f"No Delta Exchange candles returned for {symbol}")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")
    df = _normalize_ist_index(df)
    if resolution == "12h":
        df = _resample_ohlcv(df, "12h")
    return df


def load_yfinance(
    symbol: str,
    interval: str = "4h",
    days: int = 730,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_warmup: bool = True,
) -> pd.DataFrame:
    """
    Loads Yahoo Finance OHLCV:
      - Daily ('1d'): supports arbitrary multi-year start_date/end_date (e.g. 2020-2026).
      - Intraday ('1h','2h','4h','6h','12h'): Yahoo caps 1h history at 729 days; fetches 1h and resamples.
    """
    import yfinance as yf

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval}. Choose from {SUPPORTED_INTERVALS}")

    fetch_start, fetch_end, _, _ = resolve_date_window(interval, days, start_date, end_date, include_warmup)

    if interval == "1d":
        start_str = fetch_start.strftime("%Y-%m-%d")
        end_str = (fetch_end + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start_str, end=end_str, interval="1d", progress=False)
    else:
        # Yahoo Finance 1h endpoint strictly rejects dates older than 729 days
        min_allowed = datetime.now(timezone.utc) - timedelta(days=728)
        if fetch_start < min_allowed:
            print(f"  [Note] Yahoo Finance caps {interval} data at 729 days for {symbol}; clamping to recent 729d. Use --interval 1d or --crypto-only for 2021-2023 tests.")
            fetch_start = min_allowed
        if fetch_end <= min_allowed:
            raise ValueError(
                f"Yahoo Finance does not serve intraday ({interval}) data for {symbol} ending {end_date} (>729 days ago). "
                f"Use --interval 1d for multi-year portfolio OOS, or --crypto-only for 2021-2023 intraday crypto OOS."
            )
        df = yf.download(
            symbol,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=(fetch_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1h",
            progress=False,
        )

    if df.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {symbol} ({interval})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = _normalize_ist_index(df)

    if interval in ("2h", "4h", "6h", "12h"):
        df = _resample_ohlcv(df, interval)
    return df


def resolve_upstox_key(symbol: str, exchange: str = "NSE") -> str:
    if "|" in symbol:
        return symbol
    sym_upper = symbol.upper()
    if sym_upper in ("NIFTY", "NIFTY50", "^NSEI"):
        return "NSE_INDEX|Nifty 50"
    if sym_upper in ("BANKNIFTY", "^NSEBANK"):
        return "NSE_INDEX|Nifty Bank"

    exch = "MCX" if exchange.upper() == "MCX" or sym_upper.startswith(("CRUDE", "GOLDM", "SILVERM")) else "NSE"
    resp = requests.get(UPSTOX_MASTERS[exch], timeout=20)
    resp.raise_for_status()
    instruments = json.loads(gzip.decompress(resp.content).decode("utf-8"))

    if exch == "MCX":
        matches = [
            inst for inst in instruments
            if inst.get("instrument_type") == "FUT"
            and inst.get("trading_symbol", "").startswith(sym_upper)
            and inst.get("expiry") is not None
        ]
        if not matches:
            raise ValueError(f"No active MCX futures found for {sym_upper}")
        matches.sort(key=lambda x: x["expiry"])
        return matches[0]["instrument_key"]
    else:
        matches = [
            inst for inst in instruments
            if inst.get("segment") == "NSE_EQ"
            and inst.get("trading_symbol", "").upper() == sym_upper
        ]
        if not matches:
            raise ValueError(f"No NSE_EQ instrument found for {sym_upper}")
        return matches[0]["instrument_key"]


def load_upstox(
    symbol: str,
    interval: str = "1d",
    days: int = 730,
    exchange: str = "NSE",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_warmup: bool = True,
) -> pd.DataFrame:
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise ValueError("UPSTOX_ACCESS_TOKEN missing in .env")

    key = resolve_upstox_key(symbol, exchange)
    fetch_start, fetch_end, _, _ = resolve_date_window(interval, days, start_date, end_date, include_warmup)

    if interval == "1d":
        unit, step, chunk_days = "days", 1, 365
    else:
        # Fetch 1h from Upstox V3 and resample to 2h/4h/6h/12h for consistent alignment
        unit, step, chunk_days = "hours", 1, 85

    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    encoded_key = quote(key, safe="")

    frames = []
    cur_end = fetch_end
    while cur_end > fetch_start:
        cur_start = max(fetch_start, cur_end - timedelta(days=chunk_days))
        url = f"{UPSTOX_BASE}/historical-candle/{encoded_key}/{unit}/{step}/{cur_end.strftime('%Y-%m-%d')}/{cur_start.strftime('%Y-%m-%d')}"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            candles = r.json().get("data", {}).get("candles", [])
            if candles:
                df_c = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
                df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], utc=True)
                frames.append(df_c[["timestamp", "open", "high", "low", "close", "volume"]])
        cur_end = cur_start - timedelta(days=1)
        time.sleep(0.2)

    if not frames:
        raise RuntimeError(f"No Upstox data for {symbol}")
    df = pd.concat(frames, ignore_index=True).set_index("timestamp")
    df = _normalize_ist_index(df)
    if interval in ("2h", "4h", "6h", "12h"):
        df = _resample_ohlcv(df, interval)
    return df
