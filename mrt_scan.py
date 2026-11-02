#!/usr/bin/env python3
"""
mrt_scan.py  -  daily scan for the Mean Reversion Paper Tracker

Pulls daily OHLC for the rotation from Yahoo Finance (free, no key), computes
Bollinger Bands, Keltner Channels, RSI, ADX and ATR, looks up the next earnings
date, and writes a CSV the tracker's "Import scan" button understands.

Setup (once):
    pip install yfinance pandas numpy

Run (after the close, or any time for the latest completed bar):
    python mrt_scan.py                      # writes scan.csv next to this file
    python mrt_scan.py --out ~/Desktop/scan.csv
    python mrt_scan.py --tickers my_list.txt   # one symbol per line, overrides the built-in list
    python mrt_scan.py --no-earnings           # skip the (slow) earnings lookups
    python mrt_scan.py --print                 # also print the CSV to the terminal for copy/paste

Indicator settings match common charting defaults; change them below if your
chart uses different ones so the scan agrees with what you see.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Rotation (same list as the tracker). ETFs get no earnings lookup.
# ----------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "AMD", "MU", "QCOM", "AVGO",
    "CRM", "ADBE", "ORCL", "PLTR", "UBER", "NFLX",
    "COIN", "HOOD", "PYPL",
    "JPM", "GS", "MS", "SCHW", "V",
    "TSLA", "HD", "NKE", "SBUX", "DIS", "TGT",
    "UAL", "ABNB",
    "WMT", "PG",
    "UNH", "ABBV", "MRK", "CVS", "ISRG",
    "XOM", "CVX", "COP",
    "BA", "CAT", "GE", "FDX",
    "FCX", "NEM", "NUE",
    "BABA",
    "SPY", "QQQ", "IWM", "XLE", "XLF", "GLD",
]
ETFS = {"SPY", "QQQ", "IWM", "XLE", "XLF", "GLD"}

# ----------------------------------------------------------------------------
# Indicator settings
# ----------------------------------------------------------------------------
BB_LEN, BB_MULT = 20, 2.0           # Bollinger Bands: 20 SMA, 2 population std devs
KC_LEN, KC_MULT, KC_ATR_LEN = 20, 1.5, 20   # Keltner: 20 EMA, 1.5 x ATR(20) (TTM squeeze convention)
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
BAND_TOL_ATR = 0.0   # extra reach, as a fraction of ATR. 0 = the day's range must touch the band.
HISTORY = "9mo"      # enough bars for the EMAs to settle
EARNINGS_CACHE_DAYS = 7


# ----------------------------------------------------------------------------
# Indicators (Wilder smoothing where the textbook uses it)
# ----------------------------------------------------------------------------
def wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(df), n)


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder(gain, n)
    avg_loss = wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss != 0, 100.0)


def adx(df: pd.DataFrame, n: int):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = wilder(true_range(df), n)
    plus_di = 100.0 * wilder(plus_dm, n) / tr_s
    minus_di = 100.0 * wilder(minus_dm, n) / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx.fillna(0.0), n), plus_di, minus_di


def bollinger(close: pd.Series, n: int, k: float):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def keltner(df: pd.DataFrame, n: int, k: float, atr_len: int):
    mid = df["Close"].ewm(span=n, adjust=False).mean()
    a = atr(df, atr_len)
    return mid + k * a, mid, mid - k * a


def scan_frame(ticker: str, df: pd.DataFrame):
    """Return one scan row (dict) for a ticker's daily OHLC frame, or None if too short."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < max(BB_LEN, KC_LEN, ADX_LEN * 3) + 5:
        return None
    close = df["Close"]
    bb_u, bb_m, bb_l = bollinger(close, BB_LEN, BB_MULT)
    kc_u, kc_m, kc_l = keltner(df, KC_LEN, KC_MULT, KC_ATR_LEN)
    a = atr(df, ATR_LEN)
    r = rsi(close, RSI_LEN)
    x, _, _ = adx(df, ADX_LEN)

    last = df.iloc[-1]
    tol = BAND_TOL_ATR * float(a.iloc[-1])
    touched_lower = float(last["Low"]) <= float(bb_l.iloc[-1]) + tol
    touched_upper = float(last["High"]) >= float(bb_u.iloc[-1]) - tol
    if touched_lower and touched_upper:
        bb = "lower" if float(last["Close"]) < float(bb_m.iloc[-1]) else "upper"
    elif touched_lower:
        bb = "lower"
    elif touched_upper:
        bb = "upper"
    else:
        bb = "none"
    squeeze = bool(float(bb_u.iloc[-1]) < float(kc_u.iloc[-1]) and float(bb_l.iloc[-1]) > float(kc_l.iloc[-1]))

    return {
        "ticker": ticker,
        "price": round(float(last["Close"]), 2),
        "atr": round(float(a.iloc[-1]), 2),
        "earnings": "",
        "bb": bb,
        "squeeze": "true" if squeeze else "false",
        "rsi": round(float(r.iloc[-1]), 1),
        "adx": round(float(x.iloc[-1]), 1),
        "notes": "",
        "bb_upper": round(float(bb_u.iloc[-1]), 2),
        "bb_lower": round(float(bb_l.iloc[-1]), 2),
        "kc_upper": round(float(kc_u.iloc[-1]), 2),
        "kc_lower": round(float(kc_l.iloc[-1]), 2),
        "asof": pd.Timestamp(df.index[-1]).date().isoformat(),
    }


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def frame_for(data: pd.DataFrame, ticker: str):
    """Pull one ticker's OHLC frame out of a yf.download result, whichever way it came back."""
    if isinstance(data.columns, pd.MultiIndex):
        lv0 = data.columns.get_level_values(0)
        lv1 = data.columns.get_level_values(1)
        if ticker in lv0:
            df = data[ticker]
        elif ticker in lv1:
            df = data.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        df = data
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return None
    return df


def load_prices(tickers, period):
    import yfinance as yf
    data = yf.download(
        tickers, period=period, interval="1d", group_by="ticker",
        auto_adjust=False, threads=True, progress=False,
    )
    if data is None or len(data) == 0:
        raise SystemExit("No price data came back. Check your connection or try again in a minute.")
    return data


def next_earnings(ticker: str, cache: dict, today: dt.date):
    """Next earnings date as ISO string, with a small on-disk cache so this is not fetched every run."""
    c = cache.get(ticker)
    if c:
        try:
            checked = dt.date.fromisoformat(c["checked"])
            d = c.get("date")
            still_future = (d is None) or (dt.date.fromisoformat(d) >= today)
            if (today - checked).days < EARNINGS_CACHE_DAYS and still_future:
                return d or ""
        except Exception:
            pass

    import yfinance as yf
    date = None
    try:
        tk = yf.Ticker(ticker)
        candidates = []
        cal = None
        try:
            cal = tk.calendar
        except Exception:
            cal = None
        if isinstance(cal, dict):
            v = cal.get("Earnings Date") or []
            candidates = list(v) if isinstance(v, (list, tuple, pd.Series)) else [v]
        elif cal is not None and hasattr(cal, "loc"):
            try:
                candidates = list(cal.loc["Earnings Date"])
            except Exception:
                candidates = []
        dates = []
        for d in candidates:
            try:
                dates.append(pd.Timestamp(d).date())
            except Exception:
                continue
        future = sorted(d for d in dates if d >= today)
        if future:
            date = future[0].isoformat()
        if date is None:
            try:
                ed = tk.get_earnings_dates(limit=12)
                if ed is not None and len(ed):
                    fut = sorted(pd.Timestamp(i).date() for i in ed.index if pd.Timestamp(i).date() >= today)
                    if fut:
                        date = fut[0].isoformat()
            except Exception:
                pass
    except Exception:
        date = None
    cache[ticker] = {"date": date, "checked": today.isoformat()}
    return date or ""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
COLUMNS = ["ticker", "price", "atr", "earnings", "bb", "squeeze", "rsi", "adx", "notes",
           "bb_upper", "bb_lower", "kc_upper", "kc_lower", "asof"]


def main():
    ap = argparse.ArgumentParser(description="Daily scan for the Mean Reversion Paper Tracker")
    here = Path(__file__).resolve().parent
    ap.add_argument("--out", default=str(here / "scan.csv"), help="where to write the CSV")
    ap.add_argument("--tickers", help="text file with one symbol per line (overrides the built-in list)")
    ap.add_argument("--no-earnings", action="store_true", help="skip earnings-date lookups")
    ap.add_argument("--print", action="store_true", help="also print the CSV to stdout")
    ap.add_argument("--period", default=HISTORY, help="history to download (default %(default)s)")
    args = ap.parse_args()

    tickers = UNIVERSE
    if args.tickers:
        tickers = [ln.strip().upper() for ln in Path(args.tickers).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if not tickers:
        raise SystemExit("No tickers to scan.")

    try:
        import yfinance  # noqa: F401
    except ImportError:
        raise SystemExit("yfinance is not installed. Run:  pip install yfinance pandas numpy")

    today = dt.date.today()
    print(f"Downloading {len(tickers)} symbols ({args.period} of daily bars)...")
    data = load_prices(tickers, args.period)

    cache_path = here / ".earnings_cache.json"
    cache = {}
    if not args.no_earnings and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    rows, missing = [], []
    for t in tickers:
        df = frame_for(data, t)
        row = scan_frame(t, df) if df is not None else None
        if row is None:
            missing.append(t)
            continue
        if not args.no_earnings and t not in ETFS:
            row["earnings"] = next_earnings(t, cache, today)
        rows.append(row)

    if not args.no_earnings:
        try:
            cache_path.write_text(json.dumps(cache, indent=1))
        except Exception:
            pass

    out = pd.DataFrame(rows, columns=COLUMNS)
    out_path = Path(args.out).expanduser()
    out.to_csv(out_path, index=False)
    # scan.js next to the CSV: the tracker page loads it on launch as window.MRT_SCAN (no import step needed).
    js_path = out_path.with_name("scan.js")
    try:
        js_path.write_text("window.MRT_SCAN = " + json.dumps(rows) + ";\n")
    except Exception:
        js_path = None

    # Summary
    setups = out[(out["bb"] != "none") & (out["squeeze"] == "false")]
    squeezes = out[out["squeeze"] == "true"]
    print(f"\nWrote {len(out)} rows to {out_path}" + (f" and {js_path.name}" if js_path else "") + (f"  (no data for: {', '.join(missing)})" if missing else ""))
    if len(out):
        print(f"Data as of {out['asof'].max()}")
    if len(setups):
        print(f"\nAt a band ({len(setups)}):")
        for _, r in setups.sort_values("rsi").iterrows():
            side = "oversold  -> bull call" if r["bb"] == "lower" else "overbought -> bear put"
            earn = f"  earnings {r['earnings']}" if r["earnings"] else ""
            print(f"  {r['ticker']:<6} {side}   RSI {r['rsi']:>5}  ADX {r['adx']:>5}  ${r['price']}{earn}")
    else:
        print("\nNothing at a band today.")
    print(f"In a squeeze (skip): {len(squeezes)}")
    print("\nOpen the tracker, Scan tab, Import scan, and paste or upload the CSV.")

    if args.print:
        print("\n" + out.to_csv(index=False))


if __name__ == "__main__":
    main()
