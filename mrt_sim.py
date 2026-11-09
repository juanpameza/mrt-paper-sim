#!/usr/bin/env python3
"""
mrt_sim.py  -  rules-only paper trading simulator for the Mean Reversion Paper Tracker

Run it once per trading day after the close (same folder as mrt_scan.py and the
tracker HTML). Each run:

  1. Downloads daily bars, computes Bollinger / Keltner / RSI / ADX / ATR.
  2. Marks every open spread from the live option chain (Black-Scholes fallback
     if the chain can't be read) and exits on the rules: 50% stop, 100% target,
     expiration, or an earnings date that moved inside the trade window.
  3. Walks the scan for new setups and applies the entry checklist in order.
     The first rule a candidate fails is logged as the reason it was skipped.
     Candidates that pass get a $5-wide debit spread (or $10 above the price
     threshold) built from the chain, sized under the deployment cap.
  4. Saves sim_state.json, writes sim.js for the tracker page and sim_trades.csv.

Setup (once):   pip install yfinance pandas numpy
Run:            python mrt_sim.py                 one evaluation on the latest daily bar (after the close)
                python mrt_sim.py --intraday      one evaluation now, on today's bar as it forms (for a scheduler)
                python mrt_sim.py --loop          keep running: every 15 min during market hours, idle otherwise
Options:        --every N   minutes between loop passes (default 15; Yahoo option quotes are 15-min delayed anyway)
                --force     evaluate even if no new daily bar since the last run
                --dry-run   show what would happen, change nothing
                --reset     start a fresh sim (asks for confirmation)

Intraday passes use today's partial daily bar (Yahoo includes it while the market is open), so a band
touch at 10:40 can be entered at 10:45 and a stop can fire mid-session instead of on the close mark.
Entries happen at most once per ticker per day; exits are checked on every pass.

Every rule lives in the RULES block below. Change a number there, not in the code.
"""

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
import uuid
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # Python < 3.9
    NY = None

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "sim_state.json"
JS_PATH = HERE / "sim.js"
JSON_PATH = HERE / "sim.json"
CSV_PATH = HERE / "sim_trades.csv"
EARNINGS_CACHE = HERE / ".earnings_cache.json"

# ----------------------------------------------------------------------------
# Rotation
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
# Indicator settings (keep identical to mrt_scan.py so the page and the sim agree)
# ----------------------------------------------------------------------------
BB_LEN, BB_MULT = 20, 2.0
KC_LEN, KC_MULT, KC_ATR_LEN = 20, 1.5, 20
RSI_LEN, ADX_LEN, ATR_LEN = 14, 14, 14
BAND_TOL_ATR = 0.0
HISTORY = "9mo"
EARNINGS_CACHE_DAYS = 7

# ----------------------------------------------------------------------------
# THE RULES. Each one is checked, in this order, for every candidate, every day.
# ----------------------------------------------------------------------------
RULES = {
    # Account
    "START_CAPITAL": 2500.0,
    "GOAL": 25000.0,
    "DEPLOY_CAP_PCT": 30.0,        # total cost of open spreads <= this % of equity
    "MAX_POSITION_PCT": 10.0,      # one spread's cost <= this % of equity
    "MAX_OPEN": 5,                 # never more open positions than this
    "ONE_PER_TICKER": True,        # no adding to a position, no averaging down
    "NO_REENTRY_SAME_DAY": True,   # intraday: once a name has traded today, leave it alone until tomorrow

    # Setup (from the scan)
    "RSI_LONG_MAX": 30.0,          # bull call only if RSI <= this
    "RSI_SHORT_MIN": 70.0,         # bear put only if RSI >= this
    "ADX_MAX": 40.0,               # don't fade a runaway trend: skip if ADX is above this. None = off
    "REQUIRE_ADX_COOLING": False,  # stricter: ADX today must be <= yesterday. On a band-break day ADX is
                                   # nearly always rising, so this blocks almost every setup. Off by default.

    # Earnings
    "EARNINGS_AFTER_EXPIRY": True, # the next report must land after the spread expires
    "SKIP_IF_EARNINGS_UNKNOWN": True,

    # Structure
    "WIDTH": 5.0,
    "WIDE_PRICE": 400.0,           # above this price use WIDTH_WIDE
    "WIDTH_WIDE": 10.0,
    "DEBIT_MIN_FRAC": 0.35,        # debit paid must be between these fractions of the width...
    "DEBIT_MAX_FRAC": 0.50,        # ...so a full winner doubles
    "STRIKE_SEARCH": 2,            # try the ATM pair and up to this many strikes either side

    # Time
    "MIN_DTE": 20,                 # window wide enough that a monthly expiration is always inside it
    "MAX_DTE": 55,
    "TARGET_DTE": 35,
    "PREFER_MONTHLY": True,        # third-Friday expirations carry the open interest; weeklies only if no monthly fits

    # Liquidity
    "MIN_OI": 1000,                # open interest on each leg
    "MAX_LEG_SPREAD_PCT": 0.10,    # bid/ask width per leg <= this fraction of the mid...
    "MAX_LEG_SPREAD_ABS": 0.15,    # ...or this many dollars, whichever is larger
    "REQUIRE_LIVE_QUOTES": False,  # after hours Yahoo often shows 0 bid/ask; False = fall back to last trade

    # Exits (set at entry, never changed)
    "STOP_PCT": 50.0,
    "TARGET_PCT": 100.0,
    "EXIT_ON_EARNINGS_INSIDE_WINDOW": True,

    # Friction
    "SLIPPAGE_PER_SPREAD": 0.05,   # paid on top of mid at entry, given up at exit
    "COMMISSION_PER_CONTRACT_LEG": 0.65,
    "RISK_FREE_RATE": 0.04,        # only for the Black-Scholes fallback

    # Shadow books: the same signals and rules, traded on paper under different entry timing, so the
    # timing question is answered with data. "first" is the live book (first pass that passes every
    # rule). "close" buys at the first pass after 4pm New York on the finished daily bar. "confirm"
    # waits for a close back inside the band after a touch, then buys at that day's close.
    "SHADOW_BOOKS": ["close", "confirm"],
    "CONFIRM_LOOKBACK": 5,         # a valid touch within this many sessions counts for the confirm book

    # Signal book: every setup that passes the rules, one contract each, no capacity limits (no deployment
    # cap, no position size, no max open). The live book is what the account can afford; this is the
    # strategy's whole sample, so win rate and expectancy are measured on all of it.
    "SIGNALS_BOOK": True,
}


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def wilder(series, n):
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(df):
    prev_close = df["Close"].shift(1)
    return pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)


def atr(df, n):
    return wilder(true_range(df), n)


def rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder(gain, n)
    avg_loss = wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss != 0, 100.0)


def adx(df, n):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = wilder(true_range(df), n)
    plus_di = 100.0 * wilder(plus_dm, n) / tr_s
    minus_di = 100.0 * wilder(minus_dm, n) / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx.fillna(0.0), n)


def bollinger(close, n, k):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def keltner(df, n, k, atr_len):
    mid = df["Close"].ewm(span=n, adjust=False).mean()
    a = atr(df, atr_len)
    return mid + k * a, mid, mid - k * a


def scan_row(ticker, df):
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < max(BB_LEN, KC_LEN, ADX_LEN * 3) + 5:
        return None
    close = df["Close"]
    bb_u, bb_m, bb_l = bollinger(close, BB_LEN, BB_MULT)
    kc_u, _, kc_l = keltner(df, KC_LEN, KC_MULT, KC_ATR_LEN)
    a = atr(df, ATR_LEN)
    r = rsi(close, RSI_LEN)
    x = adx(df, ADX_LEN)
    last = df.iloc[-1]
    tol = BAND_TOL_ATR * float(a.iloc[-1])
    t_low = float(last["Low"]) <= float(bb_l.iloc[-1]) + tol
    t_high = float(last["High"]) >= float(bb_u.iloc[-1]) - tol
    if t_low and t_high:
        bb = "lower" if float(last["Close"]) < float(bb_m.iloc[-1]) else "upper"
    elif t_low:
        bb = "lower"
    elif t_high:
        bb = "upper"
    else:
        bb = "none"
    squeeze = bool(float(bb_u.iloc[-1]) < float(kc_u.iloc[-1]) and float(bb_l.iloc[-1]) > float(kc_l.iloc[-1]))

    # Confirmation view for the "confirm" shadow book: a valid touch in the last few sessions (band
    # touched with RSI past its threshold), and today a close back inside the band, up (or down) on the
    # day, with RSI turning. The live book never uses these.
    look = int(RULES["CONFIRM_LOOKBACK"])
    confirm = None
    close_now, close_prev = float(close.iloc[-1]), float(close.iloc[-2])
    rsi_now, rsi_prev = float(r.iloc[-1]), float(r.iloc[-2])
    inside = float(bb_l.iloc[-1]) < close_now < float(bb_u.iloc[-1])
    touch_long = touch_short = None
    for k in range(2, look + 2):
        if len(df) < k:
            break
        lo_k, hi_k, rsi_k = float(df["Low"].iloc[-k]), float(df["High"].iloc[-k]), float(r.iloc[-k])
        if touch_long is None and lo_k <= float(bb_l.iloc[-k]) and rsi_k <= RULES["RSI_LONG_MAX"]:
            touch_long = k - 1
        if touch_short is None and hi_k >= float(bb_u.iloc[-k]) and rsi_k >= RULES["RSI_SHORT_MIN"]:
            touch_short = k - 1
    if inside and touch_long is not None and close_now > close_prev and rsi_now >= rsi_prev:
        confirm = "bull-call"
    elif inside and touch_short is not None and close_now < close_prev and rsi_now <= rsi_prev:
        confirm = "bear-put"

    return {
        "ticker": ticker,
        "price": round(float(last["Close"]), 2),
        "high": round(float(last["High"]), 2),
        "low": round(float(last["Low"]), 2),
        "atr": round(float(a.iloc[-1]), 2),
        "bb": bb,
        "squeeze": squeeze,
        "rsi": round(float(r.iloc[-1]), 1),
        "adx": round(float(x.iloc[-1]), 1),
        "adx_prev": round(float(x.iloc[-2]), 1),
        "confirm": confirm,
        "touch_days_ago": touch_long if confirm == "bull-call" else (touch_short if confirm == "bear-put" else None),
        "bb_upper": round(float(bb_u.iloc[-1]), 2),
        "bb_lower": round(float(bb_l.iloc[-1]), 2),
        "kc_upper": round(float(kc_u.iloc[-1]), 2),
        "kc_lower": round(float(kc_l.iloc[-1]), 2),
        "asof": pd.Timestamp(df.index[-1]).date().isoformat(),
    }


def frame_for(data, ticker):
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
    if not {"Open", "High", "Low", "Close"}.issubset(set(df.columns)):
        return None
    return df


# ----------------------------------------------------------------------------
# Black-Scholes (fallback marks only)
# ----------------------------------------------------------------------------
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, call=True, r=0.04):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K) if call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


# ----------------------------------------------------------------------------
# Market data (Yahoo). Tests swap in a mock with the same four methods.
# ----------------------------------------------------------------------------
class YahooData:
    def __init__(self):
        import yfinance as yf
        self.yf = yf
        self._tk = {}
        self._chains = {}

    def reset(self):
        """Drop cached chains so a new pass marks against fresh quotes."""
        self._chains = {}

    def ticker(self, t):
        if t not in self._tk:
            self._tk[t] = self.yf.Ticker(t)
        return self._tk[t]

    def prices(self, tickers, period=HISTORY):
        data = self.yf.download(list(tickers), period=period, interval="1d", group_by="ticker",
                                auto_adjust=False, threads=True, progress=False)
        if data is None or len(data) == 0:
            raise RuntimeError("No price data came back from Yahoo.")
        return data

    def expiries(self, t):
        try:
            return list(self.ticker(t).options)
        except Exception:
            return []

    def chain(self, t, expiry):
        """Returns (calls, puts) DataFrames with strike, bid, ask, lastPrice, openInterest, impliedVolatility."""
        key = (t, expiry)
        if key not in self._chains:
            oc = self.ticker(t).option_chain(expiry)
            self._chains[key] = (oc.calls.copy(), oc.puts.copy())
        return self._chains[key]

    def earnings(self, t, today, cache):
        c = cache.get(t)
        if c:
            try:
                checked = dt.date.fromisoformat(c["checked"])
                d = c.get("date")
                still_future = (d is None) or (dt.date.fromisoformat(d) >= today)
                if (today - checked).days < EARNINGS_CACHE_DAYS and still_future:
                    return d
            except Exception:
                pass
        date = None
        try:
            tk = self.ticker(t)
            candidates = []
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
                    pass
            fut = sorted(d for d in dates if d >= today)
            if fut:
                date = fut[0].isoformat()
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
        cache[t] = {"date": date, "checked": today.isoformat()}
        return date


# ----------------------------------------------------------------------------
# Option helpers
# ----------------------------------------------------------------------------
def leg_quote(df, strike):
    rows = df[np.isclose(df["strike"].astype(float), float(strike))]
    if rows.empty:
        return None
    r = rows.iloc[0]
    bid = float(r.get("bid", 0) or 0)
    ask = float(r.get("ask", 0) or 0)
    last = float(r.get("lastPrice", 0) or 0)
    oi = int(r.get("openInterest", 0) or 0)
    iv = float(r.get("impliedVolatility", 0) or 0)
    live = bid > 0 and ask > 0 and ask >= bid
    if live:
        mid = (bid + ask) / 2.0
        spread = ask - bid
    elif last > 0:
        mid = last
        spread = None
    else:
        return None
    return {"strike": float(strike), "bid": bid, "ask": ask, "last": last, "mid": round(mid, 2),
            "spread": None if spread is None else round(spread, 2), "oi": oi, "iv": round(iv, 4), "live": live}


def is_monthly(d):
    """Standard monthly expiration: the third Friday (or the Thursday before it when that Friday is a holiday)."""
    if d.weekday() == 4 and 15 <= d.day <= 21:
        return True
    if d.weekday() == 3 and 14 <= d.day <= 20:
        return True
    return False


def choose_expiry(expiries, today):
    inside = []
    for e in expiries:
        try:
            d = dt.date.fromisoformat(str(e)[:10])
        except Exception:
            continue
        dte = (d - today).days
        if RULES["MIN_DTE"] <= dte <= RULES["MAX_DTE"]:
            inside.append((abs(dte - RULES["TARGET_DTE"]), d))
    if not inside:
        return None
    if RULES.get("PREFER_MONTHLY"):
        monthly = [x for x in inside if is_monthly(x[1])]
        if monthly:
            inside = monthly
    inside.sort()
    return inside[0][1].isoformat()


def leg_ok(q):
    """Liquidity rule for one leg. Returns (ok, reason)."""
    if q is None:
        return False, "no quote"
    if q["oi"] < RULES["MIN_OI"]:
        return False, f"open interest {q['oi']} < {RULES['MIN_OI']}"
    if q["live"]:
        allowed = max(RULES["MAX_LEG_SPREAD_ABS"], RULES["MAX_LEG_SPREAD_PCT"] * q["mid"])
        if q["spread"] > allowed + 1e-9:
            return False, f"bid/ask {q['spread']:.2f} wider than {allowed:.2f}"
    elif RULES["REQUIRE_LIVE_QUOTES"]:
        return False, "no live bid/ask"
    return True, ""


def build_spread(calls, puts, price, side, width):
    """Try the ATM pair and neighbours; return (best_candidate, reasons_for_rejects)."""
    df = calls if side == "bull-call" else puts
    strikes = sorted(set(float(s) for s in df["strike"].astype(float).tolist()))
    if not strikes:
        return None, ["empty chain"]
    if side == "bull-call":
        below = [s for s in strikes if s <= price]
        anchor = max(below) if below else min(strikes)
    else:
        above = [s for s in strikes if s >= price]
        anchor = min(above) if above else max(strikes)
    idx = strikes.index(anchor)
    passing, reasons = [], []
    # At-the-money first, then one step either side, and so on: the first reason reported is the ATM pair's.
    shifts = sorted(range(-RULES["STRIKE_SEARCH"], RULES["STRIKE_SEARCH"] + 1), key=lambda s: (abs(s), s))
    for shift in shifts:
        i = idx + shift
        if i < 0 or i >= len(strikes):
            continue
        lk = strikes[i]
        sk = lk + width if side == "bull-call" else lk - width
        if not any(np.isclose(sk, s) for s in strikes):
            reasons.append(f"{lk:g}/{sk:g}: no {width:g}-wide pair")
            continue
        lq, sq = leg_quote(df, lk), leg_quote(df, sk)
        for name, q in (("long", lq), ("short", sq)):
            ok, why = leg_ok(q)
            if not ok:
                reasons.append(f"{lk:g}/{sk:g}: {name} leg {why}")
                break
        else:
            mid_debit = lq["mid"] - sq["mid"]
            fill = round(mid_debit + RULES["SLIPPAGE_PER_SPREAD"], 2)
            frac = fill / width
            if mid_debit <= 0:
                reasons.append(f"{lk:g}/{sk:g}: debit not positive")
            elif frac < RULES["DEBIT_MIN_FRAC"]:
                reasons.append(f"{lk:g}/{sk:g}: debit {fill:.2f} under {RULES['DEBIT_MIN_FRAC']:.0%} of width")
            elif frac > RULES["DEBIT_MAX_FRAC"]:
                reasons.append(f"{lk:g}/{sk:g}: debit {fill:.2f} over {RULES['DEBIT_MAX_FRAC']:.0%} of width")
            else:
                passing.append({"longK": lk, "shortK": sk, "width": width, "midDebit": round(mid_debit, 2),
                                "debit": fill, "frac": frac, "long": lq, "short": sq})
    if not passing:
        return None, reasons
    # Closest to half the width from below = most probability bought while a full winner still doubles.
    passing.sort(key=lambda c: (RULES["DEBIT_MAX_FRAC"] - c["frac"], abs(c["longK"] - anchor)))
    return passing[0], reasons


def spread_value_from_chain(calls, puts, pos):
    df = calls if pos["type"] == "bull-call" else puts
    lq, sq = leg_quote(df, pos["longK"]), leg_quote(df, pos["shortK"])
    if lq is None or sq is None:
        return None, None
    v = lq["mid"] - sq["mid"]
    return max(0.0, min(pos["width"], v)), (lq, sq)


def spread_value_model(pos, S, today):
    expiry = dt.date.fromisoformat(pos["expiry"])
    T = max(0.0, (expiry - today).days / 365.0)
    call = pos["type"] == "bull-call"
    iv_l = pos.get("entry", {}).get("long", {}).get("iv") or 0.5
    iv_s = pos.get("entry", {}).get("short", {}).get("iv") or iv_l
    v = bs_price(S, pos["longK"], T, iv_l, call, RULES["RISK_FREE_RATE"]) - bs_price(S, pos["shortK"], T, iv_s, call, RULES["RISK_FREE_RATE"])
    return max(0.0, min(pos["width"], v))


def intrinsic(pos, S):
    if pos["type"] == "bull-call":
        return max(0.0, min(pos["width"], S - pos["longK"]))
    return max(0.0, min(pos["width"], pos["longK"] - S))


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
def new_state():
    return {
        "version": 1,
        "createdAt": dt.datetime.now().isoformat(timespec="seconds"),
        "rules": dict(RULES),
        "equity": RULES["START_CAPITAL"],
        "trades": {},
        "log": [],
        "lastBar": None,
        "lastRun": None,
    }


def load_state():
    if STATE_PATH.exists():
        try:
            s = json.loads(STATE_PATH.read_text())
            if isinstance(s, dict) and "trades" in s:
                s["rules"] = dict(RULES)   # rules in the file are informational; the block above wins
                return s
        except Exception as e:
            raise SystemExit(f"sim_state.json is unreadable ({e}). Move it aside or run --reset.")
    return new_state()


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=1))


def log(state, today, ticker, action, reason="", detail=None, when=None):
    entry = {"date": today.isoformat(), "ticker": ticker, "action": action, "reason": reason}
    if when:
        entry["time"] = when
    if detail:
        entry["detail"] = detail
    state["log"].append(entry)
    if len(state["log"]) > 900:
        # Entries and exits are few and worth keeping; skips are many and only the recent ones matter.
        keep = [e for e in state["log"] if e["action"] != "skip"]
        skips = [e for e in state["log"] if e["action"] == "skip"][-400:]
        state["log"] = sorted(keep + skips, key=lambda e: (e["date"], e.get("time") or ""))


def log_skip(state, today, ticker, reason, when=None):
    """Intraday passes repeat every few minutes; only record a skip when the reason changes for that name today."""
    last = state.setdefault("lastSkip", {}).get(ticker)
    if last and last.get("date") == today.isoformat() and last.get("reason") == reason:
        return
    state["lastSkip"][ticker] = {"date": today.isoformat(), "reason": reason}
    log(state, today, ticker, "skip", reason, when=when)


_clock = None   # tests can set this to a function returning a New York datetime


def ny_now():
    if _clock:
        return _clock()
    return dt.datetime.now(NY) if NY else dt.datetime.now()


def is_close_pass(intraday):
    """The pass that sees the finished daily bar: any daily-mode run, or an intraday pass after 4pm New York."""
    if not intraday:
        return True
    now = ny_now()
    return now.time() >= dt.time(16, 0)


def new_book():
    return {"equity": RULES["START_CAPITAL"], "trades": {}, "log": [], "lastSkip": {}}


def shadow_book(state, name):
    books = state.setdefault("shadow", {})
    if name not in books:
        books[name] = new_book()
    return books[name]


def market_open_now(now=None):
    """Regular session, weekdays 9:30-16:00 New York. Holidays show up as 'no bar for today' and are skipped by the data check."""
    now = now or ny_now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt.time(9, 30) <= t <= dt.time(16, 0)


def traded_today(state, ticker, today):
    d = today.isoformat()
    return any(t["ticker"] == ticker and (t["opened"] == d or t.get("closed") == d) for t in state["trades"].values())


def open_positions(state):
    return [t for t in state["trades"].values() if t["status"] == "open"]


def closed_positions(state):
    return sorted([t for t in state["trades"].values() if t["status"] == "closed"], key=lambda t: (t["closed"], t["createdAt"]))


def deployed(state):
    return round(sum(t["cost"] for t in open_positions(state)), 2)


def fees_for(n, legs=2):
    return round(RULES["COMMISSION_PER_CONTRACT_LEG"] * legs * n, 2)


# ----------------------------------------------------------------------------
# One trading day
# ----------------------------------------------------------------------------
def manage_exits(md, state, rows, today, cache, dry, when=None):
    events = []
    for pos in open_positions(state):
        t = pos["ticker"]
        row = rows.get(t)
        if row is None:
            events.append(f"{t}: no price data today, position held")
            continue
        S = row["price"]
        expiry = dt.date.fromisoformat(pos["expiry"])
        dte = (expiry - today).days
        value, source = None, None
        if dte <= 0:
            value, source = intrinsic(pos, S), "expiry"
        else:
            try:
                calls, puts = md.chain(t, pos["expiry"])
                value, legs = spread_value_from_chain(calls, puts, pos)
                if value is not None:
                    source = "chain"
            except Exception:
                value = None
            if value is None:
                value, source = spread_value_model(pos, S, today), "model"
        value = round(value, 2)
        pos["lastMark"] = value
        pos["lastMarkDate"] = today.isoformat()
        pos["lastMarkTime"] = when
        pos["markSource"] = source
        marks = pos.setdefault("marks", [])
        if marks and marks[-1]["date"] == today.isoformat():
            marks[-1].update({"mark": value, "source": source, "price": S, "time": when})   # one row per day, latest value
        else:
            marks.append({"date": today.isoformat(), "mark": value, "source": source, "price": S, "time": when})
        pos["marks"] = marks[-90:]

        reason = None
        if source == "expiry":
            reason = "expired"
        elif RULES["EXIT_ON_EARNINGS_INSIDE_WINDOW"] and t not in ETFS:
            e = md.earnings(t, today, cache)
            if e and e <= pos["expiry"]:
                reason = f"earnings {e} moved inside the window"
        if reason is None:
            if value <= pos["stop"]:
                reason = "stop"
            elif value >= pos["target"]:
                reason = "target"
        if reason is None:
            events.append(f"{t}: hold  mark {value:.2f} ({source})  stop {pos['stop']:.2f}  target {pos['target']:.2f}  {dte} DTE")
            continue

        if source == "expiry":
            exit_fill = value
            fees = 0.0
        else:
            exit_fill = round(max(0.0, min(pos["width"], value - RULES["SLIPPAGE_PER_SPREAD"])), 2)
            fees = fees_for(pos["contracts"])
        gross = round((exit_fill - pos["debit"]) * 100 * pos["contracts"], 2)
        net = round(gross - fees - pos["entryFees"], 2)
        if not dry:
            pos.update({"status": "closed", "closed": today.isoformat(), "exitPrice": exit_fill, "exitMark": value,
                        "exitReason": reason, "exitFees": fees, "fees": round(fees + pos["entryFees"], 2),
                        "pnlGross": gross, "pnl": net, "closedAt": dt.datetime.now().isoformat(timespec="seconds"),
                        "closedTime": when})
            state["equity"] = round(state["equity"] + net, 2)
            log(state, today, t, "exit", reason, {"exit": exit_fill, "pnl": net}, when=when)
        events.append(f"{t}: EXIT ({reason})  {pos['debit']:.2f} -> {exit_fill:.2f}  P&L {net:+.2f}")
    return events


def candidate_order(rows):
    """Most stretched first: RSI distance from 50, then band touch."""
    cands = [r for r in rows.values() if r["bb"] in ("lower", "upper")]
    cands.sort(key=lambda r: -abs(r["rsi"] - 50))
    return cands


def try_entry(md, state, row, today, cache, dry, side=None, book_name="live", unconstrained=False):
    """Apply the checklist in order. Returns (entered, reason).
    `side` preset (confirm book): the touch-day checks were done in scan_row; today's squeeze/ADX still apply.
    `unconstrained` (signal book): skip the capacity rules and size one contract."""
    t = row["ticker"]
    # 1. Is it a setup at all?
    if row["squeeze"]:
        return False, "squeeze: bands inside Keltner"
    if side is None:
        side = "bull-call" if row["bb"] == "lower" else "bear-put"
        if side == "bull-call" and row["rsi"] > RULES["RSI_LONG_MAX"]:
            return False, f"RSI {row['rsi']} above {RULES['RSI_LONG_MAX']:g}"
        if side == "bear-put" and row["rsi"] < RULES["RSI_SHORT_MIN"]:
            return False, f"RSI {row['rsi']} below {RULES['RSI_SHORT_MIN']:g}"
    if RULES["ADX_MAX"] is not None and row["adx"] > RULES["ADX_MAX"]:
        return False, f"ADX {row['adx']} above {RULES['ADX_MAX']:g}: trend too strong to fade"
    if RULES["REQUIRE_ADX_COOLING"] and row["adx"] > row["adx_prev"]:
        return False, f"ADX climbing ({row['adx_prev']} -> {row['adx']})"
    # 2. Is there room for it?
    opens = open_positions(state)
    if RULES["ONE_PER_TICKER"] and any(p["ticker"] == t for p in opens):
        return False, "valid setup, already have a position"
    if RULES["NO_REENTRY_SAME_DAY"] and traded_today(state, t, today):
        return False, "valid setup, already traded today"
    if not unconstrained and len(opens) >= RULES["MAX_OPEN"]:
        return False, f"valid setup, max open positions ({RULES['MAX_OPEN']}) reached"
    # 3. Time, earnings, structure, size
    expiries = md.expiries(t)
    expiry = choose_expiry(expiries, today)
    if not expiry:
        return False, f"no expiration {RULES['MIN_DTE']}-{RULES['MAX_DTE']} days out"

    if t not in ETFS:
        e = md.earnings(t, today, cache)
        if e is None:
            if RULES["SKIP_IF_EARNINGS_UNKNOWN"]:
                return False, "earnings date unknown"
        elif RULES["EARNINGS_AFTER_EXPIRY"] and e <= expiry:
            return False, f"earnings {e} before expiry {expiry}"

    width = RULES["WIDTH_WIDE"] if row["price"] > RULES["WIDE_PRICE"] else RULES["WIDTH"]
    try:
        calls, puts = md.chain(t, expiry)
    except Exception as ex:
        return False, f"could not read option chain ({ex.__class__.__name__})"
    spread, rejects = build_spread(calls, puts, row["price"], side, width)
    if spread is None:
        return False, f"no strike pair passed ({expiry}): " + ("; ".join(rejects[:3]) if rejects else "none available")

    equity = state["equity"]
    cost_per = round(spread["debit"] * 100, 2)
    if unconstrained:
        n = 1
    else:
        cap_room = round(equity * RULES["DEPLOY_CAP_PCT"] / 100.0 - deployed(state), 2)
        pos_room = round(equity * RULES["MAX_POSITION_PCT"] / 100.0, 2)
        budget = min(cap_room, pos_room)
        n = int(budget // cost_per) if cost_per > 0 else 0
        if n < 1:
            if cap_room < cost_per:
                return False, f"deployment cap: {cap_room:.0f} left, spread costs {cost_per:.0f}"
            return False, f"position limit {pos_room:.0f} < spread cost {cost_per:.0f}"

    stop = round(spread["debit"] * (1 - RULES["STOP_PCT"] / 100.0), 2)
    target = round(min(spread["debit"] * (1 + RULES["TARGET_PCT"] / 100.0), width), 2)
    entry_fees = fees_for(n)
    trade = {
        "id": "s" + uuid.uuid4().hex[:10],
        "ticker": t, "type": side, "opened": today.isoformat(), "expiry": expiry,
        "longK": spread["longK"], "shortK": spread["shortK"], "width": width,
        "debit": spread["debit"], "midDebit": spread["midDebit"], "contracts": n,
        "cost": round(cost_per * n, 2), "stop": stop, "target": target,
        "stopPct": RULES["STOP_PCT"], "targetPct": RULES["TARGET_PCT"],
        "entryFees": entry_fees, "status": "open", "book": book_name,
        "signal": {"price": row["price"], "rsi": row["rsi"], "adx": row["adx"], "adx_prev": row["adx_prev"], "bb": row["bb"], "atr": row["atr"],
                   "confirm": row.get("confirm"), "touchDaysAgo": row.get("touch_days_ago")},
        "entry": {"long": spread["long"], "short": spread["short"], "quotes": "live" if spread["long"]["live"] and spread["short"]["live"] else "last trade"},
        "createdAt": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if not dry:
        trade["openedAt"] = ny_now().strftime("%H:%M") if NY else None
        state["trades"][trade["id"]] = trade
        log(state, today, t, "entry", f"{side} {spread['longK']:g}/{spread['shortK']:g} exp {expiry} @ {spread['debit']:.2f} x{n}",
            {"stop": stop, "target": target, "cost": trade["cost"]}, when=trade["openedAt"])
    return True, f"{side} {spread['longK']:g}/{spread['shortK']:g} exp {expiry} @ {spread['debit']:.2f} x{n} (stop {stop:.2f}, target {target:.2f})"


def run_day(md, state, force=False, dry=False, today_override=None, intraday=False):
    if hasattr(md, "reset"):
        md.reset()
    tickers = sorted(set(UNIVERSE) | {p["ticker"] for p in open_positions(state)})
    data = md.prices(tickers)
    rows = {}
    for t in tickers:
        df = frame_for(data, t)
        r = scan_row(t, df) if df is not None else None
        if r:
            rows[t] = r
    if not rows:
        raise RuntimeError("No usable price history.")
    asof = max(r["asof"] for r in rows.values())
    today = today_override or dt.date.fromisoformat(asof)
    when = ny_now().strftime("%H:%M") if (intraday and NY) else None
    summary = {"asof": asof, "today": today.isoformat(), "exits": [], "entries": [], "skipped": [], "ran": True, "intraday": intraday}

    if intraday:
        # Only act on a bar that is forming today; a stale bar means the market is closed (holiday, weekend, early run).
        ny_today = (today_override or ny_now().date())
        if asof != ny_today.isoformat() and not force:
            summary["ran"] = False
            summary["why"] = f"latest bar is {asof}, market not open today"
            return summary, rows
    elif state.get("lastBar") == asof and not force:
        summary["ran"] = False
        summary["why"] = "no new daily bar since the last run"
        return summary, rows

    cache = {}
    if EARNINGS_CACHE.exists():
        try:
            cache = json.loads(EARNINGS_CACHE.read_text())
        except Exception:
            cache = {}

    summary["exits"] = manage_exits(md, state, rows, today, cache, dry, when=when)

    for row in candidate_order(rows):
        entered, reason = try_entry(md, state, row, today, cache, dry)
        if entered:
            summary["entries"].append(f"{row['ticker']}: {reason}")
        else:
            summary["skipped"].append(f"{row['ticker']} ({row['bb']}, RSI {row['rsi']}): {reason}")
            if not dry:
                if intraday:
                    log_skip(state, today, row["ticker"], reason, when=when)
                else:
                    log(state, today, row["ticker"], "skip", reason)

    # Shadow books: exits every pass (same marks), entries only on the close pass.
    summary["shadow"] = {}
    close_pass = is_close_pass(intraday)
    for name in RULES.get("SHADOW_BOOKS", []):
        book = shadow_book(state, name)
        ev = manage_exits(md, book, rows, today, cache, dry, when=when)
        entries = []
        if close_pass:
            if name == "close":
                cands = [(r, None) for r in candidate_order(rows)]
            elif name == "confirm":
                cands = [(r, r["confirm"]) for r in rows.values() if r.get("confirm")]
                cands.sort(key=lambda rc: -abs(rc[0]["rsi"] - 50))
            else:
                cands = []
            for row, side in cands:
                entered, reason = try_entry(md, book, row, today, cache, dry, side=side, book_name=name)
                if entered:
                    entries.append(f"{row['ticker']}: {reason}")
                elif not dry:
                    log_skip(book, today, row["ticker"], reason, when=when)
        summary["shadow"][name] = {"exits": [e for e in ev if "EXIT" in e], "entries": entries}

    # Signal book: everything that qualifies, one contract each, every pass (same timing as the live book).
    if RULES.get("SIGNALS_BOOK"):
        book = state.setdefault("signals", new_book())
        ev = manage_exits(md, book, rows, today, cache, dry, when=when)
        entries = []
        for row in candidate_order(rows):
            entered, reason = try_entry(md, book, row, today, cache, dry, book_name="signals", unconstrained=True)
            if entered:
                entries.append(f"{row['ticker']}: {reason}")
            elif not dry:
                log_skip(book, today, row["ticker"], reason, when=when)
        summary["signals"] = {"exits": [e for e in ev if "EXIT" in e], "entries": entries}

    if not dry:
        state["lastBar"] = asof
        state["lastRun"] = dt.datetime.now().isoformat(timespec="seconds")
        state["lastPoll"] = (ny_now().isoformat(timespec="minutes") if NY else state["lastRun"])
        state["mode"] = "intraday" if intraday else "daily"
        try:
            EARNINGS_CACHE.write_text(json.dumps(cache, indent=1))
        except Exception:
            pass
    return summary, rows


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def export_log(entries, recent_skips):
    """Every entry and exit, plus only the most recent skips, in time order."""
    keep = [e for e in entries if e["action"] != "skip"]
    skips = [e for e in entries if e["action"] == "skip"][-recent_skips:]
    return sorted(keep + skips, key=lambda e: (e["date"], e.get("time") or ""))


def export(state):
    opens = open_positions(state)
    payload = {
        "generatedAt": dt.datetime.now().isoformat(timespec="seconds"),
        "asof": state.get("lastBar"),
        "lastPoll": state.get("lastPoll"),
        "mode": state.get("mode", "daily"),
        "settings": {
            "startCapital": RULES["START_CAPITAL"], "goal": RULES["GOAL"], "deployCap": RULES["DEPLOY_CAP_PCT"],
            "stopPct": RULES["STOP_PCT"], "targetPct": RULES["TARGET_PCT"],
        },
        "rules": {k: v for k, v in RULES.items()},
        "equity": state["equity"],
        "deployed": deployed(state),
        "trades": state["trades"],
        "log": export_log(state["log"], 250),
        "shadow": {
            name: {"equity": b["equity"], "deployed": deployed(b), "trades": b["trades"], "log": export_log(b["log"], 80)}
            for name, b in state.get("shadow", {}).items()
        },
    }
    if state.get("signals"):
        b = state["signals"]
        payload["signals"] = {"equity": b["equity"], "trades": b["trades"], "log": export_log(b["log"], 120)}
    JSON_PATH.write_text(json.dumps(payload, indent=1))
    JS_PATH.write_text("window.MRT_SIM = " + json.dumps(payload) + ";\n")
    cols = ["id", "ticker", "type", "opened", "expiry", "longK", "shortK", "width", "debit", "contracts", "cost",
            "stop", "target", "status", "lastMark", "exitPrice", "exitReason", "closed", "fees", "pnlGross", "pnl"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in sorted(state["trades"].values(), key=lambda x: (x["opened"], x["createdAt"])):
            w.writerow(t)
    shadow_rows = []
    for name, b in state.get("shadow", {}).items():
        for t in b["trades"].values():
            shadow_rows.append(dict(t, book=name))
    if shadow_rows:
        with (HERE / "shadow_trades.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["book"] + cols, extrasaction="ignore")
            w.writeheader()
            for t in sorted(shadow_rows, key=lambda x: (x["opened"], x["book"], x["createdAt"])):
                w.writerow(t)
    if state.get("signals") and state["signals"]["trades"]:
        with (HERE / "signals_trades.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for t in sorted(state["signals"]["trades"].values(), key=lambda x: (x["opened"], x["createdAt"])):
                w.writerow(t)


def print_summary(state, summary):
    print(f"\nData as of {summary['asof']}" + (f"  ({ny_now().strftime('%H:%M')} New York)" if summary.get("intraday") and NY else ""))
    if not summary["ran"]:
        print("Nothing evaluated: " + summary.get("why", "no change") + ". Use --force to override.")
    else:
        print("\nOpen positions:")
        if summary["exits"]:
            for e in summary["exits"]:
                print("  " + e)
        else:
            print("  none")
        print("\nNew entries:")
        for e in summary["entries"] or ["  none"]:
            print("  " + e if not e.startswith("  ") else e)
        print(f"\nCandidates skipped ({len(summary['skipped'])}):")
        for s in summary["skipped"][:25]:
            print("  " + s)
        if len(summary["skipped"]) > 25:
            print(f"  ... {len(summary['skipped']) - 25} more in sim_state.json")
    closed = closed_positions(state)
    wins = sum(1 for t in closed if t["pnl"] > 0)
    dep = deployed(state)
    print(f"\nEquity {state['equity']:,.2f}  (start {RULES['START_CAPITAL']:,.0f}, goal {RULES['GOAL']:,.0f})")
    print(f"Open {len(open_positions(state))}  deployed {dep:,.0f} ({(dep / state['equity'] * 100) if state['equity'] else 0:.1f}% of equity, cap {RULES['DEPLOY_CAP_PCT']:g}%)")
    if closed:
        print(f"Closed {len(closed)}  wins {wins}  win rate {wins / len(closed) * 100:.0f}%  realized {state['equity'] - RULES['START_CAPITAL']:+,.2f}")
    if state.get("shadow"):
        print("\nShadow books (entry timing):")
        for name, b in state["shadow"].items():
            c = closed_positions(b); w = sum(1 for t in c if t["pnl"] > 0)
            today_bits = summary.get("shadow", {}).get(name, {})
            extra = ""
            if today_bits.get("entries"):
                extra += "  entered today: " + "; ".join(today_bits["entries"])
            if today_bits.get("exits"):
                extra += "  exits today: " + "; ".join(today_bits["exits"])
            print(f"  {name:<8} equity {b['equity']:,.2f}  open {len(open_positions(b))}  closed {len(c)}" + (f"  wins {w}" if c else "") + extra)
    if state.get("signals"):
        b = state["signals"]; c = closed_positions(b); w = sum(1 for t in c if t["pnl"] > 0)
        bits = summary.get("signals", {})
        print(f"\nSignal book (all qualifying setups, 1 contract each): open {len(open_positions(b))}  closed {len(c)}"
              + (f"  win rate {w / len(c) * 100:.0f}%  net {b['equity'] - RULES['START_CAPITAL']:+,.2f}" if c else "")
              + (f"\n  entered today: " + "; ".join(bits["entries"]) if bits.get("entries") else ""))
    print(f"\nWrote {JS_PATH.name}, {JSON_PATH.name}, {CSV_PATH.name}")


def one_pass(md, state, args, intraday):
    today = dt.date.fromisoformat(args.date) if args.date else None
    if intraday and not args.force and NY:
        now = ny_now()
        settle_window = now.weekday() < 5 and dt.time(16, 0) < now.time() <= dt.time(16, 45)
        if not market_open_now(now) and not settle_window:
            return {"asof": state.get("lastBar"), "today": now.date().isoformat(), "exits": [], "entries": [], "skipped": [],
                    "ran": False, "intraday": True, "why": f"outside market hours ({now.strftime('%a %H:%M')} New York)"}
    summary, _ = run_day(md, state, force=args.force, dry=args.dry_run, today_override=today, intraday=intraday)
    if not args.dry_run and summary["ran"]:
        save_state(state)
        export(state)
    return summary


def run_loop(md, state, args):
    every = max(1, int(args.every))
    print(f"Loop mode: every {every} min while the market is open (New York hours). Ctrl-C to stop.")
    while True:
        try:
            if market_open_now():
                summary = one_pass(md, state, args, intraday=True)
                print_summary(state, summary)
                sleep_s = every * 60
            else:
                now = ny_now()
                print(f"{now.strftime('%a %H:%M')} NY: market closed, idle.")
                sleep_s = 300
        except KeyboardInterrupt:
            print("Stopped.")
            return
        except Exception as ex:
            print(f"Pass failed ({ex.__class__.__name__}: {ex}); will retry.")
            sleep_s = 120
        try:
            time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("Stopped.")
            return


def main():
    ap = argparse.ArgumentParser(description="Rules-only paper trading simulator")
    ap.add_argument("--intraday", action="store_true", help="one pass on today's forming bar (use from a scheduler during market hours)")
    ap.add_argument("--loop", action="store_true", help="keep running: a pass every --every minutes while the market is open")
    ap.add_argument("--every", default=15, help="minutes between loop passes (default 15)")
    ap.add_argument("--force", action="store_true", help="evaluate even if the daily bar has not changed / market is closed")
    ap.add_argument("--dry-run", action="store_true", help="print decisions without saving anything")
    ap.add_argument("--reset", action="store_true", help="delete the sim state and start over")
    ap.add_argument("--date", help="treat this YYYY-MM-DD as today (testing)")
    args = ap.parse_args()

    if args.reset:
        if STATE_PATH.exists():
            ans = input("Delete sim_state.json and start a fresh sim? Type yes to confirm: ").strip().lower()
            if ans != "yes":
                print("Left as is.")
                return
            STATE_PATH.unlink()
        print("Fresh sim.")

    try:
        import yfinance  # noqa: F401
    except ImportError:
        raise SystemExit("yfinance is not installed. Run:  pip install yfinance pandas numpy")

    state = load_state()
    md = YahooData()
    if args.loop:
        run_loop(md, state, args)
        return
    print(f"Downloading daily bars for {len(UNIVERSE)} symbols...")
    summary = one_pass(md, state, args, intraday=args.intraday)
    if not args.dry_run and not summary["ran"] and not (JS_PATH.exists() and JSON_PATH.exists()):
        save_state(state)
        export(state)   # first run on a closed market still writes the files so the page has something to read
    print_summary(state, summary)
    if args.dry_run:
        print("\nDry run: nothing was saved.")


if __name__ == "__main__":
    main()
