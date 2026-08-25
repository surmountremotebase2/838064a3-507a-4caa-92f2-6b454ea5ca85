"""Surmount-compatible ATHENA proxy.

This file uses only the documented Surmount inputs: OHLCV, SocialSentiment,
and InsiderTrading. It approximates PMC/flow/options with price-volume proxies.
It cannot access ATHENA's private PMC, order-flow, GEX, option-chain, or news
blackout state. Use athena_equity_full_strategy.py for the real ATHENA path.
"""
from datetime import datetime
from math import isfinite

from surmount.base_class import Strategy, TargetAllocation
from surmount.data import InsiderTrading, SocialSentiment


ETF = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI"]
LEV_LONG = {"TQQQ", "SOXL", "UPRO", "LABU"}
LEV_SHORT = {"SQQQ", "SOXS", "SPXU", "LABD"}
LEV = LEV_LONG | LEV_SHORT
STOCKS = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL", "AVGO", "MU", "PLTR"]
SYMBOLS = ETF + sorted(LEV) + STOCKS + ["VIXY"]


def _bars(data, sym):
    return [row[sym] for row in data.get("ohlcv", []) if sym in row]


def _v(xs, key):
    return [float(x.get(key, 0) or 0) for x in xs]


def _sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


def _ema(xs, n):
    if len(xs) < n:
        return None
    value = sum(xs[:n]) / n
    k = 2.0 / (n + 1.0)
    for x in xs[n:]:
        value = k * x + (1.0 - k) * value
    return value


def _atr(xs, n=14):
    if len(xs) < n + 1:
        return None
    tr = []
    for p, b in zip(xs[-n - 1:-1], xs[-n:]):
        h, l, pc = float(b["high"]), float(b["low"]), float(p["close"])
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(tr) / n


def _rsi(c, n=14):
    if len(c) < n + 1:
        return None
    d = [b - a for a, b in zip(c[-n - 1:-1], c[-n:])]
    g = sum(max(x, 0.0) for x in d) / n
    l = sum(max(-x, 0.0) for x in d) / n
    return 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)


def _vwap(xs):
    vol = sum(float(x.get("volume", 0) or 0) for x in xs)
    return (sum(float(x["close"]) * float(x.get("volume", 0) or 0) for x in xs) / vol
            if vol else None)


def _ret(c, n):
    return c[-1] / c[-n - 1] - 1.0 if len(c) > n else None


def _social_ok(data, sym):
    rows = data.get(("social_sentiment", sym), []) or []
    if not rows:
        return 0.0
    row = rows[-1]
    vals = [row.get("stocktwitsSentiment"), row.get("twitterSentiment")]
    vals = [float(x) for x in vals if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = SYMBOLS
        self.data_list = [SocialSentiment(s) for s in SYMBOLS] + [InsiderTrading(s) for s in SYMBOLS]

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def _regime(self, data):
        spy, qqq = _bars(data, "SPY"), _bars(data, "QQQ")
        if len(spy) < 60 or len(qqq) < 60:
            return "NO_TRADE"
        sc, qc = _v(spy, "close"), _v(qqq, "close")
        se9, se21, qe9, qe21 = _ema(sc, 9), _ema(sc, 21), _ema(qc, 9), _ema(qc, 21)
        sv, qv = _vwap(spy[-78:]), _vwap(qqq[-78:])
        a = _atr(spy)
        if None in (se9, se21, qe9, qe21, sv, qv, a):
            return "NO_TRADE"
        if a / sc[-1] > 0.018:
            return "HIGH_VOL"
        if sc[-1] > sv and qc[-1] > qv and se9 > se21 and qe9 > qe21:
            return "BULL"
        if sc[-1] < sv and qc[-1] < qv and se9 < se21 and qe9 < qe21:
            return "BEAR"
        return "RANGE"

    def _score(self, sym, data, regime):
        if sym in {"VIXY"} or (regime == "BULL" and sym in LEV_SHORT) or (regime == "BEAR" and sym in LEV_LONG):
            return None
        if regime == "RANGE" and sym in LEV:
            return None
        xs = _bars(data, sym)
        if len(xs) < 90:
            return None
        c, vol = _v(xs, "close"), _v(xs, "volume")
        price, a = c[-1], _atr(xs)
        e9, e21, e50 = _ema(c, 9), _ema(c, 21), _ema(c, 50)
        vw, rs, av = _vwap(xs[-78:]), _rsi(c), _sma(vol, 20)
        if None in (price, a, e9, e21, e50, vw, rs, av) or av <= 0:
            return None
        rv = vol[-1] / av
        m5, m20 = _ret(c, 5), _ret(c, 20)
        if None in (m5, m20) or price < 10 or a / price > 0.025:
            return None
        score = 0.0
        if regime == "BULL":
            if not (e9 > e21 > e50 and price > vw and m20 > 0):
                return None
            if c[-1] > max(float(x["high"]) for x in xs[-7:-1]) and rv >= 1.25:
                score += 3.0  # ORB/breakout proxy
            if price >= vw and c[-2] <= vw and m5 > 0:
                score += 2.0  # VWAP pullback
            score += max(0.0, m5 * 30.0) + min(1.5, max(0.0, rv - 1.0))
        elif regime == "BEAR":
            if not (e9 < e21 < e50 and price < vw and m20 < 0):
                return None
            if c[-1] < min(float(x["low"]) for x in xs[-7:-1]) and rv >= 1.25:
                score += 3.0
            if price <= vw and c[-2] >= vw and m5 < 0:
                score += 2.0
            score += max(0.0, -m5 * 30.0) + min(1.5, max(0.0, rv - 1.0))
        elif regime == "RANGE":
            if sym in LEV:
                return None
            z = (price - vw) / a
            if z > -1.0 or rs > 42:
                return None
            score = 2.0 + min(1.0, -z / 2.0)
        else:
            return None
        sentiment = _social_ok(data, sym)
        score += max(-0.5, min(0.5, (sentiment - 0.5) * 2.0))
        # Insider sales are a veto for a long equity candidate when recent.
        insider = data.get(("insider_trading", sym), []) or []
        if insider and "Sale" in str(insider[-1].get("transactionType", "")):
            score -= 0.5
        score -= min(2.0, a / price * 80.0)
        return score if score > 1.0 else None

    def run(self, data):
        regime = self._regime(data)
        if regime in {"NO_TRADE", "HIGH_VOL"}:
            return TargetAllocation({})
        scored = [(self._score(s, data, regime), s) for s in self.tickers]
        scored = sorted([(x, s) for x, s in scored if x is not None], reverse=True)
        selected = scored[:2]
        if not selected:
            return TargetAllocation({})
        out = {}
        for score, sym in selected:
            lev = sym in LEV
            weight = 0.18 if lev else 0.35
            out[sym] = min(weight, max(0.05, weight * min(1.0, score / 4.0)))
        total = sum(out.values())
        if total > 0.70:
            out = {k: v * 0.70 / total for k, v in out.items()}
        return TargetAllocation(out)