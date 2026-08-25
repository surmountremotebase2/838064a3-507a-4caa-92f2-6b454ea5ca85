"""Complete Surmount-compatible ATHENA-style equity/ETF rotation.

This is the most complete version possible inside Surmount Code Builder. It
uses OHLCV plus Surmount's documented SocialSentiment and InsiderTrading feeds.
PMC, true order-flow, live news, SPX option-chain/GEX, broker brackets, and
fixed-dollar risk are not exposed by the documented Surmount Strategy API; this
file therefore uses explicit price/volume proxies and fails closed when data is
missing. It is for paper testing, not a profitability guarantee.
"""
from datetime import datetime
from math import sqrt

from surmount.base_class import Strategy, TargetAllocation
from surmount.data import InsiderTrading, SocialSentiment


CORE_ETF = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI", "ARKK"]
DEFENSIVE = ["BIL", "SHY", "TLT", "GLD"]
LONG_LEV = ["TQQQ", "SOXL", "UPRO", "LABU", "TECL"]
INVERSE_LEV = ["SQQQ", "SOXS", "SPXU", "LABD", "FAZ", "PSQ"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL", "AVGO", "MU", "PLTR", "JPM", "XOM"]
UNIVERSE = CORE_ETF + DEFENSIVE + LONG_LEV + INVERSE_LEV + STOCKS + ["VIXY"]
LEV = set(LONG_LEV + INVERSE_LEV)
INVERSE = set(INVERSE_LEV)


def _bars(data, symbol):
    return [row[symbol] for row in data.get("ohlcv", []) if symbol in row]


def _series(xs, key):
    return [float(x.get(key, 0) or 0) for x in xs]


def _sma(v, n):
    return sum(v[-n:]) / n if len(v) >= n else None


def _ema(v, n):
    if len(v) < n:
        return None
    out = sum(v[:n]) / n
    k = 2.0 / (n + 1.0)
    for x in v[n:]:
        out = k * x + (1.0 - k) * out
    return out


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
    volume = sum(float(x.get("volume", 0) or 0) for x in xs)
    return (sum(float(x["close"]) * float(x.get("volume", 0) or 0) for x in xs) / volume
            if volume else None)


def _rvol(xs, n=20):
    if len(xs) < n + 1:
        return None
    base = sum(float(x.get("volume", 0) or 0) for x in xs[-n - 1:-1]) / n
    return float(xs[-1].get("volume", 0) or 0) / base if base else None


def _roc(c, n):
    return c[-1] / c[-n - 1] - 1.0 if len(c) > n else None


def _bb(c, n=20, mult=2.0):
    if len(c) < n:
        return None
    avg = _sma(c, n)
    sd = sqrt(sum((x - avg) ** 2 for x in c[-n:]) / n)
    return avg - mult * sd, avg, avg + mult * sd


def _bar_time(data):
    rows = data.get("ohlcv", [])
    if not rows:
        return None
    row = next(iter(rows[-1].values()))
    raw = row.get("date") or row.get("datetime") or row.get("time")
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _social(data, symbol):
    rows = data.get(("social_sentiment", symbol), []) or []
    if not rows:
        return None
    row = rows[-1]
    vals = [row.get("stocktwitsSentiment"), row.get("twitterSentiment")]
    vals = [float(x) for x in vals if x is not None]
    return sum(vals) / len(vals) if vals else None


def _recent_insider_sale(data, symbol):
    rows = data.get(("insider_trading", symbol), []) or []
    if not rows:
        return False
    return "sale" in str(rows[-1].get("transactionType", "")).lower()


class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = UNIVERSE
        self.data_list = ([SocialSentiment(s) for s in UNIVERSE]
                          + [InsiderTrading(s) for s in UNIVERSE])

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def _market(self, data):
        required = {s: _bars(data, s) for s in ("SPY", "QQQ", "IWM", "VIXY")}
        if any(len(required[s]) < 120 for s in required):
            return "NO_TRADE", 0.0
        trend = []
        for s in ("SPY", "QQQ", "IWM"):
            xs, c = required[s], _series(required[s], "close")
            e20, e50, e100, v = _ema(c, 20), _ema(c, 50), _ema(c, 100), _vwap(xs[-78:])
            if None in (e20, e50, e100, v):
                return "NO_TRADE", 0.0
            trend.append((c[-1] > v, e20 > e50 > e100, c[-1] < v, e20 < e50 < e100))
        vix = required["VIXY"]
        vc = _series(vix, "close")
        vfast, vslow = _ema(vc, 10), _ema(vc, 30)
        spy_atr = _atr(required["SPY"])
        spy_price = _series(required["SPY"], "close")[-1]
        if None in (vfast, vslow, spy_atr) or not spy_price:
            return "NO_TRADE", 0.0
        vol_ratio = spy_atr / spy_price
        if vol_ratio > 0.020 or vfast > vslow * 1.12:
            return "HIGH_VOL", vol_ratio
        bull = sum(a and b for a, b, _, _ in trend)
        bear = sum(c and d for _, _, c, d in trend)
        if bull >= 2:
            return "BULL", vol_ratio
        if bear >= 2:
            return "BEAR", vol_ratio
        return "RANGE", vol_ratio

    def _session(self, data):
        stamp = _bar_time(data)
        if stamp is None:
            return "UNKNOWN"
        minutes = stamp.hour * 60 + stamp.minute
        if minutes < 9 * 60 + 45:
            return "OPEN"
        if minutes < 11 * 60:
            return "MORNING"
        if minutes < 14 * 60:
            return "MIDDAY"
        if minutes < 15 * 60 + 20:
            return "AFTERNOON"
        if minutes < 15 * 60 + 50:
            return "CLOSE"
        return "FLATTEN"

    def _candidate(self, symbol, data, regime, phase):
        if symbol == "VIXY" or phase in {"UNKNOWN", "OPEN", "FLATTEN"}:
            return None
        if regime == "RANGE" and symbol in LEV:
            return None
        if regime == "BULL" and symbol in INVERSE:
            return None
        if regime == "BEAR" and symbol not in INVERSE and symbol not in DEFENSIVE:
            return None
        xs = _bars(data, symbol)
        if len(xs) < 120:
            return None
        c = _series(xs, "close")
        a, vw, rv = _atr(xs), _vwap(xs[-78:]), _rvol(xs)
        e9, e21, e50, e100 = _ema(c, 9), _ema(c, 21), _ema(c, 50), _ema(c, 100)
        rsi, r5, r20 = _rsi(c), _roc(c, 5), _roc(c, 20)
        if any(x is None for x in (a, vw, rv, e9, e21, e50, e100, rsi, r5, r20)):
            return None
        price = c[-1]
        if price < 5 or a / price > 0.04:
            return None
        score, lanes = 0.0, []
        long_side = regime in {"BULL", "RANGE"}
        if regime == "BULL":
            if not (price > vw and e9 > e21 > e50 and r20 > 0):
                return None
            lanes.append("MOMENTUM_TREND")
            score += 2.0
            if price > max(float(x["high"]) for x in xs[-7:-1]) and rv >= 1.25:
                lanes.append("ORB_VOLUME_BREAKOUT")
                score += 2.5
            if c[-2] <= vw <= price and r5 > 0:
                lanes.append("VWAP_PULLBACK")
                score += 2.0
            if rv >= 1.15:
                lanes.append("RELATIVE_VOLUME")
                score += 1.0
        elif regime == "BEAR":
            # Long-only Surmount expresses a bearish view through inverse ETFs
            # rather than shorting ordinary equities.
            if symbol not in INVERSE:
                return None
            underlying = "QQQ" if symbol in {"SQQQ", "PSQ"} else "SPY"
            u = _bars(data, underlying)
            uc = _series(u, "close")
            uv = _vwap(u[-78:])
            if len(u) < 120 or uv is None or not (uc[-1] < uv and _roc(uc, 5) < 0):
                return None
            lanes.append("BEAR_INVERSE_CONFIRMATION")
            score += 2.5
            if rv >= 1.15:
                lanes.append("RELATIVE_VOLUME")
                score += 1.0
        elif regime == "RANGE":
            if symbol not in CORE_ETF and symbol not in STOCKS:
                return None
            z = (price - vw) / a
            bb = _bb(c)
            if z > -1.0 or rsi > 42 or bb is None:
                return None
            lanes.append("VWAP_MEAN_REVERSION")
            score += 2.0
            if price <= bb[0]:
                lanes.append("BOUNDED_GRID_EDGE")
                score += 1.0
        else:
            return None
        sentiment = _social(data, symbol)
        if sentiment is not None:
            if (long_side and sentiment >= 0.50) or (not long_side and sentiment <= 0.50):
                lanes.append("SENTIMENT_CONFIRMATION")
                score += 0.35
            elif (long_side and sentiment < 0.30) or (not long_side and sentiment > 0.70):
                return None
        if _recent_insider_sale(data, symbol) and symbol not in INVERSE:
            score -= 0.50
        # Approximate flow: current bar closes in its upper/lower range with
        # expanding volume. It is not true bid/ask delta.
        last = xs[-1]
        span = max(float(last["high"]) - float(last["low"]), 0.01)
        close_location = (float(last["close"]) - float(last["low"])) / span
        if (long_side and close_location >= 0.70 and rv >= 1.1) or (not long_side and close_location <= 0.30 and rv >= 1.1):
            lanes.append("SIGNED_VOLUME_PROXY")
            score += 0.75
        if phase == "MIDDAY":
            score -= 0.50
        if phase == "CLOSE":
            score -= 1.25
        if score < 3.0:
            return None
        return score, lanes, a, price, vw

    def run(self, data):
        regime, vol_ratio = self._market(data)
        phase = self._session(data)
        if regime == "NO_TRADE" or phase in {"UNKNOWN", "FLATTEN", "OPEN"}:
            return TargetAllocation({})
        if regime == "HIGH_VOL":
            # No leveraged or momentum entry in stress; use a small defensive
            # allocation only when valid defensive history is available.
            return TargetAllocation({"BIL": 0.20}) if len(_bars(data, "BIL")) >= 120 else TargetAllocation({})
        ranked = []
        for symbol in self.tickers:
            result = self._candidate(symbol, data, regime, phase)
            if result:
                ranked.append((result[0], symbol, result))
        ranked.sort(reverse=True)
        selected, families = [], set()
        for score, symbol, result in ranked:
            family = "NASDAQ" if symbol in {"QQQ", "TQQQ", "SQQQ", "SMH", "SOXL", "SOXS"} else "SP500" if symbol in {"SPY", "UPRO", "SPXU", "DIA"} else symbol
            if family in families:
                continue
            selected.append((score, symbol, result))
            families.add(family)
            if len(selected) == 2:
                break
        if not selected:
            return TargetAllocation({})
        allocation = {}
        for score, symbol, result in selected:
            weight = 0.12 if symbol in LEV else 0.30
            if "BOUNDED_GRID_EDGE" in result[1]:
                weight = 0.15
            weight *= min(1.0, max(0.25, score / 6.0))
            # Volatility scaling; no Surmount account-equity input is assumed.
            weight *= min(1.0, 0.012 / max(result[2] / result[3], 0.004))
            allocation[symbol] = min(weight, 0.35)
        total = sum(allocation.values())
        if total > 0.65:
            allocation = {k: v * 0.65 / total for k, v in allocation.items()}
        return TargetAllocation(allocation)