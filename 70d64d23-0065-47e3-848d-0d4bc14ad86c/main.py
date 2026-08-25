"""
ATHENA-style equity/ETF intraday regime rotation for Surmount Code Builder.

This is an allocation strategy because Surmount Code Builder returns only
TargetAllocation. A non-zero allocation is an entry/hold instruction and zero
is an exit instruction at the next rebalance. Native broker stop/target orders
and guaranteed 15:55 liquidation are not exposed by the documented Surmount
Strategy API; those require a separate IBKR execution layer.
"""

from datetime import datetime
from surmount.base_class import Strategy, TargetAllocation


CORE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI", "ARKK"]
LEVERAGED_LONG = {"TQQQ", "SOXL", "UPRO", "LABU"}
LEVERAGED_SHORT = {"SQQQ", "SOXS", "SPXU", "LABD"}
LEVERAGED = LEVERAGED_LONG | LEVERAGED_SHORT
UNIVERSE = CORE + sorted(LEVERAGED) + [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL",
    "NFLX", "AVGO", "MU", "COIN", "PLTR", "LLY", "JPM", "XOM",
]


def _bars(data, symbol):
    return [row[symbol] for row in data.get("ohlcv", []) if symbol in row]


def _close(xs):
    return [float(x["close"]) for x in xs if x.get("close") is not None]


def _vol(xs):
    return [float(x.get("volume", 0) or 0) for x in xs]


def _sma(v, n):
    return sum(v[-n:]) / n if len(v) >= n else None


def _ema(v, n):
    if len(v) < n:
        return None
    value = sum(v[:n]) / n
    alpha = 2.0 / (n + 1.0)
    for item in v[n:]:
        value = alpha * item + (1 - alpha) * value
    return value


def _atr(xs, n=14):
    if len(xs) < n + 1:
        return None
    out = []
    for previous, current in zip(xs[-n - 1:-1], xs[-n:]):
        high, low = float(current["high"]), float(current["low"])
        previous_close = float(previous["close"])
        out.append(max(high - low, abs(high - previous_close),
                       abs(low - previous_close)))
    return sum(out) / n


def _rsi(v, n=14):
    if len(v) < n + 1:
        return None
    changes = [b - a for a, b in zip(v[-n - 1:-1], v[-n:])]
    gain = sum(max(x, 0.0) for x in changes) / n
    loss = sum(max(-x, 0.0) for x in changes) / n
    return 100.0 if loss == 0 else 100.0 - (100.0 / (1.0 + gain / loss))


def _vwap(xs):
    volume = sum(float(x.get("volume", 0) or 0) for x in xs)
    if volume <= 0:
        return None
    return sum(float(x["close"]) * float(x.get("volume", 0) or 0)
               for x in xs) / volume


def _return(v, n):
    return v[-1] / v[-n - 1] - 1.0 if len(v) > n else None


def _time_of_last_bar(data):
    rows = data.get("ohlcv", [])
    if not rows:
        return None
    sample = next(iter(rows[-1].values()))
    raw = sample.get("date") or sample.get("datetime") or sample.get("time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = UNIVERSE
        self.data_list = []

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def _market_snapshot(self, data):
        snapshots = {}
        for symbol in ("SPY", "QQQ", "IWM"):
            xs = _bars(data, symbol)
            c = _close(xs)
            if len(xs) < 60:
                return None
            snapshots[symbol] = {
                "price": c[-1],
                "ema9": _ema(c, 9),
                "ema21": _ema(c, 21),
                "ema50": _ema(c, 50),
                "vwap": _vwap(xs[-78:]),
                "atr": _atr(xs),
                "r5": _return(c, 5),
                "r20": _return(c, 20),
            }
        good = all(all(v is not None for v in x.values()) for x in snapshots.values())
        if not good:
            return None
        up = sum(x["price"] > x["vwap"] and x["ema9"] > x["ema21"]
                 for x in snapshots.values())
        down = sum(x["price"] < x["vwap"] and x["ema9"] < x["ema21"]
                   for x in snapshots.values())
        atr_pct = snapshots["SPY"]["atr"] / snapshots["SPY"]["price"]
        if atr_pct >= 0.018:
            regime = "HIGH_VOL"
        elif up >= 3 and sum(x["r20"] > 0 for x in snapshots.values()) >= 2:
            regime = "BULL"
        elif down >= 3 and sum(x["r20"] < 0 for x in snapshots.values()) >= 2:
            regime = "BEAR"
        elif up <= 1 and down <= 1:
            regime = "RANGE"
        else:
            regime = "MIXED"
        return {"regime": regime, "symbols": snapshots}

    def _session_phase(self, data):
        stamp = _time_of_last_bar(data)
        if stamp is None:
            return "UNKNOWN"
        minutes = stamp.hour * 60 + stamp.minute
        if minutes < 9 * 60 + 45:
            return "OPEN"
        if minutes < 11 * 60:
            return "MORNING"
        if minutes < 14 * 60:
            return "MIDDAY"
        if minutes < 15 * 60 + 15:
            return "AFTERNOON"
        if minutes < 15 * 60 + 50:
            return "CLOSE_WINDOW"
        return "FLATTEN"

    def _daily_reference(self, xs):
        # Uses the first 78 bars as a prior-session proxy when daily bars are
        # unavailable. The strategy remains conservative if history is short.
        if len(xs) < 90:
            return None
        prior = xs[-79:-1]
        return {
            "high": max(float(x["high"]) for x in prior),
            "low": min(float(x["low"]) for x in prior),
            "open": float(prior[0]["open"]),
            "close": float(prior[-1]["close"]),
        }

    def _candidate(self, symbol, data, regime, phase):
        xs = _bars(data, symbol)
        c, volume = _close(xs), _vol(xs)
        if len(xs) < 90 or len(c) < 90:
            return None
        price, a = c[-1], _atr(xs)
        e9, e21, e50 = _ema(c, 9), _ema(c, 21), _ema(c, 50)
        vw, rs = _vwap(xs[-78:]), _rsi(c)
        avg_volume = _sma(volume, 20)
        if any(x is None for x in (price, a, e9, e21, e50, vw, rs, avg_volume)):
            return None
        if price < 10 or avg_volume <= 0 or a / price > 0.025:
            return None
        relvol = volume[-1] / avg_volume
        r5, r20 = _return(c, 5), _return(c, 20)
        reference = self._daily_reference(xs)
        if r5 is None or r20 is None or reference is None:
            return None
        gap = price / reference["close"] - 1.0
        above_vwap = price > vw
        below_vwap = price < vw
        trend_long = e9 > e21 > e50 and above_vwap
        trend_short = e9 < e21 < e50 and below_vwap
        breakout = price > reference["high"] + 0.10 * a
        breakdown = price < reference["low"] - 0.10 * a
        # The score is a ranking score, not a promise of profitability.
        if regime == "BULL":
            if symbol in LEVERAGED_SHORT or not trend_long:
                return None
            strategies = []
            if breakout and relvol >= 1.25:
                strategies.append(("ORB_MOMENTUM", 3.0))
            if above_vwap and r5 > 0 and r20 > 0 and relvol >= 1.10:
                strategies.append(("MOMENTUM", 2.0))
            if price >= vw and c[-2] <= vw and r20 > 0:
                strategies.append(("VWAP_PULLBACK", 2.0))
            if gap >= 0.02 and breakout:
                strategies.append(("GAP_GO", 1.0))
        elif regime == "BEAR":
            if symbol in LEVERAGED_LONG or not trend_short:
                return None
            strategies = []
            if breakdown and relvol >= 1.25:
                strategies.append(("ORB_BREAKDOWN", 3.0))
            if below_vwap and r5 < 0 and r20 < 0 and relvol >= 1.10:
                strategies.append(("DOWNSIDE_MOMENTUM", 2.0))
            if price <= vw and c[-2] >= vw and r20 < 0:
                strategies.append(("VWAP_REJECTION", 2.0))
            if gap <= -0.02 and breakdown:
                strategies.append(("GAP_DOWN", 1.0))
        elif regime == "RANGE":
            if symbol in LEVERAGED:
                return None
            z = (price - vw) / a
            strategies = []
            if z <= -1.0 and rs <= 40:
                strategies.append(("RANGE_LONG", 2.0))
            # A long-only allocation engine cannot safely express a short
            # mean-reversion trade. Overbought range conditions therefore
            # produce no allocation rather than accidentally buying the ETF.
        else:
            return None
        if not strategies:
            return None
        name, base = max(strategies, key=lambda x: x[1])
        # Relative strength and volume improve ranking; excessive volatility
        # reduces it. Strategies are eligible only after all gates above pass.
        score = base + max(-2.0, min(2.0, r20 * 20.0))
        score += min(1.5, max(0.0, relvol - 1.0))
        score -= min(2.0, (a / price) * 80.0)
        if phase == "MIDDAY":
            score -= 0.75
        if phase == "CLOSE_WINDOW":
            score -= 1.5
        return {
            "symbol": symbol, "score": score, "strategy": name,
            "atr_pct": a / price, "leveraged": symbol in LEVERAGED,
        }

    def _weight(self, candidate, regime):
        # Risk proxy: lower weight for high ATR and leveraged products.
        weight = 0.40 if candidate["leveraged"] else 0.50
        weight *= max(0.35, min(1.0, 0.012 / max(candidate["atr_pct"], 0.004)))
        if regime == "HIGH_VOL":
            weight *= 0.35
        return min(weight, 0.50)

    def run(self, data):
        market = self._market_snapshot(data)
        phase = self._session_phase(data)
        if market is None or phase in {"UNKNOWN", "FLATTEN"}:
            return TargetAllocation({})
        regime = market["regime"]
        if regime in {"HIGH_VOL", "MIXED"}:
            return TargetAllocation({})
        candidates = []
        for symbol in self.tickers:
            candidate = self._candidate(symbol, data, regime, phase)
            if candidate and candidate["score"] > 0:
                candidates.append(candidate)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        # Do not stack highly correlated index products in one decision.
        chosen, families = [], set()
        for candidate in candidates:
            family = ("NASDAQ" if candidate["symbol"] in {"QQQ", "TQQQ", "SQQQ", "SMH", "SOXL", "SOXS"}
                      else "SP500" if candidate["symbol"] in {"SPY", "UPRO", "SPXU", "XLK"}
                      else candidate["symbol"])
            if family in families:
                continue
            chosen.append(candidate)
            families.add(family)
            if len(chosen) == 2:
                break
        if not chosen:
            return TargetAllocation({})
        allocations = {c["symbol"]: self._weight(c, regime) for c in chosen}
        total = sum(allocations.values())
        if total > 0.80:
            scale = 0.80 / total
            allocations = {k: v * scale for k, v in allocations.items()}
        return TargetAllocation(allocations)