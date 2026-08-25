"""
Surmount Intraday Equity / ETF Strategy — VLP/ILP-Inspired Research Version

Purpose:
- Allocation-only intraday signal strategy for Surmount.
- ATHENA must independently own stops, partial exits, trailing exits,
  time stops, EOD flattening, PMC/order-flow validation, and execution.

Important:
- Research/paper only until validated with walk-forward, realistic spread,
  slippage, and allocation-turnover assumptions.
- All session times are US/Eastern regular-session times.
"""

from datetime import datetime
from math import sqrt
from zoneinfo import ZoneInfo

from surmount.base_class import Strategy, TargetAllocation
from surmount.data import InsiderTrading, SocialSentiment


ET = ZoneInfo("America/New_York")

CORE_ETF = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI"]
DEFENSIVE = ["BIL", "SHY", "TLT", "GLD"]
LONG_LEVERAGED = ["TQQQ", "SOXL", "UPRO", "LABU", "TECL"]
INVERSE_LEVERAGED = ["SQQQ", "SOXS", "SPXU", "LABD", "FAZ", "PSQ"]
STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL",
    "AVGO", "MU", "PLTR", "JPM", "XOM",
]

UNIVERSE = (
    CORE_ETF
    + DEFENSIVE
    + LONG_LEVERAGED
    + INVERSE_LEVERAGED
    + STOCKS
    + ["VIXY"]
)

LEVERAGED = set(LONG_LEVERAGED + INVERSE_LEVERAGED)
INVERSE = set(INVERSE_LEVERAGED)

INVERSE_UNDERLYING = {
    "SQQQ": "QQQ",
    "PSQ": "QQQ",
    "SPXU": "SPY",
    "SOXS": "SMH",
    "LABD": "XBI",
    "FAZ": "XLF",
}

TECH_FAMILY = {
    "QQQ", "SMH", "XLK", "TQQQ", "SOXL", "TECL",
    "NVDA", "AMD", "AVGO", "MU",
}
SP_FAMILY = {
    "SPY", "DIA", "UPRO", "AAPL", "MSFT", "AMZN",
    "META", "GOOGL", "JPM", "XOM",
}

MIN_HISTORY_BARS = 1_200  # About 20 regular sessions of 5-minute bars.
MAX_TOTAL_ALLOCATION = 0.40
MAX_SINGLE_UNLEVERED = 0.18
MAX_SINGLE_LEVERED = 0.08


def _bars(data, symbol):
    return [row[symbol] for row in data.get("ohlcv", []) if symbol in row]


def _values(xs, key):
    return [float(x.get(key, 0) or 0) for x in xs]


def _sma(values, n):
    return sum(values[-n:]) / n if len(values) >= n else None


def _ema(values, n):
    if len(values) < n:
        return None

    value = sum(values[:n]) / n
    alpha = 2.0 / (n + 1.0)

    for item in values[n:]:
        value = alpha * item + (1.0 - alpha) * value

    return value


def _roc(values, n):
    if len(values) <= n or values[-n - 1] == 0:
        return None
    return values[-1] / values[-n - 1] - 1.0


def _atr(xs, n=14):
    if len(xs) < n + 1:
        return None

    true_ranges = []
    for previous, current in zip(xs[-n - 1:-1], xs[-n:]):
        high = float(current["high"])
        low = float(current["low"])
        prior_close = float(previous["close"])
        true_ranges.append(
            max(high - low, abs(high - prior_close), abs(low - prior_close))
        )

    return sum(true_ranges) / n if true_ranges else None


def _rsi(values, n=14):
    if len(values) < n + 1:
        return None

    changes = [
        newer - older
        for older, newer in zip(values[-n - 1:-1], values[-n:])
    ]
    average_gain = sum(max(change, 0.0) for change in changes) / n
    average_loss = sum(max(-change, 0.0) for change in changes) / n

    if average_loss == 0:
        return 100.0

    rs = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _efficiency(values, n=20):
    if len(values) < n + 1:
        return None

    displacement = abs(values[-1] - values[-n - 1])
    path = sum(
        abs(newer - older)
        for older, newer in zip(values[-n - 1:-1], values[-n:])
    )

    return displacement / path if path > 0 else 0.0


def _bollinger(values, n=20, multiple=2.0):
    if len(values) < n:
        return None

    middle = _sma(values, n)
    if middle is None:
        return None

    variance = sum((value - middle) ** 2 for value in values[-n:]) / n
    deviation = sqrt(variance)

    return (
        middle - multiple * deviation,
        middle,
        middle + multiple * deviation,
    )


def _parse_stamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ET)
        return parsed.astimezone(ET)
    except (TypeError, ValueError):
        return None


def _stamp(bar):
    return _parse_stamp(bar.get("date") or bar.get("datetime") or bar.get("time"))


def _last_stamp(data):
    rows = data.get("ohlcv", [])
    if not rows:
        return None

    latest_row = rows[-1]
    if not latest_row:
        return None

    sample = next(iter(latest_row.values()))
    return _stamp(sample)


def _is_regular_session_bar(bar):
    timestamp = _stamp(bar)
    if timestamp is None or timestamp.weekday() >= 5:
        return False

    minute = timestamp.hour * 60 + timestamp.minute
    return 9 * 60 + 30 <= minute < 16 * 60


def _day_key(bar):
    timestamp = _stamp(bar)
    return timestamp.date().isoformat() if timestamp else None


def _session_bars(xs):
    session = [bar for bar in xs if _is_regular_session_bar(bar)]
    if not session:
        return [], []

    latest_day = _day_key(session[-1])
    today = [bar for bar in session if _day_key(bar) == latest_day]
    prior = [bar for bar in session if _day_key(bar) != latest_day]
    return today, prior


def _session_vwap(xs):
    today, _ = _session_bars(xs)
    if not today:
        return None

    total_volume = 0.0
    total_value = 0.0

    for bar in today:
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar.get("volume", 0) or 0)

        typical_price = (high + low + close) / 3.0
        total_volume += volume
        total_value += typical_price * volume

    return total_value / total_volume if total_volume > 0 else None


def _bar_slot(bar):
    timestamp = _stamp(bar)
    if timestamp is None:
        return None
    return timestamp.hour * 60 + timestamp.minute


def _tod_rvol(xs, lookback_sessions=15):
    today, prior = _session_bars(xs)
    if not today or not prior:
        return None

    latest = today[-1]
    slot = _bar_slot(latest)
    if slot is None:
        return None

    prior_days = []
    seen = set()

    for bar in reversed(prior):
        day = _day_key(bar)
        if day and day not in seen:
            seen.add(day)
            prior_days.append(day)
        if len(prior_days) >= lookback_sessions:
            break

    comparable_volumes = [
        float(bar.get("volume", 0) or 0)
        for bar in prior
        if _bar_slot(bar) == slot and _day_key(bar) in set(prior_days)
    ]

    if len(comparable_volumes) < 5:
        return None

    baseline = sum(comparable_volumes) / len(comparable_volumes)
    current = float(latest.get("volume", 0) or 0)
    return current / baseline if baseline > 0 else None


def _social(data, symbol):
    rows = data.get(("social_sentiment", symbol), []) or []
    if not rows:
        return None, False

    latest = rows[-1]
    values = [
        latest.get("stocktwitsSentiment"),
        latest.get("twitterSentiment"),
    ]
    values = [float(value) for value in values if value is not None]

    if not values:
        return None, False

    sentiment = sum(values) / len(values)
    return sentiment, sentiment < 0.15 or sentiment > 0.85


def _insider_sale(data, symbol):
    rows = data.get(("insider_trading", symbol), []) or []
    if not rows:
        return False

    transaction = str(rows[-1].get("transactionType", "")).lower()
    return "sale" in transaction


def _market_family(symbol):
    if symbol in TECH_FAMILY:
        return "TECH"
    if symbol in SP_FAMILY:
        return "BROAD_MARKET"
    if symbol in {"XLF", "FAZ", "JPM"}:
        return "FINANCIALS"
    if symbol in {"XBI", "LABU", "LABD"}:
        return "BIOTECH"
    if symbol in {"XLE", "XOM"}:
        return "ENERGY"
    if symbol in DEFENSIVE:
        return "DEFENSIVE"
    return symbol


class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = UNIVERSE
        alt_symbols = CORE_ETF + STOCKS
        self.data_list = (
            [SocialSentiment(symbol) for symbol in alt_symbols]
            + [InsiderTrading(symbol) for symbol in alt_symbols]
        )

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def _phase(self, data):
        timestamp = _last_stamp(data)
        if timestamp is None:
            return "UNKNOWN"

        minutes = timestamp.hour * 60 + timestamp.minute

        if minutes < 9 * 60 + 35:
            return "PRE_OPEN"
        if minutes < 9 * 60 + 50:
            return "OPENING_AUCTION"
        if minutes < 10 * 60 + 30:
            return "OPENING_DRIVE"
        if minutes < 11 * 60 + 30:
            return "MORNING"
        if minutes < 14 * 60:
            return "MIDDAY"
        if minutes < 15 * 60 + 30:
            return "AFTERNOON"
        if minutes < 15 * 60 + 50:
            return "CLOSE_WINDOW"
        return "FLATTEN"

    def _market_context(self, data):
        required = ("SPY", "QQQ", "IWM", "VIXY", "TLT", "GLD")
        market = {symbol: _bars(data, symbol) for symbol in required}

        if any(len(market[symbol]) < MIN_HISTORY_BARS for symbol in required):
            return {"regime": "NO_TRADE"}

        trends = []
        efficiencies = []

        for symbol in ("SPY", "QQQ", "IWM"):
            xs = market[symbol]
            closes = _values(xs, "close")
            price = closes[-1]
            vwap = _session_vwap(xs)
            ema20 = _ema(closes, 20)
            ema50 = _ema(closes, 50)
            ema100 = _ema(closes, 100)
            return_30m = _roc(closes, 6)
            return_100m = _roc(closes, 20)
            efficiency = _efficiency(closes, 20)

            if None in (
                vwap, ema20, ema50, ema100,
                return_30m, return_100m, efficiency,
            ):
                return {"regime": "NO_TRADE"}

            trends.append({
                "symbol": symbol,
                "price": price,
                "vwap": vwap,
                "up": price > vwap and ema20 > ema50 > ema100 and return_100m > 0,
                "down": price < vwap and ema20 < ema50 < ema100 and return_100m < 0,
                "return_30m": return_30m,
            })
            efficiencies.append(efficiency)

        spy_atr = _atr(market["SPY"])
        spy_price = trends[0]["price"]
        vix_closes = _values(market["VIXY"], "close")
        vix_fast = _ema(vix_closes, 10)
        vix_slow = _ema(vix_closes, 30)

        if None in (spy_atr, vix_fast, vix_slow) or spy_price <= 0:
            return {"regime": "NO_TRADE"}

        realized_volatility = spy_atr / spy_price
        up_count = sum(item["up"] for item in trends)
        down_count = sum(item["down"] for item in trends)
        mean_efficiency = sum(efficiencies) / len(efficiencies)

        high_volatility = (
            realized_volatility > 0.012
            or vix_fast > vix_slow * 1.08
        )

        if high_volatility:
            regime = "HIGH_VOL"
        elif up_count >= 2:
            regime = "BULL_TREND"
        elif down_count >= 2:
            regime = "BEAR_TREND"
        elif mean_efficiency < 0.20:
            regime = "RANGE_MEAN_REVERSION"
        else:
            regime = "CHOPPY"

        return {
            "regime": regime,
            "market": market,
            "trends": trends,
            "realized_volatility": realized_volatility,
            "market_downside_pressure": (
                trends[0]["return_30m"] < -0.004
                and trends[0]["price"] < trends[0]["vwap"]
            ),
        }

    def _reference(self, xs):
        today, prior = _session_bars(xs)

        if len(today) < 6 or not prior:
            return None

        prior_close = float(prior[-1]["close"])
        opening_price = float(today[0]["open"])
        orb15 = today[:3]
        orb30 = today[:6]

        return {
            "prior_close": prior_close,
            "open": opening_price,
            "gap": opening_price / prior_close - 1.0 if prior_close else 0.0,
            "orb15_high": max(float(bar["high"]) for bar in orb15),
            "orb15_low": min(float(bar["low"]) for bar in orb15),
            "orb30_high": max(float(bar["high"]) for bar in orb30),
            "orb30_low": min(float(bar["low"]) for bar in orb30),
        }

    def _candidate(self, symbol, data, ctx, phase):
        regime = ctx["regime"]

        if symbol == "VIXY":
            return None

        if phase in {"UNKNOWN", "PRE_OPEN", "OPENING_AUCTION", "FLATTEN"}:
            return None

        if regime in {"NO_TRADE", "HIGH_VOL"}:
            return None

        if regime in {"RANGE_MEAN_REVERSION", "CHOPPY"} and symbol in LEVERAGED:
            return None

        if regime == "BULL_TREND" and symbol in INVERSE:
            return None

        if regime == "BEAR_TREND" and symbol not in INVERSE:
            return None

        xs = _bars(data, symbol)
        if len(xs) < MIN_HISTORY_BARS:
            return None

        closes = _values(xs, "close")
        volumes = _values(xs, "volume")
        price = closes[-1]

        atr = _atr(xs)
        vwap = _session_vwap(xs)
        rvol = _tod_rvol(xs)
        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        ema50 = _ema(closes, 50)
        ema100 = _ema(closes, 100)
        rsi = _rsi(closes, 14)
        roc5 = _roc(closes, 5)
        roc21 = _roc(closes, 21)
        reference = self._reference(xs)

        required = (
            atr, vwap, rvol, ema9, ema21, ema50,
            ema100, rsi, roc5, roc21, reference,
        )
        if any(value is None for value in required):
            return None

        if price < 5 or atr / price > 0.035:
            return None

        dollar_volumes = [close * volume for close, volume in zip(closes, volumes)]
        average_dollar_volume = _sma(dollar_volumes, 20)

        if average_dollar_volume is None or average_dollar_volume < 20_000_000:
            return None

        current = xs[-1]
        bar_range = max(
            float(current["high"]) - float(current["low"]),
            0.01,
        )
        close_location = (
            float(current["close"]) - float(current["low"])
        ) / bar_range

        score = 0.0
        lanes = []

        if regime == "BULL_TREND":
            if not (price > vwap and ema9 > ema21 > ema50 and roc21 > 0):
                return None

            score += 2.0
            lanes.append("TREND_ALIGNMENT")

            if price > reference["orb15_high"] and rvol >= 1.20:
                score += 2.0
                lanes.append("ORB15_BREAKOUT")

            if (
                float(xs[-2]["close"]) <= vwap
                and price > vwap
                and roc5 > 0
                and rvol >= 1.05
            ):
                score += 1.75
                lanes.append("VWAP_RECLAIM")

            if (
                reference["gap"] >= 0.01
                and price > reference["open"]
                and rvol >= 1.15
            ):
                score += 1.25
                lanes.append("GAP_CONTINUATION")

        elif regime == "BEAR_TREND":
            underlying = INVERSE_UNDERLYING.get(symbol)
            if underlying is None:
                return None

            underlying_bars = _bars(data, underlying)
            underlying_closes = _values(underlying_bars, "close")
            underlying_vwap = _session_vwap(underlying_bars)

            if (
                len(underlying_bars) < MIN_HISTORY_BARS
                or underlying_vwap is None
                or _roc(underlying_closes, 5) is None
            ):
                return None

            if not (
                underlying_closes[-1] < underlying_vwap
                and _roc(underlying_closes, 5) < 0
                and price > vwap
                and ema9 > ema21
            ):
                return None

            score += 2.5
            lanes.append("UNDERLYING_BREAKDOWN_CONFIRMED")

            if rvol >= 1.15:
                score += 1.25
                lanes.append("TIME_OF_DAY_RVOL")

            if roc5 > 0:
                score += 1.0
                lanes.append("INVERSE_MOMENTUM")

        elif regime in {"RANGE_MEAN_REVERSION", "CHOPPY"}:
            if symbol not in CORE_ETF and symbol not in STOCKS:
                return None

            if ctx["market_downside_pressure"]:
                return None

            bands = _bollinger(closes)
            if bands is None:
                return None

            vwap_distance_atr = (price - vwap) / atr

            if not (
                price <= bands[0]
                and vwap_distance_atr <= -1.0
                and rsi <= 38
                and close_location >= 0.60
            ):
                return None

            score += 2.0
            lanes.append("LIQUIDITY_SWEEP_RECLAIM_PROXY")

            if rvol >= 0.85:
                score += 1.0
                lanes.append("NORMAL_OR_BETTER_LIQUIDITY")

            if price > float(xs[-2]["low"]):
                score += 0.75
                lanes.append("FAILED_BREAKDOWN_PROXY")

        else:
            return None

        sentiment, sentiment_extreme = _social(data, symbol)
        if sentiment_extreme:
            return None

        if sentiment is not None:
            if sentiment >= 0.55:
                score += 0.25
                lanes.append("SENTIMENT_SUPPORT")
            elif sentiment < 0.25 and symbol not in INVERSE:
                return None

        if _insider_sale(data, symbol) and symbol not in INVERSE:
            score -= 0.50
            lanes.append("INSIDER_SALE_PENALTY")

        if close_location >= 0.70 and rvol >= 1.05:
            score += 0.50
            lanes.append("CLOSE_STRENGTH")

        if phase == "MIDDAY":
            score -= 0.75
        elif phase == "CLOSE_WINDOW":
            score -= 1.50

        if score < 4.0:
            return None

        return {
            "symbol": symbol,
            "score": score,
            "lanes": lanes,
            "price": price,
            "atr": atr,
            "atr_pct": atr / price,
            "leveraged": symbol in LEVERAGED,
            "family": _market_family(symbol),
        }

    def _allocation(self, candidate, regime):
        cap = (
            MAX_SINGLE_LEVERED
            if candidate["leveraged"]
            else MAX_SINGLE_UNLEVERED
        )

        score_multiplier = min(1.0, max(0.40, candidate["score"] / 7.0))
        volatility_multiplier = min(
            1.0,
            0.010 / max(candidate["atr_pct"], 0.004),
        )

        weight = cap * score_multiplier * volatility_multiplier

        if regime in {"CHOPPY", "RANGE_MEAN_REVERSION"}:
            weight *= 0.65

        return min(weight, cap)

    def run(self, data):
        context = self._market_context(data)
        phase = self._phase(data)
        regime = context["regime"]

        if phase in {"UNKNOWN", "PRE_OPEN", "OPENING_AUCTION", "FLATTEN"}:
            return TargetAllocation({})

        if regime == "NO_TRADE":
            return TargetAllocation({})

        if regime == "HIGH_VOL":
            return TargetAllocation({"BIL": 0.20})

        ranked = []
        for symbol in self.tickers:
            candidate = self._candidate(symbol, data, context, phase)
            if candidate is not None:
                ranked.append(candidate)

        ranked.sort(key=lambda item: item["score"], reverse=True)

        selected = []
        used_families = set()

        for candidate in ranked:
            if candidate["family"] in used_families:
                continue

            selected.append(candidate)
            used_families.add(candidate["family"])

            if len(selected) >= 2:
                break

        if not selected:
            return TargetAllocation({})

        allocations = {
            candidate["symbol"]: self._allocation(candidate, regime)
            for candidate in selected
        }

        total_allocation = sum(allocations.values())
        if total_allocation > MAX_TOTAL_ALLOCATION:
            scale = MAX_TOTAL_ALLOCATION / total_allocation
            allocations = {
                symbol: allocation * scale
                for symbol, allocation in allocations.items()
            }

        return TargetAllocation(allocations)