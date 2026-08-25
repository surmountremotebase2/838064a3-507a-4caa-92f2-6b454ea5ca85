"""
ATHENA / PMC-INSPIRED SURMOUNT INTRADAY MASTER STRATEGY
========================================================

Paste this entire file into Surmount Code Builder as main.py.

What this strategy does
-----------------------
- Uses 5-minute US equity/ETF OHLCV bars.
- Trades only when a specific, qualified setup is present.
- Uses a broad-market regime router.
- Supports:
  1. Bull-trend continuation.
  2. Fresh opening-range breakout.
  3. Fresh VWAP reclaim.
  4. Optional range reversal.
  5. Optional inverse-ETF continuation.
- Uses time-of-day adjusted relative volume when enough sessions exist.
- Uses a candle-pressure proxy, not true order-flow delta.
- Allocates based on ATR-risk proxy and strict caps.
- Fails closed to cash when data, regime, phase, or setup is uncertain.

What this strategy cannot do
----------------------------
- Guarantee profit or prevent capital loss.
- Submit broker stops, profit targets, bracket orders, or stop losses.
- See true order book, bid/ask delta, GEX, live news, or broker fills.
- Preserve correct live position state through Surmount alone.

ATHENA must own:
- Broker reconciliation.
- PMC and order-flow confirmation.
- Bid/ask spread and slippage guards.
- Stops, partial profits, trailing exits, time stops.
- Daily loss limit, drawdown limit, and kill switch.
- Forced flatten before end of session.
- Paper/live promotion gates.

Initial configuration
---------------------
Start with:
- Bull trend only.
- SPY / QQQ and liquid ETFs only.
- One position maximum.
- Maximum 15% allocation.
- No leverage.
- No inverse ETFs.
- No stocks.
- No midday entries.
- No range reversal.
- No automatic BIL allocation.

Do not enable additional modes until they improve cost-adjusted,
out-of-sample results in separate tests.
"""

from datetime import datetime
from math import sqrt

from surmount.base_class import Strategy, TargetAllocation


# ============================================================================
# FEATURE FLAGS
# ============================================================================
# Start conservative. Change only ONE flag per research experiment.

ENABLE_BULL_TREND = True
ENABLE_RANGE_REVERSAL = False
ENABLE_INVERSE = False
ENABLE_STOCKS = False
ENABLE_LEVERAGED = False
ENABLE_MIDDAY_ENTRIES = False
ENABLE_BIL_DEFENSIVE = False
ENABLE_CROSS_SECTIONAL_RANKING = False


# ============================================================================
# CORE RISK CONFIGURATION
# ============================================================================

INTERVAL = "5min"

MIN_MARKET_BARS = 250
MIN_CANDIDATE_BARS = 250

MIN_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 5_000_000.0

MAX_ATR_PCT_NORMAL = 0.045
MAX_ATR_PCT_LEVERAGED = 0.075

MAX_POSITIONS = 1
MAX_TOTAL_ALLOCATION = 0.15
MAX_NORMAL_ALLOCATION = 0.15
MAX_LEVERAGED_ALLOCATION = 0.06
MAX_RANGE_ALLOCATION = 0.10
BIL_ALLOCATION = 0.10

# Allocation risk proxy. ATHENA must replace this with account-equity-aware
# dollar risk using actual entry, stop, and filled position size.
RISK_PER_TRADE = 0.0015

# Reject trades whose theoretical favorable movement is too small to reasonably
# clear a conservative round-trip friction estimate.
MIN_EXPECTED_MOVE_COST_MULTIPLE = 3.0

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 100

ATR_PERIOD = 14
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
EFFICIENCY_PERIOD = 20

RVOL_LOOKBACK_SESSIONS = 10

# Important: These phase gates assume timestamps are US/Eastern. Confirm this
# from a short Surmount backtest before trusting intraday phase behavior.
NO_ENTRY_BEFORE = 9 * 60 + 50
MIDDAY_START = 11 * 60 + 30
MIDDAY_END = 13 * 60 + 30
NO_ENTRY_AFTER = 15 * 60 + 15
FORCE_FLATTEN_AFTER = 15 * 60 + 45


# ============================================================================
# UNIVERSE
# ============================================================================

CORE_ETF = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "SMH",
    "XBI",
]

DEFENSIVE = [
    "BIL",
    "SHY",
    "GLD",
]

LONG_LEVERAGED = [
    "TQQQ",
    "SOXL",
    "UPRO",
    "TECL",
]

INVERSE_LEVERAGED = [
    "SQQQ",
    "SOXS",
    "SPXU",
    "PSQ",
]

STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "MU",
    "JPM",
]

LEVERAGED = set(LONG_LEVERAGED + INVERSE_LEVERAGED)
INVERSE = set(INVERSE_LEVERAGED)

INVERSE_UNDERLYING = {
    "SQQQ": "QQQ",
    "PSQ": "QQQ",
    "SPXU": "SPY",
    "SOXS": "SMH",
}

TECH_FAMILY = {
    "QQQ",
    "XLK",
    "SMH",
    "TQQQ",
    "SOXL",
    "TECL",
    "SQQQ",
    "PSQ",
    "SOXS",
    "NVDA",
    "AMD",
    "AVGO",
    "MU",
}

BROAD_FAMILY = {
    "SPY",
    "DIA",
    "UPRO",
    "SPXU",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "JPM",
}


def build_universe():
    """
    Build the active symbols from explicit feature flags.

    Keep the default universe small for clear attribution. More symbols
    increase correlation, data failure risk, and parameter-mining risk.
    """
    symbols = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "XLK",
        "SMH",
    ]

    if ENABLE_RANGE_REVERSAL:
        symbols.extend([
            "XLF",
            "XLE",
            "XBI",
        ])

    if ENABLE_STOCKS:
        symbols.extend(STOCKS)

    if ENABLE_LEVERAGED:
        symbols.extend(LONG_LEVERAGED)

    if ENABLE_INVERSE:
        symbols.extend(INVERSE_LEVERAGED)

    if ENABLE_BIL_DEFENSIVE:
        symbols.append("BIL")

    return list(dict.fromkeys(symbols))


# ============================================================================
# SAFE DATA AND INDICATOR HELPERS
# ============================================================================

def bars_for(data, symbol):
    """Return valid OHLCV bars for a symbol or an empty list."""
    if not isinstance(data, dict):
        return []

    rows = data.get("ohlcv", [])

    if not isinstance(rows, list):
        return []

    output = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        bar = row.get(symbol)

        if isinstance(bar, dict):
            output.append(bar)

    return output


def float_value(bar, field):
    """Return a valid finite numeric value, otherwise None."""
    try:
        value = float(bar.get(field))
    except (AttributeError, TypeError, ValueError):
        return None

    if value != value:
        return None

    if value in (float("inf"), float("-inf")):
        return None

    return value


def series_for(bars, field):
    """
    Return the numerical field series.

    A missing OHLCV field invalidates the series instead of silently becoming
    zero. Failing closed is safer than calculating a distorted indicator.
    """
    output = []

    for bar in bars:
        value = float_value(bar, field)

        if value is None:
            return []

        output.append(value)

    return output


def sma(values, length):
    if length <= 0 or len(values) < length:
        return None

    return sum(values[-length:]) / float(length)


def ema(values, length):
    """Chronological exponential moving average."""
    if length <= 0 or len(values) < length:
        return None

    result = sum(values[:length]) / float(length)
    alpha = 2.0 / (length + 1.0)

    for value in values[length:]:
        result = alpha * value + (1.0 - alpha) * result

    return result


def roc(values, length):
    if length <= 0 or len(values) <= length:
        return None

    prior = values[-length - 1]

    if prior == 0:
        return None

    return values[-1] / prior - 1.0


def atr(bars, length=ATR_PERIOD):
    """Average true range from completed OHLCV bars."""
    if length <= 0 or len(bars) < length + 1:
        return None

    true_ranges = []

    for previous, current in zip(bars[-length - 1:-1], bars[-length:]):
        high = float_value(current, "high")
        low = float_value(current, "low")
        prior_close = float_value(previous, "close")

        if None in (high, low, prior_close):
            return None

        if high < low or low <= 0 or prior_close <= 0:
            return None

        true_ranges.append(
            max(
                high - low,
                abs(high - prior_close),
                abs(low - prior_close),
            )
        )

    if len(true_ranges) != length:
        return None

    return sum(true_ranges) / float(length)


def rsi(values, length=RSI_PERIOD):
    if length <= 0 or len(values) < length + 1:
        return None

    changes = [
        current - prior
        for prior, current in zip(values[-length - 1:-1], values[-length:])
    ]

    average_gain = sum(max(change, 0.0) for change in changes) / float(length)
    average_loss = sum(max(-change, 0.0) for change in changes) / float(length)

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss

    return 100.0 - 100.0 / (1.0 + relative_strength)


def bollinger(values, length=BOLLINGER_PERIOD, multiple=2.0):
    if length <= 0 or len(values) < length:
        return None

    middle = sma(values, length)

    if middle is None:
        return None

    window = values[-length:]
    variance = sum((value - middle) ** 2 for value in window) / float(length)
    deviation = sqrt(variance)

    return (
        middle - multiple * deviation,
        middle,
        middle + multiple * deviation,
    )


def efficiency(values, length=EFFICIENCY_PERIOD):
    """Directional movement divided by total path length."""
    if length <= 0 or len(values) < length + 1:
        return None

    window = values[-length - 1:]
    displacement = abs(window[-1] - window[0])

    path = sum(
        abs(current - prior)
        for prior, current in zip(window[:-1], window[1:])
    )

    return displacement / path if path > 0 else 0.0


def candle_pressure_proxy(bars, length=5):
    """
    Candle-body weighted volume proxy.

    This is not true PMC, bid/ask delta, market depth, or order-flow
    imbalance. ATHENA must independently verify real order flow.
    """
    if length <= 0 or len(bars) < length:
        return None

    weighted_sum = 0.0
    total_volume = 0.0

    for bar in bars[-length:]:
        high = float_value(bar, "high")
        low = float_value(bar, "low")
        open_price = float_value(bar, "open")
        close = float_value(bar, "close")
        volume = float_value(bar, "volume")

        if None in (high, low, open_price, close, volume):
            return None

        if volume < 0:
            return None

        bar_width = max(high - low, 0.01)

        weighted_sum += (
            ((close - open_price) / bar_width)
            * volume
        )

        total_volume += volume

    return weighted_sum / total_volume if total_volume > 0 else None


def close_location(bar):
    """Normalized close location in the latest candle: 0 to 1."""
    high = float_value(bar, "high")
    low = float_value(bar, "low")
    close = float_value(bar, "close")

    if None in (high, low, close):
        return None

    return (close - low) / max(high - low, 0.01)


# ============================================================================
# TIME, SESSION, VWAP, AND RELATIVE VOLUME
# ============================================================================

def parse_timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def bar_timestamp(bar):
    if not isinstance(bar, dict):
        return None

    return parse_timestamp(
        bar.get("date")
        or bar.get("datetime")
        or bar.get("time")
    )


def day_key(bar):
    timestamp = bar_timestamp(bar)

    if timestamp is None:
        return None

    return timestamp.date().isoformat()


def session_groups(bars):
    """
    Group bars by calendar date.

    Verify Surmount timestamp timezone. If timestamps are UTC, convert to
    America/New_York before enabling strict intraday phase enforcement.
    """
    groups = {}
    order = []

    for bar in bars:
        key = day_key(bar)

        if key is None:
            continue

        if key not in groups:
            groups[key] = []
            order.append(key)

        groups[key].append(bar)

    return [(key, groups[key]) for key in order]


def today_and_prior(bars):
    grouped = session_groups(bars)

    if not grouped:
        return [], []

    today = grouped[-1][1]
    prior = []

    for _, group in grouped[:-1]:
        prior.extend(group)

    return today, prior


def session_vwap(bars):
    """Current session VWAP using typical price."""
    today, _ = today_and_prior(bars)

    if not today:
        return None

    total_volume = 0.0
    total_value = 0.0

    for bar in today:
        high = float_value(bar, "high")
        low = float_value(bar, "low")
        close = float_value(bar, "close")
        volume = float_value(bar, "volume")

        if None in (high, low, close, volume):
            continue

        if volume <= 0:
            continue

        typical_price = (high + low + close) / 3.0

        total_volume += volume
        total_value += typical_price * volume

    return total_value / total_volume if total_volume > 0 else None


def session_rvol(bars, lookback_sessions=RVOL_LOOKBACK_SESSIONS):
    """
    Relative volume vs the same intraday slot across prior sessions.

    Falls back to a local trailing 20-bar baseline only when session history
    is unavailable, which is less reliable but permits short backtests.
    """
    grouped = session_groups(bars)

    if len(grouped) >= 4:
        today = grouped[-1][1]
        slot = len(today) - 1

        current_volume = float_value(today[-1], "volume")

        if current_volume is not None:
            prior_volumes = []

            for _, group in grouped[-lookback_sessions - 1:-1]:
                if len(group) > slot:
                    value = float_value(group[slot], "volume")

                    if value is not None and value >= 0:
                        prior_volumes.append(value)

            if len(prior_volumes) >= 3:
                baseline = sum(prior_volumes) / float(len(prior_volumes))

                if baseline > 0:
                    return current_volume / baseline

    if len(bars) < 21:
        return None

    volumes = series_for(bars, "volume")

    if not volumes:
        return None

    baseline = sma(volumes[-21:-1], 20)

    if baseline is None or baseline <= 0:
        return None

    return volumes[-1] / baseline


def reference_levels(bars):
    """Build session gap and opening-range reference levels."""
    today, prior = today_and_prior(bars)

    if len(today) < 3 or not prior:
        return None

    prior_close = float_value(prior[-1], "close")
    opening_price = float_value(today[0], "open")

    if prior_close is None or opening_price is None or prior_close <= 0:
        return None

    orb15 = today[:3]
    orb30 = today[:6] if len(today) >= 6 else []

    highs15 = [float_value(bar, "high") for bar in orb15]
    lows15 = [float_value(bar, "low") for bar in orb15]

    if any(value is None for value in highs15 + lows15):
        return None

    result = {
        "today": today,
        "prior_close": prior_close,
        "open": opening_price,
        "gap": opening_price / prior_close - 1.0,
        "orb15_high": max(highs15),
        "orb15_low": min(lows15),
        "orb30_high": None,
        "orb30_low": None,
    }

    if orb30:
        highs30 = [float_value(bar, "high") for bar in orb30]
        lows30 = [float_value(bar, "low") for bar in orb30]

        if not any(value is None for value in highs30 + lows30):
            result["orb30_high"] = max(highs30)
            result["orb30_low"] = min(lows30)

    return result


# ============================================================================
# CLASSIFICATION AND COST HELPERS
# ============================================================================

def family_for(symbol):
    if symbol in TECH_FAMILY:
        return "TECH"

    if symbol in BROAD_FAMILY:
        return "BROAD"

    if symbol == "IWM":
        return "SMALL_CAP"

    if symbol == "XLF":
        return "FINANCIALS"

    if symbol == "XLE":
        return "ENERGY"

    if symbol == "XBI":
        return "BIOTECH"

    if symbol in DEFENSIVE:
        return "DEFENSIVE"

    return symbol


def benchmark_for(symbol):
    if symbol in {"SMH", "SOXS"}:
        return "SMH"

    if symbol in TECH_FAMILY:
        return "QQQ"

    return "SPY"


def estimated_round_trip_cost(symbol):
    """
    Conservative estimated total friction.

    Replace these estimates with ATHENA paper-fill measurements before live
    deployment; this is only an eligibility filter, not a fill simulator.
    """
    if symbol in LEVERAGED:
        return 0.0018

    if symbol in CORE_ETF:
        return 0.0008

    return 0.0012


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


# ============================================================================
# SURMOUNT STRATEGY
# ============================================================================

class TradingStrategy(Strategy):
    """
    Stateless regime router.

    Position lifecycle and exit-state memory are intentionally not maintained
    here because Surmount runtime persistence is not guaranteed. ATHENA must
    manage actual open positions, stops, targets, and execution.
    """

    def __init__(self):
        self.tickers = build_universe()

    @property
    def interval(self):
        return INTERVAL

    @property
    def assets(self):
        return self.tickers

    def phase(self, data):
        """Classify current regular-session time."""
        spy_bars = bars_for(data, "SPY")

        if not spy_bars:
            return "UNKNOWN"

        timestamp = bar_timestamp(spy_bars[-1])

        if timestamp is None:
            return "UNKNOWN"

        minute_of_day = timestamp.hour * 60 + timestamp.minute

        if minute_of_day < NO_ENTRY_BEFORE:
            return "PRE_OPEN"

        if minute_of_day < MIDDAY_START:
            return "MORNING"

        if minute_of_day < MIDDAY_END:
            return "MIDDAY"

        if minute_of_day < NO_ENTRY_AFTER:
            return "AFTERNOON"

        if minute_of_day < FORCE_FLATTEN_AFTER:
            return "CLOSE_WINDOW"

        return "FLATTEN"

    def market_context(self, data):
        """
        Classify broad market into trend, range, high-volatility, or no-trade.
        """
        symbols = ["SPY", "QQQ", "IWM"]

        market = {
            symbol: bars_for(data, symbol)
            for symbol in symbols
        }

        if any(
            len(market[symbol]) < MIN_MARKET_BARS
            for symbol in symbols
        ):
            return {
                "regime": "NO_TRADE",
                "market": market,
            }

        states = []
        route_efficiencies = []

        for symbol in symbols:
            bars = market[symbol]
            closes = series_for(bars, "close")

            if not closes:
                return {
                    "regime": "NO_TRADE",
                    "market": market,
                }

            current_vwap = session_vwap(bars)
            fast = ema(closes, EMA_MID)
            slow = ema(closes, EMA_SLOW)
            trend = ema(closes, EMA_TREND)
            momentum = roc(closes, 20)
            route_efficiency = efficiency(closes, EFFICIENCY_PERIOD)

            if any(
                value is None
                for value in [
                    current_vwap,
                    fast,
                    slow,
                    trend,
                    momentum,
                    route_efficiency,
                ]
            ):
                return {
                    "regime": "NO_TRADE",
                    "market": market,
                }

            price = closes[-1]

            states.append({
                "up": (
                    price > current_vwap
                    and fast > slow
                    and price > trend
                    and momentum > 0
                ),
                "down": (
                    price < current_vwap
                    and fast < slow
                    and price < trend
                    and momentum < 0
                ),
            })

            route_efficiencies.append(route_efficiency)

        spy_bars = market["SPY"]
        spy_closes = series_for(spy_bars, "close")
        spy_atr = atr(spy_bars)

        if not spy_closes or spy_atr is None or spy_closes[-1] <= 0:
            return {
                "regime": "NO_TRADE",
                "market": market,
            }

        realized_intraday_vol = spy_atr / spy_closes[-1]
        up_count = sum(state["up"] for state in states)
        down_count = sum(state["down"] for state in states)
        average_efficiency = (
            sum(route_efficiencies)
            / float(len(route_efficiencies))
        )

        if realized_intraday_vol >= 0.012:
            regime = "HIGH_VOL"

        elif up_count >= 2:
            regime = "BULL_TREND"

        elif down_count >= 2:
            regime = "BEAR_TREND"

        elif average_efficiency <= 0.22:
            regime = "RANGE"

        else:
            regime = "CHOPPY"

        return {
            "regime": regime,
            "market": market,
            "intraday_volatility": realized_intraday_vol,
            "efficiency": average_efficiency,
        }

    def candidate(self, symbol, data, context, phase):
        """
        Evaluate one symbol and return either:
        - Candidate dictionary; or
        - None, indicating no valid allocation.
        """
        regime = context["regime"]

        if phase in {
            "UNKNOWN",
            "PRE_OPEN",
            "CLOSE_WINDOW",
            "FLATTEN",
        }:
            return None

        if phase == "MIDDAY" and not ENABLE_MIDDAY_ENTRIES:
            return None

        if regime in {"NO_TRADE", "HIGH_VOL"}:
            return None

        if regime == "BULL_TREND" and not ENABLE_BULL_TREND:
            return None

        if regime == "BEAR_TREND" and not ENABLE_INVERSE:
            return None

        if regime in {"RANGE", "CHOPPY"} and not ENABLE_RANGE_REVERSAL:
            return None

        if symbol in STOCKS and not ENABLE_STOCKS:
            return None

        if symbol in LEVERAGED and not ENABLE_LEVERAGED:
            return None

        if symbol in INVERSE and not ENABLE_INVERSE:
            return None

        if regime == "BULL_TREND" and symbol in INVERSE:
            return None

        if regime == "BEAR_TREND" and symbol not in INVERSE:
            return None

        if regime in {"RANGE", "CHOPPY"} and (
            symbol in LEVERAGED
            or symbol in INVERSE
        ):
            return None

        bars = bars_for(data, symbol)

        if len(bars) < MIN_CANDIDATE_BARS:
            return None

        closes = series_for(bars, "close")
        volumes = series_for(bars, "volume")

        if not closes or not volumes:
            return None

        price = closes[-1]

        current_atr = atr(bars)
        current_vwap = session_vwap(bars)
        current_rvol = session_rvol(bars)
        fast = ema(closes, EMA_FAST)
        mid = ema(closes, EMA_MID)
        slow = ema(closes, EMA_SLOW)
        trend = ema(closes, EMA_TREND)
        roc5 = roc(closes, 5)
        roc15 = roc(closes, 15)
        roc20 = roc(closes, 20)
        pressure = candle_pressure_proxy(bars, 5)
        references = reference_levels(bars)
        location = close_location(bars[-1])

        indicators = [
            current_atr,
            current_vwap,
            current_rvol,
            fast,
            mid,
            slow,
            trend,
            roc5,
            roc15,
            roc20,
            pressure,
            references,
            location,
        ]

        if any(value is None for value in indicators):
            return None

        if price <= 0 or price < MIN_PRICE:
            return None

        atr_pct = current_atr / price

        max_atr_pct = (
            MAX_ATR_PCT_LEVERAGED
            if symbol in LEVERAGED
            else MAX_ATR_PCT_NORMAL
        )

        if atr_pct > max_atr_pct:
            return None

        dollar_volumes = [
            close * volume
            for close, volume in zip(closes[-20:], volumes[-20:])
        ]

        average_dollar_volume = sma(dollar_volumes, 20)

        if (
            average_dollar_volume is None
            or average_dollar_volume < MIN_AVG_DOLLAR_VOLUME
        ):
            return None

        benchmark = benchmark_for(symbol)

        benchmark_bars = (
            context["market"].get(benchmark)
            or bars_for(data, benchmark)
        )

        benchmark_closes = series_for(benchmark_bars, "close")

        if len(benchmark_closes) < 21:
            return None

        benchmark_roc5 = roc(benchmark_closes, 5)
        benchmark_roc15 = roc(benchmark_closes, 15)
        benchmark_roc20 = roc(benchmark_closes, 20)

        if None in (
            benchmark_roc5,
            benchmark_roc15,
            benchmark_roc20,
        ):
            return None

        alpha5 = roc5 - benchmark_roc5
        alpha15 = roc15 - benchmark_roc15
        alpha20 = roc20 - benchmark_roc20

        score = 0.0
        setup = None
        stop_atr = 1.0
        target_r = 2.0
        lanes = []

        # --------------------------------------------------------------------
        # BULL TREND
        # --------------------------------------------------------------------

        if regime == "BULL_TREND":
            if symbol in INVERSE:
                return None

            trend_ok = (
                price > current_vwap
                and fast > mid > slow
                and price > trend
                and roc20 > 0
                and roc15 > 0
            )

            if not trend_ok:
                return None

            score += 2.0
            lanes.append("TREND_ALIGNMENT")

            if alpha20 > 0.002:
                score += 0.75
                lanes.append("RELATIVE_STRENGTH")

            previous_close = closes[-2]

            orb15_cross = (
                phase == "MORNING"
                and previous_close <= references["orb15_high"]
                and price > references["orb15_high"]
                and price <= references["orb15_high"] + 0.75 * current_atr
                and current_rvol >= 1.20
                and pressure >= 0.10
            )

            orb30_cross = (
                phase == "MORNING"
                and references["orb30_high"] is not None
                and previous_close <= references["orb30_high"]
                and price > references["orb30_high"]
                and price <= references["orb30_high"] + 0.75 * current_atr
                and current_rvol >= 1.15
                and pressure >= 0.10
            )

            vwap_reclaim = (
                min(
                    float_value(bars[-1], "low"),
                    float_value(bars[-2], "low"),
                ) <= current_vwap + 0.15 * current_atr
                and previous_close <= current_vwap + 0.10 * current_atr
                and price > current_vwap
                and roc5 > 0
                and pressure >= 0.05
                and price - current_vwap <= 1.25 * current_atr
            )

            momentum_continuation = (
                roc5 > 0.001
                and current_rvol >= 1.15
                and pressure >= 0.12
                and alpha15 >= 0.0
                and price - current_vwap <= 1.00 * current_atr
            )

            if orb15_cross:
                score += 3.0
                setup = "ORB15"
                target_r = 2.5
                lanes.append("FRESH_ORB15_CROSS")

            elif orb30_cross:
                score += 2.75
                setup = "ORB30"
                target_r = 2.5
                lanes.append("FRESH_ORB30_CROSS")

            elif vwap_reclaim:
                score += 2.5
                setup = "VWAP_RECLAIM"
                target_r = 2.2
                lanes.append("FRESH_VWAP_RECLAIM")

            elif momentum_continuation:
                score += 2.0
                setup = "MOMENTUM"
                target_r = 2.0
                lanes.append("MOMENTUM_CONTINUATION")

            else:
                return None

            if current_rvol >= 1.30:
                score += 0.50
                lanes.append("STRONG_RVOL")

            if pressure >= 0.15:
                score += 0.50
                lanes.append("PRESSURE_CONFIRMATION")

        # --------------------------------------------------------------------
        # BEAR TREND / INVERSE ETF
        # --------------------------------------------------------------------

        elif regime == "BEAR_TREND":
            if not ENABLE_INVERSE or symbol not in INVERSE:
                return None

            underlying = INVERSE_UNDERLYING.get(symbol)

            if underlying is None:
                return None

            underlying_bars = (
                context["market"].get(underlying)
                or bars_for(data, underlying)
            )

            underlying_closes = series_for(underlying_bars, "close")
            underlying_vwap = session_vwap(underlying_bars)
            underlying_fast = ema(underlying_closes, EMA_FAST)
            underlying_mid = ema(underlying_closes, EMA_MID)
            underlying_roc5 = roc(underlying_closes, 5)
            underlying_pressure = candle_pressure_proxy(
                underlying_bars,
                5,
            )

            if None in (
                underlying_vwap,
                underlying_fast,
                underlying_mid,
                underlying_roc5,
                underlying_pressure,
            ):
                return None

            underlying_bearish = (
                underlying_closes[-1] < underlying_vwap
                and underlying_fast < underlying_mid
                and underlying_roc5 < 0
                and underlying_pressure <= -0.05
            )

            inverse_confirmed = (
                price > current_vwap
                and fast > mid
                and roc5 > 0
                and pressure >= 0.08
                and current_rvol >= 1.10
            )

            if not (underlying_bearish and inverse_confirmed):
                return None

            score = 4.0
            setup = "INVERSE_CONTINUATION"
            stop_atr = 0.85
            target_r = 2.25

            lanes.extend([
                "UNDERLYING_BEAR_CONFIRMATION",
                "INVERSE_RECLAIM",
            ])

            if current_rvol >= 1.25:
                score += 0.50
                lanes.append("STRONG_RVOL")

            if pressure >= 0.15:
                score += 0.50
                lanes.append("PRESSURE_CONFIRMATION")

        # --------------------------------------------------------------------
        # RANGE / CHOPPY REVERSAL
        # --------------------------------------------------------------------

        elif regime in {"RANGE", "CHOPPY"}:
            if not ENABLE_RANGE_REVERSAL:
                return None

            if symbol in LEVERAGED or symbol in INVERSE:
                return None

            bands = bollinger(closes, BOLLINGER_PERIOD, 2.0)
            current_rsi = rsi(closes, RSI_PERIOD)
            route_efficiency = efficiency(closes, EFFICIENCY_PERIOD)

            if None in (
                bands,
                current_rsi,
                route_efficiency,
            ):
                return None

            lower_band, _, _ = bands

            previous_low = float_value(bars[-2], "low")
            previous_close = float_value(bars[-2], "close")
            latest_open = float_value(bars[-1], "open")
            latest_low = float_value(bars[-1], "low")

            if None in (
                previous_low,
                previous_close,
                latest_open,
                latest_low,
            ):
                return None

            vwap_extension = (
                (price - current_vwap)
                / max(current_atr, 0.01)
            )

            lower_touch = (
                latest_low <= lower_band + 0.20 * current_atr
                or previous_low <= lower_band + 0.20 * current_atr
                or vwap_extension <= -1.0
            )

            reversal = (
                price > latest_open
                and price > previous_close
                and location >= 0.65
                and pressure >= 0.00
            )

            spy_bars = context["market"].get("SPY", [])
            spy_closes = series_for(spy_bars, "close")
            spy_vwap = session_vwap(spy_bars)

            broad_market_safe = (
                spy_closes
                and spy_vwap is not None
                and spy_closes[-1] >= spy_vwap - 0.50 * current_atr
            )

            if not (
                lower_touch
                and reversal
                and route_efficiency <= 0.32
                and current_rsi <= 42
                and vwap_extension >= -2.25
                and broad_market_safe
            ):
                return None

            score = 3.5
            setup = "RANGE_REVERSAL"
            stop_atr = 0.80
            target_r = 1.25

            lanes.extend([
                "RANGE_EDGE_TOUCH",
                "REVERSAL_CANDLE",
            ])

            if current_rvol >= 0.90:
                score += 0.50
                lanes.append("ACCEPTABLE_VOLUME")

            if location >= 0.75 and pressure >= 0.08:
                score += 0.50
                lanes.append("RECLAIM_STRENGTH")

        else:
            return None

        # --------------------------------------------------------------------
        # FINAL CANDIDATE CHECKS
        # --------------------------------------------------------------------

        if phase == "MIDDAY":
            score -= 0.75

        expected_move = target_r * stop_atr * atr_pct
        required_move = (
            MIN_EXPECTED_MOVE_COST_MULTIPLE
            * estimated_round_trip_cost(symbol)
        )

        if expected_move < required_move:
            return None

        required_score = (
            5.0
            if regime == "BULL_TREND"
            else 4.5
        )

        if score < required_score:
            return None

        return {
            "symbol": symbol,
            "score": score,
            "setup": setup,
            "lanes": lanes,
            "family": family_for(symbol),
            "leveraged": symbol in LEVERAGED,
            "atr_pct": atr_pct,
            "stop_atr": stop_atr,
            "target_r": target_r,
            "momentum_raw": (
                0.20 * roc5
                + 0.35 * roc15
                + 0.45 * roc20
            ),
            "alpha_raw": (
                0.50 * alpha15
                + 0.50 * alpha20
            ),
            "volume_raw": clamp(current_rvol, 0.0, 3.0),
        }

    def allocation_for(self, candidate, regime):
        """
        Convert setup risk to Surmount target allocation.

        This is an allocation proxy only. ATHENA must calculate actual
        quantity using portfolio equity, broker fill, actual stop, and total
        correlated exposure.
        """
        risk_distance = max(
            candidate["stop_atr"] * candidate["atr_pct"],
            0.003,
        )

        score_scale = clamp(
            candidate["score"] / 7.0,
            0.55,
            1.0,
        )

        allocation = RISK_PER_TRADE / risk_distance
        allocation *= score_scale

        if candidate["leveraged"]:
            maximum = MAX_LEVERAGED_ALLOCATION

        elif candidate["setup"] == "RANGE_REVERSAL":
            maximum = MAX_RANGE_ALLOCATION

        else:
            maximum = MAX_NORMAL_ALLOCATION

        if regime in {"RANGE", "CHOPPY"}:
            allocation *= 0.65

        if candidate["setup"] == "INVERSE_CONTINUATION":
            allocation *= 0.75

        return clamp(allocation, 0.0, maximum)

    def run(self, data):
        """
        Surmount-required strategy entry point.

        Returns TargetAllocation only. Empty allocation is intentional:
        cash is the correct output when no validated opportunity exists.
        """
        try:
            phase = self.phase(data)

            # Do not create new exposures during forbidden windows.
            if phase in {
                "UNKNOWN",
                "PRE_OPEN",
                "CLOSE_WINDOW",
                "FLATTEN",
            }:
                return TargetAllocation({})

            if phase == "MIDDAY" and not ENABLE_MIDDAY_ENTRIES:
                return TargetAllocation({})

            context = self.market_context(data)
            regime = context["regime"]

            if regime == "NO_TRADE":
                return TargetAllocation({})

            if regime == "HIGH_VOL":
                if ENABLE_BIL_DEFENSIVE:
                    bil_bars = bars_for(data, "BIL")

                    if len(bil_bars) >= MIN_MARKET_BARS:
                        return TargetAllocation({
                            "BIL": BIL_ALLOCATION,
                        })

                return TargetAllocation({})

            candidates = []

            for symbol in self.tickers:
                candidate = self.candidate(
                    symbol,
                    data,
                    context,
                    phase,
                )

                if candidate is not None:
                    candidates.append(candidate)

            if not candidates:
                return TargetAllocation({})

            if ENABLE_CROSS_SECTIONAL_RANKING and len(candidates) > 1:
                for field in (
                    "momentum_raw",
                    "volume_raw",
                    "alpha_raw",
                ):
                    ranked = sorted(
                        candidates,
                        key=lambda item: item[field],
                    )

                    denominator = max(len(ranked) - 1, 1)

                    for index, item in enumerate(ranked):
                        item[field + "_rank"] = (
                            index / float(denominator)
                        )

                for item in candidates:
                    item["rank_bonus"] = (
                        0.40 * item["momentum_raw_rank"]
                        + 0.30 * item["volume_raw_rank"]
                        + 0.30 * item["alpha_raw_rank"]
                    )

            else:
                for item in candidates:
                    item["rank_bonus"] = 0.0

            candidates.sort(
                key=lambda item: (
                    item["score"] + item["rank_bonus"],
                    item["score"],
                ),
                reverse=True,
            )

            selected = []
            used_families = set()

            for candidate in candidates:
                if candidate["family"] in used_families:
                    continue

                selected.append(candidate)
                used_families.add(candidate["family"])

                if len(selected) >= MAX_POSITIONS:
                    break

            allocations = {
                candidate["symbol"]: self.allocation_for(
                    candidate,
                    regime,
                )
                for candidate in selected
            }

            total_allocation = sum(allocations.values())

            if total_allocation > MAX_TOTAL_ALLOCATION:
                scale = MAX_TOTAL_ALLOCATION / total_allocation

                allocations = {
                    symbol: allocation * scale
                    for symbol, allocation in allocations.items()
                }

            return TargetAllocation({
                symbol: allocation
                for symbol, allocation in allocations.items()
                if allocation > 0
            })

        except Exception:
            return TargetAllocation({})