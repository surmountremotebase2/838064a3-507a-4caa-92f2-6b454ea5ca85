"""
ATHENA / PMC-INSPIRED SURMOUNT INTRADAY MASTER STRATEGY
========================================================

Single-file Surmount Code Builder strategy.

Initial enabled configuration
-----------------------------
- Bull trend: enabled.
- Momentum leader selection: enabled.
- ETF rotation: enabled.
- Stock rotation: enabled.
- Cross-sectional ranking: enabled.
- Range reversal: disabled.
- Inverse ETFs: disabled.
- Leveraged ETFs: disabled.
- Midday entries: disabled.
- Maximum one active risk position.
- Maximum 15% allocation to the active risk position.
- Remaining capital explicitly targets BIL.

This strategy is allocation-only. ATHENA must enforce:
- Broker reconciliation.
- True PMC/order-flow checks.
- Spread/slippage gates.
- Stop, target, partial-profit, trailing, and time exits.
- Max daily loss, drawdown, correlation, and kill-switch controls.
- Forced end-of-day flatten.

No strategy guarantees profit or capital preservation. This is intended for
cost-loaded backtesting and paper validation before any live deployment.
"""

from datetime import datetime
from math import sqrt

from surmount.base_class import Strategy, TargetAllocation


# ============================================================================
# INITIAL SETTINGS — ACTIVE FOR FIRST BACKTEST
# ============================================================================

ENABLE_BULL_TREND = True
ENABLE_RANGE_REVERSAL = False
ENABLE_INVERSE = False
ENABLE_STOCKS = True
ENABLE_LEVERAGED = False
ENABLE_MIDDAY_ENTRIES = False
ENABLE_CROSS_SECTIONAL_RANKING = True

ENABLE_MOMENTUM_LEADER_SELECTION = True
ENABLE_ETF_ROTATION = True
ENABLE_STOCK_ROTATION = True


# ============================================================================
# CAPITAL / RISK SETTINGS
# ============================================================================

INTERVAL = "5min"

CASH_SYMBOL = "BIL"
USE_BIL_CASH_SLEEVE = True

MAX_POSITIONS = 1
MAX_TOTAL_RISK_ALLOCATION = 0.15
MAX_NORMAL_ALLOCATION = 0.15
MAX_LEVERAGED_ALLOCATION = 0.06
MAX_RANGE_ALLOCATION = 0.10

RISK_PER_TRADE = 0.0015
MIN_EXPECTED_MOVE_COST_MULTIPLE = 3.0

HIGH_VOL_SPY_ATR_PCT = 0.012


# ============================================================================
# DATA / INDICATOR SETTINGS
# ============================================================================

MIN_MARKET_BARS = 250
MIN_CANDIDATE_BARS = 250

MIN_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 10_000_000.0
MIN_CURRENT_DOLLAR_VOLUME = 2_500_000.0

MAX_ATR_PCT_NORMAL = 0.045
MAX_ATR_PCT_LEVERAGED = 0.075

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 100

ATR_PERIOD = 14
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
EFFICIENCY_PERIOD = 20
RVOL_LOOKBACK_SESSIONS = 10

MIN_RVOL_FOR_MOMENTUM = 1.20
MIN_RVOL_FOR_BREAKOUT = 1.35
MIN_RVOL_FOR_RANGE = 0.90

MIN_ALPHA_5 = 0.0000
MIN_ALPHA_15 = 0.0010
MIN_ALPHA_20 = 0.0020

MAX_VWAP_EXTENSION_ATR = 1.25

MIN_LEADER_SCORE = 6.00
MAX_ROTATION_CANDIDATES = 5

MIN_INSTITUTIONAL_RVOL = 1.25
MIN_INSTITUTIONAL_PRESSURE = 0.12
MIN_INSTITUTIONAL_CLOSE_LOCATION = 0.70

# Assumes Surmount timestamps are US/Eastern. Verify before production use.
NO_ENTRY_BEFORE = 9 * 60 + 50
MIDDAY_START = 11 * 60 + 30
MIDDAY_END = 13 * 60 + 30
NO_ENTRY_AFTER = 15 * 60 + 15
FORCE_FLATTEN_AFTER = 15 * 60 + 45


# ============================================================================
# UNIVERSE
# ============================================================================

CORE_ETF = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "SMH", "XBI",
]

LONG_LEVERAGED = [
    "TQQQ", "SOXL", "UPRO", "TECL",
]

INVERSE_LEVERAGED = [
    "SQQQ", "SOXS", "SPXU", "PSQ",
]

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN",
    "META", "TSLA", "AVGO", "MU", "JPM",
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
    "QQQ", "XLK", "SMH",
    "TQQQ", "SOXL", "TECL",
    "SQQQ", "PSQ", "SOXS",
    "NVDA", "AMD", "AVGO", "MU",
}

BROAD_FAMILY = {
    "SPY", "DIA", "UPRO", "SPXU",
    "AAPL", "MSFT", "AMZN", "META", "JPM",
}


def build_universe():
    """Build active universe. BIL is always included as the cash sleeve."""
    symbols = [
        "SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", CASH_SYMBOL,
    ]

    if ENABLE_ETF_ROTATION:
        symbols.extend(["XLF", "XLE", "XBI"])

    if ENABLE_STOCKS and ENABLE_STOCK_ROTATION:
        symbols.extend(STOCKS)

    if ENABLE_LEVERAGED:
        symbols.extend(LONG_LEVERAGED)

    if ENABLE_INVERSE:
        symbols.extend(INVERSE_LEVERAGED)

    return list(dict.fromkeys(symbols))


# ============================================================================
# SAFE DATA HELPERS
# ============================================================================

def bars_for(data, symbol):
    """Return valid OHLCV bars for one symbol."""
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
    """Return finite numerical OHLCV field or None."""
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
    """Return numeric series only if every required field is valid."""
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
    if length <= 0 or len(bars) < length + 1:
        return None

    ranges = []

    for previous, current in zip(bars[-length - 1:-1], bars[-length:]):
        high = float_value(current, "high")
        low = float_value(current, "low")
        prior_close = float_value(previous, "close")

        if None in (high, low, prior_close):
            return None

        if high < low or low <= 0 or prior_close <= 0:
            return None

        ranges.append(
            max(
                high - low,
                abs(high - prior_close),
                abs(low - prior_close),
            )
        )

    if len(ranges) != length:
        return None

    return sum(ranges) / float(length)


def rsi(values, length=RSI_PERIOD):
    if length <= 0 or len(values) < length + 1:
        return None

    changes = [
        current - previous
        for previous, current in zip(values[-length - 1:-1], values[-length:])
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
    """Directional movement divided by total movement path."""
    if length <= 0 or len(values) < length + 1:
        return None

    window = values[-length - 1:]
    displacement = abs(window[-1] - window[0])

    path = sum(
        abs(current - previous)
        for previous, current in zip(window[:-1], window[1:])
    )

    return displacement / path if path > 0 else 0.0


def candle_pressure_proxy(bars, length=5):
    """
    OHLCV candle-volume pressure proxy.

    This is not true order flow, bid/ask delta, CVD, book imbalance, or PMC.
    ATHENA must validate real microstructure conditions independently.
    """
    if length <= 0 or len(bars) < length:
        return None

    weighted = 0.0
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

        width = max(high - low, 0.01)

        weighted += ((close - open_price) / width) * volume
        total_volume += volume

    return weighted / total_volume if total_volume > 0 else None


def close_location(bar):
    """Normalized close location in the current candle."""
    high = float_value(bar, "high")
    low = float_value(bar, "low")
    close = float_value(bar, "close")

    if None in (high, low, close):
        return None

    return (close - low) / max(high - low, 0.01)


# ============================================================================
# TIME / SESSION / VWAP / RVOL
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
    Group bars by their calendar date.

    Confirm Surmount's timestamp timezone before trusting exact session gates.
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
    groups = session_groups(bars)

    if not groups:
        return [], []

    today = groups[-1][1]
    prior = []

    for _, group in groups[:-1]:
        prior.extend(group)

    return today, prior


def session_vwap(bars):
    """Current-session VWAP using typical price."""
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
    Relative volume versus the same intraday slot in prior sessions.

    Falls back to a trailing 20-bar baseline for short historical windows.
    """
    groups = session_groups(bars)

    if len(groups) >= 4:
        today = groups[-1][1]
        slot = len(today) - 1

        current_volume = float_value(today[-1], "volume")

        if current_volume is not None:
            prior_volumes = []

            for _, group in groups[-lookback_sessions - 1:-1]:
                if len(group) > slot:
                    volume = float_value(group[slot], "volume")

                    if volume is not None and volume >= 0:
                        prior_volumes.append(volume)

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
    """Prior close, opening price, and opening-range levels."""
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
# LEADER / ROTATION / PARTICIPATION HELPERS
# ============================================================================

def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def percentile_rank(value, values):
    """Cross-sectional percentile rank from 0.0 to 1.0."""
    clean = [item for item in values if item is not None]

    if len(clean) <= 1:
        return 0.5

    count = sum(item <= value for item in clean)

    return (count - 1) / float(len(clean) - 1)


def institutional_activity_score(
    rvol,
    pressure,
    close_position,
    current_dollar_volume,
    average_dollar_volume,
):
    """
    Observable high-participation proxy.

    This does not prove institutional flow. It measures unusual dollar volume,
    directional candle pressure, strong close location, and volume expansion.
    """
    if any(
        value is None
        for value in [
            rvol,
            pressure,
            close_position,
            current_dollar_volume,
            average_dollar_volume,
        ]
    ):
        return 0.0

    if current_dollar_volume < MIN_CURRENT_DOLLAR_VOLUME:
        return 0.0

    if average_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
        return 0.0

    score = 0.0

    if rvol >= MIN_INSTITUTIONAL_RVOL:
        score += 1.0

    if pressure >= MIN_INSTITUTIONAL_PRESSURE:
        score += 1.0

    if close_position >= MIN_INSTITUTIONAL_CLOSE_LOCATION:
        score += 1.0

    if current_dollar_volume >= average_dollar_volume * 1.10:
        score += 1.0

    return score


def rotation_score(
    momentum_5,
    momentum_15,
    momentum_20,
    alpha_5,
    alpha_15,
    alpha_20,
    rvol,
    pressure,
    close_position,
):
    """Score directional momentum, relative alpha, volume, and confirmation."""
    values = [
        momentum_5,
        momentum_15,
        momentum_20,
        alpha_5,
        alpha_15,
        alpha_20,
        rvol,
        pressure,
        close_position,
    ]

    if any(value is None for value in values):
        return 0.0

    score = 0.0

    if momentum_5 > 0:
        score += 0.75

    if momentum_15 > 0:
        score += 1.00

    if momentum_20 > 0:
        score += 1.00

    if alpha_5 >= MIN_ALPHA_5:
        score += 0.50

    if alpha_15 >= MIN_ALPHA_15:
        score += 1.00

    if alpha_20 >= MIN_ALPHA_20:
        score += 1.25

    if rvol >= MIN_RVOL_FOR_MOMENTUM:
        score += 1.00

    if rvol >= MIN_RVOL_FOR_BREAKOUT:
        score += 0.50

    if pressure >= MIN_INSTITUTIONAL_PRESSURE:
        score += 0.75

    if close_position >= MIN_INSTITUTIONAL_CLOSE_LOCATION:
        score += 0.50

    return score


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

    if symbol == CASH_SYMBOL:
        return "CASH"

    return symbol


def benchmark_for(symbol):
    if symbol in {"SMH", "SOXS"}:
        return "SMH"

    if symbol in TECH_FAMILY:
        return "QQQ"

    return "SPY"


def estimated_round_trip_cost(symbol):
    """
    Conservative friction floor.

    Replace these values with cost measurements from ATHENA paper fills before
    considering live use.
    """
    if symbol in LEVERAGED:
        return 0.0018

    if symbol in CORE_ETF:
        return 0.0008

    return 0.0012


def add_cash_sleeve(active_allocations):
    """
    Explicitly allocate all residual capital to BIL.

    {} -> {"BIL": 1.0}
    {"SPY": 0.15} -> {"SPY": 0.15, "BIL": 0.85}
    """
    active = {}

    for symbol, allocation in active_allocations.items():
        try:
            weight = float(allocation)
        except (TypeError, ValueError):
            continue

        if weight > 0 and symbol != CASH_SYMBOL:
            active[str(symbol).upper()] = weight

    total = sum(active.values())

    if total > 1.0:
        scale = 1.0 / total

        active = {
            symbol: weight * scale
            for symbol, weight in active.items()
        }

        total = 1.0

    if USE_BIL_CASH_SLEEVE:
        active[CASH_SYMBOL] = max(0.0, 1.0 - total)

    return {
        symbol: weight
        for symbol, weight in active.items()
        if weight > 0
    }


# ============================================================================
# SURMOUNT STRATEGY
# ============================================================================

class TradingStrategy(Strategy):
    """
    Intraday regime router and momentum-leader selector.

    Surmount supplies target allocations. ATHENA must independently decide
    whether targets pass execution, risk, PMC, and broker-state requirements.
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
        """
        Return a session phase from the latest SPY timestamp.

        Assumption: timestamp is US/Eastern. Validate before live/paper use.
        """
        spy_bars = bars_for(data, "SPY")

        if not spy_bars:
            return "UNKNOWN"

        timestamp = bar_timestamp(spy_bars[-1])

        if timestamp is None:
            return "UNKNOWN"

        minutes = timestamp.hour * 60 + timestamp.minute

        if minutes < NO_ENTRY_BEFORE:
            return "PRE_OPEN"

        if minutes < MIDDAY_START:
            return "MORNING"

        if minutes < MIDDAY_END:
            return "MIDDAY"

        if minutes < NO_ENTRY_AFTER:
            return "AFTERNOON"

        if minutes < FORCE_FLATTEN_AFTER:
            return "CLOSE_WINDOW"

        return "FLATTEN"

    def market_context(self, data):
        """Classify broad market state from SPY, QQQ, and IWM."""
        benchmark_symbols = ["SPY", "QQQ", "IWM"]

        market = {
            symbol: bars_for(data, symbol)
            for symbol in benchmark_symbols
        }

        if any(
            len(market[symbol]) < MIN_MARKET_BARS
            for symbol in benchmark_symbols
        ):
            return {
                "regime": "NO_TRADE",
                "market": market,
            }

        states = []
        efficiency_values = []

        for symbol in benchmark_symbols:
            bars = market[symbol]
            closes = series_for(bars, "close")

            if not closes:
                return {
                    "regime": "NO_TRADE",
                    "market": market,
                }

            current_vwap = session_vwap(bars)
            mid = ema(closes, EMA_MID)
            slow = ema(closes, EMA_SLOW)
            trend = ema(closes, EMA_TREND)
            momentum = roc(closes, 20)
            route_efficiency = efficiency(
                closes,
                EFFICIENCY_PERIOD,
            )

            if any(
                value is None
                for value in [
                    current_vwap,
                    mid,
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
                    and mid > slow
                    and price > trend
                    and momentum > 0
                ),
                "down": (
                    price < current_vwap
                    and mid < slow
                    and price < trend
                    and momentum < 0
                ),
            })

            efficiency_values.append(route_efficiency)

        spy_closes = series_for(market["SPY"], "close")
        spy_atr = atr(market["SPY"])

        if not spy_closes or spy_atr is None or spy_closes[-1] <= 0:
            return {
                "regime": "NO_TRADE",
                "market": market,
            }

        intraday_volatility = spy_atr / spy_closes[-1]

        up_count = sum(state["up"] for state in states)
        down_count = sum(state["down"] for state in states)

        average_efficiency = (
            sum(efficiency_values)
            / float(len(efficiency_values))
        )

        if intraday_volatility >= HIGH_VOL_SPY_ATR_PCT:
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
            "intraday_volatility": intraday_volatility,
            "efficiency": average_efficiency,
        }

    def candidate(self, symbol, data, context, phase):
        """Return a fully qualified trade candidate, or None."""
        regime = context["regime"]

        if symbol == CASH_SYMBOL:
            return None

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

        if symbol in STOCKS and (
            not ENABLE_STOCKS
            or not ENABLE_STOCK_ROTATION
        ):
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

        if any(
            value is None
            for value in [
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
        ):
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

        recent_dollar_volume = [
            close * volume
            for close, volume in zip(closes[-20:], volumes[-20:])
        ]

        average_dollar_volume = sma(recent_dollar_volume, 20)
        current_dollar_volume = price * volumes[-1]

        if (
            average_dollar_volume is None
            or average_dollar_volume < MIN_AVG_DOLLAR_VOLUME
            or current_dollar_volume < MIN_CURRENT_DOLLAR_VOLUME
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

        leader_strength = rotation_score(
            momentum_5=roc5,
            momentum_15=roc15,
            momentum_20=roc20,
            alpha_5=alpha5,
            alpha_15=alpha15,
            alpha_20=alpha20,
            rvol=current_rvol,
            pressure=pressure,
            close_position=location,
        )

        participation_strength = institutional_activity_score(
            rvol=current_rvol,
            pressure=pressure,
            close_position=location,
            current_dollar_volume=current_dollar_volume,
            average_dollar_volume=average_dollar_volume,
        )

        if regime in {"BULL_TREND", "BEAR_TREND"}:
            if (
                ENABLE_MOMENTUM_LEADER_SELECTION
                and leader_strength < MIN_LEADER_SCORE
            ):
                return None

            if participation_strength < 2.0:
                return None

        score = 0.0
        setup = None
        stop_atr = 1.0
        target_r = 2.0
        lanes = []

        # --------------------------------------------------------------------
        # BULL TREND ENTRY LOGIC
        # --------------------------------------------------------------------

        if regime == "BULL_TREND":
            trend_alignment = (
                price > current_vwap
                and fast > mid > slow
                and price > trend
                and roc20 > 0
                and roc15 > 0
            )

            if not trend_alignment:
                return None

            score += 2.0
            lanes.append("TREND_ALIGNMENT")

            if alpha20 > MIN_ALPHA_20:
                score += 0.75
                lanes.append("RELATIVE_STRENGTH")

            previous_close = closes[-2]
            latest_low = float_value(bars[-1], "low")
            prior_low = float_value(bars[-2], "low")

            if latest_low is None or prior_low is None:
                return None

            orb15_cross = (
                phase == "MORNING"
                and previous_close <= references["orb15_high"]
                and price > references["orb15_high"]
                and price <= references["orb15_high"] + 0.75 * current_atr
                and current_rvol >= MIN_RVOL_FOR_BREAKOUT
                and pressure >= MIN_INSTITUTIONAL_PRESSURE
            )

            orb30_cross = (
                phase == "MORNING"
                and references["orb30_high"] is not None
                and previous_close <= references["orb30_high"]
                and price > references["orb30_high"]
                and price <= references["orb30_high"] + 0.75 * current_atr
                and current_rvol >= MIN_RVOL_FOR_MOMENTUM
                and pressure >= MIN_INSTITUTIONAL_PRESSURE
            )

            vwap_reclaim = (
                min(latest_low, prior_low)
                <= current_vwap + 0.15 * current_atr
                and previous_close <= current_vwap + 0.10 * current_atr
                and price > current_vwap
                and roc5 > 0
                and pressure >= 0.05
                and price - current_vwap
                <= MAX_VWAP_EXTENSION_ATR * current_atr
            )

            momentum_continuation = (
                roc5 > 0.001
                and current_rvol >= MIN_RVOL_FOR_MOMENTUM
                and pressure >= MIN_INSTITUTIONAL_PRESSURE
                and alpha15 >= MIN_ALPHA_15
                and price - current_vwap <= 1.00 * current_atr
            )

            if orb15_cross:
                score += 3.0
                setup = "ORB15_BREAKOUT"
                target_r = 2.5
                lanes.append("FRESH_ORB15_CROSS")

            elif orb30_cross:
                score += 2.75
                setup = "ORB30_BREAKOUT"
                target_r = 2.5
                lanes.append("FRESH_ORB30_CROSS")

            elif vwap_reclaim:
                score += 2.5
                setup = "VWAP_RECLAIM"
                target_r = 2.2
                lanes.append("FRESH_VWAP_RECLAIM")

            elif momentum_continuation:
                score += 2.0
                setup = "MOMENTUM_CONTINUATION"
                target_r = 2.0
                lanes.append("MOMENTUM_CONTINUATION")

            else:
                return None

            if current_rvol >= MIN_RVOL_FOR_BREAKOUT:
                score += 0.50
                lanes.append("STRONG_RVOL")

            if pressure >= 0.15:
                score += 0.50
                lanes.append("PRESSURE_CONFIRMATION")

            score += min(1.0, leader_strength / 10.0)

        # --------------------------------------------------------------------
        # INVERSE ETF ENTRY LOGIC
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
                and current_rvol >= MIN_RVOL_FOR_MOMENTUM
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

            if current_rvol >= MIN_RVOL_FOR_BREAKOUT:
                score += 0.50
                lanes.append("STRONG_RVOL")

            if pressure >= 0.15:
                score += 0.50
                lanes.append("PRESSURE_CONFIRMATION")

            score += min(1.0, leader_strength / 10.0)

        # --------------------------------------------------------------------
        # RANGE REVERSAL ENTRY LOGIC
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

            reversal_candle = (
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
                and spy_closes[-1]
                >= spy_vwap - 0.50 * current_atr
            )

            if not (
                lower_touch
                and reversal_candle
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

            if current_rvol >= MIN_RVOL_FOR_RANGE:
                score += 0.50
                lanes.append("ACCEPTABLE_VOLUME")

            if location >= 0.75 and pressure >= 0.08:
                score += 0.50
                lanes.append("RECLAIM_STRENGTH")

        else:
            return None

        # --------------------------------------------------------------------
        # FINAL RISK / COST / SCORE GATES
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

        minimum_score = (
            5.0
            if regime == "BULL_TREND"
            else 4.5
        )

        if score < minimum_score:
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
            "leader_strength": leader_strength,
            "participation_strength": participation_strength,
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
            "current_dollar_volume": current_dollar_volume,
        }

    def allocation_for(self, candidate, regime):
        """
        Convert qualified candidate into capped portfolio allocation.

        This is not a broker-safe quantity. ATHENA must separately validate
        account equity, correlation, stop distance, real-time volatility,
        execution quality, and position state.
        """
        risk_distance = max(
            candidate["stop_atr"] * candidate["atr_pct"],
            0.003,
        )

        score_scale = clamp(
            candidate["score"] / 8.0,
            0.55,
            1.0,
        )

        weight = RISK_PER_TRADE / risk_distance
        weight *= score_scale

        if candidate["leveraged"]:
            cap = MAX_LEVERAGED_ALLOCATION

        elif candidate["setup"] == "RANGE_REVERSAL":
            cap = MAX_RANGE_ALLOCATION

        else:
            cap = MAX_NORMAL_ALLOCATION

        if regime in {"RANGE", "CHOPPY"}:
            weight *= 0.65

        if candidate["setup"] == "INVERSE_CONTINUATION":
            weight *= 0.75

        return clamp(weight, 0.0, cap)

    def run(self, data):
        """
        Surmount-required execution method.

        Explicit allocation outcomes:
        - No valid setup: 100% BIL.
        - Valid candidate: risk allocation plus remaining BIL.
        """
        try:
            phase = self.phase(data)

            if phase in {
                "UNKNOWN",
                "PRE_OPEN",
                "CLOSE_WINDOW",
                "FLATTEN",
            }:
                return TargetAllocation(add_cash_sleeve({}))

            if phase == "MIDDAY" and not ENABLE_MIDDAY_ENTRIES:
                return TargetAllocation(add_cash_sleeve({}))

            context = self.market_context(data)
            regime = context["regime"]

            if regime in {"NO_TRADE", "HIGH_VOL"}:
                return TargetAllocation(add_cash_sleeve({}))

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
                return TargetAllocation(add_cash_sleeve({}))

            if ENABLE_CROSS_SECTIONAL_RANKING and len(candidates) > 1:
                fields = [
                    "momentum_raw",
                    "volume_raw",
                    "alpha_raw",
                    "leader_strength",
                    "participation_strength",
                ]

                for field in fields:
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
                        0.20 * item["momentum_raw_rank"]
                        + 0.20 * item["volume_raw_rank"]
                        + 0.20 * item["alpha_raw_rank"]
                        + 0.25 * item["leader_strength_rank"]
                        + 0.15 * item["participation_strength_rank"]
                    )

            else:
                for item in candidates:
                    item["rank_bonus"] = 0.0

            candidates.sort(
                key=lambda item: (
                    item["score"] + item["rank_bonus"],
                    item["leader_strength"],
                    item["participation_strength"],
                ),
                reverse=True,
            )

            candidates = candidates[:MAX_ROTATION_CANDIDATES]

            selected = []
            used_families = set()

            for candidate in candidates:
                if candidate["family"] in used_families:
                    continue

                selected.append(candidate)
                used_families.add(candidate["family"])

                if len(selected) >= MAX_POSITIONS:
                    break

            active_allocations = {
                candidate["symbol"]: self.allocation_for(
                    candidate,
                    regime,
                )
                for candidate in selected
            }

            active_total = sum(active_allocations.values())

            if active_total > MAX_TOTAL_RISK_ALLOCATION and active_total > 0:
                scale = MAX_TOTAL_RISK_ALLOCATION / active_total

                active_allocations = {
                    symbol: allocation * scale
                    for symbol, allocation in active_allocations.items()
                }

            return TargetAllocation(
                add_cash_sleeve(active_allocations)
            )

        except Exception:
            return TargetAllocation(add_cash_sleeve({}))