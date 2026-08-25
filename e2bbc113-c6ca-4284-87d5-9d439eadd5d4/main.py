"""
SURMOUNT EQUITY & ETF INTRADAY STRATEGY — COMPREHENSIVE V3
===========================================================

Purpose
-------
A single-file, allocation-only, multi-regime 5-minute US equity/ETF strategy.

Playbooks
---------
1. BULL_TREND:
   - Long trend continuation / VWAP reclaim / short breakout.
2. BEAR_TREND:
   - Long explicitly mapped inverse ETFs when their underlying confirms weakness.
3. RANGE:
   - Conservative long mean-reversion after downside extension and reclaim.
4. HIGH_VOL:
   - Defensive BIL allocation only.
5. NO_TRADE:
   - Flat during insufficient data, unsupported phase, or unresolved errors.

Critical ATHENA Responsibilities
-------------------------------
This file only returns target allocations. ATHENA must independently enforce:
- Order-book / PMC checks and execution quality.
- Spread, marketability, slippage, and fill validation.
- Initial stop, take-profit, partial exit, trailing stop, time stop.
- Maximum loss, daily loss, exposure, concentration, and kill-switch rules.
- Flattening before the end of the regular session.
- Broker reconciliation and paper/live environment verification.

Research Notice
---------------
This is a testable baseline, not a claim of profitability. Use chronological
walk-forward testing, an untouched holdout, and cost/slippage stress testing
before any paper or live deployment.

Surmount Interface
------------------
- Must define TradingStrategy(Strategy).
- Must return TargetAllocation({symbol: decimal_weight}).
- Allocation values are decimal portfolio weights; 0.10 means 10%.
"""

from datetime import datetime
from math import sqrt

from surmount.base_class import Strategy, TargetAllocation

try:
    from surmount.logging import log
except Exception:
    def log(message):
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

INTERVAL = "5min"

MIN_BARS = 120
MIN_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 20_000_000.0
MAX_ATR_PCT = 0.035

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
ATR_PERIOD = 14
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
EFFICIENCY_PERIOD = 20
RVOL_PERIOD = 20

MAX_POSITIONS = 2
MAX_TOTAL_ALLOCATION = 0.40
MAX_UNLEVERAGED_ALLOCATION = 0.18
MAX_LEVERAGED_ALLOCATION = 0.08
DEFENSIVE_BIL_ALLOCATION = 0.20

MIN_SCORE = 4.00

# Phase times assume the data provider emits timestamps in US/Eastern.
# If timestamps are UTC, phase logic must be converted before backtesting.
OPEN_TRADE_START_MINUTE = 9 * 60 + 50
MORNING_END_MINUTE = 11 * 60 + 30
MIDDAY_END_MINUTE = 14 * 60
AFTERNOON_END_MINUTE = 15 * 60 + 30
CLOSE_WINDOW_END_MINUTE = 15 * 60 + 50


# ============================================================================
# UNIVERSE
# ============================================================================

CORE_ETF = [
    "SPY", "QQQ", "IWM", "DIA", "XLK",
    "XLF", "XLE", "SMH", "XBI",
]

DEFENSIVE = [
    "BIL", "SHY", "TLT", "GLD",
]

LONG_LEVERAGED = [
    "TQQQ", "SOXL", "UPRO", "LABU", "TECL",
]

INVERSE_LEVERAGED = [
    "SQQQ", "SOXS", "SPXU", "LABD", "FAZ", "PSQ",
]

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN",
    "META", "TSLA", "GOOGL", "AVGO", "MU",
    "PLTR", "JPM", "XOM",
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

# Every inverse ETF must reference the economically relevant underlying.
INVERSE_UNDERLYING = {
    "SQQQ": "QQQ",
    "PSQ": "QQQ",
    "SPXU": "SPY",
    "SOXS": "SMH",
    "LABD": "XBI",
    "FAZ": "XLF",
}

SYMBOL_FAMILY = {
    "SPY": "BROAD_MARKET",
    "DIA": "BROAD_MARKET",
    "UPRO": "BROAD_MARKET",
    "SPXU": "BROAD_MARKET",
    "IWM": "SMALL_CAP",

    "QQQ": "TECH",
    "XLK": "TECH",
    "SMH": "TECH",
    "TQQQ": "TECH",
    "SOXL": "TECH",
    "TECL": "TECH",
    "SQQQ": "TECH",
    "PSQ": "TECH",
    "SOXS": "TECH",

    "XBI": "BIOTECH",
    "LABU": "BIOTECH",
    "LABD": "BIOTECH",

    "XLF": "FINANCIALS",
    "FAZ": "FINANCIALS",
    "JPM": "FINANCIALS",

    "XLE": "ENERGY",
    "XOM": "ENERGY",

    "AAPL": "MEGA_CAP_TECH",
    "MSFT": "MEGA_CAP_TECH",
    "NVDA": "SEMICONDUCTORS",
    "AMD": "SEMICONDUCTORS",
    "AVGO": "SEMICONDUCTORS",
    "MU": "SEMICONDUCTORS",
    "AMZN": "CONSUMER_TECH",
    "META": "CONSUMER_TECH",
    "GOOGL": "CONSUMER_TECH",
    "TSLA": "HIGH_BETA",
    "PLTR": "HIGH_BETA",

    "BIL": "DEFENSIVE",
    "SHY": "DEFENSIVE",
    "TLT": "DEFENSIVE",
    "GLD": "DEFENSIVE",
}


# ============================================================================
# SAFE DATA HELPERS
# ============================================================================

def get_bars(data, symbol):
    """Return valid OHLCV bars for one symbol."""
    if not isinstance(data, dict):
        return []

    ohlcv = data.get("ohlcv", [])
    if not isinstance(ohlcv, list):
        return []

    result = []

    for row in ohlcv:
        if not isinstance(row, dict):
            continue

        bar = row.get(symbol)
        if isinstance(bar, dict):
            result.append(bar)

    return result


def get_float(bar, key):
    """Safely return a finite numeric field, or None."""
    try:
        value = float(bar.get(key))
    except (AttributeError, TypeError, ValueError):
        return None

    if value != value:
        return None

    if value == float("inf") or value == float("-inf"):
        return None

    return value


def get_series(bars, field):
    """Return a numeric series only when every bar has a valid field."""
    series = []

    for bar in bars:
        value = get_float(bar, field)
        if value is None:
            return []

        series.append(value)

    return series


def sma(values, period):
    if period <= 0 or len(values) < period:
        return None

    window = values[-period:]
    return sum(window) / float(period)


def ema(values, period):
    """
    Conventional EMA:
    - Seed with SMA of the first period observations.
    - Update sequentially across remaining observations.
    """
    if period <= 0 or len(values) < period:
        return None

    value = sum(values[:period]) / float(period)
    alpha = 2.0 / (period + 1.0)

    for observation in values[period:]:
        value = alpha * observation + (1.0 - alpha) * value

    return value


def roc(values, period):
    if period <= 0 or len(values) <= period:
        return None

    base = values[-period - 1]
    if base == 0:
        return None

    return values[-1] / base - 1.0


def atr(bars, period=ATR_PERIOD):
    if period <= 0 or len(bars) < period + 1:
        return None

    true_ranges = []

    for previous, current in zip(bars[-period - 1:-1], bars[-period:]):
        high = get_float(current, "high")
        low = get_float(current, "low")
        prior_close = get_float(previous, "close")

        if high is None or low is None or prior_close is None:
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

    if len(true_ranges) != period:
        return None

    return sum(true_ranges) / float(period)


def rsi(values, period=RSI_PERIOD):
    if period <= 0 or len(values) < period + 1:
        return None

    changes = [
        current - previous
        for previous, current in zip(values[-period - 1:-1], values[-period:])
    ]

    average_gain = sum(max(change, 0.0) for change in changes) / float(period)
    average_loss = sum(max(-change, 0.0) for change in changes) / float(period)

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def bollinger(values, period=BOLLINGER_PERIOD, deviation=2.0):
    if period <= 0 or len(values) < period:
        return None

    middle = sma(values, period)
    if middle is None:
        return None

    window = values[-period:]
    variance = sum((value - middle) ** 2 for value in window) / float(period)
    standard_deviation = sqrt(variance)

    return (
        middle - deviation * standard_deviation,
        middle,
        middle + deviation * standard_deviation,
    )


def directional_efficiency(values, period=EFFICIENCY_PERIOD):
    if period <= 0 or len(values) < period + 1:
        return None

    window = values[-period - 1:]
    displacement = abs(window[-1] - window[0])

    path = sum(
        abs(current - previous)
        for previous, current in zip(window[:-1], window[1:])
    )

    if path == 0:
        return 0.0

    return displacement / path


def timestamp_from_bar(bar):
    raw_value = (
        bar.get("date")
        or bar.get("datetime")
        or bar.get("time")
    )

    if raw_value is None:
        return None

    try:
        return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def date_key(bar):
    timestamp = timestamp_from_bar(bar)
    return timestamp.date().isoformat() if timestamp is not None else None


def current_session_bars(bars):
    """
    Return bars matching the calendar date of the latest bar.

    Important:
    This assumes provider timestamps are session-consistent. It does not
    attempt timezone conversion because Surmount runtime compatibility for
    zoneinfo/pytz should be verified separately.
    """
    if not bars:
        return []

    latest_key = date_key(bars[-1])
    if latest_key is None:
        return []

    return [
        bar for bar in bars
        if date_key(bar) == latest_key
    ]


def session_vwap(bars):
    """
    Current-session VWAP using typical price: (high + low + close) / 3.
    """
    session = current_session_bars(bars)
    if not session:
        return None

    total_volume = 0.0
    total_value = 0.0

    for bar in session:
        high = get_float(bar, "high")
        low = get_float(bar, "low")
        close = get_float(bar, "close")
        volume = get_float(bar, "volume")

        if None in (high, low, close, volume):
            continue

        if volume <= 0:
            continue

        typical_price = (high + low + close) / 3.0
        total_volume += volume
        total_value += typical_price * volume

    if total_volume <= 0:
        return None

    return total_value / total_volume


def relative_volume(bars, period=RVOL_PERIOD):
    """
    Local relative-volume proxy.

    This compares the current bar to the prior 20 bars. It is deliberately
    not described as a time-of-day RVOL model, because proper intraday RVOL
    requires same-clock-slot historical sessions.
    """
    if period <= 0 or len(bars) < period + 1:
        return None

    current_volume = get_float(bars[-1], "volume")
    prior_volumes = [
        get_float(bar, "volume")
        for bar in bars[-period - 1:-1]
    ]

    if current_volume is None or any(volume is None for volume in prior_volumes):
        return None

    baseline = sum(prior_volumes) / float(period)
    if baseline <= 0:
        return None

    return current_volume / baseline


def prior_high(bars, period=3):
    if period <= 0 or len(bars) < period + 1:
        return None

    highs = [get_float(bar, "high") for bar in bars[-period - 1:-1]]

    if any(high is None for high in highs):
        return None

    return max(highs)


def close_location(bar):
    high = get_float(bar, "high")
    low = get_float(bar, "low")
    close = get_float(bar, "close")

    if None in (high, low, close):
        return None

    width = high - low
    if width <= 0:
        return None

    return (close - low) / width


# ============================================================================
# STRATEGY
# ============================================================================

class TradingStrategy(Strategy):
    """
    Complete Surmount allocation strategy.

    All failures return an empty TargetAllocation rather than raising an
    execution error or emitting a malformed allocation.
    """

    def __init__(self):
        self.tickers = UNIVERSE

    @property
    def interval(self):
        return INTERVAL

    @property
    def assets(self):
        return self.tickers

    def phase(self, data):
        """
        Determine intraday phase.

        IMPORTANT:
        The timestamp may be UTC depending on the data source. During the
        initial backtest, log the raw timestamp and verify it corresponds to
        US/Eastern market time. If it is UTC, this function must be adapted.
        """
        spy_bars = get_bars(data, "SPY")
        if not spy_bars:
            return "UNKNOWN"

        timestamp = timestamp_from_bar(spy_bars[-1])
        if timestamp is None:
            return "UNKNOWN"

        minutes = timestamp.hour * 60 + timestamp.minute

        if minutes < OPEN_TRADE_START_MINUTE:
            return "PRE_OPEN"

        if minutes < MORNING_END_MINUTE:
            return "MORNING"

        if minutes < MIDDAY_END_MINUTE:
            return "MIDDAY"

        if minutes < AFTERNOON_END_MINUTE:
            return "AFTERNOON"

        if minutes < CLOSE_WINDOW_END_MINUTE:
            return "CLOSE_WINDOW"

        return "FLATTEN"

    def market_regime(self, data):
        """
        Classify broad market state from SPY, QQQ, IWM, and VIXY proxy.
        """
        spy_bars = get_bars(data, "SPY")
        qqq_bars = get_bars(data, "QQQ")
        iwm_bars = get_bars(data, "IWM")
        vixy_bars = get_bars(data, "VIXY")

        if min(
            len(spy_bars),
            len(qqq_bars),
            len(iwm_bars),
            len(vixy_bars),
        ) < MIN_BARS:
            return "NO_TRADE"

        spy = get_series(spy_bars, "close")
        qqq = get_series(qqq_bars, "close")
        iwm = get_series(iwm_bars, "close")
        vixy = get_series(vixy_bars, "close")

        if not all([spy, qqq, iwm, vixy]):
            return "NO_TRADE"

        spy_vwap = session_vwap(spy_bars)
        qqq_vwap = session_vwap(qqq_bars)
        iwm_vwap = session_vwap(iwm_bars)

        spy_fast = ema(spy, EMA_MID)
        spy_slow = ema(spy, EMA_SLOW)
        qqq_fast = ema(qqq, EMA_MID)
        qqq_slow = ema(qqq, EMA_SLOW)
        iwm_fast = ema(iwm, EMA_MID)
        iwm_slow = ema(iwm, EMA_SLOW)

        vixy_fast = ema(vixy, 10)
        vixy_slow = ema(vixy, 30)

        spy_atr = atr(spy_bars)
        spy_price = spy[-1]

        required = [
            spy_vwap, qqq_vwap, iwm_vwap,
            spy_fast, spy_slow,
            qqq_fast, qqq_slow,
            iwm_fast, iwm_slow,
            vixy_fast, vixy_slow,
            spy_atr,
        ]

        if any(value is None for value in required):
            return "NO_TRADE"

        if spy_price <= 0:
            return "NO_TRADE"

        realized_volatility = spy_atr / spy_price

        if (
            realized_volatility > 0.012
            or vixy_fast > vixy_slow * 1.08
        ):
            return "HIGH_VOL"

        states = [
            (spy[-1], spy_vwap, spy_fast, spy_slow),
            (qqq[-1], qqq_vwap, qqq_fast, qqq_slow),
            (iwm[-1], iwm_vwap, iwm_fast, iwm_slow),
        ]

        up_count = sum(
            price > vwap and fast > slow
            for price, vwap, fast, slow in states
        )

        down_count = sum(
            price < vwap and fast < slow
            for price, vwap, fast, slow in states
        )

        if up_count >= 2:
            return "BULL_TREND"

        if down_count >= 2:
            return "BEAR_TREND"

        qqq_efficiency = directional_efficiency(qqq, EFFICIENCY_PERIOD)

        if qqq_efficiency is not None and qqq_efficiency < 0.22:
            return "RANGE"

        return "CHOPPY"

    def candidate(self, symbol, data, regime, phase):
        """
        Return a fully-qualified candidate dictionary or None.
        """
        if symbol == "VIXY":
            return None

        if phase in {"UNKNOWN", "PRE_OPEN", "FLATTEN"}:
            return None

        if regime in {"NO_TRADE", "HIGH_VOL"}:
            return None

        if regime == "BULL_TREND" and symbol in INVERSE:
            return None

        if regime == "BEAR_TREND" and symbol not in INVERSE:
            return None

        if regime in {"RANGE", "CHOPPY"} and symbol in LEVERAGED:
            return None

        bars = get_bars(data, symbol)
        if len(bars) < MIN_BARS:
            return None

        closes = get_series(bars, "close")
        volumes = get_series(bars, "volume")

        if not closes or not volumes:
            return None

        price = closes[-1]
        current_atr = atr(bars)
        current_vwap = session_vwap(bars)
        current_rvol = relative_volume(bars)
        ema_fast = ema(closes, EMA_FAST)
        ema_mid = ema(closes, EMA_MID)
        ema_slow = ema(closes, EMA_SLOW)
        short_roc = roc(closes, 5)
        medium_roc = roc(closes, 21)
        current_rsi = rsi(closes, RSI_PERIOD)
        current_close_location = close_location(bars[-1])

        required = [
            current_atr, current_vwap, current_rvol,
            ema_fast, ema_mid, ema_slow,
            short_roc, medium_roc,
            current_rsi, current_close_location,
        ]

        if any(value is None for value in required):
            return None

        if price < MIN_PRICE or price <= 0:
            return None

        atr_pct = current_atr / price
        if atr_pct > MAX_ATR_PCT:
            return None

        dollar_volumes = [
            close * volume
            for close, volume in zip(closes, volumes)
        ]

        average_dollar_volume = sma(dollar_volumes, 20)

        if (
            average_dollar_volume is None
            or average_dollar_volume < MIN_AVG_DOLLAR_VOLUME
        ):
            return None

        score = 0.0
        lanes = []

        # --------------------------------------------------------------------
        # Bull trend: long continuation and breakout / reclaim conditions.
        # --------------------------------------------------------------------
        if regime == "BULL_TREND":
            if not (
                price > current_vwap
                and ema_fast > ema_mid > ema_slow
                and medium_roc > 0
            ):
                return None

            score += 2.00
            lanes.append("TREND_ALIGNMENT")

            if short_roc > 0:
                score += 0.75
                lanes.append("SHORT_TERM_MOMENTUM")

            if current_rvol >= 1.20:
                score += 1.25
                lanes.append("VOLUME_CONFIRMATION")

            recent_high = prior_high(bars, 3)
            if recent_high is not None and price > recent_high:
                score += 1.25
                lanes.append("SHORT_BREAKOUT")

            prior_close = get_float(bars[-2], "close")
            if (
                prior_close is not None
                and prior_close <= current_vwap
                and price > current_vwap
            ):
                score += 1.00
                lanes.append("VWAP_RECLAIM")

        # --------------------------------------------------------------------
        # Bear trend: buy an inverse ETF only when the mapped underlying
        # confirms its own downside structure.
        # --------------------------------------------------------------------
        elif regime == "BEAR_TREND":
            underlying = INVERSE_UNDERLYING.get(symbol)
            if underlying is None:
                return None

            underlying_bars = get_bars(data, underlying)
            underlying_closes = get_series(underlying_bars, "close")
            underlying_vwap = session_vwap(underlying_bars)

            if (
                len(underlying_bars) < MIN_BARS
                or not underlying_closes
                or underlying_vwap is None
            ):
                return None

            underlying_roc = roc(underlying_closes, 5)
            underlying_fast = ema(underlying_closes, EMA_FAST)
            underlying_mid = ema(underlying_closes, EMA_MID)

            if None in (underlying_roc, underlying_fast, underlying_mid):
                return None

            if not (
                underlying_closes[-1] < underlying_vwap
                and underlying_fast < underlying_mid
                and underlying_roc < 0
                and price > current_vwap
                and ema_fast > ema_mid
            ):
                return None

            score += 2.50
            lanes.append("UNDERLYING_BREAKDOWN")

            if current_rvol >= 1.15:
                score += 1.25
                lanes.append("INVERSE_VOLUME_CONFIRMATION")

            if short_roc > 0:
                score += 1.00
                lanes.append("INVERSE_MOMENTUM")

            if current_close_location >= 0.65:
                score += 0.50
                lanes.append("CLOSE_STRENGTH")

        # --------------------------------------------------------------------
        # Range/choppy: conservative long mean reversion only.
        # Do not apply to inverse or leveraged ETFs.
        # --------------------------------------------------------------------
        elif regime in {"RANGE", "CHOPPY"}:
            if symbol in LEVERAGED or symbol in INVERSE:
                return None

            bands = bollinger(closes, BOLLINGER_PERIOD, 2.0)
            if bands is None:
                return None

            lower_band, _, _ = bands
            vwap_distance_atr = (price - current_vwap) / current_atr

            if not (
                price <= lower_band
                and vwap_distance_atr <= -1.0
                and current_rsi <= 38
                and current_close_location >= 0.60
            ):
                return None

            score += 2.25
            lanes.append("DOWNSIDE_EXTENSION")

            if short_roc > -0.010:
                score += 0.75
                lanes.append("SELLING_DECELERATION")

            if current_rvol >= 0.90:
                score += 1.00
                lanes.append("LIQUIDITY_ACCEPTABLE")

            if current_close_location >= 0.75:
                score += 0.75
                lanes.append("RECLAIM_STRENGTH")

        else:
            return None

        # Phase adjustments deliberately lower confidence in low-quality hours.
        if phase == "MIDDAY":
            score -= 0.50

        if phase == "CLOSE_WINDOW":
            score -= 1.25

        if score < MIN_SCORE:
            return None

        return {
            "symbol": symbol,
            "score": score,
            "atr_pct": atr_pct,
            "leveraged": symbol in LEVERAGED,
            "family": SYMBOL_FAMILY.get(symbol, symbol),
            "lanes": lanes,
        }

    def position_size(self, candidate, regime):
        """
        Convert candidate strength and volatility into target allocation.
        """
        maximum = (
            MAX_LEVERAGED_ALLOCATION
            if candidate["leveraged"]
            else MAX_UNLEVERAGED_ALLOCATION
        )

        score_multiplier = min(
            1.0,
            max(0.50, candidate["score"] / 6.0),
        )

        volatility_multiplier = min(
            1.0,
            0.010 / max(candidate["atr_pct"], 0.004),
        )

        weight = maximum * score_multiplier * volatility_multiplier

        if regime in {"RANGE", "CHOPPY"}:
            weight *= 0.65

        return min(weight, maximum)

    def run(self, data):
        """
        Surmount entry point. Always returns TargetAllocation.
        """
        try:
            phase = self.phase(data)

            if phase in {"UNKNOWN", "PRE_OPEN", "FLATTEN"}:
                return TargetAllocation({})

            regime = self.market_regime(data)

            if regime == "NO_TRADE":
                return TargetAllocation({})

            if regime == "HIGH_VOL":
                bil_bars = get_bars(data, "BIL")
                if len(bil_bars) >= MIN_BARS:
                    return TargetAllocation({
                        "BIL": DEFENSIVE_BIL_ALLOCATION,
                    })
                return TargetAllocation({})

            candidates = []

            for symbol in self.tickers:
                try:
                    result = self.candidate(symbol, data, regime, phase)

                    if result is not None:
                        candidates.append(result)

                except Exception as error:
                    log("candidate_error symbol=" + symbol + " error=" + str(error))

            candidates.sort(
                key=lambda item: item["score"],
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

            allocation = {}

            for candidate in selected:
                weight = self.position_size(candidate, regime)

                if weight > 0:
                    allocation[candidate["symbol"]] = weight

            total = sum(allocation.values())

            if total > MAX_TOTAL_ALLOCATION and total > 0:
                scale = MAX_TOTAL_ALLOCATION / total

                allocation = {
                    symbol: weight * scale
                    for symbol, weight in allocation.items()
                }

            if allocation:
                log(
                    "phase=" + phase
                    + " regime=" + regime
                    + " candidates=" + str(len(candidates))
                    + " selected=" + str([
                        item["symbol"] for item in selected
                    ])
                    + " allocation=" + str(allocation)
                )

            return TargetAllocation(allocation)

        except Exception as error:
            log("strategy_error=" + str(error))
            return TargetAllocation({})