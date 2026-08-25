"""
ATHENA / SURMOUNT INTRADAY ORB + VWAP RECLAIM BASELINE
======================================================

Purpose
-------
A low-turnover, event-driven intraday research strategy for Surmount.

Initial universe
----------------
- SPY
- QQQ
- BIL as explicit idle-capital sleeve

Enabled entries
---------------
1. Fresh ORB15 breakout:
   - First 15-minute opening range is complete.
   - A completed 5-minute bar closes above its high.
   - Broad market confirmation, relative volume, pressure, and extension
     filters all pass.

2. Fresh VWAP reclaim:
   - Price recently interacts with session VWAP.
   - A completed bar closes above VWAP with positive pressure.
   - Price is not too extended from VWAP.

Risk design
-----------
- One active risk position maximum.
- Maximum active allocation: 10%.
- Remaining allocation: BIL.
- No trade in early open, midday, late afternoon, high volatility,
  insufficient history, or unclear market conditions.
- No stocks, inverse ETFs, leveraged ETFs, sector rotation, or range trades.

ATHENA duties
-------------
ATHENA must independently enforce:
- Broker position/order reconciliation.
- Spread and slippage guards.
- PMC / order-flow confirmation.
- Initial stop, partial take profit, trailing stop, and time stop.
- Daily loss limit, consecutive-loss limit, and kill switch.
- Forced flatten near the close.

Backtest requirements
---------------------
- Verify Surmount timestamp timezone.
- Verify BIL sleeve allocation semantics.
- Add realistic commissions, spread, and slippage if the platform allows it.
- Compare this baseline to a 100% BIL control.
- Use chronological walk-forward periods and a final untouched holdout.
"""

from datetime import datetime
from math import sqrt

from surmount.base_class import Strategy, TargetAllocation

ENABLE_BULL_TREND = True
ENABLE_RANGE_REVERSAL = False
ENABLE_INVERSE = False
ENABLE_STOCKS = False
ENABLE_LEVERAGED = False
ENABLE_MIDDAY_ENTRIES = False
ENABLE_CROSS_SECTIONAL_RANKING = False

ENABLE_MOMENTUM_LEADER_SELECTION = False
ENABLE_ETF_ROTATION = False
ENABLE_STOCK_ROTATION = False

MAX_POSITIONS = 1
MAX_TOTAL_RISK_ALLOCATION = 0.10
MAX_NORMAL_ALLOCATION = 0.10
MIN_RVOL_FOR_MOMENTUM = 1.35
MIN_RVOL_FOR_BREAKOUT = 1.50
MIN_ALPHA_15 = 0.0020
MIN_ALPHA_20 = 0.0040
MIN_LEADER_SCORE = 6.75
MIN_INSTITUTIONAL_RVOL = 1.35
MIN_INSTITUTIONAL_PRESSURE = 0.15
# ============================================================================
# INITIAL EXPERIMENT SETTINGS
# ============================================================================

INTERVAL = "5min"

CASH_SYMBOL = "BIL"

ACTIVE_SYMBOLS = [
    "SPY",
    "QQQ",
]

MAX_POSITIONS = 1
MAX_ACTIVE_ALLOCATION = 0.10
RISK_PER_TRADE = 0.0010

MIN_MARKET_BARS = 250
MIN_CANDIDATE_BARS = 250

MIN_PRICE = 5.00
MIN_AVG_DOLLAR_VOLUME = 50_000_000.0
MIN_CURRENT_DOLLAR_VOLUME = 10_000_000.0

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
EMA_TREND = 100

ATR_PERIOD = 14
RVOL_LOOKBACK_SESSIONS = 10

MAX_SPY_ATR_PCT = 0.010
MAX_CANDIDATE_ATR_PCT = 0.025

MIN_RVOL_ORB = 1.50
MIN_RVOL_VWAP = 1.35

MIN_PRESSURE_ORB = 0.15
MIN_PRESSURE_VWAP = 0.10

MIN_ALPHA_15 = 0.0020
MIN_ALPHA_20 = 0.0040

MAX_ORB_EXTENSION_ATR = 0.75
MAX_VWAP_EXTENSION_ATR = 0.75

MIN_EXPECTED_MOVE_COST_MULTIPLE = 3.0

# Assumes timestamps are US/Eastern. Confirm with a small backtest.
NO_ENTRY_BEFORE = 9 * 60 + 50
MORNING_END = 11 * 60 + 30
NO_ENTRY_AFTER = 15 * 60 + 15


# ============================================================================
# SAFE DATA HELPERS
# ============================================================================

def bars_for(data, symbol):
    if not isinstance(data, dict):
        return []

    rows = data.get("ohlcv", [])

    if not isinstance(rows, list):
        return []

    output = []

    for row in rows:
        if isinstance(row, dict) and isinstance(row.get(symbol), dict):
            output.append(row[symbol])

    return output


def float_value(bar, field):
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

    previous = values[-length - 1]

    if previous == 0:
        return None

    return values[-1] / previous - 1.0


def atr(bars, length=ATR_PERIOD):
    if length <= 0 or len(bars) < length + 1:
        return None

    ranges = []

    for previous, current in zip(bars[-length - 1:-1], bars[-length:]):
        high = float_value(current, "high")
        low = float_value(current, "low")
        previous_close = float_value(previous, "close")

        if None in (high, low, previous_close):
            return None

        if high < low or low <= 0 or previous_close <= 0:
            return None

        ranges.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    if len(ranges) != length:
        return None

    return sum(ranges) / float(length)


def candle_pressure(bars, length=5):
    """
    OHLCV directional-pressure proxy; not true order flow or PMC.
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


# ============================================================================
# TIME / SESSION / VWAP / RVOL HELPERS
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
    today, _ = today_and_prior(bars)

    if not today:
        return None

    volume_sum = 0.0
    value_sum = 0.0

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

        volume_sum += volume
        value_sum += typical_price * volume

    return value_sum / volume_sum if volume_sum > 0 else None


def session_rvol(bars, lookback_sessions=RVOL_LOOKBACK_SESSIONS):
    """
    Latest bar volume versus equivalent bar index in preceding sessions.
    """
    groups = session_groups(bars)

    if len(groups) >= 4:
        today = groups[-1][1]
        slot = len(today) - 1
        current_volume = float_value(today[-1], "volume")

        if current_volume is not None:
            historical = []

            for _, group in groups[-lookback_sessions - 1:-1]:
                if len(group) > slot:
                    volume = float_value(group[slot], "volume")

                    if volume is not None and volume >= 0:
                        historical.append(volume)

            if len(historical) >= 3:
                baseline = sum(historical) / float(len(historical))

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


def opening_range(bars):
    """
    Return today’s first 15-minute opening range, defined as the first
    three 5-minute bars.
    """
    today, _ = today_and_prior(bars)

    if len(today) < 3:
        return None

    first_three = today[:3]

    highs = [float_value(bar, "high") for bar in first_three]
    lows = [float_value(bar, "low") for bar in first_three]

    if any(value is None for value in highs + lows):
        return None

    return {
        "high": max(highs),
        "low": min(lows),
        "bars": first_three,
    }


# ============================================================================
# ALLOCATION / COST HELPERS
# ============================================================================

def estimated_round_trip_cost(symbol):
    """Conservative friction estimate; replace with measured paper-fill data."""
    if symbol in {"SPY", "QQQ"}:
        return 0.0008

    return 0.0012


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def with_bil_sleeve(active_allocations):
    """
    Always emit fully specified allocation targets.

    No trade:
        {"BIL": 1.0}

    SPY trade:
        {"SPY": 0.10, "BIL": 0.90}
    """
    active = {}

    for symbol, allocation in active_allocations.items():
        try:
            weight = float(allocation)
        except (TypeError, ValueError):
            continue

        if weight > 0 and symbol != CASH_SYMBOL:
            active[symbol] = weight

    total = sum(active.values())

    if total > 1.0:
        scale = 1.0 / total
        active = {
            symbol: weight * scale
            for symbol, weight in active.items()
        }
        total = 1.0

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
    Event-driven SPY/QQQ intraday baseline.

    This strategy is intentionally stateless. ATHENA, not Surmount, should
    retain entry state, stop/target state, holding time, and broker truth.
    """

    def __init__(self):
        self.tickers = ACTIVE_SYMBOLS + [CASH_SYMBOL]

    @property
    def interval(self):
        return INTERVAL

    @property
    def assets(self):
        return self.tickers

    def phase(self, data):
        spy_bars = bars_for(data, "SPY")

        if not spy_bars:
            return "UNKNOWN"

        timestamp = bar_timestamp(spy_bars[-1])

        if timestamp is None:
            return "UNKNOWN"

        minute = timestamp.hour * 60 + timestamp.minute

        if minute < NO_ENTRY_BEFORE:
            return "PRE_OPEN"

        if minute < MORNING_END:
            return "MORNING"

        if minute < NO_ENTRY_AFTER:
            return "NO_NEW_ENTRY"

        return "FLATTEN"

    def market_is_bullish(self, data):
        """
        Require SPY and QQQ broad confirmation before considering any long.
        """
        required = {}

        for symbol in ACTIVE_SYMBOLS:
            bars = bars_for(data, symbol)

            if len(bars) < MIN_MARKET_BARS:
                return False, {}

            closes = series_for(bars, "close")
            current_vwap = session_vwap(bars)
            fast = ema(closes, EMA_FAST)
            mid = ema(closes, EMA_MID)
            slow = ema(closes, EMA_SLOW)
            trend = ema(closes, EMA_TREND)
            momentum_20 = roc(closes, 20)

            if any(
                value is None
                for value in [
                    current_vwap,
                    fast,
                    mid,
                    slow,
                    trend,
                    momentum_20,
                ]
            ):
                return False, {}

            required[symbol] = {
                "bars": bars,
                "closes": closes,
                "vwap": current_vwap,
                "fast": fast,
                "mid": mid,
                "slow": slow,
                "trend": trend,
                "roc20": momentum_20,
            }

        spy_atr = atr(required["SPY"]["bars"])
        spy_price = required["SPY"]["closes"][-1]

        if spy_atr is None or spy_price <= 0:
            return False, {}

        if spy_atr / spy_price >= MAX_SPY_ATR_PCT:
            return False, {}

        bullish = all(
            item["closes"][-1] > item["vwap"]
            and item["fast"] > item["mid"] > item["slow"]
            and item["closes"][-1] > item["trend"]
            and item["roc20"] > 0
            for item in required.values()
        )

        return bullish, required

    def candidate(self, symbol, market):
        """
        Evaluate one fresh ORB15 or VWAP-reclaim event.

        A candidate must satisfy:
        1. Liquidity and volatility requirements.
        2. Broad-market bullish confirmation.
        3. Relative strength versus the other index.
        4. Fresh entry event, not a stale condition.
        5. High time-adjusted volume and positive candle pressure.
        """
        bars = market[symbol]["bars"]
        closes = market[symbol]["closes"]

        if len(bars) < MIN_CANDIDATE_BARS:
            return None

        price = closes[-1]
        previous_close = closes[-2]

        current_atr = atr(bars)
        current_vwap = market[symbol]["vwap"]
        current_rvol = session_rvol(bars)
        pressure = candle_pressure(bars, 5)
        current_range = opening_range(bars)

        roc5 = roc(closes, 5)
        roc15 = roc(closes, 15)
        roc20 = roc(closes, 20)

        if any(
            value is None
            for value in [
                current_atr,
                current_vwap,
                current_rvol,
                pressure,
                current_range,
                roc5,
                roc15,
                roc20,
            ]
        ):
            return None

        if price < MIN_PRICE or price <= 0:
            return None

        atr_pct = current_atr / price

        if atr_pct > MAX_CANDIDATE_ATR_PCT:
            return None

        volumes = series_for(bars, "volume")

        if not volumes:
            return None

        average_dollar_volume = sma(
            [
                close * volume
                for close, volume in zip(closes[-20:], volumes[-20:])
            ],
            20,
        )

        current_dollar_volume = price * volumes[-1]

        if (
            average_dollar_volume is None
            or average_dollar_volume < MIN_AVG_DOLLAR_VOLUME
            or current_dollar_volume < MIN_CURRENT_DOLLAR_VOLUME
        ):
            return None

        other = "QQQ" if symbol == "SPY" else "SPY"
        other_closes = market[other]["closes"]

        other_roc15 = roc(other_closes, 15)
        other_roc20 = roc(other_closes, 20)

        if other_roc15 is None or other_roc20 is None:
            return None

        alpha15 = roc15 - other_roc15
        alpha20 = roc20 - other_roc20

        if alpha15 < MIN_ALPHA_15 or alpha20 < MIN_ALPHA_20:
            return None

        latest_low = float_value(bars[-1], "low")
        prior_low = float_value(bars[-2], "low")

        if latest_low is None or prior_low is None:
            return None

        score = 0.0
        setup = None
        target_r = 2.0
        stop_atr = 1.0

        # Fresh ORB15 breakout: previous close below/equal range high,
        # completed current bar closes above range high.
        orb_breakout = (
            previous_close <= current_range["high"]
            and price > current_range["high"]
            and price <= current_range["high"] + MAX_ORB_EXTENSION_ATR * current_atr
            and current_rvol >= MIN_RVOL_ORB
            and pressure >= MIN_PRESSURE_ORB
        )

        # Fresh VWAP reclaim: recent interaction with VWAP, then a confirmed
        # close back above VWAP without excessive extension.
        vwap_reclaim = (
            min(latest_low, prior_low)
            <= current_vwap + 0.15 * current_atr
            and previous_close <= current_vwap + 0.10 * current_atr
            and price > current_vwap
            and roc5 > 0
            and current_rvol >= MIN_RVOL_VWAP
            and pressure >= MIN_PRESSURE_VWAP
            and price - current_vwap <= MAX_VWAP_EXTENSION_ATR * current_atr
        )

        if orb_breakout:
            score = 6.0
            setup = "ORB15_BREAKOUT"
            target_r = 2.5
            stop_atr = 1.0

        elif vwap_reclaim:
            score = 5.5
            setup = "VWAP_RECLAIM"
            target_r = 2.2
            stop_atr = 0.90

        else:
            return None

        if current_rvol >= 1.75:
            score += 0.50

        if pressure >= 0.20:
            score += 0.50

        if alpha20 >= 0.006:
            score += 0.50

        expected_move = target_r * stop_atr * atr_pct

        if expected_move < (
            MIN_EXPECTED_MOVE_COST_MULTIPLE
            * estimated_round_trip_cost(symbol)
        ):
            return None

        return {
            "symbol": symbol,
            "score": score,
            "setup": setup,
            "atr_pct": atr_pct,
            "stop_atr": stop_atr,
            "target_r": target_r,
            "rvol": current_rvol,
            "pressure": pressure,
            "alpha15": alpha15,
            "alpha20": alpha20,
        }

    def allocation_for(self, candidate):
        """
        Conservative ATR-risk allocation proxy.

        Active exposure is capped at MAX_ACTIVE_ALLOCATION. ATHENA must apply
        live dollar-risk, actual stop, slippage, and correlation controls.
        """
        risk_distance = max(
            candidate["stop_atr"] * candidate["atr_pct"],
            0.003,
        )

        score_scale = clamp(
            candidate["score"] / 7.0,
            0.70,
            1.0,
        )

        weight = RISK_PER_TRADE / risk_distance
        weight *= score_scale

        return clamp(
            weight,
            0.0,
            MAX_ACTIVE_ALLOCATION,
        )

    def run(self, data):
        """
        Return one active SPY/QQQ target at most, plus BIL residual allocation.
        """
        try:
            phase = self.phase(data)

            if phase != "MORNING":
                return TargetAllocation(with_bil_sleeve({}))

            bullish, market = self.market_is_bullish(data)

            if not bullish:
                return TargetAllocation(with_bil_sleeve({}))

            candidates = []

            for symbol in ACTIVE_SYMBOLS:
                candidate = self.candidate(symbol, market)

                if candidate is not None:
                    candidates.append(candidate)

            if not candidates:
                return TargetAllocation(with_bil_sleeve({}))

            candidates.sort(
                key=lambda item: (
                    item["score"],
                    item["alpha20"],
                    item["rvol"],
                ),
                reverse=True,
            )

            best = candidates[0]

            allocation = {
                best["symbol"]: self.allocation_for(best),
            }

            active_total = sum(allocation.values())

            if active_total > MAX_ACTIVE_ALLOCATION:
                scale = MAX_ACTIVE_ALLOCATION / active_total

                allocation = {
                    symbol: weight * scale
                    for symbol, weight in allocation.items()
                }

            return TargetAllocation(with_bil_sleeve(allocation))

        except Exception:
            return TargetAllocation(with_bil_sleeve({}))