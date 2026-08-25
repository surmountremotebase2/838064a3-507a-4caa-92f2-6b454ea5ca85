"""Audited ATHENA/PMC-inspired intraday strategy for Surmount.

Paste this entire file into Surmount Code Builder as main.py.

Why this version is different from the Claude draft:
* opening-range entries require a fresh cross and pressure confirmation;
* volume is time-of-day adjusted instead of comparing the open with midday;
* momentum entries require relative strength versus SPY/QQQ;
* range entries require a reversal candle (no blind catch-a-falling-knife);
* allocation is scaled from an explicit ATR risk budget and capped by product;
* realized SPY volatility replaces the Claude draft's VIXY-as-VIX assumption;
* existing holdings are preserved until a deterministic invalidation/exit;
* optional sentiment/insider feeds are soft filters and never become a stale
  hard stop; and
* the file fails closed when the broad-market data is genuinely insufficient.

Important Surmount limitation:
TargetAllocation is not a broker order. This file can express a desired
position and a zero-allocation exit, but it cannot submit a native stop,
profit-target, order-flow delta, live news, option-chain/GEX, or ATHENA PMC
bracket. Use the ATHENA-native RiskKernel/bracket layer for those controls.
This is paper-test code and makes no profitability guarantee.
"""

from datetime import datetime
from math import sqrt

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - old runtime fallback
    ZoneInfo = None

from surmount.base_class import Strategy, TargetAllocation
from surmount.data import InsiderTrading, SocialSentiment


# Keep this finite and liquid. Adding many symbols increases fetch failures and
# makes a five-minute backtest less meaningful. Expand only after this list
# passes a cost-loaded out-of-sample test.
CORE_ETF = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI",
]
DEFENSIVE = ["BIL", "SHY", "GLD"]
LONG_LEVERAGED = ["TQQQ", "SOXL", "UPRO", "TECL"]
INVERSE_LEVERAGED = ["SQQQ", "SOXS", "SPXU", "PSQ"]
STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA",
    "AVGO", "MU", "JPM",
]

UNIVERSE = list(dict.fromkeys(
    CORE_ETF + DEFENSIVE + LONG_LEVERAGED + INVERSE_LEVERAGED + STOCKS
))
LEVERAGED = set(LONG_LEVERAGED + INVERSE_LEVERAGED)
INVERSE = set(INVERSE_LEVERAGED)

# Only these feeds are requested. They are useful as soft context, but are not
# a substitute for timestamped news or an options feed.
ALT_DATA_SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "NVDA"]

# Risk controls are expressed as portfolio fractions because Surmount exposes
# target allocations rather than account equity and broker stop orders.
RISK_PER_TRADE = 0.0020       # 0.20% notional-risk proxy per position
MAX_TOTAL_ALLOCATION = 0.22   # one active risk position; cash is intentional
MAX_NORMAL_ALLOCATION = 0.22
MAX_LEVERAGED_ALLOCATION = 0.08
MAX_GRID_ALLOCATION = 0.14
REBALANCE_THRESHOLD = 0.08
MIN_HOLD_BARS = 6             # 30 minutes; lifecycle belongs in ATHENA
MIN_DOLLAR_VOLUME_PER_BAR = 5_000_000.0
MIN_MARKET_BARS = 250         # avoid two-session regime decisions
MIN_CANDIDATE_BARS = 250
ROBUST_HISTORY_BARS = 2000    # about 25 regular sessions of 5-minute bars

TECH_FAMILY = {
    "QQQ", "SMH", "TQQQ", "SOXL", "TECL", "NVDA", "AMD", "AVGO", "MU"
}
BROAD_FAMILY = {
    "SPY", "UPRO", "DIA", "XLK", "AAPL", "MSFT", "AMZN", "META", "JPM"
}
INVERSE_UNDERLYING = {
    "SQQQ": "QQQ",
    "PSQ": "QQQ",
    "SOXS": "SMH",
    "SPXU": "SPY",
}


def _bars(data, symbol):
    out = []
    for row in data.get("ohlcv", []) or []:
        if isinstance(row, dict) and isinstance(row.get(symbol), dict):
            out.append(row[symbol])
    return out


def _values(xs, key):
    out = []
    for item in xs:
        try:
            out.append(float(item.get(key, 0) or 0))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _sma(values, length):
    if len(values) < length:
        return None
    return sum(values[-length:]) / float(length)


def _ema(values, length):
    if len(values) < length:
        return None
    result = sum(values[:length]) / float(length)
    alpha = 2.0 / (length + 1.0)
    for value in values[length:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _atr(xs, length=14):
    if len(xs) < length + 1:
        return None
    true_ranges = []
    for previous, current in zip(xs[-length - 1:-1], xs[-length:]):
        try:
            high = float(current["high"])
            low = float(current["low"])
            previous_close = float(previous["close"])
        except (KeyError, TypeError, ValueError):
            continue
        true_ranges.append(max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        ))
    return sum(true_ranges) / len(true_ranges) if true_ranges else None


def _rsi(closes, length=14):
    if len(closes) < length + 1:
        return None
    changes = [
        b - a for a, b in zip(closes[-length - 1:-1], closes[-length:])
    ]
    gains = sum(max(change, 0.0) for change in changes) / float(length)
    losses = sum(max(-change, 0.0) for change in changes) / float(length)
    return 100.0 if losses == 0 else 100.0 - 100.0 / (1.0 + gains / losses)


def _vwap(xs):
    total_volume = 0.0
    total_value = 0.0
    for item in xs:
        try:
            volume = float(item.get("volume", 0) or 0)
            close = float(item["close"])
        except (KeyError, TypeError, ValueError):
            continue
        total_volume += max(volume, 0.0)
        total_value += close * max(volume, 0.0)
    return total_value / total_volume if total_volume > 0 else None


def _roc(closes, length):
    if len(closes) <= length or closes[-length - 1] == 0:
        return None
    return closes[-1] / closes[-length - 1] - 1.0


def _bollinger(closes, length=20, multiple=2.0):
    if len(closes) < length:
        return None
    middle = _sma(closes, length)
    if middle is None:
        return None
    variance = sum((value - middle) ** 2 for value in closes[-length:]) / length
    deviation = sqrt(variance)
    return (
        middle - multiple * deviation,
        middle,
        middle + multiple * deviation,
    )


def _efficiency(closes, length=20):
    if len(closes) < length + 1:
        return None
    displacement = abs(closes[-1] - closes[-length - 1])
    path = sum(
        abs(b - a)
        for a, b in zip(closes[-length - 1:-1], closes[-length:])
    )
    return displacement / path if path else 0.0


def _realized_vol(closes, length=50):
    """Annualized realized volatility from returns, not VIXY."""
    if len(closes) < length + 1:
        return None
    returns = []
    for previous, current in zip(closes[-length - 1:-1], closes[-length:]):
        if previous:
            returns.append(current / previous - 1.0)
    if len(returns) < max(20, length // 2):
        return None
    # 78 five-minute regular-session bars per day, 252 trading days/year.
    return sqrt(sum(value * value for value in returns) / len(returns)) * sqrt(78.0 * 252.0)


def _pressure(xs, length=5):
    """Signed-volume proxy; true bid/ask delta is unavailable in Surmount."""
    weighted = 0.0
    volume_total = 0.0
    for item in xs[-length:]:
        try:
            high = float(item["high"])
            low = float(item["low"])
            open_price = float(item["open"])
            close = float(item["close"])
            volume = max(float(item.get("volume", 0) or 0), 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        width = max(high - low, 0.01)
        weighted += ((close - open_price) / width) * volume
        volume_total += volume
    return weighted / volume_total if volume_total else 0.0


def _close_location(item):
    try:
        high = float(item["high"])
        low = float(item["low"])
        close = float(item["close"])
    except (KeyError, TypeError, ValueError):
        return 0.5
    return (close - low) / max(high - low, 0.01)


def _parse_stamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _local_stamp(value):
    stamp = _parse_stamp(value)
    if stamp is None:
        return None
    if stamp.tzinfo is not None and ZoneInfo is not None:
        try:
            return stamp.astimezone(ZoneInfo("America/New_York")).replace(
                tzinfo=None
            )
        except Exception:
            pass
    return stamp.replace(tzinfo=None)


def _bar_stamp(item):
    if not isinstance(item, dict):
        return None
    return _local_stamp(
        item.get("date") or item.get("datetime") or item.get("time")
    )


def _day_key(item):
    stamp = _bar_stamp(item)
    return stamp.date().isoformat() if stamp else None


def _last_stamp(data):
    rows = data.get("ohlcv", []) or []
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        for item in row.values():
            stamp = _bar_stamp(item)
            if stamp is not None:
                return stamp
    return None


def _session_groups(xs):
    groups = {}
    order = []
    for item in xs:
        key = _day_key(item)
        if key is None:
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    return [(key, groups[key]) for key in order]


def _session_bars(xs):
    groups = _session_groups(xs)
    if groups:
        today = groups[-1][1]
        prior = [item for _, group in groups[:-1] for item in group]
        return today, prior
    if not xs:
        return [], []
    return xs[-78:], xs[:-78]


def _session_rvol(xs, lookback_days=10):
    """Compare a bar with the same intraday slot on prior sessions."""
    groups = _session_groups(xs)
    if len(groups) >= 2:
        today = groups[-1][1]
        slot = len(today) - 1
        try:
            current_volume = float(today[-1].get("volume", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            current_volume = 0.0
        prior_volumes = []
        for _, group in groups[-lookback_days - 1:-1]:
            if len(group) > slot:
                try:
                    prior_volumes.append(float(group[slot].get("volume", 0) or 0))
                except (AttributeError, TypeError, ValueError):
                    continue
        if len(prior_volumes) >= 3:
            baseline = sum(prior_volumes) / len(prior_volumes)
            return current_volume / baseline if baseline > 0 else None
    # A short backtest may have fewer than three completed sessions.
    return _rvol_fallback(xs)


def _rvol_fallback(xs, length=20):
    if len(xs) < length + 1:
        return None
    volumes = _values(xs, "volume")
    baseline = _sma(volumes[:-1], length)
    return volumes[-1] / baseline if baseline and baseline > 0 else None


def _social(data, symbol, anchor_stamp):
    rows = data.get(("social_sentiment", symbol), []) or []
    if not rows:
        return None, False, False
    latest = rows[-1]
    values = []
    for key in ("stocktwitsSentiment", "twitterSentiment"):
        try:
            if latest.get(key) is not None:
                values.append(float(latest[key]))
        except (TypeError, ValueError):
            continue
    if not values:
        return None, False, False
    stamp = _local_stamp(latest.get("date"))
    fresh = (
        stamp is not None
        and anchor_stamp is not None
        and abs((anchor_stamp.date() - stamp.date()).days) <= 5
    )
    value = sum(values) / len(values)
    extreme = fresh and (value < 0.18 or value > 0.82)
    return value, fresh, extreme


def _recent_insider_sale(data, symbol, anchor_stamp):
    rows = data.get(("insider_trading", symbol), []) or []
    if not rows:
        return False
    latest = rows[-1]
    transaction = str(latest.get("transactionType", "")).lower()
    if "sale" not in transaction:
        return False
    stamp = _local_stamp(
        latest.get("transactionDate") or latest.get("filingDate")
    )
    return bool(
        stamp is not None
        and anchor_stamp is not None
        and 0 <= (anchor_stamp.date() - stamp.date()).days <= 60
    )


def _family(symbol):
    if symbol in TECH_FAMILY:
        return "TECH"
    if symbol in BROAD_FAMILY:
        return "BROAD"
    if symbol in {"IWM"}:
        return "SMALL_CAP"
    if symbol in {"XLF"}:
        return "FINANCIAL"
    if symbol in {"XLE"}:
        return "ENERGY"
    if symbol in {"XBI"}:
        return "BIOTECH"
    if symbol in DEFENSIVE:
        return "DEFENSIVE"
    return symbol


def _clamp(value, low, high):
    return max(low, min(high, value))


def _estimated_cost_pct(symbol):
    """Conservative round-trip cost floor used as an entry sanity check."""
    if symbol in LEVERAGED:
        return 0.0018
    if symbol in CORE_ETF:
        return 0.0008
    return 0.0012


class TradingStrategy(Strategy):
    """Regime router for liquid US equities, ETFs, and inverse ETFs."""

    def __init__(self):
        self.tickers = UNIVERSE
        self.data_list = (
            [SocialSentiment(symbol) for symbol in ALT_DATA_SYMBOLS]
            + [InsiderTrading(symbol) for symbol in ALT_DATA_SYMBOLS]
        )
        # This is only an allocation-layer approximation. ATHENA must own the
        # durable entry/stop/target state and broker execution.
        self._position_state = {}

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
        stamp = _last_stamp(data)
        if stamp is None:
            return "UNKNOWN"
        minutes = stamp.hour * 60 + stamp.minute
        if minutes < 9 * 60 + 35:
            return "PRE_OPEN"
        if minutes < 9 * 60 + 50:
            return "OPEN"
        if minutes < 10 * 60 + 15:
            return "ORB15"
        if minutes < 11 * 60:
            return "MORNING"
        if minutes < 13 * 60 + 30:
            return "MIDDAY"
        if minutes < 15 * 60 + 15:
            return "AFTERNOON"
        if minutes < 15 * 60 + 45:
            return "CLOSE_WINDOW"
        return "FLATTEN"

    def _reference(self, xs):
        today, prior = _session_bars(xs)
        if len(today) < 3 or not prior:
            return None
        prior_close = float(prior[-1]["close"])
        first_open = float(today[0]["open"])
        orb15 = today[:3]
        orb30 = today[:6] if len(today) >= 6 else []
        return {
            "today": today,
            "gap": first_open / prior_close - 1.0 if prior_close else 0.0,
            "open": first_open,
            "prior_close": prior_close,
            "orb15_high": max(float(item["high"]) for item in orb15),
            "orb15_low": min(float(item["low"]) for item in orb15),
            "orb30_high": (
                max(float(item["high"]) for item in orb30) if orb30 else None
            ),
            "orb30_low": (
                min(float(item["low"]) for item in orb30) if orb30 else None
            ),
        }

    def _market_context(self, data):
        broad_symbols = ("SPY", "QQQ", "IWM", "DIA")
        market = {symbol: _bars(data, symbol) for symbol in broad_symbols}
        if any(len(market[symbol]) < MIN_MARKET_BARS for symbol in ("SPY", "QQQ")):
            return {
                "regime": "NO_TRADE",
                "breadth": None,
                "efficiency": None,
                "volatility": None,
                "market": market,
                "event": True,
            }

        trends = []
        efficiencies = []
        for symbol in broad_symbols:
            xs = market[symbol]
            if len(xs) < 120:
                continue
            closes = _values(xs, "close")
            vwap = _vwap(xs[-78:])
            ema20 = _ema(closes, 20)
            ema50 = _ema(closes, 50)
            ema100 = _ema(closes, 100)
            roc5 = _roc(closes, 5)
            roc20 = _roc(closes, 20)
            efficiency = _efficiency(closes, 20)
            if None in (vwap, ema20, ema50, ema100, roc5, roc20, efficiency):
                continue
            ema200 = _ema(closes, 200) if len(closes) >= 200 else None
            macro_up = ema200 is None or closes[-1] > ema200
            # If the backtest window is shorter than 200 bars, the 200-bar
            # anchor is unavailable; the multi-index trend becomes the
            # fallback rather than disabling every bearish regime.
            macro_down = ema200 is None or closes[-1] < ema200
            trends.append({
                "symbol": symbol,
                "up": closes[-1] > vwap and ema20 > ema50 > ema100 and roc20 > 0,
                "down": closes[-1] < vwap and ema20 < ema50 < ema100 and roc20 < 0,
                "macro_up": macro_up,
                "macro_down": macro_down,
                "price": closes[-1],
                "vwap": vwap,
            })
            efficiencies.append(efficiency)

        if len(trends) < 2:
            return {
                "regime": "NO_TRADE",
                "breadth": None,
                "efficiency": None,
                "volatility": None,
                "market": market,
                "event": True,
            }

        spy = market["SPY"]
        spy_closes = _values(spy, "close")
        spy_atr = _atr(spy)
        spy_price = spy_closes[-1]
        volatility = spy_atr / spy_price if spy_atr and spy_price else None
        realized_volatility = _realized_vol(spy_closes, 50)
        breadth = (
            sum(item["up"] for item in trends) -
            sum(item["down"] for item in trends)
        ) / float(len(trends))
        average_efficiency = sum(efficiencies) / len(efficiencies)

        shock = False
        if len(spy) >= 4 and spy_closes[-4] != 0:
            shock = max(
                abs(float(item["high"]) - float(item["low"])) / spy_closes[-4]
                for item in spy[-3:]
            ) > 0.012

        anchor = _last_stamp(data)
        sentiment_extremes = sum(
            _social(data, symbol, anchor)[2]
            for symbol in ("SPY", "QQQ", "IWM")
        )
        spy_ref = self._reference(spy)
        gap_event = bool(
            spy_ref is not None and abs(spy_ref["gap"]) >= 0.035
        )
        event = sentiment_extremes >= 2 or gap_event
        high_vol = bool(
            (volatility is not None and volatility >= 0.012)
            or (realized_volatility is not None and realized_volatility >= 0.45)
            or shock
        )

        if high_vol:
            regime = "HIGH_VOL"
        elif event:
            regime = "EVENT_RISK"
        elif breadth >= 0.34 and sum(item["macro_up"] for item in trends) >= 2:
            regime = "BULL_TREND"
        elif breadth <= -0.34 and sum(item["macro_down"] for item in trends) >= 2:
            regime = "BEAR_TREND"
        elif average_efficiency <= 0.24:
            regime = "GRID_RANGE"
        else:
            regime = "CHOPPY"
        return {
            "regime": regime,
            "breadth": breadth,
            "efficiency": average_efficiency,
            "volatility": volatility,
            "realized_volatility": realized_volatility,
            "history_scale": _clamp(
                len(spy) / float(ROBUST_HISTORY_BARS), 0.50, 1.0
            ),
            "market": market,
            "event": event,
            "trends": trends,
        }

    def _benchmark(self, symbol, ctx):
        if symbol in TECH_FAMILY or symbol in {"SQQQ", "PSQ", "SOXS"}:
            return "QQQ" if symbol not in {"SOXS"} else "SMH"
        return "SPY"

    def _candidate(self, symbol, data, ctx, phase):
        regime = ctx["regime"]
        if (
            symbol == "VIXY"
            or phase not in {"ORB15", "MORNING", "MIDDAY", "AFTERNOON"}
            or regime in {"NO_TRADE", "HIGH_VOL", "EVENT_RISK"}
        ):
            return None
        if regime == "BEAR_TREND" and symbol not in INVERSE:
            return None
        if regime == "BULL_TREND" and symbol in INVERSE:
            return None
        if regime in {"GRID_RANGE", "CHOPPY"} and (
            symbol in LEVERAGED or symbol in INVERSE
        ):
            return None

        xs = _bars(data, symbol)
        if len(xs) < MIN_CANDIDATE_BARS:
            return None
        closes = _values(xs, "close")
        volumes = _values(xs, "volume")
        price = closes[-1]
        atr = _atr(xs)
        reference = self._reference(xs)
        if reference is None or atr is None or price <= 5:
            return None
        session_vwap = _vwap(reference["today"])
        vwap = session_vwap if session_vwap is not None else _vwap(xs[-78:])
        rvol = _session_rvol(xs)
        ema9, ema21, ema50, ema100 = (
            _ema(closes, length) for length in (9, 21, 50, 100)
        )
        roc5, roc15, roc20 = _roc(closes, 5), _roc(closes, 15), _roc(closes, 20)
        pressure = _pressure(xs, 5)
        if None in (vwap, rvol, ema9, ema21, ema50, ema100, roc5, roc15, roc20):
            return None
        atr_pct = atr / price if price else 1.0
        if atr_pct > (0.075 if symbol in LEVERAGED else 0.045):
            return None
        average_dollar_volume = _sma(
            [close * volume for close, volume in zip(closes, volumes)], 20
        )
        if (
            average_dollar_volume is None
            or average_dollar_volume < MIN_DOLLAR_VOLUME_PER_BAR
        ):
            return None

        previous_close = closes[-2]
        close_location = _close_location(xs[-1])
        benchmark = self._benchmark(symbol, ctx)
        benchmark_bars = ctx["market"].get(benchmark) or _bars(data, benchmark)
        benchmark_closes = _values(benchmark_bars, "close")
        benchmark_roc5 = _roc(benchmark_closes, 5)
        benchmark_roc15 = _roc(benchmark_closes, 15)
        benchmark_roc20 = _roc(benchmark_closes, 20)
        alpha5 = roc5 - benchmark_roc5 if benchmark_roc5 is not None else 0.0
        alpha15 = roc15 - benchmark_roc15 if benchmark_roc15 is not None else 0.0
        alpha20 = roc20 - benchmark_roc20 if benchmark_roc20 is not None else 0.0

        score = 0.0
        lanes = []
        setup = None
        stop_atr = 1.0
        target_r = 1.8

        if regime == "BULL_TREND":
            macro_ok = True
            ema200 = _ema(closes, 200) if len(closes) >= 200 else None
            if ema200 is not None:
                macro_ok = price > ema200
            trend_ok = (
                price > vwap
                and ema9 > ema21 > ema50
                and roc20 > 0
                and roc15 > 0
                and macro_ok
                and alpha20 > -0.008
            )
            if not trend_ok:
                return None
            score += 2.0
            lanes.append("TREND_AND_MACRO")
            if alpha20 > 0.002:
                score += 0.8
                lanes.append("RELATIVE_STRENGTH_ALPHA")

            orb15_cross = (
                phase in {"ORB15", "MORNING"}
                and previous_close <= reference["orb15_high"]
                and price > reference["orb15_high"]
                and price <= reference["orb15_high"] + 0.90 * atr
                and rvol >= 1.20
                and pressure >= 0.10
            )
            orb30_cross = (
                phase in {"MORNING", "MIDDAY"}
                and reference["orb30_high"] is not None
                and previous_close <= reference["orb30_high"]
                and price > reference["orb30_high"]
                and price <= reference["orb30_high"] + 0.90 * atr
                and rvol >= 1.15
                and pressure >= 0.10
            )
            vwap_pullback = (
                min(float(xs[-1]["low"]), float(xs[-2]["low"]))
                <= vwap + 0.20 * atr
                and price > vwap
                and previous_close <= vwap + 0.10 * atr
                and roc5 > 0
                and pressure >= 0.05
            )
            gap_go = (
                0.005 <= reference["gap"] <= 0.030
                and price > reference["open"]
                and rvol >= 1.10
                and pressure >= 0.10
                and alpha5 >= 0
            )
            if orb15_cross:
                score += 3.0
                lanes.append("ORB15_FRESH_CROSS")
                setup = "ORB"
            elif orb30_cross:
                score += 2.8
                lanes.append("ORB30_FRESH_CROSS")
                setup = "ORB"
            elif vwap_pullback:
                score += 2.6
                lanes.append("VWAP_RECLAIM")
                setup = "PULLBACK"
            elif gap_go:
                score += 2.2
                lanes.append("GAP_CONTINUATION")
                setup = "GAP"
            elif (
                roc5 > 0.001
                and rvol >= 1.10
                and pressure >= 0.10
                and price - vwap <= 1.8 * atr
            ):
                score += 2.0
                lanes.append("MOMENTUM_VOLUME")
                setup = "MOMENTUM"
            else:
                return None
            if rvol >= 1.15:
                score += 0.5
                lanes.append("TIME_OF_DAY_RVOL")
            if pressure >= 0.15:
                score += 0.5
                lanes.append("PMC_GREEN_PROXY")
            # Trend trades use a 1:3 target when the native ATHENA bracket is
            # attached. The allocation proxy still exits on invalidation or
            # the end of session if price never reaches that target.
            target_r = 3.0 if setup in {"ORB", "GAP"} else 2.5

        elif regime == "BEAR_TREND":
            if symbol not in INVERSE:
                return None
            underlying = INVERSE_UNDERLYING.get(symbol, "SPY")
            underlying_bars = ctx["market"].get(underlying) or _bars(data, underlying)
            if len(underlying_bars) < 120:
                return None
            uc = _values(underlying_bars, "close")
            uvwap = _vwap(underlying_bars[-78:])
            uema21 = _ema(uc, 21)
            uema50 = _ema(uc, 50)
            uroc5 = _roc(uc, 5)
            upressure = _pressure(underlying_bars, 5)
            if None in (uvwap, uema21, uema50, uroc5):
                return None
            if not (
                uc[-1] < uvwap
                and uema21 < uema50
                and uroc5 < 0
                and upressure <= -0.05
            ):
                return None
            if not (
                price > vwap
                and ema9 > ema21
                and roc5 > 0
                and pressure >= 0.05
                and rvol >= 1.05
            ):
                return None
            score = 3.5
            lanes = ["UNDERLYING_BEAR_CONFIRMATION", "INVERSE_ETF_RECLAIM"]
            if rvol >= 1.20:
                score += 0.7
                lanes.append("TIME_OF_DAY_RVOL")
            if pressure >= 0.12:
                score += 0.5
                lanes.append("PMC_GREEN_PROXY")
            setup = "INVERSE_MOMENTUM"
            stop_atr = 0.85
            target_r = 3.0

        elif regime in {"GRID_RANGE", "CHOPPY"}:
            if symbol not in set(CORE_ETF + STOCKS):
                return None
            bands = _bollinger(closes)
            efficiency = _efficiency(closes, 20)
            rsi = _rsi(closes)
            if bands is None or efficiency is None or rsi is None:
                return None
            lower, middle, upper = bands
            z = (price - vwap) / max(atr, 0.01)
            lower_touch = price <= lower + 0.25 * atr or z <= -1.0
            reversal = (
                float(xs[-1]["close"]) > float(xs[-1]["open"])
                and price > previous_close
                and close_location >= 0.60
                and pressure >= -0.05
            )
            if (
                not lower_touch
                or not reversal
                or efficiency > 0.32
                or rsi > 48
                or z < -2.5
            ):
                return None
            score = 2.8
            lanes = ["BOUNDED_RANGE_EDGE", "REVERSAL_CONFIRMATION"]
            if price <= lower + 0.25 * atr:
                score += 0.8
                lanes.append("LOWER_BOLLINGER_TOUCH")
            if rvol >= 0.90:
                score += 0.5
                lanes.append("VOLUME_NOT_COLLAPSING")
            if close_location >= 0.72 and pressure >= 0.05:
                score += 0.6
                lanes.append("LIQUIDITY_RECLAIM_PROXY")
            setup = "RANGE"
            stop_atr = 0.80
            target_r = 1.25
        else:
            return None

        anchor = _last_stamp(data)
        sentiment, fresh_sentiment, extreme_sentiment = _social(
            data, symbol, anchor
        )
        if fresh_sentiment and sentiment is not None:
            if extreme_sentiment:
                # Extremes are a crowding penalty, not a forced liquidation.
                score -= 0.35
                lanes.append("SENTIMENT_CROWDING_PENALTY")
            elif sentiment >= 0.55:
                score += 0.20
                lanes.append("SENTIMENT_CONFIRMATION")
            elif sentiment <= 0.35 and symbol not in INVERSE:
                score -= 0.20
                lanes.append("SENTIMENT_HEADWIND")
        if _recent_insider_sale(data, symbol, anchor) and symbol not in INVERSE:
            score -= 0.25
            lanes.append("RECENT_INSIDER_SALE_PENALTY")

        minimum_score = 5.0 if regime == "BULL_TREND" else 4.5
        # Reject signals whose theoretical target is too small to overcome a
        # conservative round-trip spread/slippage/fee floor.
        expected_move = target_r * stop_atr * atr_pct
        if expected_move < 3.0 * _estimated_cost_pct(symbol):
            return None
        if score < minimum_score:
            return None
        return {
            "symbol": symbol,
            "score": score,
            "lanes": lanes,
            "setup": setup,
            "price": price,
            "atr": atr,
            "atr_pct": atr_pct,
            "rvol": rvol,
            "pressure": pressure,
            "stop_atr": stop_atr,
            "target_r": target_r,
            "history_scale": ctx.get("history_scale", 1.0),
            # Cross-sectional MVA-Alpha features used for final ranking.
            "momentum_raw": 0.20 * roc5 + 0.35 * roc15 + 0.45 * roc20,
            "alpha_raw": 0.50 * alpha15 + 0.50 * alpha20,
            "volume_raw": _clamp(rvol, 0.0, 3.0),
            "leveraged": symbol in LEVERAGED,
            "family": _family(symbol),
        }

    def _allocation(self, candidate, regime):
        risk_distance = max(candidate["stop_atr"] * candidate["atr_pct"], 0.003)
        score_scale = _clamp(candidate["score"] / 7.0, 0.55, 1.0)
        weight = RISK_PER_TRADE / risk_distance
        weight *= score_scale
        if candidate["leveraged"]:
            cap = MAX_LEVERAGED_ALLOCATION
        elif candidate["setup"] == "RANGE":
            cap = MAX_GRID_ALLOCATION
        else:
            cap = MAX_NORMAL_ALLOCATION
        if regime == "CHOPPY":
            weight *= 0.65
        if candidate["setup"] == "INVERSE_MOMENTUM":
            weight *= 0.80
        weight *= candidate.get("history_scale", 1.0)
        return _clamp(weight, 0.0, cap)

    def _held_allocations(self, holdings):
        result = {}
        for symbol, value in (holdings or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            # Surmount examples use decimal portfolio weights. Do not mistake
            # share counts from a broker adapter for allocations.
            if 0.0 < numeric <= 1.0:
                result[str(symbol).upper()] = numeric
        return result

    def _sync_position_state(self, held, data):
        for symbol in list(self._position_state):
            if symbol not in held:
                del self._position_state[symbol]
        for symbol in held:
            xs = _bars(data, symbol)
            if not xs:
                continue
            price = float(xs[-1].get("close", 0) or 0)
            if price <= 0:
                continue
            state = self._position_state.get(symbol)
            if state is None:
                self._position_state[symbol] = {
                    "entry": price,
                    "peak": price,
                    "bars": 0,
                    "stop_atr": 1.0,
                    "target_r": 3.0,
                }
            else:
                state["peak"] = max(state["peak"], price)
                state["bars"] += 1

    def _holding_exit(self, symbol, data, ctx, phase):
        if phase in {"CLOSE_WINDOW", "FLATTEN"}:
            return True
        if ctx["regime"] in {"NO_TRADE", "EVENT_RISK", "HIGH_VOL"}:
            return True
        if symbol in DEFENSIVE and ctx["regime"] != "HIGH_VOL":
            return True
        xs = _bars(data, symbol)
        if len(xs) < 30:
            return True
        closes = _values(xs, "close")
        price = closes[-1]
        atr = _atr(xs)
        vwap = _vwap(xs[-78:])
        ema21 = _ema(closes, 21)
        pressure = _pressure(xs, 5)
        if None in (atr, vwap, ema21):
            return True

        state = self._position_state.get(symbol)
        if state is not None:
            stop_mult = state.get(
                "stop_atr", 0.85 if symbol in LEVERAGED else 1.0
            )
            target_mult = state.get(
                "target_r", 2.4 if symbol in LEVERAGED else 3.0
            )
            if price <= state["entry"] - stop_mult * atr:
                return True
            if price >= state["entry"] + target_mult * atr:
                return True
            if (
                state["peak"] >= state["entry"] + 0.8 * atr
                and price <= state["peak"] - 1.1 * atr
            ):
                return True

        if symbol in INVERSE:
            if ctx["regime"] != "BEAR_TREND":
                return True
            return price < ema21 or pressure < -0.12
        if ctx["regime"] == "BULL_TREND":
            return price < vwap and price < ema21 and pressure < -0.10
        if ctx["regime"] in {"GRID_RANGE", "CHOPPY"}:
            bands = _bollinger(closes)
            if bands and price >= min(vwap, bands[1]):
                return True
            return price < ema21 and pressure < -0.15
        return False

    def run(self, data):
        ctx = self._market_context(data)
        phase = self._phase(data)
        holdings = self._held_allocations(data.get("holdings", {}))
        self._sync_position_state(holdings, data)

        # This is an intraday strategy: no overnight carry and no entry during
        # the first noisy bars or the close window.
        if phase in {"UNKNOWN", "PRE_OPEN", "OPEN", "CLOSE_WINDOW", "FLATTEN"}:
            return TargetAllocation({})
        if ctx["regime"] in {"NO_TRADE", "EVENT_RISK"}:
            return TargetAllocation({})
        if ctx["regime"] == "HIGH_VOL":
            # BIL is a cash-like temporary sleeve, not a leveraged trade.
            return (
                TargetAllocation({"BIL": 0.10})
                if len(_bars(data, "BIL")) >= 120
                else TargetAllocation({})
            )

        forced_exit = {
            symbol
            for symbol in holdings
            if self._holding_exit(symbol, data, ctx, phase)
        }
        ranked = []
        for symbol in self.tickers:
            candidate = self._candidate(symbol, data, ctx, phase)
            if candidate is not None:
                ranked.append(candidate)

        # MVA-Alpha-style cross-sectional rank: momentum 40%, volume 30%,
        # alpha 30%. Absolute gates are evaluated above; this rank only
        # chooses the strongest eligible symbols at the current timestamp.
        if ranked:
            for field in ("momentum_raw", "volume_raw", "alpha_raw"):
                ordered = sorted(ranked, key=lambda item: item[field])
                denominator = max(len(ordered) - 1, 1)
                for index, item in enumerate(ordered):
                    item[field + "__rank"] = index / float(denominator)
            for item in ranked:
                rank_score = (
                    0.40 * item["momentum_raw__rank"]
                    + 0.30 * item["volume_raw__rank"]
                    + 0.30 * item["alpha_raw__rank"]
                )
                item["rank_score"] = rank_score
                item["score"] += rank_score
        ranked.sort(
            key=lambda item: (item.get("rank_score", 0.0), item["score"]),
            reverse=True,
        )

        selected = []
        used_families = set()
        for candidate in ranked:
            if candidate["family"] in used_families:
                continue
            selected.append(candidate)
            used_families.add(candidate["family"])
            if len(selected) >= 1:
                break

        allocation = {
            symbol: value
            for symbol, value in holdings.items()
            if symbol not in forced_exit and symbol in self.tickers
        }
        # Do not rotate an open position simply because another symbol ranked
        # slightly higher.  Surmount cannot guarantee broker lifecycle state;
        # Athena must enforce the authoritative stop/target/time-stop.
        open_symbols = [s for s in allocation if s not in forced_exit]
        if open_symbols:
            selected = [c for c in selected if c["symbol"] in open_symbols]
        for candidate in selected:
            symbol = candidate["symbol"]
            if symbol in forced_exit:
                continue
            target = self._allocation(candidate, ctx["regime"])
            current = holdings.get(symbol)
            if current is not None and abs(current - target) < REBALANCE_THRESHOLD:
                target = current
            allocation[symbol] = target
            if current is not None and symbol in self._position_state:
                self._position_state[symbol]["stop_atr"] = candidate["stop_atr"]
                self._position_state[symbol]["target_r"] = candidate["target_r"]

        total = sum(max(value, 0.0) for value in allocation.values())
        if total > MAX_TOTAL_ALLOCATION:
            scale = MAX_TOTAL_ALLOCATION / total
            allocation = {
                symbol: value * scale for symbol, value in allocation.items()
            }
        return TargetAllocation({
            symbol: value for symbol, value in allocation.items() if value > 0
        })