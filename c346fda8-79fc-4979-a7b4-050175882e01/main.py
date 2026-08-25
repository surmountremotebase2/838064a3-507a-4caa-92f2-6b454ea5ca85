"""SAJO-GQX / THOMAS -- Surmount adapter (long-only ETF version).

This is a *portfolio-allocation* adapter for Surmount.  It is deliberately
smaller than the full ATHENA system:

* Surmount's Strategy API returns target allocations, not bracket orders.
* The asset universe must be declared up front, so the dynamic ATHENA stock
  screener is represented here by a curated, liquid ETF candidate universe.
* Futures, options, gamma/order-flow data, Tradovate routing, and the ATHENA
  operator-approval queue are not implemented by this file.

Use this in Surmount backtests or a Surmount paper account only.  It does not
submit broker orders directly; Surmount acts on the TargetAllocation returned
by ``run``.
"""

from math import sqrt

from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


def _number(value, default=0.0):
    """Return a finite float without requiring pandas/numpy."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value and abs(value) != float("inf") else default


def _series(ohlcv, ticker):
    """Extract Surmount's ``[{ticker: {open, high, low, close, volume}}]``."""
    out = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for row in ohlcv or []:
        if not isinstance(row, dict):
            continue
        bar = row.get(ticker)
        if not isinstance(bar, dict):
            continue
        close = _number(bar.get("close"), None)
        if close is None or close <= 0:
            continue
        out["open"].append(_number(bar.get("open"), close))
        out["high"].append(_number(bar.get("high"), close))
        out["low"].append(_number(bar.get("low"), close))
        out["close"].append(close)
        out["volume"].append(max(0.0, _number(bar.get("volume"), 0.0)))
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


def _pct_change(values, length):
    if len(values) <= length or values[-length - 1] == 0:
        return None
    return values[-1] / values[-length - 1] - 1.0


def _rsi(values, length=14):
    if len(values) <= length:
        return None
    window = values[-(length + 1):]
    gains = []
    losses = []
    for before, after in zip(window, window[1:]):
        change = after - before
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    average_gain = sum(gains) / float(length)
    average_loss = sum(losses) / float(length)
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _atr(series, length=14):
    closes = series["close"]
    highs = series["high"]
    lows = series["low"]
    if len(closes) <= length:
        return None
    true_ranges = []
    start = max(1, len(closes) - length)
    for index in range(start, len(closes)):
        true_ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        )
    return sum(true_ranges) / float(len(true_ranges)) if true_ranges else None


def _stdev(values, length):
    if len(values) < length:
        return None
    window = values[-length:]
    mean = sum(window) / float(length)
    variance = sum((value - mean) ** 2 for value in window) / float(length)
    return sqrt(max(variance, 0.0))


def _volume_ratio(values, length=20):
    if len(values) <= length:
        return None
    baseline = _sma(values[:-1], length)
    if baseline in (None, 0):
        return None
    return values[-1] / baseline


def _flat(tickers):
    return {ticker: 0.0 for ticker in tickers}


class TradingStrategy(Strategy):
    """A conservative, long-only SAJO-GQX implementation for Surmount.

    Regimes:
      * BULL_TREND: relative-strength/trend leaders, at most two positions.
      * RANGE: one oversold mean-reversion ETF, at a smaller allocation.
      * BEAR_TREND/NO_TRADE/RISK_OFF: flat.
    """

    BENCHMARK = "SPY"
    CANDIDATES = (
        "QQQ",
        "IWM",
        "SMH",
        "XLK",
        "XLF",
        "XLE",
        "XLI",
        "XLP",
    )
    BAR_INTERVAL = "5min"
    MIN_BARS = 60
    MAX_POSITIONS = 2
    MAX_GROSS_ALLOCATION = 0.50
    RANGE_ALLOCATION = 0.15
    BULL_SCORE_THRESHOLD = 60.0

    @property
    def interval(self):
        # Supported by the Surmount Strategy API and aligned with the ATHENA
        # intraday decision cadence used in the supplied log.
        return self.BAR_INTERVAL

    @property
    def assets(self):
        # Surmount requires assets to be declared before run() is called.
        return [self.BENCHMARK] + list(self.CANDIDATES)

    @property
    def data(self):
        # This adapter uses only the OHLCV stream supplied by Surmount.
        return []

    def _bull_score(self, candidate, benchmark):
        closes = candidate["close"]
        if len(closes) < self.MIN_BARS:
            return None

        close = closes[-1]
        ema20 = _ema(closes, 20)
        sma50 = _sma(closes, 50)
        rsi14 = _rsi(closes, 14)
        momentum5 = _pct_change(closes, 5)
        momentum20 = _pct_change(closes, 20)
        relative_strength = None
        benchmark_momentum = _pct_change(benchmark["close"], 20)
        if momentum20 is not None and benchmark_momentum is not None:
            relative_strength = momentum20 - benchmark_momentum
        volume_ratio = _volume_ratio(candidate["volume"], 20)
        atr14 = _atr(candidate, 14)

        if None in (ema20, sma50, rsi14, momentum5, momentum20, relative_strength):
            return None
        if atr14 is None or close <= ema20 - 1.5 * atr14:
            # A deterministic volatility stop.  TargetAllocation has no
            # bracket-order primitive, so the strategy goes flat on the next
            # rebalance when this condition is met.
            return None

        score = 0.0
        score += 25.0 if close > ema20 else 0.0
        score += 25.0 if ema20 > sma50 else 0.0
        if relative_strength > 0:
            score += min(20.0, 10.0 + relative_strength * 500.0)
        if 50.0 <= rsi14 <= 72.0:
            score += 15.0
        if volume_ratio is not None and volume_ratio >= 1.0:
            score += 10.0
        if 0.002 <= momentum5 <= 0.03:
            score += 5.0
        return score

    def _range_score(self, candidate, benchmark):
        closes = candidate["close"]
        if len(closes) < self.MIN_BARS:
            return None
        close = closes[-1]
        middle = _sma(closes, 20)
        deviation = _stdev(closes, 20)
        sma50 = _sma(closes, 50)
        rsi14 = _rsi(closes, 14)
        if None in (middle, deviation, sma50, rsi14):
            return None
        lower_band = middle - 2.0 * deviation
        # Avoid catching a falling knife: range entries must remain near the
        # longer trend instead of being deeply below it.
        if close > lower_band or rsi14 >= 35.0 or close < 0.95 * sma50:
            return None
        return max(0.0, min(100.0, 100.0 - rsi14))

    def run(self, data):
        ohlcv = data.get("ohlcv") or []
        allocation = _flat(self.assets)
        benchmark = _series(ohlcv, self.BENCHMARK)
        benchmark_closes = benchmark["close"]

        if len(benchmark_closes) < self.MIN_BARS:
            log("NO_TRADE reason=insufficient_history")
            return TargetAllocation(allocation)

        spy_close = benchmark_closes[-1]
        spy_ema20 = _ema(benchmark_closes, 20)
        spy_sma50 = _sma(benchmark_closes, 50)
        spy_return5 = _pct_change(benchmark_closes, 5)
        spy_return20 = _pct_change(benchmark_closes, 20)
        if None in (spy_ema20, spy_sma50, spy_return5, spy_return20):
            log("NO_TRADE reason=benchmark_indicators_unavailable")
            return TargetAllocation(allocation)

        # Market-wide circuit breaker.  This is evaluated before any entry.
        if spy_return5 <= -0.025:
            regime = "RISK_OFF"
        elif spy_close > spy_sma50 and spy_ema20 > spy_sma50 and spy_return20 > 0.005:
            regime = "BULL_TREND"
        elif spy_close < spy_sma50 and spy_ema20 < spy_sma50 and spy_return20 < -0.005:
            regime = "BEAR_TREND"
        elif abs(spy_close / spy_sma50 - 1.0) <= 0.012 or abs(spy_return20) < 0.02:
            regime = "RANGE"
        else:
            regime = "NO_TRADE"

        if regime == "BULL_TREND":
            ranked = []
            for ticker in self.CANDIDATES:
                candidate = _series(ohlcv, ticker)
                score = self._bull_score(candidate, benchmark)
                if score is not None and score >= self.BULL_SCORE_THRESHOLD:
                    ranked.append((score, ticker))
            ranked.sort(reverse=True)
            selected = ranked[: self.MAX_POSITIONS]
            if selected:
                weight = min(
                    self.MAX_GROSS_ALLOCATION / float(len(selected)),
                    1.0 / float(len(selected)),
                )
                for score, ticker in selected:
                    allocation[ticker] = weight
                log(
                    "MARKET_DAY type=BULL_TREND leaders=%s"
                    % ",".join("%s:%.1f" % (ticker, score) for score, ticker in selected)
                )
            else:
                log("MARKET_DAY type=BULL_TREND setup=NONE")
        elif regime == "RANGE":
            ranked = []
            for ticker in self.CANDIDATES:
                candidate = _series(ohlcv, ticker)
                score = self._range_score(candidate, benchmark)
                if score is not None:
                    ranked.append((score, ticker))
            ranked.sort(reverse=True)
            if ranked:
                score, ticker = ranked[0]
                allocation[ticker] = self.RANGE_ALLOCATION
                log("MARKET_DAY type=RANGE entry=%s score=%.1f" % (ticker, score))
            else:
                log("MARKET_DAY type=RANGE setup=NONE")
        else:
            log("MARKET_DAY type=%s allocation=FLAT" % regime)

        return TargetAllocation(allocation)