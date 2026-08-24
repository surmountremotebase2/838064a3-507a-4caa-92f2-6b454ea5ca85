from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
from datetime import datetime
import math


class TradingStrategy(Strategy):
    """
    ATHENA Adaptive Momentum-Volume-Alpha Intraday Strategy

    Market-day playbooks:
      1. Bull trend: momentum/EMA pullback
      2. Bear trend: inverse ETF momentum
      3. Range day: VWAP mean reversion
      4. High-volatility day: reduced-risk reversal
      5. No-trade day: remain in cash

    This is designed for Surmount backtesting and IBKR paper testing.
    """

    def __init__(self):
        # Benchmark and market-regime instruments.
        self.market_tickers = [
            "SPY", "QQQ", "IWM", "DIA"
        ]

        # Sector ETFs.
        self.etf_tickers = [
            "SPY", "QQQ", "IWM", "DIA",
            "XLK", "XLF", "XLE", "SMH"
        ]

        # Highly liquid equities.
        self.stock_tickers = [
            "NVDA", "AMD", "AAPL", "MSFT",
            "AMZN", "META", "GOOGL", "TSLA",
            "AVGO", "NFLX"
        ]

        # Bear-market instruments.
        self.inverse_tickers = [
            "SH", "PSQ", "RWM"
        ]

        self.tickers = list(dict.fromkeys(
            self.market_tickers
            + self.etf_tickers
            + self.stock_tickers
            + self.inverse_tickers
        ))

        # Position state.
        self.active_symbol = None
        self.entry_price = None
        self.initial_stop = None
        self.active_stop = None
        self.profit_target = None
        self.initial_risk = None
        self.highest_price = None
        self.entry_time = None
        self.entry_day = None
        self.breakeven_activated = False
        self.trailing_activated = False
        self.below_vwap_count = 0

        # Daily state.
        self.current_day = None
        self.daily_entries = 0
        self.daily_losses = 0
        self.daily_realized_r = 0.0
        self.last_exit_bar = {}
        self.symbol_entries = {}

        # Risk configuration.
        self.normal_risk_fraction = 0.0025
        self.reversal_risk_fraction = 0.00125
        self.maximum_allocation = 0.20
        self.minimum_allocation = 0.01

        # Selection configuration.
        self.minimum_score = 70.0
        self.minimum_relative_volume = 1.50
        self.maximum_daily_entries = 6
        self.maximum_symbol_entries = 2
        self.cooldown_bars = 6  # Six 5-minute bars = 30 minutes.

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return []

    # ------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------

    def _bars(self, ohlcv, symbol):
        output = []

        for bar in ohlcv:
            if symbol not in bar:
                continue

            item = bar[symbol]

            try:
                output.append({
                    "date": item.get("date"),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": float(item.get("volume", 0)),
                })
            except Exception:
                continue

        return output

    def _sma(self, values, length):
        if len(values) < length:
            return None

        return sum(values[-length:]) / float(length)

    def _ema(self, values, length):
        if len(values) < length:
            return None

        multiplier = 2.0 / (length + 1.0)
        result = values[0]

        for value in values[1:]:
            result = value * multiplier + result * (1.0 - multiplier)

        return result

    def _returns(self, closes, bars_back):
        if len(closes) <= bars_back:
            return None

        previous = closes[-1 - bars_back]

        if previous <= 0:
            return None

        return closes[-1] / previous - 1.0

    def _atr(self, bars, length=14):
        if len(bars) < length + 1:
            return None

        true_ranges = []

        for index in range(1, len(bars)):
            high = bars[index]["high"]
            low = bars[index]["low"]
            previous_close = bars[index - 1]["close"]

            true_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close)
            )

            true_ranges.append(true_range)

        if len(true_ranges) < length:
            return None

        return sum(true_ranges[-length:]) / float(length)

    def _vwap(self, bars):
        if not bars:
            return None

        latest_date = str(bars[-1]["date"])[:10]
        price_volume = 0.0
        total_volume = 0.0

        for bar in bars:
            if str(bar["date"])[:10] != latest_date:
                continue

            typical_price = (
                bar["high"]
                + bar["low"]
                + bar["close"]
            ) / 3.0

            price_volume += typical_price * bar["volume"]
            total_volume += bar["volume"]

        if total_volume <= 0:
            return None

        return price_volume / total_volume

    def _relative_volume(self, bars):
        if len(bars) < 21:
            return None

        historical_volumes = [
            bar["volume"] for bar in bars[-21:-1]
            if bar["volume"] > 0
        ]

        if len(historical_volumes) < 10:
            return None

        normal_volume = sum(historical_volumes) / len(historical_volumes)

        if normal_volume <= 0:
            return None

        return bars[-1]["volume"] / normal_volume

    def _swing_low(self, bars, length=5):
        if len(bars) < length + 1:
            return None

        # Exclude the currently forming/latest completed signal bar.
        return min(bar["low"] for bar in bars[-length - 1:-1])

    def _bar_number(self, ohlcv):
        return len(ohlcv)

    def _timestamp(self, ohlcv):
        if not ohlcv:
            return None

        for symbol in ["SPY", "QQQ"] + self.tickers:
            if symbol in ohlcv[-1]:
                return ohlcv[-1][symbol].get("date")

        return None

    def _parse_timestamp(self, value):
        if value is None:
            return None

        text = str(value).replace("Z", "+00:00")

        try:
            return datetime.fromisoformat(text)
        except Exception:
            try:
                return datetime.strptime(
                    str(value)[:19],
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                return None

    # ------------------------------------------------------------
    # Indicator package for one security
    # ------------------------------------------------------------

    def _metrics(self, ohlcv, symbol):
        bars = self._bars(ohlcv, symbol)

        if len(bars) < 31:
            return None

        closes = [bar["close"] for bar in bars]
        volumes = [bar["volume"] for bar in bars]

        price = closes[-1]
        ema9 = self._ema(closes[-30:], 9)
        ema20 = self._ema(closes[-40:], 20)
        atr14 = self._atr(bars, 14)
        vwap = self._vwap(bars)
        relative_volume = self._relative_volume(bars)

        return_5 = self._returns(closes, 1)
        return_15 = self._returns(closes, 3)
        return_30 = self._returns(closes, 6)

        if (
            price <= 0
            or ema9 is None
            or ema20 is None
            or atr14 is None
            or vwap is None
            or relative_volume is None
            or return_5 is None
            or return_15 is None
            or return_30 is None
        ):
            return None

        return {
            "symbol": symbol,
            "bars": bars,
            "price": price,
            "open": bars[-1]["open"],
            "high": bars[-1]["high"],
            "low": bars[-1]["low"],
            "volume": volumes[-1],
            "ema9": ema9,
            "ema20": ema20,
            "atr": atr14,
            "vwap": vwap,
            "relative_volume": relative_volume,
            "return_5": return_5,
            "return_15": return_15,
            "return_30": return_30,
            "swing_low": self._swing_low(bars, 5),
        }

    # ------------------------------------------------------------
    # Market-day classification
    # ------------------------------------------------------------

    def _classify_market_day(self, ohlcv):
        spy = self._metrics(ohlcv, "SPY")
        qqq = self._metrics(ohlcv, "QQQ")

        if spy is None or qqq is None:
            return "NO_TRADE"

        bull_points = 0
        bear_points = 0

        if spy["price"] > spy["vwap"]:
            bull_points += 1
        else:
            bear_points += 1

        if qqq["price"] > qqq["vwap"]:
            bull_points += 1
        else:
            bear_points += 1

        if spy["ema9"] > spy["ema20"]:
            bull_points += 1
        else:
            bear_points += 1

        if qqq["ema9"] > qqq["ema20"]:
            bull_points += 1
        else:
            bear_points += 1

        if spy["return_15"] > 0:
            bull_points += 1
        else:
            bear_points += 1

        if qqq["return_15"] > 0:
            bull_points += 1
        else:
            bear_points += 1

        market_relative_volume = (
            spy["relative_volume"] + qqq["relative_volume"]
        ) / 2.0

        spy_atr_fraction = spy["atr"] / spy["price"]
        qqq_atr_fraction = qqq["atr"] / qqq["price"]
        market_volatility = (
            spy_atr_fraction + qqq_atr_fraction
        ) / 2.0

        # Excessively weak volume and no direction.
        if (
            market_relative_volume < 0.70
            and abs(spy["return_30"]) < 0.002
            and abs(qqq["return_30"]) < 0.002
        ):
            return "NO_TRADE"

        # High-volatility reversal environment.
        if (
            market_volatility > 0.006
            and (
                spy["price"] * spy["open"] > 0
                or qqq["price"] * qqq["open"] > 0
            )
        ):
            spy_reversal = (
                spy["low"] < spy["vwap"]
                and spy["price"] > spy["vwap"]
            )

            qqq_reversal = (
                qqq["low"] < qqq["vwap"]
                and qqq["price"] > qqq["vwap"]
            )

            if spy_reversal or qqq_reversal:
                return "VOLATILITY_REVERSAL"

        if bull_points >= 5:
            return "BULL_TREND"

        if bear_points >= 5:
            return "BEAR_TREND"

        return "RANGE"

    # ------------------------------------------------------------
    # Cross-sectional ranking
    # ------------------------------------------------------------

    def _percentile_scores(self, values):
        if not values:
            return {}

        ordered = sorted(values.items(), key=lambda item: item[1])
        count = len(ordered)
        output = {}

        for index, item in enumerate(ordered):
            symbol = item[0]

            if count == 1:
                output[symbol] = 100.0
            else:
                output[symbol] = (
                    index / float(count - 1)
                ) * 100.0

        return output

    def _rank_long_candidates(self, ohlcv):
        spy = self._metrics(ohlcv, "SPY")

        if spy is None:
            return []

        metrics_by_symbol = {}
        momentum_values = {}
        volume_values = {}
        alpha_values = {}

        universe = self.stock_tickers + self.etf_tickers

        for symbol in universe:
            metrics = self._metrics(ohlcv, symbol)

            if metrics is None:
                continue

            # Basic liquidity and eligibility filters.
            if metrics["price"] < 5:
                continue

            if metrics["volume"] * metrics["price"] < 500000:
                continue

            momentum = (
                0.20 * metrics["return_5"]
                + 0.35 * metrics["return_15"]
                + 0.45 * metrics["return_30"]
            )

            alpha = (
                0.50 * (
                    metrics["return_15"]
                    - spy["return_15"]
                )
                + 0.50 * (
                    metrics["return_30"]
                    - spy["return_30"]
                )
            )

            metrics["alpha_15"] = (
                metrics["return_15"]
                - spy["return_15"]
            )

            metrics["alpha_30"] = (
                metrics["return_30"]
                - spy["return_30"]
            )

            metrics_by_symbol[symbol] = metrics
            momentum_values[symbol] = momentum
            volume_values[symbol] = metrics["relative_volume"]
            alpha_values[symbol] = alpha

        momentum_scores = self._percentile_scores(
            momentum_values
        )
        volume_scores = self._percentile_scores(
            volume_values
        )
        alpha_scores = self._percentile_scores(
            alpha_values
        )

        ranked = []

        for symbol, metrics in metrics_by_symbol.items():
            liquidity_score = 100.0

            score = (
                0.35 * momentum_scores.get(symbol, 0)
                + 0.25 * volume_scores.get(symbol, 0)
                + 0.25 * alpha_scores.get(symbol, 0)
                + 0.15 * liquidity_score
            )

            metrics["score"] = score

            eligible = (
                score >= self.minimum_score
                and metrics["relative_volume"]
                >= self.minimum_relative_volume
                and metrics["return_5"] > 0
                and metrics["return_15"] > 0
                and metrics["return_30"] > 0
                and metrics["alpha_15"] > 0
                and metrics["alpha_30"] > 0
                and metrics["price"] > metrics["vwap"]
                and metrics["ema9"] > metrics["ema20"]
                and metrics["price"] > metrics["open"]
            )

            if eligible:
                ranked.append(metrics)

        return sorted(
            ranked,
            key=lambda item: item["score"],
            reverse=True
        )

    def _rank_inverse_candidates(self, ohlcv):
        ranked = []

        for symbol in self.inverse_tickers:
            metrics = self._metrics(ohlcv, symbol)

            if metrics is None:
                continue

            score = 0.0

            if metrics["price"] > metrics["vwap"]:
                score += 25

            if metrics["ema9"] > metrics["ema20"]:
                score += 25

            if metrics["return_15"] > 0:
                score += 20

            if metrics["return_30"] > 0:
                score += 15

            if metrics["relative_volume"] >= 1.20:
                score += 15

            metrics["score"] = score

            if score >= 70:
                ranked.append(metrics)

        return sorted(
            ranked,
            key=lambda item: item["score"],
            reverse=True
        )

    # ------------------------------------------------------------
    # Range and volatility-reversal selection
    # ------------------------------------------------------------

    def _range_candidate(self, ohlcv):
        best = None
        best_score = 0.0

        for symbol in ["SPY", "QQQ", "IWM", "DIA"]:
            metrics = self._metrics(ohlcv, symbol)

            if metrics is None:
                continue

            distance_from_vwap = (
                metrics["vwap"] - metrics["price"]
            ) / metrics["price"]

            reclaiming = (
                metrics["price"] > metrics["open"]
                and metrics["return_5"] > 0
            )

            if (
                distance_from_vwap >= 0.002
                and reclaiming
                and metrics["relative_volume"] >= 0.80
            ):
                score = (
                    distance_from_vwap * 10000
                    + metrics["relative_volume"] * 10
                )

                metrics["score"] = score

                if score > best_score:
                    best = metrics
                    best_score = score

        return best

    def _volatility_reversal_candidate(self, ohlcv):
        candidates = self._rank_long_candidates(ohlcv)

        for candidate in candidates:
            reclaimed_vwap = (
                candidate["low"] < candidate["vwap"]
                and candidate["price"] > candidate["vwap"]
            )

            if reclaimed_vwap:
                return candidate

        return None

    # ------------------------------------------------------------
    # Position risk and lifecycle
    # ------------------------------------------------------------

    def _calculate_entry(self, metrics, risk_fraction):
        price = metrics["price"]
        atr = metrics["atr"]
        swing_low = metrics["swing_low"]

        if swing_low is None:
            return None

        atr_stop = price - atr
        structure_stop = swing_low - 0.10 * atr

        # Use the tighter valid stop.
        stop = max(atr_stop, structure_stop)

        stop_distance = price - stop

        if stop_distance <= 0:
            return None

        stop_fraction = stop_distance / price

        if stop_fraction < 0.003:
            return None

        if stop_fraction > 0.02:
            return None

        allocation = risk_fraction / stop_fraction
        allocation = min(
            allocation,
            self.maximum_allocation
        )

        if allocation < self.minimum_allocation:
            return None

        return {
            "entry": price,
            "stop": stop,
            "risk": stop_distance,
            "target": price + 3.0 * stop_distance,
            "allocation": allocation
        }

    def _open_position(
        self,
        symbol,
        setup,
        market_day,
        timestamp
    ):
        self.active_symbol = symbol
        self.entry_price = setup["entry"]
        self.initial_stop = setup["stop"]
        self.active_stop = setup["stop"]
        self.initial_risk = setup["risk"]
        self.profit_target = setup["target"]
        self.highest_price = setup["entry"]
        self.entry_time = timestamp
        self.entry_day = self.current_day
        self.breakeven_activated = False
        self.trailing_activated = False
        self.below_vwap_count = 0

        self.daily_entries += 1
        self.symbol_entries[symbol] = (
            self.symbol_entries.get(symbol, 0) + 1
        )

        log(
            "ENTRY "
            + symbol
            + " day_type="
            + market_day
            + " entry="
            + str(round(self.entry_price, 4))
            + " stop="
            + str(round(self.initial_stop, 4))
            + " target="
            + str(round(self.profit_target, 4))
            + " allocation="
            + str(round(setup["allocation"], 4))
        )

        return TargetAllocation({
            symbol: setup["allocation"]
        })

    def _close_position(
        self,
        reason,
        exit_price,
        bar_number
    ):
        if (
            self.entry_price is not None
            and self.initial_risk is not None
            and self.initial_risk > 0
        ):
            realized_r = (
                exit_price - self.entry_price
            ) / self.initial_risk

            self.daily_realized_r += realized_r

            if realized_r < 0:
                self.daily_losses += 1

            log(
                "EXIT "
                + str(self.active_symbol)
                + " reason="
                + reason
                + " price="
                + str(round(exit_price, 4))
                + " result_R="
                + str(round(realized_r, 2))
            )

        if self.active_symbol is not None:
            self.last_exit_bar[
                self.active_symbol
            ] = bar_number

        self.active_symbol = None
        self.entry_price = None
        self.initial_stop = None
        self.active_stop = None
        self.profit_target = None
        self.initial_risk = None
        self.highest_price = None
        self.entry_time = None
        self.entry_day = None
        self.breakeven_activated = False
        self.trailing_activated = False
        self.below_vwap_count = 0

        return TargetAllocation({})

    def _manage_position(
        self,
        ohlcv,
        market_day,
        current_time,
        bar_number
    ):
        metrics = self._metrics(
            ohlcv,
            self.active_symbol
        )

        if metrics is None:
            return TargetAllocation({
                self.active_symbol: 0
            })

        price = metrics["price"]
        bar_low = metrics["low"]
        bar_high = metrics["high"]

        self.highest_price = max(
            self.highest_price,
            bar_high
        )

        # Conservative same-bar rule:
        # if both stop and target occur, assume stop happened first.
        if bar_low <= self.active_stop:
            return self._close_position(
                "PROTECTIVE_STOP",
                self.active_stop,
                bar_number
            )

        if bar_high >= self.profit_target:
            return self._close_position(
                "TARGET_3R",
                self.profit_target,
                bar_number
            )

        one_r_price = (
            self.entry_price + self.initial_risk
        )

        two_r_price = (
            self.entry_price
            + 2.0 * self.initial_risk
        )

        # Move to breakeven after +1R.
        if bar_high >= one_r_price:
            self.breakeven_activated = True
            self.active_stop = max(
                self.active_stop,
                self.entry_price
            )

        # Trail after +2R.
        if bar_high >= two_r_price:
            self.trailing_activated = True

        if self.trailing_activated:
            swing_low = metrics["swing_low"]

            if swing_low is not None:
                ema_trail = (
                    metrics["ema9"]
                    - 0.25 * metrics["atr"]
                )

                structure_trail = (
                    swing_low
                    - 0.10 * metrics["atr"]
                )

                new_stop = max(
                    ema_trail,
                    structure_trail
                )

                # Stop can only move upward.
                self.active_stop = max(
                    self.active_stop,
                    new_stop
                )

        if price < metrics["vwap"]:
            self.below_vwap_count += 1
        else:
            self.below_vwap_count = 0

        deterioration_count = 0

        if self.below_vwap_count >= 2:
            deterioration_count += 1

        if metrics["ema9"] < metrics["ema20"]:
            deterioration_count += 1

        if metrics["return_15"] < 0:
            deterioration_count += 1

        if metrics["relative_volume"] < 0.80:
            deterioration_count += 1

        if deterioration_count >= 2:
            return self._close_position(
                "SIGNAL_DETERIORATION",
                price,
                bar_number
            )

        # Exit when the new market regime conflicts with the position.
        if (
            market_day == "BEAR_TREND"
            and self.active_symbol
            not in self.inverse_tickers
        ):
            return self._close_position(
                "MARKET_REGIME_CHANGE",
                price,
                bar_number
            )

        # Exit inverse ETFs when bear regime ends.
        if (
            self.active_symbol in self.inverse_tickers
            and market_day != "BEAR_TREND"
        ):
            return self._close_position(
                "BEAR_REGIME_ENDED",
                price,
                bar_number
            )

        # Same-day liquidation.
        if current_time is not None:
            minutes = (
                current_time.hour * 60
                + current_time.minute
            )

            if minutes >= 15 * 60 + 50:
                return self._close_position(
                    "END_OF_DAY",
                    price,
                    bar_number
                )

        # Preserve the current allocation.
        stop_fraction = (
            self.entry_price - self.initial_stop
        ) / self.entry_price

        if stop_fraction <= 0:
            allocation = self.minimum_allocation
        else:
            allocation = min(
                self.normal_risk_fraction
                / stop_fraction,
                self.maximum_allocation
            )

        return TargetAllocation({
            self.active_symbol: allocation
        })

    # ------------------------------------------------------------
    # Main Surmount execution method
    # ------------------------------------------------------------

    def run(self, data):
        ohlcv = data.get("ohlcv", [])

        if not ohlcv or len(ohlcv) < 40:
            return TargetAllocation({})

        timestamp_value = self._timestamp(ohlcv)
        current_time = self._parse_timestamp(
            timestamp_value
        )
        bar_number = self._bar_number(ohlcv)

        if current_time is None:
            log(
                "Unable to determine bar timestamp; "
                "remaining in cash for safety"
            )
            return TargetAllocation({})

        today = current_time.date().isoformat()

        # Reset daily controls.
        if self.current_day != today:
            self.current_day = today
            self.daily_entries = 0
            self.daily_losses = 0
            self.daily_realized_r = 0.0
            self.symbol_entries = {}

        market_day = self._classify_market_day(
            ohlcv
        )

        log(
            "MARKET_DAY "
            + market_day
            + " time="
            + str(current_time)
        )

        # Existing positions are managed before considering entries.
        if self.active_symbol is not None:
            return self._manage_position(
                ohlcv,
                market_day,
                current_time,
                bar_number
            )

        minutes = (
            current_time.hour * 60
            + current_time.minute
        )

        # Entry window: 9:45 AM through 3:00 PM.
        if minutes < 9 * 60 + 45:
            return TargetAllocation({})

        if minutes > 15 * 60:
            return TargetAllocation({})

        # Daily risk controls.
        if self.daily_entries >= self.maximum_daily_entries:
            return TargetAllocation({})

        if self.daily_losses >= 3:
            return TargetAllocation({})

        if self.daily_realized_r <= -4.0:
            return TargetAllocation({})

        if self.daily_realized_r >= 8.0:
            return TargetAllocation({})

        if market_day == "NO_TRADE":
            return TargetAllocation({})

        selected = None
        risk_fraction = self.normal_risk_fraction

        # Bull-trend momentum/EMA strategy.
        if market_day == "BULL_TREND":
            ranked = self._rank_long_candidates(
                ohlcv
            )

            if ranked:
                selected = ranked[0]

        # Bear trend: inverse ETFs only.
        elif market_day == "BEAR_TREND":
            ranked = self._rank_inverse_candidates(
                ohlcv
            )

            if ranked:
                selected = ranked[0]

        # Range day: VWAP mean-reversion candidate.
        elif market_day == "RANGE":
            selected = self._range_candidate(
                ohlcv
            )

        # Volatility reversal: half-normal risk.
        elif market_day == "VOLATILITY_REVERSAL":
            selected = (
                self._volatility_reversal_candidate(
                    ohlcv
                )
            )

            risk_fraction = (
                self.reversal_risk_fraction
            )

        if selected is None:
            return TargetAllocation({})

        symbol = selected["symbol"]

        # Maximum two entries per symbol daily.
        if (
            self.symbol_entries.get(symbol, 0)
            >= self.maximum_symbol_entries
        ):
            return TargetAllocation({})

        # Thirty-minute symbol cooldown.
        last_exit = self.last_exit_bar.get(symbol)

        if (
            last_exit is not None
            and bar_number - last_exit
            < self.cooldown_bars
        ):
            return TargetAllocation({})

        setup = self._calculate_entry(
            selected,
            risk_fraction
        )

        if setup is None:
            return TargetAllocation({})

        return self._open_position(
            symbol,
            setup,
            market_day,
            timestamp_value
        )