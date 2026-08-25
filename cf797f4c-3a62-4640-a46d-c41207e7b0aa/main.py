from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
from datetime import datetime


class TradingStrategy(Strategy):
    """
    ATHENA Adaptive SPY/QQQ Intraday Strategy

    Day types:
      BULL_TREND          -> Momentum/relative-strength strategy
      RANGE               -> VWAP mean-reversion strategy
      VOLATILITY_REVERSAL -> VWAP-reclaim strategy at reduced risk
      BEAR_TREND          -> Cash
      NO_TRADE            -> Cash

    Risk management:
      Initial stop = tighter valid ATR/structure stop
      Breakeven at +1R
      Trailing stop after +2R
      Full exit at +3R
      End-of-day liquidation
    """

    def __init__(self):
        self.tickers = ["SPY", "QQQ"]

        # Current position.
        self.active_symbol = None
        self.active_allocation = 0.0
        self.entry_price = None
        self.initial_stop = None
        self.active_stop = None
        self.initial_risk = None
        self.target_price = None
        self.highest_price = None
        self.below_vwap_bars = 0
        self.trailing_active = False

        # Daily controls.
        self.current_day = None
        self.daily_entries = 0
        self.consecutive_losses = 0
        self.daily_result_r = 0.0
        self.last_exit_bar = {}
        self.symbol_entries = {}

        # Risk settings.
        self.normal_risk = 0.0025
        self.reversal_risk = 0.00125
        self.maximum_allocation = 0.20
        self.minimum_allocation = 0.01

        # Trading limits.
        self.maximum_daily_entries = 6
        self.maximum_symbol_entries = 2
        self.cooldown_bars = 6

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return []

    # ============================================================
    # Data and indicator functions
    # ============================================================

    def bars(self, ohlcv, symbol):
        result = []

        for period in ohlcv:
            if symbol not in period:
                continue

            item = period[symbol]

            try:
                result.append({
                    "date": item.get("date"),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": float(item.get("volume", 0))
                })
            except Exception:
                continue

        return result

    def ema(self, values, length):
        if len(values) < length:
            return None

        multiplier = 2.0 / (length + 1.0)
        result = values[0]

        for value in values[1:]:
            result = (
                value * multiplier
                + result * (1.0 - multiplier)
            )

        return result

    def percentage_return(self, values, bars_back):
        if len(values) <= bars_back:
            return None

        previous = values[-1 - bars_back]

        if previous <= 0:
            return None

        return values[-1] / previous - 1.0

    def atr(self, bars, length=14):
        if len(bars) < length + 1:
            return None

        ranges = []

        for index in range(1, len(bars)):
            high = bars[index]["high"]
            low = bars[index]["low"]
            previous_close = bars[index - 1]["close"]

            ranges.append(max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close)
            ))

        if len(ranges) < length:
            return None

        return sum(ranges[-length:]) / float(length)

    def vwap(self, bars):
        if not bars:
            return None

        current_date = str(bars[-1]["date"])[:10]
        total_price_volume = 0.0
        total_volume = 0.0

        for bar in bars:
            if str(bar["date"])[:10] != current_date:
                continue

            typical_price = (
                bar["high"]
                + bar["low"]
                + bar["close"]
            ) / 3.0

            total_price_volume += (
                typical_price * bar["volume"]
            )

            total_volume += bar["volume"]

        if total_volume <= 0:
            return None

        return total_price_volume / total_volume

    def relative_volume(self, bars, length=20):
        if len(bars) < length + 1:
            return None

        previous_volumes = [
            bar["volume"]
            for bar in bars[-length - 1:-1]
            if bar["volume"] > 0
        ]

        if len(previous_volumes) < 10:
            return None

        normal_volume = (
            sum(previous_volumes)
            / float(len(previous_volumes))
        )

        if normal_volume <= 0:
            return None

        return bars[-1]["volume"] / normal_volume

    def swing_low(self, bars, length=5):
        if len(bars) < length + 1:
            return None

        return min(
            bar["low"]
            for bar in bars[-length - 1:-1]
        )

    def metrics(self, ohlcv, symbol):
        symbol_bars = self.bars(ohlcv, symbol)

        if len(symbol_bars) < 31:
            return None

        closes = [
            bar["close"] for bar in symbol_bars
        ]

        price = closes[-1]
        ema9 = self.ema(closes[-30:], 9)
        ema20 = self.ema(closes[-40:], 20)
        atr14 = self.atr(symbol_bars, 14)
        session_vwap = self.vwap(symbol_bars)
        rvol = self.relative_volume(symbol_bars)

        return_5 = self.percentage_return(closes, 1)
        return_15 = self.percentage_return(closes, 3)
        return_30 = self.percentage_return(closes, 6)

        required = [
            ema9,
            ema20,
            atr14,
            session_vwap,
            rvol,
            return_5,
            return_15,
            return_30
        ]

        if (
            price <= 0
            or any(value is None for value in required)
        ):
            return None

        return {
            "symbol": symbol,
            "bars": symbol_bars,
            "price": price,
            "open": symbol_bars[-1]["open"],
            "high": symbol_bars[-1]["high"],
            "low": symbol_bars[-1]["low"],
            "volume": symbol_bars[-1]["volume"],
            "ema9": ema9,
            "ema20": ema20,
            "atr": atr14,
            "vwap": session_vwap,
            "rvol": rvol,
            "return_5": return_5,
            "return_15": return_15,
            "return_30": return_30,
            "swing_low": self.swing_low(symbol_bars)
        }

    def current_time(self, ohlcv):
        if not ohlcv:
            return None

        latest = ohlcv[-1]

        for symbol in self.tickers:
            if symbol not in latest:
                continue

            value = latest[symbol].get("date")

            if value is None:
                continue

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
                    continue

        return None

    # ============================================================
    # Market-day classification
    # ============================================================

    def classify_day(self, spy, qqq):
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

        average_rvol = (
            spy["rvol"] + qqq["rvol"]
        ) / 2.0

        average_momentum = (
            abs(spy["return_30"])
            + abs(qqq["return_30"])
        ) / 2.0

        average_atr_fraction = (
            spy["atr"] / spy["price"]
            + qqq["atr"] / qqq["price"]
        ) / 2.0

        if (
            average_rvol < 0.70
            and average_momentum < 0.002
        ):
            return "NO_TRADE"

        spy_reclaim = (
            spy["low"] < spy["vwap"]
            and spy["price"] > spy["vwap"]
        )

        qqq_reclaim = (
            qqq["low"] < qqq["vwap"]
            and qqq["price"] > qqq["vwap"]
        )

        if (
            average_atr_fraction > 0.004
            and (spy_reclaim or qqq_reclaim)
        ):
            return "VOLATILITY_REVERSAL"

        if bull_points >= 5:
            return "BULL_TREND"

        if bear_points >= 5:
            return "BEAR_TREND"

        return "RANGE"

    # ============================================================
    # Candidate selection
    # ============================================================

    def momentum_candidate(self, spy, qqq):
        candidates = []

        for item in [spy, qqq]:
            alpha_15 = (
                item["return_15"]
                - spy["return_15"]
            )

            alpha_30 = (
                item["return_30"]
                - spy["return_30"]
            )

            momentum_score = 0.0

            if item["return_5"] > 0:
                momentum_score += 15

            if item["return_15"] > 0:
                momentum_score += 20

            if item["return_30"] > 0:
                momentum_score += 20

            if item["price"] > item["vwap"]:
                momentum_score += 15

            if item["ema9"] > item["ema20"]:
                momentum_score += 15

            if item["rvol"] >= 1.20:
                momentum_score += 10

            if alpha_15 >= 0 and alpha_30 >= 0:
                momentum_score += 5

            item["score"] = momentum_score
            item["alpha_15"] = alpha_15
            item["alpha_30"] = alpha_30

            qualifies = (
                momentum_score >= 70
                and item["return_5"] > 0
                and item["return_15"] > 0
                and item["return_30"] > 0
                and item["price"] > item["vwap"]
                and item["ema9"] > item["ema20"]
                and item["price"] > item["open"]
            )

            if qualifies:
                candidates.append(item)

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[0]

    def range_candidate(self, spy, qqq):
        candidates = []

        for item in [spy, qqq]:
            distance_below_vwap = (
                item["vwap"] - item["price"]
            ) / item["price"]

            bullish_reversal = (
                item["price"] > item["open"]
                and item["return_5"] > 0
            )

            if (
                distance_below_vwap >= 0.0015
                and bullish_reversal
                and item["rvol"] >= 0.80
            ):
                item["score"] = (
                    distance_below_vwap * 10000
                    + item["rvol"] * 10
                )

                candidates.append(item)

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[0]

    def reversal_candidate(self, spy, qqq):
        candidates = []

        for item in [spy, qqq]:
            reclaimed_vwap = (
                item["low"] < item["vwap"]
                and item["price"] > item["vwap"]
            )

            bullish_confirmation = (
                item["price"] > item["open"]
                and item["return_5"] > 0
            )

            if reclaimed_vwap and bullish_confirmation:
                item["score"] = (
                    item["rvol"] * 20
                    + item["return_15"] * 10000
                )

                candidates.append(item)

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates[0]

    # ============================================================
    # Entry and position sizing
    # ============================================================

    def calculate_setup(self, item, risk_fraction):
        entry = item["price"]
        atr_stop = entry - item["atr"]

        structure_stop = (
            item["swing_low"]
            - 0.10 * item["atr"]
        )

        stop = max(
            atr_stop,
            structure_stop
        )

        risk = entry - stop

        if risk <= 0:
            return None

        stop_fraction = risk / entry

        if stop_fraction < 0.001:
            return None

        if stop_fraction > 0.02:
            return None

        allocation = min(
            risk_fraction / stop_fraction,
            self.maximum_allocation
        )

        if allocation < self.minimum_allocation:
            return None

        return {
            "entry": entry,
            "stop": stop,
            "risk": risk,
            "target": entry + 3.0 * risk,
            "allocation": allocation
        }

    def open_position(self, item, setup, day_type):
        symbol = item["symbol"]

        self.active_symbol = symbol
        self.active_allocation = setup["allocation"]
        self.entry_price = setup["entry"]
        self.initial_stop = setup["stop"]
        self.active_stop = setup["stop"]
        self.initial_risk = setup["risk"]
        self.target_price = setup["target"]
        self.highest_price = setup["entry"]
        self.below_vwap_bars = 0
        self.trailing_active = False

        self.daily_entries += 1

        self.symbol_entries[symbol] = (
            self.symbol_entries.get(symbol, 0)
            + 1
        )

        log(
            "ENTRY"
            + " symbol=" + symbol
            + " day_type=" + day_type
            + " entry=" + str(round(
                self.entry_price, 4
            ))
            + " stop=" + str(round(
                self.initial_stop, 4
            ))
            + " target=" + str(round(
                self.target_price, 4
            ))
            + " allocation=" + str(round(
                self.active_allocation, 4
            ))
        )

        return TargetAllocation({
            symbol: self.active_allocation
        })

    # ============================================================
    # Exit and trade management
    # ============================================================

    def close_position(
        self,
        reason,
        exit_price,
        bar_number
    ):
        symbol = self.active_symbol

        result_r = (
            exit_price - self.entry_price
        ) / self.initial_risk

        self.daily_result_r += result_r

        if result_r < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.last_exit_bar[symbol] = bar_number

        log(
            "EXIT"
            + " symbol=" + symbol
            + " reason=" + reason
            + " exit=" + str(round(
                exit_price, 4
            ))
            + " result_R=" + str(round(
                result_r, 2
            ))
        )

        self.active_symbol = None
        self.active_allocation = 0.0
        self.entry_price = None
        self.initial_stop = None
        self.active_stop = None
        self.initial_risk = None
        self.target_price = None
        self.highest_price = None
        self.below_vwap_bars = 0
        self.trailing_active = False

        return TargetAllocation({})

    def manage_position(
        self,
        ohlcv,
        day_type,
        timestamp,
        bar_number
    ):
        item = self.metrics(
            ohlcv,
            self.active_symbol
        )

        if item is None:
            return TargetAllocation({
                self.active_symbol:
                self.active_allocation
            })

        price = item["price"]
        self.highest_price = max(
            self.highest_price,
            item["high"]
        )

        # Conservative backtest assumption:
        # stop is evaluated before target.
        if item["low"] <= self.active_stop:
            return self.close_position(
                "PROTECTIVE_STOP",
                self.active_stop,
                bar_number
            )

        if item["high"] >= self.target_price:
            return self.close_position(
                "PROFIT_TARGET_3R",
                self.target_price,
                bar_number
            )

        one_r = (
            self.entry_price
            + self.initial_risk
        )

        two_r = (
            self.entry_price
            + 2.0 * self.initial_risk
        )

        # Breakeven after +1R.
        if item["high"] >= one_r:
            self.active_stop = max(
                self.active_stop,
                self.entry_price
            )

        # Trailing stop after +2R.
        if item["high"] >= two_r:
            self.trailing_active = True

        if self.trailing_active:
            ema_stop = (
                item["ema9"]
                - 0.25 * item["atr"]
            )

            structure_stop = (
                item["swing_low"]
                - 0.10 * item["atr"]
            )

            proposed_stop = max(
                ema_stop,
                structure_stop
            )

            self.active_stop = max(
                self.active_stop,
                proposed_stop
            )

        if price < item["vwap"]:
            self.below_vwap_bars += 1
        else:
            self.below_vwap_bars = 0

        deterioration = 0

        if self.below_vwap_bars >= 2:
            deterioration += 1

        if item["ema9"] < item["ema20"]:
            deterioration += 1

        if item["return_15"] < 0:
            deterioration += 1

        if item["rvol"] < 0.80:
            deterioration += 1

        if deterioration >= 2:
            return self.close_position(
                "SIGNAL_DETERIORATION",
                price,
                bar_number
            )

        if day_type == "BEAR_TREND":
            return self.close_position(
                "BEAR_REGIME",
                price,
                bar_number
            )

        minutes = (
            timestamp.hour * 60
            + timestamp.minute
        )

        if minutes >= 15 * 60 + 50:
            return self.close_position(
                "END_OF_DAY",
                price,
                bar_number
            )

        return TargetAllocation({
            self.active_symbol:
            self.active_allocation
        })

    # ============================================================
    # Main Surmount method
    # ============================================================

    def run(self, data):
        ohlcv = data.get("ohlcv", [])

        if len(ohlcv) < 31:
            return TargetAllocation({})

        timestamp = self.current_time(ohlcv)

        if timestamp is None:
            log("Timestamp unavailable")
            return TargetAllocation({})

        today = timestamp.date().isoformat()
        bar_number = len(ohlcv)

        # Prevent overnight positions.
        if (
            self.current_day is not None
            and today != self.current_day
            and self.active_symbol is not None
        ):
            item = self.metrics(
                ohlcv,
                self.active_symbol
            )

            exit_price = (
                item["price"]
                if item is not None
                else self.entry_price
            )

            self.current_day = today

            return self.close_position(
                "NEW_DAY_EXIT",
                exit_price,
                bar_number
            )

        if self.current_day != today:
            self.current_day = today
            self.daily_entries = 0
            self.consecutive_losses = 0
            self.daily_result_r = 0.0
            self.symbol_entries = {}

        spy = self.metrics(ohlcv, "SPY")
        qqq = self.metrics(ohlcv, "QQQ")

        if spy is None or qqq is None:
            return TargetAllocation({})

        day_type = self.classify_day(
            spy,
            qqq
        )

        log(
            "MARKET_DAY"
            + " type=" + day_type
            + " time=" + str(timestamp)
        )

        if self.active_symbol is not None:
            return self.manage_position(
                ohlcv,
                day_type,
                timestamp,
                bar_number
            )

        minutes = (
            timestamp.hour * 60
            + timestamp.minute
        )

        # Only enter from 9:45 AM to 3:00 PM.
        if minutes < 9 * 60 + 45:
            return TargetAllocation({})

        if minutes > 15 * 60:
            return TargetAllocation({})

        # Daily risk limits.
        if (
            self.daily_entries
            >= self.maximum_daily_entries
        ):
            return TargetAllocation({})

        if self.consecutive_losses >= 3:
            return TargetAllocation({})

        if self.daily_result_r <= -4.0:
            return TargetAllocation({})

        if self.daily_result_r >= 8.0:
            return TargetAllocation({})

        if day_type in [
            "BEAR_TREND",
            "NO_TRADE"
        ]:
            return TargetAllocation({})

        candidate = None
        risk_fraction = self.normal_risk

        if day_type == "BULL_TREND":
            candidate = self.momentum_candidate(
                spy,
                qqq
            )

        elif day_type == "RANGE":
            candidate = self.range_candidate(
                spy,
                qqq
            )

        elif day_type == "VOLATILITY_REVERSAL":
            candidate = self.reversal_candidate(
                spy,
                qqq
            )

            risk_fraction = (
                self.reversal_risk
            )

        if candidate is None:
            return TargetAllocation({})

        symbol = candidate["symbol"]

        if (
            self.symbol_entries.get(symbol, 0)
            >= self.maximum_symbol_entries
        ):
            return TargetAllocation({})

        last_exit = self.last_exit_bar.get(
            symbol
        )

        if (
            last_exit is not None
            and bar_number - last_exit
            < self.cooldown_bars
        ):
            return TargetAllocation({})

        setup = self.calculate_setup(
            candidate,
            risk_fraction
        )

        if setup is None:
            return TargetAllocation({})

        return self.open_position(
            candidate,
            setup,
            day_type
        )