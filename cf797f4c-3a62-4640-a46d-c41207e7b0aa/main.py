from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
from datetime import datetime


class TradingStrategy(Strategy):
    """
    ATHENA Adaptive MVA Intraday — Reduced Data Universe

    Market-day playbooks:
      BULL_TREND         -> Momentum and relative-strength entry
      RANGE              -> VWAP mean-reversion entry
      VOLATILITY_REVERSAL-> VWAP-reclaim entry at half risk
      BEAR_TREND         -> Cash
      NO_TRADE           -> Cash

    Designed for Surmount 5-minute backtesting.
    """

    def __init__(self):
        # Reduced universe to improve Surmount data availability.
        self.market_tickers = ["SPY", "QQQ"]
        self.etf_tickers = ["SPY", "QQQ", "IWM", "SMH"]
        self.stock_tickers = ["NVDA", "AMD", "AAPL", "MSFT"]

        self.tickers = list(dict.fromkeys(
            self.market_tickers
            + self.etf_tickers
            + self.stock_tickers
        ))

        # Position state.
        self.active_symbol = None
        self.active_allocation = 0.0
        self.entry_price = None
        self.initial_stop = None
        self.active_stop = None
        self.initial_risk = None
        self.profit_target = None
        self.highest_price = None
        self.breakeven_active = False
        self.trailing_active = False
        self.below_vwap_bars = 0

        # Daily state.
        self.current_day = None
        self.daily_entries = 0
        self.consecutive_losses = 0
        self.daily_realized_r = 0.0
        self.symbol_entries = {}
        self.last_exit_bar = {}

        # Risk settings.
        self.normal_risk_fraction = 0.0025
        self.reversal_risk_fraction = 0.00125
        self.maximum_allocation = 0.20
        self.minimum_allocation = 0.01

        # Trading limits.
        self.maximum_daily_entries = 6
        self.maximum_symbol_entries = 2
        self.cooldown_bars = 6
        self.minimum_score = 70.0
        self.minimum_relative_volume = 1.20

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
    # Data helpers
    # ============================================================

    def get_bars(self, ohlcv, symbol):
        bars = []

        for period in ohlcv:
            if symbol not in period:
                continue

            item = period[symbol]

            try:
                bars.append({
                    "date": item.get("date"),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": float(item.get("volume", 0))
                })
            except Exception:
                continue

        return bars

    def sma(self, values, length):
        if len(values) < length:
            return None

        return sum(values[-length:]) / float(length)

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

    def percentage_return(self, closes, bars_back):
        if len(closes) <= bars_back:
            return None

        previous = closes[-1 - bars_back]

        if previous <= 0:
            return None

        return closes[-1] / previous - 1.0

    def atr(self, bars, length=14):
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

        return (
            sum(true_ranges[-length:])
            / float(length)
        )

    def vwap(self, bars):
        if not bars:
            return None

        latest_day = str(bars[-1]["date"])[:10]
        price_volume = 0.0
        total_volume = 0.0

        for bar in bars:
            if str(bar["date"])[:10] != latest_day:
                continue

            typical_price = (
                bar["high"]
                + bar["low"]
                + bar["close"]
            ) / 3.0

            price_volume += (
                typical_price * bar["volume"]
            )

            total_volume += bar["volume"]

        if total_volume <= 0:
            return None

        return price_volume / total_volume

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

        average_volume = (
            sum(previous_volumes)
            / float(len(previous_volumes))
        )

        if average_volume <= 0:
            return None

        return bars[-1]["volume"] / average_volume

    def swing_low(self, bars, length=5):
        if len(bars) < length + 1:
            return None

        completed = bars[-length - 1:-1]

        if not completed:
            return None

        return min(
            bar["low"] for bar in completed
        )

    def current_timestamp(self, ohlcv):
        if not ohlcv:
            return None

        latest = ohlcv[-1]

        for symbol in ["SPY", "QQQ"]:
            if symbol in latest:
                return latest[symbol].get("date")

        for symbol in self.tickers:
            if symbol in latest:
                return latest[symbol].get("date")

        return None

    def parse_timestamp(self, value):
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

    # ============================================================
    # Symbol metrics
    # ============================================================

    def metrics(self, ohlcv, symbol):
        bars = self.get_bars(ohlcv, symbol)

        if len(bars) < 31:
            return None

        closes = [
            bar["close"] for bar in bars
        ]

        price = closes[-1]
        ema9 = self.ema(closes[-30:], 9)
        ema20 = self.ema(closes[-40:], 20)
        atr14 = self.atr(bars, 14)
        session_vwap = self.vwap(bars)
        rvol = self.relative_volume(bars, 20)

        return_5 = self.percentage_return(
            closes, 1
        )

        return_15 = self.percentage_return(
            closes, 3
        )

        return_30 = self.percentage_return(
            closes, 6
        )

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
            "bars": bars,
            "price": price,
            "open": bars[-1]["open"],
            "high": bars[-1]["high"],
            "low": bars[-1]["low"],
            "volume": bars[-1]["volume"],
            "ema9": ema9,
            "ema20": ema20,
            "atr": atr14,
            "vwap": session_vwap,
            "relative_volume": rvol,
            "return_5": return_5,
            "return_15": return_15,
            "return_30": return_30,
            "swing_low": self.swing_low(bars, 5)
        }

    # ============================================================
    # Market-day classification
    # ============================================================

    def classify_market_day(self, ohlcv):
        spy = self.metrics(ohlcv, "SPY")
        qqq = self.metrics(ohlcv, "QQQ")

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

        market_rvol = (
            spy["relative_volume"]
            + qqq["relative_volume"]
        ) / 2.0

        market_momentum = (
            abs(spy["return_30"])
            + abs(qqq["return_30"])
        ) / 2.0

        market_atr_fraction = (
            spy["atr"] / spy["price"]
            + qqq["atr"] / qqq["price"]
        ) / 2.0

        # Low-volume, low-momentum session.
        if (
            market_rvol < 0.70
            and market_momentum < 0.002
        ):
            return "NO_TRADE"

        # High-volatility VWAP reversal.
        spy_reclaimed_vwap = (
            spy["low"] < spy["vwap"]
            and spy["price"] > spy["vwap"]
        )

        qqq_reclaimed_vwap = (
            qqq["low"] < qqq["vwap"]
            and qqq["price"] > qqq["vwap"]
        )

        if (
            market_atr_fraction > 0.004
            and (
                spy_reclaimed_vwap
                or qqq_reclaimed_vwap
            )
        ):
            return "VOLATILITY_REVERSAL"

        if bull_points >= 5:
            return "BULL_TREND"

        if bear_points >= 5:
            return "BEAR_TREND"

        return "RANGE"

    # ============================================================
    # Ranking and stock selection
    # ============================================================

    def percentile_scores(self, values):
        if not values:
            return {}

        ordered = sorted(
            values.items(),
            key=lambda item: item[1]
        )

        count = len(ordered)
        scores = {}

        for index, item in enumerate(ordered):
            symbol = item[0]

            if count == 1:
                scores[symbol] = 100.0
            else:
                scores[symbol] = (
                    index / float(count - 1)
                ) * 100.0

        return scores

    def rank_momentum_candidates(self, ohlcv):
        spy = self.metrics(ohlcv, "SPY")

        if spy is None:
            return []

        metrics_by_symbol = {}
        momentum_raw = {}
        volume_raw = {}
        alpha_raw = {}

        for symbol in self.tickers:
            item = self.metrics(ohlcv, symbol)

            if item is None:
                continue

            if item["price"] < 5:
                continue

            current_dollar_volume = (
                item["price"] * item["volume"]
            )

            if current_dollar_volume < 500000:
                continue

            momentum = (
                0.20 * item["return_5"]
                + 0.35 * item["return_15"]
                + 0.45 * item["return_30"]
            )

            alpha_15 = (
                item["return_15"]
                - spy["return_15"]
            )

            alpha_30 = (
                item["return_30"]
                - spy["return_30"]
            )

            alpha = (
                0.50 * alpha_15
                + 0.50 * alpha_30
            )

            item["alpha_15"] = alpha_15
            item["alpha_30"] = alpha_30

            metrics_by_symbol[symbol] = item
            momentum_raw[symbol] = momentum
            volume_raw[symbol] = (
                item["relative_volume"]
            )
            alpha_raw[symbol] = alpha

        momentum_scores = self.percentile_scores(
            momentum_raw
        )

        volume_scores = self.percentile_scores(
            volume_raw
        )

        alpha_scores = self.percentile_scores(
            alpha_raw
        )

        candidates = []

        for symbol, item in metrics_by_symbol.items():
            # OHLCV does not provide bid-ask spread;
            # eligible liquid symbols receive a fixed score.
            liquidity_score = 100.0

            composite_score = (
                0.35 * momentum_scores.get(
                    symbol, 0
                )
                + 0.25 * volume_scores.get(
                    symbol, 0
                )
                + 0.25 * alpha_scores.get(
                    symbol, 0
                )
                + 0.15 * liquidity_score
            )

            item["score"] = composite_score

            qualifies = (
                composite_score
                >= self.minimum_score
                and item["relative_volume"]
                >= self.minimum_relative_volume
                and item["return_5"] > 0
                and item["return_15"] > 0
                and item["return_30"] > 0
                and item["alpha_15"] >= 0
                and item["alpha_30"] >= 0
                and item["price"] > item["vwap"]
                and item["ema9"] > item["ema20"]
                and item["price"] > item["open"]
            )

            if qualifies:
                candidates.append(item)

        return sorted(
            candidates,
            key=lambda candidate: candidate["score"],
            reverse=True
        )

    def range_candidate(self, ohlcv):
        best = None
        best_score = None

        for symbol in [
            "SPY", "QQQ", "IWM", "SMH"
        ]:
            item = self.metrics(ohlcv, symbol)

            if item is None:
                continue

            distance_below_vwap = (
                item["vwap"] - item["price"]
            ) / item["price"]

            bullish_reversal_bar = (
                item["price"] > item["open"]
                and item["return_5"] > 0
            )

            qualifies = (
                distance_below_vwap >= 0.0015
                and bullish_reversal_bar
                and item["relative_volume"] >= 0.80
            )

            if not qualifies:
                continue

            score = (
                distance_below_vwap * 10000
                + item["relative_volume"] * 10
            )

            item["score"] = score

            if (
                best_score is None
                or score > best_score
            ):
                best = item
                best_score = score

        return best

    def reversal_candidate(self, ohlcv):
        ranked = self.rank_momentum_candidates(
            ohlcv
        )

        for item in ranked:
            reclaimed_vwap = (
                item["low"] < item["vwap"]
                and item["price"] > item["vwap"]
            )

            if reclaimed_vwap:
                return item

        # If no stock qualifies, test SPY and QQQ.
        for symbol in ["SPY", "QQQ"]:
            item = self.metrics(ohlcv, symbol)

            if item is None:
                continue

            if (
                item["low"] < item["vwap"]
                and item["price"] > item["vwap"]
                and item["price"] > item["open"]
            ):
                return item

        return None

    # ============================================================
    # Entry and risk calculation
    # ============================================================

    def calculate_setup(
        self,
        candidate,
        risk_fraction
    ):
        entry = candidate["price"]
        atr_value = candidate["atr"]
        recent_swing_low = candidate["swing_low"]

        if recent_swing_low is None:
            return None

        atr_stop = entry - atr_value

        structure_stop = (
            recent_swing_low
            - 0.10 * atr_value
        )

        # Use the tighter valid stop.
        stop = max(
            atr_stop,
            structure_stop
        )

        risk_per_share = entry - stop

        if risk_per_share <= 0:
            return None

        stop_fraction = (
            risk_per_share / entry
        )

        # Avoid excessively tight or wide stops.
        if stop_fraction < 0.003:
            return None

        if stop_fraction > 0.02:
            return None

        allocation = (
            risk_fraction / stop_fraction
        )

        allocation = min(
            allocation,
            self.maximum_allocation
        )

        if allocation < self.minimum_allocation:
            return None

        return {
            "entry": entry,
            "stop": stop,
            "risk": risk_per_share,
            "target": (
                entry + 3.0 * risk_per_share
            ),
            "allocation": allocation
        }

    def open_position(
        self,
        candidate,
        setup,
        market_day
    ):
        symbol = candidate["symbol"]

        self.active_symbol = symbol
        self.active_allocation = (
            setup["allocation"]
        )
        self.entry_price = setup["entry"]
        self.initial_stop = setup["stop"]
        self.active_stop = setup["stop"]
        self.initial_risk = setup["risk"]
        self.profit_target = setup["target"]
        self.highest_price = setup["entry"]
        self.breakeven_active = False
        self.trailing_active = False
        self.below_vwap_bars = 0

        self.daily_entries += 1

        self.symbol_entries[symbol] = (
            self.symbol_entries.get(symbol, 0)
            + 1
        )

        log(
            "ENTRY"
            + " symbol=" + symbol
            + " regime=" + market_day
            + " score="
            + str(round(
                candidate.get("score", 0), 2
            ))
            + " entry="
            + str(round(self.entry_price, 4))
            + " stop="
            + str(round(self.initial_stop, 4))
            + " target="
            + str(round(self.profit_target, 4))
            + " allocation="
            + str(round(
                self.active_allocation, 4
            ))
        )

        return TargetAllocation({
            symbol: self.active_allocation
        })

    # ============================================================
    # Exit and position management
    # ============================================================

    def close_position(
        self,
        reason,
        exit_price,
        bar_number
    ):
        symbol = self.active_symbol
        realized_r = 0.0

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
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if symbol is not None:
            self.last_exit_bar[symbol] = (
                bar_number
            )

        log(
            "EXIT"
            + " symbol=" + str(symbol)
            + " reason=" + reason
            + " exit="
            + str(round(exit_price, 4))
            + " result_R="
            + str(round(realized_r, 2))
        )

        self.active_symbol = None
        self.active_allocation = 0.0
        self.entry_price = None
        self.initial_stop = None
        self.active_stop = None
        self.initial_risk = None
        self.profit_target = None
        self.highest_price = None
        self.breakeven_active = False
        self.trailing_active = False
        self.below_vwap_bars = 0

        return TargetAllocation({})

    def manage_position(
        self,
        ohlcv,
        market_day,
        current_time,
        bar_number
    ):
        item = self.metrics(
            ohlcv,
            self.active_symbol
        )

        if item is None:
            log(
                "Position data unavailable for "
                + str(self.active_symbol)
            )

            # Maintain the current allocation rather
            # than submitting an unverified trade.
            return TargetAllocation({
                self.active_symbol:
                self.active_allocation
            })

        price = item["price"]
        bar_low = item["low"]
        bar_high = item["high"]

        self.highest_price = max(
            self.highest_price,
            bar_high
        )

        # Conservative same-bar handling:
        # if both stop and target are touched,
        # treat the protective stop as occurring first.
        if bar_low <= self.active_stop:
            return self.close_position(
                "PROTECTIVE_STOP",
                self.active_stop,
                bar_number
            )

        if bar_high >= self.profit_target:
            return self.close_position(
                "PROFIT_TARGET_3R",
                self.profit_target,
                bar_number
            )

        one_r_price = (
            self.entry_price
            + self.initial_risk
        )

        two_r_price = (
            self.entry_price
            + 2.0 * self.initial_risk
        )

        # Move stop to breakeven after +1R.
        if bar_high >= one_r_price:
            self.breakeven_active = True

            self.active_stop = max(
                self.active_stop,
                self.entry_price
            )

        # Begin profit trailing after +2R.
        if bar_high >= two_r_price:
            self.trailing_active = True

        if self.trailing_active:
            recent_swing_low = (
                item["swing_low"]
            )

            if recent_swing_low is not None:
                ema_trailing_stop = (
                    item["ema9"]
                    - 0.25 * item["atr"]
                )

                structure_trailing_stop = (
                    recent_swing_low
                    - 0.10 * item["atr"]
                )

                proposed_stop = max(
                    ema_trailing_stop,
                    structure_trailing_stop
                )

                # A protective stop can move upward,
                # but it can never move downward.
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

        if item["relative_volume"] < 0.80:
            deterioration += 1

        if deterioration >= 2:
            return self.close_position(
                "SIGNAL_DETERIORATION",
                price,
                bar_number
            )

        # Exit normal long positions when a bear
        # market-day classification is confirmed.
        if market_day == "BEAR_TREND":
            return self.close_position(
                "BEAR_REGIME",
                price,
                bar_number
            )

        # Mandatory end-of-day liquidation.
        minutes = (
            current_time.hour * 60
            + current_time.minute
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

        if not ohlcv:
            return TargetAllocation({})

        # At least 31 five-minute observations
        # are required for the indicators.
        if len(ohlcv) < 31:
            return TargetAllocation({})

        timestamp_value = self.current_timestamp(
            ohlcv
        )

        current_time = self.parse_timestamp(
            timestamp_value
        )

        if current_time is None:
            log(
                "Timestamp unavailable; remaining in cash"
            )
            return TargetAllocation({})

        today = current_time.date().isoformat()
        bar_number = len(ohlcv)

        # Prevent overnight holdings.
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
                "NEW_DAY_SAFETY_EXIT",
                exit_price,
                bar_number
            )

        # Reset daily limits.
        if self.current_day != today:
            self.current_day = today
            self.daily_entries = 0
            self.consecutive_losses = 0
            self.daily_realized_r = 0.0
            self.symbol_entries = {}

        market_day = (
            self.classify_market_day(ohlcv)
        )

        log(
            "MARKET_DAY"
            + " type=" + market_day
            + " time=" + str(current_time)
        )

        # Manage an existing position before
        # considering any new entry.
        if self.active_symbol is not None:
            return self.manage_position(
                ohlcv,
                market_day,
                current_time,
                bar_number
            )

        minutes = (
            current_time.hour * 60
            + current_time.minute
        )

        # Entry period:
        # 9:45 AM through 3:00 PM.
        if minutes < 9 * 60 + 45:
            return TargetAllocation({})

        if minutes > 15 * 60:
            return TargetAllocation({})

        # Daily risk controls.
        if (
            self.daily_entries
            >= self.maximum_daily_entries
        ):
            return TargetAllocation({})

        if self.consecutive_losses >= 3:
            return TargetAllocation({})

        # Approximately four full-risk losses.
        if self.daily_realized_r <= -4.0:
            return TargetAllocation({})

        # Daily profit lock.
        if self.daily_realized_r >= 8.0:
            return TargetAllocation({})

        # Bear and no-trade days remain in cash
        # in this reduced-universe version.
        if market_day in [
            "BEAR_TREND",
            "NO_TRADE"
        ]:
            return TargetAllocation({})

        candidate = None
        risk_fraction = (
            self.normal_risk_fraction
        )

        if market_day == "BULL_TREND":
            candidates = (
                self.rank_momentum_candidates(
                    ohlcv
                )
            )

            if candidates:
                candidate = candidates[0]

        elif market_day == "RANGE":
            candidate = self.range_candidate(
                ohlcv
            )

        elif (
            market_day
            == "VOLATILITY_REVERSAL"
        ):
            candidate = (
                self.reversal_candidate(ohlcv)
            )

            risk_fraction = (
                self.reversal_risk_fraction
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
            market_day
        )