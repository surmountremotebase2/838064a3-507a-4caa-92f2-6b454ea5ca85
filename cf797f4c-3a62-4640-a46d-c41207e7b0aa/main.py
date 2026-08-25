from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


class TradingStrategy(Strategy):

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return ["SPY", "QQQ"]

    @property
    def data(self):
        return []

    def run(self, data):
        ohlcv = data.get("ohlcv", [])

        log(
            "SPY/QQQ 5-MINUTE TEST: received "
            + str(len(ohlcv))
            + " bars"
        )

        if not ohlcv:
            return TargetAllocation({})

        latest = ohlcv[-1]

        if "SPY" not in latest:
            log("SPY missing from latest bar")
            return TargetAllocation({})

        if "QQQ" not in latest:
            log("QQQ missing from latest bar")
            return TargetAllocation({})

        spy_close = latest["SPY"].get("close")
        qqq_close = latest["QQQ"].get("close")

        log(
            "SPY=" + str(spy_close)
            + " QQQ=" + str(qqq_close)
        )

        # Diagnostic allocation only.
        return TargetAllocation({
            "SPY": 0.25,
            "QQQ": 0.25
        })