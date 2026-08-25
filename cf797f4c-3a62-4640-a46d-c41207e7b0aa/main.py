from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


class TradingStrategy(Strategy):

    @property
    def interval(self):
        return "5min"

    @property
    def assets(self):
        return ["SPY"]

    @property
    def data(self):
        return []

    def run(self, data):
        ohlcv = data.get("ohlcv", [])

        log(
            "SPY 5-MINUTE TEST: received "
            + str(len(ohlcv))
            + " bars"
        )

        if not ohlcv:
            return TargetAllocation({})

        latest = ohlcv[-1]

        if "SPY" not in latest:
            log("SPY missing from latest bar")
            return TargetAllocation({})

        price = latest["SPY"].get("close")

        log("Latest SPY close: " + str(price))

        return TargetAllocation({"SPY": 0.50})