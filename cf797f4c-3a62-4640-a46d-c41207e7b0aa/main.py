from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


class TradingStrategy(Strategy):

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        return ["SPY"]

    @property
    def data(self):
        return []

    def run(self, data):
        ohlcv = data.get("ohlcv", [])

        log("SPY TEST: received " + str(len(ohlcv)) + " bars")

        if not ohlcv:
            return TargetAllocation({})

        return TargetAllocation({"SPY": 0.50})