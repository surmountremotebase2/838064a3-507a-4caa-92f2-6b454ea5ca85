from surmount.base_class import Strategy, TargetAllocation


# Equity/ETF regime-rotation strategy for Surmount Code Builder.
# Surmount returns allocations, so exits are represented by allocation 0.
# This is not a broker-native stop-order implementation.

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "LABU", "LABD",
]
LEVERAGED = {"TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "LABU", "LABD"}


def bars(data, symbol):
    return [x[symbol] for x in data.get("ohlcv", []) if symbol in x]


def closes(xs):
    return [float(x["close"]) for x in xs if x.get("close") is not None]


def volumes(xs):
    return [float(x.get("volume", 0) or 0) for x in xs]


def sma(v, n):
    return sum(v[-n:]) / n if len(v) >= n else None


def ema(v, n):
    if len(v) < n:
        return None
    out = sum(v[:n]) / n
    k = 2.0 / (n + 1.0)
    for x in v[n:]:
        out = x * k + out * (1.0 - k)
    return out


def atr(xs, n=14):
    if len(xs) < n + 1:
        return None
    tr = []
    for a, b in zip(xs[-n - 1:-1], xs[-n:]):
        tr.append(max(float(b["high"]) - float(b["low"]),
                      abs(float(b["high"]) - float(a["close"])),
                      abs(float(b["low"]) - float(a["close"]))))
    return sum(tr) / n


def rsi(v, n=14):
    if len(v) < n + 1:
        return None
    gains, losses = [], []
    for a, b in zip(v[-n - 1:-1], v[-n:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def vwap(xs):
    pv = sum(float(x["close"]) * float(x.get("volume", 0) or 0) for x in xs)
    vol = sum(float(x.get("volume", 0) or 0) for x in xs)
    return pv / vol if vol else None


class TradingStrategy(Strategy):
    def __init__(self):
        self.tickers = UNIVERSE
        self.data_list = []

    @property
    def interval(self):
        # Use 5min for the intraday version. If the Surmount backtester lacks
        # 5-minute history, change only this value to 1hour for validation.
        return "5min"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def _market_regime(self, data):
        p = bars(data, "SPY")
        q = bars(data, "QQQ")
        if len(p) < 50 or len(q) < 50:
            return "NO_TRADE"
        pc, qc = closes(p), closes(q)
        pfast, pslow = ema(pc, 9), ema(pc, 21)
        qfast, qslow = ema(qc, 9), ema(qc, 21)
        pv, qv = vwap(p[-78:]), vwap(q[-78:])
        if None in (pfast, pslow, qfast, qslow, pv, qv):
            return "NO_TRADE"
        trend_up = pc[-1] > pv and qc[-1] > qv and pfast > pslow and qfast > qslow
        trend_dn = pc[-1] < pv and qc[-1] < qv and pfast < pslow and qfast < qslow
        a = atr(p, 14)
        if not a or pc[-1] <= 0:
            return "NO_TRADE"
        vol_ratio = a / pc[-1]
        if vol_ratio > 0.018:
            return "HIGH_VOL"
        if trend_up:
            return "BULL"
        if trend_dn:
            return "BEAR"
        return "RANGE"

    def _score(self, symbol, data, regime):
        xs = bars(data, symbol)
        if len(xs) < 50:
            return None
        c, vol = closes(xs), volumes(xs)
        price = c[-1]
        avg_vol = sma(vol, 20)
        if not price or not avg_vol or avg_vol <= 0:
            return None
        relvol = vol[-1] / avg_vol
        e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
        a, rs = atr(xs), rsi(c)
        vw = vwap(xs[-78:])
        if None in (e9, e21, e50, a, rs, vw) or a <= 0:
            return None
        mom5 = c[-1] / c[-6] - 1.0
        mom20 = c[-1] / c[-21] - 1.0
        direction = 1 if regime == "BULL" else -1 if regime == "BEAR" else 0
        if regime == "BULL" and symbol in {"SQQQ", "SOXS", "SPXU", "LABD"}:
            return None
        if regime == "BEAR" and symbol in {"TQQQ", "SOXL", "UPRO", "LABU"}:
            return None
        if regime == "RANGE" and symbol in LEVERAGED:
            return None
        # Momentum + relative volume + trend alignment; penalize excessive ATR.
        score = (direction * mom5 * 40.0) + (direction * mom20 * 20.0)
        score += min(relvol, 3.0) * 0.8
        score += 1.0 if (direction > 0 and e9 > e21 > e50 and price > vw) else 0
        score += 1.0 if (direction < 0 and e9 < e21 < e50 and price < vw) else 0
        score -= min((a / price) * 100.0, 5.0)
        # Range play: reward distance from VWAP reverting toward it.
        if regime == "RANGE":
            z = (price - vw) / a
            if abs(z) < 0.6:
                return None
            score = -abs(z) + (1.0 if (z < 0 and rs < 45) or (z > 0 and rs > 55) else 0)
        return score

    def run(self, data):
        regime = self._market_regime(data)
        if regime in {"NO_TRADE", "HIGH_VOL"}:
            return TargetAllocation({})
        scored = []
        for symbol in self.tickers:
            s = self._score(symbol, data, regime)
            if s is not None:
                scored.append((s, symbol))
        scored.sort(reverse=True)
        if not scored:
            return TargetAllocation({})

        # Maximum two positions; leveraged ETFs receive one-quarter weight.
        chosen = [x for x in scored if x[0] > 0][:2]
        if not chosen:
            return TargetAllocation({})
        alloc = {}
        raw_total = 0.0
        for score, symbol in chosen:
            weight = 0.35 if symbol in LEVERAGED else 0.50
            weight *= min(1.0, max(0.25, score / 4.0))
            alloc[symbol] = weight
            raw_total += weight
        if raw_total > 0.90:
            scale = 0.90 / raw_total
            alloc = {k: v * scale for k, v in alloc.items()}
        return TargetAllocation(alloc)