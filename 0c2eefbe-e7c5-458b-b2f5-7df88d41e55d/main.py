"""
Surmount Equity & ETF Strategy - VLP/ILP Inspired
===================================================

Multi-regime strategy combining:
- Momentum continuation (BULL_TREND)
- Mean reversion (GRID_RANGE, CHOPPY)
- Inverse ETF rotation (BEAR_TREND)
- Volatility regime detection
- Order-book and volume confluence

Author: ChatGPT + ATHENA Team
Compatibility: Surmount Code Builder API
Venue: US Equities / ETFs (5-min bars)
Mode: Allocation-based (TargetAllocation output)

CRITICAL: This strategy returns allocations, not orders.
- Stops/targets must be set by ATHENA native engine
- Order flow / PMC validation must be done post-allocation
- Real-time exits require ATHENA risk management overlay
"""

from datetime import datetime
from math import sqrt
from surmount.base_class import Strategy, TargetAllocation
from surmount.data import InsiderTrading, SocialSentiment

# ============================================================================
# UNIVERSE DEFINITION
# ============================================================================

CORE_ETF = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "XBI", "ARKK"]
DEFENSIVE = ["BIL", "SHY", "TLT", "GLD"]
LONG_LEVERAGED = ["TQQQ", "SOXL", "UPRO", "LABU", "TECL"]
INVERSE_LEVERAGED = ["SQQQ", "SOXS", "SPXU", "LABD", "FAZ", "PSQ"]
STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "TSLA", "GOOGL", 
    "AVGO", "MU", "PLTR", "JPM", "XOM"
]

UNIVERSE = CORE_ETF + DEFENSIVE + LONG_LEVERAGED + INVERSE_LEVERAGED + STOCKS + ["VIXY"]
LEVERAGED = set(LONG_LEVERAGED + INVERSE_LEVERAGED)
INVERSE = set(INVERSE_LEVERAGED)

TECH_LONG = {"QQQ", "SMH", "TQQQ", "SOXL", "TECL", "NVDA", "AMD", "AVGO", "MU"}
SP_LONG = {"SPY", "UPRO", "XLK", "DIA", "AAPL", "MSFT", "AMZN", "META", "JPM", "XOM"}

# ============================================================================
# INDICATOR CALCULATION HELPERS
# ============================================================================

def _bars(data, symbol):
    """Extract OHLCV bars for symbol from Surmount data dict."""
    return [row[symbol] for row in data.get("ohlcv", []) if symbol in row]

def _values(xs, key):
    """Extract field from bar list."""
    return [float(x.get(key, 0) or 0) for x in xs]

def _sma(v, n):
    """Simple Moving Average."""
    return sum(v[-n:]) / n if len(v) >= n else None

def _ema(v, n):
    """Exponential Moving Average."""
    if len(v) < n:
        return None
    out = sum(v[:n]) / n
    k = 2.0 / (n + 1.0)
    for item in v[n:]:
        out = k * item + (1.0 - k) * out
    return out

def _atr(xs, n=14):
    """Average True Range."""
    if len(xs) < n + 1:
        return None
    tr = []
    for previous, current in zip(xs[-n - 1:-1], xs[-n:]):
        h, l, pc = float(current["high"]), float(current["low"]), float(previous["close"])
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(tr) / n if tr else None

def _rsi(c, n=14):
    """Relative Strength Index."""
    if len(c) < n + 1:
        return None
    d = [b - a for a, b in zip(c[-n - 1:-1], c[-n:])]
    gains = sum(max(x, 0.0) for x in d) / n
    losses = sum(max(-x, 0.0) for x in d) / n
    if losses == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gains / losses)

def _vwap(xs):
    """Volume Weighted Average Price."""
    volume = sum(float(x.get("volume", 0) or 0) for x in xs)
    if volume == 0:
        return None
    return sum(float(x["close"]) * float(x.get("volume", 0) or 0) for x in xs) / volume

def _rvol(xs, n=20):
    """Relative Volume (current / 20-bar average)."""
    if len(xs) < n + 1:
        return None
    baseline = sum(float(x.get("volume", 0) or 0) for x in xs[-n - 1:-1]) / n
    current_vol = float(xs[-1].get("volume", 0) or 0)
    return current_vol / baseline if baseline > 0 else None

def _roc(c, n):
    """Rate of Change."""
    return c[-1] / c[-n - 1] - 1.0 if len(c) > n and c[-n - 1] != 0 else None

def _bollinger(c, n=20, multiple=2.0):
    """Bollinger Bands (lower, middle, upper)."""
    if len(c) < n:
        return None
    mid = _sma(c, n)
    if mid is None:
        return None
    variance = sum((x - mid) ** 2 for x in c[-n:]) / n
    sd = sqrt(variance)
    return mid - multiple * sd, mid, mid + multiple * sd

def _efficiency(c, n=20):
    """Directional Efficiency (displacement / path length)."""
    if len(c) < n + 1:
        return None
    displacement = abs(c[-1] - c[-n - 1])
    path = sum(abs(b - a) for a, b in zip(c[-n - 1:-1], c[-n:]))
    return displacement / path if path > 0 else 0.0

def _parse_stamp(value):
    """Parse timestamp string to datetime."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

def _last_stamp(data):
    """Get latest timestamp from data dict."""
    rows = data.get("ohlcv", [])
    if not rows:
        return None
    sample = next(iter(rows[-1].values()))
    return _parse_stamp(sample.get("date") or sample.get("datetime") or sample.get("time"))

def _day_key(bar):
    """Extract date key (YYYY-MM-DD) from bar."""
    stamp = _parse_stamp(bar.get("date") or bar.get("datetime") or bar.get("time"))
    return stamp.date().isoformat() if stamp else None

def _session_bars(xs):
    """Split bars into today's and prior day's."""
    if not xs:
        return [], []
    key = _day_key(xs[-1])
    if key is None:
        return xs[-78:], xs[:-78]
    today = [x for x in xs if _day_key(x) == key]
    prior = [x for x in xs if _day_key(x) != key]
    return today, prior

def _social(data, symbol):
    """Extract social sentiment (StockTwits + Twitter average)."""
    rows = data.get(("social_sentiment", symbol), []) or []
    if not rows:
        return None, False
    latest = rows[-1]
    vals = [latest.get("stocktwitsSentiment"), latest.get("twitterSentiment")]
    vals = [float(x) for x in vals if x is not None]
    if not vals:
        return None, False
    value = sum(vals) / len(vals)
    # Extreme sentiment (bullish OR bearish extremes) = event risk
    is_extreme = value < 0.18 or value > 0.82
    return value, is_extreme

def _insider_sale(data, symbol):
    """Check if recent insider sale."""
    rows = data.get(("insider_trading", symbol), []) or []
    return bool(rows and "sale" in str(rows[-1].get("transactionType", "")).lower())

def _market_family(symbol):
    """Classify symbol into market family (for single-family-per-regime rule)."""
    if symbol in TECH_LONG:
        return "NASDAQ"
    if symbol in SP_LONG:
        return "SP500"
    if symbol in DEFENSIVE:
        return "DEFENSIVE"
    return symbol

# ============================================================================
# STRATEGY CLASS
# ============================================================================

class TradingStrategy(Strategy):
    """
    Multi-regime equity/ETF strategy.
    
    Regimes:
    - BULL_TREND: Up trend, breadth positive, low VIX
    - BEAR_TREND: Down trend, breadth negative, high VIX -> inverse ETF plays
    - CHOPPY: No trend, mean reversion setup
    - GRID_RANGE: Low efficiency, bounded trading
    - HIGH_VOL: Stress regime, defensive only
    - EVENT_RISK: Extreme sentiment, reduce risk
    - NO_TRADE: Insufficient data or extreme conditions
    
    Returns: TargetAllocation(symbol -> allocation %)
    """
    
    def __init__(self):
        self.tickers = UNIVERSE
        # Optional data feeds (missing feeds cause neutral scoring, not crashes)
        alt_symbols = CORE_ETF + STOCKS
        self.data_list = ([SocialSentiment(s) for s in alt_symbols]
                          + [InsiderTrading(s) for s in alt_symbols])
    
    @property
    def interval(self):
        """5-minute bars for intraday trading."""
        return "5min"
    
    @property
    def assets(self):
        """Full universe of tradable symbols."""
        return self.tickers
    
    @property
    def data(self):
        """Optional data feeds for sentiment/insider filtering."""
        return self.data_list
    
    def _phase(self, data):
        """Classify current bar into trading phase."""
        stamp = _last_stamp(data)
        if stamp is None:
            return "UNKNOWN"
        minutes = stamp.hour * 60 + stamp.minute
        if minutes < 9 * 60 + 45:
            return "PRE_OPEN"
        if minutes < 10 * 60 + 15:
            return "OPEN"
        if minutes < 11 * 60:
            return "MORNING"
        if minutes < 14 * 60:
            return "MIDDAY"
        if minutes < 15 * 60 + 20:
            return "AFTERNOON"
        if minutes < 15 * 60 + 50:
            return "CLOSE_WINDOW"
        return "FLATTEN"
    
    def _market_context(self, data):
        """Detect current market regime."""
        names = ("SPY", "QQQ", "IWM", "VIXY", "TLT", "GLD")
        market = {s: _bars(data, s) for s in names}
        
        # Require minimum bar count (120 bars = ~10 hours of 5-min data)
        if any(len(market.get(s, [])) < 120 for s in names):
            return {"regime": "NO_TRADE", "volatility": None, "breadth": None, "event": True}
        
        # Analyze broad index trends (SPY, QQQ, IWM)
        trends, returns, efficiency = [], [], []
        for symbol in ("SPY", "QQQ", "IWM"):
            xs, c = market[symbol], _values(market[symbol], "close")
            v, e20, e50, e100 = _vwap(xs[-78:]), _ema(c, 20), _ema(c, 50), _ema(c, 100)
            r5, r20, eff = _roc(c, 5), _roc(c, 20), _efficiency(c, 20)
            
            if None in (v, e20, e50, e100, r5, r20, eff):
                return {"regime": "NO_TRADE", "volatility": None, "breadth": None, "event": True}
            
            trends.append({"up": c[-1] > v and e20 > e50 > e100,
                           "down": c[-1] < v and e20 < e50 < e100,
                           "price": c[-1], "vwap": v})
            returns.append((r5, r20))
            efficiency.append(eff)
        
        # Volatility analysis (VIX)
        vixc = _values(market["VIXY"], "close")
        if len(vixc) < 40:
            return {"regime": "NO_TRADE", "volatility": None, "breadth": None, "event": True}
        
        vfast, vslow = _ema(vixc, 10), _ema(vixc, 30)
        spy_atr = _atr(market["SPY"])
        spy_price = trends[0]["price"]
        
        if None in (vfast, vslow, spy_atr) or not spy_price:
            return {"regime": "NO_TRADE", "volatility": None, "breadth": None, "event": True}
        
        volatility = spy_atr / spy_price
        breadth = sum(x["up"] for x in trends) / 3.0 - sum(x["down"] for x in trends) / 3.0
        vix_event = vfast > vslow * 1.15 or volatility > 0.020
        
        # Sentiment extremes
        sentiment_events = []
        for symbol in ("SPY", "QQQ", "IWM"):
            _, extreme = _social(data, symbol)
            sentiment_events.append(extreme)
        
        # Regime classification
        if vix_event:
            regime = "HIGH_VOL"
        elif sum(sentiment_events) >= 2:
            regime = "EVENT_RISK"
        elif sum(x["up"] for x in trends) >= 2 and breadth >= 0.33:
            regime = "BULL_TREND"
        elif sum(x["down"] for x in trends) >= 2 and breadth <= -0.33:
            regime = "BEAR_TREND"
        elif sum(efficiency) / len(efficiency) < 0.22:
            regime = "GRID_RANGE"
        else:
            regime = "CHOPPY"
        
        return {
            "regime": regime,
            "volatility": volatility,
            "breadth": breadth,
            "event": sum(sentiment_events) >= 2,
            "market": market,
            "trends": trends,
        }
    
    def _reference(self, xs):
        """Extract session reference levels (gap, ORB, etc)."""
        today, prior = _session_bars(xs)
        if not today or not prior:
            return None
        
        prior_close = float(prior[-1]["close"])
        first_open = float(today[0]["open"])
        orb15 = today[:3]
        orb30 = today[:6]
        
        if len(orb15) < 3 or len(orb30) < 6:
            return None
        
        return {
            "gap": first_open / prior_close - 1.0 if prior_close else 0.0,
            "prior_close": prior_close,
            "open": first_open,
            "orb15_high": max(float(x["high"]) for x in orb15),
            "orb15_low": min(float(x["low"]) for x in orb15),
            "orb30_high": max(float(x["high"]) for x in orb30),
            "orb30_low": min(float(x["low"]) for x in orb30),
            "today": today,
        }
    
    def _candidate(self, symbol, data, ctx, phase):
        """
        Screen and score candidate for entry.
        Returns candidate dict or None if does not qualify.
        """
        regime = ctx["regime"]
        
        # Phase and regime filters
        if symbol == "VIXY" or phase in {"UNKNOWN", "PRE_OPEN", "OPEN", "FLATTEN"}:
            return None
        if regime in {"NO_TRADE", "HIGH_VOL"}:
            return None
        if regime in {"GRID_RANGE", "CHOPPY"} and symbol in LEVERAGED:
            return None
        if regime == "BULL_TREND" and symbol in INVERSE:
            return None
        if regime == "BEAR_TREND" and symbol not in INVERSE:
            return None
        
        # Data sufficiency
        xs = _bars(data, symbol)
        if len(xs) < 120:
            return None
        
        c, volume = _values(xs, "close"), _values(xs, "volume")
        price = c[-1]
        atr, vwap, rvol = _atr(xs), _vwap(xs[-78:]), _rvol(xs)
        ema9, ema21, ema50, ema100 = (_ema(c, n) for n in (9, 21, 50, 100))
        rsi, roc5, roc14, roc21 = _rsi(c), _roc(c, 5), _roc(c, 14), _roc(c, 21)
        ref = self._reference(xs)
        
        if any(x is None for x in (atr, vwap, rvol, ema9, ema21, ema50, ema100, 
                                   rsi, roc5, roc14, roc21, ref)):
            return None
        
        # Price range and volatility filters
        if price < 5 or atr / price > 0.045:
            return None
        
        # Liquidity filter
        avg_dollar_volume = _sma([x * y for x, y in zip(c, volume)], 20)
        if avg_dollar_volume is None or avg_dollar_volume < 10_000_000:
            return None
        
        gap = ref["gap"]
        close_location = (float(xs[-1]["close"]) - float(xs[-1]["low"])) / max(
            float(xs[-1]["high"]) - float(xs[-1]["low"]), 0.01
        )
        
        score, lanes = 0.0, []
        direction = 1
        
        # -------- BULL_TREND Setup --------
        if regime == "BULL_TREND":
            if not (price > vwap and ema9 > ema21 > ema50 and roc21 > 0):
                return None
            
            lanes.append("MOMENTUM_CONTINUATION")
            score += 2.0
            
            if price > ref["orb15_high"] and rvol >= 1.25:
                lanes.append("ORB15_VOLUME_BREAKOUT")
                score += 2.5
            
            if price > ref["orb30_high"] and rvol >= 1.20:
                lanes.append("ORB30_VOLUME_BREAKOUT")
                score += 2.0
            
            if float(xs[-2]["close"]) <= vwap <= price and roc5 > 0:
                lanes.append("VWAP_PULLBACK")
                score += 2.0
            
            if gap >= 0.02 and price > ref["open"] and rvol >= 1.25:
                lanes.append("GAP_AND_GO")
                score += 1.5
            
            if rvol >= 1.15:
                lanes.append("RELATIVE_VOLUME")
                score += 1.0
        
        # -------- BEAR_TREND Setup --------
        elif regime == "BEAR_TREND":
            if symbol not in INVERSE:
                return None
            
            underlying = "QQQ" if symbol in {"SQQQ", "PSQ"} else "SPY"
            ux = _bars(data, underlying)
            uc = _values(ux, "close")
            uv = _vwap(ux[-78:])
            
            if len(ux) < 120 or uv is None or not (uc[-1] < uv and _roc(uc, 5) < 0):
                return None
            
            lanes.append("INVERSE_ETF_CONFIRMATION")
            score += 2.5
            
            if price > vwap and ema9 > ema21 and roc5 > 0:
                lanes.append("INVERSE_MOMENTUM")
                score += 1.5
            
            if rvol >= 1.15:
                lanes.append("RELATIVE_VOLUME")
                score += 1.0
            
            direction = 1  # Inverse ETF is purchased
        
        # -------- GRID_RANGE / CHOPPY Setup --------
        elif regime in {"GRID_RANGE", "CHOPPY"}:
            if symbol not in CORE_ETF and symbol not in STOCKS:
                return None
            
            bands = _bollinger(c)
            z = (price - vwap) / atr
            
            if bands is None or z > -1.0 or rsi > 43:
                return None
            
            lanes.append("VWAP_MEAN_REVERSION")
            score += 2.0
            
            if price <= bands[0]:
                lanes.append("BOUNDED_GRID_EDGE")
                score += 1.5
            
            if close_location >= 0.65:
                lanes.append("LIQUIDITY_RECLAIM_PROXY")
                score += 1.0
        
        else:
            return None
        
        # -------- Sentiment Filter --------
        sentiment, sentiment_event = _social(data, symbol)
        if sentiment_event:
            return None  # Hard reject extreme sentiment for now
        
        if sentiment is not None and sentiment >= 0.50:
            lanes.append("SENTIMENT_CONFIRMATION")
            score += 0.35
        elif sentiment is not None and sentiment < 0.30 and symbol not in INVERSE:
            return None  # Bearish sentiment on long = disqualify
        
        # -------- Insider Filter --------
        if _insider_sale(data, symbol) and symbol not in INVERSE:
            score -= 0.50
            lanes.append("INSIDER_SALE_PENALTY")
        
        # -------- Order Flow Proxies --------
        if direction > 0 and close_location >= 0.70 and rvol >= 1.10:
            lanes.append("SIGNED_VOLUME_PROXY")
            score += 0.75
        
        # -------- PMC-Green Proxy --------
        pmc_proxy = (direction > 0 and price > vwap and roc5 > 0
                     and rvol >= 1.15 and close_location >= 0.65)
        if pmc_proxy:
            lanes.append("PMC_GREEN_PRICE_VOLUME_PROXY")
            score += 0.75
        
        # -------- Phase-based Adjustments --------
        if phase == "MIDDAY":
            score -= 0.50
        elif phase == "CLOSE_WINDOW":
            score -= 1.25
        
        # -------- Minimum Score Gate --------
        if score < 4.0:
            return None
        
        return {
            "symbol": symbol,
            "score": score,
            "lanes": lanes,
            "atr": atr,
            "price": price,
            "rvol": rvol,
            "roc21": roc21,
            "atr_pct": atr / price,
            "direction": direction,
            "leveraged": symbol in LEVERAGED,
            "family": _market_family(symbol),
        }
    
    def _allocation(self, candidate, regime):
        """Determine position size from candidate score."""
        symbol = candidate["symbol"]
        
        # Base weight: smaller for leveraged (decay risk)
        weight = 0.30 if not candidate["leveraged"] else 0.12
        
        # Bounded grid edge = single tranch (don't average down)
        if "BOUNDED_GRID_EDGE" in candidate["lanes"]:
            weight = 0.15
        
        # Score-based scaling: higher score = more capital
        weight *= min(1.0, max(0.25, candidate["score"] / 8.0))
        
        # Volatility adjustment: higher volatility = smaller position
        weight *= min(1.0, 0.012 / max(candidate["atr_pct"], 0.004))
        
        # Regime adjustment: choppy = less confident
        if regime == "CHOPPY":
            weight *= 0.60
        
        return min(weight, 0.30 if not candidate["leveraged"] else 0.12)
    
    def run(self, data):
        """
        Main strategy execution.
        
        Args:
            data: Surmount data dict with OHLCV bars and optional feeds
        
        Returns:
            TargetAllocation with symbol -> allocation % mapping
        """
        # Detect regime
        ctx = self._market_context(data)
        phase = self._phase(data)
        regime = ctx["regime"]
        
        # No-trade conditions
        if phase in {"UNKNOWN", "PRE_OPEN", "OPEN", "FLATTEN"}:
            return TargetAllocation({})
        
        if regime in {"NO_TRADE", "EVENT_RISK"}:
            return TargetAllocation({})
        
        # HIGH_VOL: Only defensive positions
        if regime == "HIGH_VOL":
            if len(_bars(data, "BIL")) >= 120:
                return TargetAllocation({"BIL": 0.20})
            return TargetAllocation({})
        
        # Screen all symbols for this regime
        ranked = []
        for symbol in self.tickers:
            candidate = self._candidate(symbol, data, ctx, phase)
            if candidate:
                ranked.append(candidate)
        
        # Sort by score (highest first)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top candidates (one per family, max 2)
        selected, used_families = [], set()
        for candidate in ranked:
            if candidate["family"] not in used_families:
                selected.append(candidate)
                used_families.add(candidate["family"])
                if len(selected) >= 2:
                    break
        
        if not selected:
            return TargetAllocation({})
        
        # Build allocation
        allocation = {c["symbol"]: self._allocation(c, regime) for c in selected}
        
        # Cap total allocation at 65% (keep 35% powder)
        total = sum(allocation.values())
        if total > 0.65:
            allocation = {k: v * 0.65 / total for k, v in allocation.items()}
        
        return TargetAllocation(allocation)