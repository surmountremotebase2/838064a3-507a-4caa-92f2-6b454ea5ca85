"""
SURMOUNT V3 STRATEGY - FINAL PRODUCTION VERSION
Fixes all backtest errors + Surmount-compatible

KEY FIXES:
✅ Class named 'TradingStrategy' (Surmount requirement)
✅ Inherits from Strategy base class
✅ Uses correct OHLCV data format
✅ Returns TargetAllocation objects
✅ No external dependencies beyond Surmount
"""

from surmount.base_class import Strategy, TargetAllocation
from datetime import datetime
import numpy as np

class TradingStrategy(Strategy):
    """V3 Strategy: Trend + Mean-Reversion"""
    
    def __init__(self):
        self.tickers = ["SPY", "QQQ", "IWM"]
        self.data_list = []
        self.consecutive_losses = 0
        self.closed_trades = []
        self.position_history = {}
        
    @property
    def interval(self):
        return "1d"
    
    @property
    def assets(self):
        return self.tickers
    
    @property
    def data(self):
        return self.data_list
    
    def _safe_float(self, v, d=0.0):
        try:
            return float(v) if v is not None else d
        except:
            return d
    
    def _get_bars(self, data, symbol):
        """Extract bars for symbol from Surmount data"""
        try:
            bars = []
            for row in data.get("ohlcv", []):
                if symbol in row:
                    bars.append(row[symbol])
            return bars
        except:
            return []
    
    def _ema(self, closes, n):
        """Calculate EMA"""
        if len(closes) < n or n <= 0:
            return closes[-1] if closes else 0
        closes = np.array([self._safe_float(c.get("close", 0)) for c in closes[-n:]])
        value = np.mean(closes)
        alpha = 2.0 / (n + 1.0)
        for price in closes:
            value = alpha * price + (1.0 - alpha) * value
        return float(value)
    
    def _sma(self, closes, n):
        """Calculate SMA"""
        if len(closes) < n or n <= 0:
            return closes[-1].get("close", 0) if closes else 0
        vals = [self._safe_float(c.get("close", 0)) for c in closes[-n:]]
        return np.mean(vals)
    
    def _rsi(self, closes, n=14):
        """Calculate RSI"""
        if len(closes) < n + 1 or n <= 0:
            return 50.0
        try:
            prices = [self._safe_float(c.get("close", 0)) for c in closes[-(n+1):]]
            prices = np.array(prices)
            changes = np.diff(prices)
            gains = np.sum(np.maximum(changes, 0)) / n
            losses = np.sum(np.maximum(-changes, 0)) / n
            if losses == 0:
                return 100.0 if gains > 0 else 50.0
            rs = gains / losses
            return float(100.0 - 100.0 / (1.0 + rs))
        except:
            return 50.0
    
    def _atr(self, bars, n=14):
        """Calculate ATR"""
        if len(bars) < n + 1:
            close = self._safe_float(bars[-1].get("close", 100)) if bars else 100
            return close * 0.01
        try:
            trs = []
            for i in range(-n, 0):
                h = self._safe_float(bars[i].get("high", 0))
                l = self._safe_float(bars[i].get("low", 0))
                pc = self._safe_float(bars[i-1].get("close", 0)) if i > -len(bars) else h
                tr = max(h - l, abs(h - pc), abs(l - pc))
                trs.append(tr)
            return float(np.mean(trs)) if trs else 100 * 0.01
        except:
            return 100 * 0.01
    
    def run(self, data):
        """Main strategy execution"""
        allocations = {}
        
        # Get timestamp
        try:
            timestamp = None
            for row in reversed(data.get("ohlcv", [])):
                if row:
                    sample = next(iter(row.values()), None)
                    if sample and "date" in sample:
                        timestamp = sample["date"]
                        break
        except:
            timestamp = None
        
        # Skip if no data
        if not data.get("ohlcv") or not timestamp:
            return TargetAllocation(allocations)
        
        # Process each symbol
        for symbol in self.tickers:
            bars = self._get_bars(data, symbol)
            if len(bars) < 50:
                continue
            
            closes = np.array([self._safe_float(b.get("close", 0)) for b in bars])
            price = closes[-1]
            
            if price <= 0:
                continue
            
            # Calculate indicators
            rsi = self._rsi(bars, 14)
            ema20 = self._ema(bars, 20)
            ema50 = self._ema(bars, 50)
            atr = self._atr(bars, 14)
            
            # UPTREND SIGNAL
            if price > ema20 > ema50:
                if rsi > 55:
                    # Strong uptrend: size = 10% allocation
                    allocations[symbol] = 0.10
                elif rsi < 30:
                    # Mean-reversion bounce
                    allocations[symbol] = 0.08
            
            # DOWNTREND SIGNAL  
            elif price < ema20 < ema50:
                if rsi < 45:
                    # Strong downtrend: size = 10% allocation
                    allocations[symbol] = -0.10
                elif rsi > 70:
                    # Mean-reversion pullback
                    allocations[symbol] = -0.08
            
            # MEAN-REVERSION (neutral)
            elif 40 < rsi < 60:
                # Oversold bounce
                if rsi < 35:
                    allocations[symbol] = 0.05
                # Overbought pullback
                elif rsi > 65:
                    allocations[symbol] = -0.05
        
        return TargetAllocation(allocations)