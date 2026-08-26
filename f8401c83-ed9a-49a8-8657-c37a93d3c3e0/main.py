"""
SURMOUNT EQUITY STRATEGY - COMPLETE PRODUCTION VERSION
Campaign: surmount_equity_v1
Deployment: Athena-2 Bot Fleet
Created: August 25, 2026

STRATEGY OVERVIEW:
- Intraday equity/ETF strategy for SPY, QQQ, IWM
- Regime classification + setup detection + allocation
- ATR-based position sizing with hard stops
- 240-minute time stop + EOD flatten at 15:50 ET

SIGNALS:
- Bullish: Price > EMA20 > EMA50, RSI > 55, MACD crossover
- Bearish: Price < EMA20 < EMA50, RSI < 45, MACD negative
- Neutral: Price between EMAs, hold or flatten

RISK MANAGEMENT:
- Max position: 2% portfolio per trade
- Hard stop: 1.5× ATR
- Profit target: 2.0× ATR  
- Time stop: 240 minutes
- EOD flatten: 15:50 ET
"""

class SurmountEquityStrategy:
    """
    Complete Surmount strategy for intraday equity trading
    """
    
    def __init__(self):
        self.name = "Surmount_Equity_V1"
        self.symbols = ["SPY", "QQQ", "IWM"]
        self.timeframe = "5m"
        self.regime = "NEUTRAL"
        self.active_trades = []
        
    def calculate_indicators(self, price_data):
        """Calculate technical indicators"""
        ema_20 = self.calculate_ema(price_data, 20)
        ema_50 = self.calculate_ema(price_data, 50)
        rsi_14 = self.calculate_rsi(price_data, 14)
        macd = self.calculate_macd(price_data)
        atr_14 = self.calculate_atr(price_data, 14)
        
        return {
            'ema_20': ema_20,
            'ema_50': ema_50,
            'rsi': rsi_14,
            'macd': macd,
            'atr': atr_14
        }
    
    def classify_regime(self, price, indicators):
        """Classify market regime"""
        price_val = price[-1]
        ema_20 = indicators['ema_20'][-1]
        ema_50 = indicators['ema_50'][-1]
        rsi = indicators['rsi'][-1]
        macd = indicators['macd'][-1]
        
        # BULL REGIME: Strong uptrend
        if price_val > ema_20 > ema_50 and rsi > 55 and macd > 0:
            return "BULL"
        
        # BEAR REGIME: Strong downtrend
        elif price_val < ema_20 < ema_50 and rsi < 45 and macd < 0:
            return "BEAR"
        
        # NEUTRAL/CHOP: Sideways market
        else:
            return "NEUTRAL"
    
    def generate_signals(self, symbol, price_data, indicators):
        """Generate buy/sell signals"""
        regime = self.classify_regime(price_data, indicators)
        
        signals = {
            'symbol': symbol,
            'regime': regime,
            'action': None,
            'confidence': 0.0
        }
        
        # BULL signals
        if regime == "BULL":
            ema_20 = indicators['ema_20'][-1]
            price_val = price_data[-1]
            
            if price_val > ema_20 * 1.001:  # Price above EMA by 0.1%
                signals['action'] = 'BUY'
                signals['confidence'] = 0.8
                
        # BEAR signals
        elif regime == "BEAR":
            ema_20 = indicators['ema_20'][-1]
            price_val = price_data[-1]
            
            if price_val < ema_20 * 0.999:  # Price below EMA by 0.1%
                signals['action'] = 'SELL'
                signals['confidence'] = 0.8
        
        return signals
    
    def calculate_position_size(self, portfolio_value, atr, price):
        """Calculate position size based on ATR and risk"""
        risk_per_trade = portfolio_value * 0.02  # 2% risk per trade
        stop_distance = atr * 1.5  # 1.5× ATR stop
        position_size = risk_per_trade / stop_distance
        
        return position_size
    
    def set_stops_and_targets(self, entry_price, atr, direction):
        """Set stop loss and profit target"""
        if direction == 'BUY':
            stop_loss = entry_price - (atr * 1.5)
            profit_target = entry_price + (atr * 2.0)
            
        else:  # SELL
            stop_loss = entry_price + (atr * 1.5)
            profit_target = entry_price - (atr * 2.0)
        
        return {
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'time_stop': 240  # minutes
        }
    
    def check_eod_flatten(self, current_time):
        """Check if time to flatten all positions (15:50 ET)"""
        hour = current_time.hour
        minute = current_time.minute
        
        return hour == 15 and minute >= 50
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        ema = []
        multiplier = 2 / (period + 1)
        
        for i in range(len(data)):
            if i == 0:
                ema.append(data[i])
            else:
                ema_val = data[i] * multiplier + ema[-1] * (1 - multiplier)
                ema.append(ema_val)
        
        return ema
    
    def calculate_rsi(self, data, period):
        """Calculate Relative Strength Index"""
        rsi = []
        gains = 0
        losses = 0
        
        for i in range(len(data)):
            if i == 0:
                rsi.append(50)
            else:
                change = data[i] - data[i-1]
                if change > 0:
                    gains = change
                    losses = 0
                else:
                    gains = 0
                    losses = abs(change)
                
                avg_gain = gains / period
                avg_loss = losses / period
                
                if avg_loss == 0:
                    rsi_val = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi_val = 100 - (100 / (1 + rs))
                
                rsi.append(rsi_val)
        
        return rsi
    
    def calculate_macd(self, data):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_12 = self.calculate_ema(data, 12)
        ema_26 = self.calculate_ema(data, 26)
        
        macd = [ema_12[i] - ema_26[i] for i in range(len(ema_12))]
        
        return macd
    
    def calculate_atr(self, data, period):
        """Calculate Average True Range"""
        atr = []
        
        for i in range(len(data)):
            if i == 0:
                atr.append(data[i] * 0.02)  # Approximate
            else:
                tr = abs(data[i] - data[i-1])
                atr_val = (atr[-1] * (period - 1) + tr) / period
                atr.append(atr_val)
        
        return atr
    
    def run_strategy(self, market_data, portfolio_value):
        """Main strategy loop"""
        trades = []
        
        for symbol in self.symbols:
            price_data = market_data[symbol]['price']
            
            # Calculate indicators
            indicators = self.calculate_indicators(price_data)
            
            # Generate signals
            signals = self.generate_signals(symbol, price_data, indicators)
            
            # Execute if signal present
            if signals['action']:
                atr = indicators['atr'][-1]
                position_size = self.calculate_position_size(
                    portfolio_value, atr, price_data[-1]
                )
                stops_targets = self.set_stops_and_targets(
                    price_data[-1], atr, signals['action']
                )
                
                trade = {
                    'symbol': symbol,
                    'direction': signals['action'],
                    'entry_price': price_data[-1],
                    'position_size': position_size,
                    'stop_loss': stops_targets['stop_loss'],
                    'profit_target': stops_targets['profit_target'],
                    'time_stop_minutes': 240,
                    'regime': signals['regime'],
                    'confidence': signals['confidence']
                }
                
                trades.append(trade)
        
        return trades


# ═════════════════════════════════════════════════════════════════════
# DEPLOYMENT CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

STRATEGY_CONFIG = {
    'name': 'Surmount_Equity_V1',
    'symbols': ['SPY', 'QQQ', 'IWM'],
    'timeframe': '5m',
    'max_position_pct': 0.02,  # 2% risk per trade
    'stop_multiplier': 1.5,     # 1.5× ATR
    'target_multiplier': 2.0,   # 2.0× ATR
    'time_stop_minutes': 240,
    'eod_flatten_time': '15:50',  # 15:50 ET
    'paper_trading': True,
    'broker': 'IBKR',
    'account': 'DUP645464'
}

if __name__ == "__main__":
    strategy = SurmountEquityStrategy()
    print(f"Strategy initialized: {strategy.name}")
    print(f"Symbols: {strategy.symbols}")
    print(f"Regime: {strategy.regime}")