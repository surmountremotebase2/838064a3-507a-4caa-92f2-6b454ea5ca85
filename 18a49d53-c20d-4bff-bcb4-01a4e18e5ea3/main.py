"""ATHENA-native equity/ETF signal engine (paper-safe, no broker calls).

This module is the full intelligence layer. It consumes normalized snapshots
from existing ATHENA producers: PMC/PMC-green, order flow, money flow,
momentum/volume scanners, news/event blackout, and canonical SPX/QQQ GEX.
It returns auditable intents for the existing RiskKernel/IBKR queue adapter.
It intentionally never imports or calls a broker transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timezone
from math import floor
from typing import Any, Iterable, Mapping, Optional


LONG_LEVERAGED = {"TQQQ", "SOXL", "UPRO", "LABU", "NVDL", "TECL"}
INVERSE_LEVERAGED = {"SQQQ", "SOXS", "SPXU", "LABD", "FAZ", "PSQ"}
LEVERAGED = LONG_LEVERAGED | INVERSE_LEVERAGED


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class PmcState:
    direction: int = 0
    green: bool = False
    cmf: Optional[float] = None
    efficiency: Optional[float] = None
    timeframes_aligned: int = 0
    timeframes_required: int = 0
    fresh: bool = False


@dataclass(frozen=True)
class FlowState:
    signed_delta: Optional[float] = None
    cvd_slope: Optional[float] = None
    money_flow: Optional[float] = None
    participant_bias: int = 0
    fresh: bool = False


@dataclass(frozen=True)
class NewsState:
    global_blackout: bool = False
    symbol_blackout: bool = False
    event_type: str = "none"
    severity: str = "none"
    age_seconds: Optional[float] = None
    headline: str = ""


@dataclass(frozen=True)
class OptionsState:
    available: bool = False
    fresh: bool = False
    regime: str = "UNKNOWN"  # POSITIVE_GAMMA, NEGATIVE_GAMMA, PIN_ZONE, ZERO_GAMMA_CROSS
    sign_conflict: bool = False
    spot: Optional[float] = None
    zero_gamma: Optional[float] = None
    pin_strike: Optional[float] = None
    distance_to_zero_gamma_pct: Optional[float] = None


@dataclass(frozen=True)
class MarketState:
    spx_bars: tuple[Bar, ...] = ()
    qqq_bars: tuple[Bar, ...] = ()
    spy_bars: tuple[Bar, ...] = ()
    vix_level: Optional[float] = None
    breadth: Optional[float] = None
    options: OptionsState = OptionsState()
    news: NewsState = NewsState()
    broker_ready: bool = False
    data_fresh: bool = False
    now_et: Optional[datetime] = None
    bars_by_symbol: Mapping[str, tuple[Bar, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    side: int
    entry: float
    stop: float
    target: float
    opened_at: Optional[datetime] = None
    strategy: str = ""


@dataclass(frozen=True)
class Candidate:
    symbol: str
    side: int
    strategy: str
    score: float
    entry: float
    stop: float
    target: float
    atr: float
    reasons: tuple[str, ...] = ()
    sector: str = "unknown"
    leveraged: bool = False


@dataclass(frozen=True)
class Intent:
    action: str  # ENTER, EXIT, HOLD, REJECT
    symbol: str
    side: int = 0
    qty: int = 0
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    strategy: str = ""
    reason: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)


def _close(xs: Iterable[Bar]) -> list[float]:
    return [x.close for x in xs]


def _ema(v: list[float], n: int) -> Optional[float]:
    if len(v) < n:
        return None
    out = sum(v[:n]) / n
    k = 2.0 / (n + 1.0)
    for x in v[n:]:
        out = k * x + (1.0 - k) * out
    return out


def _atr(xs: tuple[Bar, ...], n: int = 14) -> Optional[float]:
    if len(xs) < n + 1:
        return None
    values = []
    for p, b in zip(xs[-n - 1:-1], xs[-n:]):
        values.append(max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close)))
    return sum(values) / n


def _vwap(xs: tuple[Bar, ...]) -> Optional[float]:
    volume = sum(max(0.0, b.volume) for b in xs)
    return sum(b.close * max(0.0, b.volume) for b in xs) / volume if volume else None


def _rvol(xs: tuple[Bar, ...], n: int = 20) -> Optional[float]:
    if len(xs) < n + 1:
        return None
    average = sum(b.volume for b in xs[-n - 1:-1]) / n
    return xs[-1].volume / average if average else None


def _ret(xs: tuple[Bar, ...], n: int) -> Optional[float]:
    return xs[-1].close / xs[-n - 1].close - 1.0 if len(xs) > n else None


def _range_breakout(xs: tuple[Bar, ...], bars: int) -> tuple[bool, bool]:
    if len(xs) < bars + 1:
        return False, False
    prior = xs[-bars - 1:-1]
    return xs[-1].close > max(x.high for x in prior), xs[-1].close < min(x.low for x in prior)


class AthenaEquityEngine:
    """Pure signal/risk engine. Feed it ATHENA state adapters; route intents through RiskKernel."""

    def __init__(self, account_equity: float, *, risk_pct: float = 0.0025,
                 max_positions: int = 2, daily_loss_pct: float = 0.01,
                 allow_inverse_etf: bool = True):
        self.account_equity = float(account_equity)
        self.risk_pct = risk_pct
        self.max_positions = max_positions
        self.daily_loss_pct = daily_loss_pct
        self.allow_inverse_etf = allow_inverse_etf

    def regime(self, m: MarketState) -> str:
        if not m.broker_ready or not m.data_fresh:
            return "NO_TRADE"
        if m.news.global_blackout:
            return "EVENT_RISK"
        if m.options.sign_conflict:
            return "NO_TRADE"
        if m.vix_level is not None and m.vix_level >= 30:
            return "HIGH_VOL"
        spx, qqq = m.spx_bars, m.qqq_bars
        if len(spx) < 60 or len(qqq) < 60:
            return "NO_TRADE"
        sc, qc = _close(spx), _close(qqq)
        sv, qv = _vwap(spx[-78:]), _vwap(qqq[-78:])
        se9, se21, qe9, qe21 = _ema(sc, 9), _ema(sc, 21), _ema(qc, 9), _ema(qc, 21)
        if None in (sv, qv, se9, se21, qe9, qe21):
            return "NO_TRADE"
        if m.options.regime in {"ZERO_GAMMA_CROSS"}:
            return "NO_TRADE"
        if (m.options.regime in {"POSITIVE_GAMMA", "PIN_ZONE", "EXPIRY_MAGNET"}
                and abs(m.options.distance_to_zero_gamma_pct or 1.0) < 0.15):
            return "RANGE"
        bull = sc[-1] > sv and qc[-1] > qv and se9 > se21 and qe9 > qe21
        bear = sc[-1] < sv and qc[-1] < qv and se9 < se21 and qe9 < qe21
        if m.breadth is not None:
            bull = bull and m.breadth > 0.10
            bear = bear and m.breadth < -0.10
        if bull:
            return "BULL_TREND"
        if bear:
            return "BEAR_TREND"
        return "RANGE"

    def _exit_intents(self, positions: tuple[Position, ...], m: MarketState,
                      regime: str) -> list[Intent]:
        now = m.now_et
        out: list[Intent] = []
        flatten = bool(now and now.weekday() < 5 and now.time() >= time(15, 50))
        for p in positions:
            bars = m.bars_by_symbol.get(p.symbol, ())
            last = bars[-1].close if bars else None
            if flatten:
                out.append(Intent("EXIT", p.symbol, side=-p.side, qty=p.qty,
                                  strategy=p.strategy, reason="RTH_EOD_FLATTEN"))
            elif m.news.global_blackout or m.news.symbol_blackout:
                out.append(Intent("EXIT", p.symbol, side=-p.side, qty=p.qty,
                                  strategy=p.strategy, reason="NEWS_EVENT_BLACKOUT"))
            elif regime in {"NO_TRADE", "EVENT_RISK"}:
                out.append(Intent("EXIT", p.symbol, side=-p.side, qty=p.qty,
                                  strategy=p.strategy, reason="REGIME_OR_DATA_INVALID"))
            elif last is not None and ((p.side > 0 and last <= p.stop) or (p.side < 0 and last >= p.stop)):
                out.append(Intent("EXIT", p.symbol, side=-p.side, qty=p.qty,
                                  strategy=p.strategy, reason="STOP_LEVEL_TOUCHED"))
            elif last is not None and ((p.side > 0 and last >= p.target) or (p.side < 0 and last <= p.target)):
                out.append(Intent("EXIT", p.symbol, side=-p.side, qty=p.qty,
                                  strategy=p.strategy, reason="TARGET_LEVEL_TOUCHED"))
            else:
                out.append(Intent("HOLD", p.symbol, side=p.side, qty=p.qty,
                                  entry=p.entry, stop=p.stop, target=p.target,
                                  strategy=p.strategy, reason="POSITION_PROTECTED"))
        return out

    def _candidate(self, symbol: str, xs: tuple[Bar, ...], m: MarketState, regime: str,
                   pmc: PmcState, flow: FlowState, sector: str = "unknown") -> Optional[Candidate]:
        if len(xs) < 90:
            return None
        c = _close(xs)
        a, vw, rv = _atr(xs), _vwap(xs[-78:]), _rvol(xs)
        e9, e21, e50 = _ema(c, 9), _ema(c, 21), _ema(c, 50)
        r5, r20 = _ret(xs, 5), _ret(xs, 20)
        if None in (a, vw, rv, e9, e21, e50, r5, r20) or a <= 0:
            return None
        if c[-1] < 5 or a / c[-1] > 0.04:
            return None
        leveraged = symbol in LEVERAGED
        if regime == "RANGE" and leveraged:
            return None
        if regime == "BULL_TREND" and symbol in INVERSE_LEVERAGED:
            return None
        if regime == "BEAR_TREND" and not self.allow_inverse_etf and symbol in INVERSE_LEVERAGED:
            return None
        side = 1 if regime in {"BULL_TREND", "RANGE"} else -1
        reasons: list[str] = []
        score = 0.0
        breakout_up, breakout_dn = _range_breakout(xs, 6)
        trend_ok = (e9 > e21 > e50 and c[-1] > vw) if side > 0 else (e9 < e21 < e50 and c[-1] < vw)
        flow_ok = flow.fresh and ((flow.participant_bias == side)
                                  or (flow.signed_delta is not None and flow.signed_delta * side > 0)
                                  or (flow.money_flow is not None and flow.money_flow * side > 0))
        pmc_ok = pmc.fresh and pmc.green and pmc.direction == side and pmc.timeframes_aligned >= max(1, pmc.timeframes_required)
        if regime in {"BULL_TREND", "BEAR_TREND"}:
            if not trend_ok:
                return None
            if (side > 0 and r20 <= 0) or (side < 0 and r20 >= 0):
                return None
            score += 2.0
            reasons.append("EMA_VWAP_TREND")
            if (breakout_up if side > 0 else breakout_dn) and rv >= 1.25:
                score += 2.5
                reasons.append("ORB_VOLUME_BREAKOUT")
            if rv >= 1.15:
                score += 1.0
                reasons.append("RELATIVE_VOLUME")
            if pmc_ok:
                score += 1.5
                reasons.append("PMC_GREEN_ALIGNED")
            if flow_ok:
                score += 1.5
                reasons.append("ORDER_FLOW_ALIGNED")
            if not pmc_ok and not flow_ok and score < 4.0:
                return None
            strategy = "PMC_FLOW_MOMENTUM" if pmc_ok and flow_ok else "MOMENTUM_VOLUME"
        else:
            z = (c[-1] - vw) / a
            if side > 0 and not (z <= -1.0 and c[-1] > c[-2]):
                return None
            score += 2.0
            reasons.append("POSITIVE_GAMMA_RANGE_REVERSION")
            if pmc_ok:
                score += 1.5
                reasons.append("PMC_RECLAIM")
            if flow_ok:
                score += 1.0
                reasons.append("FLOW_RECLAIM")
            strategy = "RANGE_VWAP_PMC_RECLAIM"
        if m.news.symbol_blackout:
            return None
        # Options only modify the geometry; they never create an entry alone.
        stop_mult, target_r = 1.0, 2.0
        if m.options.regime in {"POSITIVE_GAMMA", "PIN_ZONE", "EXPIRY_MAGNET"}:
            stop_mult, target_r = 0.8, 1.4
        elif m.options.regime == "NEGATIVE_GAMMA":
            stop_mult, target_r = 1.25, 2.0
        if leveraged:
            stop_mult *= 1.25
        entry = c[-1]
        stop = entry - side * stop_mult * a
        target = entry + side * target_r * abs(entry - stop)
        return Candidate(symbol, side, strategy, score, entry, stop, target, a,
                         tuple(reasons), sector, leveraged)

    def evaluate(self, market: MarketState,
                 bars_by_symbol: Mapping[str, tuple[Bar, ...]],
                 pmc_by_symbol: Mapping[str, PmcState],
                 flow_by_symbol: Mapping[str, FlowState],
                 positions: tuple[Position, ...] = (),
                 sectors: Mapping[str, str] | None = None,
                 daily_pnl: float = 0.0) -> list[Intent]:
        regime = self.regime(market)
        intents = self._exit_intents(positions, market, regime)
        if daily_pnl <= -self.account_equity * self.daily_loss_pct:
            return intents + [Intent("REJECT", "*", reason="DAILY_LOSS_LIMIT")]
        if regime in {"NO_TRADE", "EVENT_RISK", "HIGH_VOL"}:
            return intents + [Intent("REJECT", "*", reason=regime)]
        if len([x for x in intents if x.action == "HOLD"]) >= self.max_positions:
            return intents
        candidates = []
        for symbol, bars in bars_by_symbol.items():
            candidate = self._candidate(symbol, bars, market, regime,
                                        pmc_by_symbol.get(symbol, PmcState()),
                                        flow_by_symbol.get(symbol, FlowState()),
                                        (sectors or {}).get(symbol, "unknown"))
            if candidate:
                candidates.append(candidate)
        candidates.sort(key=lambda x: x.score, reverse=True)
        used_sector = {p.strategy for p in positions}
        for candidate in candidates:
            if len([x for x in intents if x.action in {"HOLD", "ENTER"}]) >= self.max_positions:
                break
            if candidate.sector in used_sector:
                continue
            risk = self.account_equity * (self.risk_pct * (0.35 if candidate.leveraged else 1.0))
            qty = floor(risk / max(abs(candidate.entry - candidate.stop), 0.01))
            if qty <= 0:
                continue
            intents.append(Intent("ENTER", candidate.symbol, side=candidate.side, qty=qty,
                                  entry=candidate.entry, stop=candidate.stop,
                                  target=candidate.target, strategy=candidate.strategy,
                                  reason=";".join(candidate.reasons),
                                  evidence={"regime": regime, "score": candidate.score,
                                            "pmc": asdict(pmc_by_symbol.get(candidate.symbol, PmcState())),
                                            "flow": asdict(flow_by_symbol.get(candidate.symbol, FlowState())),
                                            "options": asdict(market.options)}))
            used_sector.add(candidate.sector)
        return intents