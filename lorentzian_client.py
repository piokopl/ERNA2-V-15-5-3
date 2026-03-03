#!/usr/bin/env python3
"""Lorentzian Classification Trend Client (L)

Composite trend signal from 5 components:
1. Lorentzian Classification (k-NN with Lorentzian distance) — weight 2.0
2. SuperTrend (ATR-based) — weight 1.0
3. EMA200 (trend filter) — weight 1.0
4. Volume Profile (POC/VA) — weight 1.0
5. ICT/SMC (BOS + FVG) — weight 1.0

Aggregation:
  UP components add +weight, DOWN subtract -weight, NEUTRAL = 0.
  If score >= threshold → UP; score <= -threshold → DOWN; else NEUTRAL.

All calculations use closed candles only (no lookahead).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import httpx

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ComponentSignal:
    """Signal from a single component."""
    name: str
    direction: str  # UP / DOWN / NEUTRAL
    strength: float  # 0.0–1.0
    details: str = ""


@dataclass
class LorentzianSignal:
    """Combined Lorentzian trend signal."""
    direction: str  # UP / DOWN / NEUTRAL
    score: float  # raw weighted score
    confidence: float  # abs(score) / total_weight
    components: List[ComponentSignal] = field(default_factory=list)

    def __str__(self):
        parts = ", ".join(f"{c.name}={c.direction}" for c in self.components)
        return f"Lorentzian({self.direction}, score={self.score:+.1f}, conf={self.confidence:.0%}) [{parts}]"


# ── Client ───────────────────────────────────────────────────────────────────

class LorentzianTrendClient:
    """Lorentzian Classification composite trend analyser."""

    BINANCE_API = "https://api.binance.com/api/v3"

    # Component weights
    W_LC = 2.0
    W_ST = 1.0
    W_EMA = 1.0
    W_VP = 1.0
    W_SMC = 1.0
    TOTAL_WEIGHT = W_LC + W_ST + W_EMA + W_VP + W_SMC  # 6.0

    def __init__(
        self,
        timeframe: str = "5m",
        lookback: int = 750,
        # LC params
        k_neighbors: int = 21,
        horizon_bars: int = 6,
        min_confidence: float = 0.52,
        # SuperTrend params
        st_period: int = 10,
        st_multiplier: float = 3.0,
        # EMA params
        ema_period: int = 200,
        # Volume Profile params
        vp_lookback: int = 400,
        vp_bins: int = 80,
        vp_value_area_pct: float = 0.70,
        # SMC params
        pivot_left: int = 3,
        pivot_right: int = 3,
        bos_lookback: int = 100,
        # Aggregation
        threshold: float = 2.0,
        timeout: float = 12.0,
    ):
        self.timeframe = timeframe
        self.lookback = lookback
        # LC
        self.k_neighbors = k_neighbors
        self.horizon_bars = horizon_bars
        self.min_confidence = min_confidence
        # SuperTrend
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        # EMA
        self.ema_period = ema_period
        # VP
        self.vp_lookback = vp_lookback
        self.vp_bins = vp_bins
        self.vp_value_area_pct = vp_value_area_pct
        # SMC
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.bos_lookback = bos_lookback
        # Agg
        self.threshold = threshold
        self._session = httpx.Client(timeout=timeout)
        # Cache for warmup data
        self._cache: Dict[str, Tuple[float, np.ndarray]] = {}  # symbol → (timestamp, ohlcv)
        self._cache_ttl = 60.0  # 60s cache — avoid re-fetching within same window

    # ── Data fetch ───────────────────────────────────────────────────────────

    def _fetch_klines(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch OHLCV from Binance. Returns (N, 5) array [O,H,L,C,V]. Uses cache if fresh."""
        # Check cache
        now = time.time()
        if symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < self._cache_ttl:
                return cached

        try:
            binance_sym = symbol.replace("/", "")
            resp = self._session.get(
                f"{self.BINANCE_API}/klines",
                params={"symbol": binance_sym, "interval": self.timeframe, "limit": self.lookback},
            )
            if resp.status_code != 200:
                logger.warning(f"Binance klines HTTP {resp.status_code}")
                return None
            data = resp.json()
            if len(data) < self.ema_period + 50:
                logger.warning(f"Klines too short: {len(data)} (need {self.ema_period + 50})")
                return None
            arr = np.array(
                [[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data],
                dtype=np.float64,
            )
            self._cache[symbol] = (now, arr)
            return arr
        except Exception as e:
            logger.warning(f"Klines fetch error: {e}")
            return None

    def warmup(self, symbols: List[str]) -> None:
        """Pre-fetch historical data for all symbols on startup."""
        for sym in symbols:
            logger.info(f"L: warming up {sym} ({self.lookback}x{self.timeframe} candles)...")
            data = self._fetch_klines(sym)
            if data is not None:
                logger.info(f"L: {sym} ready — {len(data)} candles loaded")
            else:
                logger.warning(f"L: {sym} warmup FAILED")

    # ── 1. Lorentzian Classification ─────────────────────────────────────────

    def _compute_lc(self, ohlcv: np.ndarray) -> ComponentSignal:
        """k-NN with Lorentzian distance on feature vector."""
        closes = ohlcv[:, 3]
        volumes = ohlcv[:, 4]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        n = len(closes)

        # ── Build features per bar (7 features) ──
        # return(1), return(3), return(6), RSI(14), volume_zscore, ATR%, slope_ema50
        rsi = self._rsi(closes, 14)
        atr = self._atr(highs, lows, closes, 14)
        ema50 = self._ema(closes, 50)

        features = np.zeros((n, 7), dtype=np.float64)
        for i in range(max(50, 14), n):
            features[i, 0] = (closes[i] / closes[i - 1] - 1.0) if closes[i - 1] > 0 else 0.0
            features[i, 1] = (closes[i] / closes[max(0, i - 3)] - 1.0) if closes[max(0, i - 3)] > 0 else 0.0
            features[i, 2] = (closes[i] / closes[max(0, i - 6)] - 1.0) if closes[max(0, i - 6)] > 0 else 0.0
            features[i, 3] = rsi[i]
            # Volume z-score (rolling 50)
            vol_window = volumes[max(0, i - 49):i + 1]
            vol_mean = vol_window.mean()
            vol_std = vol_window.std()
            features[i, 4] = (volumes[i] - vol_mean) / vol_std if vol_std > 0 else 0.0
            # ATR%
            features[i, 5] = atr[i] / closes[i] if closes[i] > 0 else 0.0
            # EMA50 slope (over 5 bars)
            features[i, 6] = (ema50[i] - ema50[max(0, i - 5)]) / closes[i] if closes[i] > 0 else 0.0

        # ── Normalise features (rolling z-score, window=100) ──
        norm_start = max(100, 50)
        norm_features = np.zeros_like(features)
        for i in range(norm_start, n):
            window = features[max(0, i - 99):i + 1]
            mu = window.mean(axis=0)
            sigma = window.std(axis=0)
            sigma[sigma == 0] = 1.0
            norm_features[i] = (features[i] - mu) / sigma

        # ── Build labelled training set ──
        # Only bars where we know the future (up to n - horizon)
        horizon = self.horizon_bars
        train_end = n - horizon  # last bar we can label
        valid_start = norm_start

        if train_end - valid_start < self.k_neighbors + 10:
            return ComponentSignal("LC", "NEUTRAL", 0.0, "not enough data")

        # Labels: sign of future return
        labels = np.zeros(n, dtype=np.int8)
        for i in range(valid_start, train_end):
            future_ret = closes[min(i + horizon, n - 1)] / closes[i] - 1.0 if closes[i] > 0 else 0.0
            labels[i] = 1 if future_ret > 0 else (-1 if future_ret < 0 else 0)

        # ── Predict for last bar ──
        query = norm_features[n - 1]
        train_features = norm_features[valid_start:train_end]
        train_labels = labels[valid_start:train_end]

        # Lorentzian distance: sum of log(1 + |xi - yi|)
        diffs = np.abs(train_features - query)
        distances = np.sum(np.log1p(diffs), axis=1)

        # k nearest neighbours
        k = min(self.k_neighbors, len(distances))
        idx = np.argpartition(distances, k)[:k]
        nn_labels = train_labels[idx]

        up_count = np.sum(nn_labels == 1)
        down_count = np.sum(nn_labels == -1)
        total = up_count + down_count
        if total == 0:
            return ComponentSignal("LC", "NEUTRAL", 0.0, f"k={k} all neutral")

        up_pct = up_count / total
        down_pct = down_count / total
        conf = max(up_pct, down_pct)

        if conf < self.min_confidence:
            return ComponentSignal("LC", "NEUTRAL", conf, f"low conf {conf:.0%}")
        if up_pct > down_pct:
            return ComponentSignal("LC", "UP", conf, f"k={k} up={up_count} dn={down_count}")
        else:
            return ComponentSignal("LC", "DOWN", conf, f"k={k} up={up_count} dn={down_count}")

    # ── 2. SuperTrend ────────────────────────────────────────────────────────

    def _compute_supertrend(self, ohlcv: np.ndarray) -> ComponentSignal:
        """ATR-based SuperTrend."""
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]
        n = len(closes)
        period = self.st_period
        mult = self.st_multiplier

        atr = self._atr(highs, lows, closes, period)

        # SuperTrend calculation
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        supertrend = np.zeros(n)
        direction = np.ones(n, dtype=np.int8)  # 1 = UP, -1 = DOWN

        for i in range(period, n):
            hl2 = (highs[i] + lows[i]) / 2.0
            upper_band[i] = hl2 + mult * atr[i]
            lower_band[i] = hl2 - mult * atr[i]

            # Keep band levels (prevent them from moving against trend)
            if i > period:
                if lower_band[i] < lower_band[i - 1] and closes[i - 1] > lower_band[i - 1]:
                    lower_band[i] = lower_band[i - 1]
                if upper_band[i] > upper_band[i - 1] and closes[i - 1] < upper_band[i - 1]:
                    upper_band[i] = upper_band[i - 1]

            # Direction
            if i > period:
                if direction[i - 1] == 1:
                    direction[i] = 1 if closes[i] > lower_band[i] else -1
                else:
                    direction[i] = -1 if closes[i] < upper_band[i] else 1

            supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

        d = direction[-1]
        dist = abs(closes[-1] - supertrend[-1]) / closes[-1] if closes[-1] > 0 else 0
        strength = min(1.0, dist * 20)  # normalise
        if d == 1:
            return ComponentSignal("ST", "UP", strength, f"above ST, dist={dist:.4f}")
        else:
            return ComponentSignal("ST", "DOWN", strength, f"below ST, dist={dist:.4f}")

    # ── 3. EMA200 ────────────────────────────────────────────────────────────

    def _compute_ema200(self, ohlcv: np.ndarray) -> ComponentSignal:
        """EMA200 trend filter. Close vs EMA determines direction; slope affects strength."""
        closes = ohlcv[:, 3]
        ema = self._ema(closes, self.ema_period)

        close = closes[-1]
        ema_val = ema[-1]
        slope = ema[-1] - ema[-6] if len(ema) > 6 else 0.0

        if ema_val <= 0:
            return ComponentSignal("EMA200", "NEUTRAL", 0.0, "invalid ema")

        distance_pct = abs(close - ema_val) / ema_val * 100
        # Dead zone: within 0.05% of EMA → NEUTRAL (very tight)
        if distance_pct < 0.05:
            return ComponentSignal("EMA200", "NEUTRAL", 0.0, f"too close: {distance_pct:.3f}%")

        base_strength = min(1.0, distance_pct / 2.0)
        # Slope confirms or weakens (but doesn't flip direction)
        slope_factor = 1.0 if (close > ema_val and slope >= 0) or (close < ema_val and slope <= 0) else 0.6

        if close > ema_val:
            return ComponentSignal("EMA200", "UP", base_strength * slope_factor,
                                   f"close={close:.2f} > ema={ema_val:.2f}, slope={slope:+.4f}")
        else:
            return ComponentSignal("EMA200", "DOWN", base_strength * slope_factor,
                                   f"close={close:.2f} < ema={ema_val:.2f}, slope={slope:+.4f}")

    # ── 4. Volume Profile ────────────────────────────────────────────────────

    def _compute_volume_profile(self, ohlcv: np.ndarray) -> ComponentSignal:
        """Simplified volume profile: POC and value area."""
        lookback = min(self.vp_lookback, len(ohlcv))
        data = ohlcv[-lookback:]
        highs = data[:, 1]
        lows = data[:, 2]
        closes = data[:, 3]
        volumes = data[:, 4]

        # Typical price
        typical = (highs + lows + closes) / 3.0

        price_min = typical.min()
        price_max = typical.max()
        price_range = price_max - price_min
        if price_range <= 0:
            return ComponentSignal("VP", "NEUTRAL", 0.0, "no price range")

        # Build histogram
        bins = self.vp_bins
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        vol_hist = np.zeros(bins)
        for i in range(len(typical)):
            idx = int((typical[i] - price_min) / price_range * (bins - 1))
            idx = max(0, min(bins - 1, idx))
            vol_hist[idx] += volumes[i]

        # POC = bin with highest volume
        poc_idx = np.argmax(vol_hist)
        poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0

        # Value Area (70% of volume)
        total_vol = vol_hist.sum()
        target_vol = total_vol * self.vp_value_area_pct

        # Expand from POC
        lo_idx = poc_idx
        hi_idx = poc_idx
        va_vol = vol_hist[poc_idx]
        while va_vol < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
            expand_low = vol_hist[lo_idx - 1] if lo_idx > 0 else 0
            expand_high = vol_hist[hi_idx + 1] if hi_idx < bins - 1 else 0
            if expand_low >= expand_high and lo_idx > 0:
                lo_idx -= 1
                va_vol += vol_hist[lo_idx]
            elif hi_idx < bins - 1:
                hi_idx += 1
                va_vol += vol_hist[hi_idx]
            else:
                lo_idx -= 1
                va_vol += vol_hist[lo_idx]

        val = (bin_edges[lo_idx] + bin_edges[lo_idx + 1]) / 2.0
        vah = (bin_edges[hi_idx] + bin_edges[hi_idx + 1]) / 2.0

        # Acceptance: 3/5 of last closes above/below POC
        last5 = closes[-5:]
        above_poc = sum(1 for c in last5 if c > poc_price)
        below_poc = sum(1 for c in last5 if c < poc_price)
        current = closes[-1]

        if current > poc_price and above_poc >= 2:
            strength = min(1.0, (current - poc_price) / price_range * 10)
            return ComponentSignal("VP", "UP", strength, f"close>{poc_price:.2f} POC, {above_poc}/5 above")
        elif current < poc_price and below_poc >= 2:
            strength = min(1.0, (poc_price - current) / price_range * 10)
            return ComponentSignal("VP", "DOWN", strength, f"close<{poc_price:.2f} POC, {below_poc}/5 below")
        else:
            return ComponentSignal("VP", "NEUTRAL", 0.0, f"close≈POC={poc_price:.2f}, VAL={val:.2f}, VAH={vah:.2f}")

    # ── 5. ICT/SMC (BOS + FVG) ──────────────────────────────────────────────

    def _compute_smc(self, ohlcv: np.ndarray) -> ComponentSignal:
        """Break of Structure + Fair Value Gap detection."""
        lookback = min(self.bos_lookback, len(ohlcv))
        data = ohlcv[-lookback:]
        highs = data[:, 1]
        lows = data[:, 2]
        closes = data[:, 3]
        n = len(data)
        pl = self.pivot_left
        pr = self.pivot_right

        # ── Find swing highs and lows ──
        swing_highs = []  # (index, price)
        swing_lows = []

        # Only confirmed pivots (need pr bars after pivot)
        for i in range(pl, n - pr):
            is_high = True
            is_low = True
            for j in range(1, pl + 1):
                if highs[i] <= highs[i - j]:
                    is_high = False
                if lows[i] >= lows[i - j]:
                    is_low = False
            for j in range(1, pr + 1):
                if highs[i] <= highs[i + j]:
                    is_high = False
                if lows[i] >= lows[i + j]:
                    is_low = False
            if is_high:
                swing_highs.append((i, highs[i]))
            if is_low:
                swing_lows.append((i, lows[i]))

        # ── BOS detection ──
        bos_bullish = False
        bos_bearish = False
        last_close = closes[-1]

        if swing_highs:
            last_sh = swing_highs[-1]
            if last_close > last_sh[1]:
                bos_bullish = True

        if swing_lows:
            last_sl = swing_lows[-1]
            if last_close < last_sl[1]:
                bos_bearish = True

        # ── FVG detection (last 20 bars) ──
        bullish_fvg = 0
        bearish_fvg = 0
        fvg_start = max(2, n - 20)
        for i in range(fvg_start, n):
            # Bullish FVG: low[i] > high[i-2]
            if lows[i] > highs[i - 2]:
                bullish_fvg += 1
            # Bearish FVG: high[i] < low[i-2]
            if highs[i] < lows[i - 2]:
                bearish_fvg += 1

        # ── Aggregate SMC signal ──
        score = 0
        if bos_bullish:
            score += 1
        if bos_bearish:
            score -= 1
        score += (bullish_fvg - bearish_fvg) * 0.3  # FVG weaker

        details = f"BOS bull={bos_bullish} bear={bos_bearish}, FVG bull={bullish_fvg} bear={bearish_fvg}"

        if score > 0.2:
            return ComponentSignal("SMC", "UP", min(1.0, abs(score)), details)
        elif score < -0.2:
            return ComponentSignal("SMC", "DOWN", min(1.0, abs(score)), details)
        else:
            return ComponentSignal("SMC", "NEUTRAL", 0.0, details)

    # ── Technical helpers ────────────────────────────────────────────────────

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        result = np.zeros_like(data)
        multiplier = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        n = len(closes)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Average True Range."""
        n = len(closes)
        tr = np.zeros(n)
        atr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        # SMA seed
        if n > period:
            atr[period] = tr[1:period + 1].mean()
            for i in range(period + 1, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    # ── Public API ───────────────────────────────────────────────────────────

    def get_trend(self, symbol: str) -> Optional[str]:
        """Return UP / DOWN / None (None means NEUTRAL/unavailable)."""
        signal = self.get_signal(symbol)
        if signal and signal.direction in ("UP", "DOWN"):
            return signal.direction
        return None

    def get_signal(self, symbol: str) -> Optional[LorentzianSignal]:
        """Full signal with components and diagnostics."""
        try:
            t0 = time.time()
            ohlcv = self._fetch_klines(symbol)
            if ohlcv is None or len(ohlcv) < self.ema_period + 50:
                logger.warning(f"L: insufficient data for {symbol}")
                return LorentzianSignal("NEUTRAL", 0.0, 0.0)

            # Compute all 5 components
            lc = self._compute_lc(ohlcv)
            st = self._compute_supertrend(ohlcv)
            ema = self._compute_ema200(ohlcv)
            vp = self._compute_volume_profile(ohlcv)
            smc = self._compute_smc(ohlcv)

            components = [lc, st, ema, vp, smc]
            weights = [self.W_LC, self.W_ST, self.W_EMA, self.W_VP, self.W_SMC]

            # Weighted score
            score = 0.0
            for comp, w in zip(components, weights):
                if comp.direction == "UP":
                    score += w
                elif comp.direction == "DOWN":
                    score -= w

            # Direction
            if score >= self.threshold:
                direction = "UP"
            elif score <= -self.threshold:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            confidence = abs(score) / self.TOTAL_WEIGHT

            dt = time.time() - t0
            logger.info(
                f"L({symbol}): {direction} score={score:+.1f} conf={confidence:.0%} "
                f"[LC={lc.direction} ST={st.direction} EMA={ema.direction} VP={vp.direction} SMC={smc.direction}] "
                f"({dt:.2f}s)"
            )

            return LorentzianSignal(direction, score, confidence, components)

        except Exception as e:
            logger.error(f"Lorentzian error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return LorentzianSignal("NEUTRAL", 0.0, 0.0)
