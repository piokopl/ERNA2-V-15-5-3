#!/usr/bin/env python3
"""Multi-Indicator Client (M) - Combined signal from 5 indicators.

Requires at least 3 indicators to align for a signal:
1. TA Predict (LONG/SHORT with confidence %)
2. Heiken Ashi (2+ consecutive red/green candles)
3. RSI (with extreme zone detection)
4. MACD (bullish/bearish crossover)
5. Delta 1m/3m (volume delta - buy vs sell pressure)

Rules:
- Rule 1: At least 3 indicators must align
- Rule 2: RSI < 25 or > 75 = reversal zone (don't chase)
- Rule 3: Delta confirms - if TA says LONG but Delta negative = weak signal
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import httpx

logger = logging.getLogger(__name__)


@dataclass
class IndicatorSignal:
    """Single indicator signal."""
    name: str
    direction: str  # "UP", "DOWN", "NEUTRAL"
    strength: float  # 0.0 - 1.0
    value: Optional[float] = None
    details: str = ""


@dataclass 
class MultiSignal:
    """Combined multi-indicator signal."""
    direction: str  # "UP", "DOWN", "NEUTRAL"
    aligned_count: int  # How many indicators align
    confidence: float  # 0.0 - 1.0
    signals: List[IndicatorSignal]
    is_reversal_zone: bool  # RSI in extreme zone
    is_confirmed: bool  # Delta confirms the direction
    
    def __str__(self):
        sigs = ", ".join([f"{s.name}={s.direction}" for s in self.signals])
        return f"Multi({self.direction}, {self.aligned_count}/5 aligned, conf={self.confidence:.0%}, reversal={self.is_reversal_zone}) [{sigs}]"


class MultiIndicatorClient:
    """Multi-indicator trend analyzer."""
    
    BINANCE_API = "https://api.binance.com/api/v3"
    TA_PREDICT_API = "https://mimir.cryptothunder.net/ta_predict"  # Example API
    
    def __init__(
        self,
        min_aligned: int = 3,
        ta_min_confidence: float = 0.75,
        rsi_period: int = 14,
        rsi_overbought: float = 75,
        rsi_oversold: float = 25,
        heiken_candles: int = 2,
        timeout: int = 10
    ):
        self.min_aligned = min_aligned
        self.ta_min_confidence = ta_min_confidence
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.heiken_candles = heiken_candles
        self.timeout = timeout
        
        self._session = httpx.Client(timeout=timeout)
    
    def _fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> List[dict]:
        """Fetch OHLCV data from Binance."""
        try:
            # Binance spot does not support 10m. Be defensive if config or callers pass it.
            if interval == "10m":
                interval = "15m"

            binance_symbol = symbol.replace("/", "")
            resp = self._session.get(
                f"{self.BINANCE_API}/klines",
                params={"symbol": binance_symbol, "interval": interval, "limit": limit}
            )
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            return [
                {
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "taker_buy_volume": float(k[9]),
                    "taker_buy_quote_volume": float(k[10]),
                }
                for k in data
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch klines: {e}")
            return []
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_heiken_ashi(self, klines: List[dict]) -> List[dict]:
        """Calculate Heiken Ashi candles."""
        if not klines:
            return []
        
        ha = []
        for i, k in enumerate(klines):
            if i == 0:
                ha_open = (k["open"] + k["close"]) / 2
            else:
                ha_open = (ha[-1]["open"] + ha[-1]["close"]) / 2
            
            ha_close = (k["open"] + k["high"] + k["low"] + k["close"]) / 4
            ha_high = max(k["high"], ha_open, ha_close)
            ha_low = min(k["low"], ha_open, ha_close)
            
            ha.append({
                "open": ha_open,
                "high": ha_high,
                "low": ha_low,
                "close": ha_close,
                "color": "green" if ha_close > ha_open else "red"
            })
        
        return ha
    
    def _calculate_macd(self, closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD."""
        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0
        
        def ema(data, period):
            if len(data) < period:
                return data[-1] if data else 0
            multiplier = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for price in data[period:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val
        
        fast_ema = ema(closes, fast)
        slow_ema = ema(closes, slow)
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        macd_values = []
        for i in range(slow, len(closes) + 1):
            f = ema(closes[:i], fast)
            s = ema(closes[:i], slow)
            macd_values.append(f - s)
        
        signal_line = ema(macd_values, signal) if len(macd_values) >= signal else 0
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_delta(self, klines: List[dict]) -> float:
        """Calculate volume delta (buy - sell pressure)."""
        if not klines:
            return 0.0
        
        total_delta = 0.0
        for k in klines:
            # taker_buy_volume = aggressive buyers
            # total_volume - taker_buy = aggressive sellers
            buy_vol = k.get("taker_buy_volume", 0)
            total_vol = k.get("volume", 0)
            sell_vol = total_vol - buy_vol
            total_delta += (buy_vol - sell_vol)
        
        return total_delta
    
    def _get_ta_predict(self, symbol: str) -> IndicatorSignal:
        """Get TA prediction from external API."""
        try:
            # Try to get prediction from API
            binance_symbol = symbol.replace("/", "")
            resp = self._session.get(
                self.TA_PREDICT_API,
                params={"symbol": binance_symbol},
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                direction = data.get("direction", "").upper()
                confidence = float(data.get("confidence", 0))
                
                if direction in ("LONG", "UP") and confidence >= self.ta_min_confidence:
                    return IndicatorSignal("TA_Predict", "UP", confidence, confidence, f"{confidence:.0%} LONG")
                elif direction in ("SHORT", "DOWN") and confidence >= self.ta_min_confidence:
                    return IndicatorSignal("TA_Predict", "DOWN", confidence, confidence, f"{confidence:.0%} SHORT")
                else:
                    return IndicatorSignal("TA_Predict", "NEUTRAL", confidence, confidence, f"{confidence:.0%} {direction}")
            
        except Exception as e:
            logger.debug(f"TA Predict API not available: {e}")
        
        # Fallback: Use simple price momentum
        klines = self._fetch_klines(symbol, "5m", 12)
        if klines and len(klines) >= 2:
            closes = [k["close"] for k in klines]
            change = (closes[-1] - closes[0]) / closes[0]
            
            if change > 0.002:  # >0.2% up
                return IndicatorSignal("TA_Predict", "UP", min(abs(change) * 50, 1.0), change, f"+{change:.2%}")
            elif change < -0.002:  # >0.2% down
                return IndicatorSignal("TA_Predict", "DOWN", min(abs(change) * 50, 1.0), change, f"{change:.2%}")
        
        return IndicatorSignal("TA_Predict", "NEUTRAL", 0.5, 0, "no data")
    
    def _get_heiken_signal(self, symbol: str) -> IndicatorSignal:
        """Get Heiken Ashi signal."""
        klines = self._fetch_klines(symbol, "3m", 20)
        if not klines:
            return IndicatorSignal("Heiken_Ashi", "NEUTRAL", 0.5, 0, "no data")
        
        ha = self._calculate_heiken_ashi(klines)
        if len(ha) < self.heiken_candles:
            return IndicatorSignal("Heiken_Ashi", "NEUTRAL", 0.5, 0, "insufficient data")
        
        # Count consecutive same-color candles at the end
        recent = ha[-5:]  # Last 5 candles
        green_count = 0
        red_count = 0
        
        # Count consecutive from the end
        for candle in reversed(recent):
            if candle["color"] == "green":
                if red_count > 0:
                    break
                green_count += 1
            else:
                if green_count > 0:
                    break
                red_count += 1
        
        if green_count >= self.heiken_candles:
            return IndicatorSignal("Heiken_Ashi", "UP", 0.8, green_count, f"green x{green_count}")
        elif red_count >= self.heiken_candles:
            return IndicatorSignal("Heiken_Ashi", "DOWN", 0.8, red_count, f"red x{red_count}")
        
        return IndicatorSignal("Heiken_Ashi", "NEUTRAL", 0.4, 0, "mixed")
    
    def _get_rsi_signal(self, symbol: str) -> Tuple[IndicatorSignal, bool]:
        """Get RSI signal and reversal zone flag.
        
        Rule 2: RSI < 25 or > 75 = high probability of reversal, don't chase!
        """
        klines = self._fetch_klines(symbol, "3m", 50)
        if not klines:
            return IndicatorSignal("RSI", "NEUTRAL", 0.5, 50, "no data"), False
        
        closes = [k["close"] for k in klines]
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        is_reversal_zone = rsi < self.rsi_oversold or rsi > self.rsi_overbought
        
        if rsi < self.rsi_oversold:
            # Oversold - DON'T chase down, expect reversal UP
            return IndicatorSignal("RSI", "UP", 0.3, rsi, f"RSI={rsi:.0f} OVERSOLD⚠️"), True
        elif rsi > self.rsi_overbought:
            # Overbought - DON'T chase up, expect reversal DOWN
            return IndicatorSignal("RSI", "DOWN", 0.3, rsi, f"RSI={rsi:.0f} OVERBOUGHT⚠️"), True
        elif rsi >= 60 and rsi <= 75:
            # 60-75: bullish but not extreme
            return IndicatorSignal("RSI", "UP", 0.7, rsi, f"RSI={rsi:.0f} bullish"), False
        elif rsi >= 25 and rsi <= 40:
            # 25-40: bearish but not extreme
            return IndicatorSignal("RSI", "DOWN", 0.7, rsi, f"RSI={rsi:.0f} bearish"), False
        else:
            # 40-60: neutral zone
            return IndicatorSignal("RSI", "NEUTRAL", 0.5, rsi, f"RSI={rsi:.0f} neutral"), False
    
    def _get_macd_signal(self, symbol: str) -> IndicatorSignal:
        """Get MACD signal."""
        klines = self._fetch_klines(symbol, "3m", 50)
        if not klines:
            return IndicatorSignal("MACD", "NEUTRAL", 0.5, 0, "no data")
        
        closes = [k["close"] for k in klines]
        macd_line, signal_line, histogram = self._calculate_macd(closes)
        
        if histogram > 0 and macd_line > signal_line:
            strength = min(abs(histogram) * 1000, 1.0)
            return IndicatorSignal("MACD", "UP", max(strength, 0.6), histogram, f"bullish")
        elif histogram < 0 and macd_line < signal_line:
            strength = min(abs(histogram) * 1000, 1.0)
            return IndicatorSignal("MACD", "DOWN", max(strength, 0.6), histogram, f"bearish")
        
        return IndicatorSignal("MACD", "NEUTRAL", 0.4, histogram, f"neutral")
    
    def _get_delta_signal(self, symbol: str) -> Tuple[IndicatorSignal, IndicatorSignal, bool]:
        """Get Delta signals for 1m and 3m.
        
        Rule 3: Both deltas must align for confirmation.
        """
        # 1m delta
        klines_1m = self._fetch_klines(symbol, "1m", 5)
        delta_1m = self._calculate_delta(klines_1m) if klines_1m else 0
        
        if delta_1m > 0:
            dir_1m = "UP"
        elif delta_1m < 0:
            dir_1m = "DOWN"
        else:
            dir_1m = "NEUTRAL"
        
        sig_1m = IndicatorSignal("Delta_1m", dir_1m, 0.7, delta_1m, f"{delta_1m:+.0f}")
        
        # 3m delta
        klines_3m = self._fetch_klines(symbol, "3m", 3)
        delta_3m = self._calculate_delta(klines_3m) if klines_3m else 0
        
        if delta_3m > 0:
            dir_3m = "UP"
        elif delta_3m < 0:
            dir_3m = "DOWN"
        else:
            dir_3m = "NEUTRAL"
        
        sig_3m = IndicatorSignal("Delta_3m", dir_3m, 0.7, delta_3m, f"{delta_3m:+.0f}")
        
        # Both must align for confirmation
        both_aligned = (dir_1m == dir_3m) and (dir_1m != "NEUTRAL")
        
        return sig_1m, sig_3m, both_aligned
    
    def get_trend(self, symbol: str) -> Optional[str]:
        """Get combined trend signal.
        
        Returns:
            "UP", "DOWN", or None if no clear signal
        """
        signal = self.get_signal(symbol)
        
        # Rule 2: Don't chase in reversal zones
        if signal.is_reversal_zone:
            logger.info(f"[Multi] {symbol}: RSI in reversal zone - NO SIGNAL")
            return None
        
        # Rule 1: Need at least 3 aligned
        if signal.aligned_count >= self.min_aligned:
            # Rule 3: Check delta confirmation
            if signal.is_confirmed:
                logger.info(f"[Multi] {symbol}: {signal.direction} ({signal.aligned_count}/5) CONFIRMED by Delta")
                return signal.direction
            else:
                # Delta doesn't confirm - need 4+ for weak signal
                if signal.aligned_count >= 4:
                    logger.info(f"[Multi] {symbol}: {signal.direction} ({signal.aligned_count}/5) weak (Delta mismatch)")
                    return signal.direction
                else:
                    logger.info(f"[Multi] {symbol}: {signal.direction} ({signal.aligned_count}/5) - Delta mismatch, need 4+")
                    return None
        
        logger.info(f"[Multi] {symbol}: Only {signal.aligned_count}/5 aligned - NO SIGNAL")
        return None
    
    def get_signal(self, symbol: str) -> MultiSignal:
        """Get detailed multi-indicator signal."""
        signals = []
        
        # 1. TA Predict
        ta_signal = self._get_ta_predict(symbol)
        signals.append(ta_signal)
        logger.info(f"  📊 TA_Predict: {ta_signal.direction} ({ta_signal.details})")
        
        # 2. Heiken Ashi
        ha_signal = self._get_heiken_signal(symbol)
        signals.append(ha_signal)
        logger.info(f"  📊 Heiken_Ashi: {ha_signal.direction} ({ha_signal.details})")
        
        # 3. RSI
        rsi_signal, is_reversal_zone = self._get_rsi_signal(symbol)
        signals.append(rsi_signal)
        logger.info(f"  📊 RSI: {rsi_signal.direction} ({rsi_signal.details})")
        
        # 4. MACD
        macd_signal = self._get_macd_signal(symbol)
        signals.append(macd_signal)
        logger.info(f"  📊 MACD: {macd_signal.direction} ({macd_signal.details})")
        
        # 5. Delta (1m + 3m)
        delta_1m, delta_3m, delta_confirmed = self._get_delta_signal(symbol)
        
        # Combined delta direction
        if delta_1m.direction == delta_3m.direction and delta_1m.direction != "NEUTRAL":
            delta_combined = IndicatorSignal(
                "Delta", delta_1m.direction, 0.8, 
                (delta_1m.value or 0) + (delta_3m.value or 0),
                f"1m:{delta_1m.details} 3m:{delta_3m.details}"
            )
        else:
            delta_combined = IndicatorSignal("Delta", "NEUTRAL", 0.4, 0, f"1m:{delta_1m.direction} 3m:{delta_3m.direction}")
        signals.append(delta_combined)
        logger.info(f"  📊 Delta: {delta_combined.direction} ({delta_combined.details})")
        
        # Count aligned signals
        up_count = sum(1 for s in signals if s.direction == "UP")
        down_count = sum(1 for s in signals if s.direction == "DOWN")
        
        if up_count >= down_count and up_count >= self.min_aligned:
            direction = "UP"
            aligned_count = up_count
        elif down_count > up_count and down_count >= self.min_aligned:
            direction = "DOWN"
            aligned_count = down_count
        else:
            direction = "NEUTRAL"
            aligned_count = max(up_count, down_count)
        
        # Calculate confidence
        aligned_signals = [s for s in signals if s.direction == direction]
        confidence = sum(s.strength for s in aligned_signals) / len(signals) if signals else 0
        
        # Check if Delta confirms direction
        is_confirmed = delta_combined.direction == direction and delta_confirmed
        
        logger.info(f"  ═══ RESULT: {direction} ({aligned_count}/5) reversal={is_reversal_zone} delta_confirm={is_confirmed}")
        
        return MultiSignal(
            direction=direction,
            aligned_count=aligned_count,
            confidence=confidence,
            signals=signals,
            is_reversal_zone=is_reversal_zone,
            is_confirmed=is_confirmed
        )


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    client = MultiIndicatorClient()
    
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print('='*60)
        
        trend = client.get_trend(symbol)
        print(f"\n>>> FINAL TREND: {trend}")
