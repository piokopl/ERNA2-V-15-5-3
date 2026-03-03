#!/usr/bin/env python3
"""VKI Client - RSI (Kernel Optimized) trend indicator.

Based on Pine Script "RSI (Kernel Optimized) | Flux Charts".
Uses KDE (Kernel Density Estimation) to optimize RSI pivot detection.

Analyzes multiple timeframes (15m, 5m, 1m) and aggregates signals.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class VKIResult:
    """Result from VKI analysis for single timeframe."""
    timeframe: str
    rsi: float
    high_prob: float  # Probability of bearish pivot
    low_prob: float   # Probability of bullish pivot
    signal: Optional[str]  # "UP", "DOWN", or None
    kde_high_sum: float
    kde_low_sum: float


class VKIClient:
    """
    VKI (Volatility Kernel Indicator) Client.
    
    Implements RSI with KDE optimization from Pine Script.
    Analyzes 15m, 5m, 1m timeframes and aggregates signals.
    """
    
    # KDE Settings (from Pine Script defaults)
    RSI_LENGTH = 14
    PIVOT_LENGTH = 10  # Reduced from 21 for better pivot detection with limited data
    KDE_BANDWIDTH = 2.71828  # e
    KDE_STEPS = 100
    ACTIVATION_THRESHOLD = 0.25  # Medium
    KDE_LIMIT = 300
    
    # Timeframes to analyze (must be supported by Binance: 1m, 3m, 5m, 15m, 30m, 1h, etc.)
    TIMEFRAMES = ["15m", "5m", "3m"]

    # Binance-supported intervals (spot) + a small compatibility map for common mistakes.
    _BINANCE_INTERVALS = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M",
    }
    _INTERVAL_COMPAT = {
        # Binance does not support 10m; map to nearest higher timeframe.
        "10m": "5m",
    }
    
    # Binance API
    BINANCE_API = "https://api.binance.com/api/v3/klines"
    
    def __init__(
        self,
        rsi_length: int = 14,
        pivot_length: int = 21,
        bandwidth: float = 2.71828,
        activation_threshold: float = 0.25,
        timeframes: Optional[List[str]] = None,
        timeout: int = 10,
    ):
        self.rsi_length = rsi_length
        self.pivot_length = pivot_length
        self.bandwidth = bandwidth
        self.activation_threshold = activation_threshold
        self.timeframes = timeframes or self.TIMEFRAMES
        self.timeout = timeout
        
        # Cache for pivot RSI values (simulates Pine Script vars)
        self._high_pivot_rsis: dict[str, List[float]] = {}
        self._low_pivot_rsis: dict[str, List[float]] = {}
    
    def _fetch_klines(self, symbol: str, interval: str, limit: int = 500) -> List[dict]:
        """Fetch OHLCV data from Binance."""
        try:
            # Be defensive: if someone configured an unsupported interval (e.g. 10m), map it.
            interval = self._INTERVAL_COMPAT.get(interval, interval)
            if interval not in self._BINANCE_INTERVALS:
                logger.warning(f"Unsupported Binance interval '{interval}' for VKI; falling back to 15m")
                interval = "15m"

            # Convert symbol format: "BTC/USDT" -> "BTCUSDT"
            binance_symbol = symbol.replace("/", "")
            
            resp = requests.get(
                self.BINANCE_API,
                params={
                    "symbol": binance_symbol,
                    "interval": interval,
                    "limit": limit,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            
            klines = []
            for k in resp.json():
                klines.append({
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "timestamp": k[0],
                })
            return klines
            
        except Exception as e:
            logger.warning(f"Failed to fetch klines for {symbol} {interval}: {e}")
            return []
    
    def _calculate_rsi(self, closes: List[float], length: int = 14) -> List[float]:
        """Calculate RSI series."""
        if len(closes) < length + 1:
            return []
        
        rsi_values = []
        
        # Calculate price changes
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Initial average gain/loss (SMA)
        gains = [max(0, c) for c in changes[:length]]
        losses = [abs(min(0, c)) for c in changes[:length]]
        
        avg_gain = sum(gains) / length
        avg_loss = sum(losses) / length
        
        # First RSI
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        # Subsequent RSI values using Wilder's smoothing
        for i in range(length, len(changes)):
            change = changes[i]
            gain = max(0, change)
            loss = abs(min(0, change))
            
            avg_gain = (avg_gain * (length - 1) + gain) / length
            avg_loss = (avg_loss * (length - 1) + loss) / length
            
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    def _detect_pivots(self, highs: List[float], lows: List[float], length: int) -> Tuple[List[int], List[int]]:
        """
        Detect pivot highs and lows.
        Returns indices of pivots (offset by length for lookback).
        """
        high_pivots = []
        low_pivots = []
        
        # Need at least 2*length+1 bars
        if len(highs) < 2 * length + 1:
            return [], []
        
        for i in range(length, len(highs) - length):
            # Check pivot high
            is_high_pivot = True
            for j in range(i - length, i + length + 1):
                if j != i and highs[j] >= highs[i]:
                    is_high_pivot = False
                    break
            if is_high_pivot:
                high_pivots.append(i)
            
            # Check pivot low
            is_low_pivot = True
            for j in range(i - length, i + length + 1):
                if j != i and lows[j] <= lows[i]:
                    is_low_pivot = False
                    break
            if is_low_pivot:
                low_pivots.append(i)
        
        return high_pivots, low_pivots
    
    def _gaussian_kernel(self, distance: float, bandwidth: float = 1.0) -> float:
        """Gaussian kernel function."""
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * (distance / bandwidth) ** 2)
    
    def _kde(self, arr: List[float], bandwidth: float, steps: int) -> Tuple[List[float], List[float]]:
        """
        Kernel Density Estimation.
        Returns (x_values, y_values/densities).
        """
        if not arr:
            return [], []
        
        arr_min = min(arr)
        arr_max = max(arr)
        arr_range = arr_max - arr_min
        
        if arr_range == 0:
            arr_range = 1.0  # Avoid division by zero
        
        step_size = arr_range / steps
        
        # Create density range (steps * 2 points like in Pine Script)
        density_range = [arr_min + i * step_size for i in range(steps * 2)]
        
        x_arr = []
        y_arr = []
        
        inv_bandwidth = 1.0 / bandwidth
        arr_size = len(arr)
        
        for x in density_range:
            temp = 0.0
            for val in arr:
                temp += self._gaussian_kernel(x - val, inv_bandwidth)
            
            x_arr.append(x)
            y_arr.append((1.0 / arr_size) * temp)
        
        return x_arr, y_arr
    
    def _prefix_sum(self, arr: List[float]) -> List[float]:
        """Calculate prefix sum array."""
        result = []
        total = 0.0
        for val in arr:
            total += val
            result.append(total)
        return result
    
    def _binary_search_nearest(self, arr: List[float], target: float) -> int:
        """Find index of nearest value in sorted array."""
        if not arr:
            return 0
        
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        # Check neighbors for nearest
        best_idx = left
        best_diff = abs(arr[left] - target)
        
        if left > 0 and abs(arr[left - 1] - target) < best_diff:
            best_idx = left - 1
        if left < len(arr) - 1 and abs(arr[left + 1] - target) < best_diff:
            best_idx = left + 1
        
        return best_idx
    
    def _analyze_timeframe(self, symbol: str, interval: str) -> Optional[VKIResult]:
        """Analyze single timeframe and return VKI result."""
        # Fetch data
        klines = self._fetch_klines(symbol, interval, limit=500)
        if len(klines) < self.pivot_length * 2 + self.rsi_length + 50:
            logger.warning(f"Insufficient data for {symbol} {interval}: got {len(klines)} klines")
            return None
        
        closes = [k["close"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        
        # Calculate RSI
        rsi_values = self._calculate_rsi(closes, self.rsi_length)
        if not rsi_values:
            logger.warning(f"Failed to calculate RSI for {symbol} {interval}")
            return None
        
        # Align RSI with price data (RSI starts from index rsi_length)
        rsi_offset = len(closes) - len(rsi_values)
        
        # Detect pivots
        high_pivots, low_pivots = self._detect_pivots(highs, lows, self.pivot_length)
        logger.debug(f"{symbol} {interval}: Found {len(high_pivots)} high pivots, {len(low_pivots)} low pivots")
        
        # Initialize pivot RSI lists for this timeframe
        cache_key = f"{symbol}_{interval}"
        if cache_key not in self._high_pivot_rsis:
            self._high_pivot_rsis[cache_key] = []
            self._low_pivot_rsis[cache_key] = []
        
        # Collect RSI values at pivot points
        high_pivot_rsis = self._high_pivot_rsis[cache_key]
        low_pivot_rsis = self._low_pivot_rsis[cache_key]
        
        for idx in high_pivots:
            rsi_idx = idx - rsi_offset
            if 0 <= rsi_idx < len(rsi_values):
                if len(high_pivot_rsis) >= self.KDE_LIMIT:
                    high_pivot_rsis.pop(0)
                high_pivot_rsis.append(rsi_values[rsi_idx])
        
        for idx in low_pivots:
            rsi_idx = idx - rsi_offset
            if 0 <= rsi_idx < len(rsi_values):
                if len(low_pivot_rsis) >= self.KDE_LIMIT:
                    low_pivot_rsis.pop(0)
                low_pivot_rsis.append(rsi_values[rsi_idx])
        
        # Current RSI
        current_rsi = rsi_values[-1]
        
        # Calculate KDE for high pivots (bearish)
        high_prob = 0.0
        kde_high_sum = 0.0
        if high_pivot_rsis:
            kde_high_x, kde_high_y = self._kde(high_pivot_rsis, self.bandwidth, self.KDE_STEPS)
            if kde_high_x and kde_high_y:
                kde_high_y_sum = self._prefix_sum(kde_high_y)
                nearest_idx = self._binary_search_nearest(kde_high_x, current_rsi)
                nearest_idx = min(nearest_idx, len(kde_high_y_sum) - 1)
                high_prob = kde_high_y_sum[nearest_idx] if nearest_idx >= 0 else 0.0
                kde_high_sum = sum(kde_high_y)
        
        # Calculate KDE for low pivots (bullish)
        low_prob = 0.0
        kde_low_sum = 0.0
        if low_pivot_rsis:
            kde_low_x, kde_low_y = self._kde(low_pivot_rsis, self.bandwidth, self.KDE_STEPS)
            if kde_low_x and kde_low_y:
                kde_low_y_sum = self._prefix_sum(kde_low_y)
                nearest_idx = self._binary_search_nearest(kde_low_x, current_rsi)
                nearest_idx = min(nearest_idx, len(kde_low_y_sum) - 1)
                # For low pivots, we want probability from nearest to end
                if nearest_idx >= 0 and nearest_idx < len(kde_low_y_sum):
                    low_prob = kde_low_y_sum[-1] - (kde_low_y_sum[nearest_idx - 1] if nearest_idx > 0 else 0)
                kde_low_sum = sum(kde_low_y)
        
        # Determine signal based on KDE probabilities
        signal = None
        
        # Log probabilities for debugging
        logger.info(f"{symbol} {interval}: RSI={current_rsi:.1f}, high_prob={high_prob:.4f}, low_prob={low_prob:.4f}, kde_high_sum={kde_high_sum:.4f}, kde_low_sum={kde_low_sum:.4f}")
        
        # Strategy: Compare relative probabilities
        # Higher low_prob means current RSI is in area where low pivots (reversals up) happened
        # Higher high_prob means current RSI is in area where high pivots (reversals down) happened
        
        if kde_low_sum > 0 and kde_high_sum > 0:
            # Normalize probabilities
            low_ratio = low_prob / kde_low_sum if kde_low_sum > 0 else 0
            high_ratio = high_prob / kde_high_sum if kde_high_sum > 0 else 0
            
            logger.info(f"{symbol} {interval}: low_ratio={low_ratio:.4f}, high_ratio={high_ratio:.4f}")
            
            # Use threshold of 0.5 (50%) - if we're in top half of probability distribution
            threshold = 0.5
            
            if low_ratio > threshold and low_ratio > high_ratio:
                signal = "UP"
            elif high_ratio > threshold and high_ratio > low_ratio:
                signal = "DOWN"
            elif low_ratio > high_ratio:
                # Lower threshold - just pick the stronger one
                signal = "UP"
            elif high_ratio > low_ratio:
                signal = "DOWN"
        elif kde_low_sum > 0:
            # Only have low pivot data
            low_ratio = low_prob / kde_low_sum
            if low_ratio > 0.5:
                signal = "UP"
        elif kde_high_sum > 0:
            # Only have high pivot data
            high_ratio = high_prob / kde_high_sum
            if high_ratio > 0.5:
                signal = "DOWN"
        
        # RSI extreme fallback - only strong signals, otherwise None
        if signal is None:
            if current_rsi <= 35:
                signal = "UP"  # Oversold zone - strong signal
                logger.info(f"{symbol} {interval}: RSI={current_rsi:.1f} <= 35 -> UP (oversold)")
            elif current_rsi >= 65:
                signal = "DOWN"  # Overbought zone - strong signal
                logger.info(f"{symbol} {interval}: RSI={current_rsi:.1f} >= 65 -> DOWN (overbought)")
            elif current_rsi <= 40:
                signal = "UP"  # Weak bullish
                logger.info(f"{symbol} {interval}: RSI={current_rsi:.1f} <= 40 -> UP (weak bullish)")
            elif current_rsi >= 60:
                signal = "DOWN"  # Weak bearish
                logger.info(f"{symbol} {interval}: RSI={current_rsi:.1f} >= 60 -> DOWN (weak bearish)")
            else:
                # Neutral zone (40-60) - no clear signal
                logger.info(f"{symbol} {interval}: RSI={current_rsi:.1f} in neutral zone (40-60) -> None")
                signal = None
        
        return VKIResult(
            timeframe=interval,
            rsi=current_rsi,
            high_prob=high_prob,
            low_prob=low_prob,
            signal=signal,
            kde_high_sum=kde_high_sum,
            kde_low_sum=kde_low_sum,
        )
    
    def get_trend(self, symbol: str) -> Optional[str]:
        """
        Get aggregated VKI trend from multiple timeframes.
        
        Returns:
            "UP" - Bullish signal (majority of timeframes agree)
            "DOWN" - Bearish signal (majority of timeframes agree)
            None - No clear signal
        """
        results: List[VKIResult] = []
        
        for tf in self.timeframes:
            try:
                result = self._analyze_timeframe(symbol, tf)
                if result:
                    results.append(result)
                    logger.debug(f"VKI {symbol} {tf}: RSI={result.rsi:.1f}, signal={result.signal}")
            except Exception as e:
                logger.warning(f"VKI analysis failed for {symbol} {tf}: {e}")
        
        if not results:
            logger.warning(f"VKI {symbol}: No results from any timeframe")
            return None
        
        # Count signals
        up_count = sum(1 for r in results if r.signal == "UP")
        down_count = sum(1 for r in results if r.signal == "DOWN")
        none_count = sum(1 for r in results if r.signal is None)
        
        logger.info(f"VKI {symbol}: UP={up_count}, DOWN={down_count}, None={none_count} from {len(results)} timeframes")
        
        # Aggregate: majority wins (at least 2 out of 3)
        if up_count >= 2:
            logger.info(f"VKI {symbol}: UP ({up_count}/{len(results)} timeframes)")
            return "UP"
        elif down_count >= 2:
            logger.info(f"VKI {symbol}: DOWN ({down_count}/{len(results)} timeframes)")
            return "DOWN"
        
        # If 1 signal each or only 1 signal total, use weighted approach (longer timeframe has more weight)
        # 15m > 5m > 3m
        for r in results:
            if r.timeframe == "15m" and r.signal:
                logger.info(f"VKI {symbol}: {r.signal} (15m decides)")
                return r.signal
        
        for r in results:
            if r.timeframe == "5m" and r.signal:
                logger.info(f"VKI {symbol}: {r.signal} (5m decides)")
                return r.signal
        
        for r in results:
            if r.timeframe == "1m" and r.signal:
                logger.info(f"VKI {symbol}: {r.signal} (1m decides)")
                return r.signal
        
        logger.info(f"VKI {symbol}: No clear signal")
        return None
    
    def get_detailed_analysis(self, symbol: str) -> dict:
        """Get detailed VKI analysis for debugging/logging."""
        results: List[VKIResult] = []
        
        for tf in self.timeframes:
            try:
                result = self._analyze_timeframe(symbol, tf)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"VKI analysis failed for {symbol} {tf}: {e}")
        
        return {
            "symbol": symbol,
            "timeframes": [
                {
                    "tf": r.timeframe,
                    "rsi": round(r.rsi, 2),
                    "high_prob": round(r.high_prob, 4),
                    "low_prob": round(r.low_prob, 4),
                    "signal": r.signal,
                }
                for r in results
            ],
            "aggregated_trend": self.get_trend(symbol),
        }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    client = VKIClient()
    
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print(f"{'='*50}")
        
        trend = client.get_trend(symbol)
        print(f"Trend: {trend}")
        
        details = client.get_detailed_analysis(symbol)
        for tf_data in details["timeframes"]:
            print(f"  {tf_data['tf']}: RSI={tf_data['rsi']}, signal={tf_data['signal']}")
