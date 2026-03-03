#!/usr/bin/env python3
"""
MACD Client — local MACD-based trend calculator.

Implementation aligned with common Pine Script conventions:
- MACD = EMA(fast) - EMA(slow)
- Signal = EMA(MACD, signal_length)
- Histogram = MACD - Signal

Trend heuristic:
- UP: histogram > 0 and rising
- DOWN: histogram < 0 and falling
- NEUTRAL: histogram flips direction or insufficient data
"""


import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


@dataclass
class MACDResult:
    """MACD calculation result."""
    macd: float
    signal: float
    histogram: float
    hist_prev: float
    trend: str  # UP, DOWN, NEUTRAL
    timestamp: str


class MACDClient:
    """
    MACD client that computes a trend from price data.
    
    Fetches data from public APIs (Binance, CoinGecko) and computes MACD locally.
    """
    
    def __init__(
        self,
        fast_length: int = 12,
        slow_length: int = 26,
        signal_length: int = 9,
        ma_type: str = "EMA",
        timeframe: str = "15m",
        price_source: str = "binance",
        timeout: int = 10,
        retries: int = 3,
        retry_sleep: float = 1.0,
        **_extra
    ):
        """
        Args:
            fast_length: Fast MA period (default: 12)
            slow_length: Slow MA period (default: 26)
            signal_length: Signal line period (default: 9)
            ma_type: Typ MA - "EMA" lub "SMA"
            timeframe: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            price_source: Data source - "binance" or "coingecko"
            timeout: HTTP request timeout (seconds)
            retries: Number of retries
            retry_sleep: Sleep between retries (seconds)
        """
        self.fast_length = int(fast_length)
        self.slow_length = int(slow_length)
        self.signal_length = int(signal_length)
        self.ma_type = str(ma_type).upper()
        self.timeframe = str(timeframe)
        self.price_source = str(price_source).lower()
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.retry_sleep = float(retry_sleep)
        
        # EMA cache (helps keep calculations consistent)
        self._ema_cache: Dict[str, List[float]] = {}
        
        # Minimum candles required for calculation
        self.min_candles = self.slow_length + self.signal_length + 10
        
        # Symbol mapping for different APIs
        self._symbol_map = {
            "BTC/USDT": {"binance": "BTCUSDT", "coingecko": "bitcoin"},
            "ETH/USDT": {"binance": "ETHUSDT", "coingecko": "ethereum"},
            "XRP/USDT": {"binance": "XRPUSDT", "coingecko": "ripple"},
            "SOL/USDT": {"binance": "SOLUSDT", "coingecko": "solana"},
        }
    
    def _get_binance_klines(self, symbol: str, limit: int = 100) -> Optional[List[float]]:
        """Fetch candle close prices from Binance."""
        binance_symbol = self._symbol_map.get(symbol, {}).get("binance", symbol.replace("/", ""))
        
        # Map timeframe to Binance interval format
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        interval = tf_map.get(self.timeframe, "15m")
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": limit
        }
        
        for attempt in range(self.retries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                
                # Binance klines: [open_time, open, high, low, close, volume, ...]
                # Use close price (index 4)
                closes = [float(candle[4]) for candle in data]
                return closes
                
            except Exception as e:
                logger.warning(f"Binance API error (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_sleep)
        
        return None
    
    def _get_coingecko_prices(self, symbol: str, days: int = 2) -> Optional[List[float]]:
        """Pobierz dane z CoinGecko (fallback)"""
        coin_id = self._symbol_map.get(symbol, {}).get("coingecko")
        if not coin_id:
            return None
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        for attempt in range(self.retries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                
                # CoinGecko: {"prices": [[timestamp, price], ...]}
                prices = [p[1] for p in data.get("prices", [])]
                return prices
                
            except Exception as e:
                logger.warning(f"CoinGecko API error (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_sleep)
        
        return None
    
    def _get_prices(self, symbol: str) -> Optional[List[float]]:
        """Fetch prices from the selected data source."""
        if self.price_source == "binance":
            prices = self._get_binance_klines(symbol, limit=self.min_candles)
            if prices:
                return prices
            # Fallback do CoinGecko
            logger.warning("Binance failed, trying CoinGecko...")
            return self._get_coingecko_prices(symbol)
        else:
            prices = self._get_coingecko_prices(symbol)
            if prices:
                return prices
            # Fallback do Binance
            logger.warning("CoinGecko failed, trying Binance...")
            return self._get_binance_klines(symbol, limit=self.min_candles)
    
    def _calculate_sma(self, prices: List[float], length: int) -> List[Optional[float]]:
        """Oblicz Simple Moving Average"""
        result: List[Optional[float]] = []
        for i in range(len(prices)):
            if i < length - 1:
                result.append(None)
            else:
                window = prices[i - length + 1:i + 1]
                result.append(sum(window) / length)
        return result
    
    def _calculate_ema(self, prices: List[float], length: int) -> List[Optional[float]]:
        """
        Oblicz Exponential Moving Average (zgodnie z TradingView).
        
        Metoda:
        1. First `length` values -> no EMA yet (None)
        2. Value at index `length-1` -> SMA used as seed
        3. Next values -> EMA: price * k + ema_prev * (1-k)
        
        gdzie k = 2 / (length + 1)
        """
        if len(prices) < length:
            return [None] * len(prices)
        
        result: List[Optional[float]] = [None] * (length - 1)
        
        # Seed: SMA of first `length` values
        first_sma = sum(prices[:length]) / length
        result.append(first_sma)
        
        # EMA for the remaining values
        multiplier = 2 / (length + 1)
        
        for i in range(length, len(prices)):
            ema = (prices[i] * multiplier) + (result[-1] * (1 - multiplier))
            result.append(ema)
        
        return result
    
    def _calculate_ma(self, prices: List[float], length: int) -> List[Optional[float]]:
        """Compute MA according to the selected type."""
        if self.ma_type == "SMA":
            return self._calculate_sma(prices, length)
        else:
            return self._calculate_ema(prices, length)
    
    def calculate_macd(self, prices: List[float]) -> Optional[MACDResult]:
        """
        Oblicz MACD z listy cen.
        
        Returns:
            MACDResult or None if there is not enough data
        """
        if len(prices) < self.min_candles:
            logger.warning(f"Not enough price data: {len(prices)} < {self.min_candles}")
            return None
        
        # Oblicz fast i slow MA
        fast_ma = self._calculate_ma(prices, self.fast_length)
        slow_ma = self._calculate_ma(prices, self.slow_length)
        
        # MACD line = fast - slow
        macd_line = []
        for i in range(len(prices)):
            if fast_ma[i] is not None and slow_ma[i] is not None:
                macd_line.append(fast_ma[i] - slow_ma[i])
            else:
                macd_line.append(0)
        
        # Signal line = MA(MACD, signal_length)
        signal_line = self._calculate_ma(macd_line, self.signal_length)
        
        # Histogram = MACD - Signal
        histogram = []
        for i in range(len(macd_line)):
            if signal_line[i] is not None:
                histogram.append(macd_line[i] - signal_line[i])
            else:
                histogram.append(0)
        
        # Current and previous values
        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0
        current_macd = macd_line[-1]
        current_signal = signal_line[-1] if signal_line[-1] else 0
        
        # Determine trend from histogram
        # UP: histogram > 0 and rising
        # DOWN: histogram < 0 i maleje
        # NEUTRAL: w innym przypadku
        
        if current_hist > 0 and current_hist > prev_hist:
            trend = "UP"
        elif current_hist < 0 and current_hist < prev_hist:
            trend = "DOWN"
        else:
            # Histogram zmienia kierunek lub jest neutralny
            # Alternatively you can use only the histogram sign:
            if current_hist > 0:
                trend = "UP"
            elif current_hist < 0:
                trend = "DOWN"
            else:
                trend = "NEUTRAL"
        
        return MACDResult(
            macd=current_macd,
            signal=current_signal,
            histogram=current_hist,
            hist_prev=prev_hist,
            trend=trend,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def get_trend(self, symbol: str) -> Optional[str]:
        """
        Pobierz trend MACD dla symbolu.
        
        Args:
            symbol: Para handlowa (np. "BTC/USDT")
            
        Returns:
            "UP", "DOWN", "NEUTRAL" or None on error
        """
        prices = self._get_prices(symbol)
        if not prices:
            logger.error(f"Failed to get prices for {symbol}")
            return None
        
        result = self.calculate_macd(prices)
        if not result:
            return None
        
        logger.info(
            f"MACD {symbol}: hist={result.histogram:.4f} "
            f"(prev={result.hist_prev:.4f}) -> {result.trend}"
        )
        
        return result.trend
    
    def get_full_analysis(self, symbol: str) -> Optional[MACDResult]:
        """
        Get a full MACD analysis for a symbol.
        
        Returns:
            MACDResult with all values.
        """
        prices = self._get_prices(symbol)
        if not prices:
            logger.error(f"Failed to get prices for {symbol}")
            return None
        
        return self.calculate_macd(prices)
    
    def get_btc_trend(self) -> Optional[str]:
        """Shortcut dla BTC"""
        return self.get_trend("BTC/USDT")


def macd_trend_to_side(trend: str) -> Optional[str]:
    """Map MACD trend to market side: UP->YES, DOWN->NO."""
    if not trend:
        return None
    t = str(trend).strip().upper()
    if t in {"UP", "BULLISH", "LONG"}:
        return "YES"
    if t in {"DOWN", "BEARISH", "SHORT"}:
        return "NO"
    return None  # NEUTRAL


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    client = MACDClient(
        fast_length=12,
        slow_length=26,
        signal_length=9,
        timeframe="15m",
        price_source="binance"
    )
    
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]
    
    print("\n" + "="*60)
    print("MACD Analysis")
    print("="*60)
    
    for symbol in symbols:
        result = client.get_full_analysis(symbol)
        if result:
            print(f"\n{symbol}:")
            print(f"  MACD:      {result.macd:.4f}")
            print(f"  Signal:    {result.signal:.4f}")
            print(f"  Histogram: {result.histogram:.4f} (prev: {result.hist_prev:.4f})")
            print(f"  Trend:     {result.trend}")
            print(f"  Side:      {macd_trend_to_side(result.trend)}")
        else:
            print(f"\n{symbol}: Failed to calculate")
