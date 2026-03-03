"""
Keltner Channels (KC) Trend Indicator Client

Calculates Keltner Channels using EMA and ATR to detect oversold conditions.
Signal: Price < Lower Band = UP (potential reversal up)

Parameters:
- Length: 20 periods (1-minute candles)
- ATR Multiplier: 2.0
- MA Type: EMA (Exponential Moving Average) for middle band
- ATR Method: Wilder's RMA (standard TradingView method)

Formula:
- Middle Band = EMA(Close, 20)
- ATR = RMA(True Range, 20)  # Wilder's method, alpha = 1/period
- Upper Band = EMA(Close, 20) + (2.0 × ATR)
- Lower Band = EMA(Close, 20) - (2.0 × ATR)

Signal Logic:
- Current Price < Lower Band → UP (oversold, expect bounce)
- Current Price > Upper Band → DOWN (overbought, expect drop)
- Price between bands → None (neutral)

Note: Uses 80 candles (4x period) for warm-up to ensure stable calculations.
"""

import logging
import time
from typing import Optional, List, Dict
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class KeltnerResult:
    """Result from Keltner Channels analysis."""
    current_price: float
    ema: float
    atr: float
    upper_band: float
    lower_band: float
    signal: Optional[str]  # "UP", "DOWN", or None


class KeltnerClient:
    """
    Keltner Channels trend indicator using Binance 1m data.
    
    Detects oversold/overbought conditions based on price position
    relative to Keltner Channel bands.
    """
    
    BINANCE_API = "https://api.binance.com/api/v3"
    
    def __init__(
        self,
        length: int = 20,
        atr_multiplier: float = 2.0,
        timeout: int = 10
    ):
        self.length = length
        self.atr_multiplier = atr_multiplier
        self.timeout = timeout
        
        logger.info(f"KeltnerClient initialized: length={length}, ATR mult={atr_multiplier}")
    
    def _fetch_klines(self, symbol: str, limit: int = 100) -> List[dict]:
        """Fetch 1m klines from Binance."""
        try:
            binance_symbol = symbol.replace("/", "")
            
            response = requests.get(
                f"{self.BINANCE_API}/klines",
                params={
                    "symbol": binance_symbol,
                    "interval": "1m",
                    "limit": limit
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            raw = response.json()
            klines = []
            for k in raw:
                klines.append({
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                })
            return klines
            
        except Exception as e:
            logger.warning(f"Failed to fetch klines for {symbol}: {e}")
            return []
    
    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # First EMA is SMA
        sma = sum(values[:period]) / period
        ema.append(sma)
        
        # Calculate EMA for rest
        for i in range(period, len(values)):
            new_ema = (values[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(new_ema)
        
        return ema
    
    def _calculate_rma(self, values: List[float], period: int) -> List[float]:
        """Calculate Wilder's Moving Average (RMA) - used for ATR.
        
        RMA uses alpha = 1/period instead of EMA's 2/(period+1).
        This is the standard method used by TradingView and most platforms.
        """
        if len(values) < period:
            return []
        
        rma = []
        alpha = 1 / period  # Wilder's smoothing factor
        
        # First RMA is SMA
        sma = sum(values[:period]) / period
        rma.append(sma)
        
        # Calculate RMA for rest
        for i in range(period, len(values)):
            new_rma = (values[i] * alpha) + (rma[-1] * (1 - alpha))
            rma.append(new_rma)
        
        return rma
    
    def _calculate_atr(self, klines: List[dict], period: int) -> List[float]:
        """Calculate Average True Range using Wilder's method (RMA)."""
        if len(klines) < period + 1:
            return []
        
        # Calculate True Range
        tr_values = []
        for i in range(1, len(klines)):
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_close = klines[i-1]["close"]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        # ATR is RMA (Wilder's) of True Range - standard method
        atr = self._calculate_rma(tr_values, period)
        return atr
    
    def _check_data_integrity(self, klines: List[dict]) -> bool:
        """Check for missing candles (gaps > 1 minute)."""
        if len(klines) < 2:
            return False
        
        for i in range(1, len(klines)):
            prev_close_time = klines[i-1]["close_time"]
            curr_open_time = klines[i]["open_time"]
            
            # Gap should be ~0ms (close_time of prev = open_time of next - 1ms)
            gap = curr_open_time - prev_close_time
            if gap > 60000:  # More than 1 minute gap
                logger.warning(f"Data gap detected: {gap}ms between candles")
                return False
        
        return True
    
    def analyze(self, symbol: str) -> Optional[KeltnerResult]:
        """
        Analyze symbol using Keltner Channels.
        
        Returns:
            KeltnerResult with signal:
            - "UP" if price < lower band (oversold)
            - "DOWN" if price > upper band (overbought)
            - None if price is within bands (neutral)
        """
        # Fetch enough data for stable EMA/RMA calculations (warm-up period)
        # EMA/RMA needs ~3-4x period length to stabilize
        required_candles = self.length * 4  # 80 candles for length=20
        klines = self._fetch_klines(symbol, limit=required_candles)
        
        if not klines:
            logger.warning(f"No klines data for {symbol}")
            return None
        
        if len(klines) < self.length + 1:
            logger.warning(f"Insufficient data for {symbol}: got {len(klines)}, need {self.length + 1}")
            return None
        
        # Check data integrity
        if not self._check_data_integrity(klines):
            logger.warning(f"Data integrity check failed for {symbol} - missing candles")
            return None
        
        # Extract close prices
        closes = [k["close"] for k in klines]
        current_price = closes[-1]
        
        # Calculate EMA
        ema_values = self._calculate_ema(closes, self.length)
        if not ema_values:
            logger.warning(f"Failed to calculate EMA for {symbol}")
            return None
        
        current_ema = ema_values[-1]
        
        # Calculate ATR
        atr_values = self._calculate_atr(klines, self.length)
        if not atr_values:
            logger.warning(f"Failed to calculate ATR for {symbol}")
            return None
        
        current_atr = atr_values[-1]
        
        # Calculate Keltner Bands
        upper_band = current_ema + (self.atr_multiplier * current_atr)
        lower_band = current_ema - (self.atr_multiplier * current_atr)
        
        # Determine signal
        signal = None
        if current_price < lower_band:
            signal = "UP"  # Oversold - expect bounce up
            logger.info(f"{symbol}: Price ${current_price:.2f} < Lower ${lower_band:.2f} -> UP (oversold)")
        elif current_price > upper_band:
            signal = "DOWN"  # Overbought - expect drop
            logger.info(f"{symbol}: Price ${current_price:.2f} > Upper ${upper_band:.2f} -> DOWN (overbought)")
        else:
            # Price within bands - check proximity
            band_width = upper_band - lower_band
            position = (current_price - lower_band) / band_width if band_width > 0 else 0.5
            logger.info(f"{symbol}: Price ${current_price:.2f} within bands (position: {position:.1%}) -> None")
        
        return KeltnerResult(
            current_price=current_price,
            ema=current_ema,
            atr=current_atr,
            upper_band=upper_band,
            lower_band=lower_band,
            signal=signal
        )
    
    def get_trend(self, symbol: str) -> Optional[str]:
        """
        Get Keltner trend signal for symbol.
        
        Returns:
            "UP" - Price below lower band (oversold)
            "DOWN" - Price above upper band (overbought)
            None - Price within bands (neutral)
        """
        result = self.analyze(symbol)
        if result:
            return result.signal
        return None
    
    def get_detailed_analysis(self, symbol: str) -> dict:
        """Get detailed Keltner analysis for debugging."""
        result = self.analyze(symbol)
        if not result:
            return {"error": "Analysis failed", "symbol": symbol}
        
        return {
            "symbol": symbol,
            "current_price": result.current_price,
            "ema_20": result.ema,
            "atr_20": result.atr,
            "upper_band": result.upper_band,
            "lower_band": result.lower_band,
            "band_width": result.upper_band - result.lower_band,
            "signal": result.signal,
            "parameters": {
                "length": self.length,
                "atr_multiplier": self.atr_multiplier,
            }
        }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    client = KeltnerClient(length=20, atr_multiplier=2.0)
    
    for symbol in ["ETH/USDT", "BTC/USDT"]:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print('='*50)
        
        trend = client.get_trend(symbol)
        print(f"Trend: {trend}")
        
        details = client.get_detailed_analysis(symbol)
        for k, v in details.items():
            print(f"  {k}: {v}")
