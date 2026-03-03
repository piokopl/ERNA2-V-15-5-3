#!/usr/bin/env python3
"""PolySniper In-Window v4 — multi-asset runner with Martingale.

In-Window strategy (Martingale 2.0):
- Place orders AFTER the window starts (not before)
- Monitor price and sell once the target is reached
- Martingale increases stake after a LOSS and resets after a WIN
- Martingale steps are tracked independently per asset

Flow:
1) Wait for the next window
2) At window start: fetch trend and buy YES/NO
3) Monitor price:
   - bid >= target -> SELL -> WIN -> reset martingale
   - window ends without a sell -> LOSS -> advance martingale
4) Repeat

Usage:
  python runner.py            # demo
  python runner.py --live     # live
"""


from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
import yaml
from dotenv import load_dotenv

from adx_client import ADXClient
from macd_client import MACDClient
from vki_client import VKIClient
from keltner_client import KeltnerClient
from multi_client import MultiIndicatorClient
from lorentzian_client import LorentzianTrendClient
from ai_client import AITrendClient
from bot_registry import REGISTRY, AssetStatus, TradeRecord
from polymarket_client import PolymarketClient, MarketInfo, CloudflareBlockError, InsufficientBalanceError, OrderStatus, ExecutionInfo
from ws_client import PolymarketWebSocket


WINDOW_SECONDS = 900  # 15 minutes
FILL_DEADLINE_BUFFER = 10  # seconds before window end to treat unfilled as no-trade
SELL_FEE_BUFFER = 0.03  # sell 3% less than filled to account for Polymarket token fees

# Martingale 2.0 - default multipliers (overridden by config.yaml)
DEFAULT_MARTINGALE_MULTIPLIERS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # 6 levels (0-5), max 32x


def ensure_dirs():
    Path('logs').mkdir(exist_ok=True)
    Path('state').mkdir(exist_ok=True)


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def make_logger(asset: str) -> logging.Logger:
    ensure_dirs()
    lg = logging.getLogger(f"bot.{asset}")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not lg.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(f'logs/polysniper_{asset}.log', encoding='utf-8')
        fh.setFormatter(fmt)
        lg.addHandler(sh)
        lg.addHandler(fh)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    return lg


def get_next_window(now: Optional[float] = None, grace_period: int = 30) -> Tuple[int, int]:
    """Return (start_ts, end_ts) for the next 15-minute window to trade.
    
    If we're within grace_period seconds of a window start, return that window
    (allows entering current window if we just finished previous one).
    """
    n = int(now if now is not None else time.time())
    current_start = (n // WINDOW_SECONDS) * WINDOW_SECONDS
    current_end = current_start + WINDOW_SECONDS
    
    # If we're within grace_period of current window start, use current window
    seconds_into_window = n - current_start
    if seconds_into_window <= grace_period:
        return current_start, current_end
    
    # Otherwise, wait for next window
    next_start = current_start + WINDOW_SECONDS
    next_end = next_start + WINDOW_SECONDS
    return next_start, next_end


def get_current_window(now: Optional[float] = None) -> Tuple[int, int]:
    """Return (start_ts, end_ts) for the CURRENT 15-minute window."""
    n = int(now if now is not None else time.time())
    current_start = (n // WINDOW_SECONDS) * WINDOW_SECONDS
    current_end = current_start + WINDOW_SECONDS
    return current_start, current_end


@dataclass
class OrderState:
    """State for a single order"""
    side: str  # "YES" or "NO"
    order_id: Optional[str] = None
    token_id: Optional[str] = None  # CTF token ID for sell
    filled: bool = False
    filled_size: float = 0.0
    sell_size: float = 0.0  # filled_size minus fee buffer, rounded down to 0.01
    limit_price: float = 0.0
    avg_fill_price: float = 0.0
    sold: bool = False
    sell_price: float = 0.0
    sell_order_id: Optional[str] = None
    target_hit_price: float = 0.0  # Price when target was reached
    pnl: float = 0.0
    fill_state: str = "PENDING"  # PENDING → CONFIRMED_FILLED | UNFILLED_AT_DEADLINE


@dataclass
class MartingaleState:
    """Martingale state for an asset"""
    multipliers: List[float] = field(default_factory=lambda: DEFAULT_MARTINGALE_MULTIPLIERS.copy())
    step: int = 0
    consecutive_losses: int = 0
    last_result: Optional[str] = None  # WIN, LOSS, LOSS_RESET, None
    
    def get_multiplier(self) -> float:
        """Get current size multiplier."""
        return self.multipliers[min(self.step, len(self.multipliers) - 1)]
    
    def on_win(self):
        """Reset on win."""
        self.step = 0
        self.consecutive_losses = 0
        self.last_result = "WIN"
    
    def on_loss(self):
        """Increase step on loss. Reset if already at max."""
        self.consecutive_losses += 1
        if self.step >= len(self.multipliers) - 1:
            # Already at max step - reset to 0
            self.step = 0
            self.last_result = "LOSS_RESET"
        else:
            self.step += 1
            self.last_result = "LOSS"


class InWindowBot(threading.Thread):
    """Bot that places orders AFTER market window starts with Martingale."""
    
    def __init__(self, asset: str, cfg: Dict[str, Any], is_demo: bool, shared_stop: threading.Event):
        super().__init__(daemon=True)
        self.asset = asset.upper()
        self.cfg = cfg
        self.is_demo = is_demo
        self.stop_event = shared_stop
        self.log = make_logger(self.asset)
        
        # Config
        self.enabled = bool(cfg.get('enabled', True))
        self.symbol = cfg.get('symbol', f"{self.asset}/USDT")
        self.base_order_size = float(cfg.get('order_size', 5.0))  # Base size for martingale
        self.buy_price = float(cfg.get('buy_price', 0.55))  # Max price to buy
        self.sell_target = float(cfg.get('sell_target', 0.85))  # Target price to sell
        
        # Target reduction for high martingale steps (to exit faster)
        self.target_reduction_second_last = float(cfg.get('_target_reduction_second_last', 0.0))
        self.target_reduction_last = float(cfg.get('_target_reduction_last', 0.0))
        
        # Timing
        self.order_delay = int(cfg.get('order_delay', 0))  # Seconds after window start (0 = immediate)
        self.multi_fetch_before = int(cfg.get('multi_fetch_before', 10))  # Seconds before window to fetch Multi trend
        
        # Blacklist hours (UTC) - bot won't trade during these hours
        self.blacklist_hours = cfg.get('blacklist_hours', [])
        if self.blacklist_hours:
            self.log.info(f"Blacklisted hours (UTC): {self.blacklist_hours}")
        
        # Decision trend (P, S, D, V, K, M, L, A, VOTE, or combinations like P+V)
        # P = Primary ADX, S = Secondary ADX, D = MACD, V = VKI, K = Keltner, M = Multi-indicator, L = Lorentzian, A = AI
        self.decision_trend = cfg.get('decision_trend') or cfg.get('_decision_trend')
        if self.decision_trend:
            self.decision_trend = str(self.decision_trend).upper()
            # Validate: single trend, VOTE, or combination with +
            valid_singles = ('P', 'S', 'D', 'V', 'K', 'M', 'L', 'A', 'VOTE')
            if '+' in self.decision_trend:
                # Combination mode - validate each part
                parts = [t.strip() for t in self.decision_trend.split('+')]
                valid_parts = ('P', 'S', 'D', 'V', 'K', 'M', 'L', 'A')
                if not all(p in valid_parts for p in parts):
                    self.log.warning(f"Invalid combination '{self.decision_trend}', using M")
                    self.decision_trend = 'M'
            elif self.decision_trend not in valid_singles:
                self.log.warning(f"Invalid decision_trend '{self.decision_trend}', using M")
                self.decision_trend = 'M'
        else:
            self.decision_trend = 'M'
        
        # ADX clients
        adx_cfg = cfg.get('_adx', {})
        primary_cfg = adx_cfg.get('primary', {})
        secondary_cfg = adx_cfg.get('secondary', {})
        
        self.primary_adx = ADXClient(
            api_url=primary_cfg.get('api_url', 'https://mimir.cryptothunder.net/bot_adx'),
            timeout=primary_cfg.get('timeout', 10)
        ) if primary_cfg else None
        
        self.secondary_adx = ADXClient(
            api_url=secondary_cfg.get('api_url', 'https://adx.cryptothunder.net/bot_adx'),
            timeout=secondary_cfg.get('timeout', 10)
        ) if secondary_cfg else None
        
        # MACD client
        macd_cfg = cfg.get('_macd', {})
        self.macd = MACDClient(
            fast_length=macd_cfg.get('fast_length', 12),
            slow_length=macd_cfg.get('slow_length', 26),
            signal_length=macd_cfg.get('signal_length', 9),
            timeframe=macd_cfg.get('timeframe', '3m')
        ) if macd_cfg else None
        
        # VKI client
        self.vki = VKIClient()
        
        # Keltner Channels client
        keltner_cfg = cfg.get('_keltner', {})
        self.keltner = KeltnerClient(
            length=keltner_cfg.get('length', 20),
            atr_multiplier=keltner_cfg.get('atr_multiplier', 2.0),
            timeout=keltner_cfg.get('timeout', 10)
        )
        
        # Multi-indicator client (for M trend)
        multi_cfg = cfg.get('_multi', {})
        self.multi = MultiIndicatorClient(
            min_aligned=multi_cfg.get('min_aligned', 3),
            ta_min_confidence=multi_cfg.get('ta_min_confidence', 0.75),
            rsi_period=multi_cfg.get('rsi_length', 14),
            rsi_overbought=multi_cfg.get('rsi_overbought', 75),
            rsi_oversold=multi_cfg.get('rsi_oversold', 25),
            timeout=multi_cfg.get('timeout', 10)
        )
        
        # Lorentzian Classification client (for L trend)
        lor_cfg = cfg.get('_lorentzian', {})
        self.lorentzian = LorentzianTrendClient(
            timeframe=lor_cfg.get('timeframe', '5m'),
            lookback=lor_cfg.get('lookback', 750),
            k_neighbors=lor_cfg.get('k_neighbors', 21),
            horizon_bars=lor_cfg.get('horizon_bars', 6),
            min_confidence=lor_cfg.get('min_confidence', 0.52),
            st_period=lor_cfg.get('st_period', 10),
            st_multiplier=lor_cfg.get('st_multiplier', 3.0),
            ema_period=lor_cfg.get('ema_period', 200),
            vp_lookback=lor_cfg.get('vp_lookback', 400),
            vp_bins=lor_cfg.get('vp_bins', 80),
            vp_value_area_pct=lor_cfg.get('vp_value_area_pct', 0.70),
            pivot_left=lor_cfg.get('pivot_left', 3),
            pivot_right=lor_cfg.get('pivot_right', 3),
            bos_lookback=lor_cfg.get('bos_lookback', 100),
            threshold=lor_cfg.get('threshold', 2.0),
            timeout=lor_cfg.get('timeout', 12.0),
        )
        
        # AI prediction client (for A trend)
        ai_cfg = cfg.get('_ai', {})
        self.ai_client = AITrendClient(
            model=ai_cfg.get('model', 'claude-haiku-4-5-20251001'),
            api_key=ai_cfg.get('api_key') or os.getenv('ANTHROPIC_API_KEY', ''),
            api_url=ai_cfg.get('api_url', 'https://api.anthropic.com/v1/messages'),
            max_searches=ai_cfg.get('max_searches', 1),
            timeout=ai_cfg.get('timeout', 30.0),
        )
        has_key = bool(self.ai_client._api_key)
        self.log.info(f"🤖 AI client: model={self.ai_client.model}, key={'OK' if has_key else '❌ BRAK!'}, max_searches={self.ai_client.max_searches}")
        
        # Polymarket client - load credentials from environment
        self.pm = PolymarketClient(
            private_key=os.getenv('POLYMARKET_PRIVATE_KEY', ''),
            funder_address=os.getenv('POLYMARKET_PROXY_ADDRESS', ''),
            api_key=os.getenv('POLYMARKET_API_KEY', ''),
            api_secret=os.getenv('POLYMARKET_API_SECRET', ''),
            api_passphrase=os.getenv('POLYMARKET_API_PASSPHRASE', ''),
            is_demo=is_demo
        )
        
        # Transactions file
        self.transactions_file = f"transactions_{self.asset.lower()}.csv"
        self._init_csv()
        
        # State
        self.current_order: Optional[OrderState] = None
        self.target_window_start: int = 0
        self.target_window_end: int = 0
        self.current_market: Optional[MarketInfo] = None
        self.ws: Optional[PolymarketWebSocket] = None
        
        # Martingale state
        martingale_multipliers = cfg.get('_martingale_multipliers', DEFAULT_MARTINGALE_MULTIPLIERS)
        self.martingale = MartingaleState(multipliers=martingale_multipliers)
        
        # Current prices from WebSocket
        self.yes_ask: float = 0.0
        self.yes_bid: float = 0.0
        self.no_ask: float = 0.0
        self.no_bid: float = 0.0
        
        # Cached trends
        self.cached_primary_trend: Optional[str] = None
        self.cached_secondary_trend: Optional[str] = None
        self.cached_macd_trend: Optional[str] = None
        self.cached_vki_trend: Optional[str] = None
        self.cached_keltner_trend: Optional[str] = None  # Keltner Channels (K)
        self.cached_multi_trend: Optional[str] = None  # Multi-indicator (M)
        self.cached_lorentzian_trend: Optional[str] = None  # Lorentzian Classification (L)
        self.cached_ai_trend: Optional[str] = None  # AI prediction (A)

    def _init_csv(self):
        if not Path(self.transactions_file).exists():
            with open(self.transactions_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'timestamp', 'asset', 'market', 'side', 'size', 
                    'buy_price', 'sell_price', 'pnl', 'result', 
                    'martingale_step', 'primary_trend', 'secondary_trend', 
                    'macd_trend', 'vki_trend', 'keltner_trend', 'multi_trend', 'winning_side'
                ])

    def _log_trade_csv(self, market_slug: str, side: str, size: float, 
                       buy_price: float, sell_price: float, pnl: float, 
                       result: str, winning_side: Optional[str] = None):
        with open(self.transactions_file, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                utc_ts(), self.asset, market_slug, side, f"{size:.4f}",
                f"{buy_price:.4f}", f"{sell_price:.4f}", f"{pnl:.4f}", result,
                self.martingale.step,
                self.cached_primary_trend or "", 
                self.cached_secondary_trend or "",
                self.cached_macd_trend or "",
                self.cached_vki_trend or "",
                self.cached_keltner_trend or "",
                self.cached_multi_trend or "",
                self.cached_lorentzian_trend or "",
                self.cached_ai_trend or "",
                winning_side or ""
            ])

    def _fetch_trends(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Fetch all trend indicators in PARALLEL for speed (including Multi + Lorentzian + AI)."""
        results = {}
        
        def fetch_primary():
            if self.primary_adx:
                return ('primary', self.primary_adx.get_trend(self.symbol))
            return ('primary', None)
        
        def fetch_secondary():
            if self.secondary_adx:
                return ('secondary', self.secondary_adx.get_trend(self.symbol))
            return ('secondary', None)
        
        def fetch_macd():
            if self.macd:
                return ('macd', self.macd.get_trend(self.symbol))
            return ('macd', None)
        
        def fetch_vki():
            return ('vki', self.vki.get_trend(self.symbol))
        
        def fetch_keltner():
            return ('keltner', self.keltner.get_trend(self.symbol))
        
        def fetch_multi():
            signal = self.multi.get_signal(self.symbol)
            if signal:
                if signal.direction != "NEUTRAL":
                    return ('multi', signal.direction, signal)
                return ('multi', None, signal)
            return ('multi', None, None)
        
        def fetch_lorentzian():
            return ('lorentzian', self.lorentzian.get_trend(self.symbol))
        
        def fetch_ai():
            return ('ai', self.ai_client.get_trend(self.symbol))
        
        # Run all fetches in parallel
        fetch_start = time.time()
        multi_signal = None
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(fetch_primary),
                executor.submit(fetch_secondary),
                executor.submit(fetch_macd),
                executor.submit(fetch_vki),
                executor.submit(fetch_keltner),
                executor.submit(fetch_multi),
                executor.submit(fetch_lorentzian),
                executor.submit(fetch_ai),
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=15)
                    if result[0] == 'multi':
                        results['multi'] = result[1]
                        multi_signal = result[2] if len(result) > 2 else None
                    else:
                        results[result[0]] = result[1]
                except Exception as e:
                    self.log.warning(f"Trend fetch error: {e}")
        
        fetch_time = time.time() - fetch_start
        
        # Extract results
        primary = results.get('primary')
        secondary = results.get('secondary')
        macd = results.get('macd')
        vki = results.get('vki')
        keltner = results.get('keltner')
        multi = results.get('multi')
        lorentzian = results.get('lorentzian')
        ai = results.get('ai')
        
        # Log all at once (after parallel fetch completes)
        self.log.info(f"📊 Trends in {fetch_time:.2f}s: P={primary} S={secondary} D={macd} V={vki} K={keltner} M={multi} L={lorentzian} A={ai}")
        
        # Log Multi details if available
        if multi_signal:
            if multi_signal.is_reversal_zone:
                self.log.warning(f"⚠️ [{self.asset}] RSI in reversal zone!")
            if not multi_signal.is_confirmed:
                self.log.warning(f"⚠️ [{self.asset}] Delta does not confirm signal")
        
        return primary, secondary, macd, vki, keltner, multi, lorentzian, ai

    def _get_decision_trend_value(self) -> Optional[str]:
        """Get the value of the configured decision trend. Returns 'UP'/'DOWN' or None."""
        raw = None
        if self.decision_trend == 'P':
            raw = self.cached_primary_trend
        elif self.decision_trend == 'S':
            raw = self.cached_secondary_trend
        elif self.decision_trend == 'D':  # D = MACD (Divergence)
            raw = self.cached_macd_trend
        elif self.decision_trend == 'V':
            raw = self.cached_vki_trend
        elif self.decision_trend == 'K':  # K = Keltner Channels
            raw = self.cached_keltner_trend
        elif self.decision_trend == 'M':  # M = Multi-indicator
            raw = self.cached_multi_trend
        elif self.decision_trend == 'L':  # L = Lorentzian Classification
            raw = self.cached_lorentzian_trend
        elif self.decision_trend == 'A':  # A = AI prediction
            raw = self.cached_ai_trend
        elif self.decision_trend == 'VOTE':
            raw = self._get_vote_decision()
        elif '+' in self.decision_trend:
            raw = self._get_combination_decision()
        
        # Only accept UP/DOWN — reject None, "NONE", "NEUTRAL", empty, etc.
        if raw in ("UP", "DOWN"):
            return raw
        return None
    
    def _get_combination_decision(self) -> Optional[str]:
        """Get decision by requiring all specified trends to agree."""
        trend_map = {
            'P': ('Primary', self.cached_primary_trend),
            'S': ('Secondary', self.cached_secondary_trend),
            'D': ('MACD', self.cached_macd_trend),
            'V': ('VKI', self.cached_vki_trend),
            'K': ('Keltner', self.cached_keltner_trend),
            'M': ('Multi', self.cached_multi_trend),
            'L': ('Lorentzian', self.cached_lorentzian_trend),
            'A': ('AI', self.cached_ai_trend),
        }
        
        # Parse combination like "P+V" or "P+V+D"
        required = [t.strip().upper() for t in self.decision_trend.split('+')]
        
        values = []
        details = []
        for t in required:
            if t in trend_map:
                name, value = trend_map[t]
                # Only accept UP/DOWN as valid signals
                valid_value = value if value in ("UP", "DOWN") else None
                values.append(valid_value)
                details.append(f"{t}={value or 'None'}")
            else:
                self.log.warning(f"Unknown trend '{t}' in combination")
                return None
        
        self.log.info(f"🔗 [{self.asset}] Combination check: {', '.join(details)}")
        
        # All must have a value (not None)
        if None in values:
            none_trends = [required[i] for i, v in enumerate(values) if v is None]
            self.log.warning(f"⚠️ [{self.asset}] Missing trend data: {','.join(none_trends)} - skipping")
            return None
        
        # All must agree
        if len(set(values)) == 1:
            result = values[0]
            self.log.info(f"🔗 [{self.asset}] All {len(required)} trends agree: {result}")
            return result
        else:
            self.log.warning(f"⚠️ [{self.asset}] Trends don't agree ({', '.join(details)}) - skipping")
            return None
    
    def _get_vote_decision(self) -> Optional[str]:
        """Get decision by voting across 7 trends (excluding K - Keltner is too rare)."""
        # K (Keltner) excluded - only signals when price is outside bands (rare)
        trends = {
            'P': self.cached_primary_trend,
            'S': self.cached_secondary_trend,
            'D': self.cached_macd_trend,
            'V': self.cached_vki_trend,
            'M': self.cached_multi_trend,
            'L': self.cached_lorentzian_trend,
            'A': self.cached_ai_trend,
        }
        
        up_votes = []
        down_votes = []
        none_votes = []
        
        for name, value in trends.items():
            if value == "UP":
                up_votes.append(name)
            elif value == "DOWN":
                down_votes.append(name)
            else:
                none_votes.append(name)
        
        up_count = len(up_votes)
        down_count = len(down_votes)
        total_votes = up_count + down_count
        
        self.log.info(f"🗳️ [{self.asset}] VOTE: UP={up_count} ({','.join(up_votes) or '-'}) | DOWN={down_count} ({','.join(down_votes) or '-'}) | None={len(none_votes)}")
        
        # Majority wins (need 4+ out of 7 valid votes)
        if total_votes == 0:
            self.log.warning(f"⚠️ [{self.asset}] VOTE: No valid votes - skipping")
            return None
        
        if up_count >= 4:
            self.log.info(f"🗳️ [{self.asset}] VOTE result: UP ({up_count}/7)")
            return "UP"
        elif down_count >= 4:
            self.log.info(f"🗳️ [{self.asset}] VOTE result: DOWN ({down_count}/7)")
            return "DOWN"
        else:
            self.log.warning(f"⚠️ [{self.asset}] VOTE: No majority (UP={up_count}, DOWN={down_count}) - skipping")
            return None
    
    def _fetch_multi_trend(self) -> Optional[str]:
        """Fetch Multi-indicator trend (should be called 10s before window)."""
        try:
            signal = self.multi.get_signal(self.symbol)
            if signal:
                if signal.direction != "NEUTRAL":
                    self.log.info(f"📊 Multi: {signal.direction} ({signal.aligned_count}/{len(signal.signals)} aligned)")
                    if signal.is_reversal_zone:
                        self.log.warning(f"⚠️ [{self.asset}] RSI in reversal zone!")
                    if not signal.is_confirmed:
                        self.log.warning(f"⚠️ [{self.asset}] Delta does not confirm signal")
                    return signal.direction
                else:
                    self.log.info(f"📊 Multi: NEUTRAL ({signal.aligned_count}/{len(signal.signals)} aligned - need 3+)")
                    return None
            else:
                self.log.warning(f"📊 Multi: No signal (API error?)")
            return None
        except Exception as e:
            self.log.warning(f"Multi-indicator error: {e}")
            return None

    def _find_market_for_window(self, window_start_ts: int) -> Optional[MarketInfo]:
        """Find market for specific window."""
        return self.pm.find_15min_market(self.asset, target_start_ts=window_start_ts)

    def _is_blacklisted_hour(self, window_start_ts: int) -> bool:
        """Check if window start time falls in blacklisted hours (UTC).
        
        Blacklist only applies on weekdays (Mon-Fri).
        Weekends (Sat-Sun) are unrestricted — bots trade 24h.
        """
        if not self.blacklist_hours:
            return False
        
        from datetime import datetime, timezone
        window_dt = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
        
        # weekday(): 0=Mon, 1=Tue, ..., 4=Fri, 5=Sat, 6=Sun
        if window_dt.weekday() >= 5:
            return False  # Weekend — no restrictions
        
        hour_utc = window_dt.hour
        minute_utc = window_dt.minute
        
        if hour_utc in self.blacklist_hours:
            next_allowed = f"{(hour_utc + 1) % 24:02d}:00 UTC"
            self.log.warning(f"⛔ [{self.asset}] {hour_utc:02d}:{minute_utc:02d} UTC is blacklisted (weekday), skipping until {next_allowed}")
            return True
        return False

    def _get_current_order_size(self) -> float:
        """Get order size with martingale multiplier."""
        multiplier = self.martingale.get_multiplier()
        size = self.base_order_size * multiplier
        return size

    # Minimum sell target — w prediction market token musi być > $0.50 żeby wygrać
    MIN_SELL_TARGET = 0.51

    def _get_current_sell_target(self) -> float:
        """Get sell target with reduction for high martingale steps.
        
        On second-last step: target = sell_target - target_reduction_second_last
        On last step: target = sell_target - target_reduction_last
        
        Floor: nigdy poniżej MIN_SELL_TARGET ($0.51) — poniżej $0.50 token przegrywa.
        """
        max_step = len(self.martingale.multipliers) - 1
        current_step = self.martingale.step
        
        if current_step == max_step and self.target_reduction_last > 0:
            reduced_target = self.sell_target - self.target_reduction_last
            self.log.debug(f"🎯 [{self.asset}] Target reduced: ${self.sell_target} - ${self.target_reduction_last} = ${reduced_target:.2f} (LAST step {current_step})")
        elif current_step == max_step - 1 and self.target_reduction_second_last > 0:
            reduced_target = self.sell_target - self.target_reduction_second_last
            self.log.debug(f"🎯 [{self.asset}] Target reduced: ${self.sell_target} - ${self.target_reduction_second_last} = ${reduced_target:.2f} (second-last step {current_step})")
        else:
            return self.sell_target
        
        # Floor: sell target nigdy poniżej $0.51
        if reduced_target < self.MIN_SELL_TARGET:
            self.log.warning(
                f"⚠️ [{self.asset}] Target ${reduced_target:.2f} below floor ${self.MIN_SELL_TARGET} "
                f"— clamping to ${self.MIN_SELL_TARGET}"
            )
            reduced_target = self.MIN_SELL_TARGET
        
        return reduced_target

    def _place_buy_order(self, token_id: str, side: str, size: float) -> Optional[OrderStatus]:
        """Place a buy limit order. Returns OrderStatus with initial fill info."""
        neg_risk = self.current_market.neg_risk if self.current_market else True
        self.log.info(f"📤 [{self.asset}] Placing {side} BUY @ ${self.buy_price} for {size} tokens (neg_risk={neg_risk})")

        try:
            result = self.pm.place_order(
                token_id=token_id,
                side="BUY",
                size=size,
                price=self.buy_price,
                neg_risk=neg_risk,
            )
            if result:
                self.log.info(f"✅ [{self.asset}] Buy order placed: {result.order_id}")
                self.log.info(
                    f"   Status: {result.status}, filled: {result.filled_size}, avg_price: ${result.avg_fill_price:.4f}"
                )
                return result
            else:
                self.log.error(f"❌ [{self.asset}] Buy order returned None")
                if self.pm.last_order_error:
                    self.log.error(f"❌ [{self.asset}] Error details: {self.pm.last_order_error}")
                if self.pm.last_order_response:
                    self.log.error(f"❌ [{self.asset}] API response: {self.pm.last_order_response}")
            return None
        except CloudflareBlockError:
            raise  # Propagate to main loop for circuit breaker
        except InsufficientBalanceError:
            raise  # Propagate to main loop — stop bot
        except Exception as e:
            self.log.error(f"❌ [{self.asset}] Buy order failed: {e}")
            if self.pm.last_order_error:
                self.log.error(f"❌ [{self.asset}] Error details: {self.pm.last_order_error}")
            return None

    def _place_sell_order(self, token_id: str, side: str, size: float, price: float) -> Optional[OrderStatus]:
        """Place a sell limit order. Returns OrderStatus with initial fill info."""
        neg_risk = self.current_market.neg_risk if self.current_market else True
        self.log.info(f"📤 [{self.asset}] Placing {side} SELL @ ${price} for {size} tokens (neg_risk={neg_risk})")
        self.log.info(f"📋 [{self.asset}] SELL params: token={token_id[:40]}..., side=SELL, size={size}, price={price}")
        
        try:
            result = self.pm.place_order(
                token_id=token_id,
                side="SELL",
                size=size,
                price=price,
                neg_risk=neg_risk,
            )
            if result:
                self.log.info(f"✅ [{self.asset}] Sell order placed: {result.order_id}")
                self.log.info(f"   Status: {result.status}, filled: {result.filled_size}, avg_price: ${result.avg_fill_price:.4f}")
                return result
            else:
                self.log.error(f"❌ [{self.asset}] Sell order returned None (API rejected)")
                # Loguj szczegóły błędu z klienta
                if self.pm.last_order_error:
                    self.log.error(f"❌ [{self.asset}] Error details: {self.pm.last_order_error}")
                if self.pm.last_order_response:
                    self.log.error(f"❌ [{self.asset}] API response: {self.pm.last_order_response}")
            return None
        except Exception as e:
            self.log.error(f"❌ [{self.asset}] Sell order exception: {e}")
            if self.pm.last_order_error:
                self.log.error(f"❌ [{self.asset}] Error details: {self.pm.last_order_error}")
            import traceback
            self.log.error(f"❌ [{self.asset}] Traceback:\n{traceback.format_exc()}")
            return None

    def _wait_for_fill(self, order_id: str, deadline: float) -> Tuple[bool, float, float]:
        """Wait for order to fill until deadline."""
        poll_interval = 0.5
        status_errors = 0
        max_status_errors = 10  # Po tylu błędach z rzędu - loguj ostrzeżenie
        
        while time.time() < deadline and not self.stop_event.is_set():
            try:
                status = self.pm.get_order_status(order_id)
                if status:
                    status_errors = 0  # Reset error counter
                    if status.status.upper() == "MATCHED":
                        self.log.info(f"✅ [{self.asset}] Order FILLED: {status.filled_size} tokens @ ${status.avg_fill_price:.4f}")
                        return True, status.filled_size, status.avg_fill_price
                    elif status.status.upper() in ("CANCELLED", "EXPIRED"):
                        self.log.warning(f"⚠️ [{self.asset}] Order {status.status}")
                        return False, status.filled_size, status.avg_fill_price
                    elif status.filled_size > 0:
                        self.log.info(f"📊 [{self.asset}] Partial fill: {status.filled_size}")
                else:
                    status_errors += 1
                    if status_errors == max_status_errors:
                        self.log.warning(f"⚠️ [{self.asset}] get_order_status returning None repeatedly ({status_errors}x) - API may be having issues")
                        if self.pm.last_status_error:
                            self.log.warning(f"⚠️ [{self.asset}] Last error: {self.pm.last_status_error}")
                        if self.pm.last_status_response:
                            self.log.warning(f"⚠️ [{self.asset}] Last response: {self.pm.last_status_response}")
                    elif status_errors > max_status_errors and status_errors % 20 == 0:
                        self.log.warning(f"⚠️ [{self.asset}] Still getting None from get_order_status ({status_errors}x)")
            except Exception as e:
                self.log.warning(f"Error checking order status: {e}")
                status_errors += 1
            
            time.sleep(poll_interval)
        
        # Deadline reached
        self.log.info(f"⏰ [{self.asset}] Fill deadline reached, checking final status...")
        try:
            status = self.pm.get_order_status(order_id)
            if status:
                self.log.info(f"📋 [{self.asset}] Final status: {status.status}, filled: {status.filled_size}")
                if status.filled_size > 0:
                    self.log.info(f"⏰ [{self.asset}] Deadline - partial/full fill: {status.filled_size}")
                    if status.status.upper() != "MATCHED":
                        self.pm.cancel_order(order_id)
                    return True, status.filled_size, status.avg_fill_price
                else:
                    self.log.warning(f"⏰ [{self.asset}] Deadline - NOT filled (status={status.status}), cancelling")
                    self.pm.cancel_order(order_id)
                    return False, 0.0, 0.0
            else:
                self.log.error(f"❌ [{self.asset}] Deadline - could not get order status (API returned None), cancelling order")
                self.log.error(f"❌ [{self.asset}] Order ID: {order_id}")
                if self.pm.last_status_error:
                    self.log.error(f"❌ [{self.asset}] Last status error: {self.pm.last_status_error}")
                if self.pm.last_status_response:
                    self.log.error(f"❌ [{self.asset}] Last API response: {self.pm.last_status_response}")
                self.log.error(f"❌ [{self.asset}] UWAGA: Sprawdź ręcznie na Polymarket czy zamówienie zostało wypełnione!")
                self.pm.cancel_order(order_id)
                return False, 0.0, 0.0
        except Exception as e:
            self.log.error(f"Error at deadline: {e}")
            return False, 0.0, 0.0

    def _start_websocket(self, market: MarketInfo):
        """Start WebSocket monitoring for market."""
        try:
            self.ws = PolymarketWebSocket()
            self.ws.start()
            time.sleep(1)
            self.ws.subscribe_market(market.yes_token_id, market.no_token_id)
            self.log.info(f"📡 [{self.asset}] WebSocket started")
        except Exception as e:
            self.log.warning(f"Failed to start WebSocket: {e}")
            self.ws = None

    def _stop_websocket(self):
        """Stop WebSocket monitoring."""
        if self.ws:
            try:
                self.ws.stop()
            except:
                pass
            self.ws = None
        self.yes_ask = 0.0
        self.yes_bid = 0.0
        self.no_ask = 0.0
        self.no_bid = 0.0

    def _update_prices_from_ws(self):
        """Update prices from WebSocket."""
        if self.ws:
            quote = self.ws.get_current_quote()
            if quote:
                self.yes_ask = quote.yes_ask or 0.0
                self.yes_bid = quote.yes_bid or 0.0
                self.no_ask = quote.no_ask or 0.0
                self.no_bid = quote.no_bid or 0.0

    def _get_current_bid(self, side: str) -> float:
        """Get current bid price for our side."""
        return self.yes_bid if side == "YES" else self.no_bid

    def _update_registry(self, state: str, market_slug: str = ""):
        """Update bot registry for dashboard."""
        now = time.time()
        
        if state in ("WAITING",):
            time_left = max(0, int(self.target_window_start - now)) if self.target_window_start else 0
        else:
            time_left = max(0, int(self.target_window_end - now)) if self.target_window_end else 0
        
        yes_triggered = self.current_order is not None and self.current_order.side == "YES"
        no_triggered = self.current_order is not None and self.current_order.side == "NO"
        yes_filled = self.current_order.filled if self.current_order and self.current_order.side == "YES" else False
        no_filled = self.current_order.filled if self.current_order and self.current_order.side == "NO" else False
        yes_filled_size = self.current_order.filled_size if self.current_order and self.current_order.side == "YES" else 0.0
        no_filled_size = self.current_order.filled_size if self.current_order and self.current_order.side == "NO" else 0.0
        yes_order_id = self.current_order.order_id if self.current_order and self.current_order.side == "YES" else ""
        no_order_id = self.current_order.order_id if self.current_order and self.current_order.side == "NO" else ""
        yes_sold = self.current_order.sold if self.current_order and self.current_order.side == "YES" else False
        no_sold = self.current_order.sold if self.current_order and self.current_order.side == "NO" else False
        target_hit_price = self.current_order.target_hit_price if self.current_order else 0.0
        
        # Actual fill price (0 if not filled yet)
        actual_fill_price = self.current_order.avg_fill_price if self.current_order and self.current_order.filled else 0.0
        
        current_size = self._get_current_order_size() if state == "WAITING" else (self.current_order.filled_size if self.current_order else self.base_order_size)
        
        status = AssetStatus(
            asset=self.asset,
            enabled=self.enabled,
            trend_mode="in_window",
            primary_trend=self.cached_primary_trend,
            secondary_trend=self.cached_secondary_trend,
            macd_trend=self.cached_macd_trend,
            vki_trend=self.cached_vki_trend,
            keltner_trend=self.cached_keltner_trend,
            multi_trend=self.cached_multi_trend,
            lorentzian_trend=self.cached_lorentzian_trend,
            ai_trend=self.cached_ai_trend,
            decision=state,
            market_slug=market_slug,
            market_question=self.current_market.question if self.current_market else "",
            market_end_ts=self.target_window_end,
            time_left=time_left,
            yes_ask=self.yes_ask,
            yes_bid=self.yes_bid,
            no_ask=self.no_ask,
            no_bid=self.no_bid,
            yes_triggered=yes_triggered,
            no_triggered=no_triggered,
            yes_filled=yes_filled,
            no_filled=no_filled,
            yes_filled_size=yes_filled_size,
            no_filled_size=no_filled_size,
            yes_order_id=yes_order_id or "",
            no_order_id=no_order_id or "",
            yes_sold=yes_sold,
            no_sold=no_sold,
            trigger_price_yes=self.sell_target,
            trigger_price_no=self.sell_target,
            buy_limit_price=self.buy_price,
            avg_fill_price=actual_fill_price,
            sell_price=self.sell_target,
            order_size=current_size,
            target_hit_price=target_hit_price,
        )
        REGISTRY.upsert_status(status)

    def _record_trade(self, result: str, sell_price: float, winning_side: str):
        """Record completed trade."""
        if not self.current_order:
            return
        
        order = self.current_order
        market_slug = f"{self.asset.lower()}-updown-15m-{self.target_window_start}"
        
        # Determine fill price - prefer actual fill price over limit
        if order.avg_fill_price > 0:
            fill_price = order.avg_fill_price
            self.log.debug(f"📊 [{self.asset}] Using avg_fill_price: ${fill_price:.4f}")
        else:
            fill_price = order.limit_price
            self.log.warning(f"⚠️ [{self.asset}] avg_fill_price=0, using limit_price: ${fill_price:.4f}")
        
        if result == "WIN":
            # Revenue from selling (sell_size, not filled_size — fee reduces actual tokens held)
            sell_size = order.sell_size if order.sell_size > 0 else order.filled_size
            pnl = (sell_size * sell_price) - (order.filled_size * fill_price)
        else:
            pnl = -(order.filled_size * fill_price)
        
        order.pnl = pnl
        
        self._log_trade_csv(
            market_slug=market_slug,
            side=order.side,
            size=order.filled_size,
            buy_price=fill_price,
            sell_price=sell_price,
            pnl=pnl,
            result=result,
            winning_side=winning_side
        )
        
        trade = TradeRecord(
            ts=utc_ts(),
            asset=self.asset,
            market_slug=market_slug,
            side=order.side,
            size=order.filled_size,
            buy_price=fill_price,
            sell_price=sell_price,
            pnl=pnl,
            result=result,
            primary_trend=self.cached_primary_trend,
            secondary_trend=self.cached_secondary_trend,
            macd_trend=self.cached_macd_trend,
            vki_trend=self.cached_vki_trend,
            keltner_trend=self.cached_keltner_trend,
            multi_trend=self.cached_multi_trend,
            lorentzian_trend=self.cached_lorentzian_trend,
            ai_trend=self.cached_ai_trend,
            winning_side=winning_side
        )
        REGISTRY.add_trade(trade)
        
        # Update martingale - ONLY based on prediction (target reached or not)
        if result == "WIN":
            self.martingale.on_win()
            self.log.info(f"🎉 [{self.asset}] Prediction WIN! Martingale reset to step 0")
        else:
            self.martingale.on_loss()
            if self.martingale.last_result == "LOSS_RESET":
                self.log.warning(f"💀 [{self.asset}] Prediction LOSS at MAX STEP! Martingale RESET to step 0")
            else:
                self.log.info(f"💀 [{self.asset}] Prediction LOSS! Martingale step → {self.martingale.step}")

    def _verify_sell_order_thread(self, sell_order_id: str, order_info: dict):
        """
        Verify sell order in separate thread (doesn't block main loop).
        Updates trade stats with actual fill price.
        """
        try:
            self.log.info(f"🔍 [{self.asset}] Verification thread started for order {sell_order_id[:16]}...")
            
            # Wait a bit for order to settle
            time.sleep(5)
            
            max_attempts = 12  # ~60 seconds total
            actual_fill_price = None
            is_matched = False
            
            for attempt in range(max_attempts):
                try:
                    status = self.pm.get_order_status(sell_order_id)
                    if status:
                        if status.status.upper() == "MATCHED":
                            is_matched = True
                            api_price = status.avg_fill_price
                            target_price = order_info.get('target_hit_price', 0)
                            
                            # API often returns limit price ($0.05 floor) instead of actual fill price.
                            # Use target_hit_price (bid when target was triggered) if:
                            # - API price is suspiciously low (below buy price)
                            # - target_hit_price is available
                            if api_price > 0 and target_price > 0 and api_price < order_info.get('buy_price', 0):
                                actual_fill_price = target_price
                                self.log.info(
                                    f"✅ [{self.asset}] VERIFIED: SELL MATCHED — API reported ${api_price:.4f} (floor), "
                                    f"using target bid ${target_price:.4f}"
                                )
                            elif api_price > 0:
                                actual_fill_price = api_price
                                self.log.info(f"✅ [{self.asset}] VERIFIED: SELL MATCHED @ ${actual_fill_price:.4f}")
                            elif target_price > 0:
                                actual_fill_price = target_price
                                self.log.info(f"✅ [{self.asset}] VERIFIED: SELL MATCHED @ ${actual_fill_price:.4f} (from target bid)")
                            else:
                                actual_fill_price = order_info.get('buy_price', 0)
                                self.log.warning(f"⚠️ [{self.asset}] VERIFIED: SELL MATCHED — no price data, using buy price ${actual_fill_price:.4f}")
                            break
                        elif status.status.upper() in ("CANCELLED", "EXPIRED"):
                            self.log.warning(f"⚠️ [{self.asset}] VERIFIED: SELL {status.status} - trade LOSS (martingale unchanged)")
                            break
                except Exception as e:
                    self.log.warning(f"Verification attempt {attempt+1} failed: {e}")
                
                time.sleep(5)
            
            # Update trade record with actual results
            market_slug = order_info['market_slug']
            buy_price = order_info['buy_price']
            filled_size = order_info['filled_size']
            sell_size = order_info.get('sell_size', filled_size)  # sell_size accounts for fee buffer
            side = order_info['side']
            
            if is_matched and actual_fill_price:
                actual_pnl = (sell_size * actual_fill_price) - (filled_size * buy_price)
                self.log.info(f"📊 [{self.asset}] Final PnL: ${actual_pnl:.2f} (bought {filled_size} @ ${buy_price:.4f}, sold {sell_size} @ ${actual_fill_price:.4f})")
                
                # Update the trade in registry with actual values
                self._update_trade_verification(market_slug, "WIN", actual_fill_price, actual_pnl)
            else:
                # SELL failed - record as LOSS but don't change martingale (already handled)
                actual_pnl = -(filled_size * buy_price)
                self.log.warning(f"📊 [{self.asset}] SELL NOT FILLED - Final: LOSS, PnL: ${actual_pnl:.2f}")
                
                # Update the trade in registry - mark as LOSS
                self._update_trade_verification(market_slug, "LOSS", 0.0, actual_pnl)
                
        except Exception as e:
            self.log.error(f"❌ [{self.asset}] Verification thread error: {e}")
            import traceback
            traceback.print_exc()

    def _update_trade_verification(self, market_slug: str, verified_result: str, actual_sell_price: float, actual_pnl: float):
        """Update trade record with verified results."""
        try:
            # Find and update trade in registry
            for trade in REGISTRY.trades:
                if trade.market_slug == market_slug and trade.asset == self.asset:
                    trade.result = verified_result
                    trade.sell_price = actual_sell_price
                    trade.pnl = actual_pnl
                    REGISTRY._save_trades()
                    self.log.info(f"📝 [{self.asset}] Trade record updated: {verified_result}, PnL: ${actual_pnl:.2f}")
                    break
        except Exception as e:
            self.log.error(f"Failed to update trade verification: {e}")

    def emergency_sell(self) -> bool:
        """Sell open position on shutdown. Returns True if sold or nothing to sell."""
        order = self.current_order
        if not order:
            self.log.info(f"🛑 [{self.asset}] Shutdown: no open position")
            return True
        
        # PENDING order (not yet filled) — cancel it
        if order.fill_state == "PENDING" and order.order_id:
            self.log.info(f"🛑 [{self.asset}] Shutdown: cancelling PENDING order {order.order_id[:16]}...")
            self.pm.safe_cancel(order.order_id)
            return True
        
        # Already sold
        if order.sold:
            self.log.info(f"🛑 [{self.asset}] Shutdown: position already sold")
            return True
        
        # CONFIRMED_FILLED but not sold — must sell
        if order.fill_state != "CONFIRMED_FILLED" or not order.filled or order.filled_size <= 0:
            self.log.info(f"🛑 [{self.asset}] Shutdown: no filled position to sell (state={order.fill_state})")
            return True
        
        token_id = order.token_id
        if not token_id:
            self.log.error(f"🛑 [{self.asset}] Shutdown: CANNOT SELL — no token_id! Manual intervention needed. "
                          f"Position: {order.filled_size} {order.side} tokens")
            return False
        
        sell_size = order.sell_size if order.sell_size > 0 else math.floor(order.filled_size * (1 - SELL_FEE_BUFFER) * 100) / 100
        
        # Recheck fill size before selling
        try:
            recheck = self.pm.recheck_fill_size(order.order_id)
            if recheck and recheck.filled_size > order.filled_size:
                old_fill = order.filled_size
                order.filled_size = recheck.filled_size
                sell_size = math.floor(recheck.filled_size * (1 - SELL_FEE_BUFFER) * 100) / 100
                self.log.info(f"🛑 [{self.asset}] Shutdown recheck: fill {old_fill} → {recheck.filled_size}, sell_size={sell_size}")
        except Exception as e:
            self.log.warning(f"🛑 [{self.asset}] Shutdown recheck error: {e}")
        
        SELL_FLOOR_PRICE = 0.05
        self.log.warning(
            f"🛑 [{self.asset}] Shutdown: EMERGENCY SELL {sell_size} {order.side} tokens @ floor ${SELL_FLOOR_PRICE}"
        )
        
        # Try up to 3 times
        for attempt in range(3):
            try:
                result = self._place_sell_order(token_id, order.side, sell_size, SELL_FLOOR_PRICE)
                if result:
                    self.log.info(f"✅ [{self.asset}] Shutdown: SELL placed — {result.order_id[:16]}... status={result.status}")
                    order.sold = True
                    order.sell_order_id = result.order_id
                    return True
                self.log.warning(f"⚠️ [{self.asset}] Shutdown: SELL attempt {attempt+1}/3 returned None")
            except Exception as e:
                self.log.error(f"❌ [{self.asset}] Shutdown: SELL attempt {attempt+1}/3 error: {e}")
            time.sleep(2)
        
        self.log.critical(
            f"🚨 [{self.asset}] Shutdown: FAILED TO SELL after 3 attempts! "
            f"Manual intervention needed: {order.filled_size} {order.side} tokens, token_id={token_id[:30]}..."
        )
        return False

    def run(self):
        """Main bot loop."""
        if not self.enabled:
            self.log.info(f"[{self.asset}] Bot disabled")
            self._update_registry("DISABLED")
            while not self.stop_event.is_set():
                time.sleep(2)
            return
        
        if not self.pm.initialize():
            self.log.error("Polymarket client init failed")
            self._update_registry("ERROR")
            return
        
        self.log.info(f"🚀 Starting {self.asset} In-Window bot ({'DEMO' if self.is_demo else 'LIVE'})")
        self.log.info(f"   Decision trend: {self.decision_trend}")
        self.log.info(f"   Base size: {self.base_order_size} | Buy: ${self.buy_price} | Sell target: ${self.sell_target}")
        self.log.info(f"   Martingale: {len(self.martingale.multipliers)} steps, multipliers {self.martingale.multipliers}")
        
        # Warmup: pre-fetch historical data for Lorentzian
        self.log.info(f"📥 [{self.asset}] Lorentzian warmup — loading {self.lorentzian.lookback} candles...")
        self.lorentzian.warmup([self.symbol])
        self.log.info(f"✅ [{self.asset}] Lorentzian warmup complete")
        
        while not self.stop_event.is_set():
            try:
                now = time.time()
                
                # Get next window
                next_start, next_end = get_next_window(now)
                
                self.target_window_start = next_start
                self.target_window_end = next_end
                
                time_to_next_start = next_start - now
                market_slug = f"{self.asset.lower()}-updown-15m-{next_start}"
                
                # Check if window is in blacklisted hours
                if self._is_blacklisted_hour(next_start):
                    # Skip this window, wait until it ends
                    while time.time() < next_end and not self.stop_event.is_set():
                        self._update_registry("BLACKLISTED", market_slug)
                        time.sleep(1)
                    continue
                
                self._update_registry("WAITING", market_slug)
                
                # === PHASE 1: Wait and prepare BEFORE window starts ===
                if time_to_next_start > 0:
                    self.log.info(f"⏳ [{self.asset}] Next window: {time.strftime('%H:%M:%S', time.localtime(next_start))} - {time.strftime('%H:%M:%S', time.localtime(next_end))}")
                    self.log.info(f"   Waiting {int(time_to_next_start)}s | Martingale step {self.martingale.step} ({self.martingale.get_multiplier()}x)")
                    
                    # Calculate when to fetch trends (10s before window)
                    trend_fetch_time = next_start - self.multi_fetch_before
                    
                    # Wait until trend_fetch_before seconds before start
                    while time.time() < trend_fetch_time and not self.stop_event.is_set():
                        self._update_registry("WAITING", market_slug)
                        time.sleep(1)
                    
                    if self.stop_event.is_set():
                        break
                    
                    # === PHASE 1b: Fetch ALL trends (10s before window) ===
                    self.log.info(f"🔄 [{self.asset}] Fetching trends ({self.multi_fetch_before}s before window)...")
                    self.cached_primary_trend, self.cached_secondary_trend, self.cached_macd_trend, self.cached_vki_trend, self.cached_keltner_trend, self.cached_multi_trend, self.cached_lorentzian_trend, self.cached_ai_trend = self._fetch_trends()
                    
                    # Get decision
                    trend_value = self._get_decision_trend_value()
                    if not trend_value:
                        self.log.warning(f"⚠️ [{self.asset}] No trend available, skipping window")
                        # Wait for window to pass
                        while time.time() < next_end and not self.stop_event.is_set():
                            time.sleep(1)
                        continue
                    
                    buy_side = "YES" if trend_value == "UP" else "NO"
                    self.log.info(f"📊 [{self.asset}] Decision: {self.decision_trend}={trend_value} → BUY {buy_side}")
                    
                    # Find market BEFORE window starts
                    market = self._find_market_for_window(next_start)
                    if not market:
                        self.log.error(f"❌ [{self.asset}] Market not found")
                        while time.time() < next_end and not self.stop_event.is_set():
                            time.sleep(1)
                        continue
                    
                    self.current_market = market
                    self.log.info(f"📋 [{self.asset}] Market: neg_risk={market.neg_risk}, end_ts={int(market.end_ts)}")
                    # Keep our generated slug (condition_id may be different format)
                    
                    # === PHASE 1c: Wait for exact window start ===
                    self.log.info(f"✅ [{self.asset}] Ready! Waiting for window start...")
                    while time.time() < next_start and not self.stop_event.is_set():
                        self._update_registry("READY", market_slug)
                        time.sleep(0.1)  # Check more frequently near start
                    
                    if self.stop_event.is_set():
                        break
                else:
                    # Window already started - fetch trends now
                    self.log.info(f"🔄 [{self.asset}] Fetching trends...")
                    self.cached_primary_trend, self.cached_secondary_trend, self.cached_macd_trend, self.cached_vki_trend, self.cached_keltner_trend, self.cached_multi_trend, self.cached_lorentzian_trend, self.cached_ai_trend = self._fetch_trends()
                    
                    trend_value = self._get_decision_trend_value()
                    if not trend_value:
                        self.log.warning(f"⚠️ [{self.asset}] No trend available, skipping window")
                        continue
                    
                    buy_side = "YES" if trend_value == "UP" else "NO"
                    self.log.info(f"📊 [{self.asset}] Decision: {self.decision_trend}={trend_value} → BUY {buy_side}")
                    
                    market = self._find_market_for_window(next_start)
                    if not market:
                        self.log.error(f"❌ [{self.asset}] Market not found")
                        continue
                    
                    self.current_market = market
                    self.log.info(f"📋 [{self.asset}] Market: neg_risk={market.neg_risk}, end_ts={int(market.end_ts)}")
                    # Keep our generated slug
                
                # === PHASE 2: Window started - IMMEDIATELY place order ===
                self.log.info(f"🟢 [{self.asset}] Window STARTED!")
                
                token_id = market.yes_token_id if buy_side == "YES" else market.no_token_id
                
                # Get order size with martingale
                order_size = self._get_current_order_size()
                
                # Place buy order IMMEDIATELY
                self.log.info(f"🎯 [{self.asset}] PLACING ORDER: {buy_side} @ ${self.buy_price} x {order_size} (step {self.martingale.step})")
                try:
                    buy_result = self._place_buy_order(token_id, buy_side, order_size)
                except CloudflareBlockError:
                    # IP is blocked by Cloudflare - no point retrying rapidly
                    self.log.error(f"🛑 [{self.asset}] Cloudflare WAF block detected! Pausing until next window.")
                    self._update_registry("CF_BLOCKED", market_slug)
                    # Wait out the rest of this window
                    while time.time() < next_end and not self.stop_event.is_set():
                        time.sleep(5)
                    # Extra cooldown: wait 60s after window ends before trying again
                    cooldown = 60
                    self.log.warning(f"🛑 [{self.asset}] Cloudflare cooldown: waiting {cooldown}s before next attempt...")
                    for _ in range(cooldown):
                        if self.stop_event.is_set():
                            break
                        time.sleep(1)
                    continue
                except InsufficientBalanceError:
                    self.log.warning(
                        f"💸 [{self.asset}] INSUFFICIENT BALANCE — skipping this window. "
                        f"Martingale stays at step {self.martingale.step}."
                    )
                    self._update_registry("NO_BALANCE", market_slug)
                    # Skip this window, continue to next
                    while time.time() < next_end and not self.stop_event.is_set():
                        time.sleep(2)
                    continue
                
                if not buy_result:
                    self.log.error(f"❌ [{self.asset}] Failed to place order — waiting for window end")
                    self._update_registry("ORDER_FAILED", market_slug)
                    # Don't retry within this window — wait it out
                    while time.time() < next_end and not self.stop_event.is_set():
                        time.sleep(2)
                    continue
                
                order_id = buy_result.order_id
                
                # ── STATE: PENDING ──
                # Order sent, but NO position yet. Do NOT set filled=True until confirmed.
                self.current_order = OrderState(
                    side=buy_side,
                    order_id=order_id,
                    token_id=token_id,
                    filled=False,
                    filled_size=0.0,
                    limit_price=self.buy_price,
                    avg_fill_price=0.0,
                    fill_state="PENDING",
                )
                self._update_registry("PENDING_FILL", market_slug)
                
                # Hard deadline: window_end - FILL_DEADLINE_BUFFER
                fill_deadline_ts = next_end - FILL_DEADLINE_BUFFER
                remaining_for_fill = max(0, fill_deadline_ts - time.time())
                self.log.info(f"⏱️ [{self.asset}] Fill deadline: {int(remaining_for_fill)}s (window_end - {FILL_DEADLINE_BUFFER}s)")
                
                # ── Confirm fill: quick path from place_order response, then REST polling ──
                is_filled = False
                filled_size = 0.0
                avg_fill_price = 0.0
                
                if buy_result.status.upper() == "MATCHED" and buy_result.filled_size > 0:
                    # Response says MATCHED with fill – verify via REST for accurate avg_price
                    self.log.info(f"📋 [{self.asset}] place_order says MATCHED, verifying fill via REST...")
                    exec_info = self.pm.get_execution_info(order_id, timeout_s=min(8.0, remaining_for_fill), poll_s=1.0)
                    if exec_info and exec_info.filled_size > 0:
                        is_filled = True
                        filled_size = exec_info.filled_size
                        avg_fill_price = exec_info.avg_price
                        self.log.info(f"✅ [{self.asset}] CONFIRMED_FILLED via exec_info: {filled_size} @ ${avg_fill_price:.4f}")
                    else:
                        # exec_info unavailable – trust response data
                        is_filled = True
                        filled_size = buy_result.filled_size
                        avg_fill_price = buy_result.avg_fill_price
                        self.log.info(f"✅ [{self.asset}] CONFIRMED_FILLED from response: {filled_size} @ ${avg_fill_price:.4f} (exec_info unavailable)")
                else:
                    # Not instantly matched – poll REST until deadline
                    status_label = buy_result.status.upper()
                    self.log.info(f"⏳ [{self.asset}] Order status={status_label}, polling for fill until deadline ({int(remaining_for_fill)}s)...")
                    fill_info = self.pm.confirm_fill_until_deadline(
                        order_id=order_id,
                        deadline_ts=fill_deadline_ts,
                        min_fill_threshold=0.01,
                        poll_interval=1.5,
                    )
                    if fill_info and fill_info.filled_size > 0:
                        is_filled = True
                        filled_size = fill_info.filled_size
                        avg_fill_price = fill_info.avg_price
                        self.log.info(f"✅ [{self.asset}] CONFIRMED_FILLED via polling: {filled_size} @ ${avg_fill_price:.4f}")
                
                # ── Handle unfilled at deadline ──
                if not is_filled or filled_size <= 0:
                    self.current_order.fill_state = "UNFILLED_AT_DEADLINE"
                    self.log.warning(f"⏱️ [{self.asset}] UNFILLED_AT_DEADLINE → treating as NO TRADE (no PnL, no SELL, no history)")
                    self.pm.safe_cancel(order_id)
                    # Reset – do NOT record trade, do NOT update martingale
                    self.current_order = None
                    self.current_market = None
                    # Wait for window to finish before moving on
                    while time.time() < next_end and not self.stop_event.is_set():
                        self._update_registry("UNFILLED", market_slug)
                        time.sleep(1)
                    continue
                
                # ── STATE: CONFIRMED_FILLED – now we have a real position ──
                self.current_order.fill_state = "CONFIRMED_FILLED"
                self.current_order.filled = True
                self.current_order.filled_size = filled_size
                self.current_order.avg_fill_price = avg_fill_price
                
                # ── Safety: validate fill size vs order size ──
                if filled_size > order_size * 1.1:
                    self.log.critical(
                        f"🚨 [{self.asset}] FILL SIZE MISMATCH! Ordered {order_size} but filled {filled_size} "
                        f"({filled_size/order_size:.1f}x). Capping to {order_size}. "
                        f"Check neg_risk setting (currently: {self.current_market.neg_risk if self.current_market else 'N/A'})"
                    )
                    filled_size = order_size
                    self.current_order.filled_size = filled_size
                
                # Calculate sellable size: reduce by fee buffer (3%) and round DOWN to 2 decimals
                # Polymarket charges ~2% fee on buy, so we receive fewer tokens than filled_size
                sell_size = math.floor(filled_size * (1 - SELL_FEE_BUFFER) * 100) / 100
                self.current_order.sell_size = sell_size
                self.log.info(f"💰 [{self.asset}] Fill: {filled_size} tokens, sell size: {sell_size} (−{SELL_FEE_BUFFER*100:.0f}% fee buffer)")
                
                # === PHASE 3: Start WebSocket and monitor for sell opportunity ===
                # Calculate dynamic sell target based on martingale step
                current_sell_target = self._get_current_sell_target()
                if current_sell_target != self.sell_target:
                    self.log.info(f"📈 [{self.asset}] Position: {filled_size} {buy_side} @ ${avg_fill_price:.4f} | Target: ${current_sell_target:.2f} (reduced from ${self.sell_target} at step {self.martingale.step})")
                else:
                    self.log.info(f"📈 [{self.asset}] Position: {filled_size} {buy_side} @ ${avg_fill_price:.4f} | Target: ${current_sell_target:.2f}")
                self._start_websocket(self.current_market)
                
                # Wait for WebSocket to receive initial price data
                self.log.info(f"📡 [{self.asset}] Waiting for price data...")
                time.sleep(2)
                
                sell_order_placed = False
                sell_price_at_target = None
                sell_order_id_for_verify = None
                sell_attempts = 0
                max_sell_attempts = 3
                last_known_bid = 0.0  # Track last valid price for end-of-window check
                
                while time.time() < next_end and not self.stop_event.is_set():
                    self._update_prices_from_ws()
                    self._update_registry("HOLDING", market_slug)
                    
                    # Already placed sell - just wait for window end
                    if sell_order_placed:
                        time.sleep(1)
                        continue
                    
                    # Max attempts reached - stop trying
                    if sell_attempts >= max_sell_attempts:
                        time.sleep(1)
                        continue
                    
                    # Check if we can sell at target
                    current_bid = self._get_current_bid(buy_side)
                    
                    # Skip if invalid price (0 or >= 1.0 are not real market prices)
                    if current_bid <= 0 or current_bid >= 1.0:
                        time.sleep(1)
                        continue
                    
                    # Track last valid bid for end-of-window evaluation
                    last_known_bid = current_bid
                    
                    # Price reached target - place aggressive sell (market-like)
                    if current_bid >= current_sell_target:
                        sell_attempts += 1
                        self.log.info(f"🎯 [{self.asset}] TARGET HIT! Bid: ${current_bid:.4f} >= ${current_sell_target:.2f} (attempt {sell_attempts}/{max_sell_attempts})")
                        
                        # ── Pre-SELL: recheck fill size (catch additional fills since confirm) ──
                        try:
                            recheck = self.pm.recheck_fill_size(order_id)
                            if recheck and recheck.filled_size > filled_size:
                                old_fill = filled_size
                                filled_size = recheck.filled_size
                                avg_fill_price = recheck.avg_price if recheck.avg_price > 0 else avg_fill_price
                                self.current_order.filled_size = filled_size
                                self.current_order.avg_fill_price = avg_fill_price
                                sell_size = math.floor(filled_size * (1 - SELL_FEE_BUFFER) * 100) / 100
                                self.current_order.sell_size = sell_size
                                self.log.info(
                                    f"📈 [{self.asset}] Fill recheck: {old_fill} → {filled_size} tokens "
                                    f"(+{filled_size - old_fill:.2f}), new sell size: {sell_size}"
                                )
                        except Exception as e:
                            self.log.warning(f"⚠️ [{self.asset}] Fill recheck error: {e} — using original fill {filled_size}")
                        
                        # ── Pre-SELL balance log (informational only) ──
                        # Note: CLOB /balances API may not report conditional token balances (ERC-1155).
                        # Fill was already confirmed via confirm_fill_until_deadline, so proceed with sell.
                        if not self.is_demo and sell_size > 0:
                            try:
                                token_balance = self.pm.get_token_balance(token_id)
                                if token_balance >= sell_size:
                                    self.log.info(f"✅ [{self.asset}] Balance check OK: {token_balance} >= {sell_size}")
                                else:
                                    self.log.warning(
                                        f"⚠️ [{self.asset}] Balance API reports {token_balance}, need {sell_size}. "
                                        f"Fill was confirmed — proceeding with SELL anyway."
                                    )
                            except Exception as e:
                                self.log.warning(f"⚠️ [{self.asset}] Balance check error: {e} — proceeding with SELL")
                        
                        # Place sell order at aggressive floor price ($0.05)
                        # CLOB gives price improvement: SELL limit $0.05 fills at best bid
                        # This guarantees execution even if bid drops between check and order
                        SELL_FLOOR_PRICE = 0.05
                        self.log.info(f"📤 [{self.asset}] Placing SELL @ floor ${SELL_FLOOR_PRICE} (bid was ${current_bid:.4f})")
                        sell_result = self._place_sell_order(token_id, buy_side, sell_size, SELL_FLOOR_PRICE)
                        
                        if sell_result:
                            sell_order_placed = True
                            sell_price_at_target = current_bid  # Bid when target was hit (for PnL tracking)
                            sell_order_id_for_verify = sell_result.order_id
                            self.current_order.sold = True
                            self.current_order.sell_order_id = sell_result.order_id
                            self.current_order.target_hit_price = current_bid
                            
                            # Log actual fill info from response
                            if sell_result.status.upper() == "MATCHED":
                                # API reports floor price ($0.05), not actual fill — use current_bid
                                self.log.info(f"✅ [{self.asset}] TARGET HIT - Sell FILLED (bid was ${current_bid:.4f}, floor limit ${SELL_FLOOR_PRICE})")
                            else:
                                self.log.info(f"✅ [{self.asset}] TARGET HIT - Sell order sent @ floor ${SELL_FLOOR_PRICE} (bid was ${current_bid:.4f}, status: {sell_result.status})")
                        else:
                            self.log.warning(f"⚠️ [{self.asset}] Sell order failed, will retry ({max_sell_attempts - sell_attempts} attempts left)")
                            time.sleep(2)  # Wait before retry
                    
                    time.sleep(1)
                
                # If we hit target but all sell attempts failed, still count as WIN for martingale
                # (prediction was correct, execution failed)
                target_was_hit = sell_attempts > 0
                
                # Stop WebSocket
                self._stop_websocket()
                
                if self.stop_event.is_set():
                    break
                
                # === PHASE 4: Record result ===
                # Zasada (prediction market):
                #   1) SELL placed -> WIN (profit realized)
                #   2) No SELL, but last known price > $0.50 -> WIN (our side is winning)
                #   3) No SELL, last known price <= $0.50 -> LOSS (our side is losing)
                # $0.50 is the mid-point threshold: above = winning token
                WIN_THRESHOLD = 0.50
                
                if sell_order_placed:
                    # TIER 1: WIN — a sell order was placed (profit realized)
                    self.log.info(f"🎉 [{self.asset}] WIN — SELL placed @ ${sell_price_at_target:.4f}")
                    self._record_trade("WIN", sell_price_at_target, buy_side)
                    
                    # Spawn verification thread (doesn't block main loop)
                    verify_info = {
                        'market_slug': market_slug,
                        'buy_price': avg_fill_price,
                        'filled_size': filled_size,
                        'sell_size': sell_size,
                        'side': buy_side,
                        'target_hit_price': sell_price_at_target
                    }
                    verify_thread = threading.Thread(
                        target=self._verify_sell_order_thread,
                        args=(sell_order_id_for_verify, verify_info),
                        daemon=True
                    )
                    verify_thread.start()
                    self.log.info(f"🔄 [{self.asset}] Verification thread spawned")
                elif last_known_bid > WIN_THRESHOLD:
                    # TIER 2: WIN — we did not sell, but price > $0.50 means our side is winning
                    self.log.info(
                        f"✅ [{self.asset}] WIN (price > ${WIN_THRESHOLD}) — last bid ${last_known_bid:.4f} "
                        f"(buy was ${avg_fill_price:.4f})"
                    )
                    self._record_trade("WIN", last_known_bid, buy_side)
                else:
                    # TIER 3: LOSS — cena ≤ $0.50 = nasza strona przegrywa
                    losing_side = "NO" if buy_side == "YES" else "YES"
                    self.log.info(
                        f"💀 [{self.asset}] LOSS — last bid ${last_known_bid:.4f} <= ${WIN_THRESHOLD} "
                        f"(buy was ${avg_fill_price:.4f})"
                    )
                    self._record_trade("LOSS", last_known_bid, losing_side)
                
                # Reset for next window
                self.current_order = None
                self.current_market = None
                self.target_window_start = 0
                self.target_window_end = 0
                self.cached_primary_trend = None
                self.cached_secondary_trend = None
                self.cached_macd_trend = None
                self.cached_vki_trend = None
                self.cached_keltner_trend = None
                self.cached_multi_trend = None
                self.cached_lorentzian_trend = None
                self.cached_ai_trend = None
                
            except Exception as e:
                self.log.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)


def start_dashboard(port: int, stop_event: threading.Event):
    """Start dashboard in separate thread."""
    from dashboard import app
    
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except Exception:
        pass


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='PolySniper In-Window Bot with Martingale')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--dashboard-port', type=int, default=8050)
    parser.add_argument('--no-dashboard', action='store_true')
    args = parser.parse_args()
    
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)
    
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    is_demo = not args.live
    if cfg.get('mode', {}).get('demo') is False:
        is_demo = False
    
    # Dashboard settings
    dashboard_cfg = cfg.get('dashboard', {})
    REGISTRY.dashboard_title = dashboard_cfg.get('title', 'PolySniper In-Window')
    
    # Strategy settings
    strategy_cfg = cfg.get('strategy', {})
    buy_price = float(strategy_cfg.get('buy_price', 0.55))
    sell_target = float(strategy_cfg.get('sell_target', 0.85))
    order_delay = int(strategy_cfg.get('order_delay', 3))
    global_order_size = float(strategy_cfg.get('order_size', 5.0))
    
    # Decision trend
    decision_trend = cfg.get('decision_trend', 'M')
    
    # Blacklist hours (UTC)
    blacklist_hours = cfg.get('blacklist_hours', [])
    if blacklist_hours:
        print(f"  Blacklisted hours (UTC): {blacklist_hours}")
    
    # ADX config
    adx_cfg = cfg.get('adx', {})
    adx_primary = adx_cfg.get('primary', {})
    adx_secondary = adx_cfg.get('secondary', {})
    
    # MACD config
    macd_cfg = cfg.get('macd', {})
    
    # Keltner Channels config
    keltner_cfg = cfg.get('keltner', {})
    
    # Multi-indicator config
    multi_cfg = cfg.get('multi', {})
    multi_fetch_before = int(multi_cfg.get('fetch_before', 10))
    
    # Lorentzian Classification config
    lorentzian_cfg = cfg.get('lorentzian', {})
    
    # AI prediction config
    ai_cfg = cfg.get('ai', {})
    
    # Martingale config
    martingale_cfg = cfg.get('martingale', {})
    martingale_multipliers = martingale_cfg.get('multipliers', DEFAULT_MARTINGALE_MULTIPLIERS)
    # Validate multipliers
    if not isinstance(martingale_multipliers, list) or len(martingale_multipliers) == 0:
        print(f"Warning: Invalid martingale multipliers, using defaults")
        martingale_multipliers = DEFAULT_MARTINGALE_MULTIPLIERS
    
    # Target reduction for high martingale steps
    target_reduction_cfg = martingale_cfg.get('target_reduction', {})
    target_reduction_second_last = float(target_reduction_cfg.get('second_last', 0.0))
    target_reduction_last = float(target_reduction_cfg.get('last', 0.0))
    if target_reduction_second_last > 0 or target_reduction_last > 0:
        print(f"  Target reduction: second_last={target_reduction_second_last}, last={target_reduction_last}")
    
    # Assets
    assets_cfg = cfg.get('assets', {})
    if not assets_cfg:
        assets_cfg = {
            'BTC': {'enabled': True, 'symbol': 'BTC/USDT'},
            'ETH': {'enabled': True, 'symbol': 'ETH/USDT'},
            'XRP': {'enabled': True, 'symbol': 'XRP/USDT'},
            'SOL': {'enabled': True, 'symbol': 'SOL/USDT'},
        }
    
    shared_stop = threading.Event()
    bots: List[InWindowBot] = []
    
    print(f"\n{'='*60}")
    print(f"  PolySniper In-Window v4 - {'DEMO' if is_demo else '🔴 LIVE'} MODE")
    print(f"  Decision Trend: {decision_trend}")
    print(f"  Buy Price: ${buy_price} | Sell Target: ${sell_target}")
    print(f"  Base Order Size: {global_order_size}")
    print(f"  Martingale: {len(martingale_multipliers)} steps {martingale_multipliers}")
    print(f"{'='*60}\n")
    
    # Start bots
    for asset, a_cfg in assets_cfg.items():
        merged = dict(a_cfg)
        merged['buy_price'] = buy_price
        merged['sell_target'] = sell_target
        merged['order_delay'] = order_delay
        merged['multi_fetch_before'] = multi_fetch_before
        merged['order_size'] = a_cfg.get('order_size', global_order_size)
        merged['_decision_trend'] = a_cfg.get('decision_trend') or decision_trend
        merged['_adx'] = {
            'primary': adx_primary,
            'secondary': adx_secondary,
        }
        merged['_macd'] = macd_cfg
        merged['_keltner'] = keltner_cfg
        merged['_multi'] = multi_cfg
        merged['_lorentzian'] = lorentzian_cfg
        merged['_ai'] = ai_cfg
        merged['blacklist_hours'] = blacklist_hours
        merged['_martingale_multipliers'] = martingale_multipliers
        merged['_target_reduction_second_last'] = target_reduction_second_last
        merged['_target_reduction_last'] = target_reduction_last
        
        bot = InWindowBot(
            asset=asset,
            cfg=merged,
            is_demo=is_demo,
            shared_stop=shared_stop
        )
        bots.append(bot)
        bot.start()
    
    # Start dashboard
    dash_thread = None
    if not args.no_dashboard:
        dash_thread = threading.Thread(
            target=start_dashboard,
            args=(args.dashboard_port, shared_stop),
            daemon=True
        )
        dash_thread.start()
        print(f"📊 Dashboard: http://localhost:{args.dashboard_port}")
    
    def graceful_shutdown(reason="unknown"):
        """Sell open positions and stop all bots."""
        print(f"\n⏹️ Stopping bots ({reason}) — selling open positions...")
        shared_stop.set()
        
        # Emergency sell all open positions before threads exit
        has_failures = False
        for bot in bots:
            if not bot.is_alive():
                continue
            try:
                if not bot.emergency_sell():
                    has_failures = True
            except Exception as e:
                print(f"❌ {bot.asset} emergency sell error: {e}")
                has_failures = True
        
        if has_failures:
            print("🚨 Some positions could NOT be sold — check logs and manually close on Polymarket!")
        
        for bot in bots:
            bot.join(timeout=10)
        print("✅ Done")
    
    # Handle SIGTERM (systemd stop/restart)
    def sigterm_handler(signum, frame):
        graceful_shutdown("SIGTERM")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    # Main loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        graceful_shutdown("Ctrl+C")


if __name__ == "__main__":
    main()
