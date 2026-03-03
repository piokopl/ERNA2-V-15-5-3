#!/usr/bin/env python3
"""In-memory registry shared between bots and the web dashboard.

This module provides:
- read-only dashboard views
- lightweight status and stats aggregation
- persistent trade history (saved to JSON)

Thread safety: uses a single global lock.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


TRADES_FILE = "state/trades_history.json"


@dataclass
class TradeRecord:
    ts: str
    asset: str
    market_slug: str
    side: str  # YES/NO (what we bought)
    size: float
    buy_price: float
    sell_price: float
    pnl: float
    result: str  # WIN/LOSS
    # Trend tracking
    primary_trend: Optional[str] = None   # UP/DOWN/None
    secondary_trend: Optional[str] = None # UP/DOWN/None
    macd_trend: Optional[str] = None      # UP/DOWN/None
    vki_trend: Optional[str] = None       # UP/DOWN/None (RSI Kernel Optimized)
    keltner_trend: Optional[str] = None   # UP/DOWN/None (Keltner Channels)
    multi_trend: Optional[str] = None     # UP/DOWN/None (Multi-indicator)
    lorentzian_trend: Optional[str] = None # UP/DOWN/None (Lorentzian Classification)
    ai_trend: Optional[str] = None        # UP/DOWN/None (AI prediction)
    winning_side: Optional[str] = None    # YES/NO - which side actually won


@dataclass
class AssetStatus:
    asset: str
    enabled: bool
    trend_mode: str
    primary_trend: Optional[str] = None
    secondary_trend: Optional[str] = None
    macd_trend: Optional[str] = None
    vki_trend: Optional[str] = None
    keltner_trend: Optional[str] = None  # Keltner Channels (K)
    multi_trend: Optional[str] = None  # Multi-indicator (M)
    lorentzian_trend: Optional[str] = None  # Lorentzian Classification (L)
    ai_trend: Optional[str] = None  # AI prediction (A)
    decision: str = ""  # IDLE/MONITORING/TRADING/etc
    market_slug: str = ""
    market_question: str = ""
    market_end_ts: int = 0
    time_left: int = 0  # seconds until market ends
    
    # Current prices from WebSocket
    yes_ask: float = 0.0
    yes_bid: float = 0.0
    no_ask: float = 0.0
    no_bid: float = 0.0
    
    # YES order state
    yes_triggered: bool = False
    yes_order_id: str = ""
    yes_filled: bool = False
    yes_filled_size: float = 0.0
    yes_sold: bool = False
    yes_pnl: float = 0.0
    
    # NO order state
    no_triggered: bool = False
    no_order_id: str = ""
    no_filled: bool = False
    no_filled_size: float = 0.0
    no_sold: bool = False
    no_pnl: float = 0.0
    
    # Config
    trigger_price_yes: float = 0.0
    trigger_price_no: float = 0.0
    buy_limit_price: float = 0.0  # Config limit price
    avg_fill_price: float = 0.0   # Actual fill price from exchange
    sell_price: float = 0.0
    order_size: float = 0.0
    target_hit_price: float = 0.0  # Price when target was hit (for display)
    
    position_side: str = ""  # Legacy compatibility
    position_tokens: float = 0.0
    martingale_level: int = 0
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BotRegistry:
    def __init__(self, trades_file: str = TRADES_FILE):
        self._lock = threading.Lock()
        self._status: Dict[str, AssetStatus] = {}
        self._trades: List[TradeRecord] = []
        self._trades_file = trades_file
        self.dashboard_title: str = "PolySniper In-Window"
        
        # Ensure state directory exists
        Path(self._trades_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing trades from disk
        self._load_trades()

    @property
    def trades(self) -> List[TradeRecord]:
        """Public read access to trades list."""
        return self._trades

    def _load_trades(self) -> None:
        """Load trades from JSON file on startup."""
        try:
            if Path(self._trades_file).exists():
                with open(self._trades_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                trades_data = data.get("trades", [])
                for t in trades_data:
                    self._trades.append(TradeRecord(**t))
                
                print(f"📊 Loaded {len(self._trades)} trades from {self._trades_file}")
        except Exception as e:
            print(f"⚠️ Failed to load trades history: {e}")

    def _save_trades(self) -> None:
        """Save trades to JSON file."""
        try:
            data = {
                "trades": [asdict(t) for t in self._trades],
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Write to temp file first, then rename (atomic operation)
            temp_file = self._trades_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            Path(temp_file).rename(self._trades_file)
            
        except Exception as e:
            print(f"⚠️ Failed to save trades history: {e}")

    def upsert_status(self, status: AssetStatus) -> None:
        with self._lock:
            status.last_update = datetime.utcnow().isoformat()
            self._status[status.asset] = status

    def add_trade(self, trade: TradeRecord, max_records: int = 5000) -> None:
        with self._lock:
            self._trades.append(trade)
            if len(self._trades) > max_records:
                self._trades = self._trades[-max_records:]
            
            # Persist to disk
            self._save_trades()

    def update_trade_result(self, market_slug: str, side: str, new_result: str, new_pnl: float, winning_side: str) -> bool:
        """Update PENDING/SETTLEMENT trade to WIN/LOSS based on actual market outcome.
        
        Args:
            market_slug: Market identifier
            side: YES or NO (which side we bought)
            new_result: WIN or LOSS
            new_pnl: Updated PnL
            winning_side: YES or NO (which side actually won)
        
        Returns:
            True if trade was updated, False if not found
        """
        with self._lock:
            for trade in self._trades:
                if (trade.market_slug == market_slug and 
                    trade.side == side and 
                    trade.result in ("SETTLEMENT", "PENDING")):
                    trade.result = new_result
                    trade.pnl = new_pnl
                    trade.winning_side = winning_side
                    # Set sell_price based on payout
                    if new_result == "WIN":
                        trade.sell_price = 1.0  # Winning token pays out $1.00
                    else:
                        trade.sell_price = 0.0  # Losing token pays out $0.00
                    self._save_trades()
                    return True
            return False

    def get_settlement_trades(self, market_slug: str) -> List[TradeRecord]:
        """Get all SETTLEMENT trades for a specific market."""
        with self._lock:
            return [t for t in self._trades 
                    if t.market_slug == market_slug and t.result == "SETTLEMENT"]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": {k: asdict(v) for k, v in self._status.items()},
                "trades": [asdict(t) for t in self._trades],
            }

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: asdict(v) for k, v in self._status.items()}

    def get_trades(self, asset: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            trades = self._trades
            if asset:
                trades = [t for t in trades if t.asset == asset]
            return [asdict(t) for t in trades[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Return per-asset stats + TOTAL."""
        with self._lock:
            per: Dict[str, Dict[str, Any]] = {}
            total = {"trades": 0, "wins": 0, "losses": 0, "settlements": 0, "pnl": 0.0}
            for t in self._trades:
                s = per.setdefault(t.asset, {"trades": 0, "wins": 0, "losses": 0, "settlements": 0, "pnl": 0.0})
                s["trades"] += 1
                total["trades"] += 1
                if t.result == "WIN":
                    s["wins"] += 1
                    total["wins"] += 1
                elif t.result == "LOSS":
                    s["losses"] += 1
                    total["losses"] += 1
                elif t.result == "SETTLEMENT":
                    s["settlements"] += 1
                    total["settlements"] += 1
                s["pnl"] += float(t.pnl)
                total["pnl"] += float(t.pnl)

            # win-rate (excluding settlements from denominator)
            for k, s in per.items():
                decided = s["wins"] + s["losses"]
                s["win_rate"] = (s["wins"] / decided * 100.0) if decided else 0.0
            decided_total = total["wins"] + total["losses"]
            total["win_rate"] = (total["wins"] / decided_total * 100.0) if decided_total else 0.0
            return {"per_asset": per, "total": total}

    def get_trend_accuracy(self) -> Dict[str, Any]:
        """
        Calculate accuracy for each trend source.
        
        For Dual Limit strategy:
        - Trend UP predicts YES will win
        - Trend DOWN predicts NO will win
        
        A trend is "correct" if:
        - It predicted UP and winning_side was YES
        - It predicted DOWN and winning_side was NO
        
        A trend is "wrong" if:
        - It predicted UP but winning_side was NO
        - It predicted DOWN but winning_side was YES
        
        Trades with unknown winning_side (SETTLEMENT) are excluded.
        """
        with self._lock:
            # Initialize counters for each trend source
            trends = {
                "primary": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
                "secondary": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
                "macd": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
                "vki": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
                "multi": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
                "lorentzian": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
                "ai": {"correct": 0, "wrong": 0, "total": 0, "accuracy": 0.0},
            }
            
            for t in self._trades:
                # Skip trades without known outcome
                if not t.winning_side or t.winning_side not in ("YES", "NO"):
                    continue
                
                winning_side = t.winning_side
                
                # Check primary trend
                if t.primary_trend and t.primary_trend in ("UP", "DOWN"):
                    trends["primary"]["total"] += 1
                    # UP predicts YES, DOWN predicts NO
                    trend_predicts = "YES" if t.primary_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["primary"]["correct"] += 1
                    else:
                        trends["primary"]["wrong"] += 1
                
                # Check secondary trend
                if t.secondary_trend and t.secondary_trend in ("UP", "DOWN"):
                    trends["secondary"]["total"] += 1
                    trend_predicts = "YES" if t.secondary_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["secondary"]["correct"] += 1
                    else:
                        trends["secondary"]["wrong"] += 1
                
                # Check MACD trend (D)
                if t.macd_trend and t.macd_trend in ("UP", "DOWN"):
                    trends["macd"]["total"] += 1
                    trend_predicts = "YES" if t.macd_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["macd"]["correct"] += 1
                    else:
                        trends["macd"]["wrong"] += 1
                
                # Check VKI trend
                vki_trend = getattr(t, 'vki_trend', None)
                if vki_trend and vki_trend in ("UP", "DOWN"):
                    trends["vki"]["total"] += 1
                    trend_predicts = "YES" if vki_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["vki"]["correct"] += 1
                    else:
                        trends["vki"]["wrong"] += 1
                
                # Check Multi trend (M)
                multi_trend = getattr(t, 'multi_trend', None)
                if multi_trend and multi_trend in ("UP", "DOWN"):
                    trends["multi"]["total"] += 1
                    trend_predicts = "YES" if multi_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["multi"]["correct"] += 1
                    else:
                        trends["multi"]["wrong"] += 1
                
                # Check Lorentzian trend (L)
                lorentzian_trend = getattr(t, 'lorentzian_trend', None)
                if lorentzian_trend and lorentzian_trend in ("UP", "DOWN"):
                    trends["lorentzian"]["total"] += 1
                    trend_predicts = "YES" if lorentzian_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["lorentzian"]["correct"] += 1
                    else:
                        trends["lorentzian"]["wrong"] += 1
                
                # Check AI trend (A)
                ai_trend = getattr(t, 'ai_trend', None)
                if ai_trend and ai_trend in ("UP", "DOWN"):
                    trends["ai"]["total"] += 1
                    trend_predicts = "YES" if ai_trend == "UP" else "NO"
                    if trend_predicts == winning_side:
                        trends["ai"]["correct"] += 1
                    else:
                        trends["ai"]["wrong"] += 1
            
            # Calculate accuracy percentages
            for name, data in trends.items():
                if data["total"] > 0:
                    data["accuracy"] = (data["correct"] / data["total"]) * 100.0
            
            # Sort by accuracy (best first)
            ranked = sorted(
                [(name, data) for name, data in trends.items() if data["total"] > 0],
                key=lambda x: x[1]["accuracy"],
                reverse=True
            )
            
            return {
                "per_trend": trends,
                "ranked": ranked,
            }
    
    def get_trend_combinations_accuracy(self) -> Dict[str, Any]:
        """
        Calculate accuracy for combinations of trends.
        
        Analyzes:
        - Single trends (P, S, D, V, K, M)
        - Pairs (P+S, P+D, P+V, P+K, P+M, S+D, S+V, S+K, S+M, D+V, D+K, D+M, V+K, V+M, K+M)
        - Agreement patterns (ALL_X_AGREE, MAJORITY_X)
        
        Returns stats about which combinations have the best win rate.
        """
        with self._lock:
            # Initialize combination counters for all 6 single trends
            combinations = {
                'P': {"wins": 0, "losses": 0, "total": 0},
                'S': {"wins": 0, "losses": 0, "total": 0},
                'D': {"wins": 0, "losses": 0, "total": 0},
                'V': {"wins": 0, "losses": 0, "total": 0},
                'K': {"wins": 0, "losses": 0, "total": 0},
                'M': {"wins": 0, "losses": 0, "total": 0},
                'L': {"wins": 0, "losses": 0, "total": 0},
                'A': {"wins": 0, "losses": 0, "total": 0},
            }
            
            def get_trend_key(trends_dict: Dict[str, str]) -> str:
                """Create a key from active trends, e.g. 'P:UP,V:DOWN'"""
                parts = []
                for k, v in sorted(trends_dict.items()):
                    if v:
                        parts.append(f"{k}:{v}")
                return ",".join(parts) if parts else "none"
            
            def check_prediction(trend_value: str, winning_side: str) -> bool:
                """Check if trend prediction was correct."""
                if trend_value == "UP" and winning_side == "YES":
                    return True
                if trend_value == "DOWN" and winning_side == "NO":
                    return True
                return False
            
            # Process each trade
            for t in self._trades:
                if not t.winning_side or t.winning_side not in ("YES", "NO"):
                    continue
                
                winning_side = t.winning_side
                
                # Get all trends
                p = t.primary_trend if t.primary_trend in ("UP", "DOWN") else None
                s = t.secondary_trend if t.secondary_trend in ("UP", "DOWN") else None
                d = t.macd_trend if t.macd_trend in ("UP", "DOWN") else None  # D = MACD (Divergence)
                v = getattr(t, 'vki_trend', None)
                v = v if v in ("UP", "DOWN") else None
                k = getattr(t, 'keltner_trend', None)  # K = Keltner Channels
                k = k if k in ("UP", "DOWN") else None
                m = getattr(t, 'multi_trend', None)  # M = Multi-indicator
                m = m if m in ("UP", "DOWN") else None
                l = getattr(t, 'lorentzian_trend', None)  # L = Lorentzian Classification
                l = l if l in ("UP", "DOWN") else None
                a = getattr(t, 'ai_trend', None)  # A = AI prediction
                a = a if a in ("UP", "DOWN") else None
                
                # === Analyze single trends ===
                for name, trend in [("P", p), ("S", s), ("D", d), ("V", v), ("K", k), ("M", m), ("L", l), ("A", a)]:
                    if trend:
                        key = name
                        if key not in combinations:
                            combinations[key] = {"wins": 0, "losses": 0, "total": 0}
                        combinations[key]["total"] += 1
                        if check_prediction(trend, winning_side):
                            combinations[key]["wins"] += 1
                        else:
                            combinations[key]["losses"] += 1
                
                # === Analyze agreement patterns ===
                trends_list = [("P", p), ("S", s), ("D", d), ("V", v), ("K", k), ("M", m), ("L", l), ("A", a)]
                active_trends = [(name, val) for name, val in trends_list if val]
                
                if len(active_trends) >= 2:
                    # Count UP and DOWN votes
                    up_count = sum(1 for _, v in active_trends if v == "UP")
                    down_count = sum(1 for _, v in active_trends if v == "DOWN")
                    total_trends = len(active_trends)
                    
                    # All trends agree
                    if up_count == total_trends or down_count == total_trends:
                        key = f"ALL_{total_trends}_AGREE"
                        if key not in combinations:
                            combinations[key] = {"wins": 0, "losses": 0, "total": 0}
                        combinations[key]["total"] += 1
                        
                        consensus = "UP" if up_count == total_trends else "DOWN"
                        if check_prediction(consensus, winning_side):
                            combinations[key]["wins"] += 1
                        else:
                            combinations[key]["losses"] += 1
                    
                    # Majority (at least 2 agree)
                    if up_count >= 2 or down_count >= 2:
                        majority = "UP" if up_count > down_count else "DOWN"
                        key = f"MAJORITY_{total_trends}"
                        if key not in combinations:
                            combinations[key] = {"wins": 0, "losses": 0, "total": 0}
                        combinations[key]["total"] += 1
                        if check_prediction(majority, winning_side):
                            combinations[key]["wins"] += 1
                        else:
                            combinations[key]["losses"] += 1
                
                # === Analyze specific pairs ===
                pairs = [
                    ("P+S", p, s),
                    ("P+D", p, d),
                    ("P+V", p, v),
                    ("P+K", p, k),
                    ("P+M", p, m),
                    ("P+L", p, l),
                    ("S+D", s, d),
                    ("S+V", s, v),
                    ("S+K", s, k),
                    ("S+M", s, m),
                    ("S+L", s, l),
                    ("D+V", d, v),
                    ("D+K", d, k),
                    ("D+M", d, m),
                    ("D+L", d, l),
                    ("V+K", v, k),
                    ("V+M", v, m),
                    ("V+L", v, l),
                    ("K+M", k, m),
                    ("K+L", k, l),
                    ("M+L", m, l),
                    ("P+A", p, a),
                    ("S+A", s, a),
                    ("D+A", d, a),
                    ("V+A", v, a),
                    ("K+A", k, a),
                    ("M+A", m, a),
                    ("L+A", l, a),
                ]
                
                for pair_name, t1, t2 in pairs:
                    if t1 and t2 and t1 == t2:  # Both agree
                        key = f"{pair_name}_AGREE"
                        if key not in combinations:
                            combinations[key] = {"wins": 0, "losses": 0, "total": 0}
                        combinations[key]["total"] += 1
                        if check_prediction(t1, winning_side):
                            combinations[key]["wins"] += 1
                        else:
                            combinations[key]["losses"] += 1
            
            # Calculate accuracy for each combination
            for key, data in combinations.items():
                if data["total"] > 0:
                    data["accuracy"] = round((data["wins"] / data["total"]) * 100.0, 1)
                else:
                    data["accuracy"] = 0.0
            
            # Rank by accuracy (best first), require at least 3 trades
            ranked = sorted(
                [(k, v) for k, v in combinations.items() if v["total"] >= 3],
                key=lambda x: (x[1]["accuracy"], x[1]["total"]),
                reverse=True
            )
            
            # Get best single trend (P, S, D, V, K, M)
            single_trends = [(k, v) for k, v in combinations.items() 
                           if k in ("P", "S", "D", "V", "K", "M", "L", "A") and v["total"] >= 3]
            best_single = max(single_trends, key=lambda x: x[1]["accuracy"]) if single_trends else None
            
            # Get best combination (non-single)
            combo_trends = [(k, v) for k, v in combinations.items() 
                          if k not in ("P", "S", "D", "V", "K", "M", "L", "A") and v["total"] >= 3]
            best_combo = max(combo_trends, key=lambda x: x[1]["accuracy"]) if combo_trends else None
            
            return {
                "all_combinations": combinations,
                "ranked": ranked[:15],  # Top 15
                "best_single": best_single,
                "best_combination": best_combo,
                "total_trades_analyzed": len([t for t in self._trades 
                                             if t.winning_side in ("YES", "NO")]),
            }
    
    def clear_trades(self) -> int:
        """Clear all trades (for testing/reset). Returns number of trades cleared."""
        with self._lock:
            count = len(self._trades)
            self._trades = []
            self._save_trades()
            return count


REGISTRY = BotRegistry()
