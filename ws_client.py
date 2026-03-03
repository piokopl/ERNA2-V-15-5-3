#!/usr/bin/env python3
"""
PolyGrid - WebSocket Client
Real-time orderbook updates from Polymarket.
"""

import json
import time
import logging
import threading
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field
from queue import Queue, Empty

import websocket

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class OrderbookState:
    """Current orderbook state for a token"""
    token_id: str
    bids: List[Dict] = field(default_factory=list)  # [{"price": "0.55", "size": "100"}, ...]
    asks: List[Dict] = field(default_factory=list)
    best_bid: float = 0.0
    best_ask: float = 0.0  # Changed from 1.0 to avoid false triggers
    last_update: float = 0.0


@dataclass 
class Quote:
    """Current market quote"""
    yes_bid: float
    yes_ask: float
    yes_mid: float
    no_bid: float
    no_ask: float
    no_mid: float
    timestamp: float


class JsonRpcIds:
    """Track JSON-RPC IDs"""
    def __init__(self):
        self._id = 0
        self._lock = threading.Lock()
    
    def next(self) -> int:
        with self._lock:
            self._id += 1
            return self._id


class PolymarketWebSocket:
    """WebSocket client for real-time Polymarket data"""
    
    def __init__(self, on_quote: Optional[Callable[[Quote], None]] = None, verbose: bool = False):
        self.on_quote = on_quote
        self.verbose = verbose
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        
        # Set logger level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
        
        self._ids = JsonRpcIds()
        self._subscriptions: Dict[str, int] = {}  # asset_id -> subscription_id
        
        # Current state
        self._orderbooks: Dict[str, OrderbookState] = {}  # token_id -> state
        self._yes_token: Optional[str] = None
        self._no_token: Optional[str] = None
        self._lock = threading.Lock()
        
        # Quote queue for sync access
        self._quote_queue: Queue = Queue(maxsize=100)
        
        # Ping thread removed - using run_forever(ping_interval=30) instead
    
    def start(self):
        """Start WebSocket connection"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WebSocket client started")
    
    def stop(self):
        """Stop WebSocket connection"""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WebSocket client stopped")
    
    def subscribe_market(self, yes_token_id: str, no_token_id: str):
        """Subscribe to market orderbook updates"""
        with self._lock:
            self._yes_token = yes_token_id
            self._no_token = no_token_id
            
            # Initialize orderbook states
            self._orderbooks[yes_token_id] = OrderbookState(token_id=yes_token_id)
            self._orderbooks[no_token_id] = OrderbookState(token_id=no_token_id)
        
        if self._connected and self._ws:
            # If connection is already open but we have not yet sent the initial subscription payload,
            # send it now with both asset ids.
            if not getattr(self, "_initial_subscribed", False):
                try:
                    init_msg = {"assets_ids": [yes_token_id, no_token_id], "type": "market"}
                    logger.debug(f"Sending initial subscription: {json.dumps(init_msg)[:200]}")
                    self._ws.send(json.dumps(init_msg))
                    self._initial_subscribed = True
                except Exception as e:
                    logger.error(f"Initial subscribe error: {e}")
            else:
                self._send_subscribe(yes_token_id)
                self._send_subscribe(no_token_id)
    
    def unsubscribe_market(self):
        """Unsubscribe from current market"""
        with self._lock:
            if self._yes_token and self._yes_token in self._subscriptions:
                self._send_unsubscribe(self._yes_token)
            if self._no_token and self._no_token in self._subscriptions:
                self._send_unsubscribe(self._no_token)
            
            self._yes_token = None
            self._no_token = None
            self._orderbooks.clear()
            self._subscriptions.clear()
    
    def get_quote(self, timeout: float = 0.1) -> Optional[Quote]:
        """Get latest quote (non-blocking)"""
        try:
            return self._quote_queue.get(timeout=timeout)
        except Empty:
            # Return current state if no new updates
            return self._build_quote()
    
    def get_current_quote(self) -> Optional[Quote]:
        """Get current quote from state (no waiting)"""
        return self._build_quote()
    
    def has_valid_data(self) -> bool:
        """Check if we have valid orderbook data (not defaults)"""
        with self._lock:
            if not self._yes_token or not self._no_token:
                return False
            
            yes_book = self._orderbooks.get(self._yes_token)
            no_book = self._orderbooks.get(self._no_token)
            
            if not yes_book or not no_book:
                return False
            
            # Must have at least one bid or ask on each side
            has_yes_data = len(yes_book.bids) > 0 or len(yes_book.asks) > 0
            has_no_data = len(no_book.bids) > 0 or len(no_book.asks) > 0
            
            return has_yes_data and has_no_data
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected
    
    def _run_loop(self):
        """Main WebSocket loop with reconnection"""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            if self._running:
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, 
                    self._max_reconnect_delay
                )
    
    def _connect(self):
        """Establish WebSocket connection"""
        logger.debug(f"Connecting to {WS_URL}")
        
        self._ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self._ws.run_forever(
            ping_interval=30,
            ping_timeout=10
        )
    
    def _on_open(self, ws):
        """Handle connection open"""
        logger.debug("WebSocket connected")
        self._connected = True
        self._reconnect_delay = 1.0

        # Initial subscription payload (per Polymarket WSS docs)
        # For the market channel, send: {"assets_ids": [...], "type": "market"}
        with self._lock:
            assets = [t for t in [self._yes_token, self._no_token] if t]
        if assets:
            try:
                init_msg = {"assets_ids": assets, "type": "market"}
                logger.debug(f"Sending initial subscription: {json.dumps(init_msg)[:200]}")
                ws.send(json.dumps(init_msg))
                self._initial_subscribed = True
            except Exception as e:
                logger.error(f"Initial subscribe error: {e}")
                self._initial_subscribed = False

        # websocket-client's ping is enabled via run_forever(ping_interval=30)
        # Do NOT send custom PING messages - they break the connection!

    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle connection close"""
        logger.debug(f"WebSocket closed: {close_status_code} - {close_msg}")
        self._connected = False
    
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        self._connected = False
    
    def _on_message(self, ws, message):
        """Handle incoming message"""
        logger.debug(f"RAW WS MESSAGE: {str(message)[:200]}")
        # Server may send non-JSON keepalives or plain-text errors.
        if isinstance(message, str) and message in {"PONG", "PING"}:
            return
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            # Keep running; treat as informational unless you want to hard-fail on specific messages
            logger.debug(f"Non-JSON WS message: {str(message)[:200]}")
            return
        except Exception as e:
            logger.error(f"WS message decode error: {e}")
            return

        try:
            self._handle_message(data)
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    def _handle_message(self, data: dict):
        """Process incoming message"""
        # Handle list messages (some events come as arrays)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._handle_message(item)
            return
        
        if not isinstance(data, dict):
            return
        
        event_type = data.get("event_type", "")
        
        # PRIMARY: last_trade_price - this is the actual trade price we use for sniping
        if event_type == "last_trade_price":
            self._handle_last_trade_price(data)
            return
        
        # SECONDARY: book snapshot (initial subscription)
        if event_type == "book":
            self._handle_book_snapshot(data)
            return
        
        # IGNORE price_change.price (it's "indicative price", not real)
        # But we can use best_bid/best_ask from it if available
        if event_type == "price_change" or "price_changes" in data:
            # Only extract best_bid/best_ask, ignore the misleading "price" field
            self._handle_price_change_bid_ask(data)
            return
    
    def _handle_last_trade_price(self, data: dict):
        """Handle last_trade_price event - THE ACTUAL PRICE FOR SNIPING
        
        This is emitted when a trade occurs. The 'price' field is the 
        actual execution price of the trade.
        """
        token_id = data.get("asset_id", "")
        price_str = data.get("price", "")
        
        if not token_id or not price_str:
            return
        
        try:
            price = float(price_str)
        except (ValueError, TypeError):
            return
        
        with self._lock:
            if token_id not in self._orderbooks:
                return
            
            book = self._orderbooks[token_id]
            old_price = book.best_bid
            book.last_update = time.time()
            book.best_bid = price
            book.best_ask = price
            
            side = "YES" if token_id == self._yes_token else "NO"
            
            # Log price changes at DEBUG level to reduce spam
            logger.debug(f"💰 {side}: {price:.4f}")
        
        self._publish_quote()
    
    def _handle_price_change_bid_ask(self, data: dict):
        """Extract best_bid/best_ask from price_change events.
        
        NOTE: The 'price' field in price_change is "indicative price" 
        (not real market price) - we IGNORE it!
        Only best_bid/best_ask are useful from this message.
        """
        changes = data.get("price_changes", [])
        if not changes and "asset_id" in data:
            changes = [data]
        
        for change in changes:
            token_id = change.get("asset_id", "")
            if not token_id:
                continue
            
            # Only use best_bid/best_ask if present
            best_bid_str = change.get("best_bid")
            best_ask_str = change.get("best_ask")
            
            if not best_bid_str and not best_ask_str:
                continue  # No useful data
            
            with self._lock:
                if token_id not in self._orderbooks:
                    continue
                
                book = self._orderbooks[token_id]
                
                # Only update if we don't have recent last_trade_price data
                # (last_trade_price is more accurate for sniping)
                time_since_update = time.time() - book.last_update
                if time_since_update < 1.0:
                    continue  # Skip, we have fresh last_trade_price data
                
                if best_bid_str:
                    try:
                        book.best_bid = float(best_bid_str)
                    except (ValueError, TypeError):
                        pass
                
                if best_ask_str:
                    try:
                        book.best_ask = float(best_ask_str)
                    except (ValueError, TypeError):
                        pass
                
                book.last_update = time.time()
        
        self._publish_quote()
    
    def _handle_book_snapshot(self, data: dict):
        """Handle full orderbook snapshot (event_type: book)
        
        Format:
        {
          "event_type": "book",
          "asset_id": "...",
          "bids": [{"price": ".48", "size": "30"}, ...],
          "asks": [{"price": ".52", "size": "25"}, ...],
          "timestamp": "..."
        }
        """
        token_id = data.get("asset_id", "")
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        if not token_id:
            return
        
        with self._lock:
            if token_id not in self._orderbooks:
                return
            
            book = self._orderbooks[token_id]
            book.bids = bids
            book.asks = asks
            book.last_update = time.time()
            
            # Find best bid (highest price)
            if bids:
                best_bid = max(float(b.get("price", 0)) for b in bids)
                book.best_bid = best_bid
            else:
                book.best_bid = 0.0
            
            # Find best ask (lowest price)
            if asks:
                best_ask = min(float(a.get("price", 1)) for a in asks)
                book.best_ask = best_ask
            else:
                book.best_ask = 1.0
            
            side = "YES" if token_id == self._yes_token else "NO"
            logger.debug(f"📊 {side} book: {len(bids)} bids (best={book.best_bid:.4f}), {len(asks)} asks")
        
        self._publish_quote()
    
    def _handle_book_delta(self, data: dict):
        """Handle orderbook delta update"""
        assets = data.get("assets", [])
        
        for asset in assets:
            token_id = asset.get("asset_id", "")
            
            with self._lock:
                if token_id not in self._orderbooks:
                    continue
                
                book = self._orderbooks[token_id]
                
                # Apply bid changes
                for change in asset.get("bids", []):
                    self._apply_delta(book.bids, change, is_bid=True)
                
                # Apply ask changes
                for change in asset.get("asks", []):
                    self._apply_delta(book.asks, change, is_bid=False)
                
                book.last_update = time.time()
                
                # Best bid = last (highest price) bid
                # Best ask = last (lowest price) ask
                if book.bids:
                    book.best_bid = float(book.bids[-1].get("price", 0))
                else:
                    book.best_bid = 0.0
                
                if book.asks:
                    book.best_ask = float(book.asks[-1].get("price", 1))
                else:
                    book.best_ask = 1.0
        
        self._publish_quote()
    
    def _apply_delta(self, levels: List[Dict], change: Dict, is_bid: bool):
        """Apply delta change to orderbook level"""
        price = change.get("price", "")
        size = float(change.get("size", 0))
        
        # Find existing level
        for i, level in enumerate(levels):
            if level.get("price") == price:
                if size == 0:
                    levels.pop(i)
                else:
                    level["size"] = str(size)
                return
        
        # Add new level if size > 0
        if size > 0:
            levels.append({"price": price, "size": str(size)})
            # Sort: bids ascending (worst to best), asks descending (worst to best)
            # Best = last element in both cases
            if is_bid:
                # Bids: ascending order (0.01, 0.02, ..., best)
                levels.sort(key=lambda x: float(x.get("price", 0)))
            else:
                # Asks: descending order (0.99, 0.98, ..., best)
                levels.sort(key=lambda x: float(x.get("price", 0)), reverse=True)
    
    def _handle_price_changes(self, data: dict):
        """Handle price_changes message format from Polymarket WebSocket
        
        NOTE: price_changes gives the "indicative price" or last trade price,
        NOT the actual best bid/ask from orderbook. We only use this as a fallback
        when we don't have orderbook data.
        """
        price_changes = data.get("price_changes", [])
        
        for change in price_changes:
            token_id = change.get("asset_id", "")
            price_str = change.get("price", "")
            
            if not token_id or not price_str:
                continue
            
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                continue
            
            with self._lock:
                if token_id not in self._orderbooks:
                    continue
                
                book = self._orderbooks[token_id]
                book.last_update = time.time()
                
                # Only use price_changes as fallback if we have NO orderbook data
                # The real best_bid should come from actual orderbook bids
                # price_changes is indicative/last trade price, not the bid
                if len(book.bids) == 0 and len(book.asks) == 0:
                    # No orderbook data yet, use price as approximation
                    book.best_bid = price
                    book.best_ask = price
                    logger.debug(f"Price update (no orderbook) {token_id[:16]}...: {price:.4f}")
        
        # Don't publish quote from price_changes - wait for orderbook data
        # self._publish_quote()
    
    def _handle_direct_update(self, data: dict):
        """Handle direct orderbook update (asset_id with bids/asks or price)"""
        token_id = data.get("asset_id", "")
        
        if not token_id:
            return
        
        with self._lock:
            if token_id not in self._orderbooks:
                return
            
            book = self._orderbooks[token_id]
            book.last_update = time.time()
            
            # Handle bids - Polymarket sends bids sorted by price ascending or descending
            # Best bid = HIGHEST price bid
            if "bids" in data:
                bids = data.get("bids", [])
                book.bids = bids
                if bids:
                    # Log first few bids to see what we're getting
                    sample_bids = [f"{b.get('price')}@{b.get('size', '?')[:6]}" for b in bids[:3]]
                    logger.debug(f"Bids sample for {token_id[:16]}...: {sample_bids}")
                    
                    # Find highest bid price
                    best_bid_price = 0.0
                    for bid in bids:
                        try:
                            p = float(bid.get("price", 0))
                            if p > best_bid_price:
                                best_bid_price = p
                        except (ValueError, TypeError):
                            continue
                    book.best_bid = best_bid_price
                    logger.info(f"📊 {token_id[:16]}...: {len(bids)} bids, BEST BID={best_bid_price:.4f}")
                else:
                    book.best_bid = 0.0
            
            # Handle asks - Best ask = LOWEST price ask
            if "asks" in data:
                asks = data.get("asks", [])
                book.asks = asks
                if asks:
                    # Find lowest ask price
                    best_ask_price = 1.0
                    for ask in asks:
                        try:
                            p = float(ask.get("price", 1))
                            if p < best_ask_price:
                                best_ask_price = p
                        except (ValueError, TypeError):
                            continue
                    book.best_ask = best_ask_price
                else:
                    book.best_ask = 1.0
            
            # Handle direct price update (single trade or quote)
            if "price" in data and "bids" not in data and "asks" not in data:
                try:
                    price = float(data.get("price", 0))
                    # This is a trade price, only use if we have no orderbook
                    if len(book.bids) == 0:
                        book.best_bid = price
                    if len(book.asks) == 0:
                        book.best_ask = price
                except (ValueError, TypeError):
                    pass
        
        self._publish_quote()
    
    def _build_quote(self) -> Optional[Quote]:
        """Build quote from current state"""
        with self._lock:
            if not self._yes_token or not self._no_token:
                return None
            
            yes_book = self._orderbooks.get(self._yes_token)
            no_book = self._orderbooks.get(self._no_token)
            
            if not yes_book or not no_book:
                return None
            
            yes_bid = yes_book.best_bid
            yes_ask = yes_book.best_ask
            no_bid = no_book.best_bid
            no_ask = no_book.best_ask
            
            # Debug: log raw orderbook prices
            logger.debug(f"Raw YES: bid={yes_bid:.4f} ask={yes_ask:.4f} (bids={len(yes_book.bids)} asks={len(yes_book.asks)})")
            logger.debug(f"Raw NO:  bid={no_bid:.4f} ask={no_ask:.4f} (bids={len(no_book.bids)} asks={len(no_book.asks)})")
            
            # Derive from opposite side if needed
            if yes_bid == 0 and no_ask < 1:
                yes_bid = max(0, 1 - no_ask)
            if yes_ask == 1 and no_bid > 0:
                yes_ask = min(1, 1 - no_bid)
            if no_bid == 0 and yes_ask < 1:
                no_bid = max(0, 1 - yes_ask)
            if no_ask == 1 and yes_bid > 0:
                no_ask = min(1, 1 - yes_bid)
            
            logger.debug(f"Derived YES: bid={yes_bid:.4f} ask={yes_ask:.4f}")
            logger.debug(f"Derived NO:  bid={no_bid:.4f} ask={no_ask:.4f}")
            
            return Quote(
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                yes_mid=(yes_bid + yes_ask) / 2,
                no_bid=no_bid,
                no_ask=no_ask,
                no_mid=(no_bid + no_ask) / 2,
                timestamp=max(yes_book.last_update, no_book.last_update)
            )
    
    def _publish_quote(self):
        """Publish quote to queue and callback"""
        quote = self._build_quote()
        if not quote:
            return
        
        # Add to queue (drop oldest if full)
        try:
            self._quote_queue.put_nowait(quote)
        except:
            try:
                self._quote_queue.get_nowait()
                self._quote_queue.put_nowait(quote)
            except:
                pass
        
        # Callback
        if self.on_quote:
            try:
                self.on_quote(quote)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")
    
    def _send_subscribe(self, token_id: str):
        """Subscribe to additional asset ids after the initial payload."""
        logger.debug(f"_send_subscribe called: connected={self._connected}, ws={self._ws is not None}")
        if not self._ws or not self._connected:
            logger.warning("Cannot subscribe - not connected!")
            return

        # Per WSS Overview: once connected, subscribe/unsubscribe via {"assets_ids": [...], "operation": "subscribe"}
        msg = {
            "assets_ids": [token_id],
            "operation": "subscribe",
        }

        try:
            logger.debug(f"Sending subscription: {json.dumps(msg)[:200]}")
            self._ws.send(json.dumps(msg))
            self._subscriptions[token_id] = "subscribe"
            logger.debug(f"Subscribed to {token_id[:16]}...")
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
    
    def _send_unsubscribe(self, token_id: str):
        """Unsubscribe asset id (market channel)."""
        if not self._ws or not self._connected:
            return

        msg = {
            "assets_ids": [token_id],
            "operation": "unsubscribe",
        }

        try:
            logger.info(f"Sending unsubscribe: {json.dumps(msg)[:200]}")
            self._ws.send(json.dumps(msg))
            self._subscriptions.pop(token_id, None)
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")


# Convenience function for testing
def test_ws():
    """Test WebSocket connection"""
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    def on_quote(quote: Quote):
        print(f"Quote: YES {quote.yes_bid:.4f}/{quote.yes_ask:.4f} | NO {quote.no_bid:.4f}/{quote.no_ask:.4f}")
    
    ws = PolymarketWebSocket(on_quote=on_quote)
    ws.start()
    
    # Example token IDs (replace with real ones)
    if len(sys.argv) >= 3:
        yes_token = sys.argv[1]
        no_token = sys.argv[2]
        ws.subscribe_market(yes_token, no_token)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.stop()


if __name__ == "__main__":
    test_ws()
