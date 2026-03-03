#!/usr/bin/env python3
"""
PolySniper - Polymarket Client
Simplified client for 15-minute BTC markets with order management.
"""

import os
import time
import json
import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN

import httpx

logger = logging.getLogger(__name__)



def _mask_secret(value: str, keep: int = 4) -> str:
    """Mask secrets for logs (keep first/last N chars)."""
    if not value:
        return ""
    v = str(value)
    if len(v) <= keep * 2:
        return "*" * len(v)
    return f"{v[:keep]}...{v[-keep:]}"

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


class CloudflareBlockError(Exception):
    """Raised when Cloudflare WAF blocks our requests (403)."""
    pass


class InsufficientBalanceError(Exception):
    """Raised when account has insufficient balance/allowance to place order."""
    pass


@dataclass
class MarketInfo:
    """Market information"""
    market_id: str
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    start_ts: float
    end_ts: float
    neg_risk: bool = True  # Most Polymarket markets use NegRiskCTF


@dataclass 
class OrderStatus:
    """Order status"""
    order_id: str
    status: str  # "LIVE", "MATCHED", "CANCELLED"
    filled_size: float
    remaining_size: float
    avg_fill_price: float = 0.0  # Actual average fill price


@dataclass
class ExecutionInfo:
    """Executed (filled) information for an order.

    Notes
    -----
    For accuracy we compute avg execution price from the matched maker legs
    (maker_orders) when available. This avoids incorrectly using the taker
    limit price (which may be far from the actual fill).
    """

    order_id: str
    filled_size: float
    avg_price: float
    fee_paid: float
    status: str


class PolymarketClient:
    """Polymarket API client with order management"""
    
    def __init__(
        self,
        private_key: str = "",
        funder_address: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        is_demo: bool = True
    ):
        self.private_key = private_key
        if self.private_key and not self.private_key.startswith("0x"):
            self.private_key = "0x" + self.private_key
        
        self.funder_address = funder_address
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.is_demo = is_demo
        
        self._session = httpx.Client(timeout=30.0)
        self._client = None
        
        # Demo mode order tracking
        self._demo_orders = {}  # order_id -> {side, size, price, status, filled_size}
        
        # Last error for debugging
        self.last_order_error: Optional[str] = None
        self.last_order_response: Optional[dict] = None
        self.last_status_error: Optional[str] = None
        self.last_status_response: Optional[str] = None
    
    def initialize(self) -> bool:
        """Initialize the client"""
        if self.is_demo:
            logger.info("Polymarket client initialized [DEMO]")
            return True
        
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            
            host = "https://clob.polymarket.com"
            chain_id = 137
            
            # Log config for debugging
            logger.debug(f"Private key: {self.private_key[:10]}...{self.private_key[-6:]}")
            logger.debug(f"Funder (proxy address): {self.funder_address}")
            
            # Signature type 2 = POLY_GNOSIS_SAFE (for proxy wallets)
            # When using proxy wallet: key=proxy_pk, funder=proxy_address
            signature_type = 2  # POLY_GNOSIS_SAFE
            logger.info(f"Using signature type 2 (POLY_GNOSIS_SAFE)")
            
            # First try with provided API creds
            if self.api_key and self.api_secret and self.api_passphrase:
                try:
                    creds = ApiCreds(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        api_passphrase=self.api_passphrase
                    )
                    
                    self._client = ClobClient(
                        host=host,
                        chain_id=chain_id,
                        key=self.private_key,
                        creds=creds,
                        signature_type=signature_type,
                        funder=self.funder_address
                    )
                    
                    # Verify by getting open orders
                    if self._verify_client():
                        logger.info("Polymarket client initialized with provided creds [LIVE]")
                        return True
                    else:
                        logger.warning("Provided creds verification failed, will derive new ones")
                except Exception as e:
                    logger.warning(f"Failed with provided creds: {e}")
            
            # Derive fresh credentials
            self._client = ClobClient(
                host=host,
                chain_id=chain_id,
                key=self.private_key,
                signature_type=signature_type,
                funder=self.funder_address
            )
            
            # Derive and set API credentials
            derived_creds = self._client.derive_api_key()
            self._client.set_api_creds(derived_creds)
            
            # Log derived creds so user can save them
            logger.info("Derived new API credentials (values hidden in logs).")
            logger.info(f"  API_KEY={_mask_secret(derived_creds.api_key)}")
            logger.info(f"  API_SECRET={_mask_secret(derived_creds.api_secret)}")
            logger.info(f"  API_PASSPHRASE={_mask_secret(derived_creds.api_passphrase)}")
            logger.info("Save the full values to .env to avoid re-deriving.")
            
            # Verify
            if not self._verify_client():
                logger.error("Client verification failed even after deriving new creds!")
                return False
            
            logger.info("Polymarket client initialized [LIVE]")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _verify_client(self) -> bool:
        """Verify client can communicate with API"""
        try:
            # Try to get open orders (empty is fine, just checking auth works)
            orders = self._client.get_orders()
            logger.debug(f"Verification: got {len(orders) if orders else 0} open orders")
            return True
        except Exception as e:
            logger.debug(f"Client verification failed: {e}")
            return False
    
    def find_15min_market(self, asset: str = "BTC", target_start_ts: Optional[int] = None) -> Optional[MarketInfo]:
        """Find 15-minute market
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            target_start_ts: If provided, find market with this specific START timestamp.
                            Slug format: {asset}-updown-15m-{START_TIMESTAMP}
                            If None, find the market ending soonest.
        
        For sniping, we want the market that is:
        1. Still active (end_ts > now)
        2. Ending SOONEST (closest to resolution) - or matching target_start_ts
        """
        try:
            now = time.time()
            
            # Calculate current 15-min window boundaries
            # Windows are aligned to 15-min intervals (900 seconds)
            window_size = 900
            
            # If specific target_start_ts provided, try to find that exact market
            if target_start_ts:
                prefix = (asset or "BTC").strip().lower()
                slug = f"{prefix}-updown-15m-{int(target_start_ts)}"
                
                logger.debug(f"Looking for specific market: {slug}")
                
                # expected_end_ts = start + 900
                market = self._fetch_market_by_slug(slug, target_start_ts + window_size)
                if market:
                    logger.info(f"[15min] Found target market: {market.question}")
                    return market
                else:
                    logger.warning(f"Target market not found: {slug}")
                    return None
            
            # Otherwise, find the market ending soonest
            # IMPORTANT: Polymarket slug uses WINDOW START timestamp, not end!
            # Slug format: btc-updown-15m-{START_TIMESTAMP}
            # e.g. btc-updown-15m-1769679900 = window 09:45-10:00 UTC
            window_size = 900
            
            # Calculate current window START (what we're IN right now)
            current_window_start = (int(now) // window_size) * window_size
            
            from datetime import datetime, timezone
            now_utc = datetime.fromtimestamp(now, tz=timezone.utc).strftime('%H:%M:%S')
            start_utc = datetime.fromtimestamp(current_window_start, tz=timezone.utc).strftime('%H:%M:%S')
            logger.info(f"[15min] Now: {int(now)} ({now_utc} UTC), window_start={current_window_start} ({start_utc} UTC)")
            
            # Try: current window first, then previous (might still be settling), then next
            candidates = []
            
            for start_ts in [current_window_start, current_window_start - window_size, current_window_start + window_size]:
                prefix = (asset or "BTC").strip().lower()
                slug = f"{prefix}-updown-15m-{int(start_ts)}"
                
                market = self._fetch_market_by_slug(slug, start_ts + window_size)  # end_ts = start + 900
                if market:
                    time_left = market.end_ts - now
                    logger.info(f"[15min] ✓ {slug}: time_left={time_left:.0f}s")
                    
                    # Only consider markets that haven't ended yet
                    if market.end_ts > now:
                        candidates.append((time_left, market))
                    else:
                        logger.info(f"[15min] ✗ {slug}: already ended")
                else:
                    logger.debug(f"[15min] ✗ {slug}: not found")
            
            if not candidates:
                logger.debug("No active 15min markets found")
                return None
            
            # Sort by time_left (ascending) - pick the one ending soonest
            candidates.sort(key=lambda x: x[0])
            best_market = candidates[0][1]
            time_left = candidates[0][0]
            
            logger.info(f"[15min] Selected market ending in {time_left:.0f}s: {best_market.question}")
            return best_market
            
        except Exception as e:
            logger.error(f"Error finding 15min market: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_1h_market(self, asset: str = "BTC") -> Optional[MarketInfo]:
        """Find current active 1-hour market
        
        Slug format: bitcoin-up-or-down-{month}-{day}-{hour}{am/pm}-et
        e.g., bitcoin-up-or-down-january-9-8am-et
        """
        try:
            import pytz
            
            now = time.time()
            et_tz = pytz.timezone('America/New_York')
            now_et = datetime.now(et_tz)
            
            candidates = []
            
            # Try current hour and next 2 hours
            for hour_offset in range(3):
                # Calculate target hour
                target_dt = now_et + timedelta(hours=hour_offset)
                
                # Generate slug
                month_name = target_dt.strftime("%B").lower()  # january, february, etc.
                day = target_dt.day
                hour = target_dt.hour
                
                # Convert to 12-hour format
                if hour == 0:
                    hour_str = "12am"
                elif hour < 12:
                    hour_str = f"{hour}am"
                elif hour == 12:
                    hour_str = "12pm"
                else:
                    hour_str = f"{hour - 12}pm"
                
                slug = f"bitcoin-up-or-down-{month_name}-{day}-{hour_str}-et"
                
                logger.debug(f"[1h] Trying slug: {slug}")
                
                market = self._fetch_market_by_slug(slug)
                if market:
                    time_left = market.end_ts - now
                    logger.debug(f"[1h] Found market: {market.question}, time_left={time_left:.0f}s")
                    
                    # Only consider markets that haven't ended yet
                    if market.end_ts > now:
                        candidates.append((time_left, market))
            
            if not candidates:
                logger.debug("No active 1h markets found")
                return None
            
            # Sort by time_left (ascending) - pick the one ending soonest
            candidates.sort(key=lambda x: x[0])
            best_market = candidates[0][1]
            time_left = candidates[0][0]
            
            logger.info(f"[1h] Selected market ending in {time_left:.0f}s: {best_market.question}")
            return best_market
            
        except Exception as e:
            logger.error(f"Error finding 1h market: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _fetch_market_by_slug(self, slug: str, expected_end_ts: float = 0) -> Optional[MarketInfo]:
        """Fetch market by slug
        
        Args:
            slug: Market slug e.g. btc-updown-15m-1769679900 (uses window START timestamp)
            expected_end_ts: Expected end timestamp (start + 900)
        """
        try:
            resp = self._session.get(
                f"{GAMMA_API}/events",
                params={"slug": slug}
            )
            
            if resp.status_code != 200:
                logger.debug(f"API returned {resp.status_code} for {slug}")
                return None
            
            events = resp.json()
            if not events:
                return None
            
            event = events[0]
            markets = event.get("markets", [])
            
            if not markets:
                return None
            
            market = markets[0]
            
            # Parse end timestamp from API
            end_str = event.get("endDate") or market.get("endDate")
            if end_str:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                end_ts = end_dt.timestamp()
            else:
                # Fallback: slug timestamp is START, so end = start + 900
                # Extract start_ts from slug
                try:
                    slug_ts = int(slug.split("-")[-1])
                    end_ts = slug_ts + 900
                except:
                    end_ts = time.time() + 900
            
            start_ts = end_ts - 900
            
            # Get token IDs - may be JSON string or list
            tokens = market.get("clobTokenIds", [])
            if isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens)
                except:
                    tokens = []
            
            if len(tokens) < 2:
                tokens = [market.get("clobTokenId", ""), ""]
            
            # Detect neg_risk model (NegRiskCTF exchange)
            # Most Polymarket markets use neg_risk=True
            neg_risk = bool(market.get("negRisk", event.get("negRisk", True)))
            logger.info(f"Market {slug}: neg_risk={neg_risk}, YES={tokens[0][:16]}... NO={tokens[1][:16]}...")
            
            return MarketInfo(
                market_id=market.get("id", ""),
                condition_id=market.get("conditionId", ""),
                question=market.get("question", event.get("title", "")),
                yes_token_id=tokens[0],
                no_token_id=tokens[1],
                start_ts=start_ts,
                end_ts=end_ts,
                neg_risk=neg_risk
            )
            
        except Exception as e:
            logger.debug(f"Market fetch error for {slug}: {e}")
            return None
    
    def get_market_outcome(self, slug: str) -> Optional[str]:
        """Get the resolved outcome of a market (YES or NO).
        
        After market closes and resolves:
        - Winning token price = $1.00
        - Losing token price = $0.00
        
        Args:
            slug: Market slug (e.g. 'btc-updown-15m-1769682600')
        
        Returns:
            'YES' if YES won, 'NO' if NO won, None if not resolved yet or error
        """
        try:
            resp = self._session.get(
                f"{GAMMA_API}/events",
                params={"slug": slug}
            )
            
            if resp.status_code != 200:
                logger.warning(f"get_market_outcome: API returned {resp.status_code} for {slug}")
                return None
            
            events = resp.json()
            if not events:
                logger.warning(f"get_market_outcome: No events found for {slug}")
                return None
            
            event = events[0]
            markets = event.get("markets", [])
            
            if not markets:
                logger.warning(f"get_market_outcome: No markets in event for {slug}")
                return None
            
            market = markets[0]
            
            # Log all relevant fields for debugging
            logger.info(f"get_market_outcome: Checking {slug}")
            logger.info(f"  closed={market.get('closed')}, resolved={market.get('resolved')}")
            logger.info(f"  acceptingOrders={market.get('acceptingOrders')}")
            
            # Method 1: Check outcomePrices - most reliable after resolution
            # After resolution: winning token = "1", losing token = "0"
            outcome_prices = market.get("outcomePrices")
            if outcome_prices:
                logger.info(f"  outcomePrices={outcome_prices}")
                try:
                    if isinstance(outcome_prices, str):
                        outcome_prices = json.loads(outcome_prices)
                    
                    if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                        yes_price = float(outcome_prices[0])
                        no_price = float(outcome_prices[1])
                        logger.info(f"  Parsed: YES=${yes_price}, NO=${no_price}")
                        
                        # After resolution, one should be ~1.0 and other ~0.0
                        if yes_price >= 0.95:
                            logger.info(f"  → YES won (price={yes_price})")
                            return "YES"
                        elif no_price >= 0.95:
                            logger.info(f"  → NO won (price={no_price})")
                            return "NO"
                except Exception as e:
                    logger.debug(f"  outcomePrices parse error: {e}")
            
            # Method 2: Check resolutionPrice directly
            resolution_price = market.get("resolutionPrice")
            if resolution_price is not None:
                logger.info(f"  resolutionPrice={resolution_price}")
                try:
                    price = float(resolution_price)
                    if price >= 0.5:
                        logger.info(f"  → YES won (resolutionPrice={price})")
                        return "YES"
                    else:
                        logger.info(f"  → NO won (resolutionPrice={price})")
                        return "NO"
                except (ValueError, TypeError):
                    pass
            
            # Method 3: Check if market is closed but prices aren't 0/1 yet
            # This can happen if we check too early
            closed = market.get("closed")
            accepting_orders = market.get("acceptingOrders")
            
            if closed and not accepting_orders:
                logger.info(f"  Market is closed but no clear winner yet - might need more time")
            
            # Method 4: outcome field (sometimes used)
            outcome = market.get("outcome")
            if outcome:
                logger.info(f"  outcome={outcome}")
                outcome_upper = str(outcome).upper()
                if "YES" in outcome_upper or outcome_upper == "1":
                    return "YES"
                elif "NO" in outcome_upper or outcome_upper == "0":
                    return "NO"
            
            # Not resolved yet
            logger.info(f"  Market not yet resolved, will retry")
            return None
            
        except Exception as e:
            logger.error(f"get_market_outcome error for {slug}: {e}")
            return None
    
    def place_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        neg_risk: bool = True
    ) -> Optional[OrderStatus]:
        """Place limit order, returns OrderStatus with fill info"""
        # Reset error tracking
        self.last_order_error = None
        self.last_order_response = None
        
        if self.is_demo:
            order_id = f"DEMO_{int(time.time()*1000)}"
            # In demo, assume immediate fill at *current* marketable price.
            # This avoids using a placeholder limit like 0.01, which would
            # otherwise corrupt PnL accounting.
            fill_price = self.get_marketable_price(token_id=token_id, side=side) or price
            self._demo_orders[order_id] = {
                "side": side,
                "size": size,
                "price": price,
                "fill_price": float(fill_price),
                "status": "MATCHED",  # In demo we fill immediately
                "filled_size": size,
                "created_at": time.time()
            }
            logger.info(f"[DEMO] 📝 {side} {size} @ {float(fill_price):.4f} (req {price:.4f}) -> {order_id}")
            return OrderStatus(
                order_id=order_id,
                status="MATCHED",
                filled_size=size,
                remaining_size=0.0,
                avg_fill_price=float(fill_price)
            )
        
        if not self._client:
            self.last_order_error = "Client not initialized"
            logger.error("Client not initialized")
            return None
        
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            
            # IMPORTANT:
            # - `side` here means order ACTION (BUY/SELL), not market outcome (YES/NO).
            # - Outcome is encoded by `token_id`.
            side_u = (side or "").upper()
            if side_u in {"YES", "NO"}:
                # Backward-compat: older runner code passed outcome (YES/NO) as `side`.
                # That mistakenly turned every "NO" into SELL. Treat outcome as BUY.
                logger.warning(
                    "place_order got side=%s (outcome). Treating as BUY. "
                    "Fix caller to pass side='BUY' and select outcome via token_id.",
                    side_u,
                )
                side_u = "BUY"

            if side_u == "BUY":
                order_side = BUY
            elif side_u == "SELL":
                order_side = SELL
            else:
                raise ValueError(f"Invalid order action: {side}. Expected BUY or SELL")
            
            # Polymarket requires tick size 0.01 for price, 0.01 for size
            size_dec = Decimal(str(size)).quantize(Decimal('0.01'), rounding=ROUND_DOWN)
            price_dec = Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_DOWN)
            
            # Ensure minimum size
            if size_dec < Decimal('0.01'):
                logger.error(f"Size too small: {size_dec}")
                return None
            
            logger.info(f"Creating order: {side} {float(size_dec)} @ {float(price_dec)} token={token_id[:20]}...")
            
            order_args = OrderArgs(
                price=float(price_dec),
                size=float(size_dec),
                side=order_side,
                token_id=token_id
            )
            logger.info(f"📋 OrderArgs: price={float(price_dec)}, size={float(size_dec)}, side={order_side}, token={token_id[:30]}...")
            
            # Try create_and_post_order first (simpler, handles signing internally)
            try:
                logger.info(f"📤 Trying create_and_post_order: {side} {float(size_dec)} @ {float(price_dec)} neg_risk={neg_risk}")
                try:
                    result = self._client.create_and_post_order(order_args, neg_risk=neg_risk)
                except TypeError:
                    logger.info("📋 neg_risk not supported by client, retrying without it")
                    result = self._client.create_and_post_order(order_args)
                self.last_order_response = result if isinstance(result, dict) else {"raw": str(result)}
                logger.info(f"📥 create_and_post_order response: {result}")
                
                if result and isinstance(result, dict):
                    order_id = result.get("orderID") or result.get("order_id") or result.get("id")
                    if order_id:
                        # Extract fill info from response
                        status = result.get("status", "LIVE")
                        size_matched = float(result.get("size_matched", 0) or 0)
                        original_size = float(result.get("original_size", size) or size)
                        avg_price = float(result.get("average_price", 0) or result.get("avg_price", 0) or 0)
                        
                        logger.info(f"✅ Order placed: {side} {size} @ {price:.4f} -> {order_id}")
                        logger.info(f"   Status: {status}, filled: {size_matched}/{original_size}, avg_price: ${avg_price:.4f}")
                        
                        # Return OrderStatus with initial fill info
                        return OrderStatus(
                            order_id=order_id,
                            status=status,
                            filled_size=size_matched,
                            remaining_size=original_size - size_matched,
                            avg_fill_price=avg_price if avg_price > 0 else price
                        )
                    if "error" in result:
                        self.last_order_error = f"API error: {result['error']}"
                        logger.error(f"❌ API returned error: {result['error']}")
                        raise Exception(result["error"])
                    # No order_id and no error - log full response
                    self.last_order_error = f"No order_id in response: {result}"
                    logger.warning(f"⚠️ No order_id in response: {result}")
                elif result is None:
                    self.last_order_error = "create_and_post_order returned None"
                    logger.warning(f"⚠️ create_and_post_order returned None")
                else:
                    self.last_order_error = f"Unexpected response type {type(result)}: {result}"
                    logger.warning(f"⚠️ Unexpected response type {type(result)}: {result}")
                        
            except AttributeError:
                # create_and_post_order not available, use two-step approach
                logger.info("create_and_post_order not available, using two-step")
            except Exception as e:
                # If we are out of funds / allowance, do not spam retries with two-step.
                msg = str(e)
                if "not enough balance" in msg.lower() or "allowance" in msg.lower():
                    logger.error(f"❌ create_and_post_order failed (balance/allowance): {e}")
                    raise InsufficientBalanceError(str(e)) from e
                # Detect Cloudflare WAF block (403 with HTML block page)
                if "403" in msg and ("cloudflare" in msg.lower() or "you have been blocked" in msg.lower() or "Attention Required" in msg):
                    logger.error(f"🛑 Cloudflare WAF is blocking our IP! Orders cannot be placed.")
                    raise CloudflareBlockError("IP blocked by Cloudflare WAF on polymarket.com") from e
                logger.warning(f"⚠️ create_and_post_order failed: {e}, trying two-step approach")
                import traceback
                logger.info(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: two-step create + post
            logger.info(f"📤 Trying two-step approach: create_order + post_order")
            try:
                signed = self._client.create_order(order_args, neg_risk=neg_risk)
            except TypeError:
                logger.info("📋 neg_risk not supported by client (two-step), retrying without it")
                signed = self._client.create_order(order_args)
            logger.info(f"📝 Signed order created: {type(signed)}")
            
            result = self._client.post_order(signed, OrderType.GTC)
            self.last_order_response = result if isinstance(result, dict) else {"raw": str(result)}
            logger.info(f"📥 post_order response: {result}")
            
            order_id = result.get("orderID") or result.get("order_id") or result.get("id")
            if order_id:
                # Extract fill info from response
                status = result.get("status", "LIVE")
                size_matched = float(result.get("size_matched", 0) or 0)
                original_size = float(result.get("original_size", size) or size)
                avg_price = float(result.get("average_price", 0) or result.get("avg_price", 0) or 0)
                
                logger.info(f"✅ Order placed (two-step): {side} {size} @ {price:.4f} -> {order_id}")
                logger.info(f"   Status: {status}, filled: {size_matched}/{original_size}, avg_price: ${avg_price:.4f}")
                
                return OrderStatus(
                    order_id=order_id,
                    status=status,
                    filled_size=size_matched,
                    remaining_size=original_size - size_matched,
                    avg_fill_price=avg_price if avg_price > 0 else price
                )
            
            # Check for error in result
            if "error" in result:
                self.last_order_error = f"API error (two-step): {result['error']}"
                logger.error(f"❌ Order error from API: {result['error']}")
            else:
                self.last_order_error = f"Order failed (two-step), no order_id. Response: {result}"
                logger.error(f"❌ Order failed, full response: {result}")
            return None
            
        except CloudflareBlockError:
            self.last_order_error = "Cloudflare WAF block"
            raise  # Let circuit breaker in runner handle this
        except Exception as e:
            self.last_order_error = f"Exception: {e}"
            logger.error(f"❌ Order exception: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        if self.is_demo:
            # Pobierz zapisane zlecenie demo
            order_data = self._demo_orders.get(order_id)
            if order_data:
                return OrderStatus(
                    order_id=order_id,
                    status=order_data["status"],
                    filled_size=order_data["filled_size"],
                    remaining_size=order_data["size"] - order_data["filled_size"],
                    avg_fill_price=order_data.get("fill_price", order_data.get("price", 0.0))
                )
            else:
                # Fallback for legacy orders
                return OrderStatus(
                    order_id=order_id,
                    status="MATCHED",
                    filled_size=0.0,
                    remaining_size=0.0,
                    avg_fill_price=0.0
                )
        
        if not self._client:
            self.last_status_error = "Client not initialized"
            return None
        
        # Reset status tracking
        self.last_status_error = None
        self.last_status_response = None
        
        try:
            logger.info(f"📋 Calling get_order for: {order_id}")
            result = self._client.get_order(order_id)
            self.last_status_response = str(result)[:500] if result else "None"
            logger.info(f"📋 get_order raw response type={type(result)}: {str(result)[:300]}")
            
            # Handle case where API returns string instead of dict
            if isinstance(result, str):
                # Try to parse as JSON
                try:
                    result = json.loads(result)
                    logger.info(f"📋 Parsed string response as JSON: {result}")
                except json.JSONDecodeError:
                    self.last_status_error = f"Unparseable string: {result[:200]}"
                    logger.warning(f"⚠️ get_order returned unparseable string: {result[:200]}...")
                    return None
            
            if result is None:
                self.last_status_error = "API returned None"
                logger.warning(f"⚠️ get_order returned None for order {order_id}")
                return None
            
            if not isinstance(result, dict):
                self.last_status_error = f"Unexpected type {type(result)}: {result}"
                logger.warning(f"⚠️ get_order returned unexpected type {type(result)}: {result}")
                return None
            
            # Check for error in response
            if "error" in result:
                self.last_status_error = f"API error: {result.get('error')}"
                logger.warning(f"⚠️ get_order returned error: {result.get('error')}")
                return None
            
            status = result.get("status", "UNKNOWN")
            # Parse size fields - may be strings
            original_size = float(result.get("original_size", 0) or 0)
            size_matched = float(result.get("size_matched", 0) or 0)
            
            # Try to get average price from API - check multiple possible fields
            avg_price = 0.0
            
            # 1. Try associate_trades first (actual fill data)
            associate_trades = result.get("associate_trades", [])
            if associate_trades and isinstance(associate_trades, list):
                total_value = 0.0
                total_size = 0.0
                for trade in associate_trades:
                    # Check if trade is a dict (has price/size) or just a string (trade ID)
                    if isinstance(trade, dict):
                        trade_price = float(trade.get("price", 0) or 0)
                        trade_size = float(trade.get("size", 0) or 0)
                        if trade_price > 0 and trade_size > 0:
                            total_value += trade_price * trade_size
                            total_size += trade_size
                    # Skip if trade is just a string (ID) - can't extract price from it
                if total_size > 0:
                    avg_price = total_value / total_size
                    logger.info(f"📊 Calculated avg_fill_price from trades: ${avg_price:.4f}")
            
            # 2. Try average_price field
            if avg_price == 0:
                avg_price = float(result.get("average_price", 0) or 0)
                if avg_price > 0:
                    logger.debug(f"📊 Using average_price field: ${avg_price:.4f}")
            
            # 3. Try avg_price field
            if avg_price == 0:
                avg_price = float(result.get("avg_price", 0) or 0)
                if avg_price > 0:
                    logger.debug(f"📊 Using avg_price field: ${avg_price:.4f}")
            
            # 4. Last resort: use limit price (NOT ideal, but better than 0)
            if avg_price == 0:
                avg_price = float(result.get("price", 0) or 0)
                if avg_price > 0:
                    logger.warning(f"⚠️ Using limit price as fallback: ${avg_price:.4f} (actual fill price unknown)")
            
            return OrderStatus(
                order_id=order_id,
                status=status,
                filled_size=size_matched,
                remaining_size=original_size - size_matched,
                avg_fill_price=avg_price
            )
            
        except Exception as e:
            import traceback
            self.last_status_error = f"Exception: {e}"
            logger.warning(f"Order status error for {order_id}: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if self.is_demo:
            logger.info(f"[DEMO] Cancel {order_id}")
            return True
        
        if not self._client:
            return False
        
        try:
            self._client.cancel(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.warning(f"Cancel error: {e}")
            return False
    
    def get_orderbook(self, token_id: str) -> Optional[dict]:
        """Get orderbook for token"""
        try:
            resp = self._session.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id}
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            best_bid = float(bids[0]["price"]) if bids else 0
            best_ask = float(asks[0]["price"]) if asks else 1
            
            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bids": bids[:5],
                "asks": asks[:5]
            }
            
        except Exception as e:
            logger.debug(f"Orderbook error: {e}")
            return None

    def get_marketable_price(self, token_id: str, side: str) -> Optional[float]:
        """Get a *marketable* price approximation for a token and action.

        We prefer the public /price endpoint because it reflects the best
        executable price for a BUY/SELL at that moment.

        Parameters
        ----------
        token_id: str
            Outcome token id.
        side: str
            Order action: BUY or SELL.
        """
        try:
            side_u = (side or "").upper()
            if side_u not in {"BUY", "SELL"}:
                # If caller accidentally passes YES/NO, treat as BUY.
                side_u = "BUY"

            resp = self._session.get(
                f"{CLOB_API}/price",
                params={"token_id": token_id, "side": side_u},
            )
            if resp.status_code != 200:
                return None

            data = resp.json() or {}
            p = data.get("price")
            if p is None:
                return None
            return float(p)
        except Exception as e:
            logger.debug(f"Get price error: {e}")
            return None

    def confirm_fill_until_deadline(
        self,
        order_id: str,
        deadline_ts: float,
        min_fill_threshold: float = 0.01,
        poll_interval: float = 1.5,
    ) -> Optional[ExecutionInfo]:
        """Poll for order fill confirmation until hard deadline.

        This is the **source of truth** for determining whether an order was
        actually filled.  It returns ``ExecutionInfo`` with ``filled_size`` and
        ``avg_price`` when the fill is confirmed, or ``None`` when the deadline
        is reached without confirmation.

        Parameters
        ----------
        order_id:
            The order id returned by ``place_order``.
        deadline_ts:
            Unix timestamp – stop polling after this moment (typically
            ``window_end - 10``).
        min_fill_threshold:
            Minimum ``filled_size`` to consider the order filled.
        poll_interval:
            Seconds between REST polls.
        """
        if self.is_demo:
            # Demo mode – orders fill immediately.
            o = self._demo_orders.get(order_id)
            if not o:
                return None
            filled = float(o.get("filled_size", 0.0))
            if filled < min_fill_threshold:
                return None
            fill_price = float(o.get("fill_price", o.get("price", 0.0)))
            return ExecutionInfo(
                order_id=order_id,
                filled_size=filled,
                avg_price=fill_price,
                fee_paid=0.0,
                status=str(o.get("status", "MATCHED")),
            )

        if not self._client:
            return None

        consecutive_errors = 0
        max_errors_before_warn = 5

        while time.time() < deadline_ts:
            try:
                order = self._client.get_order(order_id)
                if not order or not isinstance(order, dict):
                    consecutive_errors += 1
                    if consecutive_errors == max_errors_before_warn:
                        logger.warning(
                            f"confirm_fill: get_order returning None/bad response {consecutive_errors}x for {order_id}"
                        )
                    time.sleep(poll_interval)
                    continue

                consecutive_errors = 0
                status = str(order.get("status", "UNKNOWN")).upper()
                size_matched = float(order.get("size_matched", 0) or 0)

                # --- Fast path: status MATCHED + size_matched > threshold ---
                if status == "MATCHED" and size_matched >= min_fill_threshold:
                    # Try to get accurate avg_price via get_execution_info (short timeout)
                    exec_info = self.get_execution_info(order_id, timeout_s=8.0, poll_s=1.0)
                    if exec_info and exec_info.filled_size >= min_fill_threshold:
                        logger.info(
                            f"confirm_fill: CONFIRMED via exec_info – "
                            f"size={exec_info.filled_size}, avg_price=${exec_info.avg_price:.4f}"
                        )
                        return exec_info

                    # Fallback: build from order-level data
                    avg_price = float(order.get("average_price", 0) or order.get("avg_price", 0) or 0)
                    if avg_price <= 0:
                        avg_price = float(order.get("price", 0) or 0)
                    logger.info(
                        f"confirm_fill: CONFIRMED via order status – "
                        f"size={size_matched}, avg_price=${avg_price:.4f} (fallback)"
                    )
                    return ExecutionInfo(
                        order_id=order_id,
                        filled_size=size_matched,
                        avg_price=avg_price,
                        fee_paid=0.0,
                        status=status,
                    )

                # --- Partial fill above threshold (order still LIVE) ---
                # Do NOT return immediately! More fills may come. Track best seen.
                if size_matched >= min_fill_threshold and status in ("LIVE", "OPEN"):
                    avg_price = float(order.get("average_price", 0) or order.get("avg_price", 0) or 0)
                    if avg_price <= 0:
                        avg_price = float(order.get("price", 0) or 0)
                    # Log progress only when fill size changes
                    if not hasattr(self, '_last_partial_log') or self._last_partial_log != size_matched:
                        logger.info(
                            f"confirm_fill: PARTIAL FILL {size_matched} (order still {status}), "
                            f"avg_price=${avg_price:.4f} — waiting for more fills..."
                        )
                        self._last_partial_log = size_matched

                # --- Order cancelled / expired ---
                if status in ("CANCELLED", "EXPIRED"):
                    if size_matched >= min_fill_threshold:
                        avg_price = float(order.get("average_price", 0) or order.get("price", 0) or 0)
                        logger.info(f"confirm_fill: Order {status} but had partial fill {size_matched}")
                        return ExecutionInfo(
                            order_id=order_id,
                            filled_size=size_matched,
                            avg_price=avg_price,
                            fee_paid=0.0,
                            status=status,
                        )
                    logger.info(f"confirm_fill: Order {status} with no fill")
                    return None

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 2:
                    logger.warning(f"confirm_fill: poll error: {e}")
                elif consecutive_errors == max_errors_before_warn:
                    logger.error(f"confirm_fill: repeated errors ({consecutive_errors}x): {e}")

            time.sleep(poll_interval)

        # --- Deadline reached: one final check ---
        self._last_partial_log = None  # cleanup
        try:
            order = self._client.get_order(order_id)
            if order and isinstance(order, dict):
                size_matched = float(order.get("size_matched", 0) or 0)
                status_final = str(order.get("status", "UNKNOWN")).upper()
                if size_matched >= min_fill_threshold:
                    avg_price = float(order.get("average_price", 0) or order.get("avg_price", 0) or order.get("price", 0) or 0)
                    logger.info(
                        f"confirm_fill: CONFIRMED at deadline – size={size_matched}, "
                        f"avg_price=${avg_price:.4f}, status={status_final}"
                    )
                    # If order still LIVE, cancel remaining unfilled portion
                    if status_final in ("LIVE", "OPEN"):
                        logger.info(f"confirm_fill: Cancelling remaining unfilled portion of {order_id}")
                        try:
                            self._client.cancel(order_id)
                        except Exception:
                            pass  # Best effort
                    return ExecutionInfo(
                        order_id=order_id,
                        filled_size=size_matched,
                        avg_price=avg_price,
                        fee_paid=0.0,
                        status=status_final,
                    )
        except Exception as e:
            logger.warning(f"confirm_fill: final check error: {e}")

        logger.warning(f"confirm_fill: UNFILLED at deadline for order {order_id}")
        return None

    def recheck_fill_size(self, order_id: str) -> Optional[ExecutionInfo]:
        """Re-check order fill size before selling.
        
        Call this right before placing a SELL to catch any additional fills
        that arrived after initial confirm_fill_until_deadline returned.
        For example: ordered 20, initial confirm saw 10 (partial), 
        but by now all 20 may be filled.
        """
        if self.is_demo:
            return None
        if not self._client:
            return None
        
        try:
            order = self._client.get_order(order_id)
            if not order or not isinstance(order, dict):
                return None
            
            size_matched = float(order.get("size_matched", 0) or 0)
            status = str(order.get("status", "UNKNOWN")).upper()
            avg_price = float(
                order.get("average_price", 0) 
                or order.get("avg_price", 0) 
                or order.get("price", 0) 
                or 0
            )
            
            if size_matched > 0:
                return ExecutionInfo(
                    order_id=order_id,
                    filled_size=size_matched,
                    avg_price=avg_price,
                    fee_paid=0.0,
                    status=status,
                )
        except Exception as e:
            logger.warning(f"recheck_fill_size error for {order_id}: {e}")
        
        return None

    def safe_cancel(self, order_id: str) -> bool:
        """Attempt to cancel an order, swallowing errors.

        Returns True if cancel succeeded or order was already done.
        """
        try:
            return self.cancel_order(order_id)
        except Exception as e:
            logger.warning(f"safe_cancel error for {order_id}: {e}")
            return False

    def get_execution_info(self, order_id: str, timeout_s: float = 15.0, poll_s: float = 1.0) -> Optional[ExecutionInfo]:
        """Return executed avg price/fees for an order.

        LIVE mode: derives execution from associated trade objects and their
        maker legs (maker_orders).

        DEMO mode: uses the stored demo fill_price.
        """
        if self.is_demo:
            o = self._demo_orders.get(order_id)
            if not o:
                return None
            filled = float(o.get("filled_size", 0.0))
            fill_price = float(o.get("fill_price", o.get("price", 0.0)))
            return ExecutionInfo(
                order_id=order_id,
                filled_size=filled,
                avg_price=fill_price,
                fee_paid=0.0,
                status=str(o.get("status", "MATCHED")),
            )

        if not self._client:
            return None

        t0 = time.time()
        last_err = None
        while time.time() - t0 <= timeout_s:
            try:
                order = self._client.get_order(order_id)  # L2
                status = str(order.get("status", "UNKNOWN"))
                size_matched = float(order.get("size_matched", 0) or 0)
                assoc = order.get("associate_trades") or []

                # If no associated trades yet, wait.
                if not assoc or size_matched <= 0:
                    time.sleep(poll_s)
                    continue

                # Fetch each trade and compute execution from maker legs.
                trades: List[dict] = []
                from py_clob_client.clob_types import TradeParams

                # Preferred: fetch trade objects by their ids.
                for tid in assoc:
                    try:
                        resp = self._client.get_trades(TradeParams(id=tid))
                        if resp:
                            # SDK may return list or dict wrapper; normalize.
                            if isinstance(resp, list):
                                trades.extend(resp)
                            elif isinstance(resp, dict) and "trades" in resp:
                                trades.extend(resp.get("trades") or [])
                            elif isinstance(resp, dict):
                                trades.append(resp)
                    except Exception:
                        # ignore per-trade failures and try fallback below
                        pass

                # Fallback: scan recent trades for this market and filter by our order id.
                if not trades:
                    try:
                        market_id = order.get("market")
                        maker_addr = self.funder_address or ""
                        resp = self._client.get_trades(TradeParams(maker_address=maker_addr, market=market_id))
                        if resp:
                            if isinstance(resp, list):
                                candidates = resp
                            elif isinstance(resp, dict) and "trades" in resp:
                                candidates = resp.get("trades") or []
                            else:
                                candidates = [resp] if isinstance(resp, dict) else []

                            for tr in candidates:
                                if tr.get("taker_order_id") == order_id:
                                    trades.append(tr)
                    except Exception:
                        pass

                if not trades:
                    time.sleep(poll_s)
                    continue

                exec_notional = 0.0
                exec_size = 0.0
                fee_paid = 0.0

                for tr in trades:
                    # Use maker legs if present (true execution prices).
                    maker_legs = tr.get("maker_orders") or []
                    if maker_legs:
                        for leg in maker_legs:
                            amt = float(leg.get("matched_amount", 0) or 0)
                            pr = float(leg.get("price", 0) or 0)
                            if amt > 0 and pr > 0:
                                exec_size += amt
                                exec_notional += amt * pr
                    else:
                        # Fallback: use trade-level price/size.
                        amt = float(tr.get("size", 0) or 0)
                        pr = float(tr.get("price", 0) or 0)
                        if amt > 0 and pr > 0:
                            exec_size += amt
                            exec_notional += amt * pr

                    # Fees (bps) apply to taker notional for that trade.
                    try:
                        bps = float(tr.get("fee_rate_bps", 0) or 0)
                        # Prefer maker_leg notional if available for that trade.
                        trade_notional = 0.0
                        if maker_legs:
                            for leg in maker_legs:
                                amt = float(leg.get("matched_amount", 0) or 0)
                                pr = float(leg.get("price", 0) or 0)
                                trade_notional += amt * pr
                        else:
                            trade_notional = float(tr.get("size", 0) or 0) * float(tr.get("price", 0) or 0)
                        if trade_notional > 0 and bps > 0:
                            fee_paid += trade_notional * (bps / 10000.0)
                    except Exception:
                        pass

                if exec_size <= 0:
                    time.sleep(poll_s)
                    continue

                avg_price = exec_notional / exec_size
                return ExecutionInfo(
                    order_id=order_id,
                    filled_size=exec_size,
                    avg_price=avg_price,
                    fee_paid=fee_paid,
                    status=status,
                )

            except Exception as e:
                last_err = e
                time.sleep(poll_s)

        if last_err:
            logger.warning(f"Execution info timeout for {order_id}: {last_err}")
        return None
    
    def get_token_balance(self, token_id: str) -> float:
        """Get balance of specific token. Returns 0 if error or no balance."""
        if self.is_demo:
            # W demo mode nie mamy rzeczywistego balansu
            return 0.0
        
        if not self._client:
            return 0.0
        
        try:
            # Try via py-clob-client
            balances = self._client.get_balances()
            if balances:
                for balance in balances:
                    if balance.get("asset_id") == token_id or balance.get("token_id") == token_id:
                        return float(balance.get("balance", 0))
            return 0.0
        except Exception as e:
            logger.debug(f"Get balance via client error: {e}")
        
        # Fallback: direct API call
        try:
            if not self.funder_address:
                return 0.0
            
            resp = self._session.get(
                f"{CLOB_API}/balances",
                params={"address": self.funder_address}
            )
            
            if resp.status_code == 200:
                data = resp.json()
                balances = data if isinstance(data, list) else data.get("balances", [])
                for balance in balances:
                    if balance.get("asset_id") == token_id or balance.get("token_id") == token_id:
                        return float(balance.get("balance", 0))
            return 0.0
        except Exception as e:
            logger.debug(f"Get balance via API error: {e}")
            return 0.0
