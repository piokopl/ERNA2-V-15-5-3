#!/usr/bin/env python3
"""Live web dashboard for PolySniper In-Window bot.

Features:
- Auto-refresh every second (no page reload needed)
- Live prices from WebSocket
- Order status tracking
- Position monitoring

Endpoints:
- GET /           : HTML dashboard with live updates
- GET /api/status : live status per asset (JSON)
- GET /api/stats  : stats per asset + total (JSON)
- GET /api/trades : recent trades (JSON)
"""

from __future__ import annotations

import html
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from bot_registry import REGISTRY


# ── API field masking ─────────────────────────────────────────────────────────
# Maps internal field names → opaque names so API consumers can't identify
# the indicators, strategy, or internal structure.
FIELD_MASK = {
    # Trend fields
    "primary_trend": "t1", "secondary_trend": "t2", "macd_trend": "t3",
    "vki_trend": "t4", "keltner_trend": "t5", "multi_trend": "t6",
    "lorentzian_trend": "t7", "ai_trend": "t8",
    # Strategy params
    "buy_limit_price": "lim", "avg_fill_price": "fill_px",
    "sell_price": "tgt", "order_size": "qty",
    "trigger_price_yes": "tr_a", "trigger_price_no": "tr_b",
    "martingale_level": "lvl", "target_hit_price": "hit_px",
    "buy_price": "entry",
    # Order state
    "yes_order_id": "a_oid", "no_order_id": "b_oid",
    "yes_triggered": "a_trig", "no_triggered": "b_trig",
    "yes_filled": "a_fl", "no_filled": "b_fl",
    "yes_filled_size": "a_fqty", "no_filled_size": "b_fqty",
    "yes_sold": "a_sold", "no_sold": "b_sold",
    "yes_pnl": "a_pnl", "no_pnl": "b_pnl",
    "yes_ask": "a_ask", "yes_bid": "a_bid",
    "no_ask": "b_ask", "no_bid": "b_bid",
    # Meta
    "market_slug": "mkt", "market_question": "q",
    "market_end_ts": "end_ts", "trend_mode": "mode",
    "position_side": "p_side", "position_tokens": "p_qty",
    "winning_side": "outcome",
    # Stats / trend accuracy
    "per_trend": "signals", "per_asset": "by_asset",
    "primary": "t1", "secondary": "t2", "macd": "t3",
    "vki": "t4", "keltner": "t5", "multi": "t6",
    "lorentzian": "t7", "ai": "t8",
}

COMBO_KEY_MASK = {"P": "T1", "S": "T2", "D": "T3", "V": "T4", "K": "T5", "M": "T6", "L": "T7", "A": "T8"}


def _mask_combo_key(key: str) -> str:
    """Remap trend combination keys: P+S_AGREE → T1+T2_AGREE."""
    if key in COMBO_KEY_MASK:
        return COMBO_KEY_MASK[key]
    if "+" in key:
        parts = key.split("_", 1)
        trend_part = parts[0]
        suffix = "_" + parts[1] if len(parts) > 1 else ""
        masked = "+".join(COMBO_KEY_MASK.get(t, t) for t in trend_part.split("+"))
        return masked + suffix
    return key


def _mask(obj):
    """Recursively mask field names in API response."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # First try FIELD_MASK, then combo key mask (for P, S, D, V, K, M, P+S_AGREE etc.)
            new_key = FIELD_MASK.get(k, None) or _mask_combo_key(k)
            out[new_key] = _mask(v)
        return out
    elif isinstance(obj, list):
        return [_mask(item) for item in obj]
    elif isinstance(obj, tuple):
        lst = list(obj)
        if len(lst) == 2 and isinstance(lst[0], str):
            lst[0] = _mask_combo_key(lst[0])
        return [_mask(item) for item in lst]
    return obj


app = FastAPI(title="PolySniper Dashboard", docs_url=None, redoc_url=None)


@app.get("/api/status")
def api_status():
    return JSONResponse(_mask(REGISTRY.get_status()))


@app.get("/api/stats")
def api_stats():
    return JSONResponse(_mask(REGISTRY.get_stats()))


@app.get("/api/trades")
def api_trades(asset: str | None = Query(default=None), limit: int = Query(default=200, ge=1, le=2000)):
    asset_u = asset.upper() if asset else None
    return JSONResponse(_mask(REGISTRY.get_trades(asset=asset_u, limit=limit)))


@app.get("/api/trend_accuracy")
def api_trend_accuracy():
    return JSONResponse(_mask(REGISTRY.get_trend_accuracy()))


@app.get("/api/trend_combinations")
def api_trend_combinations():
    """Get accuracy statistics for trend combinations."""
    return JSONResponse(_mask(REGISTRY.get_trend_combinations_accuracy()))


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the live dashboard HTML."""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{DASHBOARD_TITLE}} - Live Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; 
            background: #0f0f1a; 
            color: #e0e0e0;
            padding: 20px;
        }
        
        h1 { 
            color: #00d4ff; 
            margin-bottom: 5px;
            font-size: 28px;
        }
        .subtitle { color: #888; font-size: 14px; margin-bottom: 20px; }
        h2 { 
            color: #00d4ff; 
            font-size: 18px; 
            margin: 25px 0 15px 0;
            border-bottom: 1px solid #333;
            padding-bottom: 8px;
        }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
        
        .card {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #333;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .asset-name {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }
        
        .status-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-MONITORING { background: #1e3a5f; color: #5dade2; }
        .status-TRADING { background: #1e5f3a; color: #58d68d; }
        .status-IDLE { background: #3d3d3d; color: #888; }
        .status-DISABLED { background: #5f1e1e; color: #e74c3c; }
        .status-NO_MARKET { background: #5f4a1e; color: #f39c12; }
        .status-ERROR { background: #5f1e1e; color: #e74c3c; }
        .status-BLACKLISTED { background: #4a1e5f; color: #9b59b6; }
        
        .market-info {
            background: #252538;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
            font-size: 13px;
        }
        .market-info .label { color: #888; }
        .market-info .value { color: #fff; font-family: monospace; }
        .market-info .time-left { 
            font-size: 20px; 
            font-weight: bold; 
            color: #f39c12;
        }
        .market-info .time-left.urgent { color: #e74c3c; }
        
        .prices-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .price-box {
            background: #252538;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .price-box.yes { border-left: 4px solid #27ae60; }
        .price-box.no { border-left: 4px solid #e74c3c; }
        
        .price-box .token-label {
            font-size: 14px;
            color: #888;
            margin-bottom: 8px;
        }
        .price-box .prices {
            display: flex;
            justify-content: space-around;
        }
        .price-box .price-item {
            text-align: center;
        }
        .price-box .price-label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
        }
        .price-box .price-value {
            font-size: 18px;
            font-weight: bold;
            font-family: monospace;
        }
        .price-box .price-value.ask { color: #e74c3c; }
        .price-box .price-value.bid { color: #27ae60; }
        
        .order-status {
            background: #252538;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .order-status .order-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .order-status .order-title {
            font-weight: bold;
            font-size: 14px;
        }
        .order-status.yes .order-title { color: #27ae60; }
        .order-status.no .order-title { color: #e74c3c; }
        
        .order-status .order-badges {
            display: flex;
            gap: 6px;
        }
        .order-badge {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }
        .order-badge.triggered { background: #f39c12; color: #000; }
        .order-badge.filled { background: #27ae60; color: #fff; }
        .order-badge.sold { background: #3498db; color: #fff; }
        .order-badge.waiting { background: #555; color: #999; }
        .order-badge.winning { background: #27ae60; color: #fff; animation: pulse-green 1s infinite; }
        .order-badge.losing { background: #e74c3c; color: #fff; animation: pulse-red 1s infinite; }
        .order-badge.neutral { background: #f39c12; color: #000; }
        
        @keyframes pulse-green {
            0%, 100% { background: #27ae60; }
            50% { background: #2ecc71; }
        }
        @keyframes pulse-red {
            0%, 100% { background: #e74c3c; }
            50% { background: #c0392b; }
        }
        
        .order-details {
            font-size: 12px;
            color: #888;
            font-family: monospace;
        }
        .order-details .pnl-positive { color: #27ae60; font-weight: bold; }
        .order-details .pnl-negative { color: #e74c3c; font-weight: bold; }
        
        .trends {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .trend-item {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            background: #333;
        }
        .trend-item.UP { background: #1e5f3a; color: #58d68d; }
        .trend-item.DOWN { background: #5f1e1e; color: #e74c3c; }
        
        .config-info {
            font-size: 11px;
            color: #666;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #333;
        }
        
        .stats-section {
            margin-top: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .stat-box {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-box .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #00d4ff;
        }
        .stat-box .stat-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .stat-box.positive .stat-value { color: #27ae60; }
        .stat-box.negative .stat-value { color: #e74c3c; }
        
        .last-update {
            text-align: right;
            font-size: 11px;
            color: #555;
            margin-top: 20px;
        }
        
        .history-section {
            margin-top: 30px;
        }
        
        .history-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .history-controls select {
            background: #252538;
            color: #e0e0e0;
            border: 1px solid #444;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
        }
        
        .trade-count {
            color: #888;
            font-size: 13px;
            margin-left: auto;
        }
        
        .btn-clear {
            background: #5f1e1e;
            color: #e74c3c;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }
        .btn-clear:hover {
            background: #7a2828;
        }
        
        .history-table-wrapper {
            background: #1a1a2e;
            border-radius: 12px;
            overflow: hidden;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        
        .history-table th {
            background: #252538;
            color: #00d4ff;
            padding: 12px 8px;
            text-align: left;
            position: sticky;
            top: 0;
            font-weight: 600;
        }
        
        .history-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #2a2a3e;
            color: #ccc;
        }
        
        .history-table tr:hover {
            background: #252538;
        }
        
        .history-table .result-WIN { color: #27ae60; font-weight: bold; }
        .history-table .result-LOSS { color: #e74c3c; font-weight: bold; }
        .history-table .result-SETTLEMENT { color: #f39c12; font-weight: bold; }
        
        .history-table .pnl-positive { color: #27ae60; }
        .history-table .pnl-negative { color: #e74c3c; }
        
        .history-table .side-YES { color: #27ae60; }
        .history-table .side-NO { color: #e74c3c; }
        
        .history-table .trends {
            font-size: 10px;
            color: #888;
        }
        
        .history-table .market-slug {
            font-family: monospace;
            font-size: 10px;
            color: #666;
        }
        
        .history-table .correct-many { color: #27ae60; font-weight: bold; }
        .history-table .correct-one { color: #f39c12; }
        .history-table .correct-none { color: #e74c3c; }
        
        .trades-section {
            margin-top: 30px;
        }
        
        .trades-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #1a1a2e;
            border-radius: 8px;
            overflow: hidden;
        }
        .trades-table th {
            background: #252538;
            color: #888;
            font-weight: 600;
            text-align: left;
            padding: 12px 10px;
            font-size: 11px;
            text-transform: uppercase;
            border-bottom: 1px solid #333;
        }
        .trades-table td {
            padding: 10px;
            border-bottom: 1px solid #2a2a3e;
            color: #ccc;
        }
        .trades-table tr:hover {
            background: #252538;
        }
        .trades-table .asset { font-weight: bold; color: #fff; }
        .trades-table .side-YES { color: #27ae60; font-weight: bold; }
        .trades-table .side-NO { color: #e74c3c; font-weight: bold; }
        .trades-table .result-WIN { color: #27ae60; font-weight: bold; }
        .trades-table .result-LOSS { color: #e74c3c; font-weight: bold; }
        .trades-table .result-SETTLEMENT { color: #f39c12; font-weight: bold; }
        .trades-table .pnl-positive { color: #27ae60; }
        .trades-table .pnl-negative { color: #e74c3c; }
        .trades-table .trend-UP { color: #27ae60; }
        .trades-table .trend-DOWN { color: #e74c3c; }
        .trades-table .timestamp { color: #666; font-size: 11px; }
        
        .no-data {
            color: #666;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #27ae60;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <h1>🎯 {{DASHBOARD_TITLE}}</h1>
    <div class="subtitle"><span class="live-indicator"></span>Live Dashboard - Auto-refreshing every second</div>
    
    <div id="assets-container" class="grid">
        <div class="no-data">Loading...</div>
    </div>
    
    <div class="stats-section">
        <h2>📊 Statistics</h2>
        <div id="stats-container" class="stats-grid">
            <div class="no-data">Loading...</div>
        </div>
    </div>
    
    <div class="trend-analysis-section" style="margin-top: 20px;">
        <h2>🎯 Trend Analysis</h2>
        <div class="trend-analysis-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px;">
            <!-- Single Trends Performance -->
            <div class="trend-card" style="background: #1a1a2e; border-radius: 10px; padding: 15px;">
                <h3 style="color: #00d4aa; margin-bottom: 10px; font-size: 14px;">📈 Single Trend Accuracy</h3>
                <div id="single-trends" style="font-size: 13px;">
                    <div class="no-data">Loading...</div>
                </div>
            </div>
            
            <!-- Best Combinations -->
            <div class="trend-card" style="background: #1a1a2e; border-radius: 10px; padding: 15px;">
                <h3 style="color: #00d4aa; margin-bottom: 10px; font-size: 14px;">🏆 Best Combinations</h3>
                <div id="best-combinations" style="font-size: 13px;">
                    <div class="no-data">Loading...</div>
                </div>
            </div>
            
            <!-- Recommendations -->
            <div class="trend-card" style="background: #1a1a2e; border-radius: 10px; padding: 15px;">
                <h3 style="color: #00d4aa; margin-bottom: 10px; font-size: 14px;">💡 Recommendation</h3>
                <div id="trend-recommendation" style="font-size: 13px;">
                    <div class="no-data">Loading...</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="history-section">
        <h2>📜 Trade History</h2>
        <div class="history-controls">
            <select id="filter-asset" onchange="updateHistory()">
                <option value="">All Assets</option>
                <option value="BTC">BTC</option>
                <option value="ETH">ETH</option>
                <option value="SOL">SOL</option>
                <option value="XRP">XRP</option>
            </select>
            <select id="filter-result" onchange="updateHistory()">
                <option value="">All Results</option>
                <option value="WIN">WIN</option>
                <option value="LOSS">LOSS</option>
                <option value="SETTLEMENT">SETTLEMENT</option>
            </select>
            <span id="trade-count" class="trade-count">0 trades</span>
        </div>
        <div id="history-container" class="history-table-wrapper">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Asset</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Buy Price</th>
                        <th>Sell Price</th>
                        <th>PnL</th>
                        <th>Result</th>
                        <th>Market</th>
                        <th>Trends</th>
                        <th>Correct</th>
                    </tr>
                </thead>
                <tbody id="history-tbody">
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="last-update" id="last-update"></div>
    
    <script>
        function formatTime(seconds) {
            if (seconds <= 0) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
        
        function formatPrice(price) {
            return price > 0 ? '$' + price.toFixed(4) : '-';
        }
        
        function formatPnl(pnl) {
            if (pnl === 0) return '-';
            const sign = pnl >= 0 ? '+' : '';
            const cls = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            return `<span class="${cls}">${sign}$${pnl.toFixed(4)}</span>`;
        }
        
        function renderAssetCard(asset, data) {
            const state = data.decision || 'IDLE';
            const timeLeft = data.time_left || 0;
            const timeClass = timeLeft < 60 ? 'urgent' : '';
            
            // Order badges
            function getOrderBadges(triggered, filled, sold, targetHitPrice) {
                let badges = [];
                if (sold) {
                    const priceStr = targetHitPrice > 0 ? ` @$${targetHitPrice.toFixed(2)}` : '';
                    badges.push(`<span class="order-badge sold">TARGET HIT${priceStr}</span>`);
                }
                else if (filled) badges.push('<span class="order-badge filled">FILLED</span>');
                else if (triggered) badges.push('<span class="order-badge triggered">TRIGGERED</span>');
                else badges.push('<span class="order-badge waiting">WAITING</span>');
                return badges.join('');
            }
            
            // P/L badge for active positions
            function getPnlBadge(filled, sold, currentBid, buyPrice) {
                if (!filled || sold || !currentBid || !buyPrice) return '';
                
                const diff = currentBid - buyPrice;
                const pct = ((diff / buyPrice) * 100).toFixed(1);
                
                if (diff > 0) {
                    return `<span class="order-badge winning">▲ WINNING +${pct}%</span>`;
                } else if (diff < 0) {
                    return `<span class="order-badge losing">▼ LOSING ${pct}%</span>`;
                } else {
                    return `<span class="order-badge neutral">= BREAK EVEN</span>`;
                }
            }
            
            // Trend badges
            function getTrendBadge(name, value) {
                if (!value) return '';
                return `<span class="trend-item ${value}">${name}: ${value}</span>`;
            }
            
            return `
                <div class="card">
                    <div class="card-header">
                        <span class="asset-name">${asset}</span>
                        <span class="status-badge status-${state}">${state}</span>
                    </div>
                    
                    <div class="market-info">
                        <div><span class="label">Market:</span> <span class="value">${data.mkt || '-'}</span></div>
                        <div><span class="label">Question:</span> <span class="value">${data.q || '-'}</span></div>
                        <div style="margin-top: 8px;">
                            <span class="label">Time left:</span> 
                            <span class="time-left ${timeClass}">${formatTime(timeLeft)}</span>
                        </div>
                    </div>
                    
                    <div class="prices-grid">
                        <div class="price-box yes">
                            <div class="token-label">YES Token</div>
                            <div class="prices">
                                <div class="price-item">
                                    <div class="price-label">Ask</div>
                                    <div class="price-value ask">${formatPrice(data.a_ask)}</div>
                                </div>
                                <div class="price-item">
                                    <div class="price-label">Bid</div>
                                    <div class="price-value bid">${formatPrice(data.a_bid)}</div>
                                </div>
                            </div>
                        </div>
                        <div class="price-box no">
                            <div class="token-label">NO Token</div>
                            <div class="prices">
                                <div class="price-item">
                                    <div class="price-label">Ask</div>
                                    <div class="price-value ask">${formatPrice(data.b_ask)}</div>
                                </div>
                                <div class="price-item">
                                    <div class="price-label">Bid</div>
                                    <div class="price-value bid">${formatPrice(data.b_bid)}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    ${(data.mode === 'pre_market' || data.mode === 'in_window') ? `
                    <!-- In-Window / In-Window: Single Position -->
                    <div class="order-status ${data.a_trig || data.a_fl ? 'yes' : 'no'}">
                        <div class="order-header">
                            <span class="order-title">Position: ${data.a_trig || data.a_fl ? 'YES' : data.b_trig || data.b_fl ? 'NO' : 'NONE'}</span>
                            <div class="order-badges">
                                ${data.a_sold || data.b_sold 
                                    ? `<span class="order-badge sold">TARGET HIT @$${(data.hit_px || 0).toFixed(2)}</span>`
                                    : data.a_fl || data.b_fl 
                                        ? '<span class="order-badge filled">FILLED</span>' 
                                        : data.a_trig || data.b_trig 
                                            ? '<span class="order-badge triggered">PENDING</span>'
                                            : '<span class="order-badge waiting">WAITING</span>'}
                                ${data.a_fl && !data.a_sold ? getPnlBadge(true, data.a_sold, data.a_bid, data.fill_px || data.lim) : ''}
                                ${data.b_fl && !data.b_sold ? getPnlBadge(true, data.b_sold, data.b_bid, data.fill_px || data.lim) : ''}
                            </div>
                        </div>
                        <div class="order-details">
                            ${data.a_oid ? `ID: ${data.a_oid.substring(0, 20)}...` : ''}
                            ${data.b_oid ? `ID: ${data.b_oid.substring(0, 20)}...` : ''}
                            ${data.a_fl ? ` | Filled: ${data.a_fqty} YES @ $${(data.fill_px || data.lim).toFixed(4)}` : ''}
                            ${data.b_fl ? ` | Filled: ${data.b_fqty} NO @ $${(data.fill_px || data.lim).toFixed(4)}` : ''}
                            ${data.a_fl && !data.a_sold ? ` | Current: $${(data.a_bid || 0).toFixed(4)} | Target: $${data.tgt || 0}` : ''}
                            ${data.b_fl && !data.b_sold ? ` | Current: $${(data.b_bid || 0).toFixed(4)} | Target: $${data.tgt || 0}` : ''}
                            ${(() => {
                                const fillPrice = data.fill_px || data.lim;
                                if (data.a_fl && !data.a_sold && data.a_bid > 0 && fillPrice > 0) {
                                    const livePnl = (data.a_fqty * data.a_bid) - (data.a_fqty * fillPrice);
                                    const sign = livePnl >= 0 ? '+' : '';
                                    const cls = livePnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                                    return ' | <span class="' + cls + '">Live PnL: ' + sign + '$' + livePnl.toFixed(4) + '</span>';
                                }
                                if (data.b_fl && !data.b_sold && data.b_bid > 0 && fillPrice > 0) {
                                    const livePnl = (data.b_fqty * data.b_bid) - (data.b_fqty * fillPrice);
                                    const sign = livePnl >= 0 ? '+' : '';
                                    const cls = livePnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                                    return ' | <span class="' + cls + '">Live PnL: ' + sign + '$' + livePnl.toFixed(4) + '</span>';
                                }
                                return '';
                            })()}
                        </div>
                    </div>
                    ` : `
                    <!-- In-Window: YES and NO Orders -->
                    <div class="order-status yes">
                        <div class="order-header">
                            <span class="order-title">YES Order</span>
                            <div class="order-badges">
                                ${getOrderBadges(data.a_trig, data.a_fl, data.a_sold, data.hit_px || 0)}
                                ${getPnlBadge(data.a_fl, data.a_sold, data.a_bid, data.fill_px || data.lim)}
                            </div>
                        </div>
                        <div class="order-details">
                            ${data.a_oid ? `ID: ${data.a_oid.substring(0, 20)}...` : ''}
                            ${data.a_fl ? ` | Filled: ${data.a_fqty} tokens @ $${(data.fill_px || data.lim).toFixed(4)}` : ''}
                            ${data.a_fl && !data.a_sold ? ` | Current: $${(data.a_bid || 0).toFixed(4)}` : ''}
                            ${data.a_sold ? ` | PnL: ${formatPnl(data.a_pnl)}` : ''}
                        </div>
                    </div>
                    
                    <div class="order-status no">
                        <div class="order-header">
                            <span class="order-title">NO Order</span>
                            <div class="order-badges">
                                ${getOrderBadges(data.b_trig, data.b_fl, data.b_sold, data.hit_px || 0)}
                                ${getPnlBadge(data.b_fl, data.b_sold, data.b_bid, data.fill_px || data.lim)}
                            </div>
                        </div>
                        <div class="order-details">
                            ${data.b_oid ? `ID: ${data.b_oid.substring(0, 20)}...` : ''}
                            ${data.b_fl ? ` | Filled: ${data.b_fqty} tokens @ $${(data.fill_px || data.lim).toFixed(4)}` : ''}
                            ${data.b_fl && !data.b_sold ? ` | Current: $${(data.b_bid || 0).toFixed(4)}` : ''}
                            ${data.b_sold ? ` | PnL: ${formatPnl(data.b_pnl)}` : ''}
                        </div>
                    </div>
                    `}
                    
                    <div class="trends">
                        ${getTrendBadge('Thunder', data.t1)}
                        ${getTrendBadge('Machine', data.t2)}
                        ${getTrendBadge('Magellan', data.t3)}
                        ${getTrendBadge('Viki', data.t4)}
                        ${getTrendBadge('Karol', data.t5)}
                        ${getTrendBadge('Vater', data.t6)}
                    </div>
                    
                    <div class="config-info">
                        ${(data.mode === 'pre_market' || data.mode === 'in_window')
                            ? `In-Window | Limit @$${data.lim || 0}${data.fill_px > 0 ? ' | <strong style="color:#00d4ff">Fill @$' + data.fill_px.toFixed(4) + '</strong>' : ''} | Target @$${data.tgt || 0} | Size: ${data.qty || 0}`
                            : `Trigger: YES ≤$${data.tr_a || 0} | NO ≤$${data.tr_b || 0} | Limit @$${data.lim || 0}${data.fill_px > 0 ? ' | Fill @$' + data.fill_px.toFixed(4) : ''} | Sell @$${data.tgt || 0} | Size: ${data.qty || 0}`
                        }
                    </div>
                </div>
            `;
        }
        
        function renderStats(stats) {
            const total = stats.total || {};
            const pnlClass = (total.pnl || 0) >= 0 ? 'positive' : 'negative';
            
            return `
                <div class="stat-box">
                    <div class="stat-value">${total.trades || 0}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-box positive">
                    <div class="stat-value">${total.wins || 0}</div>
                    <div class="stat-label">Wins</div>
                </div>
                <div class="stat-box negative">
                    <div class="stat-value">${total.losses || 0}</div>
                    <div class="stat-label">Losses</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${total.settlements || 0}</div>
                    <div class="stat-label">Settlements</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${(total.win_rate || 0).toFixed(1)}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="stat-box ${pnlClass}">
                    <div class="stat-value">$${(total.pnl || 0).toFixed(2)}</div>
                    <div class="stat-label">Total PnL</div>
                </div>
            `;
        }
        
        // Store all trades for filtering
        let allTrades = [];
        
        function renderHistory() {
            const assetFilter = document.getElementById('filter-asset').value;
            const resultFilter = document.getElementById('filter-result').value;
            
            let filtered = allTrades;
            if (assetFilter) {
                filtered = filtered.filter(t => t.asset === assetFilter);
            }
            if (resultFilter) {
                filtered = filtered.filter(t => t.result === resultFilter);
            }
            
            document.getElementById('trade-count').textContent = `${filtered.length} trades`;
            
            const tbody = document.getElementById('history-tbody');
            
            if (filtered.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" class="no-data">No trades</td></tr>';
                return;
            }
            
            tbody.innerHTML = filtered.map(t => {
                const pnl = parseFloat(t.pnl) || 0;
                const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                const pnlSign = pnl >= 0 ? '+' : '';
                
                const trends = [
                    t.t1 || '-',
                    t.t2 || '-', 
                    t.t3 || '-',
                    t.t4 || '-',
                    t.t5 || '-',
                    t.t6 || '-'
                ].join('/');
                
                // Calculate which trends were correct
                // UP predicts YES wins, DOWN predicts NO wins
                const winningSide = t.outcome; // YES or NO
                let correctTrends = [];
                
                if (winningSide === 'YES') {
                    if (t.t1 === 'UP') correctTrends.push('T');
                    if (t.t2 === 'UP') correctTrends.push('Ma');
                    if (t.t3 === 'UP') correctTrends.push('Mg');
                    if (t.t4 === 'UP') correctTrends.push('Vi');
                    if (t.t5 === 'UP') correctTrends.push('K');
                    if (t.t6 === 'UP') correctTrends.push('Va');
                } else if (winningSide === 'NO') {
                    if (t.t1 === 'DOWN') correctTrends.push('T');
                    if (t.t2 === 'DOWN') correctTrends.push('Ma');
                    if (t.t3 === 'DOWN') correctTrends.push('Mg');
                    if (t.t4 === 'DOWN') correctTrends.push('Vi');
                    if (t.t5 === 'DOWN') correctTrends.push('K');
                    if (t.t6 === 'DOWN') correctTrends.push('Va');
                }
                
                const correctStr = correctTrends.length > 0 
                    ? correctTrends.join(', ') 
                    : (winningSide ? 'None' : '-');
                const correctClass = correctTrends.length >= 2 ? 'correct-many' : 
                                    (correctTrends.length === 1 ? 'correct-one' : 'correct-none');
                
                return `
                    <tr>
                        <td>${t.ts || '-'}</td>
                        <td><strong>${t.asset || '-'}</strong></td>
                        <td class="side-${t.side}">${t.side || '-'}</td>
                        <td>${parseFloat(t.size || 0).toFixed(2)}</td>
                        <td>$${parseFloat(t.entry || 0).toFixed(4)}</td>
                        <td>$${parseFloat(t.tgt || 0).toFixed(4)}</td>
                        <td class="${pnlClass}">${pnlSign}$${pnl.toFixed(4)}</td>
                        <td class="result-${t.result}">${t.result || '-'}</td>
                        <td class="market-slug">${t.mkt || '-'}</td>
                        <td class="trends">${trends}</td>
                        <td class="${correctClass}">${correctStr}</td>
                    </tr>
                `;
            }).join('');
        }
        
        function updateHistory() {
            renderHistory();
        }
        
        async function clearTrades() {
            if (!confirm('Clear all trade history?')) return;
            try {
                // clear_trades disabled
                allTrades = [];
                renderHistory();
            } catch (e) {
                console.error('Clear trades error:', e);
            }
        }
        
        async function updateDashboard() {
            try {
                // Fetch status
                const statusResp = await fetch('/api/status');
                const status = await statusResp.json();
                
                // Fetch stats
                const statsResp = await fetch('/api/stats');
                const stats = await statsResp.json();
                
                // Fetch trades
                const tradesResp = await fetch('/api/trades?limit=500');
                const trades = await tradesResp.json();
                allTrades = trades;
                
                // Fetch trend combinations (every 5 seconds to reduce load)
                if (!window.lastTrendUpdate || Date.now() - window.lastTrendUpdate > 5000) {
                    const trendResp = await fetch('/api/trend_combinations');
                    const trendData = await trendResp.json();
                    renderTrendAnalysis(trendData);
                    window.lastTrendUpdate = Date.now();
                }
                
                // Render assets
                const assetsContainer = document.getElementById('assets-container');
                const assets = Object.entries(status);
                
                if (assets.length === 0) {
                    assetsContainer.innerHTML = '<div class="no-data">No bots running</div>';
                } else {
                    assetsContainer.innerHTML = assets
                        .sort((a, b) => a[0].localeCompare(b[0]))
                        .map(([asset, data]) => renderAssetCard(asset, data))
                        .join('');
                }
                
                // Render stats
                const statsContainer = document.getElementById('stats-container');
                statsContainer.innerHTML = renderStats(stats);
                
                // Render history
                renderHistory();
                
                // Update timestamp
                document.getElementById('last-update').textContent = 
                    'Last update: ' + new Date().toLocaleTimeString();
                    
            } catch (error) {
                console.error('Dashboard update error:', error);
            }
        }
        
        function renderTrendAnalysis(data) {
            // Render single trends
            const singleTrendsEl = document.getElementById('single-trends');
            const trendNames = {
                'T1': 'Thunder',
                'T2': 'Machine', 
                'T3': 'Magellan',
                'T4': 'Viki',
                'T5': 'Karol',
                'T6': 'Vater',
                'T7': 'Laurent',
                'T8': 'Apollo'
            };
            const singleTrends = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'].map(key => {
                const t = data.all_combinations[key];
                if (!t || t.total === 0) {
                    return `<div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #333;">
                        <span>${trendNames[key] || key}</span>
                        <span style="color: #666;">-</span>
                        <span style="color: #555;">0W/0L (0)</span>
                    </div>`;
                }
                const color = t.accuracy >= 60 ? '#00d4aa' : t.accuracy >= 50 ? '#ffc107' : '#ff4757';
                return `<div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #333;">
                    <span>${trendNames[key] || key}</span>
                    <span style="color: ${color}; font-weight: bold;">${t.accuracy.toFixed(1)}%</span>
                    <span style="color: #888;">${t.wins}W/${t.losses}L (${t.total})</span>
                </div>`;
            }).join('');
            
            singleTrendsEl.innerHTML = singleTrends;
            
            // Remap internal trend keys to display names
            const keyRemap = {'T1': 'T', 'T2': 'Ma', 'T3': 'Mg', 'T4': 'Vi', 'T5': 'Ka', 'T6': 'Va', 'T7': 'La', 'T8': 'Ap'};
            function remapComboKey(key) {
                return key.split('+').map(k => keyRemap[k] || k).join('+');
            }
            
            // Render best combinations
            const bestCombosEl = document.getElementById('best-combinations');
            const ranked = data.ranked || [];
            const bestCombos = ranked.slice(0, 8).map((item, idx) => {
                const [key, stats] = item;
                const color = stats.accuracy >= 70 ? '#00d4aa' : stats.accuracy >= 55 ? '#ffc107' : '#ff4757';
                const medal = idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : '';
                return `<div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #333;">
                    <span>${medal} ${remapComboKey(key)}</span>
                    <span style="color: ${color}; font-weight: bold;">${stats.accuracy.toFixed(1)}%</span>
                    <span style="color: #888;">${stats.wins}/${stats.total}</span>
                </div>`;
            }).join('');
            
            bestCombosEl.innerHTML = bestCombos || '<div class="no-data">Not enough data (min 3 trades)</div>';
            
            // Render recommendation
            const recEl = document.getElementById('trend-recommendation');
            let recommendation = '';
            
            if (data.best_single) {
                const [key, stats] = data.best_single;
                const name = trendNames[key] || key;
                const displayKey = keyRemap[key] || key;
                recommendation += `<p style="margin-bottom: 10px;"><strong>Best single trend:</strong><br>
                    <span style="color: #00d4aa; font-size: 16px;">${name} (${displayKey})</span> - ${stats.accuracy.toFixed(1)}% accuracy</p>`;
            }
            
            if (data.best_combination) {
                const [key, stats] = data.best_combination;
                recommendation += `<p style="margin-bottom: 10px;"><strong>Best combination:</strong><br>
                    <span style="color: #ffc107; font-size: 16px;">${remapComboKey(key)}</span> - ${stats.accuracy.toFixed(1)}% accuracy</p>`;
            }
            
            // Add config suggestion
            if (data.best_single) {
                const [key, stats] = data.best_single;
                if (stats.accuracy >= 55) {
                    const displayKey = keyRemap[key] || key;
                    recommendation += `<p style="color: #888; font-size: 12px; margin-top: 10px;">
                        💡 Suggested: <code style="background: #333; padding: 2px 5px; border-radius: 3px;">${displayKey}</code>
                    </p>`;
                }
            }
            
            recommendation += `<p style="color: #666; font-size: 11px; margin-top: 10px;">
                Based on ${data.total_trades_analyzed || 0} trades analyzed
            </p>`;
            
            recEl.innerHTML = recommendation || '<div class="no-data">Not enough data</div>';
        }
        
        // Initial load
        updateDashboard();
        
        // Auto-refresh every second
        setInterval(updateDashboard, 1000);
    </script>
</body>
</html>"""
    
    title = html.escape(REGISTRY.dashboard_title)
    html_out = html_content.replace("{{DASHBOARD_TITLE}}", title)
    return HTMLResponse(content=html_out)
