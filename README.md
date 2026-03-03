# PolySniper In-Window v4 (Martingale 2.0)

A bot for trading Polymarket 15-minute markets using an **in-window** strategy with a **Martingale 2.0** sizing system.

> ⚠️ Disclaimer: This repository is for educational/research purposes. LIVE trading involves real money and risk.

## How the strategy works

The bot places a BUY **right after** the 15-minute market window starts and tries to SELL for a profit before the window ends.

Example log:

```
10:14:50  ⏳ Next window: 10:15:00 - 10:30:00
         Waiting 10s | Martingale step 0 (1.0x)
10:14:50  🔄 Fetching trends (10s before window)...
10:14:51  📊 Primary ADX: UP
10:14:51  📊 Secondary ADX: UP
10:14:51  📊 MACD: DOWN
10:14:51  📊 VKI: UP
10:14:51  📊 Multi: UP (4/5)
10:14:51  📊 Decision: M=UP → BUY YES
10:14:52  ✅ Ready! Waiting for window start...
10:15:00  🟢 Window STARTED!
10:15:00  🎯 PLACING ORDER: YES @ $0.55 x 5.0 (step 0)
10:15:00  ✅ FILLED: 5.0 YES @ $0.54
10:15:00  📈 Position: 5.0 YES @ $0.5400 | Target: $0.85
...
10:22:15  🎯 Target reached! Bid: $0.87 >= $0.85
10:22:16  ✅ SOLD @ $0.87
10:22:16  🎉 WIN! PnL: $1.65 | Martingale reset to step 0
```

## Martingale 2.0

Sizing is controlled by `martingale.multipliers` in `config.yaml` (independent per asset).

Example for base size `order_size=5.0` and multipliers `[1, 2, 4, 8, 16, 32]`:

| Step | Multiplier | Size example |
|------|------------|--------------|
| 0    | 1x         | 5.0 tokens   |
| 1    | 2x         | 10.0 tokens  |
| 2    | 4x         | 20.0 tokens  |
| 3    | 8x         | 40.0 tokens  |
| 4    | 16x        | 80.0 tokens  |
| 5    | 32x (max)  | 160.0 tokens |

Rules:
- **WIN** (sold at/above target) → reset to step 0
- **LOSS** (window ended without selling) → advance step

## Repository layout

| File | Purpose |
|------|---------|
| `runner.py` | Main bot (in-window + martingale + verification) |
| `dashboard.py` | Web dashboard (FastAPI) |
| `config.yaml` | Bot configuration (strategy + indicators) |
| `polymarket_client.py` | Polymarket CLOB API client wrapper |
| `ws_client.py` | WebSocket price feed |
| `adx_client.py` | ADX trend API client |
| `macd_client.py` | MACD trend calculator |
| `vki_client.py` | VKI trend API client |
| `multi_client.py` | Multi-indicator aggregation (M trend) |
| `lorentzian_client.py` | Lorentzian composite trend (L) |
| `keltner_client.py` | Keltner Channels trend (K) |
| `ai_client.py` | AI trend client (A) via Anthropic + web_search |
| `bot_registry.py` | State/trade registry used by the dashboard |

## Installation

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml`:

- `mode.demo`: `true` for simulation, `false` for LIVE trading
- `strategy.buy_price`, `strategy.sell_target`, `strategy.order_size`
- `decision_trend`: which indicator drives the buy decision (`V`, `M`, `L`, `A`, `VOTE`, etc.)
- indicator sections: `adx`, `macd`, `vki`, `multi`, `lorentzian`, `keltner`, `ai`

## Environment variables

Create a `.env` file (never commit it). See `.env.example`.

Required for LIVE trading:
- `POLYMARKET_PRIVATE_KEY`
- `POLYMARKET_FUNDER_ADDRESS`
- `POLYMARKET_API_KEY`
- `POLYMARKET_API_SECRET`
- `POLYMARKET_API_PASSPHRASE`

Optional:
- `ANTHROPIC_API_KEY` (only if `ai.enabled: true`)

## Running

```bash
# Demo mode
python runner.py

# Live mode
python runner.py --live

# Without dashboard
python runner.py --no-dashboard

# Custom dashboard port
python runner.py --dashboard-port 8080
```

Dashboard: `http://localhost:8050`

## Notes on safety and secrets

- Do **not** log or commit private keys / API credentials.
- Keep `.env` in `.gitignore`.
- Consider running LIVE with a dedicated wallet and strict limits.
