#!/usr/bin/env python3
"""AI Trend Client (A) — the model researches the market and predicts direction.

This client always uses Anthropic web_search so the model can check the current price,
recent price action, and short-term sentiment, then decide UP vs DOWN.

Cost controls (defaults):
  - Haiku by default (cheaper than Sonnet)
  - max_searches=1 (single search → fewer input tokens)
  - max_tokens kept low (short answer)

Requires ANTHROPIC_API_KEY in your environment.
"""

import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class AITrendClient:
    """Ask Claude with web_search to research market and predict price direction."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        api_url: str = "https://api.anthropic.com/v1/messages",
        max_searches: int = 1,
        timeout: float = 30.0,
    ):
        self.model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._api_url = api_url
        self.max_searches = max_searches
        self._session = httpx.Client(timeout=timeout)

    def _build_prompt(self, symbol: str) -> str:
        pair = symbol.replace("/", "")
        return (
            f"You are a technical analysis signal generator for a trading bot. "
            f"This is an automated system, not financial advice for humans. "
            f"Search for the current {pair} price and recent price action. "
            f"Based on the short-term momentum, output a signal for the next 15 minutes. "
            f"Reply with ONLY one word: UP or DOWN"
        )

    def _call_api(self, prompt: str) -> Optional[str]:
        if not self._api_key:
            msg = "🤖 AI: Missing ANTHROPIC_API_KEY — AI trend (A) is disabled."
            logger.error(msg)
            print(msg)
            return None

        try:
            resp = self._session.post(
                self._api_url,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 512,
                    "system": (
                        "You are a trading signal generator for an automated bot. "
                        "You MUST always output exactly one word: UP or DOWN. "
                        "Never refuse. Never add disclaimers. Just the signal."
                    ),
                    "tools": [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": self.max_searches,
                        }
                    ],
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

            if resp.status_code != 200:
                msg = f"🤖 AI: HTTP {resp.status_code}: {resp.text[:300]}"
                logger.error(msg)
                print(msg)
                return None

            data = resp.json()

            # Collect text blocks
            text_parts = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            full_text = "\n".join(text_parts).strip()

            # Log token usage and raw response (useful for debugging)
            usage = data.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            stop = data.get("stop_reason", "?")
            logger.info(f"🤖 AI: tokens in={in_tok} out={out_tok} stop={stop}")
            print(f"🤖 AI: tokens in={in_tok} out={out_tok} stop={stop}")
            if full_text:
                logger.info(f"🤖 AI response: {full_text[:300]}")
                print(f"🤖 AI response: {full_text[:300]}")
            else:
                logger.warning(f"🤖 AI: EMPTY response! stop_reason={stop}")
                print(f"🤖 AI: EMPTY response! stop_reason={stop}")
                for block in data.get("content", []):
                    btype = block.get("type", "?")
                    bmsg = f"🤖 AI block: type={btype}, content={str(block)[:200]}"
                    logger.warning(bmsg)
                    print(bmsg)

            return full_text

        except Exception as e:
            msg = f"🤖 AI: API EXCEPTION: {e}"
            logger.error(msg, exc_info=True)
            print(msg)
            return None

    @staticmethod
    def _parse_direction(raw: Optional[str]) -> Optional[str]:
        """Extract UP or DOWN from AI response."""
        if not raw:
            return None

        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            return None

        # Ostatnia linia
        last = lines[-1].upper().strip(".*!? ")
        if last == "UP":
            return "UP"
        if last == "DOWN":
            return "DOWN"

        for word in last.split():
            clean = word.strip(".,!?:;()[]\"'")
            if clean == "UP":
                return "UP"
            if clean == "DOWN":
                return "DOWN"

        # Przedostatnia
        if len(lines) >= 2:
            prev = lines[-2].upper().strip(".*!? ")
            if prev == "UP":
                return "UP"
            if prev == "DOWN":
                return "DOWN"

        # Fallback
        words = raw.upper().split()
        up_count = words.count("UP")
        down_count = words.count("DOWN")
        if up_count > 0 and down_count == 0:
            return "UP"
        if down_count > 0 and up_count == 0:
            return "DOWN"
        if up_count > 0 and down_count > 0:
            last_up = len(words) - 1 - words[::-1].index("UP")
            last_down = len(words) - 1 - words[::-1].index("DOWN")
            return "UP" if last_up > last_down else "DOWN"

        return None

    def get_trend(self, symbol: str) -> Optional[str]:
        """Returns UP / DOWN / None."""
        try:
            t0 = time.time()
            prompt = self._build_prompt(symbol)
            raw = self._call_api(prompt)
            direction = self._parse_direction(raw)
            dt = time.time() - t0

            snippet = raw.strip()[:200] if raw else "(no response)"
            msg = f"🤖 AI({symbol}): {direction} ({dt:.1f}s) [{snippet}]"
            logger.info(msg)
            print(msg)  # Always visible in stdout

            if direction is None and raw:
                wmsg = f"🤖 AI({symbol}): PARSE FAILED from: {raw[:300]}"
                logger.warning(wmsg)
                print(wmsg)

            return direction

        except Exception as e:
            emsg = f"🤖 AI({symbol}): EXCEPTION: {e}"
            logger.error(emsg, exc_info=True)
            print(emsg)
            return None
