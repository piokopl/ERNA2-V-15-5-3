import time
from typing import Optional

import requests


def trend_to_side(trend: Optional[str]) -> Optional[str]:
    """Map a trend label to the Polymarket market side.

    - UP / LONG / BULLISH    -> YES
    - DOWN / SHORT / BEARISH -> NO
    - NEUTRAL / NONE / FLAT  -> None
    """
    if not trend:
        return None
    t = str(trend).strip().upper()
    if t in {"UP", "LONG", "BULL", "BULLISH"}:
        return "YES"
    if t in {"DOWN", "SHORT", "BEAR", "BEARISH"}:
        return "NO"
    if t in {"NO", "NONE", "NEUTRAL", "FLAT"}:
        return None
    return None


class ADXClient:
    """HTTP client that fetches a trend value from an ADX endpoint.

    Compatible with alternative config field names:
      - api_url / url
      - timeout
      - retries / retry_attempts
      - retry_sleep_s / retry_sleep
    """

    def __init__(
        self,
        api_url: str | None = None,
        url: str | None = None,
        timeout: int = 10,
        retries: int | None = None,
        retry_attempts: int | None = None,
        retry_sleep_s: float | None = None,
        retry_sleep: float | None = None,
        **_extra,
    ):
        # Support alternative names
        self.api_url = api_url or url
        if not self.api_url:
            raise ValueError("ADXClient requires api_url (or url)")

        self.timeout = int(timeout) if timeout is not None else 10

        if retries is None:
            retries = retry_attempts
        self.retries = int(retries) if retries is not None else 3

        if retry_sleep_s is None:
            retry_sleep_s = retry_sleep
        self.retry_sleep_s = float(retry_sleep_s) if retry_sleep_s is not None else 1.0

    def get_trend(self, symbol: str) -> Optional[str]:
        """Return a trend value for a symbol (e.g. "BTC/USDT").

        Supported JSON formats:
          1) {"BTC/USDT": "UP", ...}
          2) {"BTC/USDT": {"trend": "UP"}, ...}
          3) {"BTC/USDT": {"signal": "UP"}, ...}
        """
        for _ in range(max(1, self.retries)):
            try:
                resp = requests.get(self.api_url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                if not isinstance(data, dict):
                    return None
                if symbol not in data:
                    return None

                v = data[symbol]
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    for key in ("trend", "signal", "direction", "value"):
                        vv = v.get(key)
                        if isinstance(vv, str):
                            return vv
                    return None
                return None

            except Exception:
                time.sleep(self.retry_sleep_s)

        return None

    def get_btc_trend(self) -> Optional[str]:
        return self.get_trend("BTC/USDT")
