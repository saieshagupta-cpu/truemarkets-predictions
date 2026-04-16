"""
TrueMarkets REST API client using curl_cffi (Chrome impersonation for Cloudflare).

Auth: OTP → access_token. Tokens persisted in backend/tm_tokens.json.
Env var overrides: TM_ACCESS_TOKEN, TM_REFRESH_TOKEN.

Discovered endpoints:
  GET /v1/defi/core/profile/me
  GET /v1/conductor/assets                  → full asset list incl. BTC metadata
  GET /v1/defi/market/prices?symbol=BTC     → current-interval candles (30s/1h/24h)
  GET /v1/defi/market/prices/history        → time-series
      ?symbol=BTC&window={5m|1h|1d|7d|1M|3M|6M|365d}&resolution={5s|5m|1h|1d}
"""

import json
import os
import time
import logging
from typing import Any

from curl_cffi import requests

logger = logging.getLogger("truemarkets.api")

API_HOST = os.getenv("TRUEMARKETS_API_BASE", "https://api.truemarkets.co")
API_VERSION = os.getenv("TRUEMARKETS_API_VERSION", "2026-01-26")
USER_AGENT = "tm/0.0.11"

TOKEN_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "tm_tokens.json")


class TMApiClient:
    def __init__(self):
        self._session = requests.Session(impersonate="chrome")
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        })
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0
        self._load_tokens()

    def _load_tokens(self):
        at = os.getenv("TM_ACCESS_TOKEN")
        rt = os.getenv("TM_REFRESH_TOKEN")
        if at:
            self._access_token = at
            self._refresh_token = rt
            self._expires_at = time.time() + 3600
            return
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE) as f:
                    data = json.load(f)
                self._access_token = data.get("access_token")
                self._refresh_token = data.get("refresh_token")
                # expires_in from OTP flow is an ISO timestamp, not seconds
                exp = data.get("expires_in")
                if isinstance(exp, str):
                    try:
                        import datetime
                        dt = datetime.datetime.fromisoformat(exp.replace("Z", "+00:00"))
                        self._expires_at = dt.timestamp()
                    except Exception:
                        self._expires_at = time.time() + 3600
                elif isinstance(exp, (int, float)):
                    saved_at = data.get("saved_at") or int(time.time())
                    self._expires_at = saved_at + exp
                else:
                    self._expires_at = time.time() + 3600
            except Exception as e:
                logger.warning(f"Failed to load tokens: {e}")

    def _save_tokens(self, data: dict):
        data["saved_at"] = int(time.time())
        try:
            with open(TOKEN_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save tokens: {e}")

    def _refresh_access_token(self) -> bool:
        """POST /v1/auth/token/refresh. Refresh tokens are single-use — the response
        returns a new refresh_token which we persist back."""
        if not self._refresh_token:
            return False
        try:
            r = self._session.post(
                f"{API_HOST}/v1/auth/token/refresh",
                json={"refresh_token": self._refresh_token},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if r.status_code == 200:
                tokens = r.json()
                self._access_token = tokens.get("access_token") or self._access_token
                self._refresh_token = tokens.get("refresh_token") or self._refresh_token
                exp = tokens.get("expires_in")
                if isinstance(exp, (int, float)):
                    self._expires_at = time.time() + exp
                elif isinstance(exp, str):
                    try:
                        import datetime
                        self._expires_at = datetime.datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        self._expires_at = time.time() + 3600
                else:
                    self._expires_at = time.time() + 3600
                self._save_tokens(tokens)
                logger.info(f"Refreshed access token, new expiry: {self._expires_at}")
                return True
            else:
                logger.error(f"Refresh failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.error(f"Refresh exception: {e}")
        return False

    def _request(self, method: str, path: str, **kwargs) -> Any:
        if not self._access_token:
            raise RuntimeError("No access token — run authenticate.py")

        # Proactive refresh 5 min before expiry (access token lives ~1 hour)
        if self._expires_at and time.time() > self._expires_at - 300:
            self._refresh_access_token()

        headers = kwargs.pop("headers", {}) or {}
        headers["Authorization"] = f"Bearer {self._access_token}"
        params = kwargs.pop("params", {}) or {}
        params.setdefault("version", API_VERSION)

        r = self._session.request(method, f"{API_HOST}{path}", headers=headers, params=params, timeout=15, **kwargs)
        if r.status_code == 401 and self._refresh_access_token():
            headers["Authorization"] = f"Bearer {self._access_token}"
            r = self._session.request(method, f"{API_HOST}{path}", headers=headers, params=params, timeout=15, **kwargs)
        r.raise_for_status()
        return r.json()

    def get(self, path: str, **kwargs) -> Any:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> Any:
        return self._request("POST", path, **kwargs)


_client: TMApiClient | None = None


def _get_client() -> TMApiClient:
    global _client
    if _client is None:
        _client = TMApiClient()
    return _client


# ─── Market data helpers ─────────────────────────────────────────────


def fetch_price_history(symbol: str = "BTC", window: str = "1d", resolution: str = "1h") -> dict:
    """Returns: {symbol, window, resolution, points: [{t, price}, ...]}"""
    return _get_client().get(
        "/v1/defi/market/prices/history",
        params={"symbol": symbol, "window": window, "resolution": resolution},
    )


def fetch_prices(symbol: str = "BTC") -> dict:
    """Returns candles at multiple intervals: {symbol, candles: [{interval, openPrice, closePrice, highPrice, lowPrice}, ...]}"""
    return _get_client().get("/v1/defi/market/prices", params={"symbol": symbol})


def fetch_conductor_assets() -> list:
    """Full asset list with metadata (icon, description, supply, etc.)."""
    resp = _get_client().get("/v1/conductor/assets")
    return resp.get("data", []) if isinstance(resp, dict) else resp
