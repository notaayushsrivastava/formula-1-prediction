"""Shared JSON-over-HTTP helper for the newer clients (OpenF1, Open-Meteo).

Same rules as the Jolpica client: a client-side rate limiter, a disk cache,
exponential backoff on 429/5xx, and empty answers are never cached.
"""
from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlencode

import requests

from ..cache import DiskCache
from ..errors import AccessBlockedError, SourceRequestError
from .jolpica import RateLimiter

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
USER_AGENT = "pitwall-f1/2.0 (educational F1 analysis library)"


class JsonHttp:
    """GET JSON with rate limiting, caching and polite retries.

    404 means "no data" and returns None. 401/403 are refusals: one attempt, a
    clear error, and never a workaround.
    """

    def __init__(self, name: str, limiter: RateLimiter, cache: DiskCache | None = None,
                 timeout: float = 30.0, max_retries: int = 3) -> None:
        self.name = name
        self.limiter = limiter
        self.cache = cache or DiskCache()
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT

    def get(self, url: str, params: dict[str, Any], max_age: float | None = None) -> Any:
        """JSON for url + params, from the cache when fresh (max_age in s; None = forever)."""
        key = f"{url}?{urlencode(sorted(params.items()))}"
        cached = self.cache.get(key, max_age=max_age)
        if cached is not None:
            return cached
        data = self._download(url, params)
        if data:  # never cache "no data (yet)"
            self.cache.set(key, data)
        return data

    def _download(self, url: str, params: dict[str, Any]) -> Any:
        """One GET with up to max_retries retries (2 s, 4 s, 8 s ...) on temporary errors."""
        attempts = self.max_retries + 1
        delay = 2.0
        for attempt in range(1, attempts + 1):
            self.limiter.wait()
            log.info("Downloading %s %s", url, params)
            try:
                resp = self.http.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                problem = type(exc).__name__
            else:
                if resp.ok:
                    return resp.json()
                if resp.status_code == 404:
                    return None
                self._raise_if_final(resp)
                problem = f"HTTP {resp.status_code}"
            if attempt == attempts:
                break
            log.warning("%s request failed (%s); retry %d in %.0f s",
                        self.name, problem, attempt, delay)
            time.sleep(delay)
            delay *= 2
        raise SourceRequestError(f"{self.name}: giving up on {url} after {attempts} attempts")

    def _raise_if_final(self, resp: requests.Response) -> None:
        """Raise for errors that retrying cannot fix."""
        status = resp.status_code
        if status in (401, 403):
            raise AccessBlockedError(
                f"{self.name} refused the request (HTTP {status}). pitwall only uses the "
                "free public data and never works around a refusal.")
        if status not in RETRY_STATUSES:
            raise SourceRequestError(f"{self.name}: HTTP {status} for {resp.url}")
