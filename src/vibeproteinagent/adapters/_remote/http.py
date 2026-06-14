from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

USER_AGENT = "VibeProteinAgent/1.0 (+https://github.com/) protein-retrieval"

# Upper bound on how long we honor a server-supplied Retry-After before
# falling back to standard exponential backoff. Prevents bogus or hostile
# values from stalling agent runs.
RETRY_AFTER_CEILING_S = 60.0


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP `Retry-After` header value (RFC 7231).

    Accepts either a non-negative integer/float seconds or an HTTP-date.
    Returns the number of seconds to wait, or ``None`` if absent / unparseable.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        pass
    else:
        return seconds if seconds >= 0 else None
    try:
        retry_dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if retry_dt is None:
        return None
    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=UTC)
    delta = (retry_dt - datetime.now(UTC)).total_seconds()
    return max(0.0, delta)


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    headers: dict[str, str]
    text: str
    from_cache: bool = False

    def json(self) -> Any:
        return json.loads(self.text)


class HttpError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        url: str | None = None,
        retry_after_seconds: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.url = url
        # Server-supplied Retry-After (in seconds), if any. Populated for
        # HTTP 429 responses when the header is present and parseable.
        self.retry_after_seconds = retry_after_seconds


Transport = Callable[[str, dict[str, str], float], HttpResponse]


class HttpClient:
    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        retries: int = 2,
        backoff_s: float = 0.25,
        cache_dir: str | Path | None = None,
        user_agent: str = USER_AGENT,
        transport: Transport | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.user_agent = user_agent
        self.transport = transport
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "method": method.upper(),
            "url": url,
            "body_sha256": hashlib.sha256(body or b"").hexdigest(),
            "headers": headers or {},
            "extra": extra or {},
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        extra_cache_key: dict[str, Any] | None = None,
        cache_namespace: str = "http",
        use_cache: bool = True,
    ) -> Any:
        if params:
            query = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None},
                doseq=True,
            )
            sep = "&" if urllib.parse.urlparse(url).query else "?"
            url = f"{url}{sep}{query}"
        last_exc: json.JSONDecodeError | None = None
        for attempt in range(self.retries + 1):
            response = self.get(
                url,
                extra_cache_key=extra_cache_key,
                cache_namespace=cache_namespace,
                use_cache=use_cache and attempt == 0,
            )
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise HttpError(
                        f"GET {response.url} returned invalid JSON: {exc}",
                        status=response.status,
                        url=response.url,
                    ) from exc
                sleep_s = self.backoff_s * (2 ** attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        raise HttpError(f"GET {url} returned invalid JSON: {last_exc}", url=url)

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        extra_cache_key: dict[str, Any] | None = None,
        cache_namespace: str = "http",
        use_cache: bool = True,
    ) -> HttpResponse:
        if params:
            query = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None},
                doseq=True,
            )
            sep = "&" if urllib.parse.urlparse(url).query else "?"
            url = f"{url}{sep}{query}"
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/plain;q=0.8, */*;q=0.5",
        }
        key = self.cache_key("GET", url, headers=headers, extra=extra_cache_key)
        cache_root = self.cache_dir / cache_namespace if self.cache_dir else None
        cache_path = cache_root / f"{key}.json" if cache_root else None
        if use_cache and cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return HttpResponse(
                url=cached["url"],
                status=cached["status"],
                headers=cached["headers"],
                text=cached["text"],
                from_cache=True,
            )

        last_exc: Exception | None = None
        retry_after_seconds: float | None = None
        for attempt in range(self.retries + 1):
            retry_after_seconds = None
            try:
                if self.transport is not None:
                    out = self.transport(url, headers, self.timeout_s)
                else:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                        text = response.read().decode("utf-8")
                        out = HttpResponse(
                            url=url,
                            status=int(getattr(response, "status", 200)),
                            headers={str(k): str(v) for k, v in response.headers.items()},
                            text=text,
                        )
                if out.status >= 400:
                    if out.status == 429:
                        retry_after_seconds = _parse_retry_after(
                            out.headers.get("Retry-After")
                            or out.headers.get("retry-after")
                        )
                    raise HttpError(
                        f"GET {url} failed with HTTP {out.status}",
                        status=out.status,
                        url=url,
                        retry_after_seconds=retry_after_seconds,
                    )
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(out.__dict__, indent=2, sort_keys=True) + "\n")
                return out
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429:
                    header_value = None
                    if getattr(exc, "headers", None) is not None:
                        header_value = exc.headers.get("Retry-After")
                    retry_after_seconds = _parse_retry_after(header_value)
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise HttpError(
                        f"GET {url} failed with HTTP {exc.code}",
                        status=exc.code,
                        url=url,
                        retry_after_seconds=retry_after_seconds,
                    ) from exc
            except HttpError as exc:
                last_exc = exc
                if retry_after_seconds is None:
                    retry_after_seconds = exc.retry_after_seconds
                retryable = exc.status in {408, 429, 500, 502, 503, 504} or exc.status is None
                if not retryable or attempt >= self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise HttpError(f"GET {url} failed: {exc}", url=url) from exc
            if retry_after_seconds is not None:
                sleep_s = min(retry_after_seconds, RETRY_AFTER_CEILING_S)
            else:
                sleep_s = self.backoff_s * (2 ** attempt)
            if sleep_s > 0:
                time.sleep(sleep_s)
        raise HttpError(f"GET {url} failed: {last_exc}", url=url)
