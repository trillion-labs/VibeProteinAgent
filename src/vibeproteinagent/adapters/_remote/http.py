from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_AGENT = "VibeProteinAgent/1.0 (+https://github.com/) protein-retrieval"


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
    def __init__(self, message: str, *, status: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.url = url


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
        return self.get(
            url,
            extra_cache_key=extra_cache_key,
            cache_namespace=cache_namespace,
            use_cache=use_cache,
        ).json()

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
        for attempt in range(self.retries + 1):
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
                    raise HttpError(
                        f"GET {url} failed with HTTP {out.status}",
                        status=out.status,
                        url=url,
                    )
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(out.__dict__, indent=2, sort_keys=True) + "\n")
                return out
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise HttpError(
                        f"GET {url} failed with HTTP {exc.code}",
                        status=exc.code,
                        url=url,
                    ) from exc
            except HttpError as exc:
                last_exc = exc
                retryable = exc.status in {408, 429, 500, 502, 503, 504} or exc.status is None
                if not retryable or attempt >= self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise HttpError(f"GET {url} failed: {exc}", url=url) from exc
            time.sleep(self.backoff_s * (2 ** attempt))
        raise HttpError(f"GET {url} failed: {last_exc}", url=url)
