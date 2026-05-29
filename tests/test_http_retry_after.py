"""Retry-After honor + exponential-backoff fallback tests for HttpClient.

Track B-2 piece: when a remote API (EuropePMC, UniProt, etc.) returns HTTP 429
with a `Retry-After` header, HttpClient should sleep for that duration before
retrying (capped at RETRY_AFTER_CEILING_S). When the header is missing or
unparseable, HttpClient falls back to its standard exponential backoff.

The HttpClient transport hook is used to drive synthetic 429 responses
without spinning up a real HTTP server.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Callable
from unittest.mock import patch

import pytest

from vibeproteinagent.adapters._remote.http import (
    RETRY_AFTER_CEILING_S,
    HttpClient,
    HttpError,
    HttpResponse,
    _parse_retry_after,
)


def _make_transport(scripted_responses: list) -> tuple[Callable, dict]:
    """Build a transport callable that yields scripted responses or exceptions.

    Each item in scripted_responses is either:
      - HttpResponse: returned as-is
      - HttpError: raised
      - dict: shorthand `{"status": int, "headers": {...}, "text": "..."}`,
        converted into an HttpResponse
    """
    state = {"call": 0}

    def transport(url: str, headers: dict, timeout: float) -> HttpResponse:
        idx = state["call"]
        state["call"] += 1
        item = scripted_responses[min(idx, len(scripted_responses) - 1)]
        if isinstance(item, HttpError):
            raise item
        if isinstance(item, HttpResponse):
            return item
        return HttpResponse(
            url=url,
            status=item.get("status", 200),
            headers=item.get("headers", {}),
            text=item.get("text", ""),
        )

    return transport, state


def _failing_response(status: int, headers: dict[str, str] | None = None) -> dict:
    return {"status": status, "headers": headers or {}, "text": ""}


# --- _parse_retry_after pure helper --------------------------------------


def test_parse_retry_after_integer_seconds() -> None:
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after("0") == 0.0
    assert _parse_retry_after("  12  ") == 12.0


def test_parse_retry_after_float_seconds() -> None:
    assert _parse_retry_after("2.5") == 2.5


def test_parse_retry_after_negative_rejected() -> None:
    assert _parse_retry_after("-3") is None


def test_parse_retry_after_http_date() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=10)
    header = format_datetime(future, usegmt=True)
    parsed = _parse_retry_after(header)
    assert parsed is not None
    assert 5.0 <= parsed <= 15.0


def test_parse_retry_after_past_http_date_clamped_to_zero() -> None:
    past = datetime.now(timezone.utc) - timedelta(seconds=60)
    header = format_datetime(past, usegmt=True)
    assert _parse_retry_after(header) == 0.0


def test_parse_retry_after_missing_or_garbage() -> None:
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("   ") is None
    assert _parse_retry_after("not-a-thing") is None


# --- HttpClient honors Retry-After on 429 --------------------------------


def test_429_with_retry_after_seconds_overrides_backoff() -> None:
    transport, _ = _make_transport(
        [
            _failing_response(429, {"Retry-After": "3"}),
            _failing_response(200, {}),  # ignored — transport script
        ]
    )
    transport_call_count = {"n": 0}

    def counting_transport(url, headers, timeout):
        transport_call_count["n"] += 1
        if transport_call_count["n"] == 1:
            return HttpResponse(url=url, status=429, headers={"Retry-After": "3"}, text="")
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(transport=counting_transport, retries=2, backoff_s=0.01)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep") as mock_sleep:
        resp = client.get("https://example.test/api")
    assert resp.status == 200
    # First sleep should honor Retry-After (3.0), not exponential backoff (0.01)
    assert mock_sleep.call_count == 1
    slept = mock_sleep.call_args_list[0].args[0]
    assert slept == 3.0


def test_429_retry_after_capped_at_ceiling() -> None:
    def transport_429_then_ok(url, headers, timeout):
        if not getattr(transport_429_then_ok, "_done", False):
            transport_429_then_ok._done = True
            return HttpResponse(url=url, status=429, headers={"Retry-After": "999"}, text="")
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(transport=transport_429_then_ok, retries=2, backoff_s=0.01)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep") as mock_sleep:
        resp = client.get("https://example.test/api")
    assert resp.status == 200
    slept = mock_sleep.call_args_list[0].args[0]
    assert slept == RETRY_AFTER_CEILING_S


def test_429_without_retry_after_falls_back_to_exponential_backoff() -> None:
    def transport_429_then_ok(url, headers, timeout):
        if not getattr(transport_429_then_ok, "_done", False):
            transport_429_then_ok._done = True
            return HttpResponse(url=url, status=429, headers={}, text="")
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(transport=transport_429_then_ok, retries=2, backoff_s=0.25)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep") as mock_sleep:
        resp = client.get("https://example.test/api")
    assert resp.status == 200
    # Exponential backoff: backoff_s * 2^attempt = 0.25 * 1 = 0.25 on first retry
    slept = mock_sleep.call_args_list[0].args[0]
    assert slept == 0.25


def test_429_with_malformed_retry_after_falls_back_to_backoff() -> None:
    def transport_429_then_ok(url, headers, timeout):
        if not getattr(transport_429_then_ok, "_done", False):
            transport_429_then_ok._done = True
            return HttpResponse(
                url=url, status=429, headers={"Retry-After": "not-a-number"}, text=""
            )
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(transport=transport_429_then_ok, retries=2, backoff_s=0.5)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep") as mock_sleep:
        resp = client.get("https://example.test/api")
    assert resp.status == 200
    slept = mock_sleep.call_args_list[0].args[0]
    assert slept == 0.5  # exponential backoff fallback


def test_500_does_not_consult_retry_after() -> None:
    """5xx errors use standard exponential backoff; Retry-After parsing skipped."""

    def transport_500_then_ok(url, headers, timeout):
        if not getattr(transport_500_then_ok, "_done", False):
            transport_500_then_ok._done = True
            # Even if a 500 had Retry-After, current policy is exponential backoff
            return HttpResponse(
                url=url, status=500, headers={"Retry-After": "10"}, text=""
            )
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(transport=transport_500_then_ok, retries=2, backoff_s=0.3)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep") as mock_sleep:
        resp = client.get("https://example.test/api")
    assert resp.status == 200
    slept = mock_sleep.call_args_list[0].args[0]
    assert slept == 0.3  # exponential backoff, not 10s


def test_429_retries_exhausted_raises_with_retry_after_metadata() -> None:
    def always_429(url, headers, timeout):
        return HttpResponse(url=url, status=429, headers={"Retry-After": "2"}, text="")

    client = HttpClient(transport=always_429, retries=1, backoff_s=0.0)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep"):
        with pytest.raises(HttpError) as excinfo:
            client.get("https://example.test/api")
    err = excinfo.value
    assert err.status == 429
    assert err.retry_after_seconds == 2.0


def test_lowercase_retry_after_header_is_honored() -> None:
    """Some servers/transports return lowercase header names. Both work."""

    def transport_lower_then_ok(url, headers, timeout):
        if not getattr(transport_lower_then_ok, "_done", False):
            transport_lower_then_ok._done = True
            return HttpResponse(
                url=url, status=429, headers={"retry-after": "4"}, text=""
            )
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(transport=transport_lower_then_ok, retries=2, backoff_s=0.01)
    with patch("vibeproteinagent.adapters._remote.http.time.sleep") as mock_sleep:
        resp = client.get("https://example.test/api")
    assert resp.status == 200
    slept = mock_sleep.call_args_list[0].args[0]
    assert slept == 4.0


def test_httperror_carries_retry_after_field() -> None:
    err = HttpError("test", status=429, retry_after_seconds=5.0)
    assert err.retry_after_seconds == 5.0


def test_httperror_default_retry_after_none() -> None:
    err = HttpError("test", status=500)
    assert err.retry_after_seconds is None


def test_get_json_invalid_json_retries_and_raises_http_error() -> None:
    state = {"call": 0}

    def invalid_json(url, headers, timeout):
        state["call"] += 1
        return HttpResponse(url=url, status=200, headers={}, text="<html>not json</html>")

    client = HttpClient(transport=invalid_json, retries=1, backoff_s=0.0)
    with pytest.raises(HttpError, match="returned invalid JSON") as excinfo:
        client.get_json("https://example.test/api")

    assert state["call"] == 2
    assert excinfo.value.status == 200


def test_get_json_refetches_after_invalid_cached_json(tmp_path) -> None:
    state = {"call": 0}

    def invalid_then_valid(url, headers, timeout):
        state["call"] += 1
        if state["call"] == 1:
            return HttpResponse(url=url, status=200, headers={}, text="<html>not json</html>")
        return HttpResponse(url=url, status=200, headers={}, text='{"ok": true}')

    client = HttpClient(
        transport=invalid_then_valid,
        retries=1,
        backoff_s=0.0,
        cache_dir=tmp_path,
    )

    assert client.get_json("https://example.test/api") == {"ok": True}
    assert state["call"] == 2
