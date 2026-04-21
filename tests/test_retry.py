"""Tests for retry classification."""

import pytest

from mini_agent.retry import RetryConfig, async_retry


class FakeBadRequestError(Exception):
    status_code = 400


@pytest.mark.asyncio
async def test_async_retry_does_not_retry_invalid_request_errors():
    """Deterministic 4xx request errors should fail fast without retries."""
    calls = 0

    @async_retry(RetryConfig(enabled=True, max_retries=3))
    async def call_api():
        nonlocal calls
        calls += 1
        raise FakeBadRequestError("invalid params, context window exceeds limit (2013)")

    with pytest.raises(FakeBadRequestError):
        await call_api()

    assert calls == 1