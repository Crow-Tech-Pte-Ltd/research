import httpx
import pytest

from llm_arbitrary_choice_study.openrouter import (
    OpenRouterAuthError,
    OpenRouterBudgetError,
    OpenRouterRateLimitError,
    complete,
    get_credits,
    redact_secrets,
)


@pytest.mark.asyncio
async def test_complete_sends_key_only_in_headers(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "blue"}}]})

    request_json, response_json = await complete(
        model="provider/model",
        prompt="Choose one: blue or red.",
        temperature=0.7,
        max_tokens=8,
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    assert seen_requests[0].headers["authorization"] == "Bearer completion-secret-for-tests"
    assert "completion-secret-for-tests" not in str(request_json)
    assert response_json["choices"][0]["message"]["content"] == "blue"


@pytest.mark.asyncio
async def test_complete_maps_402_to_budget_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"code": 402, "message": "insufficient credits"}})

    with pytest.raises(OpenRouterBudgetError):
        await complete(
            model="provider/model",
            prompt="Choose one: blue or red.",
            temperature=0.7,
            max_tokens=8,
            timeout_seconds=1,
            rate_limit_max_retries=0,
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_complete_retries_429_with_retry_after(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"code": 429, "message": "slow down"}}),
        httpx.Response(200, json={"choices": [{"message": {"content": "red"}}]}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    _, response_json = await complete(
        model="provider/model",
        prompt="Choose one: blue or red.",
        temperature=0.7,
        max_tokens=8,
        timeout_seconds=1,
        rate_limit_max_retries=1,
        transport=httpx.MockTransport(handler),
    )

    assert response_json["choices"][0]["message"]["content"] == "red"
    assert responses == []


@pytest.mark.asyncio
async def test_complete_raises_rate_limit_after_bounded_retries(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": 429, "message": "slow down"}})

    with pytest.raises(OpenRouterRateLimitError):
        await complete(
            model="provider/model",
            prompt="Choose one: blue or red.",
            temperature=0.7,
            max_tokens=8,
            timeout_seconds=1,
            rate_limit_max_retries=0,
            transport=httpx.MockTransport(handler),
        )


def test_redact_secrets_removes_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "management-secret-for-tests")

    assert redact_secrets("Authorization: Bearer completion-secret-for-tests") == "Authorization: Bearer [redacted]"
    assert redact_secrets("key=management-secret-for-tests") == "key=[redacted]"


@pytest.mark.asyncio
async def test_get_credits_sends_management_key_only_in_headers(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "management-secret-for-tests")
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"data": {"total_credits": 10.0, "total_usage": 2.5}},
        )

    snapshot = await get_credits(timeout_seconds=1, transport=httpx.MockTransport(handler))

    assert seen_requests[0].method == "GET"
    assert seen_requests[0].url.path == "/api/v1/credits"
    assert seen_requests[0].headers["authorization"] == "Bearer management-secret-for-tests"
    assert seen_requests[0].content == b""
    assert snapshot.remaining_credits == 7.5
    assert "management-secret-for-tests" not in str(snapshot.as_public_dict())


@pytest.mark.asyncio
async def test_get_credits_falls_back_to_completion_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "completion-secret-for-tests")
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "")
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"data": {"total_credits": 3.0, "total_usage": 1.0}},
        )

    snapshot = await get_credits(timeout_seconds=1, transport=httpx.MockTransport(handler))

    assert seen_requests[0].headers["authorization"] == "Bearer completion-secret-for-tests"
    assert snapshot.remaining_credits == 2.0


@pytest.mark.asyncio
async def test_get_credits_maps_403_to_auth_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "management-secret-for-tests")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"code": 403, "message": "management key required"}},
        )

    with pytest.raises(OpenRouterAuthError):
        await get_credits(timeout_seconds=1, transport=httpx.MockTransport(handler))
