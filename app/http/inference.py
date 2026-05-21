"""OpenAI-compatible /v1/chat/completions HTTP client (async)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass

import httpx

from app.http._correlation import correlation_headers
from app.http.usage import UsageTokens, parse_usage
from app.logging_config import logger


def resolve_conversation_id(raw: str | None) -> str:
    """Thread id for chat completions (OpenAI-compatible extension used by inference gateways).

    Omitted or blank values get a fresh ``conv_<hex>`` id, matching
    ``layer-gateway-inference-v1`` behavior.
    """
    s = (raw or "").strip()
    return s if s else f"conv_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    usage: UsageTokens | None


async def chat_complete(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    timeout: float = 60.0,
) -> ChatCompletionResult:
    """Return assistant content and optional token usage from one chat completion call.

    Correlation forwarded as ``X-Request-Id`` / ``X-Session-Id`` / ``X-Trace-Id`` (last only
    when set). When ``conversation_id`` is non-empty after strip, it is sent in the JSON body.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    cid = (conversation_id or "").strip()
    if cid:
        payload["conversation_id"] = cid
    headers = {
        "Content-Type": "application/json",
        **correlation_headers(request_id, session_id, trace_id=trace_id),
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError) as e:
        raise RuntimeError(f"Unexpected chat response shape: {data!r}") from e

    reply = content if isinstance(content, str) else str(content)
    usage = parse_usage(data)
    logger.info(
        "chat_complete ok url=%s model=%s max_tokens=%s reply_chars=%s",
        url,
        model,
        max_tokens,
        len(reply),
    )
    return ChatCompletionResult(content=reply, usage=usage)


async def chat_complete_collect(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    timeout: float = 60.0,
) -> ChatCompletionResult:
    """Buffer a streaming chat completion; return full text and usage from the final chunk.

    Requests ``stream_options.include_usage`` so gateways that support OpenAI streaming
    usage emit token counts on the last SSE frame.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    cid = (conversation_id or "").strip()
    if cid:
        payload["conversation_id"] = cid
    headers = {
        "Content-Type": "application/json",
        **correlation_headers(request_id, session_id, trace_id=trace_id),
    }
    t0 = time.perf_counter()
    ttft_ms: int | None = None
    char_count = 0
    buf: list[str] = []
    usage: UsageTokens | None = None
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    body = line[6:]
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    chunk_usage = parse_usage(chunk)
                    if chunk_usage:
                        usage = chunk_usage
                    choices = chunk.get("choices") or [{}]
                    delta = (choices[0] or {}).get("delta", {}).get("content") or ""
                    if not delta:
                        continue
                    if ttft_ms is None:
                        ttft_ms = int(round((time.perf_counter() - t0) * 1000))
                    char_count += len(delta)
                    buf.append(delta)
    except asyncio.CancelledError:
        cancel_ms = int(round((time.perf_counter() - t0) * 1000))
        cancel_ttft = ttft_ms if ttft_ms is not None else 0
        logger.warning(
            "chat_complete_stream cancelled url=%s model=%s max_tokens=%s "
            "reply_chars=%s ttft_ms=%s gen_ms=%s",
            url,
            model,
            max_tokens,
            char_count,
            cancel_ttft,
            cancel_ms,
            extra={
                "ttft_ms": cancel_ttft,
                "gen_ms": cancel_ms,
                "reason": "client_cancelled",
            },
        )
        raise
    gen_ms = int(round((time.perf_counter() - t0) * 1000))
    final_ttft = ttft_ms if ttft_ms is not None else 0
    logger.info(
        "chat_complete_stream ok url=%s model=%s max_tokens=%s reply_chars=%s ttft_ms=%s gen_ms=%s",
        url,
        model,
        max_tokens,
        char_count,
        final_ttft,
        gen_ms,
        extra={"ttft_ms": final_ttft, "gen_ms": gen_ms},
    )
    return ChatCompletionResult(content="".join(buf), usage=usage)
