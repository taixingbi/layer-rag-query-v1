"""Prometheus metrics for HTTP and RAG query handling."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

_HTTP_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
_RAG_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
)

HTTP_REQUESTS_TOTAL = Counter(
    "rag_query_http_requests_total",
    "Total HTTP requests handled by layer-rag-query.",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "rag_query_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "path"),
    buckets=_HTTP_LATENCY_BUCKETS,
)

RAG_QUERY_REQUESTS_TOTAL = Counter(
    "rag_query_requests_total",
    "RAG query invocations (``POST /v1/rag/query``).",
    ("status", "stream"),
)
RAG_QUERY_DURATION_SECONDS = Histogram(
    "rag_query_duration_seconds",
    "Wall time for a completed RAG query (seconds).",
    ("stream",),
    buckets=_RAG_LATENCY_BUCKETS,
)
RAG_PHASE_DURATION_SECONDS = Histogram(
    "rag_query_phase_duration_seconds",
    "Per-phase latency from ``latency_ms`` (seconds).",
    ("phase",),
    buckets=_RAG_LATENCY_BUCKETS,
)
RAG_CACHE_OPS_TOTAL = Counter(
    "rag_cache_ops_total",
    "RAG Redis cache lookups by layer and result.",
    ("layer", "result"),
)


def observe_http(method: str, path: str, status_code: int, latency_s: float) -> None:
    status = str(int(status_code))
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(max(0.0, latency_s))


def observe_cache_op(layer: str, result: str) -> None:
    """Record cache hit, miss, or error (fail-open paths use error sparingly)."""
    RAG_CACHE_OPS_TOTAL.labels(layer=layer, result=result).inc()


def observe_rag_query(
    *,
    status_code: int,
    stream: bool,
    latency_ms: dict[str, int] | None = None,
) -> None:
    """Record RAG-specific counters/histograms after a query completes."""
    status = str(int(status_code))
    stream_label = "true" if stream else "false"
    RAG_QUERY_REQUESTS_TOTAL.labels(status=status, stream=stream_label).inc()
    if latency_ms:
        total_ms = latency_ms.get("total")
        if isinstance(total_ms, (int, float)):
            RAG_QUERY_DURATION_SECONDS.labels(stream=stream_label).observe(
                max(0.0, float(total_ms) / 1000.0)
            )
        for phase, ms in latency_ms.items():
            if phase == "total" or not isinstance(ms, (int, float)):
                continue
            RAG_PHASE_DURATION_SECONDS.labels(phase=phase).observe(
                max(0.0, float(ms) / 1000.0)
            )


def metrics_payload() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
