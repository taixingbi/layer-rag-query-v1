"""Infrastructure: config, logging, request context, asyncio helpers."""

from app.core.asyncio_util import run_async
from app.core.config import (
    VECTOR_SIZE,
    get_embedding_model,
    get_embedding_url,
    get_inference_max_tokens,
    get_inference_model,
    get_inference_url,
    get_qdrant_api_key,
    get_qdrant_url,
)
from app.core.logging_config import logger, setup_logging
from app.core.metrics import metrics_content_type, metrics_payload, observe_http, observe_rag_query
from app.core.version import APP_NAME, get_app_version
from app.core.request_context import (
    bind_http_context,
    bind_request_context,
    get_conversation_id,
    get_http_method,
    get_http_path,
    get_http_status,
    get_request_id,
    get_session_id,
    get_trace_id,
    get_user_id,
)

__all__ = [
    "APP_NAME",
    "VECTOR_SIZE",
    "bind_http_context",
    "bind_request_context",
    "get_conversation_id",
    "get_embedding_model",
    "get_embedding_url",
    "get_http_method",
    "get_http_path",
    "get_http_status",
    "get_inference_max_tokens",
    "get_inference_model",
    "get_inference_url",
    "get_qdrant_api_key",
    "get_app_version",
    "get_qdrant_url",
    "get_request_id",
    "get_session_id",
    "get_trace_id",
    "get_user_id",
    "logger",
    "metrics_content_type",
    "metrics_payload",
    "observe_http",
    "observe_rag_query",
    "run_async",
    "setup_logging",
]
