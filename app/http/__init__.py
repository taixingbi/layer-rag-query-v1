"""HTTP clients for embedding, chat-completions, and rerank APIs."""

from app.http.embed import embed_text, embed_texts
from app.http.inference import ChatCompletionResult, chat_complete, chat_complete_collect, resolve_conversation_id
from app.http.rerank import rerank_texts
from app.http.usage import UsageTokens, build_usage_payload, merge_usage

__all__ = [
    "ChatCompletionResult",
    "UsageTokens",
    "build_usage_payload",
    "embed_text",
    "embed_texts",
    "chat_complete",
    "chat_complete_collect",
    "merge_usage",
    "resolve_conversation_id",
    "rerank_texts",
]
