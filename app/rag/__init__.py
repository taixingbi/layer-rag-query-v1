"""RAG domain: access control, hybrid retrieval, answer pipeline, follow-ups."""

from app.rag.access import RagUser, build_qdrant_access_filter, compact_for_log
from app.rag.follow_up import generate_follow_ups
from app.rag.rag_answer import complete_rag_answer, complete_rag_answer_stream
from app.rag.retrieval import query_chunks

__all__ = [
    "RagUser",
    "build_qdrant_access_filter",
    "compact_for_log",
    "complete_rag_answer",
    "complete_rag_answer_stream",
    "generate_follow_ups",
    "query_chunks",
]
