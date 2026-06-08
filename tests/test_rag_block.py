"""Unit tests for the terminal ``rag`` response block."""

from __future__ import annotations

from app.rag.rag_answer import _RagPrep, _build_rag_block, _estimate_context_tokens


def _minimal_prep(**overrides) -> _RagPrep:
    base = dict(
        infer_base="http://infer",
        model="test-model",
        embed_model="BAAI/bge-m3",
        rerank_base="http://rerank",
        rerank_model="BAAI/bge-reranker-v2-m3",
        max_tokens=512,
        rerank_return_top_k=10,
        retrieve_fallback_n=5,
        final_context_top_k=5,
        follow_up_candidates=8,
        follow_up_final=3,
        system_msg={"role": "system", "content": "sys"},
        chunks_full=[
            {"chunk_id": "a", "source": "doc", "score": 0.05, "text": "hello world"},
            {"chunk_id": "b", "source": "doc2", "score": 0.04, "text": "more text"},
        ],
        candidate_chunks=[
            {"chunk_id": "a", "source": "doc", "rerank_score": 0.98, "text": "hello world"},
        ],
        reranked_for_hits=[
            {"chunk_id": "a", "source": "doc", "rerank_score": 0.98, "text": "hello world"},
        ],
        embed_ms=10,
        retrieve_ms=20,
        chunk_rerank_ms=30,
        initial_k=1,
    )
    base.update(overrides)
    return _RagPrep(**base)


def test_estimate_context_tokens():
    chunks = [{"text": "abcd" * 10}]
    assert _estimate_context_tokens(chunks) == 10


def test_build_rag_block_shape():
    prep = _minimal_prep()
    context = prep.candidate_chunks[:1]
    block = _build_rag_block(
        question="What is the visa status?",
        collection_base="taixing_knowledge",
        prep=prep,
        context_chunks=context,
        context_k=1,
        k_max=40,
        use_reranker=True,
    )
    assert block["collection"] == "taixing_knowledge"
    assert block["query"]["original"] == "What is the visa status?"
    retrieval = block["retrieval"]
    assert retrieval["embed_model"] == "BAAI/bge-m3"
    assert retrieval["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    assert retrieval["top_k"] == 40
    assert retrieval["retrieved_chunks"] == 2
    assert retrieval["reranked_chunks"] == 1
    assert retrieval["context_chunks"] == 1
    assert retrieval["context_tokens"] > 0
    assert retrieval["top_score"] == 0.98
    assert retrieval["confidence"] == "high"
    assert len(block["sources"]) == 1
    assert block["sources"][0]["chunk_id"] == "a"
