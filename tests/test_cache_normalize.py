"""Unit tests for cache key normalization."""

from __future__ import annotations

from app.cache.keys import acl_fingerprint, acl_fingerprint_for_user, build_embed_key, chunk_ids_fingerprint
from app.cache.normalize import normalize_query, short_hash
from app.rag.access import RagUser


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  What   Visa?  ") == "what visa?"


def test_short_hash_stable():
    assert short_hash("a", "b") == short_hash("a", "b")
    assert short_hash("a", "b") != short_hash("a", "c")


def test_build_embed_key_includes_model():
    k1 = build_embed_key("hello", model="BAAI/bge-m3")
    k2 = build_embed_key("hello", model="other-model")
    assert k1 != k2
    assert "emb:" in k1


def test_acl_fingerprint_admin():
    assert acl_fingerprint_for_user(RagUser(roles=["admin"])) == "admin"
    assert acl_fingerprint(is_admin=True) == "admin"


def test_chunk_ids_fingerprint_order_sensitive():
    a = [{"chunk_id": "1"}, {"chunk_id": "2"}]
    b = [{"chunk_id": "2"}, {"chunk_id": "1"}]
    assert chunk_ids_fingerprint(a) != chunk_ids_fingerprint(b)
