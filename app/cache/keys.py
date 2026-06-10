"""Redis key builders for HuntAI RAG cache layers."""

from __future__ import annotations

from app.cache.normalize import normalize_query, short_hash
from app.core.config import get_embedding_model, get_env, get_kb_cache_revision


def _prefix() -> str:
    env = get_env()
    return f"huntai:{env}:" if env else "huntai:"


def acl_fingerprint(
    *,
    roles: list[str] | None = None,
    groups: list[str] | None = None,
    teams: list[str] | None = None,
    is_admin: bool = False,
) -> str:
    if is_admin:
        return "admin"
    return short_hash(
        ",".join(sorted(roles or [])),
        ",".join(sorted(groups or [])),
        ",".join(sorted(teams or [])),
    )


def acl_fingerprint_for_user(user: object | None) -> str:
    """Build ACL scope without importing :mod:`app.rag` at module load (avoids cycles)."""
    if user is None:
        return "admin"
    if bool(getattr(user, "is_admin", False)):
        return "admin"
    return acl_fingerprint(
        roles=list(getattr(user, "roles", []) or []),
        groups=list(getattr(user, "groups", []) or []),
        teams=list(getattr(user, "teams", []) or []),
    )


def chunk_ids_fingerprint(chunks: list[dict]) -> str:
    ids = [str(c.get("chunk_id") or "") for c in chunks]
    return short_hash(*ids)


def follow_up_cfg_fingerprint(
    *,
    infer_model: str,
    rerank_model: str,
    follow_up_candidates: int,
    follow_up_final: int,
    min_context_score: float,
) -> str:
    return short_hash(
        infer_model,
        rerank_model,
        str(follow_up_candidates),
        str(follow_up_final),
        f"{min_context_score:.4f}",
    )


def build_embed_key(text: str, *, model: str | None = None) -> str:
    m = model or get_embedding_model()
    qhash = short_hash(normalize_query(text))
    safe_model = m.replace(":", "_").replace("/", "_")
    return f"{_prefix()}emb:{safe_model}:{qhash}"


def build_follow_up_key(
    *,
    collection_base: str,
    question: str,
    answer: str,
    chunks: list[dict],
    acl: str,
    cfg_hash: str,
) -> str:
    qhash = short_hash(normalize_query(question))
    ahash = short_hash((answer or "").strip())
    chash = chunk_ids_fingerprint(chunks)
    kb_rev = get_kb_cache_revision()
    coll = collection_base.strip() or "default"
    return f"{_prefix()}followup:{coll}:{kb_rev}:{acl}:{qhash}:{ahash}:{chash}:{cfg_hash}"


def build_miss_key(
    *,
    collection_base: str,
    question: str,
    chunks: list[dict],
    acl: str,
    cfg_hash: str,
) -> str:
    qhash = short_hash(normalize_query(question))
    chash = chunk_ids_fingerprint(chunks)
    kb_rev = get_kb_cache_revision()
    coll = collection_base.strip() or "default"
    return f"{_prefix()}miss:{coll}:{kb_rev}:{acl}:{qhash}:{chash}:{cfg_hash}"
