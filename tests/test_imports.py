"""CI smoke: package imports without external services."""

from __future__ import annotations


def test_import_public_api() -> None:
    from app import embed_text, embed_texts, query_chunks

    assert callable(query_chunks)
    assert callable(embed_text)
    assert callable(embed_texts)


def test_import_mcp_entry() -> None:
    import fastmcp
    from app.main import MCP_HTTP_PATH, mcp

    assert mcp.name == "layer-rag-query"
    assert MCP_HTTP_PATH == "/v1/mcp"
    assert fastmcp.settings.streamable_http_path == "/v1/mcp"


def test_mcp_rag_tool_names() -> None:
    from app.main import answer_from_inference, mcp, rag_query, rag_query_stream

    assert callable(rag_query)
    assert callable(rag_query_stream)
    assert callable(answer_from_inference)
    assert mcp.name == "layer-rag-query"


def test_version_and_metrics_helpers() -> None:
    from app.core.metrics import metrics_content_type, metrics_payload
    from app.core.version import APP_NAME, get_app_version

    assert APP_NAME == "layer-rag-query"
    assert get_app_version()
    body = metrics_payload()
    assert b"rag_query_http_requests_total" in body
    assert "text/plain" in metrics_content_type()
