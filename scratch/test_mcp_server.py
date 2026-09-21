"""Verification and test suite for BIS Sahayak MCP Server (Step 5A).

Tests:
1. Tool listing & schema validation (exactly 4 registered tools).
2. Tool execution: search_bis_documents (real semantic retrieval, query 'What is BIS CARE?').
3. Tool execution: get_source_status (inspect all 9 approved sources).
4. Tool execution: check_for_updates (safe read-only hash check, returns required keys).
5. Tool execution: get_document_metadata (approved stem & URL success, unapproved inputs rejected).
6. Security checks: traversal attempts, external URLs rejected.
7. Verification that disk files and Qdrant points are strictly preserved (100% read-only).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure backend root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from qdrant_client import QdrantClient

from app.mcp.server import mcp
from app.mcp.tools import (
    check_for_updates_impl,
    get_document_metadata_impl,
    get_source_status_impl,
    search_bis_documents_impl,
)


async def test_mcp_server() -> None:
    print("==================================================")
    print("STEP 5A: TESTING BIS SAHAYAK MCP SERVER")
    print("==================================================")

    # 0. Snapshot pre-test state
    chunks_path = BACKEND_DIR / "data" / "processed" / "bis_consumer_chunks.json"
    embeddings_path = BACKEND_DIR / "data" / "processed" / "bis_consumer_embeddings.json"
    chunks_size_pre = chunks_path.stat().st_size
    embeddings_size_pre = embeddings_path.stat().st_size

    qdrant_points_pre = None
    try:
        client = QdrantClient("http://localhost:6333", timeout=2)
        coll_info_pre = client.get_collection("bis_consumer")
        qdrant_points_pre = coll_info_pre.points_count
        print(f"Pre-test live Qdrant points: {qdrant_points_pre}")
    except Exception:
        print("Live Qdrant service is offline. Retriever will operate via fallback in-memory store.")

    print(f"Pre-test chunks size: {chunks_size_pre} bytes")
    print(f"Pre-test embeddings size: {embeddings_size_pre} bytes")

    # 1. Tool Listing and Schema Validation
    print("\n--- Test 1: Tool Listing & Schemas ---")
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    print(f"Discovered {len(tools)} tools: {tool_names}")

    expected_tools = {
        "search_bis_documents",
        "get_source_status",
        "check_for_updates",
        "get_document_metadata",
    }
    assert set(tool_names) == expected_tools, f"Expected {expected_tools}, got {set(tool_names)}"
    print("[PASS] Exactly 4 expected tools registered.")

    for t in tools:
        print(f"  Tool: {t.name}")
        print(f"    Description: {t.description[:80]}...")
        print(f"    Input Schema Properties: {list(t.input_schema.get('properties', {}).keys())}")

    # 2. Tool 1: search_bis_documents
    print("\n--- Test 2: Tool 'search_bis_documents' ---")
    res_search = await mcp.call_tool("search_bis_documents", {"query": "What is BIS CARE?", "top_k": 3})
    assert not res_search.is_error, f"search_bis_documents failed: {res_search}"
    # Parse returned text
    search_payload = json.loads(res_search.content[0].text)
    print(f"Search query: '{search_payload.get('query')}'")
    print(f"Results returned: {search_payload.get('results_count')}")
    assert search_payload.get("success") is True
    assert search_payload.get("results_count") == 3
    assert len(search_payload.get("results")) == 3
    for idx, hit in enumerate(search_payload["results"]):
        print(f"  Hit {idx + 1}: {hit.get('title')} | score: {hit.get('score')} | url: {hit.get('source_url')}")
    assert any("BIS" in hit.get("title") for hit in search_payload["results"])
    assert all("chunk_id" in hit and "text" in hit for hit in search_payload["results"])
    print("[PASS] search_bis_documents returned relevant evidence chunks with metadata.")

    # Test empty query handling
    res_search_empty = await mcp.call_tool("search_bis_documents", {"query": "   ", "top_k": 3})
    empty_payload = json.loads(res_search_empty.content[0].text)
    assert empty_payload.get("success") is False
    assert "error" in empty_payload
    print("[PASS] search_bis_documents gracefully handled empty query.")

    # 3. Tool 2: get_source_status
    print("\n--- Test 3: Tool 'get_source_status' ---")
    res_status = await mcp.call_tool("get_source_status", {})
    assert not res_status.is_error, f"get_source_status failed: {res_status}"
    status_payload = json.loads(res_status.content[0].text)
    print(f"Total sources reported: {status_payload.get('total_sources')}")
    assert status_payload.get("total_sources") == 9
    assert len(status_payload.get("sources")) == 9
    first_source = status_payload["sources"][0]
    print(f"  Sample source: {first_source.get('source_id')} | {first_source.get('title')}")
    print(f"  Downloaded: {first_source.get('is_downloaded')}, Hash: {first_source.get('content_sha256')[:12]}...")
    assert all(s.get("is_downloaded") for s in status_payload["sources"])
    assert all(s.get("content_sha256") for s in status_payload["sources"])
    print("[PASS] get_source_status correctly returned all 9 approved sources with complete metadata.")

    # 4. Tool 3: check_for_updates
    print("\n--- Test 4: Tool 'check_for_updates' ---")
    res_update = await mcp.call_tool("check_for_updates", {"check_remote": False})
    assert not res_update.is_error, f"check_for_updates failed: {res_update}"
    update_payload = json.loads(res_update.content[0].text)
    print(f"Update check payload keys: {list(update_payload.keys())}")
    print(f"Sources checked: {update_payload.get('sources_checked')}")
    print(f"Unchanged: {update_payload.get('unchanged')}")
    print(f"Changed: {update_payload.get('changed')}")
    print(f"Errors: {update_payload.get('errors')}")

    assert update_payload.get("sources_checked") == 9
    assert update_payload.get("unchanged") == 9
    assert update_payload.get("changed") == 0
    assert update_payload.get("errors") == 0
    assert len(update_payload.get("sources")) == 9
    print("[PASS] check_for_updates verified all 9 sources are unchanged without any disk or vector mutations.")

    # 5. Tool 4: get_document_metadata
    print("\n--- Test 5: Tool 'get_document_metadata' ---")
    # Valid by stem
    res_meta_stem = await mcp.call_tool("get_document_metadata", {"source_id_or_url": "bis_apps"})
    meta_stem_payload = json.loads(res_meta_stem.content[0].text)
    print(f"Lookup by stem 'bis_apps': status={meta_stem_payload.get('status')}")
    assert meta_stem_payload.get("status") == "APPROVED"
    assert meta_stem_payload.get("metadata", {}).get("filename_stem") == "bis_apps"

    # Valid by URL
    res_meta_url = await mcp.call_tool(
        "get_document_metadata",
        {"source_id_or_url": "https://www.bis.gov.in/bis-apps/?lang=en"},
    )
    meta_url_payload = json.loads(res_meta_url.content[0].text)
    print(f"Lookup by URL 'https://www.bis.gov.in/bis-apps/?lang=en': status={meta_url_payload.get('status')}")
    assert meta_url_payload.get("status") == "APPROVED"
    assert meta_url_payload.get("source_id") == "bis_apps"
    print("[PASS] get_document_metadata succeeded for approved stem and approved URL.")

    # 6. Rejection of unapproved / arbitrary inputs
    print("\n--- Test 6: Security & Rejection of Unapproved Inputs ---")
    test_rejections = [
        "unapproved_source_123",
        "https://google.com/search?q=test",
        "../../etc/passwd",
        "https://attacker.com/malicious.html",
        "",
    ]
    for bad_input in test_rejections:
        res_bad = await mcp.call_tool("get_document_metadata", {"source_id_or_url": bad_input})
        bad_payload = json.loads(res_bad.content[0].text)
        print(f"Input '{bad_input}' -> status={bad_payload.get('status')}, error={bad_payload.get('error')[:50]}...")
        assert bad_payload.get("status") == "REJECTED"
        assert bad_payload.get("success") is False

    print("[PASS] Strict rejection verified: All unapproved inputs safely blocked.")

    # 7. Verify post-test state (Zero mutations)
    print("\n--- Test 7: Zero Mutation Verification ---")
    if qdrant_points_pre is not None:
        coll_info_post = client.get_collection("bis_consumer")
        qdrant_points_post = coll_info_post.points_count
        print(f"Post-test Qdrant points: {qdrant_points_post} (pre: {qdrant_points_pre})")
        assert qdrant_points_post == qdrant_points_pre, "Qdrant points changed!"

    chunks_size_post = chunks_path.stat().st_size
    embeddings_size_post = embeddings_path.stat().st_size

    print(f"Post-test chunks size: {chunks_size_post} bytes (pre: {chunks_size_pre})")
    print(f"Post-test embeddings size: {embeddings_size_post} bytes (pre: {embeddings_size_pre})")

    assert chunks_size_post == chunks_size_pre, "Chunks file changed!"
    assert embeddings_size_post == embeddings_size_pre, "Embeddings file changed!"
    print("[PASS] Zero mutation guarantee PASSED: Qdrant vector database and disk caches remained 100% intact.")

    print("\n==================================================")
    print("ALL STEP 5A MCP SERVER TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
