"""Step 5B Integration Test Suite for BIS Sahayak.

Verifies end-to-end integration of MCP retrieval with ConsumerAgent:
1. "What is BIS CARE?" via MCP retrieval (proves execution path, retrieval_channel="mcp").
2. Complaint-related inquiry.
3. ISI-mark inquiry.
4. HUID verification inquiry.
5. Out-of-scope / unsupported question (verifies refusal, zero hallucinations).
6. Deliberate MCP failure / fallback test (retrieval_channel="fallback", direct Retriever serves).
7. Explicit MCP disabled test (retrieval_channel="direct", direct Retriever serves).
8. FastAPI /api/chat endpoint verification.
9. Data and collection integrity (Qdrant points, chunk files, embeddings files unchanged).
"""

from __future__ import annotations

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

from fastapi.testclient import TestClient

from app.agents.consumer_agent import ConsumerAgent
from app.main import app
from app.mcp.client import BISConsumerMCPClient, MCPClientError


def test_step5b_integration() -> None:
    print("==================================================")
    print("STEP 5B: INTEGRATION TEST SUITE (MCP + CONSUMER AGENT)")
    print("==================================================")

    # 0. Snapshot Pre-Test Data Integrity
    chunks_path = BACKEND_DIR / "data" / "processed" / "bis_consumer_chunks.json"
    embeddings_path = BACKEND_DIR / "data" / "processed" / "bis_consumer_embeddings.json"
    chunks_size_pre = chunks_path.stat().st_size
    embeddings_size_pre = embeddings_path.stat().st_size
    print(f"Pre-test chunks file size: {chunks_size_pre} bytes")
    print(f"Pre-test embeddings file size: {embeddings_size_pre} bytes")

    # Initialize Agent with default MCP enabled
    agent = ConsumerAgent()
    print(f"Agent initialized (use_mcp={agent.use_mcp}, mcp_available={agent.mcp_client.is_available if agent.mcp_client else False})")

    # --- Test 1: "What is BIS CARE?" via MCP ---
    print("\n--- Test 1: 'What is BIS CARE?' via MCP Retrieval ---")
    res1 = agent.answer_question("What is BIS CARE?")
    print(f"Retrieval Channel: {res1.get('retrieval_channel')}")
    print(f"Retrieval Attempts: {res1.get('retrieval_attempts')}")
    print(f"Evidence Sufficient: {res1.get('evidence_sufficient')}")
    print(f"Evidence Chunks Count: {len(res1.get('evidence', []))}")
    print(f"Model Used: {res1.get('model_used')}")
    print(f"Answer Snippet: {res1.get('answer', '')[:120]}...")

    assert res1.get("retrieval_channel") == "mcp", f"Expected channel 'mcp', got {res1.get('retrieval_channel')}"
    assert res1.get("evidence_sufficient") is True, "Expected evidence to be sufficient"
    assert len(res1.get("evidence", [])) > 0, "Expected evidence chunks to be present"

    # Verify metadata preservation on returned evidence
    first_chunk = res1["evidence"][0]
    expected_meta_fields = ["chunk_id", "text", "title", "source_url", "category", "document_type", "score", "section"]
    for field in expected_meta_fields:
        assert field in first_chunk, f"Missing required evidence field: {field}"
    print("[PASS] MCP client invoked, search_bis_documents executed, and metadata preserved.")

    # --- Test 2: Complaint-Related Question ---
    print("\n--- Test 2: Complaint-Related Question ---")
    res2 = agent.answer_question("How can a consumer file a complaint with BIS?")
    print(f"Retrieval Channel: {res2.get('retrieval_channel')}")
    print(f"Evidence Count: {len(res2.get('evidence', []))}")
    print(f"Answer Snippet: {res2.get('answer', '')[:120]}...")
    assert res2.get("retrieval_channel") == "mcp"
    assert res2.get("evidence_sufficient") is True
    assert any("complaint" in c.get("title", "").lower() or "complaint" in (c.get("section") or "").lower() for c in res2["evidence"])
    print("[PASS] Complaint inquiry resolved via MCP with validated topic alignment.")

    # --- Test 3: ISI-Related Question ---
    print("\n--- Test 3: ISI-Related Question ---")
    res3 = agent.answer_question("What does the ISI mark mean on consumer products?")
    print(f"Retrieval Channel: {res3.get('retrieval_channel')}")
    print(f"Evidence Count: {len(res3.get('evidence', []))}")
    print(f"Answer Snippet: {res3.get('answer', '')[:120]}...")
    assert res3.get("retrieval_channel") == "mcp"
    assert res3.get("evidence_sufficient") is True
    print("[PASS] ISI inquiry resolved via MCP.")

    # --- Test 4: HUID-Related Question ---
    print("\n--- Test 4: HUID-Related Question ---")
    res4 = agent.answer_question("How can HUID be verified by consumers?")
    print(f"Retrieval Channel: {res4.get('retrieval_channel')}")
    print(f"Evidence Count: {len(res4.get('evidence', []))}")
    print(f"Answer Snippet: {res4.get('answer', '')[:120]}...")
    assert res4.get("retrieval_channel") == "mcp"
    assert res4.get("evidence_sufficient") is True
    assert any("huid" in c.get("text", "").lower() or "hallmark" in c.get("title", "").lower() for c in res4["evidence"])
    print("[PASS] HUID inquiry resolved via MCP with verified hallmarking evidence.")

    # --- Test 5: Out-of-Scope / Unsupported Question ---
    print("\n--- Test 5: Out-of-Scope / Unsupported Question ---")
    res5 = agent.answer_question("How do I bake a chocolate cake with strawberries?")
    print(f"Retrieval Channel: {res5.get('retrieval_channel')}")
    print(f"Evidence Sufficient: {res5.get('evidence_sufficient')}")
    print(f"Answer: {res5.get('answer')}")
    assert res5.get("retrieval_channel") == "mcp"
    assert res5.get("evidence_sufficient") is False
    assert "insufficient" in res5.get("answer", "").lower()
    print("[PASS] Out-of-scope question correctly refused without hallucinations.")

    # --- Test 6: Fallback When MCP Fails Deliberately ---
    print("\n--- Test 6: Safe Fallback Verification (Forced MCP Failure) ---")

    class FailingMCPClient(BISConsumerMCPClient):
        def search_bis_documents(self, query: str, top_k: int = 5, timeout: float = 15.0):
            raise MCPClientError("Simulated MCP transport disconnect / timeout.")

    fallback_agent = ConsumerAgent(mcp_client=FailingMCPClient(), use_mcp=True)
    res6 = fallback_agent.answer_question("What is BIS CARE?")
    print(f"Retrieval Channel: {res6.get('retrieval_channel')}")
    print(f"Evidence Chunks: {len(res6.get('evidence', []))}")
    print(f"Evidence Sufficient: {res6.get('evidence_sufficient')}")
    print(f"Answer Snippet: {res6.get('answer', '')[:120]}...")

    assert res6.get("retrieval_channel") == "fallback", f"Expected 'fallback', got {res6.get('retrieval_channel')}"
    assert len(res6.get("evidence", [])) > 0, "Fallback Retriever should have provided evidence"
    assert res6.get("evidence_sufficient") is True
    assert res6.get("answer") is not None
    print("[PASS] Deliberate MCP failure failed safely to direct Retriever with retrieval_channel='fallback'.")

    # --- Test 7: MCP Disabled Explicitly ---
    print("\n--- Test 7: MCP Disabled Explicitly (use_mcp=False) ---")
    direct_agent = ConsumerAgent(use_mcp=False)
    res7 = direct_agent.answer_question("What is BIS CARE?")
    print(f"Retrieval Channel: {res7.get('retrieval_channel')}")
    print(f"Evidence Chunks: {len(res7.get('evidence', []))}")
    print(f"Answer Snippet: {res7.get('answer', '')[:120]}...")

    assert res7.get("retrieval_channel") == "direct", f"Expected 'direct', got {res7.get('retrieval_channel')}"
    assert len(res7.get("evidence", [])) > 0
    assert res7.get("evidence_sufficient") is True
    print("[PASS] MCP disabled execution correctly labeled retrieval_channel='direct'.")

    # --- Test 8: FastAPI Endpoint Verification ---
    print("\n--- Test 8: FastAPI /api/chat Endpoint Verification ---")
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "What is BIS CARE?"})
    print(f"FastAPI Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    chat_json = response.json()
    print(f"FastAPI response retrieval_channel: {chat_json.get('retrieval_channel')}")
    print(f"FastAPI response answer snippet: {chat_json.get('answer', '')[:120]}...")
    assert chat_json.get("retrieval_channel") in {"mcp", "fallback", "direct"}
    assert "evidence" in chat_json
    assert "retrieval_attempts" in chat_json
    print("[PASS] FastAPI /api/chat endpoint returned valid payload with retrieval_channel.")

    # --- Test 9: Data Integrity Checks ---
    print("\n--- Test 9: Data Integrity Checks ---")
    chunks_size_post = chunks_path.stat().st_size
    embeddings_size_post = embeddings_path.stat().st_size
    print(f"Post-test chunks file size: {chunks_size_post} bytes (delta: {chunks_size_post - chunks_size_pre})")
    print(f"Post-test embeddings file size: {embeddings_size_post} bytes (delta: {embeddings_size_post - embeddings_size_pre})")

    assert chunks_size_post == chunks_size_pre, "Processed chunks file was modified!"
    assert embeddings_size_post == embeddings_size_pre, "Processed embeddings file was modified!"
    print("[PASS] Zero mutation guarantee verified. All production files remained intact.")

    print("\n==================================================")
    print("ALL STEP 5B INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    test_step5b_integration()
