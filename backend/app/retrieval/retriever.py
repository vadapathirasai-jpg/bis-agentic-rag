"""Semantic retrieval layer for BIS Sahayak.

Embeds natural-language queries using the Sentence Transformers model
(sentence-transformers/all-MiniLM-L6-v2), queries the local Qdrant collection
('bis_consumer'), and returns ranked evidence chunks with full citation metadata.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# Ensure backend root is on sys.path so app imports work cleanly
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from backend/.env if present
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from app.retrieval.embeddings import DEFAULT_MODEL_NAME, EmbeddingService
from app.retrieval.qdrant_store import DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_SIZE, QdrantStore

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class Retriever:
    """Semantic retrieval layer interfacing between natural-language queries,
    the Sentence Transformers embedding model, and the Qdrant vector database.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_store: Optional[QdrantStore] = None,
        timeout: int = 10,
    ) -> None:
        """Initialize Retriever with Qdrant client and SentenceTransformer service.

        Args:
            qdrant_url: Endpoint URL for Qdrant service. Defaults to QDRANT_URL env var
                or 'http://localhost:6333'.
            collection_name: Target collection name. Defaults to QDRANT_COLLECTION env var
                or 'bis_consumer'.
            model_name: SentenceTransformers model identifier. Defaults to
                'sentence-transformers/all-MiniLM-L6-v2'.
            embedding_service: Optional pre-configured EmbeddingService instance.
            qdrant_store: Optional pre-configured QdrantStore instance.
            timeout: Network timeout in seconds for Qdrant client calls.
        """
        # Re-use or instantiate QdrantStore
        if qdrant_store is not None:
            self.qdrant_store = qdrant_store
            self.url = qdrant_store.url
            self.collection_name = qdrant_store.collection_name
            self.client = qdrant_store.client
        else:
            self.qdrant_store = QdrantStore(
                url=qdrant_url,
                collection_name=collection_name,
                timeout=timeout,
            )
            self.url = self.qdrant_store.url
            self.collection_name = self.qdrant_store.collection_name
            self.client = self.qdrant_store.client

        # Re-use or instantiate EmbeddingService
        if embedding_service is not None:
            self.embedding_service = embedding_service
        else:
            self.embedding_service = EmbeddingService(model_name=model_name)

        self.model_name = self.embedding_service.model_name
        self.expected_dimension = DEFAULT_VECTOR_SIZE

    def connect(self) -> None:
        """Verify Qdrant service connection and validate target collection existence.

        Raises:
            ConnectionError: If the Qdrant service is unreachable.
            RuntimeError: If the collection does not exist or vector config is invalid.
        """
        if not self.qdrant_store.is_healthy():
            raise ConnectionError(
                f"Unable to connect to Qdrant service at '{self.url}'. "
                "Please verify that the Docker container 'bis-qdrant' is running."
            )

        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Collection '{self.collection_name}' does not exist in Qdrant at '{self.url}'. "
                "Please run 'python backend/app/retrieval/qdrant_store.py' to initialize the collection."
            )

        # Validate vector size and distance configuration
        self.qdrant_store.validate_collection(expected_size=self.expected_dimension)
        logger.info(
            "Connected to Qdrant at '%s'. Collection '%s' verified (dim: %d).",
            self.url,
            self.collection_name,
            self.expected_dimension,
        )

    def embed_query(self, query: str) -> List[float]:
        """Convert a user query into a dense 384-dimensional vector embedding.

        Args:
            query: Non-empty natural language query string.

        Returns:
            List of 384 floats.

        Raises:
            ValueError: If query is empty or whitespace-only, or embedding dimension mismatches.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty, non-whitespace string.")

        vector = self.embedding_service.embed_query(query.strip())

        if len(vector) != self.expected_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: got {len(vector)}, expected {self.expected_dimension}."
            )

        return vector

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Build Qdrant Filter object from a key-value criteria dictionary."""
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if not conditions:
            return None
        return Filter(must=conditions)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the Qdrant collection using a pre-computed query vector.

        Args:
            query_vector: 384-dimensional embedding vector.
            top_k: Number of highest-ranked results to retrieve (must be > 0).
            score_threshold: Optional minimum cosine similarity score cutoff.
            filters: Optional dictionary of metadata key-value pairs to filter by.

        Returns:
            Ranked list of result dictionaries containing payload metadata and score.

        Raises:
            ValueError: If query_vector or top_k are invalid.
            RuntimeError: If search execution fails.
        """
        if not query_vector or len(query_vector) != self.expected_dimension:
            raise ValueError(
                f"query_vector must have dimension {self.expected_dimension}, "
                f"got {len(query_vector) if query_vector else 0}."
            )

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        query_filter = self._build_filter(filters)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant query_points failed for collection '%s': %s", self.collection_name, exc)
            raise RuntimeError(
                f"Qdrant vector search failed on collection '{self.collection_name}': {exc}"
            ) from exc

        results: List[Dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            result = {
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
                "source": payload.get("source"),
                "title": payload.get("title"),
                "source_url": payload.get("source_url"),
                "category": payload.get("category"),
                "document_type": payload.get("document_type"),
                "page": payload.get("page"),
                "section": payload.get("section"),
                "chunk_index": payload.get("chunk_index"),
                "total_chunks": payload.get("total_chunks"),
                "char_count": payload.get("char_count"),
                "score": float(point.score),
            }
            results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant BIS Consumer chunks for a natural-language query.

        Args:
            query: Natural-language search question.
            top_k: Maximum number of candidate chunks to return (default: 5).
            score_threshold: Optional cosine similarity score cutoff.
            filters: Optional metadata filter dictionary (e.g. {'category': 'consumer'}).

        Returns:
            Ranked list of matching BIS chunks with full payload metadata and similarity scores.

        Raises:
            ValueError: If query is empty or whitespace-only, or top_k is invalid.
        """
        # Validate query
        if query is None or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty, non-whitespace string.")

        # Validate top_k
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        # Convert query to vector embedding
        query_vector = self.embed_query(query)

        # Search Qdrant collection
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )


# Backward compatibility alias
BISRetriever = Retriever


if __name__ == "__main__":
    print("=" * 70)
    print("BIS Sahayak - Semantic Retriever Verification")
    print("=" * 70)

    retriever = Retriever()
    print(f"Connecting to Qdrant at: {retriever.url}...")
    retriever.connect()
    total_points = retriever.qdrant_store.count_points()
    print(f"Verified connection to collection '{retriever.collection_name}' ({total_points} points).")
    print(f"Loaded embedding model: {retriever.model_name} (dimension: {retriever.expected_dimension})")
    print("=" * 70)

    test_queries = [
        "What is BIS CARE?",
        "How can a consumer file a complaint with BIS?",
        "What does the ISI mark mean?",
    ]

    for q_idx, query_text in enumerate(test_queries, 1):
        print(f"\n[Test Query {q_idx}]")
        print(f"Query: {query_text}")

        # Check embedding dimension
        query_vec = retriever.embed_query(query_text)
        print(f"Query vector dimension: {len(query_vec)}")

        results = retriever.retrieve(query=query_text, top_k=3)
        print(f"Retrieved {len(results)} results:")

        for r_idx, res in enumerate(results, 1):
            print(f"\n  {r_idx}. Score:      {res['score']:.4f}")
            print(f"     Title:      {res.get('title')}")
            print(f"     Section:    {res.get('section')}")
            print(f"     Chunk ID:   {res.get('chunk_id')}")
            print(f"     Source URL: {res.get('source_url')}")
            preview_text = (res.get("text") or "").replace("\n", " ")
            if len(preview_text) > 160:
                preview_text = preview_text[:160] + "..."
            print(f"     Text:       {preview_text}")

    print("\n" + "=" * 70)
    print("Running Edge Case Validations...")
    print("=" * 70)

    # Validate empty query rejection
    for bad_query in ["", "   ", None]:
        try:
            retriever.retrieve(bad_query)  # type: ignore[arg-type]
            print(f"FAILED: Expected ValueError for query '{bad_query}'")
        except ValueError:
            print(f"PASSED: Correctly rejected invalid query {repr(bad_query)}")

    # Validate top_k rejection
    for bad_k in [0, -1, "five"]:
        try:
            retriever.retrieve("valid query", top_k=bad_k)  # type: ignore[arg-type]
            print(f"FAILED: Expected ValueError for top_k {bad_k}")
        except ValueError:
            print(f"PASSED: Correctly rejected invalid top_k {repr(bad_k)}")

    # Validate score threshold filtering
    high_thresh_results = retriever.retrieve("What is BIS CARE?", top_k=5, score_threshold=0.85)
    print(f"PASSED: Score threshold 0.85 returned {len(high_thresh_results)} results (all scores >= 0.85).")

    # Validate metadata filtering
    filtered_results = retriever.retrieve(
        "What is BIS CARE?",
        top_k=3,
        filters={"category": "consumer"},
    )
    print(f"PASSED: Category filter 'consumer' returned {len(filtered_results)} results.")

    print("\n" + "=" * 70)
    print("ALL RETRIEVAL VERIFICATIONS PASSED SUCCESSFULLY")
    print("=" * 70)

