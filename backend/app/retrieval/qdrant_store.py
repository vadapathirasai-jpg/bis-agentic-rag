"""Qdrant vector store integration for BIS Sahayak.

Manages connection to local Qdrant instance, collection lifecycle,
vector indexing, deterministic point generation, and idempotent upserts
for official BIS Consumer Agent knowledge chunks.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Default path constants
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
DEFAULT_EMBEDDINGS_PATH = BACKEND_DIR / "data" / "processed" / "bis_consumer_embeddings.json"

# Load environment variables from backend/.env if present
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Configuration defaults
DEFAULT_COLLECTION_NAME = "bis_consumer"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_DISTANCE = Distance.COSINE
DEFAULT_BATCH_SIZE = 64

# Required fields for payload preservation
REQUIRED_METADATA_FIELDS = [
    "chunk_id",
    "text",
    "source",
    "title",
    "source_url",
    "category",
    "document_type",
    "page",
    "section",
    "chunk_index",
    "total_chunks",
    "char_count",
]


class QdrantStore:
    """Manages Qdrant vector store collections, indexing, and point persistence for BIS Sahayak."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        """Initialize connection to Qdrant vector database.

        Args:
            url: Qdrant endpoint URL. Defaults to QDRANT_URL env var, or
                constructs from QDRANT_HOST / QDRANT_PORT, or 'http://localhost:6333'.
            collection_name: Target collection name. Defaults to QDRANT_COLLECTION
                env var, or 'bis_consumer'.
            timeout: Network timeout in seconds for Qdrant client calls.
        """
        if url:
            self.url = url
        elif os.getenv("QDRANT_URL"):
            self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        else:
            host = os.getenv("QDRANT_HOST", "localhost")
            port = os.getenv("QDRANT_PORT", "6333")
            self.url = f"http://{host}:{port}"

        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or DEFAULT_COLLECTION_NAME
        )
        self.timeout = timeout
        self.client = QdrantClient(url=self.url, timeout=self.timeout)

    def is_healthy(self) -> bool:
        """Check if the Qdrant service is reachable and responsive."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Health check failed for Qdrant at %s: %s", self.url, exc)
            return False

    def collection_exists(self) -> bool:
        """Check whether the target collection exists in Qdrant."""
        try:
            return self.client.collection_exists(collection_name=self.collection_name)
        except Exception as exc:
            logger.error("Error checking collection existence '%s': %s", self.collection_name, exc)
            raise

    def create_collection(
        self,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: Distance = DEFAULT_DISTANCE,
    ) -> None:
        """Create the target collection with the specified vector dimensions and distance metric.

        Args:
            vector_size: Dimensionality of embedding vectors (default: 384).
            distance: Distance metric (default: Cosine).
        """
        logger.info(
            "Creating collection '%s' (dimension: %d, distance: %s)...",
            self.collection_name,
            vector_size,
            distance.name,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        logger.info("Collection '%s' created successfully.", self.collection_name)

    def validate_collection(
        self,
        expected_size: int = DEFAULT_VECTOR_SIZE,
        expected_distance: Distance = DEFAULT_DISTANCE,
    ) -> None:
        """Validate that an existing collection matches expected vector parameters.

        Args:
            expected_size: Expected vector size (default: 384).
            expected_distance: Expected distance metric (default: Cosine).

        Raises:
            ValueError: If collection settings do not match expected parameters.
        """
        info = self.client.get_collection(collection_name=self.collection_name)
        vectors_config = info.config.params.vectors

        if isinstance(vectors_config, dict):
            raise ValueError(
                f"Collection '{self.collection_name}' has incompatible multi-vector configuration: {vectors_config}"
            )

        actual_size = getattr(vectors_config, "size", None)
        actual_distance = getattr(vectors_config, "distance", None)

        # Distance comparison
        distance_matches = (
            actual_distance == expected_distance
            or str(actual_distance).lower().endswith("cosine")
            or getattr(actual_distance, "value", "").lower() == "cosine"
        )

        if actual_size != expected_size or not distance_matches:
            raise ValueError(
                f"Collection '{self.collection_name}' configuration mismatch! "
                f"Expected size={expected_size}, distance={expected_distance}. "
                f"Got size={actual_size}, distance={actual_distance}."
            )

        logger.info(
            "Collection '%s' verified: size=%s, distance=%s.",
            self.collection_name,
            actual_size,
            actual_distance,
        )

    def ensure_collection(
        self,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: Distance = DEFAULT_DISTANCE,
    ) -> None:
        """Ensure collection exists and has valid configuration; creates it if absent."""
        if not self.collection_exists():
            self.create_collection(vector_size=vector_size, distance=distance)
        else:
            self.validate_collection(expected_size=vector_size, expected_distance=distance)

    @staticmethod
    def generate_point_id(chunk_id: str) -> str:
        """Generate a deterministic UUID string point ID from a chunk_id.

        This guarantees that re-running upserts will overwrite points idempotently
        rather than duplicating them.

        Args:
            chunk_id: The deterministic hex chunk ID.

        Returns:
            Standard UUID5 string.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))

    def load_embeddings(
        self,
        file_path: Optional[Path | str] = None,
        expected_dimension: int = DEFAULT_VECTOR_SIZE,
    ) -> List[Dict[str, Any]]:
        """Load and validate pre-computed embeddings and metadata from a JSON file.

        Args:
            file_path: Path to embeddings JSON file. Defaults to DEFAULT_EMBEDDINGS_PATH.
            expected_dimension: Expected dimension of embedding vectors (default: 384).

        Returns:
            List of embedding records.

        Raises:
            FileNotFoundError: If the embeddings file does not exist.
            ValueError: If the file is empty, malformed, or has invalid dimensions.
        """
        path = Path(file_path) if file_path else DEFAULT_EMBEDDINGS_PATH
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found at: {path}")

        logger.info("Loading pre-computed embeddings from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Embeddings file '{path}' must contain a non-empty JSON list.")

        # Validate each record
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Record at index {idx} is not a valid JSON dictionary.")

            # Check essential fields
            if "chunk_id" not in record or not record["chunk_id"]:
                raise ValueError(f"Record at index {idx} missing required 'chunk_id'.")
            if "text" not in record or not record["text"]:
                raise ValueError(f"Record at index {idx} (chunk_id={record.get('chunk_id')}) missing required 'text'.")
            if "embedding" not in record or not isinstance(record["embedding"], list):
                raise ValueError(f"Record at index {idx} (chunk_id={record.get('chunk_id')}) missing valid 'embedding' list.")

            # Validate vector dimension
            actual_dim = len(record["embedding"])
            if actual_dim != expected_dimension:
                raise ValueError(
                    f"Record at index {idx} (chunk_id={record['chunk_id']}) has vector dimension {actual_dim}; "
                    f"expected {expected_dimension}."
                )

            # Check that required metadata fields are at least present as keys
            for field in REQUIRED_METADATA_FIELDS:
                if field not in record:
                    raise ValueError(
                        f"Record at index {idx} (chunk_id={record['chunk_id']}) missing metadata field '{field}'."
                    )

        logger.info("Successfully validated %d embedding records from %s.", len(data), path)
        return data

    def upsert_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        wait: bool = True,
    ) -> int:
        """Upsert embedding records into Qdrant in batches using deterministic point IDs.

        Preserves all 12 chunk metadata fields in the payload and achieves idempotency.

        Args:
            embeddings_data: List of validated embedding records.
            batch_size: Number of points per upsert request (default: 64).
            wait: Whether to wait for changes to be committed before returning.

        Returns:
            Number of points upserted.
        """
        if not embeddings_data:
            logger.warning("No embeddings data provided for upsert.")
            return 0

        points: List[PointStruct] = []
        for record in embeddings_data:
            point_id = self.generate_point_id(record["chunk_id"])

            # Build full metadata payload preserving all required fields
            payload: Dict[str, Any] = {
                field: record.get(field) for field in REQUIRED_METADATA_FIELDS
            }
            # Also preserve any auxiliary fields present in the record
            for k, v in record.items():
                if k != "embedding" and k not in payload:
                    payload[k] = v

            points.append(
                PointStruct(
                    id=point_id,
                    vector=record["embedding"],
                    payload=payload,
                )
            )

        total_points = len(points)
        logger.info(
            "Upserting %d points into collection '%s' in batches of %d...",
            total_points,
            self.collection_name,
            batch_size,
        )

        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=wait,
            )
            logger.debug(
                "Upserted batch %d-%d of %d points.",
                i + 1,
                min(i + batch_size, total_points),
                total_points,
            )

        logger.info("Completed upsert of %d points into '%s'.", total_points, self.collection_name)
        return total_points

    def count_points(self) -> int:
        """Return the exact number of points in the target collection.

        Returns:
            Integer point count.
        """
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return result.count

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform a vector similarity search in the collection.

        Args:
            query_vector: Dense embedding vector for the search query.
            limit: Maximum number of candidate chunks to retrieve.
            score_threshold: Optional minimum cosine similarity score.

        Returns:
            List of matching records with payload metadata and similarity scores.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results: List[Dict[str, Any]] = []
        for point in response.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            })
        return results

    def initialize(
        self,
        file_path: Optional[Path | str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Dict[str, Any]:
        """Convenience method to ensure collection, load embeddings, upsert, and verify counts.

        Args:
            file_path: Optional path to embeddings JSON file.
            batch_size: Batch size for upserts.

        Returns:
            Summary dictionary of the ingestion operation.
        """
        self.ensure_collection(vector_size=DEFAULT_VECTOR_SIZE, distance=DEFAULT_DISTANCE)
        embeddings = self.load_embeddings(file_path=file_path)
        upserted_count = self.upsert_embeddings(embeddings_data=embeddings, batch_size=batch_size)
        total_in_collection = self.count_points()

        return {
            "qdrant_url": self.url,
            "collection_name": self.collection_name,
            "vector_size": DEFAULT_VECTOR_SIZE,
            "distance": DEFAULT_DISTANCE.name,
            "embeddings_loaded": len(embeddings),
            "points_upserted": upserted_count,
            "total_points_in_collection": total_in_collection,
        }


# Backward compatibility alias
QdrantVectorStore = QdrantStore


if __name__ == "__main__":
    print("=" * 60)
    print("BIS Sahayak - Qdrant Vector Store Ingestion")
    print("=" * 60)

    store = QdrantStore()
    print(f"Connecting to Qdrant at: {store.url}...")
    if not store.is_healthy():
        print(f"ERROR: Cannot connect to Qdrant service at {store.url}")
        print("Please ensure Docker container 'bis-qdrant' is running.")
        exit(1)

    print(f"Ensuring collection '{store.collection_name}' exists with dimension 384, Cosine...")
    store.ensure_collection(vector_size=DEFAULT_VECTOR_SIZE, distance=DEFAULT_DISTANCE)

    print(f"Loading pre-computed embeddings from {DEFAULT_EMBEDDINGS_PATH}...")
    embeddings_data = store.load_embeddings()
    loaded_count = len(embeddings_data)
    print(f"Loaded {loaded_count} embedding records successfully.")

    print(f"Upserting points into '{store.collection_name}' (batch size: {DEFAULT_BATCH_SIZE})...")
    upserted_count = store.upsert_embeddings(embeddings_data, batch_size=DEFAULT_BATCH_SIZE)

    total_points = store.count_points()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Qdrant URL:                   {store.url}")
    print(f"Collection Name:              {store.collection_name}")
    print(f"Vector Dimension:             {DEFAULT_VECTOR_SIZE}")
    print(f"Distance Metric:              {DEFAULT_DISTANCE.name}")
    print(f"Number of embeddings loaded:  {loaded_count}")
    print(f"Number of points upserted:    {upserted_count}")
    print(f"Total points in collection:   {total_points}")
    print("=" * 60)

    if total_points == loaded_count:
        print("Verification Status: SUCCESS (Idempotent point persistence verified)")
    else:
        print(f"Verification Status: WARNING (Expected {loaded_count} points, found {total_points})")
    print("=" * 60)

