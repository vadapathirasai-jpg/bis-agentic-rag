"""Incremental source synchronization module for BIS Sahayak.

Manages change detection for official BIS sources via SHA-256 content hashing,
and executes granular, single-source incremental synchronization:
- If UNCHANGED: Skips re-downloading, chunking, embedding, and Qdrant operations.
- If CHANGED: Updates raw snapshot and metadata, re-extracts records, re-chunks,
  generates embeddings ONLY for new chunks, safely removes ONLY old points
  belonging to the source, and upserts new points.

Includes a comprehensive Dry-Run mode and defensive safety validations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
import requests

from app.ingestion.chunker import DocumentChunker
from app.ingestion.loader import DocumentLoader, load_document
from app.ingestion.source_fetcher import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    compute_content_sha256,
    detect_content_change,
    load_sources,
    validate_bis_url,
)
from app.retrieval.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddingService,
    generate_embeddings,
)
from app.retrieval.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DISTANCE,
    DEFAULT_VECTOR_SIZE,
    REQUIRED_METADATA_FIELDS,
    QdrantStore,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class SyncSafetyError(Exception):
    """Raised when source identity or point deletion safety checks fail."""


class SourceSynchronizer:
    """Incremental synchronization engine for official BIS knowledge sources."""

    def __init__(
        self,
        sources_manifest_path: Optional[Union[str, Path]] = None,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_chunks_path: Optional[Union[str, Path]] = None,
        processed_embeddings_path: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        qdrant_store: Optional[QdrantStore] = None,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize the synchronizer with paths, Qdrant client, and embedding service.

        Args:
            sources_manifest_path: Path to sources.json.
            raw_dir: Path to raw data directory (contains .html/.pdf and .metadata.json).
            processed_chunks_path: Path to bis_consumer_chunks.json.
            processed_embeddings_path: Path to bis_consumer_embeddings.json.
            collection_name: Qdrant collection name to synchronize.
            qdrant_store: Optional existing QdrantStore instance.
            qdrant_client: Optional raw QdrantClient (e.g. for testing with :memory:).
            embedding_service: Optional pre-configured EmbeddingService.
            model_name: SentenceTransformers model identifier.
        """
        backend_dir = Path(__file__).resolve().parent.parent.parent
        self.manifest_path = (
            Path(sources_manifest_path)
            if sources_manifest_path
            else backend_dir / "data" / "sources.json"
        )
        self.raw_dir = Path(raw_dir) if raw_dir else backend_dir / "data" / "raw"
        self.chunks_path = (
            Path(processed_chunks_path)
            if processed_chunks_path
            else backend_dir / "data" / "processed" / "bis_consumer_chunks.json"
        )
        self.embeddings_path = (
            Path(processed_embeddings_path)
            if processed_embeddings_path
            else backend_dir / "data" / "processed" / "bis_consumer_embeddings.json"
        )

        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        # Qdrant client setup
        if qdrant_client is not None:
            self.qdrant_client = qdrant_client
        elif qdrant_store is not None:
            self.qdrant_client = qdrant_store.client
        else:
            self._qdrant_store = QdrantStore(collection_name=self.collection_name)
            self.qdrant_client = self._qdrant_store.client

        # Embedding service setup
        self.model_name = model_name
        self._embedding_service = embedding_service

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy-loaded EmbeddingService instance."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(model_name=self.model_name)
        return self._embedding_service

    def get_source_point_ids(
        self,
        source_url: str,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Identify all point IDs in Qdrant that strictly match the given source_url.

        Defensive verification: Every returned point must possess a payload where
        payload['source_url'] == source_url.

        Args:
            source_url: The unique official source URL.
            collection_name: Target collection name. Defaults to self.collection_name.

        Returns:
            List of matching UUID point ID strings.
        """
        target_collection = collection_name or self.collection_name
        if not source_url or not isinstance(source_url, str):
            raise SyncSafetyError("Cannot look up points: source_url must be a non-empty string.")

        source_filter = Filter(
            must=[FieldCondition(key="source_url", match=MatchValue(value=source_url))]
        )

        matching_point_ids: List[str] = []
        offset = None

        while True:
            scroll_result, next_offset = self.qdrant_client.scroll(
                collection_name=target_collection,
                scroll_filter=source_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in scroll_result:
                payload = point.payload or {}
                if payload.get("source_url") == source_url:
                    matching_point_ids.append(str(point.id))
                else:
                    logger.warning(
                        "Safety check failed: point %s returned for source_url '%s' has mismatched payload '%s'.",
                        point.id,
                        source_url,
                        payload.get("source_url"),
                    )

            if next_offset is None:
                break
            offset = next_offset

        return matching_point_ids

    def delete_source_points(
        self,
        point_ids: List[str],
        source_url: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Defensively delete specific point IDs belonging to a source.

        Args:
            point_ids: Explicit list of point IDs to delete.
            source_url: The source URL being replaced (used for safety logs).
            collection_name: Target collection name. Defaults to self.collection_name.

        Returns:
            Number of points deleted.

        Raises:
            SyncSafetyError: If parameters are invalid or empty when deletion was intended.
        """
        target_collection = collection_name or self.collection_name
        if not point_ids:
            return 0

        if not isinstance(point_ids, list):
            raise SyncSafetyError("point_ids must be an explicit list of point IDs.")

        logger.info(
            "Safely deleting %d old points for '%s' from collection '%s'...",
            len(point_ids),
            source_url,
            target_collection,
        )

        self.qdrant_client.delete(
            collection_name=target_collection,
            points_selector=point_ids,
            wait=True,
        )
        return len(point_ids)

    def upsert_source_points(
        self,
        embedded_chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """Upsert newly embedded chunks into Qdrant using deterministic UUIDs.

        Args:
            embedded_chunks: List of embedded chunk dictionaries.
            collection_name: Target collection name. Defaults to self.collection_name.

        Returns:
            Number of points upserted.
        """
        target_collection = collection_name or self.collection_name
        if not embedded_chunks:
            return 0

        points: List[PointStruct] = []
        for chunk in embedded_chunks:
            chunk_id = chunk["chunk_id"]
            point_id = QdrantStore.generate_point_id(chunk_id)

            payload: Dict[str, Any] = {
                field: chunk.get(field) for field in REQUIRED_METADATA_FIELDS
            }
            for k, v in chunk.items():
                if k != "embedding" and k not in payload:
                    payload[k] = v

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload=payload,
                )
            )

        self.qdrant_client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )
        logger.info(
            "Upserted %d points for collection '%s'.",
            len(points),
            target_collection,
        )
        return len(points)

    def fetch_source_content(
        self,
        source: Dict[str, Any],
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Fetch upstream content for a single source definition.

        Args:
            source: Source definition from sources.json.
            timeout: Network request timeout in seconds.

        Returns:
            Tuple of (raw_bytes, http_response_headers_dict).
        """
        url = source["url"]
        validate_bis_url(url)

        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        headers_dict = {
            "status_code": resp.status_code,
            "etag": resp.headers.get("ETag") or resp.headers.get("etag"),
            "last_modified": resp.headers.get("Last-Modified") or resp.headers.get("last-modified"),
            "content_type": resp.headers.get("Content-Type", ""),
        }
        return resp.content, headers_dict

    def load_stored_metadata(self, filename_stem: str) -> Dict[str, Any]:
        """Load stored metadata for a source stem from raw_dir."""
        meta_path = self.raw_dir / f"{filename_stem}.metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.warning("Failed to read metadata for %s: %s", filename_stem, err)
            return {}

    def _is_content_identical(self, source: Dict[str, Any], current_bytes: bytes) -> bool:
        """Check if structured content extracted from current_bytes matches local snapshot.

        This protects against false-positive changes caused by dynamic boilerplate
        (such as WordPress dynamically generated random menu IDs).
        """
        stem = source["filename_stem"]
        raw_ext = ".html" if source.get("document_type") == "html" else ".pdf"
        local_path = self.raw_dir / f"{stem}{raw_ext}"
        if not local_path.exists():
            return False

        try:
            loader = DocumentLoader()
            local_records = loader.load_document(local_path)
            if not local_records:
                return False

            with tempfile.NamedTemporaryFile(suffix=raw_ext, delete=False) as tf:
                tf.write(current_bytes)
                temp_path = Path(tf.name)

            temp_meta = temp_path.with_suffix(".metadata.json")
            try:
                stored_meta = self.load_stored_metadata(stem)
                with open(temp_meta, "w", encoding="utf-8") as f:
                    json.dump(stored_meta, f)
                current_records = loader.load_document(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
                temp_meta.unlink(missing_ok=True)

            if len(local_records) != len(current_records):
                return False

            local_text = [(r.get("section"), r.get("text")) for r in local_records]
            current_text = [(r.get("section"), r.get("text")) for r in current_records]
            return local_text == current_text
        except Exception as err:
            logger.warning("Content equality check failed for %s: %s", stem, err)
            return False

    def plan_source_sync(
        self,
        source: Dict[str, Any],
        fetched_content: Optional[bytes] = None,
        fetched_headers: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a source against its stored metadata and determine sync plan.

        Does NOT modify any files or Qdrant points.

        Args:
            source: Source definition dict.
            fetched_content: Optional already-fetched raw bytes. If None, reads from local raw snapshot.
            fetched_headers: Optional headers from fetch.
            collection_name: Target collection name.

        Returns:
            Plan dictionary with status ('UNCHANGED' | 'CHANGED') and metadata.
        """
        stem = source["filename_stem"]
        url = source["url"]
        stored_meta = self.load_stored_metadata(stem)
        previous_hash = stored_meta.get("content_sha256")

        if fetched_content is not None:
            current_bytes = fetched_content
        else:
            raw_ext = ".html" if source.get("document_type") == "html" else ".pdf"
            local_raw_path = self.raw_dir / f"{stem}{raw_ext}"
            if local_raw_path.exists():
                current_bytes = local_raw_path.read_bytes()
            else:
                current_bytes = b""

        current_hash = compute_content_sha256(current_bytes) if current_bytes else None

        if previous_hash is None or current_hash is None:
            status = "CHANGED"
            reason = "Missing previous or current content hash."
        elif previous_hash == current_hash:
            status = "UNCHANGED"
            reason = "Content SHA-256 matches stored snapshot."
        else:
            # When raw SHA-256 differs, check if structured knowledge records actually changed
            # or if the change is merely dynamic web template markup (e.g. WordPress random menu IDs)
            if current_bytes and self._is_content_identical(source, current_bytes):
                status = "UNCHANGED"
                reason = "Document content records match stored snapshot (ignoring dynamic template markup)."
            else:
                status = "CHANGED"
                reason = "Content SHA-256 and document records differ from stored snapshot."

        target_coll = collection_name or self.collection_name
        existing_point_ids: List[str] = []
        try:
            existing_point_ids = self.get_source_point_ids(url, collection_name=target_coll)
        except Exception as q_err:
            logger.debug("Could not inspect existing points in Qdrant (%s): %s", target_coll, q_err)

        return {
            "source_id": stem,
            "title": source["title"],
            "url": url,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
            "status": status,
            "action": "SKIP" if status == "UNCHANGED" else "REPROCESS",
            "reason": reason,
            "existing_qdrant_points": len(existing_point_ids),
            "existing_point_ids": existing_point_ids,
            "raw_bytes_len": len(current_bytes),
            "fetched_headers": fetched_headers or {},
        }


    def sync_source(
        self,
        source: Dict[str, Any],
        raw_content: bytes,
        headers_info: Dict[str, Any],
        collection_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Perform granular synchronization for a single BIS source.

        Args:
            source: Source definition dictionary from sources.json.
            raw_content: The new raw content bytes for this source.
            headers_info: HTTP response headers dict (etag, last_modified, status_code, content_type).
            collection_name: Target Qdrant collection name.
            dry_run: If True, plans actions without modifying disk or Qdrant.

        Returns:
            Synchronization result dictionary.
        """
        stem = source["filename_stem"]
        url = source["url"]
        target_coll = collection_name or self.collection_name

        # 1. Determine plan
        plan = self.plan_source_sync(
            source=source,
            fetched_content=raw_content,
            fetched_headers=headers_info,
            collection_name=target_coll,
        )

        if plan["status"] == "UNCHANGED":
            return {
                "source_id": stem,
                "title": source["title"],
                "url": url,
                "status": "UNCHANGED",
                "action": "SKIP",
                "previous_hash": plan["previous_hash"],
                "current_hash": plan["current_hash"],
                "old_points_deleted": 0,
                "new_points_upserted": 0,
                "embeddings_generated": 0,
                "new_chunks_count": 0,
            }

        if dry_run:
            return {
                "source_id": stem,
                "title": source["title"],
                "url": url,
                "status": "CHANGED",
                "action": "DRY_RUN_REPROCESS",
                "previous_hash": plan["previous_hash"],
                "current_hash": plan["current_hash"],
                "existing_qdrant_points": plan["existing_qdrant_points"],
                "new_bytes": len(raw_content),
                "plan": plan,
            }

        # 2. REAL EXECUTION FOR CHANGED SOURCE
        logger.info("Synchronizing changed source '%s' (%s)...", source["title"], stem)

        # A. Update raw snapshot file
        doc_type = source.get("document_type", "html")
        raw_ext = ".html" if doc_type == "html" else ".pdf"
        raw_filename = f"{stem}{raw_ext}"
        raw_file_path = self.raw_dir / raw_filename

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file_path.write_bytes(raw_content)

        # B. Update metadata JSON
        retrieved_at = datetime.now(timezone.utc).isoformat()
        metadata_file_path = self.raw_dir / f"{stem}.metadata.json"
        current_sha256 = compute_content_sha256(raw_content)

        metadata_dict: Dict[str, Any] = {
            "title": source["title"],
            "source_url": url,
            "filename_stem": stem,
            "category": source.get("category", "consumer"),
            "document_type": doc_type,
            "retrieved_at": retrieved_at,
            "local_file": f"backend/data/raw/{raw_filename}",
            "file_size_bytes": len(raw_content),
            "content_sha256": current_sha256,
            "content_type_header": headers_info.get("content_type", ""),
            "etag": headers_info.get("etag"),
            "last_modified": headers_info.get("last_modified"),
            "http_status": headers_info.get("status_code", 200),
            "source": "BIS",
        }
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4, ensure_ascii=False)

        # C. Load ONLY this source's structured records
        records = load_document(raw_file_path)
        if not records:
            logger.warning("No structured records extracted from '%s'.", raw_filename)

        # D. Chunk ONLY its records
        chunker = DocumentChunker()
        new_chunks = chunker.chunk_records(records)
        for chunk in new_chunks:
            chunk["filename_stem"] = stem

        # E. Generate embeddings ONLY for its new chunks
        embedded_chunks = []
        if new_chunks:
            embedded_chunks = generate_embeddings(
                chunks=new_chunks,
                model=self.embedding_service.model,
                model_name=self.model_name,
            )

        # F. Identify and delete ONLY old points belonging to this source in Qdrant
        old_point_ids = self.get_source_point_ids(url, collection_name=target_coll)
        deleted_count = 0
        if old_point_ids:
            deleted_count = self.delete_source_points(
                point_ids=old_point_ids,
                source_url=url,
                collection_name=target_coll,
            )

        # G. Insert/upsert ONLY new points
        upserted_count = 0
        if embedded_chunks:
            upserted_count = self.upsert_source_points(
                embedded_chunks=embedded_chunks,
                collection_name=target_coll,
            )

        # H. Update combined processed cache files if paths exist and are in use
        self._update_processed_cache(url=url, new_chunks=new_chunks, new_embeddings=embedded_chunks)

        return {
            "source_id": stem,
            "title": source["title"],
            "url": url,
            "status": "CHANGED",
            "action": "SYNCHRONIZED",
            "previous_hash": plan["previous_hash"],
            "current_hash": current_sha256,
            "old_points_deleted": deleted_count,
            "new_points_upserted": upserted_count,
            "embeddings_generated": len(embedded_chunks),
            "new_chunks_count": len(new_chunks),
        }

    def _update_processed_cache(
        self,
        url: str,
        new_chunks: List[Dict[str, Any]],
        new_embeddings: List[Dict[str, Any]],
    ) -> None:
        """Update processed JSON caches by replacing entries matching url."""
        if self.chunks_path.exists():
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    existing_chunks = json.load(f)
                filtered_chunks = [c for c in existing_chunks if c.get("source_url") != url]
                filtered_chunks.extend(new_chunks)
                with open(self.chunks_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_chunks, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning("Could not update processed chunks cache: %s", e)

        if self.embeddings_path.exists():
            try:
                with open(self.embeddings_path, "r", encoding="utf-8") as f:
                    existing_embeddings = json.load(f)
                filtered_embeddings = [e for e in existing_embeddings if e.get("source_url") != url]
                filtered_embeddings.extend(new_embeddings)
                with open(self.embeddings_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_embeddings, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning("Could not update processed embeddings cache: %s", e)

    def synchronize_all(
        self,
        collection_name: Optional[str] = None,
        dry_run: bool = False,
        content_overrides: Optional[Dict[str, bytes]] = None,
    ) -> Dict[str, Any]:
        """Synchronize all sources defined in the manifest.

        Args:
            collection_name: Target collection name.
            dry_run: If True, only inspects and plans without modifying state.
            content_overrides: Optional mapping of stem -> raw_bytes (useful for testing).

        Returns:
            Summary dictionary with results per source and aggregate counts.
        """
        sources = load_sources(self.manifest_path)
        target_coll = collection_name or self.collection_name

        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "collection_name": target_coll,
            "total_sources_evaluated": len(sources),
            "sources_unchanged": 0,
            "sources_changed": 0,
            "total_points_deleted": 0,
            "total_points_upserted": 0,
            "total_embeddings_generated": 0,
            "results": [],
        }

        for source in sources:
            stem = source["filename_stem"]

            if content_overrides and stem in content_overrides:
                raw_bytes = content_overrides[stem]
                headers = {"status_code": 200, "etag": None, "last_modified": None, "content_type": "text/html"}
            else:
                try:
                    raw_bytes, headers = self.fetch_source_content(source)
                except Exception as net_err:
                    logger.warning("Could not fetch upstream %s (%s); checking local cache.", stem, net_err)
                    raw_ext = ".html" if source.get("document_type") == "html" else ".pdf"
                    local_f = self.raw_dir / f"{stem}{raw_ext}"
                    raw_bytes = local_f.read_bytes() if local_f.exists() else b""
                    headers = {"status_code": 200, "etag": None, "last_modified": None, "content_type": "text/html"}

            res = self.sync_source(
                source=source,
                raw_content=raw_bytes,
                headers_info=headers,
                collection_name=target_coll,
                dry_run=dry_run,
            )
            report["results"].append(res)

            if res["status"] == "UNCHANGED":
                report["sources_unchanged"] += 1
            else:
                report["sources_changed"] += 1
                report["total_points_deleted"] += res.get("old_points_deleted", 0)
                report["total_points_upserted"] += res.get("new_points_upserted", 0)
                report["total_embeddings_generated"] += res.get("embeddings_generated", 0)

        return report
