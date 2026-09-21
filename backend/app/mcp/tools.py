"""MCP Tool Implementations for BIS Sahayak.

Provides pure read-only operations exposing BIS consumer knowledge,
metadata inspection, semantic search, and update checking.

Strict Guarantees:
- Zero mutations to disk, raw snapshots, metadata files, or processed caches.
- Zero mutations to Qdrant vector database (no upserts, no deletes).
- No synthetic generation or LLM agent calls.
- Strict input validation and rejection of unapproved / arbitrary inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.source_fetcher import (
    DEFAULT_USER_AGENT,
    compute_content_sha256,
    load_sources,
)
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# Default directory references
DEFAULT_MANIFEST_PATH = BACKEND_DIR / "data" / "sources.json"
DEFAULT_RAW_DIR = BACKEND_DIR / "data" / "raw"

# Singleton retriever cache to avoid reloading embedding model repeatedly
_RETRIEVER_INSTANCE: Optional[Retriever] = None


class _LocalMemoryQdrantStore:
    """Read-only in-memory QdrantStore adapter populated from pre-computed embeddings."""

    def __init__(self, client: Any, collection_name: str = "bis_consumer") -> None:
        self.client = client
        self.url = ":memory:"
        self.collection_name = collection_name

    def is_healthy(self) -> bool:
        return True

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        return True

    def validate_collection(self, expected_size: int = 384) -> bool:
        return True


def _create_fallback_in_memory_retriever() -> Retriever:
    """Create a Retriever backed by an in-memory Qdrant instance populated with production embeddings."""
    import uuid
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from app.retrieval.qdrant_store import DEFAULT_COLLECTION_NAME

    mem_client = QdrantClient(":memory:")
    mem_client.create_collection(
        collection_name=DEFAULT_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    embeddings_file = BACKEND_DIR / "data" / "processed" / "bis_consumer_embeddings.json"
    if embeddings_file.exists():
        try:
            with open(embeddings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            points = []
            for item in data:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item.get("chunk_id", "")))
                payload = {k: v for k, v in item.items() if k != "embedding"}
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=item["embedding"],
                        payload=payload,
                    )
                )
            if points:
                mem_client.upsert(collection_name=DEFAULT_COLLECTION_NAME, points=points)
                logger.info(
                    "Initialized in-memory Qdrant with %d production embeddings for offline retrieval.",
                    len(points),
                )
        except Exception as err:
            logger.error("Failed to load embeddings into in-memory Qdrant: %s", err)

    mem_store = _LocalMemoryQdrantStore(client=mem_client, collection_name=DEFAULT_COLLECTION_NAME)
    return Retriever(qdrant_store=mem_store)


def get_retriever() -> Retriever:
    """Retrieve or lazily initialize the shared Retriever instance.

    Attempts connection to the live Qdrant service first. If unavailable
    (e.g. offline testing or standalone daemon), falls back to an in-memory
    Qdrant instance seeded with the 79 production embeddings and payloads.
    """
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        try:
            standard_retriever = Retriever()
            if standard_retriever.qdrant_store.is_healthy():
                _RETRIEVER_INSTANCE = standard_retriever
            else:
                logger.warning("Live Qdrant is not responsive. Using in-memory store.")
                _RETRIEVER_INSTANCE = _create_fallback_in_memory_retriever()
        except Exception as exc:
            logger.warning("Could not connect to live Qdrant (%s); using in-memory store.", exc)
            _RETRIEVER_INSTANCE = _create_fallback_in_memory_retriever()

    return _RETRIEVER_INSTANCE


def search_bis_documents_impl(
    query: str,
    top_k: int = 5,
    retriever: Optional[Retriever] = None,
) -> Dict[str, Any]:
    """Execute pure read-only semantic search over BIS consumer knowledge.

    Args:
        query: Natural language query string.
        top_k: Maximum number of relevant chunks to retrieve (default 5, clamped 1-20).
        retriever: Optional pre-configured Retriever instance.

    Returns:
        Structured response with ranked evidence chunks and citation metadata.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "error": "Query must be a non-empty string.",
            "query": query,
            "results_count": 0,
            "results": [],
        }

    # Validate and clamp top_k
    try:
        top_k_int = int(top_k)
    except (ValueError, TypeError):
        top_k_int = 5

    if top_k_int < 1:
        top_k_int = 1
    elif top_k_int > 20:
        top_k_int = 20

    active_retriever = retriever or get_retriever()

    try:
        raw_results = active_retriever.retrieve(query=query.strip(), top_k=top_k_int)
    except Exception as exc:
        logger.error("Error retrieving documents for query '%s': %s", query, exc)
        return {
            "success": False,
            "error": f"Search failed: {exc}",
            "query": query.strip(),
            "results_count": 0,
            "results": [],
        }

    formatted_results = []
    for item in raw_results:
        formatted_results.append({
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text"),
            "title": item.get("title"),
            "source_url": item.get("source_url"),
            "category": item.get("category"),
            "document_type": item.get("document_type"),
            "score": round(float(item.get("score", 0.0)), 4),
            "section": item.get("section"),
            "chunk_index": item.get("chunk_index"),
            "total_chunks": item.get("total_chunks"),
        })

    return {
        "success": True,
        "query": query.strip(),
        "results_count": len(formatted_results),
        "results": formatted_results,
    }


def get_source_status_impl(
    manifest_path: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Inspect and return current synchronization status for all approved BIS sources.

    Args:
        manifest_path: Optional path to sources.json.
        raw_dir: Optional path to raw data directory.

    Returns:
        Dictionary summarizing the status and metadata of all approved sources.
    """
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    raw_path = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR

    if not manifest_file.exists():
        return {
            "success": False,
            "error": f"Sources manifest not found at '{manifest_file}'.",
            "total_sources": 0,
            "sources": [],
        }

    sources = load_sources(manifest_file)
    sources_status = []

    for source in sources:
        stem = source.get("filename_stem", "")
        doc_type = source.get("document_type", "html")
        raw_ext = ".html" if doc_type == "html" else ".pdf"
        snapshot_path = raw_path / f"{stem}{raw_ext}"
        metadata_path = raw_path / f"{stem}.metadata.json"

        stored_meta: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                stored_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to read metadata for %s: %s", stem, exc)

        snapshot_exists = snapshot_path.exists()
        file_size = (
            stored_meta.get("file_size_bytes")
            or (snapshot_path.stat().st_size if snapshot_exists else 0)
        )

        sources_status.append({
            "source_id": stem,
            "title": source.get("title"),
            "url": source.get("url"),
            "category": source.get("category"),
            "document_type": doc_type,
            "is_downloaded": snapshot_exists,
            "file_size_bytes": file_size,
            "content_sha256": stored_meta.get("content_sha256"),
            "retrieved_at": stored_meta.get("retrieved_at"),
            "http_status": stored_meta.get("http_status", 200),
            "etag": stored_meta.get("etag"),
            "last_modified": stored_meta.get("last_modified"),
        })

    return {
        "success": True,
        "total_sources": len(sources_status),
        "sources": sources_status,
    }


def check_for_updates_impl(
    check_remote: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Perform a safe, read-only update check comparing stored hashes against current state.

    Zero mutations: Does NOT write to disk, does NOT delete or upsert Qdrant points,
    and does NOT regenerate embeddings.

    Args:
        check_remote: If True, tests remote HTTP availability. If False (default),
            validates current local snapshot hash against stored metadata.
        manifest_path: Optional path to sources.json.
        raw_dir: Optional path to raw data directory.

    Returns:
        Structured update assessment with sources_checked, unchanged, changed, errors, and sources list.
    """
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    raw_path = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR

    if not manifest_file.exists():
        return {
            "sources_checked": 0,
            "unchanged": 0,
            "changed": 0,
            "errors": 1,
            "sources": [],
            "error_detail": f"Sources manifest not found at '{manifest_file}'.",
        }

    sources = load_sources(manifest_file)
    results = []
    unchanged_count = 0
    changed_count = 0
    error_count = 0

    for source in sources:
        stem = source.get("filename_stem", "")
        url = source.get("url", "")
        doc_type = source.get("document_type", "html")
        raw_ext = ".html" if doc_type == "html" else ".pdf"
        snapshot_path = raw_path / f"{stem}{raw_ext}"
        metadata_path = raw_path / f"{stem}.metadata.json"

        stored_meta: Dict[str, Any] = {}
        stored_hash: Optional[str] = None
        if metadata_path.exists():
            try:
                stored_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                stored_hash = stored_meta.get("content_sha256")
            except Exception as exc:
                logger.warning("Error reading metadata for %s: %s", stem, exc)

        current_hash: Optional[str] = None
        status = "UNKNOWN"
        details = ""

        if not check_remote:
            # Check local snapshot hash against metadata
            if not snapshot_path.exists():
                status = "ERROR"
                details = f"Local raw file '{snapshot_path.name}' is missing."
                error_count += 1
            elif not stored_hash:
                status = "ERROR"
                details = f"Stored content_sha256 missing in metadata for '{stem}'."
                error_count += 1
            else:
                try:
                    current_bytes = snapshot_path.read_bytes()
                    current_hash = compute_content_sha256(current_bytes)
                    if current_hash == stored_hash:
                        status = "UNCHANGED"
                        details = "Local snapshot matches stored SHA-256 hash."
                        unchanged_count += 1
                    else:
                        status = "CHANGED"
                        details = "Local snapshot hash differs from stored metadata."
                        changed_count += 1
                except Exception as exc:
                    status = "ERROR"
                    details = f"Failed to compute local hash: {exc}"
                    error_count += 1
        else:
            # Safe remote check (read-only HEAD/GET with short timeout)
            import requests

            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                    timeout=8,
                    stream=True,
                )
                if resp.status_code != 200:
                    status = "ERROR"
                    details = f"Upstream returned HTTP {resp.status_code}."
                    error_count += 1
                else:
                    remote_content = resp.content
                    current_hash = compute_content_sha256(remote_content)
                    if current_hash == stored_hash:
                        status = "UNCHANGED"
                        details = "Upstream content matches stored SHA-256 hash."
                        unchanged_count += 1
                    else:
                        status = "CHANGED"
                        details = "Upstream content SHA-256 differs from stored snapshot."
                        changed_count += 1
            except Exception as net_err:
                status = "ERROR"
                details = f"Failed to reach upstream URL: {net_err}"
                error_count += 1

        results.append({
            "source_id": stem,
            "title": source.get("title"),
            "url": url,
            "status": status,
            "stored_hash": stored_hash,
            "current_hash": current_hash,
            "hash_match": (status == "UNCHANGED"),
            "details": details,
        })

    return {
        "sources_checked": len(sources),
        "unchanged": unchanged_count,
        "changed": changed_count,
        "errors": error_count,
        "check_mode": "remote" if check_remote else "local_snapshot",
        "sources": results,
    }


def get_document_metadata_impl(
    source_id_or_url: str,
    manifest_path: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Retrieve metadata for an approved BIS source, strictly rejecting unapproved inputs.

    Args:
        source_id_or_url: Filename stem (e.g. 'bis_apps') or official URL
            (e.g. 'https://www.bis.gov.in/bis-apps/?lang=en').
        manifest_path: Optional path to sources.json.
        raw_dir: Optional path to raw data directory.

    Returns:
        Structured metadata for the approved source, or a formal rejection response.
    """
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    raw_path = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR

    if not source_id_or_url or not isinstance(source_id_or_url, str):
        return {
            "success": False,
            "status": "REJECTED",
            "error": "Source identifier or URL must be a non-empty string.",
            "input": source_id_or_url,
        }

    cleaned_input = source_id_or_url.strip()

    if not manifest_file.exists():
        return {
            "success": False,
            "status": "ERROR",
            "error": f"Sources manifest not found at '{manifest_file}'.",
            "input": cleaned_input,
        }

    sources = load_sources(manifest_file)
    approved_by_stem: Dict[str, Dict[str, Any]] = {
        s.get("filename_stem", ""): s for s in sources if "filename_stem" in s
    }
    approved_by_url: Dict[str, Dict[str, Any]] = {
        s.get("url", ""): s for s in sources if "url" in s
    }

    # Match by filename_stem or url
    matched_source: Optional[Dict[str, Any]] = None
    if cleaned_input in approved_by_stem:
        matched_source = approved_by_stem[cleaned_input]
    elif cleaned_input in approved_by_url:
        matched_source = approved_by_url[cleaned_input]
    else:
        # Check URL normalization (trailing slashes)
        normalized_input = cleaned_input.rstrip("/")
        for url, s in approved_by_url.items():
            if url.rstrip("/") == normalized_input:
                matched_source = s
                break

    # Security check: Strictly reject unapproved / arbitrary inputs
    if matched_source is None:
        logger.warning("Rejected unapproved metadata request for: '%s'", cleaned_input)
        return {
            "success": False,
            "status": "REJECTED",
            "error": (
                f"Access denied: '{cleaned_input}' is not an approved BIS source. "
                "Only sources registered in the official manifest are permitted."
            ),
            "input": cleaned_input,
            "approved_sources": sorted(list(approved_by_stem.keys())),
        }

    stem = matched_source.get("filename_stem", "")
    metadata_path = raw_path / f"{stem}.metadata.json"
    doc_type = matched_source.get("document_type", "html")
    raw_ext = ".html" if doc_type == "html" else ".pdf"
    snapshot_path = raw_path / f"{stem}{raw_ext}"

    stored_metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to parse metadata file '%s': %s", metadata_path, exc)
            return {
                "success": False,
                "status": "ERROR",
                "error": f"Failed to read metadata for '{stem}': {exc}",
                "source_id": stem,
            }
    else:
        return {
            "success": False,
            "status": "MISSING_METADATA",
            "error": f"Metadata snapshot for '{stem}' not found on disk.",
            "source_id": stem,
        }

    return {
        "success": True,
        "status": "APPROVED",
        "source_id": stem,
        "title": matched_source.get("title"),
        "url": matched_source.get("url"),
        "category": matched_source.get("category"),
        "document_type": doc_type,
        "local_snapshot_exists": snapshot_path.exists(),
        "metadata": stored_metadata,
    }
