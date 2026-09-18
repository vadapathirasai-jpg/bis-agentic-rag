"""Source fetcher module for BIS Sahayak.

Responsible for securely fetching authorized official resources from official BIS
domains, saving original raw documents (HTML/PDF), and recording source metadata
for downstream citation and retrieval.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Security: Strict domain allowlist for official BIS resources
ALLOWED_BIS_DOMAINS = {
    "www.bis.gov.in",
    "bis.gov.in",
}

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 (BIS-Sahayak/1.0)"
)
DEFAULT_TIMEOUT_SECONDS = 30

REQUIRED_MANIFEST_FIELDS = {"title", "url", "category", "document_type"}


class SourceFetcherError(Exception):
    """Base exception for source fetching errors."""


class InvalidDomainError(SourceFetcherError):
    """Raised when a URL is not from an approved official BIS domain."""


class UnsupportedContentTypeError(SourceFetcherError):
    """Raised when server response is neither HTML nor PDF."""


class DownloadError(SourceFetcherError):
    """Raised when HTTP request or download fails."""


def validate_bis_url(url: str) -> None:
    """Validate that the given URL belongs to an approved official BIS domain.

    Args:
        url: The web URL to inspect.

    Raises:
        InvalidDomainError: If domain is not in ALLOWED_BIS_DOMAINS.
        SourceFetcherError: If URL structure is invalid.
    """
    if not url or not isinstance(url, str):
        raise SourceFetcherError("URL must be a non-empty string.")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SourceFetcherError(f"Invalid URL scheme '{parsed.scheme}'. Only HTTP/HTTPS allowed.")

    netloc = parsed.netloc.lower().split(":")[0]  # strip port if present
    if netloc not in ALLOWED_BIS_DOMAINS:
        raise InvalidDomainError(
            f"Domain '{netloc}' is rejected. Only authorized official BIS domains "
            f"({', '.join(sorted(ALLOWED_BIS_DOMAINS))}) are allowed."
        )


def sanitize_filename(name: str) -> str:
    """Convert arbitrary strings into safe filenames.

    Args:
        name: Desired name or title.

    Returns:
        Safe filesystem-compatible string.
    """
    cleaned = re.sub(r"[^\w\-]", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "bis_resource"


def detect_document_type(content_type_header: str, url: str) -> str:
    """Detect whether response is HTML or PDF based on headers and URL.

    Args:
        content_type_header: The Content-Type header from HTTP response.
        url: The resource URL.

    Returns:
        'html' or 'pdf'.

    Raises:
        UnsupportedContentTypeError: If type cannot be recognized as HTML or PDF.
    """
    header = (content_type_header or "").lower()
    url_lower = url.lower()

    if "text/html" in header or "application/xhtml+xml" in header:
        return "html"
    elif "application/pdf" in header or url_lower.endswith(".pdf"):
        return "pdf"
    elif "application/octet-stream" in header and url_lower.endswith(".pdf"):
        return "pdf"
    else:
        raise UnsupportedContentTypeError(
            f"Unsupported Content-Type '{content_type_header}'. Only HTML and PDF are supported."
        )


def load_sources(manifest_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load and validate the approved sources manifest.

    Args:
        manifest_path: Path to sources JSON manifest. Defaults to backend/data/sources.json.

    Returns:
        List of validated source definition dictionaries.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        SourceFetcherError: If JSON structure or schema is invalid.
    """
    if manifest_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        path = base_dir / "data" / "sources.json"
    else:
        path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Sources manifest not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise SourceFetcherError(f"Invalid JSON format in sources manifest '{path}': {err}") from err

    if not isinstance(data, list):
        raise SourceFetcherError(f"Sources manifest '{path}' must contain a JSON array of sources.")

    validated_sources: List[Dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SourceFetcherError(f"Item #{index} in manifest is not a valid JSON object.")

        missing_fields = REQUIRED_MANIFEST_FIELDS - set(item.keys())
        if missing_fields:
            raise SourceFetcherError(
                f"Source #{index} ('{item.get('title', 'Unknown')}') is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        validated_sources.append(item)

    return validated_sources


def fetch_resource(
    url: str,
    output_dir: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Fetch an official BIS resource and save the original file and metadata.

    If the resource already exists and is non-empty, it is skipped unless force_refresh=True.

    Args:
        url: The official BIS resource URL.
        output_dir: Directory where raw files will be stored. Defaults to backend/data/raw.
        metadata: Optional metadata dictionary (e.g. title, category, filename_stem).
        timeout: HTTP request timeout in seconds.
        force_refresh: If True, re-download and overwrite existing files.

    Returns:
        Complete metadata dictionary including local file paths and download details.

    Raises:
        InvalidDomainError: If URL is outside the official BIS domains.
        DownloadError: If the HTTP request fails or content is empty.
        UnsupportedContentTypeError: If content type is neither HTML nor PDF.
    """
    # 1. Validate domain security
    validate_bis_url(url)

    metadata_in = metadata.copy() if metadata else {}
    title = metadata_in.get("title", "BIS Resource")
    category = metadata_in.get("category", "general")
    expected_doc_type = metadata_in.get("document_type")
    filename_stem = metadata_in.get("filename_stem")

    if not filename_stem:
        path_stem = Path(urlparse(url).path).stem
        base_stem = path_stem or title
        if category and category not in base_stem:
            base_stem = f"{category}_{base_stem}"
        filename_stem = sanitize_filename(base_stem)

    # 2. Resolve output directory
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        target_dir = base_dir / "data" / "raw"
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Check for existing valid download to avoid duplicate network requests
    metadata_filename = f"{filename_stem}.metadata.json"
    metadata_file_path = target_dir / metadata_filename

    # Determine candidate file extension if known in advance
    candidate_ext = f".{expected_doc_type.lower()}" if expected_doc_type else None
    candidate_raw_file = (target_dir / f"{filename_stem}{candidate_ext}") if candidate_ext else None

    if not force_refresh and metadata_file_path.exists():
        # Check if paired raw file exists and is non-empty
        try:
            with open(metadata_file_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)

            recorded_local = existing_meta.get("local_file")
            if recorded_local:
                actual_file = target_dir / Path(recorded_local).name
                if actual_file.exists() and actual_file.stat().st_size > 0:
                    print("\nSkipping (already downloaded):")
                    print(str(actual_file))
                    print(f"Source URL: {url}")
                    print(f"Metadata file: {metadata_file_path}")
                    return existing_meta
        except Exception:
            # If existing metadata is corrupted, continue to re-download
            pass

    # 4. Log start of fetch
    action_label = "Refreshing (overwriting existing):" if force_refresh and candidate_raw_file and candidate_raw_file.exists() else "Fetching:"
    print(f"\n{action_label}")
    print(url)

    # 5. HTTP GET Request
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout as err:
        raise DownloadError(f"Request timed out while accessing {url}: {err}") from err
    except requests.exceptions.RequestException as err:
        raise DownloadError(f"HTTP request failed for {url}: {err}") from err

    # 6. Check response content
    raw_content = response.content
    if not raw_content:
        raise DownloadError(f"Received empty response content from {url}.")

    # 7. Identify document type
    content_type_header = response.headers.get("Content-Type", "")
    doc_type = detect_document_type(content_type_header, url)
    extension = ".html" if doc_type == "html" else ".pdf"

    # 8. Save original raw file
    raw_filename = f"{filename_stem}{extension}"
    raw_file_path = target_dir / raw_filename

    with open(raw_file_path, "wb") as f:
        f.write(raw_content)

    # 9. Build metadata record
    retrieved_at = datetime.now(timezone.utc).isoformat()
    relative_local_file = f"backend/data/raw/{raw_filename}"

    final_metadata: Dict[str, Any] = {
        "title": title,
        "source_url": url,
        "category": category,
        "document_type": doc_type,
        "retrieved_at": retrieved_at,
        "local_file": relative_local_file,
        "file_size_bytes": len(raw_content),
        "content_type_header": content_type_header,
        "source": "BIS",
    }

    # 10. Save metadata JSON
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=4, ensure_ascii=False)

    # 11. Console output
    print("\nDownloaded:")
    print(str(raw_file_path))
    print("\nDocument type:")
    print(doc_type.upper())
    print("\nSource:")
    print("BIS")
    print(f"\nMetadata saved to:\n{metadata_file_path}")

    return final_metadata


def fetch_sources(
    manifest_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch all approved resources listed in the sources manifest.

    Args:
        manifest_path: Path to sources.json. Defaults to backend/data/sources.json.
        output_dir: Output directory for raw assets. Defaults to backend/data/raw.
        force_refresh: If True, re-downloads resources even if they exist locally.

    Returns:
        List of metadata records for all processed resources.
    """
    sources = load_sources(manifest_path)
    print(f"Loaded {len(sources)} approved source definitions from manifest.")

    results: List[Dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        print(f"\n[{index}/{len(sources)}] Processing: {source['title']}")
        url = source["url"]
        meta_payload = source.copy()

        result = fetch_resource(
            url=url,
            output_dir=output_dir,
            metadata=meta_payload,
            force_refresh=force_refresh,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BIS Sahayak - Approved Sources Ingestion Test")
    print("=" * 60)

    # 1. Test fetching from manifest
    try:
        manifest_results = fetch_sources()
        print("\n" + "=" * 60)
        print(f"Manifest processing complete. Total sources: {len(manifest_results)}")
        for item in manifest_results:
            print(f"- {item['title']}: {item['local_file']} ({item['file_size_bytes']} bytes)")
        print("=" * 60)
    except Exception as exc:
        print(f"\nManifest fetch test failed: {exc}")
        raise

    # 2. Security validation check: Reject unauthorized domains
    print("\nTesting domain security enforcement...")
    try:
        fetch_resource("https://example.com/test")
        print("SECURITY FAILURE: Unauthorized domain was not rejected!")
        raise AssertionError("Domain validation failed to reject third-party URL.")
    except InvalidDomainError as sec_err:
        print(f"SECURITY PASS: Unauthorized domain correctly rejected: {sec_err}")
