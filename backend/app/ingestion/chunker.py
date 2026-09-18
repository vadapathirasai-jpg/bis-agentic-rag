"""Document chunking module for BIS Sahayak.

Splits structured extracted document records into retrieval-friendly, semantic
chunks with preserved metadata, deterministic IDs, and structure-aware overlap.
Operates purely on local records without network access or LLMs.
"""

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Configurable initial defaults (in characters, can be extended to tokens)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150


def generate_chunk_id(
    source_url: Optional[str],
    section: Optional[str],
    chunk_index: int,
    chunk_text: str,
) -> str:
    """Generate a deterministic, stable chunk identifier.

    Args:
        source_url: The official BIS source URL.
        section: Section heading or FAQ question.
        chunk_index: The chunk's index within its parent record.
        chunk_text: The textual content of the chunk.

    Returns:
        Deterministic 16-character hexadecimal hash string.
    """
    key = f"{source_url or ''}|{section or ''}|{chunk_index}|{chunk_text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def split_text_with_overlap(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split text into overlapping segments respecting paragraph, sentence, and word boundaries.

    Args:
        text: Raw input text.
        chunk_size: Target maximum chunk size in characters.
        overlap: Character overlap between consecutive split chunks.

    Returns:
        List of non-empty text chunks.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # If text is already within size, return as single chunk without splitting
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned_text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            final_chunk = cleaned_text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        # Search window for natural boundary
        min_search = max(0, chunk_size - overlap - 150)
        window = cleaned_text[start:end]

        # 1. Prefer paragraph break (\n\n)
        para_idx = window.rfind("\n\n", min_search)
        if para_idx != -1:
            split_point = start + para_idx + 2
        else:
            # 2. Prefer sentence boundary (.!? followed by space or newline)
            sentence_match = None
            for match in re.finditer(r"([.!?])(?:\s+|\n)", window[min_search:]):
                sentence_match = match
            if sentence_match:
                split_point = start + min_search + sentence_match.end()
            else:
                # 3. Prefer newline
                nl_idx = window.rfind("\n", min_search)
                if nl_idx != -1:
                    split_point = start + nl_idx + 1
                else:
                    # 4. Prefer word boundary (space)
                    space_idx = window.rfind(" ", min_search)
                    if space_idx != -1:
                        split_point = start + space_idx + 1
                    else:
                        # Fallback to hard limit if no spaces found
                        split_point = end

        chunk = cleaned_text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        # Step back 'overlap' characters from split_point for next chunk
        next_start = max(start + 1, split_point - overlap)

        # Align next_start to a word boundary to avoid slicing words
        if next_start < text_len:
            space_fwd = cleaned_text.find(" ", next_start, next_start + 40)
            if space_fwd != -1 and space_fwd < split_point:
                next_start = space_fwd + 1

        start = next_start

    return chunks


def chunk_record(
    record: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """Convert a single structured document record into one or more chunks.

    Preserves all citation metadata and attaches deterministic chunk IDs.

    Args:
        record: Structured record from loader.py.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap size in characters for long records.

    Returns:
        List of chunk dictionaries with metadata.
    """
    text = record.get("text", "")
    text_segments = split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)

    if not text_segments:
        return []

    total_chunks = len(text_segments)
    chunks: List[Dict[str, Any]] = []

    for idx, segment in enumerate(text_segments):
        chunk_id = generate_chunk_id(
            source_url=record.get("source_url"),
            section=record.get("section"),
            chunk_index=idx,
            chunk_text=segment,
        )

        chunk_dict: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "text": segment,
            "source": record.get("source", "BIS"),
            "title": record.get("title"),
            "source_url": record.get("source_url"),
            "category": record.get("category"),
            "document_type": record.get("document_type"),
            "page": record.get("page"),
            "section": record.get("section"),
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "char_count": len(segment),
        }
        chunks.append(chunk_dict)

    return chunks


def chunk_records(
    records: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """Convert a collection of structured document records into retrieval-friendly chunks.

    Args:
        records: List of structured records from loader.py.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap size in characters for long records.

    Returns:
        Aggregated list of all generated chunk dictionaries.
    """
    all_chunks: List[Dict[str, Any]] = []

    for record in records:
        record_chunks = chunk_record(record, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(record_chunks)

    return all_chunks


def save_chunks(
    chunks: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save generated chunks and their metadata to the processed data directory.

    Args:
        chunks: List of chunk dictionaries.
        output_path: Target JSON file path. Defaults to backend/data/processed/bis_consumer_chunks.json.

    Returns:
        Path of the saved file.
    """
    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        path = base_dir / "data" / "processed" / "bis_consumer_chunks.json"
    else:
        path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return path


def load_chunks(file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load previously processed chunks from a JSON file.

    Args:
        file_path: Path to the JSON chunks file. Defaults to backend/data/processed/bis_consumer_chunks.json.

    Returns:
        List of chunk dictionaries.
    """
    if file_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        path = base_dir / "data" / "processed" / "bis_consumer_chunks.json"
    else:
        path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Processed chunks file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DocumentChunker:
    """Configurable document chunker for BIS Sahayak ingestion pipeline."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single record using configured parameters."""
        return chunk_record(record, chunk_size=self.chunk_size, overlap=self.overlap)

    def chunk_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple records using configured parameters."""
        return chunk_records(records, chunk_size=self.chunk_size, overlap=self.overlap)

    def process_and_save(
        self,
        records: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Chunk records and persist them to backend/data/processed/."""
        chunks = self.chunk_records(records)
        return save_chunks(chunks, output_path=output_path)


if __name__ == "__main__":
    import sys

    # Ensure backend path is available for imports
    base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from app.ingestion.loader import DocumentLoader

    print("=" * 60)
    print("BIS Sahayak - Document Chunker Test")
    print("=" * 60)

    # 1. Load existing local BIS records through loader.py
    raw_dir = base_dir / "data" / "raw"
    loader = DocumentLoader(raw_dir)
    records = loader.load_from_directory()

    # 2. Chunk records using DocumentChunker
    chunker = DocumentChunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
    chunks = chunker.chunk_records(records)

    # 3. Save to backend/data/processed/
    output_file = chunker.process_and_save(records)

    # 4. Compute statistics
    input_count = len(records)
    output_count = len(chunks)
    lengths = [c["char_count"] for c in chunks]
    avg_len = sum(lengths) / output_count if output_count else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    print(f"\nInput records: {input_count}")
    print(f"Generated chunks: {output_count}")
    print(f"Average chunk length: {avg_len:.1f} characters")
    print(f"Minimum chunk length: {min_len} characters")
    print(f"Maximum chunk length: {max_len} characters")
    print(f"Saved chunks to: {output_file}")

    # 5. Representative examples
    print("\n" + "=" * 60)
    print("REPRESENTATIVE CHUNK EXAMPLES")
    print("=" * 60)

    # Example 1: FAQ Chunk (Question & Answer preserved together)
    faq_chunks = [c for c in chunks if c.get("section") and ("?" in c["section"] or "What" in c["section"])]
    if faq_chunks:
        c_faq = faq_chunks[0]
        print("\n[Example 1: FAQ Chunk (Q&A Intact)]")
        print(f"Chunk ID:     {c_faq['chunk_id']}")
        print(f"Title:        {c_faq['title']}")
        print(f"Source URL:   {c_faq['source_url']}")
        print(f"Category:     {c_faq['category']}")
        print(f"Section:      {c_faq['section']}")
        print(f"Page:         {c_faq['page']}")
        print(f"Index:        {c_faq['chunk_index'] + 1} of {c_faq['total_chunks']}")
        print(f"Char Count:   {c_faq['char_count']}")
        print(f"Text:\n{'-'*40}\n{c_faq['text']}\n{'-'*40}")

    # Example 2: Normal Informational Section Chunk
    normal_chunks = [c for c in chunks if c["total_chunks"] == 1 and not (c.get("section") and "?" in c["section"])]
    if normal_chunks:
        c_norm = normal_chunks[0]
        print("\n[Example 2: Normal Section Chunk]")
        print(f"Chunk ID:     {c_norm['chunk_id']}")
        print(f"Title:        {c_norm['title']}")
        print(f"Source URL:   {c_norm['source_url']}")
        print(f"Category:     {c_norm['category']}")
        print(f"Section:      {c_norm['section']}")
        print(f"Page:         {c_norm['page']}")
        print(f"Index:        {c_norm['chunk_index'] + 1} of {c_norm['total_chunks']}")
        print(f"Char Count:   {c_norm['char_count']}")
        print(f"Text:\n{'-'*40}\n{c_norm['text']}\n{'-'*40}")

    # Example 3: Long-Record Split Chunk
    split_chunks = [c for c in chunks if c["total_chunks"] > 1]
    if split_chunks:
        c_split = split_chunks[0]
        print("\n[Example 3: Long-Record Split Chunk (with Overlap)]")
        print(f"Chunk ID:     {c_split['chunk_id']}")
        print(f"Title:        {c_split['title']}")
        print(f"Source URL:   {c_split['source_url']}")
        print(f"Category:     {c_split['category']}")
        print(f"Section:      {c_split['section']}")
        print(f"Page:         {c_split['page']}")
        print(f"Index:        {c_split['chunk_index'] + 1} of {c_split['total_chunks']}")
        print(f"Char Count:   {c_split['char_count']}")
        print(f"Text:\n{'-'*40}\n{c_split['text'][:300]}...\n{'-'*40}")
