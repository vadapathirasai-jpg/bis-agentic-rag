"""Embedding service module for BIS Sahayak.

Generates dense vector embeddings using Sentence Transformers for document chunks
and user search queries. Operates locally with batch encoding and metadata preservation.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load processed document chunks from a JSON file.

    Args:
        chunks_path: Path to the chunks JSON file. Defaults to
                     backend/data/processed/bis_consumer_chunks.json.

    Returns:
        List of chunk dictionaries.

    Raises:
        FileNotFoundError: If the chunks file does not exist.
        ValueError: If the file content is not a valid list of chunks.
    """
    if chunks_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        path = base_dir / "data" / "processed" / "bis_consumer_chunks.json"
    else:
        path = Path(chunks_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at: {path}. "
            "Please ensure chunker.py has been run to generate the processed chunks."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse chunks JSON from '{path}': {err}") from err

    if not isinstance(data, list):
        raise ValueError(f"Chunks file '{path}' must contain a JSON array of chunk objects.")

    return data


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model.

    Checks local cached path first, falling back to HuggingFace hub identifier.

    Args:
        model_name: Hugging Face model identifier or local path.

    Returns:
        Loaded SentenceTransformer instance.

    Raises:
        RuntimeError: If model loading fails.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
    local_model_dir = base_dir / "data" / "models" / "all-MiniLM-L6-v2"

    candidate_target = None
    if Path(model_name).exists():
        candidate_target = str(model_name)
    elif "all-MiniLM-L6-v2" in model_name and local_model_dir.exists():
        candidate_target = str(local_model_dir)

    target = candidate_target or model_name

    try:
        logger.info("Loading embedding model from '%s'...", target)
        return SentenceTransformer(target)
    except Exception as err:
        if candidate_target and candidate_target != model_name:
            try:
                return SentenceTransformer(model_name)
            except Exception as inner_err:
                raise RuntimeError(
                    f"Failed to load embedding model '{model_name}': {inner_err}"
                ) from inner_err
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}': {err}"
        ) from err


def get_model_dimension(model: SentenceTransformer) -> int:
    """Determine the embedding dimension dynamically from the model."""
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    elif hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
    else:
        dim = 384
    return int(dim) if dim is not None else 384


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model: Optional[SentenceTransformer] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Generate dense vector embeddings for a list of document chunks.

    Encodes ONLY the 'text' field of each chunk in batches, preserves all
    existing chunk metadata, and attaches the embedding vector, model identifier,
    and embedding dimension.

    Args:
        chunks: List of chunk dictionaries from chunker.py.
        model: Optional pre-loaded SentenceTransformer model.
        model_name: Model identifier if model is to be loaded dynamically.
        batch_size: Batch size for model encoding.

    Returns:
        List of chunk dictionaries augmented with 'embedding',
        'embedding_model', and 'embedding_dimension'.
    """
    if not chunks:
        return []

    if model is None:
        model = load_embedding_model(model_name)

    # Determine embedding dimension dynamically from the loaded model
    dimension = get_model_dimension(model)

    # Extract pure text from chunks (metadata is NOT embedded)
    texts = [chunk.get("text", "") for chunk in chunks]

    # Batch encode texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embedded_chunks: List[Dict[str, Any]] = []

    for chunk, vector in zip(chunks, embeddings):
        vec_list = vector.tolist()

        # Build output preserving every original field
        record: Dict[str, Any] = {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "source": chunk.get("source", "BIS"),
            "title": chunk.get("title"),
            "source_url": chunk.get("source_url"),
            "category": chunk.get("category"),
            "document_type": chunk.get("document_type"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "chunk_index": chunk.get("chunk_index"),
            "total_chunks": chunk.get("total_chunks"),
            "char_count": chunk.get("char_count"),
            "embedding": vec_list,
            "embedding_model": model_name,
            "embedding_dimension": dimension,
        }
        embedded_chunks.append(record)

    return embedded_chunks


def save_embeddings(
    embedded_chunks: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save embedded chunks to a JSON file.

    Args:
        embedded_chunks: List of embedded chunk dictionaries.
        output_path: Target JSON file path. Defaults to
                     backend/data/processed/bis_consumer_embeddings.json.

    Returns:
        Saved file Path.
    """
    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
        path = base_dir / "data" / "processed" / "bis_consumer_embeddings.json"
    else:
        path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    return path


class EmbeddingService:
    """Embedding service encapsulation for BIS Sahayak ingestion and retrieval."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loaded SentenceTransformer model."""
        if self._model is None:
            self._model = load_embedding_model(self.model_name)
        return self._model

    def get_dimension(self) -> int:
        """Return the dynamic embedding dimension of the loaded model."""
        return get_model_dimension(self.model)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute embeddings for a list of string passages.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in vectors]

    def embed_query(self, query: str) -> List[float]:
        """Compute a single dense embedding vector for a search query.

        Args:
            query: User query string.

        Returns:
            Float vector embedding.
        """
        vector = self.model.encode(query, convert_to_numpy=True)
        return vector.tolist()

    def process_and_save(
        self,
        chunks: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
    ) -> Path:
        """Generate embeddings for chunks and save to disk."""
        embedded = generate_embeddings(
            chunks=chunks,
            model=self.model,
            model_name=self.model_name,
            batch_size=batch_size,
        )
        return save_embeddings(embedded, output_path=output_path)


if __name__ == "__main__":
    import sys

    # Ensure backend path is available for imports
    base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    print("=" * 60)
    print("BIS Sahayak - Text Embeddings Test")
    print("=" * 60)

    # 1. Load the 79 processed chunks
    chunks_path = base_dir / "data" / "processed" / "bis_consumer_chunks.json"
    print(f"\nLoading chunks from: {chunks_path}")
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    # 2. Load model & generate embeddings
    model_name = DEFAULT_MODEL_NAME
    print(f"\nLoading embedding model: {model_name}")
    model = load_embedding_model(model_name)

    # 3. Generate embeddings
    print("Generating batch embeddings...")
    embedded_chunks = generate_embeddings(chunks, model=model, model_name=model_name)

    # 4. Save to backend/data/processed/bis_consumer_embeddings.json
    output_path = base_dir / "data" / "processed" / "bis_consumer_embeddings.json"
    saved_path = save_embeddings(embedded_chunks, output_path=output_path)

    # 5. Verification checks
    num_chunks = len(chunks)
    num_embeddings = len(embedded_chunks)
    assert num_chunks == num_embeddings, f"Mismatch: {num_chunks} chunks vs {num_embeddings} embeddings"

    first_chunk = embedded_chunks[0]
    dim = first_chunk["embedding_dimension"]
    first_emb_len = len(first_chunk["embedding"])

    # Verify all embeddings have identical dimension
    for idx, c in enumerate(embedded_chunks):
        assert len(c["embedding"]) == dim, f"Chunk {idx} embedding dimension mismatch: {len(c['embedding'])} != {dim}"
        assert c["embedding_model"] == model_name, f"Chunk {idx} model mismatch"

    # 6. Report outputs as specified in requirements
    print("\n" + "=" * 60)
    print("EMBEDDING VERIFICATION REPORT")
    print("=" * 60)
    print(f"number of chunks:       {num_embeddings}")
    print(f"embedding model:        {model_name}")
    print(f"embedding dimension:    {dim}")
    print(f"first chunk ID:         {first_chunk['chunk_id']}")
    print(f"first embedding length: {first_emb_len}")
    print(f"saved output path:      {saved_path}")
    print("=" * 60)
    print("All embeddings generated and verified successfully!")
