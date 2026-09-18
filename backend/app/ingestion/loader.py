"""Document loader module for BIS Sahayak.

Extracts structured text from raw, approved official BIS documents (HTML and PDF),
preserving full source traceability, section headings, and citation metadata.
Does NOT fetch from the internet or generate embeddings.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SUPPORTED_HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_HTML_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


def load_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata JSON associated with a raw document.

    Args:
        metadata_path: Path to the .metadata.json file or the raw file itself.

    Returns:
        Dictionary containing source metadata.
    """
    path = Path(metadata_path)

    if not path.name.endswith(".metadata.json"):
        path = path.with_name(f"{path.stem}.metadata.json")

    if not path.exists():
        logger.warning("Metadata file not found: %s", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        logger.warning("Failed to read metadata from %s: %s", path, err)
        return {}


def load_html(
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract structured sections and text from a local raw BIS HTML document.

    Preserves headings, questions, answers, and source metadata for citation.

    Args:
        file_path: Path to the local HTML file.
        metadata: Optional metadata dictionary. If not provided, attempts to load
                  from the corresponding .metadata.json file.

    Returns:
        List of structured record dictionaries.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found at: {path}")

    meta = metadata.copy() if metadata else load_metadata(path)
    title = meta.get("title", path.stem.replace("_", " ").title())
    source_url = meta.get("source_url")
    category = meta.get("category", "general")
    source = meta.get("source", "BIS")
    doc_type = "html"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    # 1. Remove non-content elements
    for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        element.decompose()

    # 2. Locate main content container
    content_root = soup.find("div", id="skip-to-main-content") or soup.find("main") or soup.body
    if content_root is None:
        return []

    # 3. Remove navigation / sidebar boilerplate inside the container
    for boilerplate in content_root.find_all(
        class_=lambda c: c
        and any(
            k in str(c).lower()
            for k in [
                "accordion-menu",
                "leftmenu",
                "breadcrumb",
                "bread-crumb",
                "screen-reader",
                "hidden-module",
                "quick_link",
                "footer_about",
            ]
        )
    ):
        boilerplate.decompose()

    records: List[Dict[str, Any]] = []

    # 4. Check for FAQ accordion pattern (div.accordion paired with div.panel)
    accordions = content_root.find_all("div", class_="accordion")

    if accordions:
        # Check for overview text before the first accordion
        first_acc = accordions[0]
        intro_paragraphs: List[str] = []

        for prev in first_acc.find_all_previous(["h1", "h2", "p"]):
            if prev in content_root.descendants:
                txt = prev.get_text(" ", strip=True)
                if txt and "MAIN MENU" not in txt and txt not in intro_paragraphs:
                    intro_paragraphs.append(txt)

        # Reverse to maintain original order since find_all_previous goes backwards
        intro_paragraphs.reverse()
        # Filter out breadcrumb text like "Home / ..."
        filtered_intro = [p for p in intro_paragraphs if not p.startswith("Home /") and len(p) > 5]

        if filtered_intro:
            records.append({
                "text": "\n\n".join(filtered_intro),
                "source": source,
                "title": title,
                "source_url": source_url,
                "category": category,
                "document_type": doc_type,
                "page": None,
                "section": "Overview",
            })

        # Process accordion Q&A units
        for acc in accordions:
            q_text = acc.get_text(" ", strip=True)
            if not q_text:
                continue

            # Find matching answer panel
            panel = acc.find_next_sibling(class_="panel")
            a_text = panel.get_text("\n", strip=True) if panel else ""

            # Clean extra newlines
            a_text = re.sub(r"\n{3,}", "\n\n", a_text)
            full_text = f"Question: {q_text}\n\nAnswer:\n{a_text}" if a_text else q_text

            records.append({
                "text": full_text,
                "source": source,
                "title": title,
                "source_url": source_url,
                "category": category,
                "document_type": doc_type,
                "page": None,
                "section": q_text,
            })

    else:
        # 5. Heading-based sectioning for standard informational pages
        current_section: Optional[str] = None
        current_blocks: List[str] = []

        elements = content_root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"]
        )

        for el in elements:
            text_val = el.get_text(" ", strip=True)
            if not text_val or "MAIN MENU" in text_val or text_val.startswith("Home /"):
                continue

            if el.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # Commit previous section
                if current_blocks:
                    combined_text = "\n\n".join(current_blocks).strip()
                    if combined_text:
                        records.append({
                            "text": combined_text,
                            "source": source,
                            "title": title,
                            "source_url": source_url,
                            "category": category,
                            "document_type": doc_type,
                            "page": None,
                            "section": current_section,
                        })
                    current_blocks = []
                current_section = text_val

            elif el.name in ["p", "table"]:
                current_blocks.append(text_val)

            elif el.name in ["ul", "ol"]:
                items = [
                    f"- {li.get_text(' ', strip=True)}"
                    for li in el.find_all("li", recursive=False)
                    if li.get_text(strip=True)
                ]
                if items:
                    current_blocks.append("\n".join(items))

        # Commit final section
        if current_blocks:
            combined_text = "\n\n".join(current_blocks).strip()
            if combined_text:
                records.append({
                    "text": combined_text,
                    "source": source,
                    "title": title,
                    "source_url": source_url,
                    "category": category,
                    "document_type": doc_type,
                    "page": None,
                    "section": current_section,
                })

    # If no records were created but content exists, fallback to full text
    if not records:
        raw_text = content_root.get_text("\n", strip=True)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        if raw_text:
            records.append({
                "text": raw_text,
                "source": source,
                "title": title,
                "source_url": source_url,
                "category": category,
                "document_type": doc_type,
                "page": None,
                "section": None,
            })

    return records


def load_pdf(
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract page-level structured records from a local raw BIS PDF document.

    Supports PyMuPDF (fitz) with automatic pure-Python fallback (pypdf) when
    native DLLs are restricted by system security policies.

    Args:
        file_path: Path to the local PDF file.
        metadata: Optional metadata dictionary. If not provided, attempts to load
                  from the corresponding .metadata.json file.

    Returns:
        List of structured record dictionaries, one per non-empty page.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    meta = metadata.copy() if metadata else load_metadata(path)
    title = meta.get("title", path.stem.replace("_", " ").title())
    source_url = meta.get("source_url")
    category = meta.get("category", "general")
    source = meta.get("source", "BIS")
    doc_type = "pdf"

    records: List[Dict[str, Any]] = []

    # Attempt primary loader: PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                cleaned_text = page_text.strip()

                if not cleaned_text:
                    continue

                records.append({
                    "text": cleaned_text,
                    "source": source,
                    "title": title,
                    "source_url": source_url,
                    "category": category,
                    "document_type": doc_type,
                    "page": page_num + 1,
                    "section": None,
                })
        finally:
            doc.close()

        return records

    except (ImportError, Exception) as fitz_err:
        logger.info("PyMuPDF unavailable (%s); attempting pure-Python pypdf loader.", fitz_err)

    # Fallback loader: pure-Python pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()

            if not cleaned_text:
                continue

            records.append({
                "text": cleaned_text,
                "source": source,
                "title": title,
                "source_url": source_url,
                "category": category,
                "document_type": doc_type,
                "page": page_num + 1,
                "section": None,
            })

        return records

    except Exception as pypdf_err:
        raise RuntimeError(
            f"Failed to extract PDF text from {path}. Neither PyMuPDF nor pypdf could process it: {pypdf_err}"
        ) from pypdf_err


def load_document(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a raw BIS document (HTML or PDF) and return structured records.

    Automatically identifies document type from file extension and loads
    corresponding citation metadata.

    Args:
        file_path: Path to the local raw document file.

    Returns:
        List of structured record dictionaries.

    Raises:
        FileNotFoundError: If the target file does not exist.
        ValueError: If the file extension is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found at: {path}")

    ext = path.suffix.lower()

    if ext in SUPPORTED_HTML_EXTENSIONS:
        return load_html(path)
    elif ext in SUPPORTED_PDF_EXTENSIONS:
        return load_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}' for file '{path.name}'. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )


class DocumentLoader:
    """Modular document loader for ingesting official BIS documents."""

    def __init__(self, raw_data_dir: Optional[Union[str, Path]] = None) -> None:
        if raw_data_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
            self.raw_data_dir = base_dir / "data" / "raw"
        else:
            self.raw_data_dir = Path(raw_data_dir)

    def load_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load a single document."""
        return load_document(file_path)

    def load_html(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Load an HTML document."""
        return load_html(file_path, metadata)

    def load_pdf(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Load a PDF document."""
        return load_pdf(file_path, metadata)

    def load_from_directory(
        self, directory: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """Scan a directory and load all supported documents.

        Args:
            directory: Directory to scan. Defaults to raw_data_dir.

        Returns:
            Aggregated list of all extracted records.
        """
        target_dir = Path(directory) if directory else self.raw_data_dir
        all_records: List[Dict[str, Any]] = []

        if not target_dir.exists():
            return all_records

        for file_path in sorted(target_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                all_records.extend(self.load_document(file_path))

        return all_records


if __name__ == "__main__":
    print("=" * 60)
    print("BIS Sahayak - Document Loader Test")
    print("=" * 60)

    # 1. Test HTML extraction with consumer_faq.html
    base_dir = Path(__file__).resolve().parent.parent.parent  # backend/
    raw_dir = base_dir / "data" / "raw"
    faq_path = raw_dir / "consumer_faq.html"

    if faq_path.exists():
        print(f"\nLoading HTML document: {faq_path.name}")
        records = load_document(faq_path)
        print(f"Document title: {records[0]['title'] if records else 'N/A'}")
        print(f"Number of extracted records: {len(records)}")
        print(f"Source URL: {records[0]['source_url'] if records else 'N/A'}")
        print(f"Document type: {records[0]['document_type'] if records else 'N/A'}")

        print("\nFirst 3 Section Headings:")
        for idx, rec in enumerate(records[:3], 1):
            print(f"  {idx}. {rec.get('section')}")

        if records:
            print("\nFirst 300 characters of extracted text:")
            sample_text = records[0]["text"][:300]
            print("-" * 40)
            print(sample_text)
            print("-" * 40)
    else:
        print(f"\nExpected test file not found: {faq_path}")

    # 2. Test PDF extraction if available
    pdf_files = list(raw_dir.glob("*.pdf")) if raw_dir.exists() else []
    if pdf_files:
        sample_pdf = pdf_files[0]
        print(f"\nLoading PDF document: {sample_pdf.name}")
        pdf_records = load_document(sample_pdf)
        print(f"Number of extracted PDF pages: {len(pdf_records)}")
        if pdf_records:
            print(f"Page 1 snippet: {pdf_records[0]['text'][:200]}...")
    else:
        print("\nNo local BIS PDF available for PDF extraction test.")