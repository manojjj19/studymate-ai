"""PDF Reader and Text Chunking Module for StudyMate AI.

Extracts text from PDF documents page-by-page using PyMuPDF (fitz)
and breaks long text into overlapping chunks while preserving page numbers.
"""

from dataclasses import dataclass, asdict
import io
import re
from typing import List, Dict, Any, Union, Optional
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDFProcessingError(Exception):
    """Custom exception raised when a PDF cannot be processed."""
    pass


class ScannedPDFWarning(PDFProcessingError):
    """Raised when a PDF contains pages but no extractable digital text layer."""
    pass


@dataclass
class DocumentChunk:
    """Represents a discrete chunk of text extracted from a document."""
    chunk_id: int
    text: str
    page_number: int  # 1-based index
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_text(text: str) -> str:
    """Cleans up excess whitespace, line breaks, and non-printable characters.
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned normalized text string.
    """
    if not text:
        return ""
    text = text.replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def extract_pages(
    file_input: Union[str, bytes, io.BytesIO],
    source_name: str = "document.pdf"
) -> List[Dict[str, Any]]:
    """Extracts text page-by-page from a PDF file path, byte buffer, or BytesIO.
    
    Args:
        file_input: File path string, raw bytes, or BytesIO buffer.
        source_name: Human-readable filename for tracking citations.
        
    Returns:
        A list of dicts: [{"page_number": int, "text": str, "source": str}, ...]
        
    Raises:
        PDFProcessingError: If the document is corrupted or invalid.
        ScannedPDFWarning: If no extractable text is found across any pages.
    """
    if fitz is None:
        raise PDFProcessingError(
            "PyMuPDF (fitz) is not installed in the environment. "
            "Please run 'pip install pymupdf' to enable PDF text extraction."
        )

    doc = None
    try:
        if isinstance(file_input, str):
            doc = fitz.open(file_input)
        elif isinstance(file_input, bytes):
            if len(file_input) == 0:
                raise PDFProcessingError("Uploaded file is empty (0 bytes).")
            doc = fitz.open(stream=file_input, filetype="pdf")
        elif hasattr(file_input, "read"):
            data = file_input.read()
            if len(data) == 0:
                raise PDFProcessingError("Uploaded file buffer is empty (0 bytes).")
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            raise PDFProcessingError(f"Unsupported file input type: {type(file_input)}")

        if doc.page_count == 0:
            raise PDFProcessingError("The PDF document contains 0 pages.")

        pages_data = []
        total_text_length = 0

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text") or ""
            cleaned = clean_text(raw_text)
            total_text_length += len(cleaned)

            pages_data.append({
                "page_number": page_idx + 1,  # 1-indexed for student-facing citations
                "text": cleaned,
                "source": source_name
            })

        # Detect scanned/image-only PDFs where no text layer exists
        if total_text_length == 0:
            raise ScannedPDFWarning(
                "No extractable text was found in this PDF. It appears to be a scanned document "
                "or image-only file. OCR (Optical Character Recognition) is not supported in this "
                "initial version of StudyMate AI. Please upload a digital text-based PDF."
            )

        return pages_data

    except fitz.FileDataError as e:
        raise PDFProcessingError(f"Invalid or corrupted PDF file: {str(e)}") from e
    except (PDFProcessingError, ScannedPDFWarning):
        raise
    except Exception as e:
        raise PDFProcessingError(f"Unexpected error while opening PDF: {str(e)}") from e
    finally:
        if doc is not None:
            doc.close()


def chunk_text_by_words(
    text: str,
    page_number: int,
    source: str,
    chunk_size: int = 150,
    overlap: int = 30,
    start_chunk_id: int = 0
) -> List[DocumentChunk]:
    """Splits a single block of page text into word-based chunks with overlap.
    
    Args:
        text: Text to chunk.
        page_number: Originating page number (1-based).
        source: Originating filename.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.
        start_chunk_id: Starting integer ID for chunks.
        
    Returns:
        List of DocumentChunk instances.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    chunk_id = start_chunk_id

    # If the text is smaller than or equal to the chunk size, keep it as one chunk
    if len(words) <= chunk_size:
        return [DocumentChunk(
            chunk_id=chunk_id,
            text=" ".join(words),
            page_number=page_number,
            source=source
        )]

    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        # Only add chunk if it contains meaningful content
        if chunk_words:
            chunk_str = " ".join(chunk_words)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_str,
                page_number=page_number,
                source=source
            ))
            chunk_id += 1

    return chunks


def process_pdf_to_chunks(
    file_input: Union[str, bytes, io.BytesIO],
    source_name: str = "document.pdf",
    chunk_size_words: int = 150,
    overlap_words: int = 30
) -> List[DocumentChunk]:
    """End-to-end pipeline: extracts PDF text and produces structured chunks.
    
    Args:
        file_input: PDF path, bytes, or file-like object.
        source_name: Name of the PDF file.
        chunk_size_words: Words per chunk (default: 150).
        overlap_words: Overlap words between chunks (default: 30).
        
    Returns:
        List of DocumentChunk objects with accurate page numbers.
    """
    pages_data = extract_pages(file_input, source_name=source_name)
    all_chunks: List[DocumentChunk] = []
    current_id = 0

    for page_info in pages_data:
        page_text = page_info["text"]
        page_num = page_info["page_number"]
        source = page_info["source"]

        if not page_text:
            continue

        page_chunks = chunk_text_by_words(
            text=page_text,
            page_number=page_num,
            source=source,
            chunk_size=chunk_size_words,
            overlap=overlap_words,
            start_chunk_id=current_id
        )
        all_chunks.extend(page_chunks)
        current_id += len(page_chunks)

    return all_chunks
