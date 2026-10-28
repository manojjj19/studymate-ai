"""Unit tests for PDF extraction and text chunking in StudyMate AI."""

import io
import pytest

try:
    import fitz
except ImportError:
    fitz = None

from src.pdf_reader import (
    clean_text,
    chunk_text_by_words,
    extract_pages,
    process_pdf_to_chunks,
    DocumentChunk,
    PDFProcessingError,
    ScannedPDFWarning
)


def create_synthetic_pdf(pages_text: list) -> bytes:
    """Helper to generate an in-memory PDF byte stream for tests."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not available for generating synthetic PDF.")
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_clean_text():
    """Verify whitespace and line break normalization."""
    raw = "  Hello   world! \r\n\r\n\n This is a   test.  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world!\n\nThis is a test."


def test_chunk_text_by_words():
    """Verify word-level chunking and overlapping."""
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    chunks = chunk_text_by_words(
        text=text,
        page_number=3,
        source="notes.pdf",
        chunk_size=5,
        overlap=2,
        start_chunk_id=0
    )
    assert len(chunks) >= 2
    assert chunks[0].page_number == 3
    assert chunks[0].source == "notes.pdf"
    assert chunks[0].chunk_id == 0
    # Overlap test: word4 appears in both chunk 0 and chunk 1
    assert "word4" in chunks[0].text and "word4" in chunks[1].text


@pytest.mark.skipif(fitz is None, reason="PyMuPDF (fitz) is not installed")
def test_extract_pages_with_synthetic_pdf():
    """Verify multi-page PDF extraction preserving 1-based page numbers."""
    page1_text = "Chapter 1: Introduction to Photosynthesis. Plants convert light into energy."
    page2_text = "Chapter 2: Cellular Respiration. Mitochondria are the powerhouse of the cell."

    pdf_bytes = create_synthetic_pdf([page1_text, page2_text])
    pages = extract_pages(pdf_bytes, source_name="biology_notes.pdf")

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Photosynthesis" in pages[0]["text"]
    assert pages[0]["source"] == "biology_notes.pdf"

    assert pages[1]["page_number"] == 2
    assert "Cellular Respiration" in pages[1]["text"]


@pytest.mark.skipif(fitz is None, reason="PyMuPDF (fitz) is not installed")
def test_scanned_pdf_warning():
    """Verify that a PDF without extractable text raises ScannedPDFWarning."""
    pdf_bytes = create_synthetic_pdf([""])
    with pytest.raises(ScannedPDFWarning) as exc_info:
        extract_pages(pdf_bytes, source_name="scanned_scan.pdf")
    assert "scanned document" in str(exc_info.value).lower()


def test_empty_buffer_or_missing_fitz_error():
    """Verify that calling extract_pages with empty buffer or without fitz raises PDFProcessingError."""
    with pytest.raises(PDFProcessingError):
        extract_pages(b"", source_name="empty.pdf")


@pytest.mark.skipif(fitz is None, reason="PyMuPDF (fitz) is not installed")
def test_process_pdf_to_chunks_pipeline():
    """Verify the full PDF to chunks processing pipeline."""
    sample_text = (
        "Operating systems manage computer hardware and software resources. "
        "The kernel is the core component that controls memory, processes, "
        "and peripheral devices."
    )
    pdf_bytes = create_synthetic_pdf([sample_text])
    chunks = process_pdf_to_chunks(
        pdf_bytes,
        source_name="os_lecture.pdf",
        chunk_size_words=10,
        overlap_words=3
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.page_number == 1
        assert chunk.source == "os_lecture.pdf"
        assert len(chunk.text) > 0
