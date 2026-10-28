"""Unit tests for semantic retrieval in StudyMate AI."""

import pytest
from src.pdf_reader import DocumentChunk
from src.retrieval import PassageRetriever


def test_empty_retriever():
    """Verify retriever handles empty chunk lists gracefully."""
    retriever = PassageRetriever(use_mock=True)
    indexed_count = retriever.index_chunks([])
    assert indexed_count == 0
    assert retriever.embeddings is None

    results = retriever.retrieve("Any question")
    assert results == []
    assert retriever.has_sufficient_evidence(results) is False


def test_retrieval_ranking_and_page_tracking():
    """Verify that retrieval accurately ranks relevant passages and retains page references."""
    retriever = PassageRetriever(use_mock=True, min_relevance_threshold=0.2)

    chunks = [
        DocumentChunk(
            chunk_id=0,
            text="Mitochondria produce ATP through aerobic cellular respiration.",
            page_number=5,
            source="bio_notes.pdf"
        ),
        DocumentChunk(
            chunk_id=1,
            text="Newton's second law states that Force equals mass times acceleration (F = ma).",
            page_number=12,
            source="physics_notes.pdf"
        ),
        DocumentChunk(
            chunk_id=2,
            text="Binary search trees maintain sorted keys with O(log n) average lookup time.",
            page_number=28,
            source="cs_notes.pdf"
        )
    ]

    count = retriever.index_chunks(chunks)
    assert count == 3

    # Test biological question
    bio_results = retriever.retrieve("How do mitochondria produce ATP?", top_k=2)
    assert len(bio_results) > 0
    top_hit = bio_results[0]
    assert top_hit["chunk"].chunk_id == 0
    assert top_hit["page_number"] == 5
    assert top_hit["source"] == "bio_notes.pdf"
    assert top_hit["is_relevant"] is True
    assert retriever.has_sufficient_evidence(bio_results) is True

    # Test physics question
    physics_results = retriever.retrieve("What is Newton's law for force and mass?", top_k=1)
    assert len(physics_results) == 1
    assert physics_results[0]["chunk"].chunk_id == 1
    assert physics_results[0]["page_number"] == 12


def test_irrelevant_query_insufficient_evidence():
    """Verify that irrelevant queries are flagged as having insufficient evidence."""
    retriever = PassageRetriever(use_mock=True, min_relevance_threshold=0.5)

    chunks = [
        DocumentChunk(
            chunk_id=0,
            text="Photosynthesis occurs in plant chloroplasts using chlorophyll to capture sunlight.",
            page_number=1,
            source="botany.pdf"
        )
    ]
    retriever.index_chunks(chunks)

    # Completely unrelated query
    results = retriever.retrieve("How does the stock market exchange calculate dividend yield?")
    assert len(results) == 1
    # Check that it is flagged as not relevant because similarity is low
    assert results[0]["is_relevant"] is False
    assert retriever.has_sufficient_evidence(results) is False
