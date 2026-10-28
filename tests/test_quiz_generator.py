"""Unit tests for QuizGenerator in StudyMate AI."""

import pytest
from src.pdf_reader import DocumentChunk
from src.retrieval import PassageRetriever
from src.quiz_generator import QuizGenerator, QuizQuestion


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id=0,
            text="Photosynthesis is the process used by plants to convert light energy into chemical energy. Chlorophyll absorbs solar radiation.",
            page_number=2,
            source="botany_101.pdf"
        ),
        DocumentChunk(
            chunk_id=1,
            text="The Calvin cycle takes place in the stroma of chloroplasts during light-independent reactions. Carbon dioxide is fixed into glucose.",
            page_number=3,
            source="botany_101.pdf"
        )
    ]


def test_empty_chunks_returns_empty_quiz():
    generator = QuizGenerator()
    questions = generator.generate_quiz_from_chunks([])
    assert questions == []


def test_extractive_quiz_generation(sample_chunks):
    generator = QuizGenerator()
    questions = generator.generate_quiz_from_chunks(sample_chunks, num_questions=2)

    assert len(questions) == 2
    for q in questions:
        assert isinstance(q, QuizQuestion)
        assert len(q.options) == 4
        assert 0 <= q.correct_option_index < 4
        assert q.options[q.correct_option_index] == q.correct_answer
        assert q.page_reference in [2, 3]
        assert q.source_doc == "botany_101.pdf"
        assert len(q.explanation) > 0


def test_targeted_revision_quiz(sample_chunks):
    retriever = PassageRetriever(use_mock=True)
    retriever.index_chunks(sample_chunks)

    generator = QuizGenerator()
    weak_topics = ["Calvin cycle"]
    revision_questions = generator.generate_targeted_revision_quiz(
        weak_topics=weak_topics,
        retriever=retriever,
        num_questions=1
    )

    assert len(revision_questions) >= 1
    # Check that question references the relevant content
    assert revision_questions[0].page_reference in [2, 3]
    assert len(revision_questions[0].options) == 4
