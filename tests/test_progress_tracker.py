"""Unit tests for ProgressTracker in StudyMate AI."""

import os
import pytest
from src.progress_tracker import ProgressTracker, QuizAttempt


@pytest.fixture
def temp_tracker(tmp_path):
    """Fixture providing a ProgressTracker with an isolated temporary storage file."""
    temp_json = str(tmp_path / "test_progress.json")
    tracker = ProgressTracker(storage_path=temp_json)
    return tracker


def test_empty_progress_tracker(temp_tracker):
    """Verify empty tracker returns clean zero-state stats."""
    stats = temp_tracker.get_summary_stats()
    assert stats["total_attempted"] == 0
    assert stats["total_correct"] == 0
    assert stats["accuracy_rate"] == 0.0
    assert stats["weak_topics_count"] == 0
    assert temp_tracker.get_topics_needing_revision() == []


def test_record_attempts_and_statistics(temp_tracker):
    """Verify recording correct and incorrect quiz attempts updates stats properly."""
    # Attempt 1: Biology, Correct
    temp_tracker.record_attempt(
        topic="Cell Biology",
        question="What is the powerhouse of the cell?",
        student_answer="Mitochondria",
        correct_answer="Mitochondria",
        is_correct=True,
        page_reference=3,
        source_doc="bio.pdf",
        confidence_level="High"
    )

    # Attempt 2: Cell Biology, Incorrect
    temp_tracker.record_attempt(
        topic="Cell Biology",
        question="Where does glycolysis occur?",
        student_answer="Nucleus",
        correct_answer="Cytoplasm",
        is_correct=False,
        page_reference=5,
        source_doc="bio.pdf",
        confidence_level="Low"
    )

    # Attempt 3: Physics, Incorrect
    temp_tracker.record_attempt(
        topic="Thermodynamics",
        question="What does the first law of thermodynamics state?",
        student_answer="Entropy always increases",
        correct_answer="Conservation of energy",
        is_correct=False,
        page_reference=10,
        source_doc="physics.pdf",
        confidence_level="Medium"
    )

    stats = temp_tracker.get_summary_stats()
    assert stats["total_attempted"] == 3
    assert stats["total_correct"] == 1
    assert stats["total_incorrect"] == 2
    assert stats["accuracy_rate"] == 33.3
    assert stats["weak_topics_count"] == 2

    # Check topic breakdown
    breakdown = temp_tracker.get_topic_breakdown()
    assert breakdown["Cell Biology"]["attempted"] == 2
    assert breakdown["Cell Biology"]["correct"] == 1
    assert breakdown["Cell Biology"]["accuracy"] == 50.0

    assert breakdown["Thermodynamics"]["attempted"] == 1
    assert breakdown["Thermodynamics"]["correct"] == 0
    assert breakdown["Thermodynamics"]["accuracy"] == 0.0

    # Revision list
    revision_topics = temp_tracker.get_topics_needing_revision()
    assert "Cell Biology" in revision_topics
    assert "Thermodynamics" in revision_topics

    # Retrieve incorrect by topic
    bio_mistakes = temp_tracker.get_incorrect_attempts_by_topic("Cell Biology")
    assert len(bio_mistakes) == 1
    assert bio_mistakes[0].student_answer == "Nucleus"


def test_json_persistence(tmp_path):
    """Verify quiz history persists to and reloads from a local JSON file."""
    temp_json = str(tmp_path / "persistent_progress.json")
    tracker1 = ProgressTracker(storage_path=temp_json)

    tracker1.record_attempt(
        topic="History",
        question="In what year did WW2 end?",
        student_answer="1945",
        correct_answer="1945",
        is_correct=True,
        page_reference=22
    )

    assert os.path.exists(temp_json)

    # Load in new tracker instance
    tracker2 = ProgressTracker(storage_path=temp_json)
    assert len(tracker2.attempts) == 1
    assert tracker2.attempts[0].topic == "History"
    assert tracker2.attempts[0].is_correct is True


def test_heuristic_disclaimer(temp_tracker):
    """Verify the confidence disclaimer is present and clearly documented."""
    assert "heuristic" in temp_tracker.HEURISTIC_DISCLAIMER.lower()
    assert "probability" in temp_tracker.HEURISTIC_DISCLAIMER.lower()
