"""Progress Tracking and Mistake Analysis Module for StudyMate AI.

Tracks quiz attempts, correct answers, and topics needing revision.
Persists history to local JSON without requiring external databases.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from typing import List, Dict, Any, Optional


@dataclass
class QuizAttempt:
    """Represents a single question attempt by a student."""
    attempt_id: str
    topic: str
    question: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    page_reference: Optional[int]
    source_doc: str
    timestamp: str
    # Student's self-assessed or heuristic confidence (e.g. 'High', 'Medium', 'Low')
    confidence_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuizAttempt":
        return cls(**data)


class ProgressTracker:
    """Manages student learning progress, quiz history, and targeted revision topics."""

    HEURISTIC_DISCLAIMER = (
        "Confidence ratings are subjective student heuristics to prioritize study review, "
        "not a mathematical probability or guarantee of correctness."
    )

    def __init__(self, storage_path: str = "progress_data.json"):
        self.storage_path = storage_path
        self.attempts: List[QuizAttempt] = []
        self.load_history()

    def record_attempt(
        self,
        topic: str,
        question: str,
        student_answer: str,
        correct_answer: str,
        is_correct: bool,
        page_reference: Optional[int] = None,
        source_doc: str = "document.pdf",
        confidence_level: str = "Medium"
    ) -> QuizAttempt:
        """Records a new quiz question attempt."""
        attempt_id = f"att_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        attempt = QuizAttempt(
            attempt_id=attempt_id,
            topic=topic.strip() or "General Notes",
            question=question.strip(),
            student_answer=student_answer.strip(),
            correct_answer=correct_answer.strip(),
            is_correct=is_correct,
            page_reference=page_reference,
            source_doc=source_doc,
            timestamp=datetime.now().isoformat(),
            confidence_level=confidence_level
        )
        self.attempts.append(attempt)
        self.save_history()
        return attempt

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates overall quiz statistics."""
        total = len(self.attempts)
        if total == 0:
            return {
                "total_attempted": 0,
                "total_correct": 0,
                "total_incorrect": 0,
                "accuracy_rate": 0.0,
                "weak_topics_count": 0
            }

        correct = sum(1 for a in self.attempts if a.is_correct)
        incorrect = total - correct
        accuracy = round((correct / total) * 100, 1)

        weak_topics = self.get_topics_needing_revision()

        return {
            "total_attempted": total,
            "total_correct": correct,
            "total_incorrect": incorrect,
            "accuracy_rate": accuracy,
            "weak_topics_count": len(weak_topics)
        }

    def get_topic_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Returns accuracy and mistake breakdown grouped by topic."""
        breakdown: Dict[str, Dict[str, Any]] = {}
        for a in self.attempts:
            if a.topic not in breakdown:
                breakdown[a.topic] = {"attempted": 0, "correct": 0, "incorrect": 0}
            breakdown[a.topic]["attempted"] += 1
            if a.is_correct:
                breakdown[a.topic]["correct"] += 1
            else:
                breakdown[a.topic]["incorrect"] += 1

        for topic, data in breakdown.items():
            data["accuracy"] = round((data["correct"] / data["attempted"]) * 100, 1)

        return breakdown

    def get_topics_needing_revision(self) -> List[str]:
        """Finds topics where the student has made mistakes or has low accuracy."""
        breakdown = self.get_topic_breakdown()
        # Topics with at least 1 mistake, sorted by highest number of mistakes first
        topics = [
            topic for topic, stats in breakdown.items()
            if stats["incorrect"] > 0
        ]
        topics.sort(key=lambda t: breakdown[t]["incorrect"], reverse=True)
        return topics

    def get_incorrect_attempts_by_topic(self, topic: Optional[str] = None) -> List[QuizAttempt]:
        """Retrieves past incorrect attempts, optionally filtered by topic."""
        incorrect = [a for a in self.attempts if not a.is_correct]
        if topic:
            incorrect = [a for a in incorrect if a.topic.lower() == topic.lower()]
        return incorrect

    def save_history(self) -> None:
        """Saves attempts to a local JSON file."""
        try:
            data = [a.to_dict() for a in self.attempts]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Avoid crashing the UI if disk write fails
            print(f"Warning: Failed to save progress data: {e}")

    def load_history(self) -> None:
        """Loads attempts from the local JSON file if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.attempts = [QuizAttempt.from_dict(item) for item in data]
            except Exception as e:
                print(f"Warning: Failed to load progress data: {e}")
                self.attempts = []
        else:
            self.attempts = []

    def reset_history(self) -> None:
        """Clears all stored quiz history."""
        self.attempts = []
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except OSError:
                pass
