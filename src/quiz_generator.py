"""Quiz Generation and Targeted Revision Module for StudyMate AI.

Generates 4-option multiple choice questions with explanations and page references
from document chunks. Also supports a Confidence-Aware Study Loop to generate
targeted revision questions for past mistakes.
"""

from dataclasses import dataclass, asdict
import json
import re
from typing import List, Dict, Any, Optional
import random

from src.pdf_reader import DocumentChunk


@dataclass
class QuizQuestion:
    """Represents a multiple-choice quiz question."""
    question_id: str
    topic: str
    question: str
    options: List[str]  # 4 distinct options
    correct_option_index: int  # 0, 1, 2, or 3
    correct_answer: str
    explanation: str
    page_reference: int
    source_doc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuizGenerator:
    """Generates multiple choice quizzes and targeted revision questions."""

    def __init__(self, ai_tutor=None):
        """Initializes QuizGenerator with an optional LLM tutor instance.
        
        Args:
            ai_tutor: Optional AITutor instance for LLM-based quiz generation.
        """
        self.ai_tutor = ai_tutor

    def generate_quiz_from_chunks(
        self,
        chunks: List[DocumentChunk],
        num_questions: int = 3,
        topic_hint: Optional[str] = None
    ) -> List[QuizQuestion]:
        """Generates multiple choice questions based on document chunks.
        
        Args:
            chunks: Document passages to base questions on.
            num_questions: Number of questions to generate.
            topic_hint: Optional topic to focus the quiz on.
            
        Returns:
            List of QuizQuestion objects.
        """
        if not chunks:
            return []

        # If LLM tutor is available and not in mock mode, attempt LLM generation
        if self.ai_tutor is not None and not getattr(self.ai_tutor, "use_mock", True):
            try:
                return self._generate_with_llm(chunks, num_questions, topic_hint)
            except Exception as e:
                print(f"LLM quiz generation failed ({e}), falling back to extractive generator.")

        # Robust extractive generator (works offline, in tests, or as fast fallback)
        return self._generate_extractively(chunks, num_questions, topic_hint)

    def generate_targeted_revision_quiz(
        self,
        weak_topics: List[str],
        retriever,
        num_questions: int = 3
    ) -> List[QuizQuestion]:
        """Confidence-Aware Study Loop: Generates targeted revision questions for topics with past mistakes.
        
        Args:
            weak_topics: List of topics where student scored poorly or requested revision.
            retriever: PassageRetriever instance to find relevant notes for these topics.
            num_questions: Number of questions to generate.
            
        Returns:
            List of QuizQuestion objects focused on weak areas.
        """
        if not weak_topics or retriever is None:
            return []

        revision_chunks = []
        for topic in weak_topics[:3]:
            # Retrieve passages for this specific weak topic
            results = retriever.retrieve(f"Key concepts and definitions in {topic}", top_k=2)
            for res in results:
                if res["chunk"] not in revision_chunks:
                    revision_chunks.append(res["chunk"])

        if not revision_chunks:
            # Fallback to available chunks if no specific match
            revision_chunks = retriever.chunks[:5]

        # Generate questions prioritizing these topics
        return self.generate_quiz_from_chunks(
            chunks=revision_chunks,
            num_questions=min(num_questions, len(revision_chunks)),
            topic_hint=weak_topics[0] if weak_topics else None
        )

    def _generate_extractively(
        self,
        chunks: List[DocumentChunk],
        num_questions: int,
        topic_hint: Optional[str] = None
    ) -> List[QuizQuestion]:
        """Deterministic extractive question generator that requires no LLM weights.
        
        Constructs factual multiple-choice questions directly from the document's sentences,
        preserving citations and factual integrity.
        """
        questions: List[QuizQuestion] = []
        # Filter chunks if topic_hint is provided
        eligible_chunks = list(chunks)
        if topic_hint:
            matched = [c for c in eligible_chunks if topic_hint.lower() in c.text.lower()]
            if matched:
                eligible_chunks = matched

        for i, chunk in enumerate(eligible_chunks[:num_questions]):
            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?]) +", chunk.text)
                if len(s.strip().split()) >= 6
            ]
            if not sentences:
                continue

            target_sentence = sentences[0]
            words = target_sentence.split()

            # Infer topic from first words or topic_hint
            topic = topic_hint or (" ".join(words[:3]).capitalize())

            # Formulate question and options
            question_text = f"According to page {chunk.page_number}, which statement is directly supported by the notes on '{topic}'?"
            correct_opt = target_sentence

            # Create distinct synthetic plausible distractors
            distractors = [
                f"The notes state the opposite: {target_sentence.lower().replace('is', 'is not').replace('can', 'cannot')[:80]}...",
                f"This concept is entirely unrelated to {topic} and was omitted from the syllabus.",
                f"The text concludes that further laboratory testing is required before reaching this result."
            ]

            options = [correct_opt] + distractors
            # Deterministic shuffle based on chunk_id so tests are reproducible
            seed_val = chunk.chunk_id + i
            rng = random.Random(seed_val)
            rng.shuffle(options)

            correct_idx = options.index(correct_opt)

            q = QuizQuestion(
                question_id=f"q_{chunk.chunk_id}_{i}",
                topic=topic,
                question=question_text,
                options=options,
                correct_option_index=correct_idx,
                correct_answer=correct_opt,
                explanation=f"Found on Page {chunk.page_number} ({chunk.source}): '{target_sentence}'",
                page_reference=chunk.page_number,
                source_doc=chunk.source
            )
            questions.append(q)

        return questions

    def _generate_with_llm(
        self,
        chunks: List[DocumentChunk],
        num_questions: int,
        topic_hint: Optional[str]
    ) -> List[QuizQuestion]:
        """Prompts the local LLM to generate structured multiple-choice questions."""
        combined_text = "\n\n".join(
            f"[Page {c.page_number} - {c.source}]:\n{c.text}"
            for c in chunks[:4]
        )

        prompt = (
            f"Generate {num_questions} multiple-choice quiz questions based ONLY on the following study text. "
            f"Each question must have 4 options (A, B, C, D), identify the correct answer, provide a short explanation, "
            f"and state the exact page number.\n\n"
            f"<UNTRUSTED_DOCUMENT_CONTEXT>\n{combined_text}\n</UNTRUSTED_DOCUMENT_CONTEXT>\n\n"
            f"Format your response as a JSON list of objects with keys: "
            f"'question', 'topic', 'options' (list of 4 strings), 'correct_option_index' (0-3), 'explanation', 'page_number'."
        )

        raw_response = self.ai_tutor.generate_raw(prompt)
        # Parse JSON from response
        json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in LLM quiz response.")

        items = json.loads(json_match.group(0))
        parsed_questions = []

        for idx, item in enumerate(items):
            options = item.get("options", [])
            if len(options) != 4:
                continue

            correct_idx = int(item.get("correct_option_index", 0)) % 4
            parsed_questions.append(QuizQuestion(
                question_id=f"llm_q_{idx}",
                topic=item.get("topic", topic_hint or "Study Notes"),
                question=item.get("question", ""),
                options=options,
                correct_option_index=correct_idx,
                correct_answer=options[correct_idx],
                explanation=item.get("explanation", ""),
                page_reference=int(item.get("page_number", chunks[0].page_number)),
                source_doc=chunks[0].source
            ))

        return parsed_questions or self._generate_extractively(chunks, num_questions, topic_hint)
