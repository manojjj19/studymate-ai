"""StudyMate AI - Command Line Interface (CLI) Runner.

Enables students and evaluators to interact with StudyMate AI directly
from the terminal without needing to launch a web server.
Demonstrates PDF extraction, grounded QA, quiz generation, and progress tracking.
"""

import sys
import os
import io

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.pdf_reader import DocumentChunk, process_pdf_to_chunks
from src.retrieval import PassageRetriever
from src.ai_tutor import AITutor
from src.quiz_generator import QuizGenerator
from src.progress_tracker import ProgressTracker
from sample_data.create_sample_pdf import SAMPLE_PAGES


def print_banner():
    print("=" * 70)
    print("[StudyMate AI] Open-Source Study Companion (CLI Edition)")
    print("=" * 70)
    print("Features: PDF Grounded Q&A | 4-Option Quizzes | Mistake Tracking\n")


def load_demo_chunks():
    """Loads synthetic CS study guide chunks."""
    chunks = []
    c_id = 0
    for p in SAMPLE_PAGES:
        chunks.append(DocumentChunk(
            chunk_id=c_id,
            text=p["content"],
            page_number=p["page_number"],
            source="Synthetic_CS_Notes.pdf"
        ))
        c_id += 1
    return chunks


def main():
    print_banner()

    tracker = ProgressTracker("progress_data.json")
    retriever = PassageRetriever(use_mock=True)
    tutor = AITutor(use_mock=True)
    quiz_gen = QuizGenerator(ai_tutor=tutor)

    print("Options:")
    print("1. Load Built-In Synthetic Notes (Computer Architecture & ML)")
    print("2. Load Custom Text File or PDF")
    choice = input("\nEnter choice (1 or 2, default 1): ").strip() or "1"

    chunks = []
    if choice == "2":
        path = input("Enter path to file: ").strip()
        if os.path.exists(path):
            try:
                chunks = process_pdf_to_chunks(path, source_name=os.path.basename(path))
                print(f"Loaded {len(chunks)} chunks from {path}")
            except Exception as e:
                print(f"Could not load PDF ({e}). Falling back to synthetic notes.")
                chunks = load_demo_chunks()
        else:
            print("File not found. Using synthetic notes.")
            chunks = load_demo_chunks()
    else:
        chunks = load_demo_chunks()

    retriever.index_chunks(chunks)
    print(f"Indexed {len(chunks)} passages across {len(set(c.page_number for c in chunks))} pages.\n")

    while True:
        print("-" * 50)
        print("Main Menu:")
        print("1. 💡 Ask a Question (Grounded Q&A)")
        print("2. 📝 Take a Practice Quiz")
        print("3. ⚡ Revise Mistakes (Confidence-Aware Study Loop)")
        print("4. 📊 View Study Progress Dashboard")
        print("5. 🚪 Exit")
        action = input("\nSelect action (1-5): ").strip()

        if action == "1":
            q = input("\nEnter your question based on the notes: ").strip()
            if not q:
                continue
            passages = retriever.retrieve(q, top_k=2)
            res = tutor.answer_question(q, passages)
            print("\n🎓 TUTOR ANSWER:")
            print(res["answer"])
            if res["sources"]:
                print("\n📚 Supporting Sources:")
                for s in res["sources"]:
                    print(f"- Page {s['page_number']} ({s['source']}) [Relevance: {s['similarity']}]")

        elif action == "2":
            questions = quiz_gen.generate_quiz_from_chunks(chunks, num_questions=2)
            if not questions:
                print("No questions could be generated from the document.")
                continue

            for idx, q in enumerate(questions):
                print(f"\nQuestion {idx + 1}: {q.question}")
                print(f"[Topic: {q.topic} | Page {q.page_reference}]")
                for opt_idx, opt in enumerate(q.options):
                    print(f"  {chr(65 + opt_idx)}) {opt}")

                ans_letter = input("Your answer (A, B, C, or D): ").strip().upper()
                selected_idx = ord(ans_letter) - 65 if ans_letter in ["A", "B", "C", "D"] else -1
                student_ans = q.options[selected_idx] if 0 <= selected_idx < 4 else "None"

                conf = input("Confidence level (Low, Medium, High, default Medium): ").strip().capitalize() or "Medium"

                is_correct = (student_ans == q.correct_answer)
                if is_correct:
                    print("✅ Correct!")
                else:
                    print(f"❌ Incorrect! Correct answer was: {q.correct_answer}")

                print(f"Explanation: {q.explanation}")

                tracker.record_attempt(
                    topic=q.topic,
                    question=q.question,
                    student_answer=student_ans,
                    correct_answer=q.correct_answer,
                    is_correct=is_correct,
                    page_reference=q.page_reference,
                    source_doc=q.source_doc,
                    confidence_level=conf
                )

        elif action == "3":
            weak = tracker.get_topics_needing_revision()
            if not weak:
                print("\n🎉 No weak topics recorded yet! Take a practice quiz first to identify weak areas.")
            else:
                print(f"\nTargeting weak topics: {', '.join(weak)}")
                rev_q = quiz_gen.generate_targeted_revision_quiz(weak, retriever, num_questions=1)
                for q in rev_q:
                    print(f"\nRevision Question: {q.question}")
                    for opt_idx, opt in enumerate(q.options):
                        print(f"  {chr(65 + opt_idx)}) {opt}")
                    ans_letter = input("Your answer (A, B, C, D): ").strip().upper()
                    selected_idx = ord(ans_letter) - 65 if ans_letter in ["A", "B", "C", "D"] else -1
                    student_ans = q.options[selected_idx] if 0 <= selected_idx < 4 else "None"
                    if student_ans == q.correct_answer:
                        print("✅ Correct! You have reinforced this topic.")
                    else:
                        print(f"❌ Still needs work. Correct answer: {q.correct_answer}")
                        print(f"Refer back to Page {q.page_reference}.")

        elif action == "4":
            stats = tracker.get_summary_stats()
            print("\n" + "=" * 40)
            print("📊 STUDY PROGRESS DASHBOARD")
            print("=" * 40)
            print(f"Total Quizzes Attempted:  {stats['total_attempted']}")
            print(f"Correct Answers:          {stats['total_correct']}")
            print(f"Overall Accuracy:         {stats['accuracy_rate']}%")
            print(f"Topics Needing Revision:  {stats['weak_topics_count']}")
            print(f"\nNote: {ProgressTracker.HEURISTIC_DISCLAIMER}")

            breakdown = tracker.get_topic_breakdown()
            if breakdown:
                print("\nTopic Breakdown:")
                for t, d in breakdown.items():
                    print(f"  • {t}: {d['correct']}/{d['attempted']} ({d['accuracy']}%)")

        elif action == "5":
            print("\nHappy studying with StudyMate AI! Goodbye.")
            break


if __name__ == "__main__":
    main()
