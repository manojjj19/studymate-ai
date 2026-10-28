"""StudyMate AI - Main Streamlit Web Application.

An open-source AI-powered study companion where students upload PDF notes,
ask questions with verified page citations, generate interactive quizzes,
and revise weak topics through a Confidence-Aware Study Loop.
"""

import io
import os
import streamlit as st

from src.pdf_reader import (
    process_pdf_to_chunks,
    DocumentChunk,
    PDFProcessingError,
    ScannedPDFWarning
)
from src.retrieval import PassageRetriever
from src.ai_tutor import AITutor
from src.quiz_generator import QuizGenerator, QuizQuestion
from src.progress_tracker import ProgressTracker


# ---------------------------------------------------------
# Page Configuration & Purple/Blue Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="StudyMate AI | Smart Open-Source Study Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global styles and color variables */
    :root {
        --primary: #4F46E5;
        --primary-light: #EEF2FF;
        --secondary: #6366F1;
        --accent: #818CF8;
        --text-dark: #1E1B4B;
        --border-color: #E0E7FF;
    }
    
    .main-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white !important;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.18);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    .study-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E7FF;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    .citation-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        border: 1px solid #C7D2FE;
        margin-right: 0.5rem;
    }

    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #92400E;
        margin: 0.8rem 0;
    }

    .security-badge {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "tracker" not in st.session_state:
    st.session_state.tracker = ProgressTracker()


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("StudyMate AI")
    st.caption("Hacktoberfest Open-Source AI Edition")
    st.divider()

    st.subheader("⚙️ AI Model Settings")
    
    # Execution mode selector
    execution_mode = st.radio(
        "Execution Mode:",
        options=["Fast Mock / Offline Mode", "Local Open-Weight LLM"],
        index=0,
        help=(
            "Fast Mock Mode provides instantaneous extractive responses for testing "
            "without needing to download large 3GB model weights. Local LLM loads Qwen2.5."
        )
    )
    is_mock = (execution_mode == "Fast Mock / Offline Mode")

    model_name = st.text_input(
        "Model Identifier:",
        value="Qwen/Qwen2.5-1.5B-Instruct",
        help="Configurable open-weight model from Hugging Face."
    )

    # Hardware detection
    detected_device = AITutor._detect_device()
    if detected_device == "cuda":
        st.success("⚡ Hardware: NVIDIA GPU (CUDA Enabled)")
    else:
        st.info("🖥️ Hardware: CPU Mode (Fallback Active)")

    st.divider()
    st.subheader("📄 Document Processing")
    chunk_size = st.slider("Chunk Size (words):", min_value=80, max_value=300, value=150, step=10)
    overlap = st.slider("Chunk Overlap (words):", min_value=10, max_value=80, value=30, step=5)

    st.divider()
    # Quick demo loader
    if st.button("🚀 Load Synthetic Sample Notes", use_container_width=True):
        from sample_data.create_sample_pdf import SAMPLE_PAGES
        synthetic_chunks = []
        c_id = 0
        for p in SAMPLE_PAGES:
            words = p["content"].split()
            synthetic_chunks.append(DocumentChunk(
                chunk_id=c_id,
                text=p["content"],
                page_number=p["page_number"],
                source="Synthetic_CS_Notes.pdf"
            ))
            c_id += 1
        
        st.session_state.chunks = synthetic_chunks
        st.session_state.doc_name = "Synthetic_CS_Notes.pdf"
        retriever = PassageRetriever(use_mock=True)
        retriever.index_chunks(synthetic_chunks)
        st.session_state.retriever = retriever
        st.session_state.current_quiz = []
        st.session_state.quiz_submitted = False
        st.success("Loaded 3 pages of synthetic CS notes!")
        st.rerun()

    if st.session_state.doc_name:
        if st.button("🗑️ Clear Current Document", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.doc_name = None
            st.session_state.retriever = None
            st.session_state.current_quiz = []
            st.session_state.quiz_submitted = False
            st.rerun()


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 StudyMate AI</h1>
        <p>Your intelligent, open-source study assistant. Upload PDF notes, receive grounded answers with exact page references, test your memory, and master weak topics.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_upload, tab_qa, tab_quiz, tab_progress = st.tabs([
    "📂 1. Upload & Notes",
    "💡 2. Ask Your Notes",
    "📝 3. Practice Quiz & Revision",
    "📊 4. Progress & Mistakes"
])


# =========================================================
# TAB 1: UPLOAD & NOTES HUB
# =========================================================
with tab_upload:
    st.subheader("📄 Upload Study Notes (PDF)")
    st.write(
        "Upload any digital text-based PDF lecture notes, summaries, or handouts. "
        "StudyMate AI will extract text page-by-page and index it for semantic retrieval."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Please upload a text-based PDF. Scanned image-only PDFs require OCR which is not yet supported."
    )

    if uploaded_file is not None:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner(f"Extracting text and chunking '{uploaded_file.name}'..."):
                try:
                    pdf_bytes = uploaded_file.read()
                    chunks = process_pdf_to_chunks(
                        file_input=pdf_bytes,
                        source_name=uploaded_file.name,
                        chunk_size_words=chunk_size,
                        overlap_words=overlap
                    )
                    
                    # Build index
                    retriever = PassageRetriever(use_mock=is_mock)
                    retriever.index_chunks(chunks)
                    
                    st.session_state.chunks = chunks
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.retriever = retriever
                    st.session_state.current_quiz = []
                    st.session_state.quiz_submitted = False
                    
                    st.success(f"Successfully processed '{uploaded_file.name}'!")

                except ScannedPDFWarning as sw:
                    st.error(f"⚠️ Scanned PDF Notice: {str(sw)}")
                except PDFProcessingError as pe:
                    st.error(f"❌ PDF Processing Error: {str(pe)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

    if st.session_state.doc_name and st.session_state.chunks:
        st.markdown("---")
        st.markdown(f"### 📑 Active Document: `{st.session_state.doc_name}`")
        
        pages_present = sorted(list(set(c.page_number for c in st.session_state.chunks)))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages Extracted", len(pages_present))
        with col2:
            st.metric("Total Chunks Created", len(st.session_state.chunks))
        with col3:
            total_words = sum(len(c.text.split()) for c in st.session_state.chunks)
            st.metric("Estimated Word Count", total_words)

        st.markdown("#### 🔍 Browse Extracted Chunks")
        selected_page = st.selectbox("Filter by Page Number:", options=["All Pages"] + [f"Page {p}" for p in pages_present])
        
        display_chunks = st.session_state.chunks
        if selected_page != "All Pages":
            page_num = int(selected_page.replace("Page ", ""))
            display_chunks = [c for c in display_chunks if c.page_number == page_num]

        for c in display_chunks[:6]:
            with st.expander(f"Chunk #{c.chunk_id} | Page {c.page_number} ({len(c.text.split())} words)"):
                st.write(c.text)
    else:
        st.info("💡 No document currently loaded. Upload a PDF above or click 'Load Synthetic Sample Notes' in the sidebar to try StudyMate AI immediately!")


# =========================================================
# TAB 2: ASK YOUR NOTES (Q&A)
# =========================================================
with tab_qa:
    st.subheader("💡 Ask Your Notes")
    st.write("Ask questions about concepts in your uploaded document. Answers are grounded in your text with page citations.")

    st.markdown(
        """
        <div class="security-badge">
            🛡️ <b>Prompt Injection Protection Active:</b> Document text is strictly encapsulated as untrusted context. The assistant answers only from verifiable notes.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.chunks or st.session_state.retriever is None:
        st.warning("⚠️ Please upload a PDF in the 'Upload & Notes' tab or load sample notes first.")
    else:
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the Von Neumann bottleneck? or Explain supervised learning.",
            help="Type your question and press Ask Question."
        )

        col_q1, col_q2 = st.columns([1, 4])
        with col_q1:
            ask_btn = st.button("🔍 Ask Tutor", use_container_width=True, type="primary")

        if ask_btn and user_query:
            with st.spinner("Searching document passages and preparing grounded explanation..."):
                retriever = st.session_state.retriever
                passages = retriever.retrieve(user_query, top_k=3)
                
                tutor = AITutor(
                    model_name=model_name,
                    use_mock=is_mock,
                    device=detected_device
                )
                
                response_data = tutor.answer_question(user_query, passages)

            st.markdown("### 🎓 Tutor Explanation")
            if response_data["has_evidence"]:
                st.markdown(
                    f"""
                    <div class="study-card">
                        {response_data["answer"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### 📚 Supporting Sources & Page Citations")
                for src in response_data["sources"]:
                    with st.expander(f"📖 Page {src['page_number']} — {src['source']} (Relevance Score: {src['similarity']})"):
                        st.write(f"**Extracted Passage:**\n> {src['snippet']}")
            else:
                st.warning(response_data["answer"])


# =========================================================
# TAB 3: PRACTICE QUIZ & REVISION
# =========================================================
with tab_quiz:
    st.subheader("📝 Practice Quiz & Targeted Revision")
    st.write("Generate 4-option multiple choice questions from your notes to test active recall.")

    if not st.session_state.chunks:
        st.warning("⚠️ Please upload notes or load sample data to generate a quiz.")
    else:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
        with col_ctrl1:
            num_q = st.selectbox("Number of Questions:", options=[1, 2, 3, 5], index=1)
        with col_ctrl2:
            gen_quiz_btn = st.button("🎲 Generate Quiz", use_container_width=True, type="primary")
        with col_ctrl3:
            # Check for weak topics from progress tracker
            weak_topics = st.session_state.tracker.get_topics_needing_revision()
            revision_label = f"⚡ Revise Weak Topics ({len(weak_topics)})" if weak_topics else "⚡ Targeted Revision"
            revision_btn = st.button(revision_label, use_container_width=True)

        generator = QuizGenerator(ai_tutor=AITutor(model_name=model_name, use_mock=is_mock))

        # Handle regular quiz generation
        if gen_quiz_btn:
            with st.spinner("Generating multiple-choice questions from notes..."):
                st.session_state.current_quiz = generator.generate_quiz_from_chunks(
                    chunks=st.session_state.chunks,
                    num_questions=num_q
                )
                st.session_state.quiz_submitted = False
                st.rerun()

        # Handle targeted revision quiz generation
        if revision_btn:
            if not weak_topics:
                st.info("No mistakes recorded yet! Take a practice quiz first to identify topics needing review.")
            else:
                with st.spinner(f"Generating targeted revision quiz for: {', '.join(weak_topics[:2])}..."):
                    st.session_state.current_quiz = generator.generate_targeted_revision_quiz(
                        weak_topics=weak_topics,
                        retriever=st.session_state.retriever,
                        num_questions=num_q
                    )
                    st.session_state.quiz_submitted = False
                    st.success(f"Loaded targeted revision questions for: {', '.join(weak_topics[:2])}")
                    st.rerun()

        # Render active quiz
        if st.session_state.current_quiz:
            st.markdown("---")
            st.markdown(
                f"""
                <div class="disclaimer-box">
                    💡 <b>Confidence-Aware Study Loop:</b> Select your confidence level for each answer. 
                    {ProgressTracker.HEURISTIC_DISCLAIMER}
                </div>
                """,
                unsafe_allow_html=True
            )

            student_answers = {}
            student_confidences = {}

            for idx, q in enumerate(st.session_state.current_quiz):
                st.markdown(f"#### Question {idx + 1}: {q.question}")
                st.markdown(f"<span class='citation-badge'>Topic: {q.topic}</span><span class='citation-badge'>Source Page {q.page_reference}</span>", unsafe_allow_html=True)

                radio_key = f"quiz_choice_{q.question_id}"
                selected = st.radio(
                    f"Select your answer for Question {idx + 1}:",
                    options=q.options,
                    key=radio_key,
                    index=None,
                    label_visibility="collapsed"
                )
                student_answers[idx] = selected

                conf_key = f"quiz_conf_{q.question_id}"
                confidence = st.select_slider(
                    f"How confident are you in this answer?",
                    options=["Low", "Medium", "High"],
                    value="Medium",
                    key=conf_key
                )
                student_confidences[idx] = confidence
                st.markdown("<br>", unsafe_allow_html=True)

            if not st.session_state.quiz_submitted:
                if st.button("✅ Submit Answers", type="primary"):
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                st.markdown("### 📋 Quiz Results & Explanations")
                correct_count = 0

                for idx, q in enumerate(st.session_state.current_quiz):
                    student_ans = student_answers.get(idx)
                    student_conf = student_confidences.get(idx, "Medium")
                    is_correct = (student_ans == q.correct_answer)

                    if is_correct:
                        correct_count += 1
                        st.success(f"**Question {idx + 1}: Correct!** ✅\n\nYour answer: `{student_ans}`")
                    else:
                        st.error(f"**Question {idx + 1}: Incorrect** ❌\n\nYour answer: `{student_ans or 'No answer selected'}`\n\n**Correct answer:** `{q.correct_answer}`")

                    # Record attempt in tracker
                    st.session_state.tracker.record_attempt(
                        topic=q.topic,
                        question=q.question,
                        student_answer=student_ans or "None",
                        correct_answer=q.correct_answer,
                        is_correct=is_correct,
                        page_reference=q.page_reference,
                        source_doc=q.source_doc,
                        confidence_level=student_conf
                    )

                    with st.expander(f"💡 Explanation & Citation (Page {q.page_reference})"):
                        st.write(q.explanation)

                st.markdown(f"### Score: **{correct_count} / {len(st.session_state.current_quiz)}**")
                if correct_count < len(st.session_state.current_quiz):
                    st.info("Tip: Head over to the **Progress & Mistakes** tab or click **Revise Weak Topics** to reinforce concepts you missed!")


# =========================================================
# TAB 4: PROGRESS & MISTAKE DASHBOARD
# =========================================================
with tab_progress:
    st.subheader("📊 Study Progress & Mistake Analysis")
    st.write("Track your learning trajectory, review missed questions, and plan targeted revision sessions.")

    stats = st.session_state.tracker.get_summary_stats()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Quizzes Attempted", stats["total_attempted"])
    with m2:
        st.metric("Correct Answers", stats["total_correct"])
    with m3:
        st.metric("Overall Accuracy", f"{stats['accuracy_rate']}%")
    with m4:
        st.metric("Topics Needing Revision", stats["weak_topics_count"])

    st.markdown(
        f"""
        <div class="disclaimer-box">
            📌 <b>Note on Confidence Metrics:</b> {ProgressTracker.HEURISTIC_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Topic breakdown
    breakdown = st.session_state.tracker.get_topic_breakdown()
    if breakdown:
        st.markdown("### 📈 Topic Accuracy Breakdown")
        for topic, data in breakdown.items():
            col_t1, col_t2 = st.columns([2, 3])
            with col_t1:
                st.write(f"**{topic}** ({data['correct']}/{data['attempted']} correct)")
            with col_t2:
                st.progress(data["accuracy"] / 100.0)

    # Mistake Review List
    incorrect_attempts = [a for a in st.session_state.tracker.attempts if not a.is_correct]
    if incorrect_attempts:
        st.markdown("### ⚠️ Mistakes to Revise")
        st.caption("Review questions you answered incorrectly, along with the source page references.")

        for attempt in reversed(incorrect_attempts[-8:]):
            with st.expander(f"❌ {attempt.topic} — Page {attempt.page_reference or 'N/A'}"):
                st.write(f"**Question:** {attempt.question}")
                st.write(f"**Your Answer:** `{attempt.student_answer}` *(Self-reported Confidence: {attempt.confidence_level})*")
                st.write(f"**Correct Answer:** `{attempt.correct_answer}`")
                st.write(f"**Source Document:** `{attempt.source_doc}` (Page {attempt.page_reference})")
    else:
        st.success("🎉 No mistakes recorded yet! Great job studying.")

    st.markdown("---")
    if st.button("🗑️ Reset Study History"):
        st.session_state.tracker.reset_history()
        st.success("Study history cleared.")
        st.rerun()
