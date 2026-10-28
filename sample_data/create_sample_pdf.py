"""Synthetic Sample PDF Generator for StudyMate AI.

Generates a realistic, multi-page synthetic student revision guide
covering fundamental concepts in Computer Science and Machine Learning.
Does not contain copyrighted textbooks or private student data.
"""

import sys
import os

SAMPLE_PAGES = [
    {
        "page_number": 1,
        "title": "Module 1: Introduction to Computer Architecture",
        "content": (
            "Computer architecture defines the functionality, organization, and implementation of computer systems.\n"
            "The Von Neumann architecture forms the foundation of modern computers. It features a central processing unit (CPU), "
            "a control unit, an arithmetic logic unit (ALU), memory, and input/output mechanisms.\n\n"
            "In this model, program instructions and data share the same physical memory and bus structure. "
            "This shared bus between memory and CPU can create a throughput bottleneck, famously known as the Von Neumann bottleneck.\n"
            "Cache memory sits between registers and main RAM to mitigate latency issues."
        )
    },
    {
        "page_number": 2,
        "title": "Module 2: Machine Learning Foundations",
        "content": (
            "Machine learning is a branch of artificial intelligence focused on building systems that learn from data.\n"
            "Supervised learning trains models on labeled datasets where inputs correspond to desired target outputs. "
            "Common supervised tasks include linear regression for continuous predictions and classification for discrete categories.\n\n"
            "Unsupervised learning uncovers hidden patterns in unlabeled data without explicit feedback. "
            "Clustering algorithms like K-Means group similar data points together based on distance metrics like Euclidean distance.\n"
            "Overfitting occurs when a model memorizes training noise instead of generalizing to unseen data."
        )
    },
    {
        "page_number": 3,
        "title": "Module 3: Neural Networks and Deep Learning",
        "content": (
            "Deep learning utilizes artificial neural networks composed of multiple layers of interconnected neurons.\n"
            "An activation function introduces non-linearity into the network, enabling it to learn complex mappings.\n"
            "Popular activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh.\n\n"
            "Backpropagation is the core algorithm used to train neural networks. It calculates the gradient of the loss function "
            "with respect to each network weight using the mathematical chain rule of calculus.\n"
            "Optimizers like Adam and SGD adjust weights in the direction that minimizes loss."
        )
    }
]


def generate_sample_pdf(output_path: str = "sample_data/sample_study_guide.pdf") -> str:
    """Creates a sample multi-page PDF using PyMuPDF if installed, or plain text if not."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        import fitz
        doc = fitz.open()
        for page_data in SAMPLE_PAGES:
            page = doc.new_page(width=595, height=842)  # A4 size
            # Title
            page.insert_text(
                (50, 70),
                f"Page {page_data['page_number']} - {page_data['title']}",
                fontsize=16,
                color=(0.2, 0.2, 0.6)
            )
            # Body text
            page.insert_textbox(
                fitz.Rect(50, 100, 545, 780),
                page_data['content'],
                fontsize=12,
                leading=16
            )
        doc.save(output_path)
        doc.close()
        print(f"Successfully generated synthetic sample PDF at: {output_path}")
        return output_path
    except ImportError:
        # Fallback to text file representation if fitz is not installed
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in SAMPLE_PAGES:
                f.write(f"--- PAGE {p['page_number']} ---\n")
                f.write(f"{p['title']}\n")
                f.write(f"{p['content']}\n\n")
        print(f"PyMuPDF not installed yet; generated text sample at: {txt_path}")
        return txt_path


if __name__ == "__main__":
    generate_sample_pdf()
