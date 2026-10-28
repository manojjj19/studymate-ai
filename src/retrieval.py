"""Semantic Retrieval Module for StudyMate AI.

Embeds document chunks using sentence-transformers (all-MiniLM-L6-v2)
and performs vector similarity search (NumPy dot product / FAISS) to retrieve
the most relevant passages with page references.
Includes pure-Python mathematical fallback for lightweight offline testing.
"""

from typing import List, Dict, Any, Optional, Union
import math

try:
    import numpy as np
except ImportError:
    np = None

from src.pdf_reader import DocumentChunk


def _dot(v1: List[float], v2: List[float]) -> float:
    """Pure Python dot product."""
    return sum(a * b for a, b in zip(v1, v2))


def _l2_normalize(vec: List[float]) -> List[float]:
    """Pure Python L2 normalization."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return [x / norm for x in vec]
    return vec


class PassageRetriever:
    """Manages document chunk embeddings and vector similarity retrieval."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_relevance_threshold: float = 0.25,
        use_mock: bool = False
    ):
        """Initializes the retriever.
        
        Args:
            model_name: Hugging Face embedding model identifier.
            min_relevance_threshold: Minimum cosine similarity score to consider a match relevant.
            use_mock: If True, uses lightweight bag-of-words vectors for fast offline tests.
        """
        self.model_name = model_name
        self.min_relevance_threshold = min_relevance_threshold
        # If numpy is not installed, force mock mode to use pure-Python vectors
        self.use_mock = use_mock or (np is None)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[Any] = None
        self._embedder = None

    def _get_embedder(self):
        """Lazy loader for the SentenceTransformer model to save memory."""
        if self._embedder is None and not self.use_mock:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
            except ImportError:
                self.use_mock = True
        return self._embedder

    def _mock_encode(self, texts: List[str]) -> Any:
        """Lightweight token-frequency vectorizer with pure Python + NumPy support."""
        vocab: Dict[str, int] = {}
        for text in texts:
            for token in text.lower().split():
                clean_tok = "".join(ch for ch in token if ch.isalnum())
                if clean_tok and clean_tok not in vocab:
                    vocab[clean_tok] = len(vocab)

        if not vocab:
            vocab["default"] = 0

        dim = max(16, len(vocab))

        if np is not None:
            vectors = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                for token in text.lower().split():
                    clean_tok = "".join(ch for ch in token if ch.isalnum())
                    if clean_tok in vocab:
                        vectors[i, vocab[clean_tok]] += 1.0
                norm = np.linalg.norm(vectors[i])
                if norm > 0:
                    vectors[i] /= norm
                else:
                    vectors[i, 0] = 1.0
            return vectors
        else:
            # Pure Python implementation
            vectors_list = []
            for text in texts:
                row = [0.0] * dim
                for token in text.lower().split():
                    clean_tok = "".join(ch for ch in token if ch.isalnum())
                    if clean_tok in vocab:
                        row[vocab[clean_tok]] += 1.0
                vectors_list.append(_l2_normalize(row))
            return vectors_list

    def embed_texts(self, texts: List[str]) -> Any:
        """Generates L2-normalized dense embeddings for a list of strings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32) if np is not None else []

        if self.use_mock:
            return self._mock_encode(texts)

        embedder = self._get_embedder()
        embeddings = embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexes a list of document chunks.
        
        Args:
            chunks: List of DocumentChunk instances.
            
        Returns:
            Number of indexed chunks.
        """
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = None
            return 0

        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self.embed_texts(texts)
        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieves top-k most semantically relevant passages for a query.
        
        Args:
            query: The student's question.
            top_k: Maximum number of passages to return.
            
        Returns:
            List of dicts with chunk, similarity_score, is_relevant, page_number, source, text.
        """
        if not query.strip() or self.embeddings is None or len(self.chunks) == 0:
            return []

        if self.use_mock:
            all_texts = [c.text for c in self.chunks] + [query]
            all_vecs = self._mock_encode(all_texts)
            chunk_vecs = all_vecs[:-1]
            query_vec = all_vecs[-1]

            if np is not None:
                scores = np.dot(chunk_vecs, query_vec).tolist()
            else:
                scores = [_dot(cv, query_vec) for cv in chunk_vecs]
        else:
            query_vec = self.embed_texts([query])[0]
            scores = np.dot(self.embeddings, query_vec).tolist()

        k = min(top_k, len(self.chunks))
        # Sort indices descending by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = indexed_scores[:k]

        results = []
        for idx, score in top_items:
            chunk = self.chunks[idx]
            is_relevant = float(score) >= self.min_relevance_threshold

            results.append({
                "chunk": chunk,
                "similarity_score": round(float(score), 4),
                "is_relevant": is_relevant,
                "page_number": chunk.page_number,
                "source": chunk.source,
                "text": chunk.text
            })

        return results

    def has_sufficient_evidence(self, results: List[Dict[str, Any]]) -> bool:
        """Checks if at least one retrieved passage exceeds the relevance threshold."""
        if not results:
            return False
        return any(r["is_relevant"] for r in results)
