from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[str(item)] = scores.get(str(item), 0.0) + 1.0 / (k + rank)
    return scores


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def rrf_score_matrix(*score_matrices: np.ndarray, k: int = 60) -> np.ndarray:
    if not score_matrices:
        raise ValueError("at least one score matrix is required")
    shape = np.asarray(score_matrices[0]).shape
    if any(np.asarray(x).shape != shape for x in score_matrices):
        raise ValueError("all score matrices must have the same shape")
    fused = np.zeros(shape, dtype=np.float32)
    for matrix in score_matrices:
        fused += 1.0 / (k + _rank_positions(np.asarray(matrix)))
    return fused


def score_matrix(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    q, c = np.asarray(query_embeddings, dtype=np.float32), np.asarray(corpus_embeddings, dtype=np.float32)
    if q.ndim != 2 or c.ndim != 2 or q.shape[1] != c.shape[1]:
        raise ValueError("embedding dimensions do not align")
    return q @ c.T


@dataclass
class BM25Retriever:
    tokenizer: Callable[[str], list[str]] | None = None
    k1: float = 1.5
    b: float = 0.75
    _vectorizer: object | None = field(default=None, init=False, repr=False)
    _doc_matrix: object | None = field(default=None, init=False, repr=False)
    _idf: np.ndarray | None = field(default=None, init=False, repr=False)
    _doc_lengths: np.ndarray | None = field(default=None, init=False, repr=False)
    _avgdl: float = field(default=0.0, init=False, repr=False)

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is not None:
            return list(self.tokenizer(str(text)))
        try:
            import jieba
            return [tok.strip() for tok in jieba.lcut(str(text)) if tok.strip()]
        except ImportError:
            import re
            return re.findall(r"[\w]+|[^\w\s]", str(text).lower(), flags=re.UNICODE)

    def fit(self, documents: Sequence[str]) -> "BM25Retriever":
        from sklearn.feature_extraction.text import CountVectorizer

        self._vectorizer = CountVectorizer(analyzer=self._tokenize, token_pattern=None, lowercase=False, dtype=np.float32)
        self._doc_matrix = self._vectorizer.fit_transform([str(x) for x in documents]).tocsr()
        df = np.asarray((self._doc_matrix > 0).sum(axis=0)).ravel()
        n = self._doc_matrix.shape[0]
        self._idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._doc_lengths = np.asarray(self._doc_matrix.sum(axis=1)).ravel().astype(np.float32)
        self._avgdl = float(self._doc_lengths.mean()) if n else 0.0
        return self

    def score(self, queries: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None or self._doc_matrix is None or self._idf is None or self._doc_lengths is None:
            raise RuntimeError("call fit before score")
        query_counts = self._vectorizer.transform([str(x) for x in queries]).tocsr()
        output = np.zeros((query_counts.shape[0], self._doc_matrix.shape[0]), dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * self._doc_lengths / max(self._avgdl, 1e-12))
        for i in range(query_counts.shape[0]):
            terms = query_counts[i].indices
            if len(terms) == 0:
                continue
            tf = self._doc_matrix[:, terms].toarray()
            contribution = self._idf[terms] * (tf * (self.k1 + 1.0) / (tf + norm[:, None]))
            output[i] = contribution.sum(axis=1)
        return output


@dataclass
class DenseRetriever:
    model_name: str
    batch_size: int = 32
    max_length: int = 512
    query_instruction: str | None = None
    model: object | None = None

    def _model(self):
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the gpu extra to run dense retrieval") from exc
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = self.max_length
        return self.model

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        values = [str(x) for x in texts]
        if is_query and self.query_instruction:
            values = [f"Instruct: {self.query_instruction}\nQuery: {x}" for x in values]
        model = self._model()
        result = model.encode(values, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(result, dtype=np.float32)

    def score(self, queries: Sequence[str], documents: Sequence[str]) -> np.ndarray:
        return score_matrix(self.encode(queries, is_query=True), self.encode(documents, is_query=False))
