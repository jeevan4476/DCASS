"""
Semantic similarity metrics for DCASS benchmarking.

This module provides metrics to evaluate how well the semantic meaning
is preserved between original messages and decoded content.

Metrics:
    - CLIP Similarity: Uses CLIP embeddings for semantic similarity
    - BERTScore: Contextual embedding similarity using BERT
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MetricResult:
    """Result from a single metric computation."""
    name: str
    score: float
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def __repr__(self) -> str:
        return f"{self.name}: {self.score:.4f}"


class BaseMetric(ABC):
    """Abstract base class for semantic similarity metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass

    @abstractmethod
    def compute(self, reference: str, candidate: str) -> MetricResult:
        """
        Compute similarity between reference and candidate text.

        Args:
            reference: Original message (ground truth)
            candidate: Decoded/reconstructed text

        Returns:
            MetricResult with score in [0, 1] range
        """
        pass

    def compute_batch(
        self,
        references: list[str],
        candidates: list[str]
    ) -> list[MetricResult]:
        """
        Compute metric for a batch of pairs.

        Args:
            references: List of original messages
            candidates: List of decoded texts

        Returns:
            List of MetricResult objects
        """
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have same length")

        return [
            self.compute(ref, cand)
            for ref, cand in zip(references, candidates)
        ]


class CLIPSimilarity(BaseMetric):
    """
    CLIP-based semantic similarity metric.

    Uses CLIP text embeddings to compute cosine similarity between
    the original message and decoded content. CLIP captures semantic
    meaning across both visual and textual concepts.
    """

    def __init__(self, device: str = None):
        """
        Args:
            device: Device for CLIP model ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None

    @property
    def name(self) -> str:
        return "CLIP Similarity"

    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is None:
            import clip
            print(f"  Loading CLIP model for metrics on {self.device}...")
            self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            self._model.eval()

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to CLIP embedding."""
        import clip
        self._load_model()

        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()

    def compute(self, reference: str, candidate: str) -> MetricResult:
        """Compute CLIP cosine similarity."""
        ref_emb = self._encode_text(reference)
        cand_emb = self._encode_text(candidate)

        # Cosine similarity (already normalized)
        similarity = float(np.dot(ref_emb, cand_emb))

        # Clamp to [0, 1] (can be negative for very dissimilar)
        similarity = max(0.0, min(1.0, similarity))

        return MetricResult(
            name=self.name,
            score=similarity,
            details={
                "reference_length": len(reference),
                "candidate_length": len(candidate),
            }
        )

    def compute_batch(
        self,
        references: list[str],
        candidates: list[str]
    ) -> list[MetricResult]:
        """Optimized batch computation using single forward pass."""
        import clip

        if len(references) != len(candidates):
            raise ValueError("references and candidates must have same length")

        self._load_model()

        with torch.no_grad():
            # Encode all texts at once
            all_texts = references + candidates
            tokens = clip.tokenize(all_texts, truncate=True).to(self.device)
            embeddings = self._model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()

        n = len(references)
        ref_embs = embeddings[:n]
        cand_embs = embeddings[n:]

        results = []
        for i in range(n):
            similarity = float(np.dot(ref_embs[i], cand_embs[i]))
            similarity = max(0.0, min(1.0, similarity))

            results.append(MetricResult(
                name=self.name,
                score=similarity,
                details={
                    "reference_length": len(references[i]),
                    "candidate_length": len(candidates[i]),
                }
            ))

        return results


class SentenceTransformerSimilarity(BaseMetric):
    """
    Sentence Transformer-based semantic similarity metric.

    Uses sentence-transformers library which is more stable and
    compatible with Python 3.13. Computes cosine similarity between
    sentence embeddings.

    This is used as an alternative to BERTScore when there are
    compatibility issues.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Args:
            model_name: Sentence transformer model name.
                       Default is a fast, high-quality model.
            device: Device for model ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    @property
    def name(self) -> str:
        return "BERTScore"  # Keep same name for compatibility

    def _load_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"  Loading Sentence Transformer ({self.model_name})...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: "
                    "pip install sentence-transformers"
                )

    def compute(self, reference: str, candidate: str) -> MetricResult:
        """Compute cosine similarity between sentence embeddings."""
        self._load_model()

        # Encode both texts
        embeddings = self._model.encode([reference, candidate], convert_to_tensor=True)

        # Compute cosine similarity
        from torch.nn.functional import cosine_similarity
        similarity = cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        # Normalize to [0, 1]
        similarity = max(0.0, min(1.0, (similarity + 1) / 2))

        return MetricResult(
            name=self.name,
            score=similarity,
            details={
                "precision": similarity,  # For compatibility
                "recall": similarity,
                "f1": similarity,
                "model": self.model_name,
            }
        )

    def compute_batch(
        self,
        references: list[str],
        candidates: list[str]
    ) -> list[MetricResult]:
        """Optimized batch computation."""
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have same length")

        self._load_model()
        from torch.nn.functional import cosine_similarity

        # Encode all at once
        ref_embeddings = self._model.encode(references, convert_to_tensor=True)
        cand_embeddings = self._model.encode(candidates, convert_to_tensor=True)

        results = []
        for i in range(len(references)):
            similarity = cosine_similarity(
                ref_embeddings[i].unsqueeze(0),
                cand_embeddings[i].unsqueeze(0)
            ).item()
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))

            results.append(MetricResult(
                name=self.name,
                score=similarity,
                details={
                    "precision": similarity,
                    "recall": similarity,
                    "f1": similarity,
                    "model": self.model_name,
                }
            ))

        return results


class BERTScoreMetric(BaseMetric):
    """
    BERTScore metric for semantic similarity.

    Uses contextual embeddings from BERT to compute precision, recall,
    and F1 scores. BERTScore captures semantic similarity at the token
    level using contextual representations.

    Note: Falls back to SentenceTransformerSimilarity if bert-score
    has compatibility issues (e.g., Python 3.13).

    Reference: https://arxiv.org/abs/1904.09675
    """

    def __init__(self, model_type: str = "microsoft/deberta-base-mnli", device: str = None):
        """
        Args:
            model_type: Hugging Face model to use for BERTScore.
                       Default is DeBERTa which performs well on semantic tasks.
            device: Device for model ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback = None
        self._use_fallback = None  # None = not determined yet

    @property
    def name(self) -> str:
        return "BERTScore"

    def _check_bertscore_compatibility(self) -> bool:
        """Check if bert-score library works on this system."""
        if self._use_fallback is not None:
            return not self._use_fallback

        try:
            from bert_score import score as bert_score_fn
            # Try a minimal test
            P, R, F1 = bert_score_fn(
                ["test"], ["test"],
                model_type=self.model_type,
                device=self.device,
                verbose=False
            )
            self._use_fallback = False
            return True
        except Exception as e:
            print(f"  BERTScore not compatible (Python 3.13 issue?): {e}")
            print("  Falling back to SentenceTransformer similarity...")
            self._use_fallback = True
            self._fallback = SentenceTransformerSimilarity(device=self.device)
            return False

    def compute(self, reference: str, candidate: str) -> MetricResult:
        """Compute BERTScore F1 (or fallback similarity)."""
        if not self._check_bertscore_compatibility():
            return self._fallback.compute(reference, candidate)

        from bert_score import score as bert_score_fn

        # BERTScore expects lists
        P, R, F1 = bert_score_fn(
            [candidate],
            [reference],
            model_type=self.model_type,
            device=self.device,
            verbose=False
        )

        f1_score = float(F1[0])

        return MetricResult(
            name=self.name,
            score=f1_score,
            details={
                "precision": float(P[0]),
                "recall": float(R[0]),
                "f1": f1_score,
                "model": self.model_type,
            }
        )

    def compute_batch(
        self,
        references: list[str],
        candidates: list[str]
    ) -> list[MetricResult]:
        """Optimized batch computation."""
        if not self._check_bertscore_compatibility():
            return self._fallback.compute_batch(references, candidates)

        from bert_score import score as bert_score_fn

        if len(references) != len(candidates):
            raise ValueError("references and candidates must have same length")

        # BERTScore handles batching internally
        P, R, F1 = bert_score_fn(
            candidates,
            references,
            model_type=self.model_type,
            device=self.device,
            verbose=False
        )

        results = []
        for i in range(len(references)):
            results.append(MetricResult(
                name=self.name,
                score=float(F1[i]),
                details={
                    "precision": float(P[i]),
                    "recall": float(R[i]),
                    "f1": float(F1[i]),
                    "model": self.model_type,
                }
            ))

        return results


class CombinedMetrics:
    """
    Combines multiple metrics for comprehensive evaluation.

    Usage:
        metrics = CombinedMetrics()
        results = metrics.evaluate("original message", "decoded content")

        # Or batch evaluation
        batch_results = metrics.evaluate_batch(originals, decoded)
    """

    def __init__(
        self,
        device: str = None,
        use_clip: bool = True,
        use_bertscore: bool = True,
        bertscore_model: str = "microsoft/deberta-base-mnli"
    ):
        """
        Args:
            device: Device for models
            use_clip: Whether to include CLIP similarity
            use_bertscore: Whether to include BERTScore
            bertscore_model: Model for BERTScore
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics: list[BaseMetric] = []

        if use_clip:
            self.metrics.append(CLIPSimilarity(device=self.device))

        if use_bertscore:
            self.metrics.append(BERTScoreMetric(
                model_type=bertscore_model,
                device=self.device
            ))

    def evaluate(self, reference: str, candidate: str) -> dict[str, MetricResult]:
        """
        Evaluate all metrics for a single pair.

        Args:
            reference: Original message
            candidate: Decoded text

        Returns:
            Dict mapping metric name to result
        """
        return {
            metric.name: metric.compute(reference, candidate)
            for metric in self.metrics
        }

    def evaluate_batch(
        self,
        references: list[str],
        candidates: list[str]
    ) -> list[dict[str, MetricResult]]:
        """
        Evaluate all metrics for a batch of pairs.

        Args:
            references: List of original messages
            candidates: List of decoded texts

        Returns:
            List of dicts, each mapping metric name to result
        """
        # Compute each metric in batch
        all_results: dict[str, list[MetricResult]] = {}

        for metric in self.metrics:
            all_results[metric.name] = metric.compute_batch(references, candidates)

        # Reorganize by sample
        n = len(references)
        return [
            {name: results[i] for name, results in all_results.items()}
            for i in range(n)
        ]

    @property
    def metric_names(self) -> list[str]:
        """Get list of metric names."""
        return [m.name for m in self.metrics]


# Convenience function
def compute_semantic_similarity(
    reference: str,
    candidate: str,
    device: str = None
) -> dict[str, float]:
    """
    Quick semantic similarity computation.

    Args:
        reference: Original message
        candidate: Decoded text
        device: Device for models

    Returns:
        Dict mapping metric name to score
    """
    metrics = CombinedMetrics(device=device)
    results = metrics.evaluate(reference, candidate)
    return {name: result.score for name, result in results.items()}
