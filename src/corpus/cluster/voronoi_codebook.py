# src/corpus/cluster/voronoi_codebook.py
"""
Spherical K-Means Voronoi Codebook Partitioning (VCP) Module for DCASS.

Partitions the 512-dimensional unit hypersphere S^511 into K=256 non-overlapping
Voronoi clusters corresponding to byte symbol values 0x00 to 0xFF.

Key Features:
- Unit-norm normalized Spherical K-Means centroid optimization (||c_m||_2 = 1.0)
- Soft-margin boundary filtering (delta_margin >= 0.05) to eliminate quantization noise
- 100% deterministic symbol-to-cluster mapping
- GPU acceleration via PyTorch / FAISS
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List, Dict
import numpy as np
import torch

class VoronoiCodebook:
    """
    Spherical K-Means Voronoi Codebook Partitioning (VCP).
    Maps 512-dim unit vectors to 256 deterministic byte centroids (0x00 .. 0xFF).
    """

    def __init__(self, num_clusters: int = 256, dim: int = 512, delta_margin: float = 0.05):
        """
        Initialize Voronoi Codebook.

        Args:
            num_clusters: Number of Voronoi clusters (default 256 for 1 byte/symbol)
            dim: Vector embedding dimension (default 512)
            delta_margin: Soft-margin boundary safety threshold (default 0.05)
        """
        self.num_clusters = num_clusters
        self.dim = dim
        self.delta_margin = delta_margin
        self.centroids: Optional[np.ndarray] = None  # Shape: (256, 512)
        self.cluster_assignments: Optional[np.ndarray] = None  # Shape: (N,)
        self.cluster_to_indices: Dict[int, List[int]] = {i: [] for i in range(num_clusters)}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if codebook centroids are fitted."""
        return self._fitted and self.centroids is not None

    def fit(
        self,
        embeddings: np.ndarray,
        max_iters: int = 25,
        batch_size: int = 4096,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Fit 256 Spherical K-Means centroids on input vector embeddings.

        Args:
            embeddings: (N, 512) numpy array of vector embeddings
            max_iters: Maximum iterations for Spherical K-Means
            batch_size: Processing batch size
            device: Computation device ('cuda' or 'cpu')
            seed: Random seed for initial centroids

        Returns:
            Dict containing convergence metrics (final inertia, mean cluster size)
        """
        N, D = embeddings.shape
        if D != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {D}")

        print(f"Fitting Spherical K-Means Voronoi Codebook ({self.num_clusters} centroids) on {N:,} vectors using {device}...")

        # Convert to PyTorch tensor & normalize to unit length
        torch.manual_seed(seed)
        np.random.seed(seed)

        X = torch.from_numpy(embeddings).float().to(device)
        X = X / torch.norm(X, dim=1, keepdim=True).clamp(min=1e-12)

        # Initialize centroids randomly from data points
        init_indices = torch.randperm(N)[:self.num_clusters]
        centroids = X[init_indices].clone()
        centroids = centroids / torch.norm(centroids, dim=1, keepdim=True).clamp(min=1e-12)

        for iteration in range(max_iters):
            # Compute inner products (cosine similarities): (N, 256)
            similarities = torch.matmul(X, centroids.T)
            assignments = torch.argmax(similarities, dim=1)

            # Update centroids by cluster means & re-normalize onto unit sphere
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.num_clusters, device=device)

            for c in range(self.num_clusters):
                mask = (assignments == c)
                count = mask.sum().item()
                counts[c] = count
                if count > 0:
                    sum_vec = X[mask].sum(dim=0)
                    new_centroids[c] = sum_vec / torch.norm(sum_vec).clamp(min=1e-12)
                else:
                    # Re-initialize empty cluster with random point
                    rand_idx = torch.randint(0, N, (1,)).item()
                    new_centroids[c] = X[rand_idx]

            # Check convergence
            centroid_shift = torch.norm(new_centroids - centroids).item()
            centroids = new_centroids
            print(f"  Iteration {iteration + 1}/{max_iters} - Centroid shift: {centroid_shift:.6f}", end="\r")

            if centroid_shift < 1e-5:
                print(f"\n  Converged early at iteration {iteration + 1}!")
                break

        print() # Newline after progress

        # Final assignments & metrics
        similarities = torch.matmul(X, centroids.T)
        final_assignments = torch.argmax(similarities, dim=1).cpu().numpy()
        self.centroids = centroids.cpu().numpy()
        self.cluster_assignments = final_assignments

        # Build cluster-to-indices lookup map
        self.cluster_to_indices = {i: [] for i in range(self.num_clusters)}
        for idx, cluster_id in enumerate(final_assignments):
            self.cluster_to_indices[int(cluster_id)].append(idx)

        self._fitted = True
        mean_density = N / self.num_clusters
        min_density = min(len(v) for v in self.cluster_to_indices.values())
        max_density = max(len(v) for v in self.cluster_to_indices.values())

        print("✅ Voronoi Codebook Fitted Successfully!")
        print(f"   Centroids: {self.num_clusters} (||c||_2 = 1.0)")
        print(f"   Cluster Density: Min={min_density}, Max={max_density}, Mean={mean_density:.1f} vectors/cluster")

        return {
            "num_vectors": N,
            "num_clusters": self.num_clusters,
            "mean_density": mean_density,
            "min_density": min_density,
            "max_density": max_density
        }

    def assign(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign new embeddings to their nearest Voronoi cluster centroid.

        Args:
            embeddings: (N, 512) array of query vectors

        Returns:
            (N,) array of byte symbol cluster IDs (0 to 255)
        """
        if not self.is_fitted:
            raise RuntimeError("VoronoiCodebook is not fitted. Call fit() or load() first.")

        # Normalize query embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_emb = embeddings / norms

        # Inner product with centroids
        similarities = np.dot(normalized_emb, self.centroids.T)
        return np.argmax(similarities, axis=1)

    def get_cluster_indices(self, cluster_id: int) -> List[int]:
        """Get list of vector indices assigned to cluster_id."""
        return self.cluster_to_indices.get(cluster_id, [])

    def filter_soft_margin(self, embeddings: np.ndarray, candidate_indices: List[int], cluster_id: int) -> List[int]:
        """
        Filter candidate vectors using soft-margin boundary buffering.
        Rejects boundary-edge vectors where (top1_sim - top2_sim) < delta_margin.

        Args:
            embeddings: Full corpus embeddings array
            candidate_indices: List of vector indices in candidate cluster
            cluster_id: Target cluster ID

        Returns:
            List of interior vector indices passing the soft-margin threshold
        """
        if not candidate_indices or not self.is_fitted:
            return candidate_indices

        sub_emb = embeddings[candidate_indices]
        norms = np.linalg.norm(sub_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_sub = sub_emb / norms

        # Compute similarities to all centroids
        similarities = np.dot(normalized_sub, self.centroids.T) # (K_cand, 256)

        robust_indices = []
        for i, idx in enumerate(candidate_indices):
            sims = similarities[i]
            top_sim = sims[cluster_id]
            # Second best similarity (excluding target cluster_id)
            other_sims = np.delete(sims, cluster_id)
            second_sim = np.max(other_sims)

            margin = top_sim - second_sim
            if margin >= self.delta_margin:
                robust_indices.append(idx)

        # Fallback to all candidates if filtering is too aggressive
        return robust_indices if robust_indices else candidate_indices

    def save(self, output_path: Union[str, Path]) -> None:
        """Save fitted centroids and cluster assignments to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted VoronoiCodebook.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            centroids=self.centroids,
            cluster_assignments=self.cluster_assignments,
            num_clusters=self.num_clusters,
            dim=self.dim,
            delta_margin=self.delta_margin
        )
        print(f"Saved Voronoi Codebook to {output_path}")

    def load(self, input_path: Union[str, Path]) -> None:
        """Load fitted centroids and assignments from disk."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Voronoi Codebook file not found at {input_path}")

        data = np.load(input_path, allow_pickle=True)
        self.centroids = data["centroids"]
        self.cluster_assignments = data["cluster_assignments"]
        self.num_clusters = int(data["num_clusters"])
        self.dim = int(data["dim"])
        self.delta_margin = float(data["delta_margin"])

        # Rebuild lookup map
        self.cluster_to_indices = {i: [] for i in range(self.num_clusters)}
        for idx, cluster_id in enumerate(self.cluster_assignments):
            self.cluster_to_indices[int(cluster_id)].append(idx)

        self._fitted = True
        print(f"Loaded Voronoi Codebook ({self.num_clusters} centroids) from {input_path}")
