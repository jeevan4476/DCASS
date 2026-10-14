#!/usr/bin/env python3
"""
Component 2: Semantic Distribution & Selection-Bias Benchmark
Tests: KS test, Wasserstein-1, Voronoi codebook entropy, multi-modal balance
"""

from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus.index.unified_index import UnifiedSemanticIndex
from src.engine.semantic_engine import SemanticEngine

# Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_MESSAGES = 50
CODEBOOK_SIZE = 256
CORPUS_SAMPLE_SIZE = 5000
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
TABLES_DIR = BENCHMARK_DIR / "tables"
PLOTS_DIR = BENCHMARK_DIR / "plots"

torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class ModalityStats:
    modality: str
    num_carriers: int
    proportion: float
    corpus_proportion: float


@dataclass
class KSResult:
    modality: str
    dimension: int
    statistic: float
    p_value: float
    reject_null: bool


@dataclass
class Component2Result:
    wasserstein_distances: Dict[str, float]
    ks_results: List[KSResult]
    cluster_entropy: float
    cluster_entropy_ratio: float
    modality_cluster_entropy: Dict[str, float]
    modality_balance: List[ModalityStats]
    manifold_plot_path: str


def load_corpus_embeddings(engine: SemanticEngine, sample_size: int = CORPUS_SAMPLE_SIZE) -> Dict[str, np.ndarray]:
    """
    Uniformly sample unit-norm embeddings directly from FAISS indices (IndexFlatIP).
    This samples the full 31,783 images, 200,000 texts, and 13,496 audio vectors directly
    without relying on stale or incomplete /tmp/ cache files.
    """
    embeddings = {}
    rng = np.random.default_rng(SEED)

    for mod in ['image', 'text', 'audio']:
        if mod not in engine.index.indices:
            print(f"Warning: modality '{mod}' not found in index")
            embeddings[mod] = np.empty((0, 512), dtype=np.float32)
            continue

        idx = engine.index.indices[mod]
        n_total = idx.ntotal
        n_samples = min(sample_size, n_total)

        if n_samples == 0:
            embeddings[mod] = np.empty((0, 512), dtype=np.float32)
            continue

        sampled_indices = rng.choice(n_total, size=n_samples, replace=False)
        vecs = np.vstack([idx.reconstruct(int(i)) for i in sampled_indices]).astype(np.float32)

        # Ensure unit normalization
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-12
        vecs = vecs / norms

        embeddings[mod] = vecs
        print(f"Loaded {n_samples:,} {mod} embeddings from FAISS index (total: {n_total:,}): shape {vecs.shape}")

    return embeddings


def generate_dcass_carriers(
    engine: SemanticEngine,
    num_messages: int = NUM_MESSAGES
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Generate DCASS carriers using DSSC mode across varied test messages.
    Reconstructs exact unit-norm embeddings and queries real Voronoi codebook clusters.
    """
    # Diverse test message corpus for benchmarking
    base_topics = [
        "The quick brown fox jumps over the lazy dog in the sunny park.",
        "Deep convolutional neural networks achieve state-of-the-art accuracy.",
        "A peaceful mountain stream flowing through an ancient evergreen forest.",
        "Global economic growth indicators show resilient consumer spending.",
        "A classical symphony performance in the grand concert hall tonight.",
        "Astronomers discover high-energy gamma ray emissions from a distant galaxy.",
        "The bustling downtown farmer's market offers fresh local produce.",
        "Autonomous electric vehicles navigate complex urban traffic safely.",
        "Gentle rain falls on the cobblestone street under vintage gas lamps.",
        "Innovative renewable energy technologies reduce carbon emissions worldwide."
    ]

    messages = [
        f"{base_topics[i % len(base_topics)]} [msg_{i:03d}]"
        for i in range(num_messages)
    ]

    all_carriers = []
    carrier_embeddings: Dict[str, List[np.ndarray]] = {'image': [], 'text': [], 'audio': []}

    rng = np.random.default_rng(SEED)

    for i, msg in enumerate(messages):
        # Generate independent 32-byte session key per message
        session_key = rng.bytes(32)
        try:
            result = engine.encode(
                msg,
                mode='dssc',
                session_key=session_key,
                modalities=['image', 'text', 'audio']
            )

            if result.dssc_result and result.dssc_result.encoded_carriers:
                for carrier in result.dssc_result.encoded_carriers:
                    mod = carrier.media.modality
                    mid = carrier.media_id

                    # Look up exact embedding in O(1) from unified index
                    emb_norm = None
                    if mid in engine.index._id_lookup:
                        target_mod, offset = engine.index._id_lookup[mid]
                        if target_mod in engine.index.indices:
                            emb = engine.index.indices[target_mod].reconstruct(offset).reshape(1, -1).astype(np.float32)
                            norm = np.linalg.norm(emb, axis=-1, keepdims=True)
                            emb_norm = emb / max(norm, 1e-12)

                    # Look up real Voronoi cluster ID (0..255)
                    real_cluster_id = None
                    if hasattr(engine, '_vcp_mapper') and engine._vcp_mapper:
                        real_cluster_id = engine._vcp_mapper.symbol_for_media_id(mid)

                    carrier_dict = {
                        'modality': mod,
                        'media_id': mid,
                        'file_path': carrier.media.file_path,
                        'cluster_id': real_cluster_id,
                        'chunk_index': carrier.chunk_index,
                        'symbol': carrier.symbol,
                        'bits': carrier.bits,
                        'family': carrier.family,
                    }

                    if emb_norm is not None:
                        carrier_embeddings[mod].append(emb_norm)
                        carrier_dict['embedding'] = emb_norm

                    # Append carrier ONCE (no duplication)
                    all_carriers.append(carrier_dict)

        except Exception as e:
            print(f"Warning: Failed to encode message {i}: {e}")
            continue

    # Convert to arrays
    final_carrier_embeddings: Dict[str, np.ndarray] = {}
    for mod in ['image', 'text', 'audio']:
        if carrier_embeddings[mod]:
            final_carrier_embeddings[mod] = np.vstack(carrier_embeddings[mod])
        else:
            final_carrier_embeddings[mod] = np.empty((0, 512), dtype=np.float32)

    print(f"Generated {len(all_carriers)} total carriers across {num_messages} messages:")
    for mod, emb in final_carrier_embeddings.items():
        print(f"  {mod}: {len(emb)} carriers, shape {emb.shape if emb.size else 'empty'}")

    return all_carriers, final_carrier_embeddings


def compute_wasserstein_distance(corpus_emb: np.ndarray, carrier_emb: np.ndarray) -> float:
    """Compute 1D Wasserstein distance on first principal component projection."""
    if corpus_emb.size == 0 or carrier_emb.size == 0:
        return float('inf')

    # Project to 1D via PCA fitted on corpus distribution
    pca = PCA(n_components=1, random_state=SEED)
    corpus_1d = pca.fit_transform(corpus_emb).flatten()
    carrier_1d = pca.transform(carrier_emb).flatten()

    w1 = wasserstein_distance(corpus_1d, carrier_1d)
    return float(w1)


def compute_ks_tests(modality: str, corpus_emb: np.ndarray, carrier_emb: np.ndarray) -> List[KSResult]:
    """Compute 2-sample Kolmogorov-Smirnov test across all 512 embedding dimensions."""
    results = []
    if corpus_emb.size == 0 or carrier_emb.size == 0:
        return results

    min_dim = min(corpus_emb.shape[1], carrier_emb.shape[1])

    for d in range(min_dim):
        stat, p_val = ks_2samp(corpus_emb[:, d], carrier_emb[:, d])
        results.append(KSResult(
            modality=modality,
            dimension=d,
            statistic=float(stat),
            p_value=float(p_val),
            reject_null=bool(p_val <= 0.05)
        ))

    # Also test 1D PCA projection
    pca = PCA(n_components=1, random_state=SEED)
    corpus_1d = pca.fit_transform(corpus_emb).flatten()
    carrier_1d = pca.transform(carrier_emb).flatten()
    stat, p_val = ks_2samp(corpus_1d, carrier_1d)
    results.append(KSResult(
        modality=modality,
        dimension=min_dim,  # PCA projection
        statistic=float(stat),
        p_value=float(p_val),
        reject_null=bool(p_val <= 0.05)
    ))

    return results


def compute_cluster_entropy(carriers: List[Dict], num_clusters: int = CODEBOOK_SIZE) -> Tuple[float, float]:
    """Compute Voronoi cluster coverage entropy H and normalized ratio H / H_max."""
    if not carriers:
        return 0.0, 0.0

    cluster_counts = np.zeros(num_clusters, dtype=int)
    for c in carriers:
        cid = c.get('cluster_id')
        if cid is not None and 0 <= cid < num_clusters:
            cluster_counts[cid] += 1

    total = cluster_counts.sum()
    if total == 0:
        return 0.0, 0.0

    probs = cluster_counts / total
    probs = probs[probs > 0]

    H = -np.sum(probs * np.log2(probs))
    H_max = np.log2(num_clusters)

    return float(H), float(H / H_max)


def compute_modality_cluster_entropy(carriers: List[Dict], num_clusters: int = CODEBOOK_SIZE) -> Dict[str, float]:
    """Compute cluster coverage entropy partitioned per modality."""
    mod_carriers: Dict[str, List[Dict]] = {'image': [], 'text': [], 'audio': []}
    for c in carriers:
        mod = c.get('modality')
        if mod in mod_carriers:
            mod_carriers[mod].append(c)

    entropies = {}
    for mod, c_list in mod_carriers.items():
        H, _ = compute_cluster_entropy(c_list, num_clusters=num_clusters)
        entropies[mod] = H
    return entropies


def analyze_modality_balance(carriers: List[Dict], engine: SemanticEngine) -> List[ModalityStats]:
    """Analyze modality balance in DCASS carriers compared to true corpus proportions."""
    modality_counts = {}
    for c in carriers:
        mod = c.get('modality', 'unknown')
        modality_counts[mod] = modality_counts.get(mod, 0) + 1

    total_carriers = sum(modality_counts.values())

    corpus_totals = {
        mod: engine.index.indices[mod].ntotal
        for mod in ['image', 'text', 'audio']
        if mod in engine.index.indices
    }
    corpus_all = sum(corpus_totals.values())

    results = []
    for mod in ['image', 'text', 'audio']:
        count = modality_counts.get(mod, 0)
        c_tot = corpus_totals.get(mod, 0)
        results.append(ModalityStats(
            modality=mod,
            num_carriers=count,
            proportion=count / total_carriers if total_carriers > 0 else 0.0,
            corpus_proportion=c_tot / corpus_all if corpus_all > 0 else 0.0
        ))

    return results


def create_manifold_plot(
    corpus_emb: Dict[str, np.ndarray],
    carrier_emb: Dict[str, np.ndarray],
    output_path: Path
):
    """Create manifold visualization with density contours and marginal distributions."""
    from sklearn.manifold import TSNE
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Ellipse
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        width_ratios=[4, 4, 4],
        height_ratios=[4, 0.6, 0.6],
        hspace=0.3, wspace=0.3
    )

    corpus_color = '#4A90D9'   # Blue
    carrier_color = '#E85D5D'  # Red

    for idx, modality in enumerate(['image', 'text', 'audio']):
        ax_main = fig.add_subplot(gs[0, idx])
        ax_marg_x = fig.add_subplot(gs[1, idx], sharex=ax_main)
        ax_marg_y = fig.add_subplot(gs[2, idx], sharey=ax_main)

        c_emb = corpus_emb.get(modality, np.empty((0, 512)))
        d_emb = carrier_emb.get(modality, np.empty((0, 512)))

        if c_emb.shape[0] < 5 or d_emb.shape[0] < 5:
            ax_main.text(0.5, 0.5, f'Insufficient {modality} data', ha='center', va='center',
                         transform=ax_main.transAxes, fontsize=12)
            ax_main.set_title(f'{modality.capitalize()} - Insufficient Data', fontsize=13, fontweight='medium')
            continue

        n_sample = min(1500, len(c_emb), len(d_emb))
        c_idx = np.random.choice(len(c_emb), n_sample, replace=False)
        d_idx = np.random.choice(len(d_emb), n_sample, replace=False)

        combined = np.vstack([c_emb[c_idx], d_emb[d_idx]])
        labels = np.hstack([np.zeros(n_sample), np.ones(n_sample)])

        # PCA initialization for t-SNE
        pca = PCA(n_components=2, random_state=SEED)
        pca_init = pca.fit_transform(combined)

        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_sample // 4)),
            random_state=SEED,
            max_iter=1000,
            init=pca_init,
            learning_rate='auto'
        )
        coords = tsne.fit_transform(combined)

        c_coords = coords[labels == 0]
        d_coords = coords[labels == 1]

        x_min, x_max = coords[:, 0].min() - 2, coords[:, 0].max() + 2
        y_min, y_max = coords[:, 1].min() - 2, coords[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.vstack([xx.ravel(), yy.ravel()]).T

        # KDE for corpus
        try:
            kde_c = gaussian_kde(c_coords.T, bw_method=0.25)
            zz_c = kde_c(grid.T).reshape(xx.shape)
            ax_main.contourf(xx, yy, zz_c, levels=15, alpha=0.3, colors=[corpus_color])
            ax_main.contour(xx, yy, zz_c, levels=8, alpha=0.6, colors=[corpus_color], linewidths=0.8)
        except Exception:
            pass

        # KDE for carriers
        try:
            kde_d = gaussian_kde(d_coords.T, bw_method=0.25)
            zz_d = kde_d(grid.T).reshape(xx.shape)
            ax_main.contourf(xx, yy, zz_d, levels=15, alpha=0.3, colors=[carrier_color])
            ax_main.contour(xx, yy, zz_d, levels=8, alpha=0.6, colors=[carrier_color], linewidths=0.8)
        except Exception:
            pass

        # Overlay scatter
        scatter_n = min(300, len(c_coords), len(d_coords))
        c_scatter_idx = np.random.choice(len(c_coords), scatter_n, replace=False)
        d_scatter_idx = np.random.choice(len(d_coords), scatter_n, replace=False)

        ax_main.scatter(
            c_coords[c_scatter_idx, 0], c_coords[c_scatter_idx, 1],
            c=corpus_color, alpha=0.20, s=10, edgecolors='none',
            label=f'Corpus ({len(c_emb):,})', zorder=3
        )
        ax_main.scatter(
            d_coords[d_scatter_idx, 0], d_coords[d_scatter_idx, 1],
            c=carrier_color, alpha=0.35, s=14, edgecolors='none',
            label=f'DCASS ({len(d_emb):,})', zorder=4
        )

        # 1-sigma ellipses with eigenvalue safety clamping
        for points, color in [(c_coords, corpus_color), (d_coords, carrier_color)]:
            if len(points) > 10:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                evals, evecs = np.linalg.eigh(cov)
                evals = np.maximum(evals, 1e-12)
                angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
                width, height = 2 * np.sqrt(evals)
                ellipse = Ellipse(
                    mean, width, height, angle=angle,
                    edgecolor=color, facecolor='none',
                    linewidth=2, linestyle='--', alpha=0.7
                )
                ax_main.add_patch(ellipse)

        ax_main.set_title(f'{modality.capitalize()} Manifold (t-SNE)', fontsize=13, fontweight='medium', pad=10)
        ax_main.set_xlabel('t-SNE 1', fontsize=11)
        ax_main.set_ylabel('t-SNE 2', fontsize=11)
        ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_main.grid(True, alpha=0.2, linestyle=':')
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.set_aspect('equal', adjustable='box')

        # Marginals X
        ax_marg_x.hist(c_coords[:, 0], bins=30, density=True, alpha=0.5, color=corpus_color, histtype='stepfilled', label='Corpus')
        ax_marg_x.hist(d_coords[:, 0], bins=30, density=True, alpha=0.5, color=carrier_color, histtype='stepfilled', label='DCASS')
        ax_marg_x.set_ylabel('Density', fontsize=9)
        ax_marg_x.tick_params(axis='x', labelbottom=False)
        ax_marg_x.grid(True, alpha=0.2, linestyle=':')
        if idx == 0:
            ax_marg_x.legend(fontsize=8, loc='upper right')

        # Marginals Y
        ax_marg_y.hist(c_coords[:, 1], bins=30, density=True, alpha=0.5, color=corpus_color, histtype='stepfilled', orientation='horizontal')
        ax_marg_y.hist(d_coords[:, 1], bins=30, density=True, alpha=0.5, color=carrier_color, histtype='stepfilled', orientation='horizontal')
        ax_marg_y.set_xlabel('Density', fontsize=9)
        ax_marg_y.tick_params(axis='y', labelleft=False)
        ax_marg_y.grid(True, alpha=0.2, linestyle=':')

    fig.suptitle('Component 2: Semantic Manifold Overlap — Corpus vs DCASS Carriers', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Manifold plot saved to {output_path}")


def create_wasserstein_heatmap(
    corpus_emb: Dict[str, np.ndarray],
    carrier_emb: Dict[str, np.ndarray],
    output_path: Path
):
    """Create a Wasserstein distance heatmap across modalities."""
    modalities = ['image', 'text', 'audio']
    n = len(modalities)
    w1_matrix = np.zeros((n, n))

    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if mod1 in corpus_emb and mod2 in carrier_emb:
                if corpus_emb[mod1].size > 0 and carrier_emb[mod2].size > 0:
                    pca = PCA(n_components=1, random_state=SEED)
                    c_proj = pca.fit_transform(corpus_emb[mod1]).ravel()
                    d_proj = pca.transform(carrier_emb[mod2]).ravel()
                    w1_matrix[i, j] = wasserstein_distance(c_proj, d_proj)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w1_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.35)

    for i in range(n):
        for j in range(n):
            color = 'white' if w1_matrix[i, j] > 0.18 else 'black'
            ax.text(j, i, f'{w1_matrix[i, j]:.4f}', ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([m.capitalize() for m in modalities], fontsize=11)
    ax.set_yticklabels([m.capitalize() for m in modalities], fontsize=11)
    ax.set_xlabel('Carrier Modality', fontsize=12)
    ax.set_ylabel('Corpus Modality', fontsize=12)
    ax.set_title('Wasserstein-1 Distance Matrix', fontsize=13, fontweight='medium', pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('W₁ Distance', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Wasserstein heatmap saved to {output_path}")


def create_ks_heatmap(ks_results: List[KSResult], output_path: Path):
    """Create KS test statistic and p-value heatmaps across dimensions per modality."""
    if not ks_results:
        return

    modalities = ['image', 'text', 'audio']
    stat_matrix = np.zeros((len(modalities), 512))
    pval_matrix = np.zeros((len(modalities), 512))

    for i, mod in enumerate(modalities):
        mod_results = [r for r in ks_results if r.modality == mod and r.dimension < 512]
        for r in mod_results:
            stat_matrix[i, r.dimension] = r.statistic
            pval_matrix[i, r.dimension] = r.p_value

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(stat_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
    ax1.set_title('KS Statistic per Dimension', fontsize=13, fontweight='medium')
    ax1.set_xlabel('Embedding Dimension (0..511)', fontsize=11)
    ax1.set_ylabel('Modality', fontsize=11)
    ax1.set_yticks(range(len(modalities)))
    ax1.set_yticklabels([m.capitalize() for m in modalities])
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='KS Statistic')

    pval_log = -np.log10(np.clip(pval_matrix, 1e-15, 1.0))
    im2 = ax2.imshow(pval_log, cmap='Blues', aspect='auto', vmin=0, vmax=10)
    ax2.set_title('-log₁₀(p-value) per Dimension', fontsize=13, fontweight='medium')
    ax2.set_xlabel('Embedding Dimension (0..511)', fontsize=11)
    ax2.set_ylabel('Modality', fontsize=11)
    ax2.set_yticks(range(len(modalities)))
    ax2.set_yticklabels([m.capitalize() for m in modalities])
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='-log₁₀(p)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"KS heatmap saved to {output_path}")


def run_component2():
    print("=" * 80)
    print(" COMPONENT 2: SEMANTIC DISTRIBUTION & SELECTION-BIAS BENCHMARK")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize SemanticEngine
    print("\n1. Initializing SemanticEngine and FAISS indices...")
    engine = SemanticEngine()
    engine.load()

    # 2. Sample true corpus embeddings from FAISS indices
    print("\n2. Sampling true corpus embeddings from FAISS indices...")
    corpus_emb = load_corpus_embeddings(engine, sample_size=CORPUS_SAMPLE_SIZE)

    # 3. Generate DCASS carriers
    print(f"\n3. Generating carriers for {NUM_MESSAGES} diverse messages...")
    carriers, carrier_emb = generate_dcass_carriers(engine, NUM_MESSAGES)

    # 4. Compute Wasserstein distances
    print("\n4. Computing Wasserstein-1 distances against true corpus...")
    wasserstein = {}
    for modality in ['image', 'text', 'audio']:
        if modality in corpus_emb and modality in carrier_emb and corpus_emb[modality].size > 0 and carrier_emb[modality].size > 0:
            w1 = compute_wasserstein_distance(corpus_emb[modality], carrier_emb[modality])
            wasserstein[modality] = w1
            print(f"  {modality.capitalize():5s}: W1 = {w1:.6f}")
        else:
            print(f"  {modality.capitalize():5s}: No data available")

    # 5. Run KS tests per modality
    print("\n5. Running 2-sample Kolmogorov-Smirnov tests per modality...")
    ks_results: List[KSResult] = []
    mod_ks_stats: Dict[str, Tuple[float, float, int]] = {}
    for modality in ['image', 'text', 'audio']:
        if modality in corpus_emb and modality in carrier_emb and corpus_emb[modality].size > 0 and carrier_emb[modality].size > 0:
            res = compute_ks_tests(modality, corpus_emb[modality], carrier_emb[modality])
            ks_results.extend(res)
            dim_results = [r for r in res if r.dimension < 512]
            mean_stat = float(np.mean([r.statistic for r in dim_results]))
            mean_p = float(np.mean([r.p_value for r in dim_results]))
            rejections = sum(1 for r in dim_results if r.reject_null)
            mod_ks_stats[modality] = (mean_stat, mean_p, rejections)
            print(f"  {modality.capitalize():5s}: Mean KS Stat = {mean_stat:.4f}, Mean p = {mean_p:.4f}, Rejections = {rejections}/512")

    # 6. Voronoi cluster coverage entropy
    print("\n6. Computing Voronoi codebook cluster coverage entropy...")
    global_H, global_H_ratio = compute_cluster_entropy(carriers, num_clusters=CODEBOOK_SIZE)
    mod_entropies = compute_modality_cluster_entropy(carriers, num_clusters=CODEBOOK_SIZE)
    print(f"  Global Cluster Entropy: H = {global_H:.4f} bits / 8.0000 bits (ratio = {global_H_ratio:.2%})")
    for mod in ['image', 'text', 'audio']:
        print(f"    {mod.capitalize():5s} Entropy: H = {mod_entropies.get(mod, 0.0):.4f} bits")

    # 7. Modality balance
    print("\n7. Analyzing carrier modality balance vs corpus proportions...")
    modality_stats = analyze_modality_balance(carriers, engine)
    for m in modality_stats:
        print(f"  {m.modality.capitalize():5s}: {m.num_carriers:4d} carriers ({m.proportion:.1%}) vs corpus ({m.corpus_proportion:.1%})")

    # 8. Visualizations
    print("\n8. Generating benchmark plots...")
    manifold_path = PLOTS_DIR / "fig_manifold_tsne.png"
    create_manifold_plot(corpus_emb, carrier_emb, manifold_path)

    w1_heatmap_path = PLOTS_DIR / "fig_wasserstein_heatmap.png"
    create_wasserstein_heatmap(corpus_emb, carrier_emb, w1_heatmap_path)

    ks_heatmap_path = PLOTS_DIR / "fig_ks_heatmap.png"
    create_ks_heatmap(ks_results, ks_heatmap_path)

    # 9. Save JSON results
    result = Component2Result(
        wasserstein_distances=wasserstein,
        ks_results=ks_results,
        cluster_entropy=global_H,
        cluster_entropy_ratio=global_H_ratio,
        modality_cluster_entropy=mod_entropies,
        modality_balance=modality_stats,
        manifold_plot_path=str(manifold_path)
    )

    output_path = BENCHMARK_DIR / "component_2_results.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

    # 10. Generate LaTeX table (modality-specific metrics)
    tex_path = TABLES_DIR / "table2_distributional_drift.tex"
    with open(tex_path, 'w') as f:
        f.write("% Component 2: Semantic Distribution Benchmark Results\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Distributional Drift Metrics for DCASS Carrier Selection}\n")
        f.write("\\label{tab:distributional_drift}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Modality & $W_1$ Distance & KS Stat (mean) & KS $p$-value & Cluster Entropy & Carrier Proportion \\\\\n")
        f.write("\\midrule\n")
        for m_stat in modality_stats:
            mod = m_stat.modality
            w1_val = wasserstein.get(mod, 0.0)
            ks_mean_stat = mod_ks_stats.get(mod, (0.0, 1.0, 0))[0]
            ks_mean_p = mod_ks_stats.get(mod, (0.0, 1.0, 0))[1]
            mod_h = mod_entropies.get(mod, 0.0)
            pct_str = f"{m_stat.proportion * 100:.1f}\\%"
            f.write(f"{mod.capitalize()} & {w1_val:.4f} & {ks_mean_stat:.4f} & {ks_mean_p:.4f} & {mod_h:.4f} bits & {pct_str} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\textbf{{Combined}} & -- & -- & -- & \\textbf{{{global_H:.4f} bits}} & 100.0\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nResults successfully saved:")
    print(f"  JSON:   {output_path}")
    print(f"  LaTeX:  {tex_path}")
    print(f"  Plots:  {manifold_path}")
    print(f"          {w1_heatmap_path}")
    print(f"          {ks_heatmap_path}")

    return result


if __name__ == "__main__":
    run_component2()