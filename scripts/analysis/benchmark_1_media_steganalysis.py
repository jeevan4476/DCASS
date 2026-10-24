#!/usr/bin/env python3
"""
DCASS Component 1: Media-Layer Steganalysis Benchmark (schema v2).

Protocol
--------
1. Image-level k-fold splitting: the pool is partitioned by image, so a cover
   and its LSB twin always land in the same split and no image is shared
   across roles. The legacy item-level split leaked twins across train/test
   for roughly 70 percent of scenes.
2. Positive control gate: a detector may only testify about DCASS after
   reaching >= 0.90 accuracy on held-out 1.0 bpp LSB twins.
3. Measured, not asserted, DCASS metrics: per-sample posteriors are recorded;
   the DCASS AUC is computed from cover versus carrier posteriors with a
   bootstrap CI over folds and a label-permutation test. The legacy plug-in KL
   is kept for continuity but calibrated against its own permutation null.
   The balanced accuracy is detection-oriented, 0.5 [P(cover decided cover) +
   P(carrier decided stego)], which equals 0.5 under blindness at any cover
   base rate rather than collapsing to the base rate itself.
4. Carrier hygiene: engine-produced carriers that overlap any image the fold's
   detector saw (train, val, or test) are excluded from the scored set; the
   exclusion counts are recorded.
5. Payload sweep: trained detectors are additionally evaluated on twins at
   0.4, 0.2, and 0.1 bpp to map how detection degrades below the training
   payload.

Artifacts (storage/benchmarks):
- component_1_fold{k}.json  split manifest, carrier provenance, posteriors,
                            per-detector metrics for fold k (resumable)
- component_1_summary.json  aggregated results across folds
- table1_steganalysis.tex   publication table
- plots/fig1_roc_curves.png/.pdf, plots/fig1_payload_sweep.png/.pdf

Detector checkpoints are written to storage/models/benchmark_1/ (local,
gitignored).

Usage:
    .venv/bin/python scripts/analysis/benchmark_1_media_steganalysis.py
    .venv/bin/python scripts/analysis/benchmark_1_media_steganalysis.py --fold 1 --force
    .venv/bin/python scripts/analysis/benchmark_1_media_steganalysis.py --summary-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.steganalysis_detectors import (  # noqa: E402
    DETECTOR_CITATIONS,
    get_detector,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_manifest import (  # noqa: E402
    build_manifest,
    caption_provenance,
    scale_table_to_textwidth,
    write_manifest,
)

SEED = 42
DEFAULT_IMAGES = 2000
DEFAULT_FOLDS = 3
DEFAULT_EPOCHS = 20
DEFAULT_DETECTORS = ["sr_net", "zhu_net", "ye_net", "xu_net", "srm_flda"]
DEFAULT_SWEEP_RATES = [1.0, 0.4, 0.2, 0.1]
DEFAULT_CARRIERS = 1000
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-3
POSITIVE_CONTROL_GATE = 0.90
KL_BINS = 50
IMAGE_SIZE = 256

FLICKR_IMAGES_DIR = PROJECT_ROOT / "storage" / "data" / "raw" / "flickr30k" / "images"
BENCHMARK_DIR = PROJECT_ROOT / "storage" / "benchmarks"
PLOTS_DIR = BENCHMARK_DIR / "plots"
TABLES_DIR = BENCHMARK_DIR / "tables"
CHECKPOINT_DIR = PROJECT_ROOT / "storage" / "models" / "benchmark_1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SHORT_REFS = {
    "srm_flda": "Fridrich \\& Kodovsky, TIFS 2012",
    "xu_net": "Xu et al., SPL 2016",
    "ye_net": "Ye et al., TIFS 2017",
    "zhu_net": "Zhu et al., TIFS 2020",
    "sr_net": "Boroumand et al., TIFS 2018",
}


# ---------------------------------------------------------------------------
# Cohort selection, image-level folds, small statistics
# ---------------------------------------------------------------------------

def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def select_image_pool(num_images: int, seed: int = SEED) -> List[Path]:
    """Deterministic Flickr30k pool: the first num_images entries of a seeded
    permutation of the sorted corpus listing."""
    all_images = sorted(
        p for p in FLICKR_IMAGES_DIR.glob("*.jpg") if not p.name.startswith("._")
    )
    if len(all_images) < num_images:
        raise ValueError(f"Requested {num_images} images, only {len(all_images)} exist")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_images), size=num_images, replace=False)
    return [all_images[i] for i in sorted(int(i) for i in chosen)]


def pool_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def make_image_level_folds(
    pool_size: int, num_folds: int, seed: int = SEED
) -> List[List[int]]:
    """Partition pool indices into num_folds disjoint image-level folds."""
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(pool_size)
    return [sorted(int(i) for i in part) for part in np.array_split(perm, num_folds)]


def split_train_val(
    rest: Sequence[int], val_ratio: float = 0.1, seed: int = SEED
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed + 2)
    order = [int(i) for i in rng.permutation(len(rest))]
    n_val = max(1, int(round(val_ratio * len(rest))))
    val = sorted(rest[o] for o in order[:n_val])
    train = sorted(rest[o] for o in order[n_val:])
    return train, val


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score(pos) > score(neg)) with average ranks for ties."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def cover_class_rate(scores: np.ndarray) -> float:
    """Fraction of carrier posteriors decided cover at the 0.5 threshold.

    Orientation matters: this is P(score <= 0.5), not P(score > 0.5). The
    inverse once leaked into a component 1 artifact as a mislabeled field."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    return float((scores <= 0.5).mean())


def balanced_accuracy(cover: np.ndarray, carrier: np.ndarray) -> float:
    """Detection-oriented balanced accuracy on the carrier-versus-cover test.

    Bal = 0.5 [P(cover decided cover) + P(carrier decided stego)]. Under the
    blindness hypothesis the two score distributions coincide, so the two
    decision rates sum to one and Bal is exactly 0.5 at any cover base rate.
    A readable signal shifts the metric by half the decision-rate gap. An
    orientation flip once leaked into an artifact as the 0.5 [P(cover cover)
    + P(carrier cover)] variant, which converges to the cover base rate in
    the high eighties or nineties instead of 0.5."""
    cover = np.asarray(cover, dtype=np.float64)
    carrier = np.asarray(carrier, dtype=np.float64)
    if cover.size == 0 or carrier.size == 0:
        return float("nan")
    return float(0.5 * ((cover <= 0.5).mean() + (carrier > 0.5).mean()))


def permutation_auc_test(
    cover: np.ndarray, carrier: np.ndarray, n_perm: int = 2000, seed: int = 0
) -> Dict[str, float]:
    """Two-sided label-permutation test for AUC == 0.5 (carrier versus cover)."""
    cover = np.asarray(cover, dtype=np.float64)
    carrier = np.asarray(carrier, dtype=np.float64)
    observed = rank_auc(carrier, cover)
    pooled = np.concatenate([cover, carrier])
    n_cover = cover.size
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        perm = rng.permutation(pooled.size)
        null[b] = rank_auc(pooled[perm[:n_cover]], pooled[perm[n_cover:]])
    deviation = abs(observed - 0.5)
    p = float((1 + np.sum(np.abs(null - 0.5) >= deviation)) / (n_perm + 1))
    return {
        "auc": float(observed),
        "p": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def bootstrap_auc_ci(
    cover: np.ndarray, carrier: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> Dict[str, float]:
    """Percentile bootstrap CI for the carrier-versus-cover AUC."""
    rng = np.random.default_rng(seed)
    cover = np.asarray(cover, dtype=np.float64)
    carrier = np.asarray(carrier, dtype=np.float64)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        c = cover[rng.integers(0, cover.size, cover.size)]
        k = carrier[rng.integers(0, carrier.size, carrier.size)]
        boot[b] = rank_auc(k, c)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def plug_in_kl(cover: np.ndarray, carrier: np.ndarray, bins: int = KL_BINS) -> float:
    """Legacy descriptive statistic kept for continuity: plug-in histogram KL
    of carrier posteriors relative to cover posteriors, 50 bins, 1e-8 floor."""
    hist_cover, edges = np.histogram(cover, bins=bins, range=(0.0, 1.0), density=True)
    hist_carrier, _ = np.histogram(carrier, bins=edges, density=True)
    hist_cover = np.clip(hist_cover, 1e-8, None)
    hist_carrier = np.clip(hist_carrier, 1e-8, None)
    hist_cover = hist_cover / hist_cover.sum()
    hist_carrier = hist_carrier / hist_carrier.sum()
    return float(np.sum(hist_carrier * np.log(hist_carrier / hist_cover)))


def kl_permutation_test(
    cover: np.ndarray, carrier: np.ndarray, n_perm: int = 1000, seed: int = 0
) -> Dict[str, float]:
    """Calibrate the plug-in KL against its own sampling null."""
    cover = np.asarray(cover, dtype=np.float64)
    carrier = np.asarray(carrier, dtype=np.float64)
    observed = plug_in_kl(cover, carrier)
    pooled = np.concatenate([cover, carrier])
    n_cover = cover.size
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(pooled.size)
        null[b] = plug_in_kl(pooled[perm[:n_cover]], pooled[perm[n_cover:]])
    p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return {
        "kl": observed,
        "p": p,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
    }


def bootstrap_mean_auc_over_folds(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]], n_boot: int = 1000, seed: int = 0
) -> Dict[str, float]:
    """CI for the mean AUC across folds via stratified resampling in folds.
    pairs: list of (positive_scores, negative_scores)."""
    point = float(np.mean([rank_auc(np.asarray(p), np.asarray(n)) for p, n in pairs]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for pos, neg in pairs:
            pos = np.asarray(pos)
            neg = np.asarray(neg)
            p = pos[rng.integers(0, pos.size, pos.size)]
            n = neg[rng.integers(0, neg.size, neg.size)]
            vals.append(rank_auc(p, n))
        boot[b] = float(np.mean(vals))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": point, "lo": float(lo), "hi": float(hi)}


def bootstrap_mean_balanced_over_folds(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]], n_boot: int = 1000, seed: int = 0
) -> Dict[str, float]:
    """CI for the fold-mean balanced accuracy between covers and carriers.
    pairs: list of (cover_scores, carrier_scores)."""
    point = float(
        np.mean([balanced_accuracy(np.asarray(c), np.asarray(k)) for c, k in pairs])
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for c, k in pairs:
            c = np.asarray(c)
            k = np.asarray(k)
            vals.append(
                balanced_accuracy(
                    c[rng.integers(0, c.size, c.size)],
                    k[rng.integers(0, k.size, k.size)],
                )
            )
        boot[b] = float(np.mean(vals))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": point, "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------------------
# Media loading and paired cover / LSB-twin dataset
# ---------------------------------------------------------------------------

def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    tensor = torch.tensor(np.array(img), dtype=torch.float32).div_(255.0)
    return tensor.unsqueeze(0)


def embed_twin(tensor: torch.Tensor, uid: int, rate: float, seed: int = SEED) -> torch.Tensor:
    """Deterministic LSB-matching twin: +-1 grey-level noise on a fixed-size
    seeded pixel subset. Determinism depends only on (uid, rate, seed)."""
    flat = tensor.flatten().clone()
    n = flat.numel()
    k = min(n, int(round(n * rate)))
    rng = np.random.default_rng([seed, int(uid), int(round(rate * 1000))])
    idx = rng.choice(n, size=k, replace=False)
    noise = rng.choice([-1.0, 1.0], size=k).astype(np.float32) / 255.0
    flat[idx] += torch.from_numpy(noise)
    return torch.clamp(flat.view_as(tensor), 0.0, 1.0)


class PairedImageDataset(Dataset):
    """Each image contributes two items: cover (even index) and its LSB twin
    (odd index). Twin noise is deterministic per (uid, rate, seed), so the
    loader worker count and access order cannot change it."""

    def __init__(self, image_paths, uids, lsb_rate: float = 1.0, seed: int = SEED):
        self.image_paths = [Path(p) for p in image_paths]
        self.uids = [int(u) for u in uids]
        self.lsb_rate = float(lsb_rate)
        self.seed = int(seed)
        if len(self.image_paths) != len(self.uids):
            raise ValueError("image_paths and uids must have equal length")

    def __len__(self) -> int:
        return 2 * len(self.image_paths)

    def __getitem__(self, index: int):
        image_index, is_stego = divmod(index, 2)
        tensor = load_image_tensor(self.image_paths[image_index])
        if is_stego:
            tensor = embed_twin(tensor, self.uids[image_index], self.lsb_rate, self.seed)
        return tensor, int(is_stego)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def seed_training(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_batch(model: nn.Module, name: str, images: torch.Tensor) -> np.ndarray:
    """Unified posterior P(stego) for every detector family.

    Scoring always runs in eval mode. BatchNorm statistics must come from the
    running estimates rather than the current batch, otherwise the score of an
    image depends on its batch co-members (8-image test batches versus
    32-image carrier batches), which fabricates a carrier-versus-cover gap
    where no signal exists."""
    model.eval()
    if name == "srm_flda":
        with torch.no_grad():
            features = model.extract_features(images).cpu().numpy()
        return np.asarray(model.flda.predict_proba(features)[:, 1], dtype=np.float64)
    with torch.no_grad():
        logits = model(images)
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float64)


def score_loader(
    model: nn.Module, name: str, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Score a PairedImageDataset loader; returns (cover, stego) posteriors and
    refuses to return misaligned arrays."""
    covers: List[np.ndarray] = []
    stegos: List[np.ndarray] = []
    seen: List[int] = []
    for images, labels in loader:
        probs = score_batch(model, name, images.to(DEVICE))
        seen.extend(int(v) for v in labels)
        covers.append(probs[::2])
        stegos.append(probs[1::2])
    if seen != [i % 2 for i in range(len(seen))]:
        raise RuntimeError("paired loader order violated; refusing misaligned scores")
    return np.concatenate(covers), np.concatenate(stegos)


def evaluate_accuracy(model: nn.Module, name: str, loader: DataLoader) -> float:
    cover, stego = score_loader(model, name, loader)
    correct = int((cover <= 0.5).sum() + (stego > 0.5).sum())
    return correct / max(cover.size + stego.size, 1)


def train_deep_detector(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    checkpoint_path: Path,
    seed: int,
) -> Tuple[nn.Module, Dict[str, object]]:
    seed_training(seed)
    model = get_detector(name).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        steps = 0
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            loss = criterion(model(images), labels)
            (loss / GRAD_ACCUM_STEPS).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            running += float(loss.item())
            steps += 1
        val_acc = evaluate_accuracy(model, name, val_loader)
        history.append(
            {"epoch": epoch + 1, "train_loss": running / max(steps, 1), "val_acc": val_acc}
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"    epoch {epoch + 1}/{epochs}: "
            f"loss={running / max(steps, 1):.4f} val_acc={val_acc:.4f} best={best_acc:.4f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, checkpoint_path)
    return model, {
        "params": int(params),
        "best_val_acc": float(best_acc),
        "runtime_s": round(time.time() - started, 1),
        "history": history,
        "checkpoint": relative(checkpoint_path),
    }


def train_srm_flda(
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    seed: int,
) -> Tuple[nn.Module, Dict[str, object]]:
    seed_training(seed)
    model = get_detector("srm_flda").to(DEVICE)
    model.eval()
    started = time.time()
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, batch_labels in train_loader:
            features.append(model.extract_features(images.to(DEVICE)).cpu())
            labels.append(batch_labels)
    model.fit_flda(torch.cat(features), torch.cat(labels))
    val_acc = evaluate_accuracy(model, "srm_flda", val_loader)
    try:
        import joblib

        joblib.dump(model.flda, checkpoint_path.with_suffix(".joblib"))
        checkpoint_note = relative(checkpoint_path.with_suffix(".joblib"))
    except Exception as exc:  # environment dependent (joblib missing)
        checkpoint_note = f"not saved: {exc}"
    return model, {
        "params": int(sum(p.numel() for p in model.parameters())),
        "best_val_acc": float(val_acc),
        "runtime_s": round(time.time() - started, 1),
        "history": [],
        "checkpoint": checkpoint_note,
    }

# ---------------------------------------------------------------------------
# Carrier generation and scoring
# ---------------------------------------------------------------------------

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from src.engine.semantic_engine import SemanticEngine

        engine = SemanticEngine()
        engine.load()
        _ENGINE = engine
    return _ENGINE


def generate_dcass_carriers(
    num_carriers: int,
    seed: int,
    exclude_names: Sequence[str],
    message_budget: int = 120,
) -> Tuple[List[Tuple[Path, str]], Dict[str, object]]:
    """Pull carriers through the real engine, deduplicate, and pad from the
    corpus if the engine yields fewer than requested. Every step is counted so
    the artifact records exactly how the set was produced."""
    engine = get_engine()
    collected: List[Tuple[Path, str]] = []
    seen: set = set()
    duplicates = 0
    failures = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for message_index in range(message_budget):
        if len(collected) >= num_carriers:
            break
        message = f"Component 1 carrier sampling seed {seed} message {message_index}"
        for mode in ("exact_vcp", "dssc"):
            if len(collected) >= num_carriers:
                break
            try:
                if mode == "dssc":
                    result = engine.encode(
                        message, mode=mode, session_key=rng.bytes(32), modalities=["image"]
                    )
                    encoded = result.dssc_result.encoded_carriers if result.dssc_result else []
                else:
                    result = engine.encode(
                        message, mode=mode, modalities=["image"], diversity_mode="round_robin"
                    )
                    encoded = result.exact_vcp_result.encoded if result.exact_vcp_result else []
            except Exception as exc:
                failures += 1
                print(f"    carrier encode failure (message {message_index}, {mode}): {exc}")
                continue
            for enc in encoded:
                if enc.media and enc.media.modality == "image":
                    path = Path(enc.media.file_path)
                    key = relative(path)
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    collected.append((path, mode))
                    if len(collected) >= num_carriers:
                        break
    padded = 0
    if len(collected) < num_carriers:
        all_images = sorted(
            p for p in FLICKR_IMAGES_DIR.glob("*.jpg") if not p.name.startswith("._")
        )
        banned = set(exclude_names) | {p.name for p, _ in collected}
        candidates = [p for p in all_images if p.name not in banned]
        pad_rng = np.random.default_rng(seed + 3)
        order = pad_rng.permutation(len(candidates))
        for o in order:
            if len(collected) >= num_carriers:
                break
            path = candidates[int(o)]
            collected.append((path, "padding_corpus"))
            padded += 1
    meta: Dict[str, object] = {
        "engine_generated": len(collected) - padded,
        "padding": padded,
        "duplicates_removed": duplicates,
        "encode_failures": failures,
    }
    return collected, meta


def score_carriers(model: nn.Module, name: str, carrier_paths: Sequence[Path]) -> np.ndarray:
    """Score carriers in BATCH_SIZE chunks. The chunk size is deliberately the
    same as every other scoring path: a batch of 32 through the widest detector
    (zhu_net) was the largest single VRAM allocation of the whole run and
    tripped the CUDA caching allocator on the 6 GB card. Scores are unchanged
    because scoring runs in eval mode, only peak memory changes."""
    out: List[np.ndarray] = []
    for i in range(0, len(carrier_paths), BATCH_SIZE):
        batch = torch.stack([load_image_tensor(p) for p in carrier_paths[i : i + BATCH_SIZE]])
        out.append(score_batch(model, name, batch.to(DEVICE)))
    return np.concatenate(out) if out else np.array([])


def score_rate(
    model: nn.Module, name: str, paths: Sequence[Path], uids: Sequence[int], rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = PairedImageDataset(paths, uids, lsb_rate=rate)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return score_loader(model, name, loader)


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------

def run_fold(config: Dict[str, object], fold: int) -> Dict[str, object]:
    protocol = {
        key: config[key]
        for key in (
            "images",
            "folds",
            "epochs",
            "detectors",
            "carriers",
            "sweep_rates",
            "seed",
        )
    }
    pool = select_image_pool(config["images"])
    folds = make_image_level_folds(len(pool), config["folds"])
    test_idx = folds[fold]
    rest_idx = [i for f in range(config["folds"]) if f != fold for i in folds[f]]
    train_idx, val_idx = split_train_val(rest_idx)
    train_paths = [pool[i] for i in train_idx]
    val_paths = [pool[i] for i in val_idx]
    test_paths = [pool[i] for i in test_idx]

    fold_path = BENCHMARK_DIR / f"component_1_fold{fold}.json"
    state: Dict[str, object]
    if fold_path.exists():
        loaded = json.loads(fold_path.read_text())
        state = loaded if loaded.get("config") == protocol and not config["force"] else {}
    else:
        state = {}
    if not state:
        state = {
            "schema": "component1-fold-v2",
            "fold": fold,
            "config": protocol,
            "dataset": {
                "images_dir": relative(FLICKR_IMAGES_DIR),
                "pool_size": len(pool),
                "pool_sha256": pool_fingerprint(pool),
                "seed": SEED,
            },
            "split": {
                "train": [p.name for p in train_paths],
                "val": [p.name for p in val_paths],
                "test": [p.name for p in test_paths],
            },
            "detectors": {},
            "complete": False,
        }
        print(
            f"  fold {fold}: train={len(train_paths)} val={len(val_paths)} "
            f"test={len(test_paths)} images"
        )
    state["dataset"]["pool"] = [p.name for p in pool]

    if "carriers" not in state:
        raw, meta = generate_dcass_carriers(
            config["carriers"], seed=SEED + 7 * fold, exclude_names=[p.name for p in pool]
        )
        split_names = {
            "train": {p.name for p in train_paths},
            "val": {p.name for p in val_paths},
            "test": {p.name for p in test_paths},
        }
        overlap = {"train": 0, "val": 0, "test": 0}
        scored: List[Tuple[Path, str]] = []
        for path, tag in raw:
            role = next((r for r, names in split_names.items() if path.name in names), None)
            if role:
                overlap[role] += 1
                continue
            scored.append((path, tag))
        mode_counts: Dict[str, int] = {}
        for _, tag in scored:
            mode_counts[tag] = mode_counts.get(tag, 0) + 1
        state["carriers"] = {
            "requested": config["carriers"],
            "engine_generated": meta["engine_generated"],
            "padded": meta["padding"],
            "duplicates_removed": meta["duplicates_removed"],
            "encode_failures": meta["encode_failures"],
            "excluded_overlap": overlap,
            "mode_counts": mode_counts,
            "ids": [relative(p) for p, _ in scored],
            "modes": [tag for _, tag in scored],
        }
        fold_path.write_text(json.dumps(state, indent=2))
        print(
            f"  fold {fold}: carriers scored={len(scored)} "
            f"(engine={meta['engine_generated']} padded={meta['padding']} "
            f"excluded={overlap} failures={meta['encode_failures']})"
        )
    carrier_paths = [PROJECT_ROOT / rel for rel in state["carriers"]["ids"]]

    for det_index, name in enumerate(protocol["detectors"]):
        entry = state["detectors"].get(name)
        if entry is not None and not config["force"]:
            print(f"  fold {fold} {name}: recorded, skipping (use --force to retrain)")
            continue
        print(f"  fold {fold} {name}: training on device {DEVICE}")
        seed = SEED + 1000 * fold + 100 * det_index
        train_ds = PairedImageDataset(train_paths, train_idx, lsb_rate=1.0)
        val_ds = PairedImageDataset(val_paths, val_idx, lsb_rate=1.0)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=config["workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        checkpoint = CHECKPOINT_DIR / f"{name}_fold{fold}.pt"
        if name == "srm_flda":
            model, info = train_srm_flda(train_loader, val_loader, checkpoint, seed)
        else:
            model, info = train_deep_detector(
                name, train_loader, val_loader, config["epochs"], checkpoint, seed
            )

        test_ds = PairedImageDataset(test_paths, test_idx, lsb_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        cover, stego = score_loader(model, name, test_loader)
        control_correct = int((cover <= 0.5).sum() + (stego > 0.5).sum())
        control_n = int(cover.size + stego.size)
        control_acc = control_correct / control_n
        control_auc = rank_auc(stego, cover)

        carrier_scores = score_carriers(model, name, carrier_paths)
        auc_perm = permutation_auc_test(cover, carrier_scores, seed=seed + 11)
        auc_ci = bootstrap_auc_ci(cover, carrier_scores, seed=seed + 12)
        kl_perm = kl_permutation_test(cover, carrier_scores, seed=seed + 13)
        cover_rate = cover_class_rate(carrier_scores)
        balanced = balanced_accuracy(cover, carrier_scores)

        sweep: Dict[str, Dict[str, object]] = {}
        for rate in protocol["sweep_rates"]:
            if abs(rate - 1.0) < 1e-9:
                r_cover, r_stego = cover, stego
            else:
                r_cover, r_stego = score_rate(model, name, test_paths, test_idx, rate)
            correct = int((r_cover <= 0.5).sum() + (r_stego > 0.5).sum())
            total = int(r_cover.size + r_stego.size)
            sweep[f"{rate:g}"] = {
                "accuracy": correct / total,
                "fpr": float((r_cover > 0.5).mean()),
                "auc": rank_auc(r_stego, r_cover),
                "correct": correct,
                "n": total,
                "cover_fp": int((r_cover > 0.5).sum()),
                "cover_n": int(r_cover.size),
            }

        state["detectors"][name] = {
            "citation": DETECTOR_CITATIONS.get(name, ""),
            "params": info["params"],
            "best_val_acc": info["best_val_acc"],
            "runtime_s": info["runtime_s"],
            "checkpoint": info["checkpoint"],
            "training_seed": seed,
            "control": {
                "accuracy": control_acc,
                "auc": control_auc,
                "correct": control_correct,
                "n": control_n,
                "cover_n": int(cover.size),
                "stego_n": int(stego.size),
                "gate_pass": bool(control_acc >= POSITIVE_CONTROL_GATE),
            },
            "dcass": {
                "auc": auc_perm["auc"],
                "perm_p": auc_perm["p"],
                "auc_ci": [auc_ci["lo"], auc_ci["hi"]],
                "kl": kl_perm["kl"],
                "kl_perm_p": kl_perm["p"],
                "kl_null_mean": kl_perm["null_mean"],
                "kl_null_p95": kl_perm["null_p95"],
                "cover_class_rate": cover_rate,
                "balanced_accuracy": balanced,
                "carrier_n": int(carrier_scores.size),
            },
            "sweep": sweep,
            "scores": {
                "cover": [round(float(v), 6) for v in cover],
                "stego": [round(float(v), 6) for v in stego],
                "carrier": [round(float(v), 6) for v in carrier_scores],
            },
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fold_path.write_text(json.dumps(state, indent=2))
        print(
            f"  fold {fold} {name}: control_acc={control_acc:.4f} "
            f"dcass_auc={auc_perm['auc']:.4f} [{auc_ci['lo']:.4f},{auc_ci['hi']:.4f}] "
            f"cover_rate={cover_rate:.4f} kl={kl_perm['kl']:.4f} p={kl_perm['p']:.3f}"
        )

    state["complete"] = all(d in state["detectors"] for d in protocol["detectors"])
    fold_path.write_text(json.dumps(state, indent=2))
    return state

# ---------------------------------------------------------------------------
# Aggregation, table, figures
# ---------------------------------------------------------------------------

def provenance() -> Dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "git_commit": commit,
        "torch": torch.__version__,
        "device": gpu,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_component1_manifest(summary: Dict[str, object]) -> None:
    """Write the Component 1 run manifest.

    It is written after the table and the plots, so the hashes it records
    are of the final files rather than of whatever was on disk from an
    earlier run.
    """
    write_manifest(
        build_manifest(
            component="component_1_media_steganalysis",
            outputs=[
                BENCHMARK_DIR / "component_1_summary.json",
                TABLES_DIR / "table1_steganalysis.tex",
                PLOTS_DIR / "fig1_roc_curves.png",
                PLOTS_DIR / "fig1_payload_sweep.png",
            ],
            inputs=sorted(
                p
                for p in (
                    BENCHMARK_DIR / f"component_1_fold{i}.json" for i in range(3)
                )
                if p.exists()
            ),
            seeds={"seed": SEED, "protocol_seed": summary["protocol"]["seed"]},
            extra={
                "protocol": summary["protocol"],
                "detectors": [d["name"] for d in summary["detectors"]],
            },
        ),
        BENCHMARK_DIR / "component_1_summary_manifest.json",
    )


def aggregate(config: Dict[str, object]) -> Dict[str, object]:
    folds_data = []
    for fold in range(config["folds"]):
        fold_path = BENCHMARK_DIR / f"component_1_fold{fold}.json"
        if not fold_path.exists():
            raise FileNotFoundError(f"missing fold artifact {fold_path.name}")
        data = json.loads(fold_path.read_text())
        missing = [d for d in config["detectors"] if d not in data.get("detectors", {})]
        if missing:
            raise RuntimeError(f"fold {fold} missing detectors: {missing}")
        folds_data.append(data)

    summary: Dict[str, object] = {
        "schema": "component1-summary-v2",
        "provenance": provenance(),
        "protocol": {
            "images": config["images"],
            "folds": config["folds"],
            "epochs": config["epochs"],
            "carriers": config["carriers"],
            "sweep_rates": config["sweep_rates"],
            "positive_control_gate": POSITIVE_CONTROL_GATE,
            "seed": SEED,
        },
        "detectors": [],
        "all_gates_passed": True,
    }
    for det_index, name in enumerate(config["detectors"]):
        entries = [fd["detectors"][name] for fd in folds_data]
        pc_correct = sum(e["control"]["correct"] for e in entries)
        pc_n = sum(e["control"]["n"] for e in entries)
        pc_acc = pc_correct / pc_n
        pc_ci = wilson_ci(pc_correct, pc_n)
        pc_aucs = [e["control"]["auc"] for e in entries]
        dcass_aucs = [e["dcass"]["auc"] for e in entries]
        dcass_ci = bootstrap_mean_auc_over_folds(
            [
                (np.asarray(e["scores"]["carrier"]), np.asarray(e["scores"]["cover"]))
                for e in entries
            ],
            seed=SEED + det_index,
        )
        control_ci = bootstrap_mean_auc_over_folds(
            [
                (np.asarray(e["scores"]["stego"]), np.asarray(e["scores"]["cover"]))
                for e in entries
            ],
            seed=SEED + det_index,
        )
        balanced_ci = bootstrap_mean_balanced_over_folds(
            [
                (np.asarray(e["scores"]["cover"]), np.asarray(e["scores"]["carrier"]))
                for e in entries
            ],
            seed=SEED + det_index,
        )
        kls = [e["dcass"]["kl"] for e in entries]
        kl_ps = [e["dcass"]["kl_perm_p"] for e in entries]
        cover_rates = [cover_class_rate(e["scores"]["carrier"]) for e in entries]
        sweep: Dict[str, Dict[str, object]] = {}
        for rate in config["sweep_rates"]:
            key = f"{rate:g}"
            cells = [e["sweep"][key] for e in entries]
            correct = sum(c["correct"] for c in cells)
            total = sum(c["n"] for c in cells)
            fp = sum(c["cover_fp"] for c in cells)
            cn = sum(c["cover_n"] for c in cells)
            sweep[key] = {
                "accuracy": correct / total,
                "accuracy_ci": list(wilson_ci(correct, total)),
                "fpr": fp / cn,
                "auc_mean": float(np.mean([c["auc"] for c in cells])),
                "n": total,
            }
        gates = [bool(e["control"]["gate_pass"]) for e in entries]
        if not all(gates):
            summary["all_gates_passed"] = False
        summary["detectors"].append(
            {
                "name": name,
                "citation": entries[0]["citation"],
                "reference": SHORT_REFS.get(name, name),
                "control": {
                    "accuracy": pc_acc,
                    "accuracy_ci": list(pc_ci),
                    "auc_mean": float(np.mean(pc_aucs)),
                    "auc_std": float(np.std(pc_aucs)),
                    "auc_ci": [control_ci["lo"], control_ci["hi"]],
                    "n": pc_n,
                    "gate_pass_all": all(gates),
                },
                "dcass": {
                    "auc_mean": float(np.mean(dcass_aucs)),
                    "auc_std": float(np.std(dcass_aucs)),
                    "auc_ci": [dcass_ci["lo"], dcass_ci["hi"]],
                    "balanced_mean": balanced_ci["mean"],
                    "balanced_ci": [balanced_ci["lo"], balanced_ci["hi"]],
                    "cover_rate_mean": float(np.mean(cover_rates)),
                    "cover_rate_std": float(np.std(cover_rates)),
                    "kl_mean": float(np.mean(kls)),
                    "kl_std": float(np.std(kls)),
                    "kl_p_values": kl_ps,
                    "kl_p_min": float(min(kl_ps)),
                    "kl_p_max": float(max(kl_ps)),
                    "carrier_n_mean": float(
                        np.mean([e["dcass"]["carrier_n"] for e in entries])
                    ),
                },
                "sweep": sweep,
                "per_fold": [
                    {
                        "fold": fd["fold"],
                        "control_acc": e["control"]["accuracy"],
                        "control_auc": e["control"]["auc"],
                        "dcass_auc": e["dcass"]["auc"],
                        "dcass_auc_ci": e["dcass"]["auc_ci"],
                        "dcass_perm_p": e["dcass"]["perm_p"],
                        "kl": e["dcass"]["kl"],
                        "kl_perm_p": e["dcass"]["kl_perm_p"],
                        "cover_class_rate": cover_class_rate(e["scores"]["carrier"]),
                        "balanced_accuracy": balanced_accuracy(
                            np.asarray(e["scores"]["cover"]),
                            np.asarray(e["scores"]["carrier"]),
                        ),
                        "checkpoint": e["checkpoint"],
                        "runtime_s": e["runtime_s"],
                    }
                    for fd, e in zip(folds_data, entries)
                ],
            }
        )
    summary_path = BENCHMARK_DIR / "component_1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_table(summary)
    plot_figures(summary)
    write_component1_manifest(summary)
    return summary


def write_table(summary: Dict[str, object], commit: str | None = None) -> None:
    lines = [
        "% Generated by scripts/analysis/benchmark_1_media_steganalysis.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Detector & Reference & PC acc (\\%), 95\\% CI & PC AUC & DCASS AUC, 95\\% CI & Cover rate (\\%) & $D_{\\mathrm{KL}}$ [perm $p$] \\\\",
        "\\midrule",
    ]
    for det in summary["detectors"]:
        c = det["control"]
        d = det["dcass"]
        lines.append(
            "{} & {} & {:.2f} [{:.2f}, {:.2f}] & {:.4f} & {:.4f} [{:.4f}, {:.4f}] & {:.2f} & {:.2f} [{:.3f}, {:.3f}] \\\\".format(
                det["name"].replace("_", "\\_"),
                det["reference"],
                100 * c["accuracy"],
                100 * c["accuracy_ci"][0],
                100 * c["accuracy_ci"][1],
                c["auc_mean"],
                d["auc_mean"],
                d["auc_ci"][0],
                d["auc_ci"][1],
                100 * d["cover_rate_mean"],
                d["kl_mean"],
                d["kl_p_min"],
                d["kl_p_max"],
            )
        )
    provenance = caption_provenance(
        n=f"{summary['protocol']['folds']} folds x {summary['protocol']['images']} images, "
        f"{len(summary['detectors'])} gated detectors",
        scope="image-only content-residual negative control",
        commit=commit or summary.get("provenance", {}).get("git_commit"),
    )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Media-layer steganalysis resistance on an image-level %d-fold protocol. "
        "PC acc: pooled positive-control accuracy on held-out 1.0 bpp LSB twins with a Wilson 95\\%% CI. "
        "PC AUC and DCASS AUC: fold-mean AUC; the DCASS AUC is measured from recorded posteriors "
        "with a 95\\%% stratified bootstrap CI over folds. Cover rate: fraction of DCASS carriers "
        "decided cover at threshold 0.5. $D_{\\mathrm{KL}}$: legacy plug-in statistic with its "
        "label-permutation $p$ range across folds.%s}"
        % (summary["protocol"]["folds"], provenance),
        "\\label{tab:steganalysis}",
        "\\end{table}",
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "table1_steganalysis.tex"
    table_path.write_text("\n".join(lines) + "\n")
    scale_table_to_textwidth(table_path)


def plot_figures(summary: Dict[str, object]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    detectors = summary["detectors"]
    names = [d["name"] for d in detectors]
    x = np.arange(len(detectors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ctrl_acc = [100 * d["control"]["accuracy"] for d in detectors]
    ctrl_acc_err = [
        [100 * (d["control"]["accuracy"] - d["control"]["accuracy_ci"][0]) for d in detectors],
        [100 * (d["control"]["accuracy_ci"][1] - d["control"]["accuracy"]) for d in detectors],
    ]
    bal = [100 * d["dcass"]["balanced_mean"] for d in detectors]
    bal_err = [
        [100 * (d["dcass"]["balanced_mean"] - d["dcass"]["balanced_ci"][0]) for d in detectors],
        [100 * (d["dcass"]["balanced_ci"][1] - d["dcass"]["balanced_mean"]) for d in detectors],
    ]
    ax1.bar(x - 0.2, ctrl_acc, 0.4, yerr=ctrl_acc_err, capsize=4, color="#c0392b", label="Positive control (1.0 bpp)")
    ax1.bar(x + 0.2, bal, 0.4, yerr=bal_err, capsize=4, color="#27ae60", label="DCASS balanced accuracy")
    ax1.axhline(50.0, color="grey", linestyle="--", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 108)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("Detection accuracy")

    ctrl_auc = [d["control"]["auc_mean"] for d in detectors]
    ctrl_auc_err = [
        [d["control"]["auc_mean"] - d["control"]["auc_ci"][0] for d in detectors],
        [d["control"]["auc_ci"][1] - d["control"]["auc_mean"] for d in detectors],
    ]
    dcass_auc = [d["dcass"]["auc_mean"] for d in detectors]
    dcass_auc_err = [
        [d["dcass"]["auc_mean"] - d["dcass"]["auc_ci"][0] for d in detectors],
        [d["dcass"]["auc_ci"][1] - d["dcass"]["auc_mean"] for d in detectors],
    ]
    ax2.bar(x - 0.2, ctrl_auc, 0.4, yerr=ctrl_auc_err, capsize=4, color="#c0392b", label="Positive control (1.0 bpp)")
    ax2.bar(x + 0.2, dcass_auc, 0.4, yerr=dcass_auc_err, capsize=4, color="#27ae60", label="DCASS carriers (measured)")
    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20)
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("ROC AUC")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"fig1_roc_curves.{ext}", dpi=200)
    plt.close(fig)

    rates = [float(r) for r in summary["protocol"]["sweep_rates"]]
    fig2, ax = plt.subplots(figsize=(8, 5))
    for det in detectors:
        accs = []
        errs = []
        for rate in rates:
            cell = det["sweep"][f"{rate:g}"]
            accs.append(100 * cell["accuracy"])
            errs.append(
                [
                    100 * (cell["accuracy"] - cell["accuracy_ci"][0]),
                    100 * (cell["accuracy_ci"][1] - cell["accuracy"]),
                ]
            )
        errs_arr = np.array(errs).T
        ax.errorbar(
            rates,
            accs,
            yerr=errs_arr,
            marker="o",
            capsize=3,
            label=det["name"],
        )
    ax.axhline(50.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel("Payload (bpp, log scale)")
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_ylim(45, 105)
    ax.legend(fontsize=9)
    ax.set_title("Payload sweep on held-out twins")
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(PLOTS_DIR / f"fig1_payload_sweep.{ext}", dpi=200)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCASS component 1 media steganalysis benchmark (schema v2)"
    )
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--fold", type=int, default=None, help="run a single fold only")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--detectors", type=str, default=",".join(DEFAULT_DETECTORS))
    parser.add_argument("--carriers", type=int, default=DEFAULT_CARRIERS)
    parser.add_argument(
        "--sweep-rates",
        type=str,
        default=",".join(str(r) for r in DEFAULT_SWEEP_RATES),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Regenerate the LaTeX table and figures from the existing "
        "component_1_summary.json without re-aggregating or re-running "
        "any measurement. The summary artifact is left untouched.",
    )
    return parser.parse_args(argv)


def gate_exit_code(summary: dict) -> int:
    """Exit status for the C1 gate run: 0 iff the positive-control gates
    passed. Before this existed the script always exited 0, so a failed
    control could sit unnoticed in a pipeline while downstream artifacts
    were regenerated from bad measurements."""
    return 0 if summary["all_gates_passed"] else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    config: Dict[str, object] = {
        "images": args.images,
        "folds": args.folds,
        "epochs": args.epochs,
        "detectors": [d.strip() for d in args.detectors.split(",") if d.strip()],
        "carriers": args.carriers,
        "sweep_rates": [float(r) for r in args.sweep_rates.split(",") if r.strip()],
        "workers": args.workers,
        "force": args.force,
        "seed": SEED,
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" DCASS COMPONENT 1: MEDIA-LAYER STEGANALYSIS BENCHMARK (schema v2)")
    print("=" * 78)
    print(f"Device: {DEVICE} | pool={config['images']} images | folds={config['folds']} "
          f"| detectors={','.join(config['detectors'])}")

    if args.table_only:
        # Regenerate the publication artifacts from the recorded summary.
        # The summary itself is read-only here: no fold is re-aggregated and
        # no measurement is re-run, so the artifact and its provenance stay
        # exactly as recorded.
        summary_path = BENCHMARK_DIR / "component_1_summary.json"
        if not summary_path.exists():
            print(f"ERROR: {summary_path} not found; run the benchmark first.")
            return None
        summary = json.loads(summary_path.read_text())
        print(f"Loaded recorded summary from {summary_path}")
        write_table(summary)
        plot_figures(summary)
        write_component1_manifest(summary)
        print(f"Artifacts: {relative(BENCHMARK_DIR / 'component_1_summary.json')}")
        return summary

    if not args.summary_only:
        fold_list = [args.fold] if args.fold is not None else list(range(config["folds"]))
        for fold in fold_list:
            if not 0 <= fold < config["folds"]:
                raise ValueError(f"fold {fold} outside 0..{config['folds'] - 1}")
            print(f"Fold {fold}")
            run_fold(config, fold)
        if args.fold is not None:
            present = all(
                (BENCHMARK_DIR / f"component_1_fold{f}.json").exists()
                for f in range(config["folds"])
            )
            if not present:
                print("single-fold mode: aggregation deferred until every fold exists")
                return 0

    summary = aggregate(config)
    print("-" * 78)
    for det in summary["detectors"]:
        c = det["control"]
        d = det["dcass"]
        print(
            f"{det['name']:>9}: PC acc {100 * c['accuracy']:.2f}% "
            f"[{100 * c['accuracy_ci'][0]:.2f},{100 * c['accuracy_ci'][1]:.2f}] | "
            f"PC AUC {c['auc_mean']:.4f} | DCASS AUC {d['auc_mean']:.4f} "
            f"[{d['auc_ci'][0]:.4f},{d['auc_ci'][1]:.4f}] | gate={'pass' if c['gate_pass_all'] else 'FAIL'}"
        )
    print(f"Artifacts: {relative(BENCHMARK_DIR / 'component_1_summary.json')}")
    return gate_exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())

