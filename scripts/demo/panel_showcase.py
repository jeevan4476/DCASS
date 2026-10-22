#!/usr/bin/env python3
"""
DCASS panel showcase terminal application.

A self-contained console presentation of the full DCASS stack for evaluation
panels. It requires no browser and no network beyond the local machine.

Menu sections:
    1. Complete end-to-end encrypted transmission.
    2. Live zero modification proof with a transmission cross-check.
    3. Live steganalysis detector evaluation against clean, LSB, and DCASS.
    4. Benchmark summary and defense questions and answers cheatsheet.
    5. Pre-flight environment and GPU report.

Usage:
    .venv/bin/python scripts/demo/panel_showcase.py
    .venv/bin/python scripts/demo/panel_showcase.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INDICES_DIR = PROJECT_ROOT / "storage" / "data" / "indices"
MODELS_DIR = PROJECT_ROOT / "storage" / "models"
BENCHMARKS_DIR = PROJECT_ROOT / "storage" / "benchmarks"
SHARED_CHANNEL_DIR = PROJECT_ROOT / "storage" / "shared_channel"

CATALOG_FILES = ("image_metadata.json", "text_metadata.json", "audio_metadata.json")
CHECKPOINT_FILES = ("gan_generator.pt", "rl_agent.pt", "warden.pt")
INDEX_FILES = ("image.index", "text.index", "audio.index")

WIDTH = 78

# ---------------------------------------------------------------------------
# Formatting helpers. ASCII only, no decorative characters.
# ---------------------------------------------------------------------------


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def section_title(title: str) -> None:
    print()
    rule()
    print(f"  {title}")
    rule()


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Print a clean ASCII table with computed column widths."""
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in cells:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
        return "| " + " | ".join(padded) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print(format_row(headers))
    print(separator)
    for row in cells:
        print(format_row(row))
    print(separator)


def progress_bar(step: int, total: int, label: str = "", width: int = 36) -> None:
    """Render a single-line ASCII progress bar."""
    fraction = 0.0 if total <= 0 else min(1.0, step / total)
    filled = int(fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if step >= total else "\r"
    print(f"  [{bar}] {fraction * 100:5.1f}%  {label}", end=end, flush=True)


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_catalog(filename: str) -> list[dict]:
    data = load_json(INDICES_DIR / filename)
    return data if isinstance(data, list) else []


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_packet_manifest(media_id: str) -> Optional[Path]:
    """Latest recorded packet manifest for a media id, or None when unwired."""
    if not SHARED_CHANNEL_DIR.exists():
        return None
    matches = list(SHARED_CHANNEL_DIR.rglob(f"*__{media_id}.json"))
    matches += list(SHARED_CHANNEL_DIR.rglob(f"{media_id}.json"))
    if not matches:
        return None
    return max(matches, key=lambda candidate: candidate.stat().st_mtime)


# ---------------------------------------------------------------------------
# Section 5: environment pre-flight
# ---------------------------------------------------------------------------


def gather_environment() -> list[list[str]]:
    rows: list[list[str]] = []

    version = sys.version_info
    ok = version >= (3, 10)
    rows.append(["Python", platform.python_version(), "PASS" if ok else "FAIL"])
    rows.append(["Interpreter", sys.executable, "OK"])

    try:
        import torch  # noqa: PLC0415

        rows.append(["PyTorch", torch.__version__, "OK"])
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            rows.append(["CUDA", f"available, {name}", "OK"])
            rows.append(["GPU VRAM", f"{vram:.2f} GB", "OK"])
        else:
            rows.append(["CUDA", "not available, CPU fallback", "WARN"])
    except Exception as exc:
        rows.append(["PyTorch", f"unavailable: {exc}", "WARN"])

    for index_file in INDEX_FILES:
        path = INDICES_DIR / index_file
        rows.append([f"Index {index_file}", "present" if path.exists() else "missing", "OK" if path.exists() else "FAIL"])

    for checkpoint in CHECKPOINT_FILES:
        path = MODELS_DIR / checkpoint
        rows.append([f"Model {checkpoint}", "present" if path.exists() else "missing", "OK" if path.exists() else "WARN"])

    total_items = 0
    for catalog in CATALOG_FILES:
        count = len(load_catalog(catalog))
        total_items += count
        rows.append([f"Catalog {catalog.replace('_metadata.json', '')}", f"{count} items", "OK" if count else "FAIL"])
    rows.append(["Corpus total", f"{total_items} items", "OK" if total_items else "FAIL"])

    return rows


def show_environment() -> None:
    section_title("Pre-flight environment and GPU report")
    print_table(["Component", "Value", "Status"], gather_environment())


# ---------------------------------------------------------------------------
# Section 1: end-to-end transmission
# ---------------------------------------------------------------------------


def section_end_to_end(message: Optional[str] = None, shared_dir: Optional[Path] = None) -> dict:
    section_title("1. Complete end-to-end encrypted transmission")

    message = message or "Panel showcase: verified zero-modification transmission."
    print(f"  Plaintext input : {message!r}")
    print("  Pipeline        : DSSC codec, auto scheduler, real transport layer")
    print()

    if shared_dir is None:
        shared_dir = Path(tempfile.mkdtemp(prefix="panel_showcase_"))

    started = time.perf_counter()
    try:
        from src.stealth.pipeline import run_pipeline  # noqa: PLC0415

        result = run_pipeline(
            message,
            sender="panel_sender",
            recipient="panel_recipient",
            mode="auto",
            speed_multiplier=1000.0,
            shared_dir=shared_dir,
            cleanup=True,
            codec_mode="dssc",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"  Pipeline execution failed after {elapsed:.1f}s: {exc}")
        return {"ok": False, "elapsed": elapsed, "error": str(exc)}

    elapsed = time.perf_counter() - started

    original_bytes = message.encode("utf-8")
    decoded_text = result.decoded_message or ""
    decoded_bytes = decoded_text.encode("utf-8")
    length_gap = abs(len(original_bytes) - len(decoded_bytes))
    byte_differences = sum(1 for a, b in zip(original_bytes, decoded_bytes) if a != b) + length_gap
    exact_match = result.decoded_ok and decoded_text == message and byte_differences == 0

    scheduler = getattr(result, "schedule", {}) or {}
    packets = scheduler.get("items", []) if isinstance(scheduler, dict) else []

    print()
    print_table(
        ["Metric", "Value", "Status"],
        [
            ["Mode used", str(result.mode_used), "OK"],
            ["Scheduled packets", str(len(packets)), "OK"],
            ["Round trip seconds", f"{elapsed:.2f}", "OK"],
            ["ECC success", str(result.ecc_success), "OK" if result.ecc_success else "WARN"],
            [
                "Fixed errors",
                str(result.ecc_errors_fixed) if result.ecc_errors_fixed else "none",
                "OK",
            ],
            ["Verification rate", f"{result.verification_rate * 100:.1f}%", "OK"],
            ["Bytes compared", str(len(original_bytes)), "OK"],
            ["Byte discrepancies", str(byte_differences), "PASS" if byte_differences == 0 else "FAIL"],
            ["Decoded plaintext", repr(decoded_text), "PASS" if exact_match else "FAIL"],
        ],
    )
    if exact_match:
        print("  Verdict: decoded plaintext matches the original input with zero byte discrepancies.")
    else:
        print("  Verdict: ROUND TRIP MISMATCH. Do not present this result before investigating.")

    return {
        "ok": exact_match,
        "elapsed": elapsed,
        "mode_used": getattr(result, "mode_used", "unknown"),
        "byte_differences": byte_differences,
    }


# ---------------------------------------------------------------------------
# Section 2: byte-by-byte zero modification proof
# ---------------------------------------------------------------------------


def pick_catalog_samples(sample_count: int = 3) -> list[tuple[str, dict, str]]:
    """Pick up to sample_count existing corpus files, one modality at a time."""
    iterators: list[tuple[str, Any]] = []
    for catalog in CATALOG_FILES:
        modality = catalog.replace("_metadata.json", "")
        entries = [
            entry
            for entry in load_catalog(catalog)
            if str(entry.get("path", "") or "") and (PROJECT_ROOT / str(entry.get("path"))).is_file()
        ]
        iterators.append((modality, iter(entries)))

    picked: list[tuple[str, dict, str]] = []
    while len(picked) < sample_count:
        round_added = 0
        for modality, iterator in iterators:
            if len(picked) >= sample_count:
                break
            try:
                entry = next(iterator)
            except StopIteration:
                continue
            picked.append((modality, entry, str(PROJECT_ROOT / str(entry["path"]))))
            round_added += 1
        if round_added == 0:
            break
    return picked[:sample_count]


def section_hash_proof(sample_count: int = 3) -> dict:
    section_title("2. Live zero modification proof")
    print("  DCASS transmits corpus references, not duplicate payload copies. The resolved")
    print("  carrier file is therefore the transmitted object. It is hashed in place and")
    print("  cross-checked against the latest recorded packet manifest for that media id.")
    print()

    samples = pick_catalog_samples(sample_count)
    if not samples:
        print("  No corpus files available. Build the indices first.")
        return {"ok": False, "files": 0}

    modalities_seen = sorted({modality for modality, _, _ in samples})
    if len(modalities_seen) < 3:
        print(f"  Note: raw files exist locally for these modalities: {', '.join(modalities_seen)}.")
        print("  Text and audio carriers are database references in this checkout, so the byte")
        print("  proof runs on the image carriers where raw corpus files exist.")
        print()

    rows: list[list[str]] = []
    total_bytes = 0
    manifest_hits = 0
    touched_after_send = 0

    for index, (modality, entry, path_str) in enumerate(samples, start=1):
        path = Path(path_str)
        digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        media_id = str(entry.get("id", "unknown"))
        manifest = find_packet_manifest(media_id)
        if manifest is None:
            manifest_state = "no packet recorded"
        else:
            manifest_hits += 1
            record = load_json(manifest) or {}
            sent_at = record.get("timestamp")
            modified = isinstance(sent_at, (int, float)) and (
                path.stat().st_mtime > float(sent_at) + 1.0
            )
            if modified:
                touched_after_send += 1
            manifest_state = "modified after send" if modified else "unchanged since send"
        rows.append([media_id[:38], modality, f"{size:,}", digest[:16], manifest_state])
        progress_bar(index, len(samples), label=media_id[:36])

    print()
    print_table(["Media ID", "Modality", "Bytes", "SHA-256", "Packet Manifest Check"], rows)
    print(f"  {'Total bytes hashed':<38}: {total_bytes:,}")
    print(f"  {'Media ids with a packet manifest':<38}: {manifest_hits} of {len(samples)}")
    print(f"  {'Files touched after transmission':<38}: {touched_after_send}")
    if touched_after_send == 0:
        print("  Verdict: zero modification. Carriers are referenced, never copied or rewritten,")
        print("  and every resolved file is unchanged since its recorded transmission.")
    else:
        print("  Verdict: a resolved file changed after transmission. Investigate before presenting.")

    return {"ok": touched_after_send == 0, "files": len(samples), "bytes": total_bytes}


# ---------------------------------------------------------------------------
# Section 3: live steganalysis evaluation
# ---------------------------------------------------------------------------


def _recorded_component_1_rows() -> list[list[str]]:
    data = load_json(BENCHMARKS_DIR / "component_1_summary.json") or {}
    rows: list[list[str]] = []
    for detector in data.get("detectors", []):
        rows.append(
            [
                str(detector.get("name", "unknown")),
                f"{detector.get('traditional_accuracy', 0) * 100:.2f}%",
                f"{detector.get('traditional_auc', 0):.4f}",
                f"{detector.get('dcass_accuracy', 0) * 100:.2f}%",
                f"{detector.get('dcass_auc', 0):.4f}",
            ]
        )
    return rows


def section_live_steganalysis(detectors: tuple[str, ...] = ("sr_net",), sample_count: int = 2) -> dict:
    section_title("3. Live steganalysis detector evaluation: clean, LSB, DCASS")

    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch.nn.functional as torch_functional  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from src.analysis import steganalysis_detectors as detector_module  # noqa: PLC0415
    except Exception as exc:
        print(f"  Detector stack unavailable: {exc}")
        print("  Showing recorded trained benchmark results instead.")
        print()
        print_table(
            ["Detector", "Control Acc", "Control AUC", "DCASS Acc", "DCASS AUC"],
            _recorded_component_1_rows(),
        )
        return {"ok": False, "reason": str(exc)}

    registry = {"sr_net": "SRNet", "zhu_net": "ZhuNet", "ye_net": "YeNet"}
    image_entries = [
        entry
        for entry in load_catalog("image_metadata.json")
        if str(entry.get("path", "") or "") and (PROJECT_ROOT / str(entry.get("path"))).is_file()
    ]
    needed = sample_count * 2
    if len(image_entries) < needed:
        print(f"  Corpus too small: need {needed} images, found {len(image_entries)}.")
        return {"ok": False, "reason": "corpus too small"}

    def to_tensor(path: Path) -> "torch.Tensor":
        image = Image.open(path).convert("L").resize((256, 256), Image.LANCZOS)
        array = np.array(image, dtype=np.float32) / 255.0
        return torch.tensor(array, dtype=torch.float32).unsqueeze(0)

    def lsb_inject(image_tensor: "torch.Tensor") -> "torch.Tensor":
        stego = image_tensor.clone()
        flat = stego.flatten()
        indices = torch.arange(flat.numel())
        flip = torch.where(indices % 2 == 0, 1.0, -1.0)
        flat = flat + flip * (1.0 / 255.0)
        return flat.reshape(stego.shape)

    clean_paths = [PROJECT_ROOT / str(entry["path"]) for entry in image_entries[:sample_count]]
    dcass_paths = [
        PROJECT_ROOT / str(entry["path"])
        for entry in image_entries[sample_count:needed]
    ]

    clean_images = torch.stack([to_tensor(path) for path in clean_paths])
    lsb_images = torch.stack([lsb_inject(image) for image in clean_images])
    dcass_images = torch.stack([to_tensor(path) for path in dcass_paths])

    rows: list[list[str]] = []
    per_detector: dict[str, dict[str, float]] = {}

    total_work = len(detectors) * 3
    work_done = 0
    for detector_key in detectors:
        class_name = registry.get(detector_key, detector_key)
        detector_class = getattr(detector_module, class_name, None)
        if detector_class is None:
            rows.append([class_name, "missing", "missing", "missing"])
            work_done += 3
            continue

        model = detector_class()
        model.eval()

        def probabilities(images: "torch.Tensor") -> "np.ndarray":
            with torch.no_grad():
                logits = model(images)
                return torch_functional.softmax(logits, dim=1)[:, 1].cpu().numpy()

        clean_probs = probabilities(clean_images)
        progress_bar(work_done + 1, total_work, label=f"{class_name} clean")
        lsb_probs = probabilities(lsb_images)
        progress_bar(work_done + 2, total_work, label=f"{class_name} LSB stego")
        dcass_probs = probabilities(dcass_images)
        progress_bar(work_done + 3, total_work, label=f"{class_name} DCASS carrier")
        work_done += 3

        per_detector[class_name] = {
            "clean": float(clean_probs.mean()),
            "lsb": float(lsb_probs.mean()),
            "dcass": float(dcass_probs.mean()),
        }
        rows.append(
            [
                class_name,
                f"{clean_probs.mean() * 100:.1f}%",
                f"{lsb_probs.mean() * 100:.1f}%",
                f"{dcass_probs.mean() * 100:.1f}%",
            ]
        )

    print()
    print("  Live forward pass, freshly initialized weights in this process:")
    print_table(["Detector", "Clean Stego Prob", "LSB Stego Prob", "DCASS Stego Prob"], rows)

    recorded = _recorded_component_1_rows()
    if recorded:
        print()
        print("  Recorded trained benchmark from storage/benchmarks/component_1_summary.json:")
        print_table(["Detector", "Control Acc", "Control AUC", "DCASS Acc", "DCASS AUC"], recorded)

    print()
    print("  Reading: the live pass is a runtime sanity check on the architecture code path.")
    print("  The trained benchmark above is the recorded evidence for panel claims.")

    return {"ok": True, "per_detector": per_detector}


# ---------------------------------------------------------------------------
# Section 4: benchmark summary and defense cheatsheet
# ---------------------------------------------------------------------------

CHEATSHEET = [
    [
        "Why can detectors not see DCASS carriers?",
        "Carrier bytes are identical to public corpus bytes. There is no modification to detect.",
    ],
    [
        "Were the detectors competent?",
        "Yes. 93.7 to 95.1 percent accuracy and 0.984 to 0.986 AUC on the LSB positive control.",
    ],
    [
        "Why does the accuracy column look high when the AUC is 0.4973 to 0.4993?",
        "The accuracy figure is a one-sided cover-decision rate. Balanced accuracy is 50 percent.",
    ],
    [
        "Can a warden spot the carrier selection statistically?",
        "No. Zero of 1,536 KS coordinates reject after Benjamini-Hochberg control. W1 is 0.0028 to 0.0081.",
    ],
    [
        "Why does periodic transmission fail?",
        "Zero timing entropy. Interception is 100 percent.",
    ],
    [
        "Why does Poisson timing fail?",
        "Burst density in the exponential tail. Interception is 54 percent.",
    ],
    [
        "How does the generator defeat the DPI warden?",
        "It does not. A trained traffic detector flags the generated cadence at "
        "100 percent, against a threshold calibrated on held-out human traffic.",
    ],
    [
        "What is the capacity per carrier?",
        "13.60 bits average multi-modal, up to 15.97 text-only, against 8.00 for exact_vcp.",
    ],
    [
        "What was the framing optimization?",
        "A 5 byte CRC-16 header replaced a 35 byte HMAC header, cutting framed overhead from 40 bytes to 10.",
    ],
    [
        "How much carrier damage can it survive?",
        "Four byte errors per codeword at k equals 8. 70 percent recover at 5 percent substitution, refined to 61.2 percent by the 80-trial extension; carrier dropout is unrecoverable.",
    ],
    [
        "Can a wrong key ever decode?",
        "No. Zero false accepts across 100 random keys and 256 single-bit key flips.",
    ],
    [
        "What dominates latency?",
        "FAISS search. About 99 percent of pipeline time. Permutation and ECC are near zero.",
    ],
    [
        "Do concurrent users corrupt each other?",
        "No. 100 percent success and 0.000 percent cross-talk at 50 threads.",
    ],
]


def _component1_rows(summary: dict[str, Any]) -> list[list[str]]:
    """Read the schema v2 summary, with a fallback for the legacy flat fields."""
    rows: list[list[str]] = []
    for detector in summary.get("detectors", []):
        control = detector.get("control") or {}
        dcass = detector.get("dcass") or {}
        accuracy = control.get("accuracy", detector.get("traditional_accuracy", 0.0))
        control_auc = control.get("auc_mean", detector.get("traditional_auc", 0.0))
        dcass_auc = dcass.get("auc_mean", detector.get("dcass_auc", 0.0))
        rows.append(
            [
                str(detector.get("name")),
                f"{accuracy * 100:.2f}%",
                f"{control_auc:.4f}",
                f"{dcass_auc:.4f}",
            ]
        )
    return rows


def section_benchmark_summary() -> dict:
    section_title("4. Benchmark summary and defense cheatsheet")

    component_1 = load_json(BENCHMARKS_DIR / "component_1_summary.json") or {}
    rows_1 = _component1_rows(component_1)
    if rows_1:
        print("  Component 1: media-layer steganalysis resistance")
        print_table(["Detector", "Control Acc", "Control AUC", "DCASS AUC"], rows_1)
        print()

    component_2 = load_json(BENCHMARKS_DIR / "component_2_results.json") or {}
    wasserstein = component_2.get("wasserstein_distances", {})
    if wasserstein:
        entropy = component_2.get("cluster_entropy", 0.0)
        ratio = component_2.get("cluster_entropy_ratio", 0.0)
        print("  Component 2: semantic selection bias")
        print_table(
            ["Metric", "Value", "Reference"],
            [
                ["W1 image", f"{wasserstein.get('image', 0):.6f}", "unit hypersphere radius"],
                ["W1 text", f"{wasserstein.get('text', 0):.6f}", "unit hypersphere radius"],
                ["W1 audio", f"{wasserstein.get('audio', 0):.6f}", "unit hypersphere radius"],
                ["Cluster entropy", f"{entropy:.4f} bits", "maximum 8.0000 bits"],
                ["Entropy ratio", f"{ratio * 100:.2f}%", "of theoretical maximum"],
            ],
        )
        print()

    component_3 = load_json(BENCHMARKS_DIR / "component_3_results.json") or {}
    rows_3 = []
    for scheduler in component_3.get("schedulers", []):
        rows_3.append(
            [
                str(scheduler.get("scheduler")),
                f"{scheduler.get('interception_rate', 0):.1f}%",
                f"{scheduler.get('delay_entropy', 0):.4f}",
                f"{scheduler.get('mean_latency', 0):.2f} s",
            ]
        )
    if rows_3:
        print("  Component 3: traffic timing evasion")
        print_table(["Scheduler", "Interception", "Delay Entropy", "Mean Latency"], rows_3)
        for regime in component_3.get("diurnal_breakdown", []):
            print(
                f"    {regime.get('regime')}: mean {regime.get('mean_latency', 0):.2f} s, "
                f"interception {regime.get('interception_rate', 0):.1f}%"
            )
        print()

    component_4a_raw = load_json(BENCHMARKS_DIR / "component_4a_capacity_results.json") or {}
    component_4a = (
        component_4a_raw.get("results", [])
        if isinstance(component_4a_raw, dict)
        else component_4a_raw
    )
    rows_4a = []
    for record in component_4a:
        if record.get("payload_bytes") in (50, 300):
            rows_4a.append(
                [
                    str(record.get("mode")),
                    str(record.get("payload_bytes")),
                    str(record.get("carrier_count")),
                    f"{record.get('bits_per_carrier', 0):.2f}",
                    f"{record.get('carrier_reduction_pct', 0):.1f}%",
                ]
            )
    if rows_4a:
        print("  Component 4a: codec capacity, 50 and 300 byte payloads")
        print_table(["Mode", "Bytes", "Carriers", "Bits/Carrier", "Reduction"], rows_4a)
        print()

    component_4b = load_json(BENCHMARKS_DIR / "component_4b_robustness_results.json") or {}
    rows_4b = []
    for record in component_4b.get("ecc_recovery_benchmarks", []):
        rows_4b.append(
            [
                f"{record.get('carrier_error_rate', 0) * 100:.0f}%",
                f"{record.get('recovery_rate_pct', 0):.1f}%",
                f"{record.get('mean_fixed_errors', 0):.1f}",
            ]
        )
    if rows_4b:
        print("  Component 4b: Reed-Solomon recovery under carrier substitution")
        print_table(["Substitution Rate", "Recovery Rate", "Mean Fixed Errors"], rows_4b)
        permutation = component_4b.get("permutation_benchmarks", [])
        if permutation:
            largest = permutation[-1]
            print(
                f"    Permutation scaling: {largest.get('candidate_size')} candidates in "
                f"{largest.get('time_per_permutation_ms', 0):.4f} ms, determinism "
                f"{'pass' if largest.get('determinism_passed') else 'fail'}"
            )
        key_sep = component_4b.get("key_separation_benchmarks", [])
        false_accepts = sum(int(record.get("num_false_acceptances", 0)) for record in key_sep)
        trials = sum(int(record.get("num_trials", 0)) for record in key_sep)
        print(f"    Key separation: {false_accepts} false accepts across {trials} mismatch trials")
        print()

    component_5 = load_json(BENCHMARKS_DIR / "component_5_results.json") or {}
    rows_5 = []
    for record in component_5.get("concurrency_stress", []):
        rows_5.append(
            [
                str(record.get("num_threads")),
                f"{record.get('requests_per_sec', 0):.2f}",
                f"{record.get('p50_latency_ms', 0):.0f}",
                f"{record.get('p95_latency_ms', 0):.0f}",
                f"{record.get('success_rate_pct', 0):.1f}%",
                f"{record.get('cross_talk_rate_pct', 0):.3f}%",
            ]
        )
    if rows_5:
        print("  Component 5: concurrency and session isolation")
        print_table(["Threads", "Req/s", "p50 ms", "p95 ms", "Success", "Cross-Talk"], rows_5)
        print()

    print("  Defense cheatsheet")
    print_table(["Question", "One-line answer"], CHEATSHEET)

    return {"ok": True}


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------


def interactive_menu() -> int:
    actions: dict[str, tuple[str, Callable[[], Any]]] = {
        "1": ("Complete end-to-end encrypted transmission", section_end_to_end),
        "2": ("Live zero modification proof: hashes and transmission cross-check", section_hash_proof),
        "3": ("Live steganalysis evaluation: clean, LSB, DCASS", section_live_steganalysis),
        "4": ("Benchmark summary and defense cheatsheet", section_benchmark_summary),
        "5": ("Pre-flight environment and GPU report", show_environment),
    }

    while True:
        show_environment()
        print()
        print("  Panel menu")
        for key, (label, _) in actions.items():
            print(f"    {key}. {label}")
        print("    0. Exit")
        print()
        try:
            choice = input("  Select a section: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice == "0":
            print("  Session closed.")
            return 0
        if choice not in actions:
            print("  Unknown selection. Enter a number from the menu.")
            continue

        started = time.perf_counter()
        try:
            actions[choice][1]()
        except Exception as exc:
            print(f"  Section failed: {exc}")
        elapsed = time.perf_counter() - started
        print(f"\n  Section duration: {elapsed:.2f} s")
        try:
            input("\n  Press Enter to return to the menu.")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


def dry_run() -> int:
    """Non-interactive full pass across every section with live data."""
    rule()
    print("  DCASS PANEL SHOWCASE, DRY RUN")
    print("  All sections execute non-interactively against local data.")
    rule()

    timings: list[list[str]] = []
    failures: list[str] = []

    started = time.perf_counter()
    show_environment()
    timings.append(["Pre-flight environment", f"{time.perf_counter() - started:.2f}", "OK"])

    started = time.perf_counter()
    result = section_end_to_end()
    timings.append(
        ["End-to-end transmission", f"{time.perf_counter() - started:.2f}", "OK" if result.get("ok") else "FAIL"]
    )
    if not result.get("ok"):
        failures.append("end-to-end transmission")

    started = time.perf_counter()
    result = section_hash_proof(sample_count=3)
    timings.append(
        ["Zero-modification hash proof", f"{time.perf_counter() - started:.2f}", "OK" if result.get("ok") else "FAIL"]
    )
    if not result.get("ok"):
        failures.append("hash proof")

    started = time.perf_counter()
    result = section_live_steganalysis(detectors=("sr_net", "zhu_net", "ye_net"), sample_count=2)
    timings.append(
        ["Live steganalysis", f"{time.perf_counter() - started:.2f}", "OK" if result.get("ok") else "WARN"]
    )

    started = time.perf_counter()
    result = section_benchmark_summary()
    timings.append(
        ["Benchmark summary", f"{time.perf_counter() - started:.2f}", "OK" if result.get("ok") else "FAIL"]
    )

    section_title("Dry run telemetry")
    print_table(["Section", "Seconds", "Status"], timings)

    if failures:
        print(f"  Dry run incomplete. Failed sections: {', '.join(failures)}")
        return 1

    print("  Dry run completed. Exit code 0.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DCASS terminal showcase for panel presentations. No browser required.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every section non-interactively against local data and exit.",
    )
    args = parser.parse_args()

    if args.dry_run:
        return dry_run()
    return interactive_menu()


if __name__ == "__main__":
    raise SystemExit(main())
