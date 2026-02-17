"""
Report generation for DCASS benchmarks.

Provides formatted CLI output and file export for benchmark results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.benchmarks.semantic_benchmark import BenchmarkResults


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def colorize(text: str, color: str) -> str:
    """Add color to text for terminal output."""
    return f"{color}{text}{Colors.END}"


def print_benchmark_report(results: "BenchmarkResults"):
    """
    Print formatted benchmark report to console.
    
    Args:
        results: BenchmarkResults object from benchmark run
    """
    width = 78
    
    # Header
    print()
    print(colorize("=" * width, Colors.CYAN))
    print(colorize(" " * 15 + "DCASS SEMANTIC RECOVERY BENCHMARK REPORT", Colors.BOLD + Colors.CYAN))
    print(colorize("=" * width, Colors.CYAN))
    
    # Metadata
    print(f"\n{colorize('Timestamp:', Colors.BOLD)} {results.timestamp}")
    print(f"{colorize('Dataset Version:', Colors.BOLD)} {results.dataset_version}")
    print(f"{colorize('Total Samples:', Colors.BOLD)} {results.total_samples}")
    print(f"{colorize('Total Time:', Colors.BOLD)} {results.total_time_seconds:.1f}s")
    
    # Overall Results
    print(f"\n{colorize('=' * width, Colors.CYAN)}")
    print(colorize(" OVERALL RESULTS", Colors.BOLD + Colors.GREEN))
    print(colorize("=" * width, Colors.CYAN))
    
    print(f"\n{'Metric':<25} {'Score':<15} {'Interpretation':<35}")
    print("-" * width)
    
    # CLIP interpretation
    clip_interp = _interpret_score(results.overall_clip_mean, "clip")
    bert_interp = _interpret_score(results.overall_bertscore_mean, "bert")
    
    clip_color = _score_color(results.overall_clip_mean)
    bert_color = _score_color(results.overall_bertscore_mean)
    
    print(f"{'CLIP Similarity':<25} {colorize(f'{results.overall_clip_mean:.3f} +/- {results.overall_clip_std:.3f}', clip_color):<25} {clip_interp:<35}")
    print(f"{'BERTScore F1':<25} {colorize(f'{results.overall_bertscore_mean:.3f} +/- {results.overall_bertscore_std:.3f}', bert_color):<25} {bert_interp:<35}")
    
    # Results by Mode
    print(f"\n{colorize('=' * width, Colors.CYAN)}")
    print(colorize(" RESULTS BY DIVERSITY MODE", Colors.BOLD + Colors.YELLOW))
    print(colorize("=" * width, Colors.CYAN))
    
    print(f"\n{'Mode':<15} {'CLIP':<18} {'BERTScore':<18} {'Enc(ms)':<10} {'Dec(ms)':<10}")
    print("-" * width)
    
    for mode, stats in results.mode_stats.items():
        clip_str = f"{stats.clip_mean:.3f} +/- {stats.clip_std:.3f}"
        bert_str = f"{stats.bertscore_mean:.3f} +/- {stats.bertscore_std:.3f}"
        
        mode_display = colorize(mode, Colors.BOLD)
        print(f"{mode:<15} {clip_str:<18} {bert_str:<18} {stats.avg_encoding_time_ms:<10.1f} {stats.avg_decoding_time_ms:<10.1f}")
    
    # Modality Distribution by Mode
    print(f"\n{colorize('Modality Distribution (%):', Colors.BOLD)}")
    print(f"{'Mode':<15} {'Image':<12} {'Text':<12} {'Audio':<12}")
    print("-" * 55)
    
    for mode, stats in results.mode_stats.items():
        dist = stats.modality_distribution
        img = f"{dist.get('image', 0):.1f}%"
        txt = f"{dist.get('text', 0):.1f}%"
        aud = f"{dist.get('audio', 0):.1f}%"
        print(f"{mode:<15} {img:<12} {txt:<12} {aud:<12}")
    
    # Results by Category
    print(f"\n{colorize('=' * width, Colors.CYAN)}")
    print(colorize(" RESULTS BY CATEGORY", Colors.BOLD + Colors.BLUE))
    print(colorize("=" * width, Colors.CYAN))
    
    print(f"\n{'Category':<15} {'N':<5} {'CLIP':<18} {'BERTScore':<18} {'Chunks':<8} {'Verified':<10}")
    print("-" * width)
    
    # Sort categories by CLIP score descending
    sorted_cats = sorted(
        results.category_stats.items(),
        key=lambda x: x[1].clip_mean,
        reverse=True
    )
    
    for cat, stats in sorted_cats:
        clip_str = f"{stats.clip_mean:.3f} +/- {stats.clip_std:.3f}"
        bert_str = f"{stats.bertscore_mean:.3f} +/- {stats.bertscore_std:.3f}"
        verified_str = f"{stats.verification_rate * 100:.0f}%"
        
        cat_color = _score_color(stats.clip_mean)
        print(f"{cat:<15} {stats.num_samples:<5} {clip_str:<18} {bert_str:<18} {stats.avg_chunks:<8.1f} {verified_str:<10}")
    
    # Best and Worst Examples
    print(f"\n{colorize('=' * width, Colors.CYAN)}")
    print(colorize(" SAMPLE RESULTS", Colors.BOLD + Colors.HEADER))
    print(colorize("=" * width, Colors.CYAN))
    
    # Top 5 by CLIP
    print(f"\n{colorize('Top 5 by CLIP Similarity:', Colors.GREEN)}")
    top_samples = sorted(results.samples, key=lambda x: x.clip_similarity, reverse=True)[:5]
    for i, s in enumerate(top_samples, 1):
        msg_preview = s.message[:35] + "..." if len(s.message) > 35 else s.message
        print(f"  {i}. [{s.diversity_mode:<11}] CLIP={s.clip_similarity:.3f} BERT={s.bertscore:.3f}")
        print(f"     \"{msg_preview}\"")
    
    # Bottom 5 by CLIP
    print(f"\n{colorize('Bottom 5 by CLIP Similarity:', Colors.RED)}")
    bottom_samples = sorted(results.samples, key=lambda x: x.clip_similarity)[:5]
    for i, s in enumerate(bottom_samples, 1):
        msg_preview = s.message[:35] + "..." if len(s.message) > 35 else s.message
        print(f"  {i}. [{s.diversity_mode:<11}] CLIP={s.clip_similarity:.3f} BERT={s.bertscore:.3f}")
        print(f"     \"{msg_preview}\"")
    
    # Summary Box
    print(f"\n{colorize('=' * width, Colors.CYAN)}")
    print(colorize(" SUMMARY", Colors.BOLD + Colors.CYAN))
    print(colorize("=" * width, Colors.CYAN))
    
    # Find best mode
    best_mode = max(results.mode_stats.items(), key=lambda x: x[1].clip_mean)
    best_cat = max(results.category_stats.items(), key=lambda x: x[1].clip_mean)
    worst_cat = min(results.category_stats.items(), key=lambda x: x[1].clip_mean)
    
    print(f"\n  - {colorize('Best performing mode:', Colors.BOLD)} {best_mode[0]} (CLIP: {best_mode[1].clip_mean:.3f})")
    print(f"  - {colorize('Best category:', Colors.BOLD)} {best_cat[0]} (CLIP: {best_cat[1].clip_mean:.3f})")
    print(f"  - {colorize('Challenging category:', Colors.BOLD)} {worst_cat[0]} (CLIP: {worst_cat[1].clip_mean:.3f})")
    
    # Overall assessment
    overall_assessment = _overall_assessment(results.overall_clip_mean, results.overall_bertscore_mean)
    print(f"\n  {colorize('Overall Assessment:', Colors.BOLD)} {overall_assessment}")
    
    print(f"\n{colorize('=' * width, Colors.CYAN)}\n")


def _interpret_score(score: float, metric_type: str) -> str:
    """Get human-readable interpretation of score."""
    if metric_type == "clip":
        if score >= 0.8:
            return "Excellent semantic alignment"
        elif score >= 0.7:
            return "Good semantic alignment"
        elif score >= 0.6:
            return "Moderate semantic alignment"
        elif score >= 0.5:
            return "Fair semantic alignment"
        else:
            return "Weak semantic alignment"
    else:  # bert
        if score >= 0.85:
            return "Excellent contextual match"
        elif score >= 0.75:
            return "Good contextual match"
        elif score >= 0.65:
            return "Moderate contextual match"
        elif score >= 0.55:
            return "Fair contextual match"
        else:
            return "Weak contextual match"


def _score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 0.75:
        return Colors.GREEN
    elif score >= 0.6:
        return Colors.YELLOW
    else:
        return Colors.RED


def _overall_assessment(clip_score: float, bert_score: float) -> str:
    """Generate overall assessment text."""
    avg = (clip_score + bert_score) / 2
    
    if avg >= 0.75:
        return colorize("The semantic recovery system shows STRONG performance. ", Colors.GREEN) + \
               "Decoded content closely matches the semantic meaning of original messages."
    elif avg >= 0.65:
        return colorize("The semantic recovery system shows GOOD performance. ", Colors.YELLOW) + \
               "Most semantic meaning is preserved in decoded content."
    elif avg >= 0.55:
        return colorize("The semantic recovery system shows MODERATE performance. ", Colors.YELLOW) + \
               "Some semantic drift observed between original and decoded content."
    else:
        return colorize("The semantic recovery system shows WEAK performance. ", Colors.RED) + \
               "Significant semantic drift between original and decoded content."


def generate_markdown_report(results: "BenchmarkResults") -> str:
    """
    Generate markdown report for documentation.
    
    Args:
        results: BenchmarkResults object
        
    Returns:
        Markdown string
    """
    lines = [
        "# DCASS Semantic Recovery Benchmark Report",
        "",
        f"**Timestamp:** {results.timestamp}",
        f"**Dataset Version:** {results.dataset_version}",
        f"**Total Samples:** {results.total_samples}",
        f"**Total Time:** {results.total_time_seconds:.1f}s",
        "",
        "## Overall Results",
        "",
        "| Metric | Score | Interpretation |",
        "|--------|-------|----------------|",
        f"| CLIP Similarity | {results.overall_clip_mean:.3f} +/- {results.overall_clip_std:.3f} | {_interpret_score(results.overall_clip_mean, 'clip')} |",
        f"| BERTScore F1 | {results.overall_bertscore_mean:.3f} +/- {results.overall_bertscore_std:.3f} | {_interpret_score(results.overall_bertscore_mean, 'bert')} |",
        "",
        "## Results by Diversity Mode",
        "",
        "| Mode | CLIP | BERTScore | Enc (ms) | Dec (ms) |",
        "|------|------|-----------|----------|----------|",
    ]
    
    for mode, stats in results.mode_stats.items():
        lines.append(
            f"| {mode} | {stats.clip_mean:.3f} +/- {stats.clip_std:.3f} | "
            f"{stats.bertscore_mean:.3f} +/- {stats.bertscore_std:.3f} | "
            f"{stats.avg_encoding_time_ms:.1f} | {stats.avg_decoding_time_ms:.1f} |"
        )
    
    lines.extend([
        "",
        "### Modality Distribution",
        "",
        "| Mode | Image | Text | Audio |",
        "|------|-------|------|-------|",
    ])
    
    for mode, stats in results.mode_stats.items():
        dist = stats.modality_distribution
        lines.append(
            f"| {mode} | {dist.get('image', 0):.1f}% | "
            f"{dist.get('text', 0):.1f}% | {dist.get('audio', 0):.1f}% |"
        )
    
    lines.extend([
        "",
        "## Results by Category",
        "",
        "| Category | N | CLIP | BERTScore | Avg Chunks | Verified |",
        "|----------|---|------|-----------|------------|----------|",
    ])
    
    sorted_cats = sorted(
        results.category_stats.items(),
        key=lambda x: x[1].clip_mean,
        reverse=True
    )
    
    for cat, stats in sorted_cats:
        lines.append(
            f"| {cat} | {stats.num_samples} | {stats.clip_mean:.3f} +/- {stats.clip_std:.3f} | "
            f"{stats.bertscore_mean:.3f} +/- {stats.bertscore_std:.3f} | "
            f"{stats.avg_chunks:.1f} | {stats.verification_rate * 100:.0f}% |"
        )
    
    return "\n".join(lines)
