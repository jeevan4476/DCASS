# src/analysis/benchmarks/__init__.py
"""
DCASS Benchmarking Package.

Provides tools for evaluating semantic recovery accuracy.

Usage:
    from src.analysis.benchmarks import SemanticBenchmark, run_benchmark
    
    # Quick run
    results = run_benchmark()
    
    # Custom run
    benchmark = SemanticBenchmark()
    results = benchmark.run(modes=["balanced"], verbose=True)
    benchmark.print_report(results)
    benchmark.save_results(results)
"""

from src.analysis.benchmarks.metrics import (
    CLIPSimilarity,
    BERTScoreMetric,
    SentenceTransformerSimilarity,
    CombinedMetrics,
    MetricResult,
    compute_semantic_similarity,
)

from src.analysis.benchmarks.semantic_benchmark import (
    SemanticBenchmark,
    BenchmarkResults,
    SampleResult,
    CategoryStats,
    ModeStats,
    run_benchmark,
)

from src.analysis.benchmarks.report import (
    print_benchmark_report,
    generate_markdown_report,
)

__all__ = [
    # Metrics
    "CLIPSimilarity",
    "BERTScoreMetric",
    "SentenceTransformerSimilarity",
    "CombinedMetrics",
    "MetricResult",
    "compute_semantic_similarity",
    # Benchmark
    "SemanticBenchmark",
    "BenchmarkResults",
    "SampleResult",
    "CategoryStats",
    "ModeStats",
    "run_benchmark",
    # Report
    "print_benchmark_report",
    "generate_markdown_report",
]
