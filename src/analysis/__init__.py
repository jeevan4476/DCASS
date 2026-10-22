"""
DCASS Analysis Package.

Provides tools for benchmarking and evaluating the DCASS steganography system.
"""

__all__ = [
    "SemanticBenchmark",
    "run_benchmark",
    "BenchmarkResults",
]


def __getattr__(name: str):
    """
    Lazily import `src.analysis.benchmarks` attributes (PEP 562).

    `src.analysis.adversarial.warden` (the GAN/RL stealth pipeline's
    discriminator) has no dependency on benchmarks — but benchmarks pulls in
    `src.engine.encoder` -> `src.corpus.index.unified_index` -> `clip`,
    which is a large optional dependency the stealth pipeline does not need.
    Eagerly importing benchmarks here meant `import
    src.analysis.adversarial.warden` (and everything under src/stealth/ that
    imports it) ALSO required `clip` to be installed, purely because both
    submodules share this package's __init__. Deferring the import until
    one of these names is actually accessed keeps the two independent.
    """
    if name in __all__:
        from src.analysis.benchmarks import (
            SemanticBenchmark,
            run_benchmark,
            BenchmarkResults,
        )

        globals().update(
            SemanticBenchmark=SemanticBenchmark,
            run_benchmark=run_benchmark,
            BenchmarkResults=BenchmarkResults,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
