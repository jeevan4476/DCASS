"""
DCASS Semantic Recovery Benchmark Runner.

This module runs comprehensive benchmarks to evaluate semantic reconstruction
accuracy across different diversity modes and message categories.

Usage:
    from src.analysis.benchmarks import SemanticBenchmark
    
    benchmark = SemanticBenchmark()
    results = benchmark.run()
    benchmark.print_report(results)
    benchmark.save_results(results, "benchmark_results.json")
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime

from src.engine.encoder import SemanticEncoder, DiversityMode
from src.engine.decoder import SemanticDecoder
from src.analysis.benchmarks.metrics import CombinedMetrics, MetricResult


@dataclass
class SampleResult:
    """Result for a single message sample."""
    message: str
    category: str
    diversity_mode: str
    media_ids: list[str]
    decoded_content: str
    num_chunks: int
    modality_breakdown: dict[str, int]
    clip_similarity: float
    bertscore: float
    bertscore_precision: float
    bertscore_recall: float
    encoding_time_ms: float
    decoding_time_ms: float
    verified: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class CategoryStats:
    """Aggregated statistics for a category."""
    category: str
    num_samples: int
    clip_mean: float
    clip_std: float
    bertscore_mean: float
    bertscore_std: float
    avg_chunks: float
    verification_rate: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModeStats:
    """Aggregated statistics for a diversity mode."""
    mode: str
    num_samples: int
    clip_mean: float
    clip_std: float
    bertscore_mean: float
    bertscore_std: float
    avg_encoding_time_ms: float
    avg_decoding_time_ms: float
    modality_distribution: dict[str, float]  # Percentage per modality
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    timestamp: str
    dataset_version: str
    total_samples: int
    total_time_seconds: float
    
    # Per-mode statistics
    mode_stats: dict[str, ModeStats]
    
    # Per-category statistics (aggregated across modes)
    category_stats: dict[str, CategoryStats]
    
    # Individual sample results
    samples: list[SampleResult]
    
    # Overall statistics
    overall_clip_mean: float
    overall_clip_std: float
    overall_bertscore_mean: float
    overall_bertscore_std: float
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dataset_version": self.dataset_version,
            "total_samples": self.total_samples,
            "total_time_seconds": self.total_time_seconds,
            "mode_stats": {k: v.to_dict() for k, v in self.mode_stats.items()},
            "category_stats": {k: v.to_dict() for k, v in self.category_stats.items()},
            "samples": [s.to_dict() for s in self.samples],
            "overall_clip_mean": self.overall_clip_mean,
            "overall_clip_std": self.overall_clip_std,
            "overall_bertscore_mean": self.overall_bertscore_mean,
            "overall_bertscore_std": self.overall_bertscore_std,
        }


class SemanticBenchmark:
    """
    Main benchmark runner for DCASS semantic recovery evaluation.
    
    Runs encode/decode cycles across multiple diversity modes and
    evaluates semantic similarity using CLIP and BERTScore metrics.
    """
    
    def __init__(
        self,
        dataset_path: Path = None,
        device: str = None,
        bertscore_model: str = "microsoft/deberta-base-mnli"
    ):
        """
        Args:
            dataset_path: Path to test_messages.json. Uses default if None.
            device: Device for models ('cuda' or 'cpu'). Auto-detected if None.
            bertscore_model: Model to use for BERTScore.
        """
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent.parent / "data" / "benchmarks" / "test_messages.json"
        
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.bertscore_model = bertscore_model
        
        # Components (lazy loaded)
        self._encoder = None
        self._decoder = None
        self._metrics = None
        self._dataset = None
    
    @property
    def encoder(self) -> SemanticEncoder:
        """Lazy load encoder."""
        if self._encoder is None:
            print("Loading encoder...")
            self._encoder = SemanticEncoder()
            self._encoder.load()
        return self._encoder
    
    @property
    def decoder(self) -> SemanticDecoder:
        """Lazy load decoder."""
        if self._decoder is None:
            print("Loading decoder...")
            self._decoder = SemanticDecoder()
            self._decoder.load()
        return self._decoder
    
    @property
    def metrics(self) -> CombinedMetrics:
        """Lazy load metrics."""
        if self._metrics is None:
            print("Loading metrics...")
            self._metrics = CombinedMetrics(
                device=self.device,
                bertscore_model=self.bertscore_model
            )
        return self._metrics
    
    @property
    def dataset(self) -> dict:
        """Load benchmark dataset."""
        if self._dataset is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self._dataset = json.load(f)
        return self._dataset
    
    def get_all_messages(self) -> list[tuple[str, str]]:
        """
        Get all messages from dataset.
        
        Returns:
            List of (category, message) tuples
        """
        messages = []
        for category, data in self.dataset["categories"].items():
            for msg in data["messages"]:
                messages.append((category, msg))
        return messages
    
    def run_single(
        self, 
        message: str, 
        category: str,
        diversity_mode: DiversityMode
    ) -> SampleResult:
        """
        Run benchmark for a single message.
        
        Args:
            message: Message to encode/decode
            category: Category name for this message
            diversity_mode: Diversity mode to use
            
        Returns:
            SampleResult with metrics
        """
        # Encode
        start_encode = time.perf_counter()
        encode_result = self.encoder.encode(message, diversity_mode=diversity_mode)
        encoding_time = (time.perf_counter() - start_encode) * 1000
        
        # Decode
        start_decode = time.perf_counter()
        decode_result = self.decoder.decode(encode_result.media_ids)
        decoding_time = (time.perf_counter() - start_decode) * 1000
        
        # Get decoded content
        decoded_content = decode_result.reconstructed_meaning
        
        # Compute metrics
        metric_results = self.metrics.evaluate(message, decoded_content)
        
        clip_result = metric_results.get("CLIP Similarity")
        bert_result = metric_results.get("BERTScore")
        
        return SampleResult(
            message=message,
            category=category,
            diversity_mode=diversity_mode,
            media_ids=encode_result.media_ids,
            decoded_content=decoded_content,
            num_chunks=len(encode_result.chunks),
            modality_breakdown=encode_result.modality_breakdown,
            clip_similarity=clip_result.score if clip_result else 0.0,
            bertscore=bert_result.score if bert_result else 0.0,
            bertscore_precision=bert_result.details.get("precision", 0.0) if bert_result else 0.0,
            bertscore_recall=bert_result.details.get("recall", 0.0) if bert_result else 0.0,
            encoding_time_ms=encoding_time,
            decoding_time_ms=decoding_time,
            verified=decode_result.all_verified,
        )
    
    def run(
        self,
        modes: list[DiversityMode] = None,
        categories: list[str] = None,
        max_samples_per_category: int = None,
        verbose: bool = True
    ) -> BenchmarkResults:
        """
        Run full benchmark across modes and categories.
        
        Args:
            modes: Diversity modes to test. Defaults to all three.
            categories: Categories to include. Defaults to all.
            max_samples_per_category: Limit samples per category (for quick tests)
            verbose: Print progress updates
            
        Returns:
            BenchmarkResults with all statistics
        """
        modes = modes or ["best", "round_robin", "balanced"]
        
        # Get messages
        all_messages = self.get_all_messages()
        
        # Filter by categories if specified
        if categories:
            all_messages = [(cat, msg) for cat, msg in all_messages if cat in categories]
        
        # Limit per category if specified
        if max_samples_per_category:
            filtered = []
            cat_counts = {}
            for cat, msg in all_messages:
                if cat_counts.get(cat, 0) < max_samples_per_category:
                    filtered.append((cat, msg))
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
            all_messages = filtered
        
        total_runs = len(all_messages) * len(modes)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f" DCASS Semantic Recovery Benchmark")
            print(f"{'='*70}")
            print(f" Messages: {len(all_messages)}")
            print(f" Modes: {modes}")
            print(f" Total runs: {total_runs}")
            print(f"{'='*70}\n")
        
        # Run benchmarks
        all_samples: list[SampleResult] = []
        start_time = time.perf_counter()
        
        for mode_idx, mode in enumerate(modes):
            if verbose:
                print(f"\n[Mode {mode_idx+1}/{len(modes)}] Running mode: {mode}")
                print("-" * 50)
            
            for i, (category, message) in enumerate(all_messages):
                if verbose:
                    progress = (mode_idx * len(all_messages) + i + 1) / total_runs * 100
                    print(f"  [{progress:5.1f}%] {category}: \"{message[:40]}{'...' if len(message) > 40 else ''}\"")
                
                try:
                    result = self.run_single(message, category, mode)
                    all_samples.append(result)
                    
                    if verbose:
                        print(f"          CLIP: {result.clip_similarity:.3f} | BERT: {result.bertscore:.3f}")
                        
                except Exception as e:
                    if verbose:
                        print(f"          ERROR: {e}")
        
        total_time = time.perf_counter() - start_time
        
        # Compute statistics
        results = self._compute_statistics(all_samples, modes, total_time)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f" Benchmark complete in {total_time:.1f}s")
            print(f"{'='*70}\n")
        
        return results
    
    def _compute_statistics(
        self, 
        samples: list[SampleResult],
        modes: list[str],
        total_time: float
    ) -> BenchmarkResults:
        """Compute aggregate statistics from samples."""
        
        # Mode statistics
        mode_stats = {}
        for mode in modes:
            mode_samples = [s for s in samples if s.diversity_mode == mode]
            
            if not mode_samples:
                continue
            
            clip_scores = [s.clip_similarity for s in mode_samples]
            bert_scores = [s.bertscore for s in mode_samples]
            encode_times = [s.encoding_time_ms for s in mode_samples]
            decode_times = [s.decoding_time_ms for s in mode_samples]
            
            # Modality distribution
            total_items = sum(sum(s.modality_breakdown.values()) for s in mode_samples)
            modality_counts = {"image": 0, "text": 0, "audio": 0}
            for s in mode_samples:
                for mod, count in s.modality_breakdown.items():
                    modality_counts[mod] = modality_counts.get(mod, 0) + count
            
            modality_dist = {
                mod: count / total_items * 100 if total_items > 0 else 0
                for mod, count in modality_counts.items()
            }
            
            mode_stats[mode] = ModeStats(
                mode=mode,
                num_samples=len(mode_samples),
                clip_mean=float(np.mean(clip_scores)),
                clip_std=float(np.std(clip_scores)),
                bertscore_mean=float(np.mean(bert_scores)),
                bertscore_std=float(np.std(bert_scores)),
                avg_encoding_time_ms=float(np.mean(encode_times)),
                avg_decoding_time_ms=float(np.mean(decode_times)),
                modality_distribution=modality_dist,
            )
        
        # Category statistics (aggregated across modes)
        categories = set(s.category for s in samples)
        category_stats = {}
        
        for cat in categories:
            cat_samples = [s for s in samples if s.category == cat]
            
            clip_scores = [s.clip_similarity for s in cat_samples]
            bert_scores = [s.bertscore for s in cat_samples]
            chunks = [s.num_chunks for s in cat_samples]
            verified = [s.verified for s in cat_samples]
            
            category_stats[cat] = CategoryStats(
                category=cat,
                num_samples=len(cat_samples),
                clip_mean=float(np.mean(clip_scores)),
                clip_std=float(np.std(clip_scores)),
                bertscore_mean=float(np.mean(bert_scores)),
                bertscore_std=float(np.std(bert_scores)),
                avg_chunks=float(np.mean(chunks)),
                verification_rate=sum(verified) / len(verified) if verified else 0.0,
            )
        
        # Overall statistics
        all_clip = [s.clip_similarity for s in samples]
        all_bert = [s.bertscore for s in samples]
        
        return BenchmarkResults(
            timestamp=datetime.now().isoformat(),
            dataset_version=self.dataset.get("version", "unknown"),
            total_samples=len(samples),
            total_time_seconds=total_time,
            mode_stats=mode_stats,
            category_stats=category_stats,
            samples=samples,
            overall_clip_mean=float(np.mean(all_clip)),
            overall_clip_std=float(np.std(all_clip)),
            overall_bertscore_mean=float(np.mean(all_bert)),
            overall_bertscore_std=float(np.std(all_bert)),
        )
    
    def save_results(self, results: BenchmarkResults, path: Path = None) -> Path:
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results to save
            path: Output path. Defaults to data/benchmarks/results/
            
        Returns:
            Path to saved file
        """
        if path is None:
            results_dir = self.dataset_path.parent / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = results_dir / f"benchmark_{timestamp}.json"
        
        path = Path(path)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        
        return path
    
    def print_report(self, results: BenchmarkResults):
        """Print formatted benchmark report to console."""
        from src.analysis.benchmarks.report import print_benchmark_report
        print_benchmark_report(results)


# Convenience function
def run_benchmark(
    modes: list[DiversityMode] = None,
    verbose: bool = True,
    save: bool = True
) -> BenchmarkResults:
    """
    Quick benchmark runner.
    
    Args:
        modes: Diversity modes to test
        verbose: Print progress
        save: Save results to file
        
    Returns:
        BenchmarkResults
    """
    benchmark = SemanticBenchmark()
    results = benchmark.run(modes=modes, verbose=verbose)
    
    if save:
        path = benchmark.save_results(results)
        print(f"\nResults saved to: {path}")
    
    benchmark.print_report(results)
    
    return results
