# src/cli/main.py
"""
DCASS CLI - Dynamic Context-Aware Semantic Steganography

A modern, user-friendly CLI for encoding and decoding steganographic messages
using semantic similarity matching across multiple modalities.

Commands:
  encode      Encode a message into media sequence
  decode      Decode media IDs back to semantic meaning
  demo        Full encode -> decode demonstration
  status      Show system status and index information
  search      Search the corpus for a query
  verify      Verify media IDs exist in corpus
  distribute  Encode and distribute with timing profiles
  benchmark   Run semantic recovery benchmark
"""

from __future__ import annotations

import sys
import argparse
import json
from typing import Literal, cast
from pathlib import Path

# -------- Engine --------
from src.engine.encoder import SemanticEncoder, DiversityMode
from src.engine.decoder import SemanticDecoder
from src.engine.chunker import SemanticChunker

# -------- Distribution Layer --------
from src.distribution.channel_registry import get_available_channels
from src.distribution.dispatcher import Dispatcher
from src.distribution.scheduler import Scheduler
from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES

# -------- Analysis --------
# Imported lazily to avoid loading heavy dependencies on startup


# =========================================================
# ANSI Colors (for terminals that support it)
# =========================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    @classmethod
    def disable(cls):
        """Disable all colors."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


# Check if colors should be disabled
if not sys.stdout.isatty() or "--no-color" in sys.argv:
    Colors.disable()


def colored(text: str, color: str) -> str:
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.RESET}"


def bold(text: str) -> str:
    """Make text bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def dim(text: str) -> str:
    """Make text dim."""
    return f"{Colors.DIM}{text}{Colors.RESET}"


# Modality colors
MODALITY_COLORS = {
    "image": Colors.BRIGHT_BLUE,
    "text": Colors.BRIGHT_GREEN,
    "audio": Colors.BRIGHT_MAGENTA,
}


def modality_badge(modality: str) -> str:
    """Create a colored badge for modality."""
    color = MODALITY_COLORS.get(modality, Colors.WHITE)
    return f"{color}[{modality}]{Colors.RESET}"


# =========================================================
# SHARED STATE (lazy loaded)
# =========================================================

_encoder: SemanticEncoder | None = None
_decoder: SemanticDecoder | None = None


def _get_encoder(quiet: bool = False) -> SemanticEncoder:
    """Get or create the shared encoder instance."""
    global _encoder
    if _encoder is None:
        if not quiet:
            print(dim("Loading encoder..."))
        _encoder = SemanticEncoder(expand_synonyms=True)
        _encoder.load()
    return _encoder


def _get_decoder(quiet: bool = False) -> SemanticDecoder:
    """Get or create the shared decoder instance."""
    global _decoder
    if _decoder is None:
        if not quiet:
            print(dim("Loading decoder..."))
        _decoder = SemanticDecoder()
        _decoder.load()
    return _decoder


# =========================================================
# OUTPUT HELPERS
# =========================================================


def print_header(title: str, char: str = "="):
    """Print a styled header."""
    width = 60
    print()
    print(colored(char * width, Colors.CYAN))
    print(colored(f" {title}", Colors.BOLD))
    print(colored(char * width, Colors.CYAN))


def print_section(title: str):
    """Print a section header."""
    print(
        f"\n{colored('[', Colors.DIM)}{colored(title, Colors.YELLOW)}{colored(']', Colors.DIM)}"
    )


def print_kv(key: str, value: str, indent: int = 2):
    """Print a key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{dim(key + ':')} {value}")


def print_success(message: str):
    """Print a success message."""
    print(f"{colored('OK', Colors.GREEN)} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{colored('WARN', Colors.YELLOW)} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"{colored('ERR', Colors.RED)} {message}")


# =========================================================
# CLI COMMANDS
# =========================================================


def cmd_encode(args):
    """Encode a message into a media sequence."""
    print_header("DCASS ENCODER")

    print_kv("Message", f'"{args.message}"')
    print_kv("Mode", args.mode)
    if args.modality != "all":
        print_kv("Modality", args.modality)

    encoder = _get_encoder()

    # Determine modalities
    if args.modality == "all":
        modalities = ["image", "text", "audio"]
    else:
        modalities = [args.modality]

    try:
        result = encoder.encode(
            args.message,
            modalities=modalities,  # type: ignore
            diversity_mode=cast(DiversityMode, args.mode),
        )
    except ValueError as e:
        print_error(str(e))
        return 1
    except RuntimeError as e:
        print_error(str(e))
        return 1

    # Show chunks
    print_section(f"CHUNKS ({len(result.chunks)})")
    for i, chunk in enumerate(result.chunks, 1):
        quote = '"'
        print(f"  {i}. {dim(quote)}{chunk.original}{dim(quote)}")

    # Show encoded media
    print_section(f"ENCODED MEDIA ({len(result.encoded)})")
    for i, enc in enumerate(result.encoded, 1):
        badge = modality_badge(enc.media.modality)
        score = f"{enc.media.normalized_score:.3f}"
        score_color = (
            Colors.GREEN if enc.media.normalized_score > 0.5 else Colors.YELLOW
        )

        print(f"  {i}. {badge} {colored(enc.media.id, Colors.CYAN)}")
        print(f"     Score: {colored(score, score_color)}")

        content = enc.media.content
        if len(content) > 55:
            content = content[:55] + "..."
        print(f"     {dim(content)}")

    # Summary
    print_section("SUMMARY")
    breakdown = result.modality_breakdown
    parts = [f"{k}: {v}" for k, v in breakdown.items()]
    print_kv("Modalities", ", ".join(parts))

    # Output media IDs
    print_section("OUTPUT")
    ids_str = ",".join(result.media_ids)
    print(f"  {ids_str}")

    # Copy-friendly format
    if args.json:
        print_section("JSON OUTPUT")
        output = {
            "message": args.message,
            "media_ids": result.media_ids,
            "modalities": breakdown,
            "chunks": [c.original for c in result.chunks],
        }
        print(json.dumps(output, indent=2))

    # Optional: immediately distribute using timing profile
    if getattr(args, "profile", None):
        rc = _run_distribution_from_media_ids(
            media_ids=result.media_ids,
            profile_name=args.profile,
            seed=getattr(args, "seed", 42),
            policy=getattr(args, "policy", "round_robin"),
            base_delay=getattr(args, "base_delay", 3),
        )
        if rc != 0:
            return rc

    return 0


def _run_distribution_from_media_ids(
    media_ids: list[str],
    profile_name: str,
    seed: int = 42,
    policy: str = "round_robin",
    base_delay: int = 3,
) -> int:
    """Distribute an already-encoded media-id sequence with noise + scheduling."""
    print_section("DISTRIBUTION")

    profile = ACTIVITY_PROFILES.get(profile_name)
    if profile is None:
        print_error(f"Unknown profile: {profile_name}")
        print(f"Available: {list(ACTIVITY_PROFILES.keys())}")
        return 1

    # Apply noise
    print(dim(f"Profile: {profile_name} | Seed: {seed} | Policy: {policy}"))
    noise = NoiseController(seed=seed, **profile)
    items, delays = noise.apply(media_ids, base_delays=[base_delay] * len(media_ids))

    if not items:
        print_warning("All items skipped by noise model")
        return 0

    print_kv("After noise", f"{len(items)} items")
    print_kv("Delays", str(delays))

    dispatcher = Dispatcher(
        channels=get_available_channels(),
        policy=policy,
    )

    scheduler = Scheduler(
        dispatcher=dispatcher,
        delays=delays,
    )

    print(dim("Executing scheduled distribution..."))
    scheduler.run(items)
    print_success("Distribution complete")
    return 0


def cmd_decode(args):
    """Decode media IDs back to semantic meaning."""
    print_header("DCASS DECODER")

    # Parse media IDs
    media_ids = [id.strip() for id in args.ids.split(",") if id.strip()]

    if not media_ids:
        print_error("No valid media IDs provided")
        return 1

    print_kv("Media IDs", str(len(media_ids)))

    decoder = _get_decoder()
    result = decoder.decode(media_ids)

    # Show decoded items
    print_section(f"DECODED ITEMS ({len(result.decoded)})")
    for i, item in enumerate(result.decoded, 1):
        badge = modality_badge(item.modality)

        if item.verified:
            status = colored("OK", Colors.GREEN)
        else:
            status = colored("UNVERIFIED", Colors.RED)

        print(f"  {i}. {status} {badge} {colored(item.media_id, Colors.CYAN)}")

        content = item.content
        if len(content) > 60:
            content = content[:60] + "..."
        print(f"     {dim(content)}")

    # Verification summary
    print_section("VERIFICATION")
    rate = result.verification_rate
    rate_str = f"{rate * 100:.1f}%"
    rate_color = (
        Colors.GREEN if rate == 1.0 else (Colors.YELLOW if rate > 0.5 else Colors.RED)
    )
    print_kv("Rate", colored(rate_str, rate_color))

    if result.all_verified:
        print_success("All items verified in corpus")
    else:
        unverified = [d.media_id for d in result.decoded if not d.verified]
        print_warning(f"Unverified IDs: {unverified}")

    # Reconstructed meaning
    print_section("RECONSTRUCTED MEANING")
    print(f"  {result.reconstructed_meaning}")

    if args.json:
        print_section("JSON OUTPUT")
        output = {
            "media_ids": media_ids,
            "verified": result.all_verified,
            "verification_rate": rate,
            "contents": result.contents,
            "reconstructed": result.reconstructed_meaning,
        }
        print(json.dumps(output, indent=2))

    return 0


def cmd_demo(args):
    """Full encode -> decode demonstration."""
    print_header("DCASS DEMO: ENCODE -> DECODE")

    print_kv("Message", f'"{args.message}"')
    print_kv("Mode", args.mode)

    # Step 1: Encode
    print_section("STEP 1: ENCODING")
    encoder = _get_encoder()

    try:
        encode_result = encoder.encode(
            args.message, diversity_mode=cast(DiversityMode, args.mode)
        )
    except (ValueError, RuntimeError) as e:
        print_error(str(e))
        return 1

    print_kv("Chunks", str(len(encode_result.chunks)))
    print_kv("Media items", str(len(encode_result.encoded)))

    breakdown = encode_result.modality_breakdown
    parts = [f"{k}={v}" for k, v in breakdown.items()]
    print_kv("Modalities", " ".join(parts))

    for i, enc in enumerate(encode_result.encoded, 1):
        badge = modality_badge(enc.media.modality)
        print(f"    {i}. {badge} {enc.media.id}")

    # Step 2: Transmission simulation
    print_section("STEP 2: TRANSMISSION")
    transmitted_ids = encode_result.media_ids
    print_kv("Transmitting", f"{len(transmitted_ids)} media item(s)")
    print(f"    IDs: {dim(','.join(transmitted_ids))}")

    # Step 3: Decode
    print_section("STEP 3: DECODING")
    decoder = _get_decoder()
    decode_result = decoder.decode(transmitted_ids)

    rate = decode_result.verification_rate
    rate_str = f"{rate * 100:.1f}%"
    rate_color = Colors.GREEN if rate == 1.0 else Colors.YELLOW
    print_kv("Verified", colored(rate_str, rate_color))

    # Step 4: Compare
    print_section("STEP 4: COMPARISON")
    print_kv("Original", f'"{args.message}"')
    print_kv("Decoded", f'"{decode_result.reconstructed_meaning}"')

    # Final status
    print()
    if decode_result.all_verified:
        print_success("Demo complete - all items verified")
    else:
        print_warning("Demo complete - some items unverified")

    return 0


def cmd_status(args):
    """Show system status and index information."""
    print_header("DCASS STATUS")

    # Check indices
    print_section("INDICES")

    from src.corpus.index.unified_index import resolve_indices_base_path

    indices_path = resolve_indices_base_path()

    index_files = {
        "image": ("image.index", "image_metadata.json"),
        "text": ("text.index", "text_metadata.json"),
        "audio": ("audio.index", "audio_metadata.json"),
    }

    total_items = 0
    for modality, (index_file, meta_file) in index_files.items():
        index_path = indices_path / index_file
        meta_path = indices_path / meta_file

        if index_path.exists() and meta_path.exists():
            try:
                import faiss

                idx = faiss.read_index(str(index_path))
                count = idx.ntotal
                total_items += count
                status = colored("OK", Colors.GREEN)
                info = f"{count:,} items"
            except Exception as e:
                status = colored("ERR", Colors.RED)
                info = str(e)
        else:
            status = colored("MISSING", Colors.RED)
            info = "Index not found"

        badge = modality_badge(modality)
        print(f"  {badge} {status} - {info}")

    print_kv("Total items", f"{total_items:,}", indent=0)

    # System info
    print_section("SYSTEM")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_color = Colors.GREEN if device == "cuda" else Colors.YELLOW
    print_kv("Device", colored(device, device_color))
    print_kv("Indices path", str(indices_path))

    return 0


def cmd_search(args):
    """Search the corpus for a query."""
    print_header("DCASS SEARCH")

    print_kv("Query", f'"{args.query}"')
    print_kv("Results", str(args.k))
    if args.modality != "all":
        print_kv("Modality", args.modality)

    encoder = _get_encoder()

    # Determine modalities
    if args.modality == "all":
        modalities = ["image", "text", "audio"]
    else:
        modalities = [args.modality]

    results = encoder.index.search(
        args.query,
        k=args.k,
        modalities=modalities,  # type: ignore
    )

    print_section(f"RESULTS ({len(results)})")

    for i, item in enumerate(results, 1):
        badge = modality_badge(item.modality)
        score = f"{item.normalized_score:.3f}"
        score_color = Colors.GREEN if item.normalized_score > 0.5 else Colors.YELLOW

        print(
            f"  {i}. {badge} {colored(item.id, Colors.CYAN)} ({colored(score, score_color)})"
        )

        content = item.content
        if len(content) > 60:
            content = content[:60] + "..."
        print(f"     {dim(content)}")

    return 0


def cmd_verify(args):
    """Verify media IDs exist in corpus."""
    print_header("DCASS VERIFY")

    # Parse media IDs
    media_ids = [id.strip() for id in args.ids.split(",") if id.strip()]

    if not media_ids:
        print_error("No valid media IDs provided")
        return 1

    print_kv("IDs to verify", str(len(media_ids)))

    decoder = _get_decoder()
    all_verified, rate = decoder.verify_sequence(media_ids)

    print_section("RESULTS")

    for media_id in media_ids:
        item = decoder.index.get_by_id(media_id)
        if item:
            badge = modality_badge(item.modality)
            print(f"  {colored('OK', Colors.GREEN)} {badge} {media_id}")
        else:
            print(f"  {colored('MISSING', Colors.RED)} {media_id}")

    print_section("SUMMARY")
    rate_str = f"{rate * 100:.1f}%"
    rate_color = Colors.GREEN if rate == 1.0 else Colors.RED
    print_kv("Verification rate", colored(rate_str, rate_color))

    if all_verified:
        print_success("All IDs verified")
        return 0
    else:
        print_warning("Some IDs not found in corpus")
        return 1


def cmd_benchmark(args):
    """Run semantic recovery benchmark."""
    from src.analysis.benchmarks import SemanticBenchmark
    from src.analysis.benchmarks.report import generate_markdown_report

    print_header("DCASS SEMANTIC RECOVERY BENCHMARK")

    # Parse modes
    if args.modes == "all":
        modes = ["best", "round_robin", "balanced"]
    else:
        modes = [m.strip() for m in args.modes.split(",")]

    print_kv("Modes", ", ".join(modes))
    print_kv("Quick mode", str(args.quick))

    # Initialize benchmark
    benchmark = SemanticBenchmark()

    # Run benchmark
    print_section("RUNNING BENCHMARK")

    if args.quick:
        # Quick mode: 3 samples per category
        results = benchmark.run(
            modes=modes,  # type: ignore
            max_samples_per_category=3,
            verbose=not args.quiet,
        )
    else:
        results = benchmark.run(
            modes=modes,  # type: ignore
            verbose=not args.quiet,
        )

    # Print report
    if not args.quiet:
        benchmark.print_report(results)

    # Save results
    output_path = Path(args.output) if args.output else None
    saved_path = benchmark.save_results(results, output_path)  # type: ignore
    print_success(f"Results saved to: {saved_path}")

    # Save markdown if requested
    if args.markdown:
        md_content = generate_markdown_report(results)
        md_path = saved_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print_success(f"Markdown report saved to: {md_path}")

    return 0


def cmd_distribute(args):
    """Encode and distribute with timing profiles."""
    print_header("DCASS DISTRIBUTION")

    print_kv("Message", f'"{args.message}"')
    print_kv("Profile", args.profile)
    print_kv("Policy", args.policy)
    print_kv("Seed", str(args.seed))

    # Encode
    print_section("ENCODING")
    encoder = _get_encoder()

    try:
        encode_result = encoder.encode(args.message)
    except (ValueError, RuntimeError) as e:
        print_error(str(e))
        return 1

    media_ids = encode_result.media_ids
    print_kv("Media items", str(len(media_ids)))

    for i, media_id in enumerate(media_ids, 1):
        print(f"    {i}. {media_id}")

    return _run_distribution_from_media_ids(
        media_ids=media_ids,
        profile_name=args.profile,
        seed=args.seed,
        policy=args.policy,
        base_delay=args.base_delay,
    )


# =========================================================
# ARGUMENT PARSER
# =========================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="dcass",
        description="Dynamic Context-Aware Semantic Steganography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s encode "Meet me at the cafe" --mode round_robin
  %(prog)s decode "flickr8k_00123,wiki_00456"
  %(prog)s demo "The secret message" --mode balanced
  %(prog)s search "dog running on beach" --k 10
  %(prog)s status
        """,
    )

    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Encode command
    encode_parser = subparsers.add_parser(
        "encode", help="Encode a message into media sequence"
    )
    encode_parser.add_argument("message", help="Message to encode")
    encode_parser.add_argument(
        "--mode",
        "-m",
        choices=["best", "round_robin", "balanced"],
        default="best",
        help="Diversity mode for modality selection (default: best)",
    )
    encode_parser.add_argument(
        "--modality",
        choices=["all", "image", "text", "audio"],
        default="all",
        help="Restrict to specific modality (default: all)",
    )
    encode_parser.add_argument(
        "--json", "-j", action="store_true", help="Output JSON format"
    )

    # Optional: one-shot encode + distribute
    encode_parser.add_argument(
        "--profile",
        "-p",
        choices=list(ACTIVITY_PROFILES.keys()),
        default=None,
        help="If set, immediately distribute using this activity profile",
    )
    encode_parser.add_argument(
        "--policy",
        choices=["round_robin", "fixed", "alternating"],
        default="round_robin",
        help="Channel dispatch policy (default: round_robin)",
    )
    encode_parser.add_argument(
        "--seed", type=int, default=42, help="Noise model RNG seed (default: 42)"
    )
    encode_parser.add_argument(
        "--base-delay",
        type=int,
        default=3,
        help="Base delay (seconds) per item before noise (default: 3)",
    )

    # Decode command
    decode_parser = subparsers.add_parser(
        "decode", help="Decode media IDs to semantic meaning"
    )
    decode_parser.add_argument("ids", help="Comma-separated media IDs")
    decode_parser.add_argument(
        "--json", "-j", action="store_true", help="Output JSON format"
    )

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", help="Full encode -> decode demonstration"
    )
    demo_parser.add_argument("message", help="Message to encode and decode")
    demo_parser.add_argument(
        "--mode",
        "-m",
        choices=["best", "round_robin", "balanced"],
        default="best",
        help="Diversity mode (default: best)",
    )

    # Status command
    subparsers.add_parser("status", help="Show system status and index information")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search the corpus for a query"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--k", "-k", type=int, default=5, help="Number of results (default: 5)"
    )
    search_parser.add_argument(
        "--modality",
        choices=["all", "image", "text", "audio"],
        default="all",
        help="Restrict to specific modality (default: all)",
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify media IDs exist in corpus"
    )
    verify_parser.add_argument("ids", help="Comma-separated media IDs to verify")

    # Distribute command
    distribute_parser = subparsers.add_parser(
        "distribute", help="Encode and distribute with timing profiles"
    )
    distribute_parser.add_argument("message", help="Message to distribute")
    distribute_parser.add_argument(
        "--profile",
        "-p",
        choices=list(ACTIVITY_PROFILES.keys()),
        default="casual",
        help="Activity profile (default: casual)",
    )

    distribute_parser.add_argument(
        "--policy",
        choices=["round_robin", "fixed", "alternating"],
        default="round_robin",
        help="Channel dispatch policy (default: round_robin)",
    )
    distribute_parser.add_argument(
        "--seed", type=int, default=42, help="Noise model RNG seed (default: 42)"
    )
    distribute_parser.add_argument(
        "--base-delay",
        type=int,
        default=3,
        help="Base delay (seconds) per item before noise (default: 3)",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run semantic recovery benchmark"
    )
    benchmark_parser.add_argument(
        "--modes",
        "-m",
        default="all",
        help="Diversity modes to test: 'all' or comma-separated (default: all)",
    )
    benchmark_parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick benchmark with fewer samples (3 per category)",
    )
    benchmark_parser.add_argument(
        "--quiet", action="store_true", help="Minimal output during benchmark"
    )
    benchmark_parser.add_argument("--output", "-o", help="Output path for results JSON")
    benchmark_parser.add_argument(
        "--markdown", "--md", action="store_true", help="Also save markdown report"
    )

    return parser


# =========================================================
# MAIN ENTRY POINT
# =========================================================


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "encode": cmd_encode,
        "decode": cmd_decode,
        "demo": cmd_demo,
        "status": cmd_status,
        "search": cmd_search,
        "verify": cmd_verify,
        "distribute": cmd_distribute,
        "benchmark": cmd_benchmark,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            print("\nInterrupted")
            return 130
        except Exception as e:
            print_error(str(e))
            if "--debug" in sys.argv:
                raise
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
