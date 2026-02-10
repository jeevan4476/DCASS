# src/cli/main.py
"""
DCASS CLI — Unified Entry Point

CLI CONTRACT
------------
Commands:
  encode <message>           - Encode a message into mixed media sequence
  decode <media_ids>         - Decode a media sequence  
  distribute <message> [profile] - Encode and distribute with timing
  status                     - Show index status

KEY FEATURE: Mixed-Modality Encoding
By default, encoding uses modality="auto" which searches ALL indices
and returns a MIX of images and texts for each message chunk.

Example:
  Input: "Secret meeting at dawn in the park"
  Output: [whisper.jpg, "The sun rises...", park_bench.jpg]
         [image,       text,              image]
"""

import sys
from pathlib import Path
from typing import Optional, Literal

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Type alias
Modality = Literal["text", "image", "audio", "auto"]


def run_encode(message: str, modality: Modality = "auto"):
    """
    Encode a message using mixed-modality search.
    
    By default (modality="auto"), searches ALL indices and returns
    the best match regardless of type. This produces a mixed sequence.
    """
    from src.engine.encoder import SemanticEncoder
    
    print("\n[DCASS] Mixed-Modality Semantic Encoding")
    print("-" * 50)
    print(f"Input: {message}")
    print(f"Mode: {modality} {'(mixed image+text)' if modality == 'auto' else ''}")
    print()
    
    # Create encoder with specified modality
    encoder = SemanticEncoder(default_modality=modality)
    
    try:
        encoder.load()  # Load all available indices
    except FileNotFoundError as e:
        print(f"Error: Index not found. Run 'python scripts/build_indices.py' first.")
        print(f"Details: {e}")
        return None
    
    # Encode message
    encoded = encoder.encode(message)
    
    # Display results
    print("Encoded Sequence:")
    print("-" * 50)
    
    for i, result in enumerate(encoded.sequence, 1):
        modality_icon = "🖼️" if result.modality == "image" else "📝"
        print(f"{i}. [{result.modality.upper()}] {modality_icon}")
        
        # Show content (truncated for text)
        content = result.content
        if result.modality == "text" and len(content) > 60:
            content = content[:60] + "..."
        print(f"   Content: {content}")
        print(f"   Score: {result.score:.4f}")
        
        if i <= len(encoded.chunks):
            print(f"   Chunk: '{encoded.chunks[i-1]}'")
        print()
    
    # Show statistics
    stats = encoder.get_statistics(encoded)
    print("Statistics:")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Media items: {stats['num_media']}")
    print(f"  Avg similarity: {stats['avg_similarity']:.4f}")
    print(f"  Mixed modality: {stats['is_mixed_modality']}")
    print(f"  Distribution: {stats['modality_distribution']}")
    print()
    
    return encoded


def run_decode(media_ids: list, modality: Modality = "auto"):
    """
    Decode a media sequence.
    """
    from src.engine.decoder import SemanticDecoder
    
    print("\n[DCASS] Semantic Decoding")
    print("-" * 40)
    print(f"Input: {len(media_ids)} media items")
    print()
    
    # Create decoder and load index
    decoder = SemanticDecoder(default_modality="image")  # Default for lookup
    
    try:
        decoder.load()  # Load all indices
    except FileNotFoundError as e:
        print(f"Error: Index not found.")
        print(f"Details: {e}")
        return None
    
    # Decode
    decoded = decoder.decode(media_ids)
    
    # Display results
    print("Decoded Message:")
    print("-" * 40)
    print(decoded.reconstructed_text)
    print()
    print(f"Confidence: {decoded.avg_confidence:.2f}")
    
    return decoded


def run_distribute(message: str, profile_name: str = "casual", modality: Modality = "auto"):
    """
    Full encoding + distribution pipeline.
    Uses mixed-modality by default.
    """
    from src.engine.encoder import SemanticEncoder
    from src.distribution.channel_registry import get_available_channels
    from src.distribution.dispatcher import Dispatcher
    from src.distribution.scheduler import Scheduler
    from src.distribution.noise import NoiseController
    from src.distribution.profiles import ACTIVITY_PROFILES
    
    print("\n[DCASS] Distribution Pipeline (Mixed-Modality)")
    print("-" * 50)
    print(f"Message: {message}")
    print(f"Profile: {profile_name}")
    print(f"Mode: {modality}")
    print()
    
    # ---- Encode with mixed modality ----
    print("[Encoding] Using mixed-modality search...")
    encoder = SemanticEncoder(default_modality=modality)
    
    try:
        encoder.load()
    except FileNotFoundError as e:
        print(f"Error: Index not found. Run 'python scripts/build_indices.py' first.")
        return
    
    encoded = encoder.encode(message)
    
    if not encoded.sequence:
        print("No media produced. Exiting.")
        return
    
    print(f"Produced {len(encoded.sequence)} media item(s)")
    print(f"Modality mix: {encoded.modality_distribution}")
    
    # ---- Distribute ----
    profile = ACTIVITY_PROFILES.get(profile_name)
    if profile is None:
        print(f"Unknown profile: {profile_name}")
        print(f"Available: {list(ACTIVITY_PROFILES.keys())}")
        return
    
    noise = NoiseController(seed=42, **profile)
    
    # Get media paths
    media_paths = encoded.media_paths
    
    media_items, delays = noise.apply(
        media_paths,
        base_delays=[3] * len(media_paths)
    )
    
    if not media_items:
        print("All items skipped by noise model.")
        return
    
    dispatcher = Dispatcher(
        channels=get_available_channels(),
        policy="round_robin"
    )
    
    scheduler = Scheduler(
        dispatcher=dispatcher,
        delays=delays
    )
    
    print("\n[Distribution] Executing scheduled distribution...")
    scheduler.run(media_items)
    
    print("\n[DCASS] Distribution complete")


def run_status():
    """
    Show status of indices.
    """
    from src.corpus.index.unified_index import UnifiedSemanticIndex
    
    print("\n[DCASS] Index Status")
    print("-" * 40)
    print("Note: All indices use CLIP embeddings for cross-modal search")
    
    index = UnifiedSemanticIndex()
    
    # Try to load each modality
    for modality in index.available_modalities:
        idx = index.get_index(modality)
        exists = idx.exists()
        
        print(f"\n{modality.upper()} Index:")
        print(f"  Path: {idx.index_path}")
        print(f"  Exists: {'Yes' if exists else 'No'}")
        
        if exists:
            try:
                idx.load()
                print(f"  Items: {idx.size}")
                print(f"  Embedding: CLIP 512-dim")
            except Exception as e:
                print(f"  Error loading: {e}")
    
    print()


def print_usage():
    """Print usage information."""
    print("\nDCASS - Dynamic Context-Aware Semantic Steganography")
    print("=" * 55)
    print("\n🔑 KEY FEATURE: Mixed-Modality Encoding")
    print("   Messages are encoded as a MIX of images AND texts!")
    print()
    print("Usage:")
    print("  python -m src.cli.main <command> [args]")
    print("\nCommands:")
    print("  encode <message>              Encode message to mixed media sequence")
    print("  decode <id1> <id2> ...        Decode media sequence")
    print("  distribute <message> [profile] Encode and distribute with timing")
    print("  status                        Show index status")
    print("\nProfiles for distribute:")
    print("  casual  - Relaxed timing (default)")
    print("  steady  - Regular intervals")
    print("  bursty  - Burst activity")
    print("\nExamples:")
    print('  python -m src.cli.main encode "Meet at dawn in the park"')
    print('  # Output might be: [image, text, image] - mixed!')
    print()
    print('  python -m src.cli.main distribute "Secret message" bursty')
    print("  python -m src.cli.main status")
    print("\nSetup:")
    print("  1. python scripts/download_flickr8k.py")
    print("  2. python scripts/build_indices.py")
    print("  3. python -m src.cli.main encode 'your message'")
    print()


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "encode":
        if len(sys.argv) < 3:
            print("Error: Message required")
            print("Usage: python -m src.cli.main encode <message>")
            return
        message = sys.argv[2]
        run_encode(message)
    
    elif command == "decode":
        if len(sys.argv) < 3:
            print("Error: Media IDs required")
            print("Usage: python -m src.cli.main decode <id1> <id2> ...")
            return
        media_ids = sys.argv[2:]
        run_decode(media_ids)
    
    elif command == "distribute":
        if len(sys.argv) < 3:
            print("Error: Message required")
            print("Usage: python -m src.cli.main distribute <message> [profile]")
            return
        message = sys.argv[2]
        profile = sys.argv[3] if len(sys.argv) > 3 else "casual"
        run_distribute(message, profile)
    
    elif command == "status":
        run_status()
    
    elif command in ["help", "-h", "--help"]:
        print_usage()
    
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
