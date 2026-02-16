# src/cli/main.py
"""
DCASS CLI - Phase 3 Demo Entry Point

CLI CONTRACT
------------
Commands:
  encode <message>              Encode a message into media sequence
  decode <media_ids>            Decode media IDs back to semantic meaning
  distribute <message> [profile] Full encode + distribution pipeline
  demo <message>                Full encode -> decode demo with verification

Profiles:
  casual | steady | bursty | night_owl | debug

Rules:
- CLI does NOT implement logic
- CLI only orchestrates modules
- All intelligence lives in src.engine and src.corpus
"""

import sys
import json

# -------- New Engine (Phase 3) --------
from src.engine.encoder import SemanticEncoder
from src.engine.decoder import SemanticDecoder

# -------- Legacy Encoder (backward compatibility) --------
from src.embeddings.step8_sentence_to_images import sentence_to_image_sequence

# -------- Distribution Layer --------
from src.distribution.channel_registry import get_available_channels
from src.distribution.dispatcher import Dispatcher
from src.distribution.scheduler import Scheduler
from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES


# =========================================================
# SHARED STATE (lazy loaded)
# =========================================================
_encoder = None
_decoder = None


def _get_encoder() -> SemanticEncoder:
    """Get or create the shared encoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = SemanticEncoder(expand_synonyms=True)
        _encoder.load()
    return _encoder


def _get_decoder() -> SemanticDecoder:
    """Get or create the shared decoder instance."""
    global _decoder
    if _decoder is None:
        _decoder = SemanticDecoder()
        _decoder.load()
    return _decoder


# =========================================================
# CLI COMMANDS
# =========================================================

def run_encode(message: str, use_new_encoder: bool = True):
    """
    Encode a message into a media sequence.
    
    Args:
        message: Secret message to encode
        use_new_encoder: Use new SemanticEncoder (True) or legacy (False)
    """
    print("\n" + "=" * 60)
    print("DCASS ENCODER")
    print("=" * 60)
    print(f"Message: \"{message}\"")
    print("-" * 60)
    
    if use_new_encoder:
        try:
            encoder = _get_encoder()
            result = encoder.encode(message)
            
            print(f"\nChunks ({len(result.chunks)}):")
            for chunk in result.chunks:
                print(f"  - \"{chunk.original}\"")
            
            print(f"\nEncoded Media Sequence:")
            for i, enc in enumerate(result.encoded, 1):
                print(f"  {i}. [{enc.media.modality}] {enc.media.id}")
                print(f"      Score: {enc.media.normalized_score:.3f}")
                content_preview = enc.media.content[:50] + "..." if len(enc.media.content) > 50 else enc.media.content
                print(f"      Content: \"{content_preview}\"")
            
            print(f"\nModality breakdown: {result.modality_breakdown}")
            print(f"\nMedia IDs for transmission:")
            print(f"  {result.media_ids}")
            
            return result.media_ids
            
        except Exception as e:
            print(f"\nNew encoder failed: {e}")
            print("Falling back to legacy encoder...")
            use_new_encoder = False
    
    if not use_new_encoder:
        # Legacy encoder
        print("\n[Using Legacy Encoder]")
        image_sequence = sentence_to_image_sequence(message)
        
        if not image_sequence:
            print("No images produced.")
            return []
        
        print(f"\nEncoded image sequence ({len(image_sequence)} images):")
        for idx, img in enumerate(image_sequence, 1):
            print(f"  {idx}. {img}")
        
        return image_sequence


def run_decode(media_ids: list[str]):
    """
    Decode a media sequence back to semantic meaning.
    
    Args:
        media_ids: List of media IDs to decode
    """
    print("\n" + "=" * 60)
    print("DCASS DECODER")
    print("=" * 60)
    print(f"Media IDs: {media_ids}")
    print("-" * 60)
    
    decoder = _get_decoder()
    result = decoder.decode(media_ids)
    
    print(f"\nDecoded Items ({len(result.decoded)}):")
    for i, item in enumerate(result.decoded, 1):
        status = "OK" if item.verified else "UNVERIFIED"
        print(f"  {i}. [{status}] {item.modality or 'unknown'}: {item.media_id}")
        content_preview = item.content[:60] + "..." if len(item.content) > 60 else item.content
        print(f"      Content: \"{content_preview}\"")
    
    print(f"\nVerification: {result.verification_rate * 100:.1f}% verified")
    print(f"\nReconstructed Meaning:")
    print(f"  \"{result.reconstructed_meaning}\"")
    
    return result


def run_demo(message: str):
    """
    Full encode -> decode demo with verification.
    
    Args:
        message: Message to encode and decode
    """
    print("\n" + "=" * 60)
    print("DCASS FULL DEMO: ENCODE -> DECODE")
    print("=" * 60)
    print(f"Original Message: \"{message}\"")
    print("=" * 60)
    
    # Encode
    print("\n[STEP 1: ENCODING]")
    encoder = _get_encoder()
    encode_result = encoder.encode(message)
    
    print(f"  Chunks: {[c.original for c in encode_result.chunks]}")
    print(f"  Media IDs: {encode_result.media_ids}")
    print(f"  Modalities: {encode_result.modality_breakdown}")
    
    # Simulate transmission
    print("\n[STEP 2: TRANSMISSION]")
    transmitted_ids = encode_result.media_ids
    print(f"  Transmitting {len(transmitted_ids)} media items...")
    print(f"  IDs: {transmitted_ids}")
    
    # Decode
    print("\n[STEP 3: DECODING]")
    decoder = _get_decoder()
    decode_result = decoder.decode(transmitted_ids)
    
    print(f"  Verified: {decode_result.verification_rate * 100:.1f}%")
    print(f"  Contents: {decode_result.semantic_content}")
    
    # Compare
    print("\n[STEP 4: VERIFICATION]")
    print(f"  Original:     \"{message}\"")
    print(f"  Reconstructed: \"{decode_result.reconstructed_meaning}\"")
    
    if decode_result.all_verified:
        print("\n  STATUS: ALL ITEMS VERIFIED IN CORPUS")
    else:
        print("\n  WARNING: Some items could not be verified!")
    
    print("\n" + "=" * 60)
    return encode_result, decode_result


def run_distribute(message: str, profile_name: str):
    """
    Full encode + distribution pipeline.
    
    Args:
        message: Message to encode and distribute
        profile_name: Activity profile (casual, steady, bursty, etc.)
    """
    print("\n" + "=" * 60)
    print("DCASS DISTRIBUTION")
    print("=" * 60)
    print(f"Message: \"{message}\"")
    print(f"Profile: {profile_name}")
    print("-" * 60)
    
    # Get profile
    profile = ACTIVITY_PROFILES.get(profile_name)
    if profile is None:
        print(f"Unknown profile: {profile_name}")
        print(f"Available profiles: {list(ACTIVITY_PROFILES.keys())}")
        return
    
    # Encode using legacy encoder (more stable for distribution)
    print("\n[ENCODING]")
    image_sequence = sentence_to_image_sequence(message, verbose=False)
    
    if not image_sequence:
        print("No images produced. Exiting.")
        return
    
    print(f"  Produced {len(image_sequence)} image(s)")
    for i, img_id in enumerate(image_sequence, 1):
        print(f"    {i}. {img_id}")
    
    # Apply noise
    print("\n[APPLYING NOISE]")
    noise = NoiseController(seed=42, **profile)
    images, delays = noise.apply(
        image_sequence,
        base_delays=[3] * len(image_sequence)
    )
    
    if not images:
        print("All images skipped by noise model.")
        return
    
    print(f"  After noise: {len(images)} images")
    print(f"  Delays: {delays}")
    
    # Setup dispatcher
    print("\n[DISTRIBUTION]")
    dispatcher = Dispatcher(
        channels=get_available_channels(),
        policy="round_robin"
    )
    
    scheduler = Scheduler(
        dispatcher=dispatcher,
        delays=delays
    )
    
    print("Executing scheduled distribution...")
    scheduler.run(images)
    
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPLETE")
    print("=" * 60)


# =========================================================
# CLI ENTRY POINT
# =========================================================

def print_usage():
    """Print CLI usage information."""
    print("""
DCASS CLI - Dynamic Context-Aware Semantic Steganography

Usage:
  python -m src.cli.main <command> [args]

Commands:
  encode <message>                 Encode message into media sequence
  decode <id1,id2,...>             Decode media IDs to semantic meaning
  demo <message>                   Full encode -> decode demonstration
  distribute <message> [profile]   Encode and distribute with timing

Profiles (for distribute):
  casual    - Relaxed posting pattern
  steady    - Consistent timing
  bursty    - Clustered activity
  night_owl - Late night pattern  
  debug     - No delays (testing)

Examples:
  python -m src.cli.main encode "Meet me at the cafe at noon"
  python -m src.cli.main decode "flickr8k_00123,flickr8k_00456"
  python -m src.cli.main demo "The secret meeting is tomorrow"
  python -m src.cli.main distribute "Hello world" casual
""")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "encode":
        if len(sys.argv) < 3:
            print("Error: encode requires a message")
            print("Usage: python -m src.cli.main encode <message>")
            return
        message = sys.argv[2]
        run_encode(message)
    
    elif command == "decode":
        if len(sys.argv) < 3:
            print("Error: decode requires media IDs")
            print("Usage: python -m src.cli.main decode <id1,id2,...>")
            return
        # Parse comma-separated IDs
        ids_str = sys.argv[2]
        media_ids = [id.strip() for id in ids_str.split(",") if id.strip()]
        run_decode(media_ids)
    
    elif command == "demo":
        if len(sys.argv) < 3:
            print("Error: demo requires a message")
            print("Usage: python -m src.cli.main demo <message>")
            return
        message = sys.argv[2]
        run_demo(message)
    
    elif command == "distribute":
        if len(sys.argv) < 3:
            print("Error: distribute requires a message")
            print("Usage: python -m src.cli.main distribute <message> [profile]")
            return
        message = sys.argv[2]
        profile = sys.argv[3] if len(sys.argv) > 3 else "casual"
        run_distribute(message, profile)
    
    elif command in ("help", "-h", "--help"):
        print_usage()
    
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
