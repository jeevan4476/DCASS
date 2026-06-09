#!/usr/bin/env python3
"""
DCASS Demo Script - End-to-End Demonstration

This script demonstrates the full DCASS steganography pipeline:
1. Encode a secret message into media sequences
2. Decode media IDs back to semantic meaning
3. Show verification and semantic similarity

Usage:
    python scripts/demo_dcass.py
    python scripts/demo_dcass.py "Your custom secret message"
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.encoder import SemanticEncoder
from src.engine.decoder import SemanticDecoder
from src.engine.chunker import SemanticChunker


def print_banner(text: str, char: str = "="):
    """Print a banner with text."""
    width = 70
    print(char * width)
    print(f" {text}")
    print(char * width)


def demo_chunking():
    """Demonstrate the chunking process."""
    print_banner("DEMO 1: Semantic Chunking", "-")
    
    chunker = SemanticChunker(expand_synonyms=True)
    
    messages = [
        "Meet me at the cafe, bring the documents",
        "A happy dog running on the beach",
        "The secret meeting is tomorrow at dawn"
    ]
    
    for msg in messages:
        print(f"\nInput: \"{msg}\"")
        chunks = chunker.chunk(msg)
        print(f"Chunks ({len(chunks)}):")
        for c in chunks:
            if c.text != c.original:
                print(f"  - \"{c.original}\" -> \"{c.text}\" (expanded)")
            else:
                print(f"  - \"{c.text}\"")


def demo_encoding(message: str = None):
    """Demonstrate encoding."""
    print_banner("DEMO 2: Semantic Encoding")
    
    message = message or "A dog running on the beach"
    print(f"\nSecret Message: \"{message}\"")
    print("-" * 50)
    
    # Initialize encoder
    print("\nInitializing encoder...")
    encoder = SemanticEncoder(expand_synonyms=True, default_modalities=["image", "text"])
    
    print("Loading indices...")
    status = encoder.load()
    print(f"  Load status: {status}")
    
    if not encoder.is_loaded():
        print("\nERROR: Failed to load indices!")
        print("Make sure data/indices/ contains:")
        print("  - image.index + image_metadata.json")
        print("  - text.index + text_metadata.json")
        return None
    
    # Encode
    print("\nEncoding message...")
    result = encoder.encode(message)
    
    print(f"\nChunks ({len(result.chunks)}):")
    for chunk in result.chunks:
        print(f"  - \"{chunk.original}\"")
    
    print(f"\nEncoded Media Sequence ({len(result.encoded)} items):")
    for i, enc in enumerate(result.encoded, 1):
        media = enc.media
        print(f"\n  {i}. Chunk: \"{enc.chunk.original}\"")
        print(f"     Media: [{media.modality}] {media.id}")
        print(f"     Score: {media.normalized_score:.3f} (raw: {media.score:.3f})")
        
        # Show content preview
        content = media.content
        if media.modality == "image":
            content = media.metadata.get("caption", content)
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"     Content: \"{preview}\"")
        
        # Show alternatives
        if enc.alternatives:
            print(f"     Alternatives: {[a.id for a in enc.alternatives[:2]]}")
    
    print(f"\nModality breakdown: {result.modality_breakdown}")
    print(f"\nMedia IDs for transmission:")
    print(f"  {result.media_ids}")
    
    return result


def demo_decoding(media_ids: list = None):
    """Demonstrate decoding."""
    print_banner("DEMO 3: Semantic Decoding")
    
    if not media_ids:
        print("\nNo media IDs provided. Using example IDs...")
        media_ids = ["flickr8k_00000", "flickr8k_00001"]
    
    print(f"\nReceived Media IDs: {media_ids}")
    print("-" * 50)
    
    # Initialize decoder
    print("\nInitializing decoder...")
    decoder = SemanticDecoder()
    
    print("Loading indices...")
    status = decoder.load()
    print(f"  Load status: {status}")
    
    if not decoder.is_loaded():
        print("\nERROR: Failed to load indices!")
        return None
    
    # Decode
    print("\nDecoding media sequence...")
    result = decoder.decode(media_ids)
    
    print(f"\nDecoded Items ({len(result.decoded)}):")
    for i, item in enumerate(result.decoded, 1):
        status_str = "VERIFIED" if item.verified else "UNVERIFIED"
        print(f"\n  {i}. [{status_str}] {item.modality or 'unknown'}: {item.media_id}")
        
        content = item.content
        preview = content[:70] + "..." if len(content) > 70 else content
        print(f"     Content: \"{preview}\"")
    
    print(f"\nVerification Rate: {result.verification_rate * 100:.1f}%")
    print(f"\nReconstructed Meaning:")
    print(f"  \"{result.reconstructed_meaning}\"")
    
    return result


def demo_full_loop(message: str = None):
    """Full encode -> decode demonstration."""
    print_banner("DEMO 4: Full Encode -> Decode Loop", "=")
    
    message = message or "Meet me at the beach, bring the dog"
    print(f"\n ORIGINAL MESSAGE: \"{message}\"")
    print("=" * 70)
    
    # Step 1: Encode
    print("\n[STEP 1: ENCODING]")
    encoder = SemanticEncoder(expand_synonyms=True, default_modalities=["image", "text"])
    encoder.load()
    
    if not encoder.is_loaded():
        print("ERROR: Encoder failed to load!")
        return
    
    encode_result = encoder.encode(message)
    
    print(f"  Message chunked into {len(encode_result.chunks)} parts")
    print(f"  Encoded to {len(encode_result.media_ids)} media items")
    print(f"  Modalities used: {encode_result.modality_breakdown}")
    
    # Step 2: Simulate transmission
    print("\n[STEP 2: TRANSMISSION]")
    transmitted_ids = encode_result.media_ids
    print(f"  Transmitting {len(transmitted_ids)} media IDs...")
    print(f"  IDs: {transmitted_ids}")
    
    # Step 3: Decode
    print("\n[STEP 3: DECODING]")
    decoder = SemanticDecoder()
    decoder.load()
    
    decode_result = decoder.decode(transmitted_ids)
    
    print(f"  Received {len(decode_result.decoded)} items")
    print(f"  Verification rate: {decode_result.verification_rate * 100:.1f}%")
    
    # Step 4: Compare
    print("\n[STEP 4: RESULTS]")
    print("-" * 50)
    print(f"  Original:      \"{message}\"")
    print(f"  Reconstructed: \"{decode_result.reconstructed_meaning}\"")
    print("-" * 50)
    
    # Show detailed mapping
    print("\n  Chunk-to-Media Mapping:")
    for i, enc in enumerate(encode_result.encoded):
        decoded = decode_result.decoded[i] if i < len(decode_result.decoded) else None
        chunk = enc.chunk.original
        media_id = enc.media.id
        content = decoded.content[:40] + "..." if decoded and len(decoded.content) > 40 else (decoded.content if decoded else "N/A")
        
        print(f"    \"{chunk}\" -> {media_id}")
        print(f"      Content: \"{content}\"")
    
    if decode_result.all_verified:
        print("\n  STATUS: ALL ITEMS VERIFIED IN CORPUS")
    else:
        print("\n  WARNING: Some items could not be verified!")
    
    print("\n" + "=" * 70)
    return encode_result, decode_result


def main():
    """Main demo entry point."""
    print("\n")
    print_banner("DCASS - Dynamic Context-Aware Semantic Steganography", "#")
    print("  Zero-Modification Media Curation Steganography Demo")
    print("#" * 70)
    
    # Get custom message from args
    custom_message = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        # Demo 1: Chunking
        demo_chunking()
        
        # Demo 2: Encoding
        print("\n")
        encode_result = demo_encoding(custom_message)
        
        if encode_result:
            # Demo 3: Decoding
            print("\n")
            demo_decoding(encode_result.media_ids)
            
            # Demo 4: Full loop
            print("\n")
            demo_full_loop(custom_message)
        else:
            print("\nSkipping decode and full loop demos (encoding failed)")
            
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n")
    print_banner("DEMO COMPLETE", "#")
    print("\nTo run your own message:")
    print('  python scripts/demo_dcass.py "Your secret message here"')
    print("\nOr use the CLI:")
    print('  python -m src.cli.main demo "Your secret message"')
    print('  python -m src.cli.main encode "Your secret message"')
    print('  python -m src.cli.main decode "id1,id2,id3"')
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
