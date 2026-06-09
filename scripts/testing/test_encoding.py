#!/usr/bin/env python3
"""
Comprehensive encoding/decoding test script.

Tests:
1. Balanced vs Round-robin mode comparison
2. Edge cases (short, long, empty, special chars)
3. Score calibration measurement
4. Modality distribution analysis

Usage:
    python scripts/test_encoding.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engine.encoder import SemanticEncoder
from src.engine.decoder import SemanticDecoder


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def test_diversity_modes(encoder: SemanticEncoder):
    """Test and compare diversity modes."""
    print_header("TEST 1: DIVERSITY MODE COMPARISON")
    
    test_message = (
        "The quick brown fox jumps over the lazy dog. "
        "Meet me at the coffee shop tomorrow morning. "
        "The weather is beautiful today with clear blue skies."
    )
    
    print(f"Test message: \"{test_message[:60]}...\"")
    
    results = {}
    
    for mode in ["best", "round_robin", "balanced"]:
        print_subheader(f"Mode: {mode}")
        
        result = encoder.encode(test_message, diversity_mode=mode)
        breakdown = result.modality_breakdown
        
        results[mode] = {
            "breakdown": breakdown,
            "total_chunks": len(result.chunks),
            "media_ids": result.media_ids[:5],  # First 5
            "scores": [e.media.normalized_score for e in result.encoded]
        }
        
        print(f"  Chunks: {len(result.chunks)}")
        print(f"  Modality breakdown: {breakdown}")
        print(f"  Avg score: {np.mean(results[mode]['scores']):.3f}")
        print(f"  Score range: {min(results[mode]['scores']):.3f} - {max(results[mode]['scores']):.3f}")
        
        # Show first 3 encodings
        for i, enc in enumerate(result.encoded[:3], 1):
            print(f"    {i}. [{enc.media.modality}] \"{enc.chunk.original[:30]}...\" -> {enc.media.id}")
    
    # Summary comparison
    print_subheader("Summary Comparison")
    print(f"{'Mode':<12} {'Image':<8} {'Text':<8} {'Audio':<8} {'Avg Score':<10}")
    print("-" * 50)
    for mode, data in results.items():
        b = data["breakdown"]
        avg = np.mean(data["scores"])
        print(f"{mode:<12} {b.get('image', 0):<8} {b.get('text', 0):<8} {b.get('audio', 0):<8} {avg:<10.3f}")
    
    return results


def test_edge_cases(encoder: SemanticEncoder, decoder: SemanticDecoder):
    """Test edge cases."""
    print_header("TEST 2: EDGE CASES")
    
    test_cases = [
        ("Very short", "Hi"),
        ("Single word", "Dog"),
        ("Numbers", "Meet at 5pm on the 23rd"),
        ("Special chars", "Hello! @#$% World???"),
        ("Question", "What is the meaning of life?"),
        ("Command", "Run to the store and buy milk"),
        ("Long sentence", "The extraordinarily magnificent and beautifully decorated ancient castle stood majestically upon the highest peak of the mountain overlooking the vast valley below where the river flowed gently through the meadows"),
    ]
    
    results = []
    
    for name, message in test_cases:
        print_subheader(f"{name}: \"{message[:40]}{'...' if len(message) > 40 else ''}\"")
        
        try:
            # Encode
            result = encoder.encode(message, diversity_mode="round_robin")
            
            print(f"  Chunks: {len(result.chunks)}")
            print(f"  Modalities: {result.modality_breakdown}")
            
            # Decode
            decode_result = decoder.decode(result.media_ids)
            
            print(f"  Verified: {decode_result.verification_rate * 100:.0f}%")
            print(f"  First content: \"{decode_result.contents[0][:50]}...\"" if decode_result.contents else "  No content")
            
            results.append({
                "name": name,
                "message": message,
                "chunks": len(result.chunks),
                "modalities": result.modality_breakdown,
                "verified": decode_result.verification_rate,
                "status": "OK"
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": name,
                "message": message,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Test empty message
    print_subheader("Empty message: \"\"")
    try:
        result = encoder.encode("")
        print(f"  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
        results.append({"name": "Empty", "message": "", "status": "OK (ValueError)"})
    except Exception as e:
        print(f"  Unexpected error: {e}")
        results.append({"name": "Empty", "message": "", "status": "ERROR", "error": str(e)})
    
    return results


def measure_score_calibration(encoder: SemanticEncoder):
    """Measure actual score distributions per modality."""
    print_header("TEST 3: SCORE CALIBRATION MEASUREMENT")
    
    # Sample queries for calibration
    queries = [
        "a dog running on the beach",
        "sunset over the ocean",
        "people walking in the city",
        "a cat sleeping on a couch",
        "children playing in the park",
        "a mountain covered with snow",
        "cars driving on the highway",
        "a bird flying in the sky",
        "food on a plate",
        "a person reading a book",
        "music playing in the background",
        "rain falling on the window",
        "a tree in the forest",
        "the sun rising in the morning",
        "a boat on the water",
        "flowers in a garden",
        "a building in the city",
        "an airplane in the sky",
        "a horse running in a field",
        "waves crashing on the shore",
    ]
    
    print(f"Running {len(queries)} sample queries against each modality...")
    
    scores_by_modality = {"image": [], "text": [], "audio": []}
    
    for query in queries:
        for modality in ["image", "text", "audio"]:
            results = encoder.index.search(query, k=5, modalities=[modality])
            for r in results:
                scores_by_modality[modality].append(r.score)  # Raw score
    
    print_subheader("Raw Score Statistics (before normalization)")
    print(f"{'Modality':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Count':<8}")
    print("-" * 60)
    
    calibration = {}
    for modality, scores in scores_by_modality.items():
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            calibration[modality] = (round(mean, 3), round(std, 3))
            print(f"{modality:<10} {mean:<10.3f} {std:<10.3f} {min(scores):<10.3f} {max(scores):<10.3f} {len(scores):<8}")
    
    print_subheader("Recommended Calibration Values")
    print("DEFAULT_CALIBRATION = {")
    for modality, (mean, std) in calibration.items():
        print(f'    "{modality}": ({mean}, {std}),')
    print("}")
    
    # Compare with current values
    print_subheader("Current vs Measured")
    current = {
        "image": (0.28, 0.06),
        "text": (0.65, 0.15),
        "audio": (0.25, 0.08),
    }
    print(f"{'Modality':<10} {'Current Mean':<15} {'Measured Mean':<15} {'Diff':<10}")
    print("-" * 55)
    for modality in ["image", "text", "audio"]:
        curr_mean = current[modality][0]
        meas_mean = calibration.get(modality, (0, 0))[0]
        diff = meas_mean - curr_mean
        print(f"{modality:<10} {curr_mean:<15.3f} {meas_mean:<15.3f} {diff:+<10.3f}")
    
    return calibration


def test_very_long_message(encoder: SemanticEncoder, decoder: SemanticDecoder):
    """Test with a very long message."""
    print_header("TEST 4: VERY LONG MESSAGE")
    
    # Generate a long message
    long_message = (
        "The quick brown fox jumps over the lazy dog. " * 10 +
        "In a galaxy far far away, there lived a brave hero who embarked on an epic journey. " * 5 +
        "The weather today is sunny with a chance of rain in the afternoon. " * 5 +
        "Technology continues to advance at an unprecedented rate, transforming how we live and work. " * 5
    )
    
    print(f"Message length: {len(long_message)} characters")
    print(f"Preview: \"{long_message[:100]}...\"")
    
    try:
        result = encoder.encode(long_message, diversity_mode="balanced")
        
        print(f"\nChunks created: {len(result.chunks)}")
        print(f"Modality breakdown: {result.modality_breakdown}")
        print(f"Media items: {len(result.media_ids)}")
        
        # Decode
        decode_result = decoder.decode(result.media_ids)
        print(f"Verification rate: {decode_result.verification_rate * 100:.0f}%")
        
        # Check for any issues
        if len(result.chunks) > 50:
            print("\nWARNING: Very high chunk count - consider adjusting chunker settings")
        
        return {"status": "OK", "chunks": len(result.chunks), "modalities": result.modality_breakdown}
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


def main():
    """Run all tests."""
    print_header("DCASS ENCODING/DECODING TEST SUITE")
    print("Initializing encoder and decoder...")
    
    # Initialize
    encoder = SemanticEncoder(expand_synonyms=True)
    encoder.load()
    
    decoder = SemanticDecoder()
    decoder.load()
    
    # Run tests
    results = {}
    
    # Test 1: Diversity modes
    results["diversity_modes"] = test_diversity_modes(encoder)
    
    # Test 2: Edge cases
    results["edge_cases"] = test_edge_cases(encoder, decoder)
    
    # Test 3: Score calibration
    results["calibration"] = measure_score_calibration(encoder)
    
    # Test 4: Very long message
    results["long_message"] = test_very_long_message(encoder, decoder)
    
    # Final summary
    print_header("TEST SUITE COMPLETE")
    print("\nAll tests completed. Review the output above for details.")
    print("\nKey findings:")
    print(f"  - Recommended calibration values have been calculated")
    print(f"  - Edge cases tested: {len(results['edge_cases'])} scenarios")
    print(f"  - Diversity modes compared: best, round_robin, balanced")
    
    return results


if __name__ == "__main__":
    main()
