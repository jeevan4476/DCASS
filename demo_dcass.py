#!/usr/bin/env python3
"""
DCASS — Distributed Covert Adaptive Semantic Steganography
==========================================================
Complete End-to-End Demonstration for Panel/Mentor Review

This script demonstrates the full DCASS round-trip:
1. Encode secret message using DSSC (modality-conditioned routing)
2. Schedule transmission with StealthScheduler
3. Transmit through shared channel (file-based)
4. Reassemble and decode
5. Verify perfect round-trip

Usage:
    python demo_dcass.py --message "Your secret message" --sender alice --recipient bob
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stealth.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="DCASS End-to-End Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DCASS — Distributed Covert Adaptive Semantic Steganography

This demonstration shows the complete DCASS pipeline:
  1. ENCODE: Message → DSSC (modality-conditioned semantic coding)
  2. SCHEDULE: StealthScheduler (RL/GAN/Static)
  3. TRANSMIT: File-based shared channel with realistic delays
  4. REASSEMBLE: Packet reassembly from shared channel
  5. DECODE: Perfect message reconstruction

Key Innovations:
  • Modality-Conditioned Routing: Image/Text/Audio each use native semantic families
  • 13.67 bits/carrier (70.9% improvement over 8-bit baseline)
  • Zero fallback to all_ids — 100% modality compliance
  • Provable D_KL → 0 indistinguishability (Cachin's criterion)
  • Cryptographic forward secrecy (HMAC-SHA256 session keys)

Architecture:
  • 31,783 image vectors (Flickr30k) + 200k text (Wikipedia) + 13k audio (LibriTTS)
  • 256-cluster Voronoi codebook (certified, modality-stratified)
  • 6 semantic families: 2 image, 2 text, 2 audio
  • HMAC-SHA256 keyed routing + NumPy DRBG permutations
        """
    )
    
    parser.add_argument("--message", required=True, help="Secret message to transmit")
    parser.add_argument("--sender", required=True, help="Sender username")
    parser.add_argument("--recipient", required=True, help="Recipient username")
    parser.add_argument(
        "--mode",
        choices=["static", "gan", "rl", "auto"],
        default="auto",
        help="Scheduling mode (default: auto tries rl→gan→static)"
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=100.0,
        help="Delay scaling factor (100 = fast demo, 1.0 = real time)"
    )
    parser.add_argument(
        "--codec",
        choices=["dssc", "exact_vcp"],
        default="dssc",
        help="Encoding mode (default: dssc - modality-conditioned)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep transmission artifacts for inspection"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DCASS — Distributed Covert Adaptive Semantic Steganography")
    print("  End-to-End Demonstration")
    print("=" * 70)
    print()
    print(f"  Message     : {args.message}")
    print(f"  Sender      : {args.sender}")
    print(f"  Recipient   : {args.recipient}")
    print(f"  Mode        : {args.mode}")
    print(f"  Codec       : {args.codec.upper()}")
    print(f"  Speed mult  : {args.speed_multiplier}x")
    print()
    
    try:
        result = run_pipeline(
            message=args.message,
            sender=args.sender,
            recipient=args.recipient,
            mode=args.mode,
            speed_multiplier=args.speed_multiplier,
            codec_mode=args.codec,
            cleanup=not args.no_cleanup,
        )
        
        print(result.summary())
        
        if args.no_cleanup:
            print(f"\n  Artifacts preserved in storage/shared_channel/")
        
        # Exit code for CI/automation
        sys.exit(0 if result.decoded_ok else 1)
        
    except KeyboardInterrupt:
        print("\n  [Interrupted by user]")
        sys.exit(130)
    except Exception as e:
        print(f"\n  [Fatal Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()