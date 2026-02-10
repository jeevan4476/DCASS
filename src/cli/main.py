# src/cli/main.py
"""
DCASS CLI — Phase 3 Demo Entry Point

CLI CONTRACT
------------
Commands:
  encode <sentence>
  distribute <sentence> [profile]

Profiles:
  casual | steady | bursty

Rules:
- CLI does NOT implement logic
- CLI only orchestrates modules
- All intelligence lives elsewhere
"""

import sys

# -------- Layer 2 (Semantic Encoding) --------
from src.embeddings.step8_sentence_to_images import sentence_to_image_sequence

# -------- Phase 3 (Behavioral Layer) --------
from src.distribution.channel_registry import get_available_channels
from src.distribution.dispatcher import Dispatcher
from src.distribution.scheduler import Scheduler
from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES


# =========================================================
# STEP 6.3 — CLI FUNCTIONS (ONLY TWO, NO EXTRA LOGIC)
# =========================================================

def run_encode(sentence: str):
    """
    Layer 2 only.
    Semantic encoding without any distribution.
    """
    print("\n[LAYER 2] Semantic Encoding")
    print("--------------------------")
    print(f"Input sentence: {sentence}\n")

    image_sequence = sentence_to_image_sequence(sentence)

    if not image_sequence:
        print("No images produced.")
        return

    print("Encoded image sequence:")
    for idx, img in enumerate(image_sequence, 1):
        print(f"{idx}. {img}")

    print("\n[LAYER 2] Done\n")


def run_distribute(sentence: str, profile_name: str):
    """
    Full Layer 2 + Phase 3 pipeline.
    """
    print("\n[DCASS] Distribution Demo")
    print("------------------------")
    print(f"Sentence : {sentence}")
    print(f"Profile  : {profile_name}\n")

    # ---- Layer 2 ----
    print("[LAYER 2] Encoding...")
    image_sequence = sentence_to_image_sequence(sentence)

    if not image_sequence:
        print("No images produced. Exiting.")
        return

    print(f"Produced {len(image_sequence)} image(s)\n")

    # ---- Phase 3 (wiring only, no logic here) ----
    profile = ACTIVITY_PROFILES.get(profile_name)
    if profile is None:
        print(f"Unknown profile: {profile_name}")
        return

    noise = NoiseController(seed=42, **profile)

    images, delays = noise.apply(
        image_sequence,
        base_delays=[3] * len(image_sequence)
    )

    if not images:
        print("All images skipped by noise model.")
        return

    dispatcher = Dispatcher(
        channels=get_available_channels(),
        policy="round_robin"
    )

    scheduler = Scheduler(
        dispatcher=dispatcher,
        delays=delays
    )

    print("[PHASE 3] Executing scheduled distribution...\n")
    scheduler.run(images)

    print("\n[PHASE 3] Done\n")


# =========================================================
# STEP 6.2 — CLI ENTRY POINT (ARGUMENT HANDLING ONLY)
# =========================================================

def print_usage():
    print("\nUsage:")
    print("  python -m src.cli.main encode <sentence>")
    print("  python -m src.cli.main distribute <sentence> [profile]\n")
    print("Available profiles:")
    for p in ACTIVITY_PROFILES.keys():
        print(f"  - {p}")
    print()


def main():
    if len(sys.argv) < 3:
        print_usage()
        return

    command = sys.argv[1]
    sentence = sys.argv[2]

    if command == "encode":
        run_encode(sentence)

    elif command == "distribute":
        profile = sys.argv[3] if len(sys.argv) > 3 else "casual"
        run_distribute(sentence, profile)

    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
