from src.distribution.channel_registry import get_available_channels
from src.distribution.dispatcher import Dispatcher
from src.distribution.scheduler import Scheduler
from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES

# IMPORT YOUR EXISTING LAYER-2 ENCODER
from src.embeddings.step8_sentence_to_images import sentence_to_images


def run_distribute(
    sentence: str,
    profile_name: str = "casual",
    seed: int = 42,
    policy: str = "round_robin"
):
    print("\n[PHASE 3] Starting distribution")
    print(f"Profile: {profile_name}")
    print(f"Seed: {seed}")
    print(f"Policy: {policy}\n")

    # ---- Layer 2 ----
    image_sequence = sentence_to_images(sentence)

    if not image_sequence:
        print("No images generated. Exiting.")
        return

    print(f"Encoded image sequence: {image_sequence}\n")

    # ---- Phase 3 ----
    channels = get_available_channels()

    dispatcher = Dispatcher(
        channels=channels,
        policy=policy
    )

    base_delays = [3] * len(image_sequence)

    profile = ACTIVITY_PROFILES.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown profile: {profile_name}")

    noise = NoiseController(
        seed=seed,
        skip_prob=profile["skip_prob"],
        jitter_range=profile["jitter_range"],
        idle_gap_prob=profile["idle_gap_prob"],
        idle_gap_range=profile["idle_gap_range"]
    )

    images, delays = noise.apply(image_sequence, base_delays)

    if not images:
        print("All images skipped due to noise. Exiting.")
        return

    scheduler = Scheduler(
        dispatcher=dispatcher,
        delays=delays
    )

    logs = scheduler.run(images)

    print("\n[PHASE 3] Distribution complete\n")
    for log in logs:
        print(log)
