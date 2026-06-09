from src.distribution.channel_registry import get_available_channels
from src.distribution.dispatcher import Dispatcher
from src.distribution.scheduler import Scheduler
from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES

channels = get_available_channels()

dispatcher = Dispatcher(
    channels=channels,
    policy="round_robin"
)

images = [
    "image_001",
    "image_002",
    "image_003",
    "image_004",
    "image_005",
    "image_006"
]

base_delays = [3] * len(images)

profile = ACTIVITY_PROFILES["casual"]

noise = NoiseController(
    seed=7,
    skip_prob=profile["skip_prob"],
    jitter_range=profile["jitter_range"],
    idle_gap_prob=profile["idle_gap_prob"],
    idle_gap_range=profile["idle_gap_range"]
)

images_noisy, delays_noisy = noise.apply(images, base_delays)

scheduler = Scheduler(
    dispatcher=dispatcher,
    delays=delays_noisy
)

logs = scheduler.run(images_noisy)

for log in logs:
    print(log)
