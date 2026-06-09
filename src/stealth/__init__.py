"""DCASS Stealth Package - GAN and RL Components.

Imports are lazy to avoid pulling in torch/faiss at package import time.
Use explicit submodule imports when you need these classes:

    from src.stealth.gan import TemporalPatternGenerator, GANTrainer
    from src.stealth.rl import PPOAgent, StealthEnvironment
    from src.stealth.stealth_scheduler import StealthScheduler
"""

__all__ = [
    # GAN
    "TemporalPatternGenerator",
    "TimingSchedule",
    "GANTrainer",
    "TrainingConfig",
    "HumanTrafficDataset",
    # RL
    "StealthEnvironment",
    "PPOAgent",
    "PPOConfig",
    "ActorCritic",
    # Scheduler
    "StealthScheduler",
]


def __getattr__(name: str):
    """Lazy import on first attribute access."""
    _gan = [
        "TemporalPatternGenerator", "TimingSchedule",
        "GANTrainer", "TrainingConfig", "HumanTrafficDataset",
    ]
    _rl = ["StealthEnvironment", "PPOAgent", "PPOConfig", "ActorCritic"]

    if name in _gan:
        from src.stealth import gan as _gan_mod
        return getattr(_gan_mod, name)
    if name in _rl:
        from src.stealth import rl as _rl_mod
        return getattr(_rl_mod, name)
    if name == "StealthScheduler":
        from src.stealth.stealth_scheduler import StealthScheduler
        return StealthScheduler
    raise AttributeError(f"module 'src.stealth' has no attribute {name!r}")
