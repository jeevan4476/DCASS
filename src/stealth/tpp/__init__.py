"""Conditional Autoregressive Log-Normal Mixture Temporal Point Process (TPP).

Replaces deterministic-latent WGAN-GP timing synthesis with an intensity-free
autoregressive Log-Normal Mixture GRU (Shchur et al., ICLR 2020) trained by
exact maximum likelihood on (log_delay_t, channel_t | history, time_of_day).
"""

from .model import TPPConfig, TemporalPointProcessGenerator
from .trainer import TPPTrainingConfig, TPPTrainer, train_tpp_model

__all__ = [
    "TPPConfig",
    "TemporalPointProcessGenerator",
    "TPPTrainingConfig",
    "TPPTrainer",
    "train_tpp_model",
]
