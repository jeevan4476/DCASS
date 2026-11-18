"""Frozen detector ensembles for PPO reward training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ...analysis.adversarial.warden import WardenVerdict


def average_warden_verdicts(verdicts: Sequence[WardenVerdict]) -> WardenVerdict:
    """Average detector probabilities and anomaly scores with shape checks."""
    if not verdicts:
        raise ValueError("an ensemble must contain at least one verdict")
    bot_shape = verdicts[0].bot_probability.shape
    anomaly_shape = verdicts[0].anomaly_scores.shape
    if any(
        verdict.bot_probability.shape != bot_shape
        or verdict.anomaly_scores.shape != anomaly_shape
        for verdict in verdicts[1:]
    ):
        raise ValueError("all ensemble verdicts must have the same shape")

    feature_keys = set.intersection(
        *(set(verdict.feature_importance) for verdict in verdicts)
    ) if verdicts else set()
    features: dict[str, torch.Tensor] = {}
    for key in feature_keys:
        values = [verdict.feature_importance[key] for verdict in verdicts]
        if all(value.shape == values[0].shape for value in values):
            features[key] = torch.stack(values).mean(dim=0)

    return WardenVerdict(
        bot_probability=torch.stack([v.bot_probability for v in verdicts]).mean(dim=0),
        anomaly_scores=torch.stack([v.anomaly_scores for v in verdicts]).mean(dim=0),
        feature_importance=features,
    )


class WardenEnsemble(nn.Module):
    """Average the outputs of frozen, independently initialized detectors."""

    def __init__(self, models: Sequence[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("WardenEnsemble requires at least one model")
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> WardenVerdict:
        return average_warden_verdicts([model(*args, **kwargs) for model in self.models])
