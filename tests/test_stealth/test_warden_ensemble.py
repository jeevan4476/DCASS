import pytest
import torch

from src.analysis.adversarial.warden import WardenVerdict
from src.stealth.rl.warden_ensemble import average_warden_verdicts


def test_ensemble_averages_probabilities_and_anomaly_scores():
    first = WardenVerdict(
        bot_probability=torch.tensor([0.2, 0.8]),
        anomaly_scores=torch.tensor([[0.1, 0.3], [0.5, 0.7]]),
        feature_importance={},
    )
    second = WardenVerdict(
        bot_probability=torch.tensor([0.6, 0.4]),
        anomaly_scores=torch.tensor([[0.3, 0.5], [0.7, 0.9]]),
        feature_importance={},
    )

    result = average_warden_verdicts([first, second])

    assert torch.allclose(result.bot_probability, torch.tensor([0.4, 0.6]))
    assert torch.allclose(
        result.anomaly_scores,
        torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
    )


def test_ensemble_rejects_empty_or_shape_mismatched_verdicts():
    with pytest.raises(ValueError, match="at least one"):
        average_warden_verdicts([])

    first = WardenVerdict(torch.tensor([0.2]), torch.zeros(1, 2), {})
    second = WardenVerdict(torch.tensor([0.2]), torch.zeros(1, 3), {})
    with pytest.raises(ValueError, match="same shape"):
        average_warden_verdicts([first, second])
