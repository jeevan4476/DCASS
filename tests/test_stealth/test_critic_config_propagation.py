# tests/test_stealth/test_critic_config_propagation.py
"""FIX-E regression surface: the critic's scoring configuration travels
with the model through every checkpoint path.

A checkpoint whose critic was trained with a bounded score is not the
same model as one whose critic was not, even though the parameter
tensors have identical shapes: `bounded_critic` decides whether scores
are clipped and `critic_score_bound` decides where. The trainer's own
`save_checkpoint` records both keys, but

  * train_gan.py's CLI checkpoint writer kept its own dictionary without
    them, so every CLI-trained checkpoint silently forgot its critic
    configuration on the next load; and
  * every re-instantiator built a fresh Warden with default
    arguments, applied `load_state_dict`, and never restored the
    configuration — a bounded-critic checkpoint loaded as unbounded
    everywhere except the trainer itself.

These tests pin both ends: the CLI output carries the keys, and
`load_frozen_warden` (train_rl.py) restores the behaviour they describe.
A third test pins the shipped checkpoint's regression behavior: its keys
are absent, so loading leaves the untouched defaults bit-for-bit (no
fabricated restoration).
"""

import importlib.util
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_script_module(relative: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, _REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def test_cli_final_checkpoint_carries_critic_config(tmp_path):
    """train_gan.py's final write must carry the same two keys
    save_checkpoint does."""
    out = tmp_path / "cli_gan_generator.pt"
    import subprocess

    result = subprocess.run(
        [
            str(_REPO_ROOT / ".venv" / "bin" / "python"),
            str(_REPO_ROOT / "scripts" / "stealth" / "train_gan.py"),
            "--data", str(tmp_path / "missing_data.json"),  # synthetic fallback
            "--epochs", "1",
            "--batch-size", "500",  # synthetic fallback has 1000 samples: ~2 steps
            "--latent-dim", "16",
            "--hidden-dim", "32",
            "--max-seq-len", "20",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    checkpoint = torch.load(out, map_location="cpu", weights_only=False)
    assert "critic_bounded" in checkpoint
    assert "critic_score_bound" in checkpoint


def test_load_frozen_warden_restores_bounded_critic(tmp_path):
    """A bounded-critic checkpoint loaded through train_rl.py's
    load_frozen_warden must score bounded again."""
    module = _load_script_module(
        "scripts/stealth/train_rl.py", "_fixe_script_train_rl"
    )
    shipped = torch.load(
        _REPO_ROOT / "storage" / "models" / "gan_generator.pt",
        map_location="cpu",
        weights_only=False,
    )
    fixture = dict(shipped)
    fixture["critic_bounded"] = True
    fixture["critic_score_bound"] = 0.95
    fixture_path = tmp_path / "bounded_gan_generator.pt"
    torch.save(fixture, fixture_path)

    warden = module.load_frozen_warden(fixture_path, num_channels=3, device="cpu")
    assert warden.bounded_critic is True
    assert warden.critic_score_bound == 0.95


def test_shipped_checkpoint_loads_with_untouched_defaults(tmp_path):
    """The shipped gan_generator.pt predates the two keys: loading it
    must leave the constructor defaults untouched (no fabricated
    restoration) — exactly what apply_critic_config's absence behavior
    promises."""
    from src.analysis.adversarial.warden import CRITIC_SCORE_BOUND

    module = _load_script_module(
        "scripts/stealth/train_rl.py", "_fixe_script_train_rl"
    )
    warden = module.load_frozen_warden(
        _REPO_ROOT / "storage" / "models" / "gan_generator.pt",
        num_channels=3,
        device="cpu",
    )
    assert warden.bounded_critic is False
    assert warden.critic_score_bound == CRITIC_SCORE_BOUND
