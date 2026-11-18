import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analysis/train_traffic_detector.py"
SPEC = importlib.util.spec_from_file_location("train_traffic_detector", SCRIPT)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def test_distinct_detector_seeds_can_share_the_same_contributor_split():
    sessions = [
        {"contributor_id": f"user-{index}", "delays": [1.0], "channels": [index % 3]}
        for index in range(40)
    ]

    first = train.split_human_sessions_with_seed(sessions, model_seed=42, split_seed=2026)
    second = train.split_human_sessions_with_seed(sessions, model_seed=1337, split_seed=2026)

    assert first == second
    groups = [
        {sessions[index]["contributor_id"] for index in indices}
        for indices in first
    ]
    assert all(not (left & right) for i, left in enumerate(groups) for right in groups[i + 1 :])
