"""Public integration API: load a trained checkpoint once, then call
`.choose(paths)` on every routing decision. This is the class other
projects should import — it hides the DQNAgent/Config/observation plumbing
behind one method.
"""

from path_agent.agent import DQNAgent
from path_agent.config import Config
from path_agent.observation import build_observation


class PathSelector:
    def __init__(self, checkpoint_path: str | None = None, num_paths: int | None = None):
        """checkpoint_path: path to a .pt file saved by train.py. Defaults to
            this config's checkpoint_path (path_agent/runs/dqn.pt).
        num_paths: optional, only used to size internal buffers consistently;
                   the network itself works with any N at call time.
        """
        self.cfg = Config(num_paths=num_paths) if num_paths else Config()
        self.agent = DQNAgent(self.cfg)
        self.agent.load(checkpoint_path or self.cfg.checkpoint_path)

    def choose(self, paths: list[dict]) -> int:
        """paths: list of raw per-path measurement dicts, see observation.py
        for the required fields. Returns the index (into `paths`) of the
        path the agent recommends sending traffic on.
        """
        if len(paths) < 2:
            raise ValueError("need at least 2 paths to choose between")
        obs = build_observation(paths)
        return self.agent.act(obs, greedy=True)
