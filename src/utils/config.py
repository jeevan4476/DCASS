import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_path: str = "config/default.yaml"):
        # Resolve path relative to the project root
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_path = self.root_dir / config_path
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot notation (e.g., 'paths.raw_data')
        """
        keys = key_path.split(".")
        value = self.data
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    def get_path(self, key_path: str) -> Path:
        """Helper to get a config value as a full Path object"""
        val = self.get(key_path)
        return self.root_dir / val if val else None

# Singleton instance
config = Config()