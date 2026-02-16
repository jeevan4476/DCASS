"""
DCASS Configuration System

Provides a singleton Config class that loads settings from YAML files
and allows override via environment variables.

Usage:
    from config.settings import config
    
    # Get values using dot notation
    model_name = config.get("embeddings.text.model")
    
    # Get as Path object
    data_dir = config.get_path("paths.data_dir")
    
    # Get with default value
    device = config.get("model.device", "cpu")
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml


class Config:
    """
    Singleton configuration manager.
    
    Loads configuration from YAML files and provides a simple API
    for accessing nested configuration values using dot notation.
    """
    
    _instance: Optional["Config"] = None
    _config: dict = {}
    _project_root: Path = Path.cwd()
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize configuration on first instantiation."""
        self._project_root = self._find_project_root()
        self._load_config()
    
    def _find_project_root(self) -> Path:
        """
        Find the project root directory.
        
        Looks for markers like .git, config/, or src/ directories.
        """
        # Start from this file's directory
        current = Path(__file__).parent
        
        # Go up until we find project markers
        for _ in range(5):  # Max 5 levels up
            if (current / ".git").exists() or (current / "src").exists():
                return current
            current = current.parent
        
        # Fallback to current working directory
        return Path.cwd()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = self._find_config_file()
        
        if config_path is not None and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            print("Warning: Config file not found, using defaults")
            self._config = {}
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find the configuration file in standard locations."""
        possible_paths = [
            self._project_root / "config" / "default.yaml",
            self._project_root / "config.yaml",
            Path.home() / ".dcass" / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Map of env vars to config paths
        env_mapping = {
            "DCASS_DEVICE": "model.device",
            "DCASS_LOG_LEVEL": "logging.level",
            "CUDA_VISIBLE_DEVICES": "model.cuda_devices",
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config_path, value)
    
    def _set_nested(self, key: str, value: Any) -> None:
        """Set a nested configuration value."""
        keys = key.split(".")
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "embeddings.text.model")
            default: Default value if key not found
            
        Returns:
            The configuration value or default
            
        Example:
            >>> config.get("embeddings.text.model")
            'all-MiniLM-L6-v2'
            >>> config.get("nonexistent.key", "default_value")
            'default_value'
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, key: str, absolute: bool = True) -> Path:
        """
        Get a configuration value as a Path object.
        
        Args:
            key: Configuration key for a path value
            absolute: If True, resolve relative paths against project root
            
        Returns:
            Path object
        """
        path_str = self.get(key, "")
        path = Path(path_str)
        
        if absolute and not path.is_absolute():
            path = self._project_root / path
        
        return path
    
    def get_device(self) -> str:
        """
        Get the compute device, resolving 'auto' to actual device.
        
        Returns:
            Device string: "cpu", "cuda", "cuda:0", etc.
        """
        device = self.get("model.device", "auto")
        
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        
        return device
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def __repr__(self) -> str:
        return f"Config(root={self._project_root})"


# Global config instance - import this in other modules
config = Config()
