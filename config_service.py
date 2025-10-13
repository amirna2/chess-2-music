"""
Configuration Service - Simple YAML loader

Loads config.yaml and provides dot-notation access.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigService:
    """
    Simple YAML configuration loader with singleton pattern.

    Usage:
        config = get_config()
        value = config.get('synthesis.sample_rate', 44100)
    """

    _instance = None
    _config = None
    _config_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            project_root = Path(__file__).parent
            self._config_path = project_root / "config.yaml"

    def load(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from YAML file."""
        if config_path:
            self._config_path = Path(config_path)

        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")

        with open(self._config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def _ensure_loaded(self):
        """Ensure config is loaded before access."""
        if self._config is None:
            self.load()

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value using dot-notation path.

        Args:
            path: Dot-separated path (e.g., 'synthesis.sample_rate')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        self._ensure_loaded()

        keys = path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def raw(self) -> Dict[str, Any]:
        """Get raw config dictionary."""
        self._ensure_loaded()
        return self._config


# Global singleton
_config = ConfigService()


def get_config() -> ConfigService:
    """Get global config instance."""
    return _config
