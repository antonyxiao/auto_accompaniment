"""
Configuration management for auto accompaniment system.

Supports environment variables, config files, and programmatic configuration.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """MySQL database configuration for Dejavu."""
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "dejavu"

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        """Load database config from environment variables."""
        return cls(
            host=os.getenv("AA_DB_HOST", "127.0.0.1"),
            port=int(os.getenv("AA_DB_PORT", "3306")),
            user=os.getenv("AA_DB_USER", "root"),
            password=os.getenv("AA_DB_PASSWORD", ""),
            database=os.getenv("AA_DB_NAME", "dejavu"),
        )

    def to_dejavu_config(self) -> dict:
        """Convert to Dejavu-compatible config dict."""
        return {
            "database": {
                "host": self.host,
                "user": self.user,
                "passwd": self.password,
                "db": self.database,
            }
        }


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    block_size: int = 2048
    channels: int = 1
    silence_threshold: float = -40.0
    input_device_index: Optional[int] = None
    recording_duration: int = 10

    @classmethod
    def from_env(cls) -> AudioConfig:
        """Load audio config from environment variables."""
        device_index = os.getenv("AA_AUDIO_DEVICE")
        return cls(
            sample_rate=int(os.getenv("AA_SAMPLE_RATE", "44100")),
            block_size=int(os.getenv("AA_BLOCK_SIZE", "2048")),
            channels=int(os.getenv("AA_CHANNELS", "1")),
            silence_threshold=float(os.getenv("AA_SILENCE_THRESHOLD", "-40.0")),
            input_device_index=int(device_index) if device_index else None,
            recording_duration=int(os.getenv("AA_RECORDING_DURATION", "10")),
        )


@dataclass
class PathConfig:
    """File and directory path configuration."""
    audio_dir: Path = field(default_factory=lambda: Path("mp3"))
    backgrounds_dir: Path = field(default_factory=lambda: Path("backgrounds"))
    intervals_file: Path = field(default_factory=lambda: Path("intervals.pkl"))

    @classmethod
    def from_env(cls) -> PathConfig:
        """Load path config from environment variables."""
        return cls(
            audio_dir=Path(os.getenv("AA_AUDIO_DIR", "mp3")),
            backgrounds_dir=Path(os.getenv("AA_BACKGROUNDS_DIR", "backgrounds")),
            intervals_file=Path(os.getenv("AA_INTERVALS_FILE", "intervals.pkl")),
        )


@dataclass
class PlaybackConfig:
    """Playback configuration."""
    offset_seconds: float = 0.45
    default_duration: int = 30

    @classmethod
    def from_env(cls) -> PlaybackConfig:
        """Load playback config from environment variables."""
        return cls(
            offset_seconds=float(os.getenv("AA_PLAYBACK_OFFSET", "0.45")),
            default_duration=int(os.getenv("AA_PLAYBACK_DURATION", "30")),
        )


@dataclass
class Config:
    """Main configuration container for auto accompaniment system."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    debug: bool = False

    @classmethod
    def from_env(cls) -> Config:
        """Load all configuration from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            audio=AudioConfig.from_env(),
            paths=PathConfig.from_env(),
            playback=PlaybackConfig.from_env(),
            debug=os.getenv("AA_DEBUG", "false").lower() == "true",
        )

    @classmethod
    def from_file(cls, path: Path | str) -> Config:
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls(
            database=DatabaseConfig(**data.get("database", {})),
            audio=AudioConfig(**data.get("audio", {})),
            paths=PathConfig(
                audio_dir=Path(data.get("paths", {}).get("audio_dir", "mp3")),
                backgrounds_dir=Path(data.get("paths", {}).get("backgrounds_dir", "backgrounds")),
                intervals_file=Path(data.get("paths", {}).get("intervals_file", "intervals.pkl")),
            ),
            playback=PlaybackConfig(**data.get("playback", {})),
            debug=data.get("debug", False),
        )

    def save(self, path: Path | str) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        data = {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "user": self.database.user,
                "password": self.database.password,
                "database": self.database.database,
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "block_size": self.audio.block_size,
                "channels": self.audio.channels,
                "silence_threshold": self.audio.silence_threshold,
                "input_device_index": self.audio.input_device_index,
                "recording_duration": self.audio.recording_duration,
            },
            "paths": {
                "audio_dir": str(self.paths.audio_dir),
                "backgrounds_dir": str(self.paths.backgrounds_dir),
                "intervals_file": str(self.paths.intervals_file),
            },
            "playback": {
                "offset_seconds": self.playback.offset_seconds,
                "default_duration": self.playback.default_duration,
            },
            "debug": self.debug,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Configuration saved to {path}")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try to load from file first, then environment
        config_path = Path(os.getenv("AA_CONFIG_FILE", "config.json"))
        if config_path.exists():
            _config = Config.from_file(config_path)
        else:
            _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
