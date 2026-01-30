"""
Database management for pitch interval storage and retrieval.

Provides persistence for the pitch interval database with support
for incremental updates and multiple storage backends.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class SongEntry:
    """Entry for a song in the database."""
    name: str
    intervals: list[float]
    source_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[dict] = None

    def __len__(self) -> int:
        return len(self.intervals)


class IntervalDatabase:
    """
    Database for storing and querying pitch interval sequences.

    Supports both pickle (legacy) and JSON storage formats.
    """

    def __init__(self, path: Optional[Path | str] = None):
        """
        Initialize the database.

        Args:
            path: Path to database file. If None, uses in-memory storage.
        """
        self.path = Path(path) if path else None
        self._data: dict[str, SongEntry] = {}
        self._dirty = False

        if self.path and self.path.exists():
            self.load()

    def __len__(self) -> int:
        """Return number of songs in database."""
        return len(self._data)

    def __contains__(self, song_name: str) -> bool:
        """Check if a song exists in database."""
        return song_name in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over song names."""
        return iter(self._data)

    def __getitem__(self, song_name: str) -> SongEntry:
        """Get a song entry by name."""
        return self._data[song_name]

    def add(
        self,
        name: str,
        intervals: list[float],
        source_file: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add or update a song in the database.

        Args:
            name: Song name (identifier).
            intervals: List of pitch intervals.
            source_file: Path to source audio file.
            duration_seconds: Duration of the song.
            metadata: Additional metadata.
        """
        self._data[name] = SongEntry(
            name=name,
            intervals=intervals,
            source_file=source_file,
            duration_seconds=duration_seconds,
            metadata=metadata,
        )
        self._dirty = True
        logger.info(f"Added song '{name}' with {len(intervals)} intervals")

    def remove(self, name: str) -> bool:
        """
        Remove a song from the database.

        Args:
            name: Song name to remove.

        Returns:
            True if song was removed, False if not found.
        """
        if name in self._data:
            del self._data[name]
            self._dirty = True
            logger.info(f"Removed song '{name}'")
            return True
        return False

    def get_intervals(self, name: str) -> list[float]:
        """
        Get intervals for a song.

        Args:
            name: Song name.

        Returns:
            List of pitch intervals.

        Raises:
            KeyError: If song not found.
        """
        return self._data[name].intervals

    def to_dict(self) -> dict[str, list[float]]:
        """
        Export database as simple dict (intervals only).

        Compatible with legacy code.

        Returns:
            Dictionary mapping song names to interval lists.
        """
        return {name: entry.intervals for name, entry in self._data.items()}

    def from_dict(self, data: dict[str, list[float]]) -> None:
        """
        Import from simple dict format.

        Args:
            data: Dictionary mapping song names to interval lists.
        """
        for name, intervals in data.items():
            self.add(name, intervals)

    def load(self, path: Optional[Path | str] = None) -> None:
        """
        Load database from file.

        Args:
            path: Path to load from. Uses self.path if None.
        """
        path = Path(path) if path else self.path
        if not path:
            raise ValueError("No path specified")

        if not path.exists():
            logger.warning(f"Database file not found: {path}")
            return

        if path.suffix == ".json":
            self._load_json(path)
        else:
            self._load_pickle(path)

        self._dirty = False
        logger.info(f"Loaded {len(self._data)} songs from {path}")

    def _load_pickle(self, path: Path) -> None:
        """Load from pickle format (legacy)."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Handle legacy format (simple dict)
        if isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                self.from_dict(data)
            else:
                # New format with SongEntry
                self._data = data

    def _load_json(self, path: Path) -> None:
        """Load from JSON format."""
        with open(path) as f:
            data = json.load(f)

        for name, entry_data in data.items():
            if isinstance(entry_data, list):
                # Simple format
                self.add(name, entry_data)
            else:
                # Full format
                self.add(
                    name=name,
                    intervals=entry_data["intervals"],
                    source_file=entry_data.get("source_file"),
                    duration_seconds=entry_data.get("duration_seconds"),
                    metadata=entry_data.get("metadata"),
                )

    def save(self, path: Optional[Path | str] = None, format: str = "auto") -> None:
        """
        Save database to file.

        Args:
            path: Path to save to. Uses self.path if None.
            format: "pickle", "json", or "auto" (based on extension).
        """
        path = Path(path) if path else self.path
        if not path:
            raise ValueError("No path specified")

        if format == "auto":
            format = "json" if path.suffix == ".json" else "pickle"

        if format == "json":
            self._save_json(path)
        else:
            self._save_pickle(path)

        self._dirty = False
        logger.info(f"Saved {len(self._data)} songs to {path}")

    def _save_pickle(self, path: Path) -> None:
        """Save to pickle format (legacy compatible)."""
        # Save in legacy format for compatibility
        data = self.to_dict()
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _save_json(self, path: Path) -> None:
        """Save to JSON format."""
        data = {}
        for name, entry in self._data.items():
            data[name] = {
                "intervals": entry.intervals,
                "source_file": entry.source_file,
                "duration_seconds": entry.duration_seconds,
                "metadata": entry.metadata,
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def list_songs(self) -> list[str]:
        """
        List all songs in database.

        Returns:
            Sorted list of song names.
        """
        return sorted(self._data.keys())

    def search(self, pattern: str) -> list[str]:
        """
        Search for songs by pattern.

        Args:
            pattern: Regex pattern to match against song names.

        Returns:
            List of matching song names.
        """
        regex = re.compile(pattern, re.IGNORECASE)
        return [name for name in self._data if regex.search(name)]

    def stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self._data:
            return {"count": 0}

        interval_counts = [len(entry.intervals) for entry in self._data.values()]
        return {
            "count": len(self._data),
            "total_intervals": sum(interval_counts),
            "avg_intervals_per_song": sum(interval_counts) / len(interval_counts),
            "min_intervals": min(interval_counts),
            "max_intervals": max(interval_counts),
        }


def extract_song_name(filename: str) -> str:
    """
    Extract song name from filename.

    Removes common suffixes like _vocal, _bg, etc.

    Args:
        filename: Filename to process.

    Returns:
        Extracted song name.
    """
    # Remove extension
    name = Path(filename).stem

    # Remove common suffixes
    suffixes = ["_vocal", "_vocals", "_bg", "_background", "_inst", "_instrumental"]
    for suffix in suffixes:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name


def get_background_filename(song_name: str, extension: str = ".mp3") -> str:
    """
    Get the expected background track filename for a song.

    Args:
        song_name: Base song name.
        extension: File extension.

    Returns:
        Background track filename.
    """
    # Handle song names that might already have suffixes
    base_name = extract_song_name(song_name)
    return f"{base_name}_bg{extension}"
