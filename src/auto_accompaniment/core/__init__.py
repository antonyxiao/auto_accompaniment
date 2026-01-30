"""Core modules for audio processing and matching."""

from auto_accompaniment.core.pitch import PitchDetector
from auto_accompaniment.core.matching import SequenceMatcher
from auto_accompaniment.core.database import IntervalDatabase

__all__ = ["PitchDetector", "SequenceMatcher", "IntervalDatabase"]
