"""
Auto Accompaniment - Automatic music accompaniment system.

Identifies songs from vocal input and synchronizes background music playback.
"""

__version__ = "1.0.0"
__author__ = "Auto Accompaniment Team"

from auto_accompaniment.core.pitch import PitchDetector
from auto_accompaniment.core.matching import SequenceMatcher
from auto_accompaniment.core.database import IntervalDatabase

__all__ = ["PitchDetector", "SequenceMatcher", "IntervalDatabase"]
