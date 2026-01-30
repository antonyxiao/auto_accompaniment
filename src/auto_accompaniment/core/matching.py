"""
Sequence matching algorithms for song identification.

Provides multiple matching strategies including simple subsequence matching,
Dynamic Time Warping (DTW), and normalized cross-correlation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MatchingMethod(Enum):
    """Available matching algorithms."""
    SIMPLE = "simple"  # Sum of absolute differences
    DTW = "dtw"  # Dynamic Time Warping
    NORMALIZED = "normalized"  # Normalized cross-correlation
    COMBINED = "combined"  # Combination of methods


@dataclass
class MatchResult:
    """Result of sequence matching."""
    song_name: str
    position: int  # Frame position in the song
    score: float  # Lower is better for distance-based, higher for correlation
    confidence: float  # 0-1 normalized confidence score
    time_offset_seconds: float  # Position in seconds

    def __str__(self) -> str:
        minutes = int(self.time_offset_seconds // 60)
        seconds = int(self.time_offset_seconds % 60)
        return (
            f"{self.song_name}: position={self.position}, "
            f"time={minutes}:{seconds:02d}, confidence={self.confidence:.2%}"
        )


class SequenceMatcherBase(ABC):
    """Abstract base class for sequence matchers."""

    @abstractmethod
    def match_subsequence(
        self,
        query: list[float],
        reference: list[float],
    ) -> tuple[int, float]:
        """
        Find the best matching position for query in reference.

        Args:
            query: Short query sequence.
            reference: Long reference sequence.

        Returns:
            Tuple of (best_position, score).
        """
        pass


class SimpleSequenceMatcher(SequenceMatcherBase):
    """
    Simple sum-of-absolute-differences matcher.

    Fast but sensitive to tempo variations.
    """

    def match_subsequence(
        self,
        query: list[float],
        reference: list[float],
    ) -> tuple[int, float]:
        """Find best match using sum of absolute differences."""
        query_len = len(query)
        if query_len > len(reference):
            return -1, float("inf")

        min_difference = float("inf")
        best_position = -1

        # Convert to numpy for vectorized operations
        query_arr = np.array(query, dtype=np.float32)
        ref_arr = np.array(reference, dtype=np.float32)

        # Sliding window comparison
        for i in range(len(reference) - query_len + 1):
            window = ref_arr[i : i + query_len]
            difference = np.sum(np.abs(query_arr - window))

            if difference < min_difference:
                min_difference = difference
                best_position = i

        return best_position, float(min_difference)


class DTWSequenceMatcher(SequenceMatcherBase):
    """
    Dynamic Time Warping based matcher.

    More robust to tempo variations but slower.
    """

    def __init__(self, window: Optional[int] = None):
        """
        Initialize DTW matcher.

        Args:
            window: Sakoe-Chiba band width for speedup. None for full DTW.
        """
        self.window = window
        self._dtw_available = False

        try:
            from dtaidistance import dtw
            self._dtw = dtw
            self._dtw_available = True
        except ImportError:
            logger.warning(
                "dtaidistance not available. Install with: pip install dtaidistance"
            )

    @property
    def is_available(self) -> bool:
        """Check if DTW is available."""
        return self._dtw_available

    def match_subsequence(
        self,
        query: list[float],
        reference: list[float],
    ) -> tuple[int, float]:
        """Find best match using DTW distance."""
        if not self._dtw_available:
            raise RuntimeError("dtaidistance not available")

        query_len = len(query)
        if query_len > len(reference):
            return -1, float("inf")

        query_arr = np.array(query, dtype=np.float64)
        ref_arr = np.array(reference, dtype=np.float64)

        best_distance = float("inf")
        best_position = -1

        # Sliding window with DTW
        for i in range(len(reference) - query_len + 1):
            window = ref_arr[i : i + query_len]
            distance = self._dtw.distance(
                query_arr,
                window,
                window=self.window,
            )

            if distance < best_distance:
                best_distance = distance
                best_position = i

        return best_position, float(best_distance)


class NormalizedMatcher(SequenceMatcherBase):
    """
    Normalized cross-correlation based matcher.

    Robust to amplitude variations.
    """

    def match_subsequence(
        self,
        query: list[float],
        reference: list[float],
    ) -> tuple[int, float]:
        """Find best match using normalized cross-correlation."""
        query_len = len(query)
        if query_len > len(reference):
            return -1, float("-inf")

        query_arr = np.array(query, dtype=np.float32)
        ref_arr = np.array(reference, dtype=np.float32)

        # Normalize query
        query_norm = query_arr - np.mean(query_arr)
        query_std = np.std(query_arr)
        if query_std > 0:
            query_norm = query_norm / query_std

        best_correlation = float("-inf")
        best_position = -1

        for i in range(len(reference) - query_len + 1):
            window = ref_arr[i : i + query_len]

            # Normalize window
            window_norm = window - np.mean(window)
            window_std = np.std(window)
            if window_std > 0:
                window_norm = window_norm / window_std

            # Cross-correlation
            correlation = float(np.sum(query_norm * window_norm) / query_len)

            if correlation > best_correlation:
                best_correlation = correlation
                best_position = i

        return best_position, best_correlation


class SequenceMatcher:
    """
    Main sequence matcher with multiple algorithm support.

    Provides a unified interface for song identification.
    """

    def __init__(
        self,
        method: MatchingMethod = MatchingMethod.SIMPLE,
        block_size: int = 2048,
        sample_rate: int = 44100,
        dtw_window: Optional[int] = None,
    ):
        """
        Initialize the sequence matcher.

        Args:
            method: Matching algorithm to use.
            block_size: Audio block size (for time calculations).
            sample_rate: Audio sample rate.
            dtw_window: DTW window size (if using DTW).
        """
        self.method = method
        self.block_size = block_size
        self.sample_rate = sample_rate

        # Initialize matchers
        self._simple = SimpleSequenceMatcher()
        self._dtw = DTWSequenceMatcher(window=dtw_window)
        self._normalized = NormalizedMatcher()

        logger.debug(f"Initialized SequenceMatcher with method={method.value}")

    def _get_matcher(self) -> SequenceMatcherBase:
        """Get the appropriate matcher for the current method."""
        if self.method == MatchingMethod.SIMPLE:
            return self._simple
        elif self.method == MatchingMethod.DTW:
            if not self._dtw.is_available:
                logger.warning("DTW not available, falling back to simple")
                return self._simple
            return self._dtw
        elif self.method == MatchingMethod.NORMALIZED:
            return self._normalized
        else:
            return self._simple

    def _position_to_time(self, position: int) -> float:
        """Convert frame position to time in seconds."""
        return position * self.block_size / self.sample_rate

    def _calculate_confidence(
        self,
        score: float,
        method: MatchingMethod,
        query_length: int,
    ) -> float:
        """
        Calculate a normalized confidence score.

        Args:
            score: Raw matching score.
            method: Matching method used.
            query_length: Length of query sequence.

        Returns:
            Confidence score between 0 and 1.
        """
        if method == MatchingMethod.NORMALIZED:
            # Correlation is already in [-1, 1], normalize to [0, 1]
            return (score + 1) / 2

        # For distance-based methods, use exponential decay
        # Normalize by query length to make it comparable
        normalized_score = score / max(query_length, 1)
        # Convert distance to similarity using exponential decay
        confidence = np.exp(-normalized_score / 100)
        return float(min(max(confidence, 0), 1))

    def match(
        self,
        query_intervals: list[float],
        reference_intervals: list[float],
        song_name: str = "unknown",
    ) -> MatchResult:
        """
        Match query intervals against a single reference.

        Args:
            query_intervals: Query pitch intervals.
            reference_intervals: Reference pitch intervals.
            song_name: Name of the reference song.

        Returns:
            MatchResult with position and score.
        """
        matcher = self._get_matcher()
        position, score = matcher.match_subsequence(
            query_intervals,
            reference_intervals,
        )

        confidence = self._calculate_confidence(
            score,
            self.method,
            len(query_intervals),
        )

        return MatchResult(
            song_name=song_name,
            position=position,
            score=score,
            confidence=confidence,
            time_offset_seconds=self._position_to_time(position),
        )

    def match_all(
        self,
        query_intervals: list[float],
        database: dict[str, list[float]],
    ) -> list[MatchResult]:
        """
        Match query against all songs in database.

        Args:
            query_intervals: Query pitch intervals.
            database: Dictionary mapping song names to interval sequences.

        Returns:
            List of MatchResult sorted by confidence (best first).
        """
        results = []

        for song_name, reference in database.items():
            result = self.match(query_intervals, reference, song_name)
            results.append(result)

        # Sort by confidence (highest first)
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def identify(
        self,
        query_intervals: list[float],
        database: dict[str, list[float]],
        min_confidence: float = 0.5,
    ) -> Optional[MatchResult]:
        """
        Identify the best matching song.

        Args:
            query_intervals: Query pitch intervals.
            database: Dictionary mapping song names to interval sequences.
            min_confidence: Minimum confidence threshold.

        Returns:
            Best MatchResult or None if no match meets threshold.
        """
        results = self.match_all(query_intervals, database)

        if not results:
            return None

        best = results[0]
        if best.confidence < min_confidence:
            logger.warning(
                f"Best match {best.song_name} has low confidence: {best.confidence:.2%}"
            )
            return None

        return best


class CombinedMatcher(SequenceMatcher):
    """
    Combines multiple matching methods for more robust identification.

    Uses voting or score averaging across methods.
    """

    def __init__(
        self,
        block_size: int = 2048,
        sample_rate: int = 44100,
        methods: Optional[list[MatchingMethod]] = None,
    ):
        """
        Initialize combined matcher.

        Args:
            block_size: Audio block size.
            sample_rate: Audio sample rate.
            methods: List of methods to combine. Defaults to SIMPLE and NORMALIZED.
        """
        super().__init__(
            method=MatchingMethod.COMBINED,
            block_size=block_size,
            sample_rate=sample_rate,
        )
        self.methods = methods or [MatchingMethod.SIMPLE, MatchingMethod.NORMALIZED]

    def match_all(
        self,
        query_intervals: list[float],
        database: dict[str, list[float]],
    ) -> list[MatchResult]:
        """
        Match using combined scores from multiple methods.

        Returns results sorted by average confidence.
        """
        # Collect results from each method
        method_results: dict[str, list[float]] = {name: [] for name in database}

        for method in self.methods:
            self.method = method
            results = super().match_all(query_intervals, database)

            for result in results:
                method_results[result.song_name].append(result.confidence)

        # Average confidences
        combined_results = []
        for song_name, confidences in method_results.items():
            avg_confidence = np.mean(confidences) if confidences else 0

            # Get position from simple matcher (fastest)
            self.method = MatchingMethod.SIMPLE
            single_result = self.match(
                query_intervals,
                database[song_name],
                song_name,
            )

            combined_results.append(
                MatchResult(
                    song_name=song_name,
                    position=single_result.position,
                    score=single_result.score,
                    confidence=float(avg_confidence),
                    time_offset_seconds=single_result.time_offset_seconds,
                )
            )

        combined_results.sort(key=lambda r: r.confidence, reverse=True)
        return combined_results
