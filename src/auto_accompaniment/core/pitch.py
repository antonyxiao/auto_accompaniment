"""
Pitch detection and interval extraction module.

Provides optimized pitch detection using Aubio with optional ML-based detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


class PitchAlgorithm(Enum):
    """Available pitch detection algorithms."""
    DEFAULT = "default"  # Aubio default (YIN-based)
    YIN = "yin"
    YINFFT = "yinfft"
    MCOMB = "mcomb"
    FCOMB = "fcomb"
    SCHMITT = "schmitt"
    SPECACF = "specacf"


@dataclass
class PitchResult:
    """Result of pitch detection for a single frame."""
    pitch: float  # Hz, 0 if silent
    confidence: float  # 0-1
    energy: float  # RMS energy


@dataclass
class PitchSequence:
    """A sequence of detected pitches with metadata."""
    pitches: list[float]
    confidences: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    sample_rate: int = 44100
    block_size: int = 2048

    @property
    def duration_seconds(self) -> float:
        """Calculate the duration of the sequence in seconds."""
        return len(self.pitches) * self.block_size / self.sample_rate

    def to_intervals(self, ignore_silence: bool = True) -> list[float]:
        """
        Convert pitch sequence to pitch intervals (differences).

        Args:
            ignore_silence: If True, skip intervals involving silent frames (pitch=0).

        Returns:
            List of pitch intervals in Hz.
        """
        intervals = []
        for i in range(1, len(self.pitches)):
            current = self.pitches[i]
            previous = self.pitches[i - 1]
            if ignore_silence and (current == 0 or previous == 0):
                continue
            intervals.append(current - previous)
        return intervals

    def to_midi_intervals(self, ignore_silence: bool = True) -> list[int]:
        """
        Convert pitch sequence to MIDI note intervals (semitones).

        More robust to octave errors and frequency variations.

        Args:
            ignore_silence: If True, skip intervals involving silent frames.

        Returns:
            List of pitch intervals in semitones.
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, falling back to Hz intervals")
            return [int(i) for i in self.to_intervals(ignore_silence)]

        intervals = []
        for i in range(1, len(self.pitches)):
            current = self.pitches[i]
            previous = self.pitches[i - 1]
            if ignore_silence and (current == 0 or previous == 0):
                continue
            if current > 0 and previous > 0:
                midi_current = librosa.hz_to_midi(current)
                midi_previous = librosa.hz_to_midi(previous)
                intervals.append(round(midi_current - midi_previous))
        return intervals


@runtime_checkable
class PitchDetectorProtocol(Protocol):
    """Protocol for pitch detectors."""

    def detect_pitch(self, audio_block: np.ndarray) -> PitchResult:
        """Detect pitch from a single audio block."""
        ...

    def process_audio(self, audio_data: np.ndarray) -> PitchSequence:
        """Process full audio data and extract pitches."""
        ...


class PitchDetector:
    """
    Optimized pitch detector using Aubio.

    Unlike the original implementation, this reuses the Aubio pitch detection
    object across frames for better performance.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 2048,
        algorithm: PitchAlgorithm = PitchAlgorithm.DEFAULT,
        silence_threshold: float = -40.0,
        unit: str = "Hz",
    ):
        """
        Initialize the pitch detector.

        Args:
            sample_rate: Audio sample rate in Hz.
            block_size: Number of samples per analysis frame.
            algorithm: Pitch detection algorithm to use.
            silence_threshold: Silence threshold in dB.
            unit: Pitch unit ("Hz", "midi", "cent").
        """
        import aubio

        self.sample_rate = sample_rate
        self.block_size = block_size
        self.algorithm = algorithm
        self.silence_threshold = silence_threshold
        self.unit = unit

        # Create reusable pitch detector
        self._detector = aubio.pitch(
            algorithm.value,
            block_size * 2,  # win_size
            block_size,  # hop_size
            sample_rate,
        )
        self._detector.set_unit(unit)
        self._detector.set_silence(silence_threshold)

        logger.debug(
            f"Initialized PitchDetector: sr={sample_rate}, "
            f"block={block_size}, algo={algorithm.value}"
        )

    def detect_pitch(self, audio_block: np.ndarray) -> PitchResult:
        """
        Detect pitch from a single audio block.

        Args:
            audio_block: Audio samples as float32 numpy array.

        Returns:
            PitchResult with pitch, confidence, and energy.
        """
        # Ensure correct type
        if audio_block.dtype != np.float32:
            audio_block = audio_block.astype(np.float32)

        # Pad if necessary
        if len(audio_block) < self.block_size:
            audio_block = np.pad(
                audio_block,
                (0, self.block_size - len(audio_block)),
                mode="constant",
            )

        pitch = float(self._detector(audio_block)[0])
        confidence = float(self._detector.get_confidence())
        energy = float(np.sqrt(np.mean(audio_block ** 2)))

        return PitchResult(pitch=pitch, confidence=confidence, energy=energy)

    def process_audio(
        self,
        audio_data: np.ndarray,
        min_confidence: float = 0.0,
    ) -> PitchSequence:
        """
        Process full audio data and extract pitch sequence.

        Args:
            audio_data: Full audio signal as numpy array.
            min_confidence: Minimum confidence threshold (0-1).

        Returns:
            PitchSequence containing all detected pitches.
        """
        # Ensure correct type
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        pitches = []
        confidences = []
        energies = []

        for i in range(0, len(audio_data), self.block_size):
            block = audio_data[i : i + self.block_size]
            result = self.detect_pitch(block)

            # Apply confidence threshold
            if result.confidence < min_confidence:
                pitches.append(0.0)
            else:
                pitches.append(result.pitch)

            confidences.append(result.confidence)
            energies.append(result.energy)

        return PitchSequence(
            pitches=pitches,
            confidences=confidences,
            energies=energies,
            sample_rate=self.sample_rate,
            block_size=self.block_size,
        )


class CREPEPitchDetector:
    """
    ML-based pitch detector using CREPE (if available).

    CREPE is a deep learning model for monophonic pitch tracking
    that is more accurate than traditional methods.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 2048,
        model_capacity: str = "tiny",  # tiny, small, medium, large, full
    ):
        """
        Initialize CREPE pitch detector.

        Args:
            sample_rate: Audio sample rate.
            block_size: Block size (for compatibility).
            model_capacity: CREPE model size.
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.model_capacity = model_capacity
        self._crepe_available = False

        try:
            import crepe
            self._crepe = crepe
            self._crepe_available = True
            logger.info(f"CREPE pitch detector initialized (model: {model_capacity})")
        except ImportError:
            logger.warning(
                "CREPE not available. Install with: pip install crepe tensorflow"
            )

    @property
    def is_available(self) -> bool:
        """Check if CREPE is available."""
        return self._crepe_available

    def process_audio(
        self,
        audio_data: np.ndarray,
        min_confidence: float = 0.5,
        step_size: int = 10,  # ms
    ) -> PitchSequence:
        """
        Process audio using CREPE.

        Args:
            audio_data: Audio signal.
            min_confidence: Minimum confidence threshold.
            step_size: Pitch estimation step size in ms.

        Returns:
            PitchSequence with detected pitches.
        """
        if not self._crepe_available:
            raise RuntimeError("CREPE not available")

        # CREPE expects mono audio
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        time, frequency, confidence, _ = self._crepe.predict(
            audio_data,
            self.sample_rate,
            model_capacity=self.model_capacity,
            step_size=step_size,
            viterbi=True,  # Apply Viterbi smoothing
        )

        # Apply confidence threshold
        pitches = [
            float(f) if c >= min_confidence else 0.0
            for f, c in zip(frequency, confidence)
        ]

        return PitchSequence(
            pitches=pitches,
            confidences=[float(c) for c in confidence],
            energies=[],  # CREPE doesn't provide energy
            sample_rate=self.sample_rate,
            block_size=int(step_size * self.sample_rate / 1000),
        )


def create_pitch_detector(
    method: str = "aubio",
    **kwargs,
) -> PitchDetector | CREPEPitchDetector:
    """
    Factory function to create a pitch detector.

    Args:
        method: Detection method ("aubio" or "crepe").
        **kwargs: Additional arguments passed to detector constructor.

    Returns:
        Pitch detector instance.
    """
    if method == "aubio":
        return PitchDetector(**kwargs)
    elif method == "crepe":
        detector = CREPEPitchDetector(**kwargs)
        if not detector.is_available:
            logger.warning("CREPE not available, falling back to Aubio")
            return PitchDetector(**kwargs)
        return detector
    else:
        raise ValueError(f"Unknown pitch detection method: {method}")
