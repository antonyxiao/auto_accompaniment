"""
Audio recording module for real-time microphone input.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from auto_accompaniment.core.pitch import PitchDetector, PitchSequence

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Records audio from microphone and extracts pitch in real-time.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 2048,
        channels: int = 1,
        device_index: Optional[int] = None,
    ):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Sample rate in Hz.
            block_size: Number of samples per block.
            channels: Number of audio channels.
            device_index: Input device index. None for default.
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.device_index = device_index

        self._pyaudio = None
        self._stream = None

    def _init_pyaudio(self):
        """Initialize PyAudio if not already done."""
        if self._pyaudio is None:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()

    def _cleanup_pyaudio(self):
        """Clean up PyAudio resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def record(
        self,
        duration_seconds: float,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Record audio from microphone.

        Args:
            duration_seconds: Duration to record.
            progress_callback: Optional callback with progress (0-1).

        Returns:
            Audio data as float32 numpy array.
        """
        import pyaudio

        self._init_pyaudio()

        frames = []
        total_frames = int(duration_seconds * self.sample_rate / self.block_size)

        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.block_size,
                input_device_index=self.device_index,
            )

            logger.info(
                f"Recording {duration_seconds}s from device {self.device_index}"
            )

            for i in range(total_frames):
                data = self._stream.read(self.block_size)
                frames.append(data)

                if progress_callback:
                    progress_callback((i + 1) / total_frames)

        finally:
            self._cleanup_pyaudio()

        # Convert to numpy array
        audio_data = np.frombuffer(b"".join(frames), dtype=np.float32)
        logger.info(f"Recorded {len(audio_data)} samples")

        return audio_data

    def record_with_pitch(
        self,
        duration_seconds: float,
        pitch_detector: PitchDetector,
        progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> PitchSequence:
        """
        Record audio and extract pitch in real-time.

        Args:
            duration_seconds: Duration to record.
            pitch_detector: PitchDetector instance.
            progress_callback: Optional callback with (progress, current_pitch).

        Returns:
            PitchSequence with detected pitches.
        """
        import pyaudio

        self._init_pyaudio()

        pitches = []
        confidences = []
        energies = []
        total_frames = int(duration_seconds * self.sample_rate / self.block_size)

        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.block_size,
                input_device_index=self.device_index,
            )

            logger.info(
                f"Recording and analyzing {duration_seconds}s from device {self.device_index}"
            )

            for i in range(total_frames):
                data = self._stream.read(self.block_size)
                samples = np.frombuffer(data, dtype=np.float32)

                result = pitch_detector.detect_pitch(samples)
                pitches.append(result.pitch)
                confidences.append(result.confidence)
                energies.append(result.energy)

                if progress_callback:
                    progress_callback((i + 1) / total_frames, result.pitch)

        finally:
            self._cleanup_pyaudio()

        return PitchSequence(
            pitches=pitches,
            confidences=confidences,
            energies=energies,
            sample_rate=self.sample_rate,
            block_size=self.block_size,
        )


def load_audio_file(
    path: str,
    sample_rate: int = 44100,
    mono: bool = True,
) -> np.ndarray:
    """
    Load audio from file using librosa.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate.
        mono: Whether to convert to mono.

    Returns:
        Audio data as float32 numpy array.
    """
    import librosa

    audio_data, sr = librosa.load(path, sr=sample_rate, mono=mono)
    logger.info(f"Loaded {path}: {len(audio_data)} samples at {sr}Hz")

    return audio_data.astype(np.float32)
