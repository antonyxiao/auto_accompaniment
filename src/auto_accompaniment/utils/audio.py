"""
Audio device utilities and helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Represents an audio input/output device."""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_input: bool
    is_output: bool

    def __str__(self) -> str:
        device_type = []
        if self.is_input:
            device_type.append("input")
        if self.is_output:
            device_type.append("output")
        return f"[{self.index}] {self.name} ({', '.join(device_type)})"


def list_audio_devices() -> list[AudioDevice]:
    """
    List all available audio devices.

    Returns:
        List of AudioDevice objects representing available devices.
    """
    try:
        import pyaudio
    except ImportError:
        logger.error("PyAudio not installed. Run: pip install pyaudio")
        return []

    p = pyaudio.PyAudio()
    devices = []

    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            device = AudioDevice(
                index=i,
                name=info["name"],
                max_input_channels=int(info["maxInputChannels"]),
                max_output_channels=int(info["maxOutputChannels"]),
                default_sample_rate=float(info["defaultSampleRate"]),
                is_input=info["maxInputChannels"] > 0,
                is_output=info["maxOutputChannels"] > 0,
            )
            devices.append(device)
    finally:
        p.terminate()

    return devices


def get_default_input_device() -> Optional[int]:
    """
    Get the default input device index.

    Returns:
        Device index or None if no input device available.
    """
    try:
        import pyaudio
    except ImportError:
        logger.error("PyAudio not installed")
        return None

    p = pyaudio.PyAudio()
    try:
        info = p.get_default_input_device_info()
        return int(info["index"])
    except IOError:
        logger.warning("No default input device found")
        return None
    finally:
        p.terminate()


def validate_device(device_index: int, require_input: bool = True) -> bool:
    """
    Validate that a device exists and has the required capabilities.

    Args:
        device_index: The device index to validate.
        require_input: Whether the device must support input.

    Returns:
        True if device is valid, False otherwise.
    """
    devices = list_audio_devices()
    for device in devices:
        if device.index == device_index:
            if require_input and not device.is_input:
                logger.error(f"Device {device_index} does not support input")
                return False
            return True

    logger.error(f"Device {device_index} not found")
    return False


def format_time(seconds: float) -> str:
    """
    Format seconds as MM:SS string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 0:
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
