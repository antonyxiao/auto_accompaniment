"""
Audio playback module using VLC.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Audio player using VLC for synchronized playback.
    """

    def __init__(self):
        """Initialize the audio player."""
        self._vlc = None
        self._instance = None
        self._player = None
        self._init_vlc()

    def _init_vlc(self):
        """Initialize VLC if available."""
        try:
            import vlc
            self._vlc = vlc
            self._instance = vlc.Instance()
            self._player = self._instance.media_player_new()
            logger.info("VLC initialized successfully")
        except ImportError:
            logger.error("VLC not available. Install with: pip install python-vlc")
        except Exception as e:
            logger.error(f"Failed to initialize VLC: {e}")

    @property
    def is_available(self) -> bool:
        """Check if VLC is available."""
        return self._vlc is not None and self._player is not None

    def play(
        self,
        file_path: Path | str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> bool:
        """
        Play an audio file from a specific position.

        Args:
            file_path: Path to audio file.
            start_time: Start position in seconds.
            duration: Optional duration to play. None for full file.

        Returns:
            True if playback started successfully.
        """
        if not self.is_available:
            logger.error("VLC not available")
            return False

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        try:
            media = self._instance.media_new(str(file_path))
            media.get_mrl()

            # Set start time option
            if start_time > 0:
                media.add_option(f"start-time={start_time}")

            # Set duration if specified
            if duration is not None:
                media.add_option(f"stop-time={start_time + duration}")

            self._player.set_media(media)
            self._player.play()

            logger.info(
                f"Playing {file_path.name} from {start_time:.2f}s"
                + (f" for {duration:.2f}s" if duration else "")
            )
            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False

    def stop(self) -> None:
        """Stop playback."""
        if self._player:
            self._player.stop()
            logger.info("Playback stopped")

    def pause(self) -> None:
        """Pause playback."""
        if self._player:
            self._player.pause()

    def resume(self) -> None:
        """Resume playback."""
        if self._player:
            self._player.play()

    def is_playing(self) -> bool:
        """Check if currently playing."""
        if self._player:
            return self._player.is_playing()
        return False

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if self._player:
            return self._player.get_time() / 1000.0
        return 0.0

    def set_volume(self, volume: int) -> None:
        """
        Set playback volume.

        Args:
            volume: Volume level (0-100).
        """
        if self._player:
            self._player.audio_set_volume(max(0, min(100, volume)))

    def wait_until_finished(self, timeout: Optional[float] = None) -> bool:
        """
        Block until playback finishes.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if playback finished, False if timeout.
        """
        if not self._player:
            return True

        start_time = time.time()
        while self._player.is_playing():
            time.sleep(0.1)
            if timeout and (time.time() - start_time) > timeout:
                return False
        return True


def play_accompaniment(
    song_name: str,
    start_time: float,
    backgrounds_dir: Path | str,
    recording_duration: float = 10.0,
    playback_offset: float = 0.45,
    duration: Optional[float] = None,
) -> bool:
    """
    Play the background track for a matched song.

    Args:
        song_name: Name of the matched song.
        start_time: Matched position in the song (seconds).
        backgrounds_dir: Directory containing background tracks.
        recording_duration: Duration of the original recording (seconds).
        playback_offset: Playback offset compensation (seconds).
        duration: Optional playback duration.

    Returns:
        True if playback started successfully.
    """
    from auto_accompaniment.core.database import get_background_filename

    backgrounds_dir = Path(backgrounds_dir)
    bg_filename = get_background_filename(song_name)
    bg_path = backgrounds_dir / bg_filename

    if not bg_path.exists():
        # Try without _bg suffix
        alt_path = backgrounds_dir / f"{song_name}.mp3"
        if alt_path.exists():
            bg_path = alt_path
        else:
            logger.error(f"Background track not found: {bg_path}")
            return False

    # Calculate actual start time
    actual_start = start_time + recording_duration + playback_offset

    player = AudioPlayer()
    return player.play(bg_path, start_time=actual_start, duration=duration)
