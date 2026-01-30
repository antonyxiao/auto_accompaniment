"""
Command-line interface for auto accompaniment system.

Provides commands for:
- Building the pitch database
- Identifying songs from microphone or file
- Playing accompaniment
- Managing configuration
- Listing audio devices
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from auto_accompaniment.utils.config import Config, get_config, set_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=False),
    help="Path to config file",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], debug: bool) -> None:
    """Auto Accompaniment - Automatic music accompaniment system."""
    ctx.ensure_object(dict)

    # Load configuration
    if config:
        cfg = Config.from_file(Path(config))
    else:
        cfg = get_config()

    cfg.debug = debug
    set_config(cfg)
    ctx.obj["config"] = cfg

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")


@cli.command()
@click.option(
    "--audio-dir",
    "-d",
    type=click.Path(exists=True),
    help="Directory containing audio files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="intervals.pkl",
    help="Output database file",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["aubio", "crepe"]),
    default="aubio",
    help="Pitch detection method",
)
@click.pass_context
def build(
    ctx: click.Context,
    audio_dir: Optional[str],
    output: str,
    method: str,
) -> None:
    """Build the pitch interval database from audio files."""
    import os
    import re

    from auto_accompaniment.core.database import IntervalDatabase, extract_song_name
    from auto_accompaniment.core.pitch import create_pitch_detector
    from auto_accompaniment.core.recorder import load_audio_file

    cfg = ctx.obj["config"]
    audio_dir = Path(audio_dir) if audio_dir else cfg.paths.audio_dir

    if not audio_dir.exists():
        click.echo(f"Error: Audio directory not found: {audio_dir}", err=True)
        sys.exit(1)

    click.echo(f"Building database from {audio_dir}")
    click.echo(f"Using {method} pitch detection")

    # Initialize pitch detector
    detector = create_pitch_detector(
        method=method,
        sample_rate=cfg.audio.sample_rate,
        block_size=cfg.audio.block_size,
    )

    # Initialize database
    db = IntervalDatabase(output)

    # Find audio files
    extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
    audio_files = [
        f for f in audio_dir.iterdir() if f.suffix.lower() in extensions
    ]

    if not audio_files:
        click.echo(f"No audio files found in {audio_dir}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(audio_files)} audio files")

    # Process each file
    with click.progressbar(audio_files, label="Processing") as files:
        for audio_file in files:
            try:
                # Load audio
                audio_data = load_audio_file(
                    str(audio_file),
                    sample_rate=cfg.audio.sample_rate,
                )

                # Extract pitches
                pitch_sequence = detector.process_audio(audio_data)

                # Convert to intervals
                intervals = pitch_sequence.to_intervals()

                # Extract song name
                song_name = extract_song_name(audio_file.name)

                # Add to database
                db.add(
                    name=song_name,
                    intervals=intervals,
                    source_file=str(audio_file),
                    duration_seconds=pitch_sequence.duration_seconds,
                )

            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")

    # Save database
    db.save()

    stats = db.stats()
    click.echo(f"\nDatabase built successfully:")
    click.echo(f"  Songs: {stats['count']}")
    click.echo(f"  Total intervals: {stats['total_intervals']}")
    click.echo(f"  Output: {output}")


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True), required=False)
@click.option(
    "--database",
    "-d",
    type=click.Path(exists=True),
    default="intervals.pkl",
    help="Path to intervals database",
)
@click.option(
    "--duration",
    "-t",
    type=float,
    default=10.0,
    help="Recording duration in seconds",
)
@click.option(
    "--device",
    "-i",
    type=int,
    help="Audio input device index",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["simple", "dtw", "normalized", "combined"]),
    default="simple",
    help="Matching method",
)
@click.option(
    "--play/--no-play",
    default=False,
    help="Play the matched accompaniment",
)
@click.option(
    "--backgrounds",
    "-b",
    type=click.Path(exists=True),
    help="Directory containing background tracks",
)
@click.pass_context
def identify(
    ctx: click.Context,
    audio_file: Optional[str],
    database: str,
    duration: float,
    device: Optional[int],
    method: str,
    play: bool,
    backgrounds: Optional[str],
) -> None:
    """Identify a song from microphone input or audio file."""
    from auto_accompaniment.core.database import IntervalDatabase
    from auto_accompaniment.core.matching import MatchingMethod, SequenceMatcher
    from auto_accompaniment.core.pitch import PitchDetector
    from auto_accompaniment.core.playback import play_accompaniment
    from auto_accompaniment.core.recorder import AudioRecorder, load_audio_file

    cfg = ctx.obj["config"]

    # Load database
    if not Path(database).exists():
        click.echo(f"Error: Database not found: {database}", err=True)
        click.echo("Run 'auto-accomp build' first to create the database.")
        sys.exit(1)

    db = IntervalDatabase(database)
    click.echo(f"Loaded database with {len(db)} songs")

    # Initialize pitch detector
    detector = PitchDetector(
        sample_rate=cfg.audio.sample_rate,
        block_size=cfg.audio.block_size,
    )

    # Get audio input
    if audio_file:
        click.echo(f"Analyzing file: {audio_file}")
        audio_data = load_audio_file(audio_file, sample_rate=cfg.audio.sample_rate)
        pitch_sequence = detector.process_audio(audio_data)
    else:
        device_idx = device if device is not None else cfg.audio.input_device_index
        click.echo(f"Recording {duration}s from device {device_idx}...")

        recorder = AudioRecorder(
            sample_rate=cfg.audio.sample_rate,
            block_size=cfg.audio.block_size,
            device_index=device_idx,
        )

        def progress_callback(progress: float, pitch: float):
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "=" * filled + "-" * (bar_width - filled)
            pitch_str = f"{pitch:.1f}Hz" if pitch > 0 else "---"
            click.echo(f"\r[{bar}] {progress*100:.0f}% {pitch_str}", nl=False)

        pitch_sequence = recorder.record_with_pitch(
            duration,
            detector,
            progress_callback=progress_callback,
        )
        click.echo()  # New line after progress

    # Convert to intervals
    intervals = pitch_sequence.to_intervals()
    click.echo(f"Extracted {len(intervals)} pitch intervals")

    if len(intervals) < 5:
        click.echo("Warning: Very few intervals detected. Try singing louder or longer.")

    # Match against database
    matching_method = MatchingMethod(method)
    matcher = SequenceMatcher(
        method=matching_method,
        block_size=cfg.audio.block_size,
        sample_rate=cfg.audio.sample_rate,
    )

    click.echo(f"\nMatching using {method} method...")
    results = matcher.match_all(intervals, db.to_dict())

    # Display results
    click.echo("\nResults:")
    click.echo("-" * 50)

    for i, result in enumerate(results[:5]):
        marker = ">>>" if i == 0 else "   "
        click.echo(
            f"{marker} {result.song_name}: "
            f"confidence={result.confidence:.1%}, "
            f"time={int(result.time_offset_seconds//60)}:"
            f"{int(result.time_offset_seconds%60):02d}"
        )

    if not results:
        click.echo("No matches found")
        sys.exit(1)

    best = results[0]
    click.echo(f"\nBest match: {best.song_name}")
    click.echo(f"Position: {best.time_offset_seconds:.2f}s")
    click.echo(f"Confidence: {best.confidence:.1%}")

    # Play accompaniment if requested
    if play:
        bg_dir = Path(backgrounds) if backgrounds else cfg.paths.backgrounds_dir
        if not bg_dir.exists():
            click.echo(f"Error: Backgrounds directory not found: {bg_dir}", err=True)
            sys.exit(1)

        click.echo(f"\nPlaying accompaniment...")
        success = play_accompaniment(
            song_name=best.song_name,
            start_time=best.time_offset_seconds,
            backgrounds_dir=bg_dir,
            recording_duration=duration,
            playback_offset=cfg.playback.offset_seconds,
            duration=cfg.playback.default_duration,
        )

        if success:
            click.echo("Press Ctrl+C to stop playback")
            try:
                import time
                time.sleep(cfg.playback.default_duration)
            except KeyboardInterrupt:
                click.echo("\nStopped")


@cli.command()
def devices() -> None:
    """List available audio input devices."""
    from auto_accompaniment.utils.audio import list_audio_devices

    devices = list_audio_devices()

    if not devices:
        click.echo("No audio devices found")
        return

    click.echo("Available audio devices:")
    click.echo("-" * 50)

    for device in devices:
        if device.is_input:
            click.echo(f"  [{device.index}] {device.name}")
            click.echo(f"      Sample rate: {device.default_sample_rate}Hz")


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.json",
    help="Output config file path",
)
@click.pass_context
def init_config(ctx: click.Context, output: str) -> None:
    """Generate a default configuration file."""
    cfg = Config()
    cfg.save(output)
    click.echo(f"Configuration saved to {output}")
    click.echo("Edit this file to customize settings.")


@cli.command()
@click.argument("database", type=click.Path(exists=True), default="intervals.pkl")
def stats(database: str) -> None:
    """Show database statistics."""
    from auto_accompaniment.core.database import IntervalDatabase

    db = IntervalDatabase(database)
    s = db.stats()

    click.echo(f"Database: {database}")
    click.echo("-" * 40)
    click.echo(f"Songs: {s['count']}")

    if s["count"] > 0:
        click.echo(f"Total intervals: {s['total_intervals']}")
        click.echo(f"Avg intervals/song: {s['avg_intervals_per_song']:.0f}")
        click.echo(f"Min intervals: {s['min_intervals']}")
        click.echo(f"Max intervals: {s['max_intervals']}")

        click.echo("\nSongs:")
        for song in db.list_songs():
            entry = db[song]
            click.echo(f"  - {song}: {len(entry)} intervals")


@cli.command()
@click.option(
    "--duration",
    "-t",
    type=float,
    default=8.0,
    help="Recording duration in seconds",
)
@click.option(
    "--device",
    "-i",
    type=int,
    help="Audio input device index",
)
@click.pass_context
def test_mic(ctx: click.Context, duration: float, device: Optional[int]) -> None:
    """Test microphone input and pitch detection."""
    from auto_accompaniment.core.pitch import PitchDetector
    from auto_accompaniment.core.recorder import AudioRecorder

    cfg = ctx.obj["config"]
    device_idx = device if device is not None else cfg.audio.input_device_index

    click.echo(f"Testing microphone (device {device_idx}) for {duration}s...")
    click.echo("Sing or play something!")
    click.echo("-" * 40)

    detector = PitchDetector(
        sample_rate=cfg.audio.sample_rate,
        block_size=cfg.audio.block_size,
    )

    recorder = AudioRecorder(
        sample_rate=cfg.audio.sample_rate,
        block_size=cfg.audio.block_size,
        device_index=device_idx,
    )

    def progress_callback(progress: float, pitch: float):
        if pitch > 0:
            note = pitch_to_note(pitch)
            click.echo(f"  {pitch:7.1f} Hz  ({note})")

    pitch_sequence = recorder.record_with_pitch(
        duration,
        detector,
        progress_callback=progress_callback,
    )

    intervals = pitch_sequence.to_intervals()
    click.echo("-" * 40)
    click.echo(f"Detected {len(pitch_sequence.pitches)} frames")
    click.echo(f"Non-silent: {sum(1 for p in pitch_sequence.pitches if p > 0)}")
    click.echo(f"Intervals: {len(intervals)}")


def pitch_to_note(frequency: float) -> str:
    """Convert frequency to musical note name."""
    if frequency <= 0:
        return "---"

    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    import math

    # A4 = 440 Hz
    semitones = 12 * math.log2(frequency / 440.0)
    note_index = int(round(semitones)) % 12
    octave = int(4 + (semitones + 9) // 12)

    return f"{notes[note_index]}{octave}"


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
