# CLAUDE.md - AI Assistant Guide

## Project Overview

**auto_accompaniment** is an automatic music accompaniment system that identifies songs from vocal input and synchronizes background music playback.

**Core Functionality:**
1. Records audio input from a microphone (or processes an audio file)
2. Detects pitch patterns using Aubio or CREPE (ML-based)
3. Matches detected pitch intervals against a pre-built database of songs
4. Identifies the best matching song and the exact time offset
5. Automatically plays the corresponding background accompaniment from that position

## Repository Structure

```
auto_accompaniment/
├── src/
│   └── auto_accompaniment/          # Main Python package
│       ├── __init__.py              # Package initialization
│       ├── __main__.py              # Entry point for python -m
│       ├── core/                    # Core functionality modules
│       │   ├── __init__.py
│       │   ├── pitch.py             # Pitch detection (Aubio/CREPE)
│       │   ├── matching.py          # Sequence matching algorithms
│       │   ├── database.py          # Interval database management
│       │   ├── recorder.py          # Audio recording from microphone
│       │   └── playback.py          # VLC audio playback
│       ├── cli/                     # Command-line interface
│       │   ├── __init__.py
│       │   └── main.py              # Click-based CLI commands
│       └── utils/                   # Utility modules
│           ├── __init__.py
│           ├── config.py            # Configuration management
│           └── audio.py             # Audio device utilities
├── tests/                           # Unit tests
│   └── __init__.py
├── mp3/                             # Audio files to index (gitignored)
├── backgrounds/                     # Background/accompaniment tracks (gitignored)
├── intervals.pkl                    # Generated pitch database (gitignored)
│
├── # Legacy files (for reference)
├── extract_pitch.py                 # Original batch extraction script
├── fingerprinter.py                 # Original Dejavu fingerprinting
├── identifier.py                    # Original song identification
├── match.py                         # Original matching algorithm
├── record_pitch.py                  # Original mic recording
│
├── # Configuration & Build
├── pyproject.toml                   # Modern Python project config
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Development dependencies
├── config.example.json              # Example configuration file
├── .gitignore                       # Git ignore patterns
└── CLAUDE.md                        # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/antonyxiao/auto_accompaniment.git
cd auto_accompaniment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[full]"
```

### Basic Usage

```bash
# List available audio devices
auto-accomp devices

# Build the pitch database from audio files
auto-accomp build --audio-dir mp3/

# Identify a song from microphone (10 seconds)
auto-accomp identify --device 0

# Identify and play accompaniment
auto-accomp identify --play --backgrounds backgrounds/

# Identify from an audio file
auto-accomp identify song_sample.wav

# Test microphone input
auto-accomp test-mic --device 0

# Generate a config file
auto-accomp init-config
```

## Architecture

### Module Overview

| Module | Purpose |
|--------|---------|
| `core/pitch.py` | Pitch detection using Aubio (default) or CREPE (ML). Extracts pitch sequences and converts to intervals. |
| `core/matching.py` | Multiple matching algorithms: simple (fast), DTW (robust), normalized correlation, combined. |
| `core/database.py` | Manages the pitch interval database with pickle/JSON persistence. |
| `core/recorder.py` | Real-time audio recording from microphone with pitch extraction. |
| `core/playback.py` | VLC-based audio playback with time synchronization. |
| `cli/main.py` | Click-based CLI with commands: build, identify, devices, stats, test-mic. |
| `utils/config.py` | Configuration management via files or environment variables. |
| `utils/audio.py` | Audio device enumeration and utilities. |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BUILD PHASE                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Audio Files (mp3/wav/flac)                                         │
│         │                                                            │
│         ▼                                                            │
│  PitchDetector.process_audio() ──► PitchSequence                    │
│         │                                                            │
│         ▼                                                            │
│  PitchSequence.to_intervals() ──► intervals list                    │
│         │                                                            │
│         ▼                                                            │
│  IntervalDatabase.add() ──► intervals.pkl                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       IDENTIFY PHASE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Microphone / Audio File                                            │
│         │                                                            │
│         ▼                                                            │
│  AudioRecorder.record_with_pitch() ──► PitchSequence                │
│         │                                                            │
│         ▼                                                            │
│  PitchSequence.to_intervals() ──► query intervals                   │
│         │                                                            │
│         ▼                                                            │
│  SequenceMatcher.identify() ──► MatchResult                         │
│         │                          (song, position, confidence)     │
│         ▼                                                            │
│  play_accompaniment() ──► VLC playback at synced position           │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AA_CONFIG_FILE` | `config.json` | Path to config file |
| `AA_DB_HOST` | `127.0.0.1` | MySQL host for Dejavu |
| `AA_DB_PORT` | `3306` | MySQL port |
| `AA_DB_USER` | `root` | MySQL username |
| `AA_DB_PASSWORD` | `` | MySQL password |
| `AA_DB_NAME` | `dejavu` | MySQL database name |
| `AA_AUDIO_DEVICE` | auto | Input device index |
| `AA_SAMPLE_RATE` | `44100` | Audio sample rate |
| `AA_BLOCK_SIZE` | `2048` | Analysis block size |
| `AA_RECORDING_DURATION` | `10` | Default recording duration |
| `AA_AUDIO_DIR` | `mp3` | Audio files directory |
| `AA_BACKGROUNDS_DIR` | `backgrounds` | Background tracks directory |
| `AA_DEBUG` | `false` | Enable debug logging |

### Config File

Copy `config.example.json` to `config.json` and modify:

```json
{
  "database": {
    "host": "127.0.0.1",
    "user": "root",
    "password": "your_password",
    "database": "dejavu"
  },
  "audio": {
    "sample_rate": 44100,
    "block_size": 2048,
    "input_device_index": null,
    "recording_duration": 10
  },
  "paths": {
    "audio_dir": "mp3",
    "backgrounds_dir": "backgrounds"
  }
}
```

## Key Classes

### PitchDetector

```python
from auto_accompaniment.core.pitch import PitchDetector, PitchAlgorithm

# Create detector
detector = PitchDetector(
    sample_rate=44100,
    block_size=2048,
    algorithm=PitchAlgorithm.DEFAULT,
    silence_threshold=-40.0,
)

# Process audio
pitch_sequence = detector.process_audio(audio_data)
intervals = pitch_sequence.to_intervals()
```

### SequenceMatcher

```python
from auto_accompaniment.core.matching import SequenceMatcher, MatchingMethod

# Create matcher
matcher = SequenceMatcher(method=MatchingMethod.SIMPLE)

# Match against database
results = matcher.match_all(query_intervals, database)
best_match = results[0]  # Sorted by confidence

print(f"Song: {best_match.song_name}")
print(f"Position: {best_match.time_offset_seconds}s")
print(f"Confidence: {best_match.confidence:.1%}")
```

### IntervalDatabase

```python
from auto_accompaniment.core.database import IntervalDatabase

# Load or create database
db = IntervalDatabase("intervals.pkl")

# Add song
db.add(name="my_song", intervals=[1.2, -0.5, 0.8, ...])

# Query
intervals = db.get_intervals("my_song")

# Export for matching
data = db.to_dict()  # {name: intervals, ...}
```

## Matching Algorithms

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| `SIMPLE` | Fast | Good | Real-time, consistent tempo |
| `DTW` | Slow | Best | Variable tempo, rubato |
| `NORMALIZED` | Fast | Good | Volume variations |
| `COMBINED` | Medium | Better | General use |

## Dependencies

### Required
- `numpy>=1.21.0` - Numerical operations
- `aubio>=0.4.9` - Pitch detection
- `librosa>=0.9.0` - Audio loading
- `click>=8.0.0` - CLI framework

### Optional
- `pyaudio>=0.2.11` - Microphone input (requires PortAudio)
- `python-vlc>=3.0.0` - Playback (requires VLC)
- `dtaidistance>=2.0.0` - DTW matching
- `dejavu` - Audio fingerprinting (requires MySQL)
- `crepe` + `tensorflow` - ML-based pitch detection

## Coding Conventions

### Style
- Type hints on all public functions
- Dataclasses for data structures
- Protocols for interfaces
- snake_case for functions/variables
- UPPER_CASE for constants

### Patterns
- Factory functions for object creation (`create_pitch_detector()`)
- Context managers for resource cleanup
- Logging instead of print statements
- Configuration via dependency injection

### Error Handling
```python
# Use specific exceptions
class PitchDetectionError(Exception):
    pass

# Log errors, don't suppress
try:
    result = process_audio(data)
except Exception as e:
    logger.error(f"Failed to process audio: {e}")
    raise
```

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=auto_accompaniment

# Specific test file
pytest tests/test_pitch.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black src/

# Lint
ruff check src/

# Type check
mypy src/
```

## Migration from Legacy Code

The legacy scripts (`extract_pitch.py`, `match.py`, etc.) are preserved for reference. To migrate:

1. Replace direct `get_pitch()` calls with `PitchDetector.process_audio()`
2. Replace `simple_sequence_matching()` with `SequenceMatcher.match()`
3. Replace pickle dict loading with `IntervalDatabase`
4. Use `Config` instead of hardcoded values

## Warnings and Known Issues

1. **Audio Device Selection**: Device indices vary by system. Use `auto-accomp devices` to find the correct index.
2. **PyAudio Installation**: Requires PortAudio system library. On Ubuntu: `sudo apt install portaudio19-dev`
3. **VLC Requirement**: Background playback requires VLC media player installed.
4. **MySQL for Dejavu**: Fingerprinting feature requires MySQL database setup.

## Quick Reference Commands

```bash
# Build database
auto-accomp build -d mp3/

# Identify from microphone
auto-accomp identify -i 0

# Identify with playback
auto-accomp identify -i 0 --play -b backgrounds/

# Identify from file
auto-accomp identify recording.wav

# Use DTW matching (more accurate)
auto-accomp identify -m dtw

# Show database stats
auto-accomp stats intervals.pkl

# Test microphone
auto-accomp test-mic -i 0 -t 5

# Generate config file
auto-accomp init-config -o config.json
```
