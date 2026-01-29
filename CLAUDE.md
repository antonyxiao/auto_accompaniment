# CLAUDE.md - AI Assistant Guide

## Project Overview

**auto_accompaniment** is an automatic music accompaniment system that identifies songs from vocal input and synchronizes background music playback.

**Core Functionality:**
1. Records audio input from a microphone (or processes an audio file)
2. Detects pitch patterns using the Aubio library
3. Matches detected pitch intervals against a pre-built database of songs
4. Identifies the best matching song and the exact time offset
5. Automatically plays the corresponding background accompaniment from that position

## Repository Structure

```
auto_accompaniment/
├── extract_pitch.py    # Batch pitch extraction from audio files to build database
├── fingerprinter.py    # Creates audio fingerprints using Dejavu library
├── identifier.py       # Main song identification and playback orchestrator
├── match.py            # Core matching algorithm with pitch-based sequence matching
├── record_pitch.py     # Records live microphone input and extracts pitch intervals
├── mp3/                # Directory for source audio files (songs to index)
├── intervals.pkl       # Serialized pitch intervals database (generated)
└── CLAUDE.md           # This file
```

## Key Files and Their Purposes

| File | Purpose | Usage |
|------|---------|-------|
| `extract_pitch.py` | Processes all audio files in `mp3/` directory and extracts pitch intervals, saving to `intervals.pkl` | Run once to build/rebuild the pitch database |
| `fingerprinter.py` | Fingerprints audio files into MySQL database using Dejavu | Run once to initialize Dejavu fingerprint database |
| `identifier.py` | Main entry point - identifies song via mic/file and plays accompaniment | `python identifier.py` or `python identifier.py <audio_file>` |
| `match.py` | Performs pitch-based sequence matching against `intervals.pkl` | `python match.py` (mic) or `python match.py <audio_file>` |
| `record_pitch.py` | Records 8 seconds of microphone input and prints pitch intervals | Development/testing utility |

## Development Workflow

### Initial Setup

1. **Install Python dependencies** (no requirements.txt exists - install manually):
   ```bash
   pip install aubio librosa pyaudio numpy dejavu vlc-python dtaidistance wave
   ```

2. **Set up MySQL database**:
   ```sql
   CREATE DATABASE dejavu;
   ```

3. **Prepare audio files**:
   - Place source audio files (vocals) in `mp3/` directory
   - Place background/accompaniment files in the backgrounds directory

4. **Build databases**:
   ```bash
   python extract_pitch.py     # Creates intervals.pkl
   python fingerprinter.py     # Populates Dejavu database
   ```

### Running the System

**Using microphone input:**
```bash
python identifier.py    # Dejavu-based identification + playback
python match.py         # Pitch-based matching only
```

**Using audio file:**
```bash
python identifier.py <path/to/audio.wav>
python match.py <path/to/audio.wav>
```

## Important Configuration Values

### Constants (hardcoded in source files)

| Constant | Value | File | Description |
|----------|-------|------|-------------|
| `BLOCK_SIZE` | 2048 | multiple | Audio frame size for pitch analysis |
| `MIC_SECOND` / `MIC_LEN` | 10 | match.py, identifier.py | Recording duration in seconds |
| `PB_OFFSET` | 0.45 | identifier.py | Playback offset compensation in seconds |
| Sample rate | 44100 Hz | all | Standard CD-quality sampling rate |
| Silence threshold | -40 dB | multiple | Aubio pitch detection silence level |

### Paths and Credentials (MUST be updated for your environment)

**Background music directory** (`identifier.py:9`):
```python
BG = '/home/antonyxiao/dejavu/backgrounds/'  # Update this path!
```

**Database credentials** (`identifier.py:13-20`, `fingerprinter.py:3-10`):
```python
config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "passwd": "debang",  # Update password!
        "db": "dejavu",
    }
}
```

**Microphone device index** (varies by file):
- Device 4: Scarlett Solo (external audio interface)
- Device 18: Laptop microphone

To list available audio devices:
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(i, p.get_device_info_by_index(i)['name'])
```

## Code Architecture

### Audio Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                             │
├─────────────────────────────────────────────────────────────────────┤
│  Audio Files (mp3/wav/flac)                                         │
│         │                                                            │
│         ▼                                                            │
│  extract_pitch.py ──► Pitch Detection ──► Interval Conversion       │
│         │                                                            │
│         ▼                                                            │
│  intervals.pkl (serialized pitch interval database)                 │
│                                                                      │
│  fingerprinter.py ──► Dejavu ──► MySQL Database (fingerprints)      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         RECOGNITION PHASE                            │
├─────────────────────────────────────────────────────────────────────┤
│  Microphone / File Input                                            │
│         │                                                            │
│         ▼                                                            │
│  Pitch Detection (Aubio) ──► convert_to_intervals()                 │
│         │                                                            │
│         ▼                                                            │
│  simple_sequence_matching() ──► Compare against intervals.pkl       │
│         │                                                            │
│         ▼                                                            │
│  Best Match Song + Time Offset                                      │
│         │                                                            │
│         ▼                                                            │
│  VLC Playback (identifier.py) at synchronized position              │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Functions

**`get_pitch(audio_data, block_size)`** - Extracts pitch values from audio data using Aubio
- Location: `extract_pitch.py:12`, `match.py:16`
- Returns: List of pitch values in Hz

**`convert_to_intervals(pitches)`** - Converts pitch sequence to pitch differences
- Location: `extract_pitch.py:37`, `match.py:41`, `record_pitch.py:59`
- Returns: List of pitch intervals (excluding silences)

**`simple_sequence_matching(short_intervals, long_intervals)`** - Finds best match position
- Location: `match.py:50`
- Returns: `(best_position, min_difference)`

**`dtw_sequence_matching(short_intervals, long_intervals)`** - DTW-based matching (unused)
- Location: `match.py:65`
- Alternative matching algorithm using Dynamic Time Warping

## Coding Conventions

### Style
- **Naming**: snake_case for functions and variables
- **Constants**: UPPER_CASE at module level
- **Imports**: Standard library first, then third-party

### Patterns
- Main execution code runs at module level (no `if __name__ == "__main__":` guard)
- Commented-out code blocks contain alternative implementations for reference
- Functions defined before use in file

### Code Duplication
Note: `get_pitch()` and `convert_to_intervals()` are duplicated across files. When modifying these functions, update all occurrences:
- `extract_pitch.py`
- `match.py`
- `record_pitch.py`

## File Naming Conventions

**Audio files:**
- Vocal tracks: `<songname>_vocal.mp3` or just `<songname>.mp3`
- Background tracks: `<songname>_bg.mp3`

The system extracts the song name using regex and appends `_bg.mp3` to find the corresponding background file.

## Dependencies

### Python Packages
- `aubio` - Real-time audio analysis and pitch detection
- `librosa` - Audio loading and processing
- `pyaudio` - Audio I/O (requires PortAudio system library)
- `numpy` - Numerical operations
- `dejavu` - Audio fingerprinting library
- `vlc` - VLC media player Python bindings
- `dtaidistance` - Dynamic Time Warping (optional, for alternative matching)
- `wave` - WAV file handling (standard library)
- `pickle` - Object serialization (standard library)

### System Requirements
- MySQL server (for Dejavu database)
- PortAudio library (for PyAudio)
- VLC media player (for playback)
- Microphone input device

## Warnings and Known Issues

1. **Hardcoded credentials**: Database password is in source code - do not commit sensitive credentials
2. **Hardcoded paths**: Background music path is user-specific
3. **Device indices**: Microphone device index must be configured per system
4. **No error handling**: Files assume correct input and database connectivity
5. **No requirements.txt**: Dependencies must be installed manually
6. **Code duplication**: Core functions duplicated across files

## Quick Reference Commands

```bash
# Build pitch database from audio files
python extract_pitch.py

# Fingerprint audio files to Dejavu database
python fingerprinter.py

# Identify song from microphone (10 seconds)
python identifier.py

# Identify song from file
python identifier.py path/to/audio.wav

# Match against pitch database only (no playback)
python match.py
python match.py path/to/audio.wav

# Test microphone pitch recording (8 seconds)
python record_pitch.py
```
