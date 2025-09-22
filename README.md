# Chess-to-Music Converter

A Python tool that converts chess games in PGN format into MIDI music compositions. This project translates chess moves, piece types, game events, and move quality annotations into corresponding musical elements, inspired by Laurie Spiegel's pioneering work on algorithmic composition and her Music Mouse software.

## Features

- **Chess Game Processing**: Parses PGN files and analyzes moves using `python-chess`
- **MIDI Generation**: Creates multi-track MIDI files with `mido` (separate tracks for white/black players)
- **Opening Recognition**: Identifies chess openings using ECO codes from Lichess database
- **Musical Mapping**:
  - Chess pieces → MIDI instruments
  - Board squares → musical pitches
  - Move timing → note duration
  - Move quality (NAG codes) → harmonic variations
  - Game events (captures, checks, checkmate) → special musical effects
- **Audio Export**: Convert MIDI to MP3 via built-in utilities

## Quick Start

### Installation

```bash
# Install dependencies
make install

# Or with virtual environment
make setup
source venv/bin/activate
```

### Basic Usage

```bash
# Convert PGN to MIDI
python3 c2m.py data/game.pgn

# Use custom configuration
python3 c2m.py data/game.pgn --config my_config.yaml

# Specify output filename
python3 c2m.py data/game.pgn --output my_song.mid

# Convert MIDI to MP3
make to-mp3 MIDI=data/game.mid
```

### Using Make Commands

```bash
# See all available commands
make help

# Convert PGN to MIDI
make run PGN=data/game.pgn

# With custom config
make run PGN=data/game.pgn CONFIG=custom_config.yaml

# Run demo
make demo

# Convert MIDI to MP3
make to-mp3 MIDI=data/game.mid

# List available files
make list-games
make list-midi

# Clean up generated files
make clean        # Remove MIDI files only
make clean-all    # Remove all audio files
```

## CLI Options

```
usage: c2m.py [-h] [-c CONFIG] [-o OUTPUT] pgn_file

Convert chess games (PGN) to MIDI music files

positional arguments:
  pgn_file              Path to the PGN file to convert

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to configuration YAML file (default: config.yaml)
  -o OUTPUT, --output OUTPUT
                        Output MIDI file path (default: <pgn_file>.mid)
```

## Musical Mapping

### Piece to Instrument
- **Pawn (P)**: Piano (0)
- **Knight (N)**: Violin (40)
- **Bishop (B)**: Oboe (68)
- **Rook (R)**: French Horn (69)
- **Queen (Q)**: Piano (0)
- **King (K)**: Organ (19)

### Special Effects
- **Move Quality (NAG codes)**: Great moves add harmonies, mistakes add discordant notes
- **Captures**: Additional harmonic notes
- **Check**: Higher pitch accent
- **Checkmate**: C major chord
- **Tempo Changes**: Game progresses from fast (140 BPM) to slow (90 BPM)
- **Stereo Panning**: White pieces pan left, black pieces pan right

## Configuration

The tool uses a YAML configuration file (`config.yaml` by default) to customize:
- Instrument assignments
- Pitch mappings
- Musical effects
- Tempo settings
- Audio panning

See `config.yaml` for the complete configuration structure.

## Project Structure

```
├── c2m.py              # Main conversion script
├── config.yaml         # Default configuration
├── data/               # PGN files and generated output
├── openings/           # ECO opening database (a.tsv - e.tsv)
├── Makefile           # Build and utility commands
└── requirements.txt   # Python dependencies
```

## Dependencies

- `python-chess`: Chess game parsing and analysis
- `mido`: MIDI file generation
- `PyYAML`: Configuration file parsing

### External Tools (for audio conversion)
- `timidity`: MIDI to WAV conversion
- `lame`: WAV to MP3 encoding

## Examples

```bash
# Basic conversion
python3 c2m.py data/game.pgn
# → Creates data/game.mid

# Full pipeline to MP3
python3 c2m.py data/game.pgn
make to-mp3 MIDI=data/game.mid
# → Creates data/game.mp3

# Using Make for everything
make run PGN=data/game.pgn
make to-mp3 MIDI=data/game.mid
```

## License

See LICENSE file for details.
