# Chess-to-Music Converter

A Python tool that converts chess games in PGN format into MIDI music compositions. This project translates chess moves, piece types, game events, and move quality annotations into corresponding musical elements, inspired by Laurie Spiegel's pioneering work on algorithmic composition and her Music Mouse software.

## Features

- **EMT-Aware Timing**: Uses per-move elapsed move time (EMT) to shape pre-move “thinking windows”
- **Windowed Drone Layer**: Expands / compresses a sustained drone section proportional to think time (log / pow curve)
- **Expression Modulation**: Classifies upcoming move (brilliant, blunder, check, forced, long, epic) to scale window & escalate modulation phases
- **Phase-Based Drone Dynamics**: Idle → swell → tension (tritone) → pulses for very long thinks
- **Short-Think Gating**: Ultra-fast moves ≤ 1s get a 1:1 micro-window; small thinks ≤ threshold stay flat (no jitter)
- **Chess Game Processing**: Parses PGN & annotations with `python-chess`
- **Opening Recognition**: ECO-based base drone pitch & harmonies
- **Musical Mapping**:
  - Pieces → configurable MIDI programs + automatic orchestral register shifts
  - Squares → pitch via selectable mapping modes
  - NAG & SAN markers → accents / effects
  - Captures, check, mate → dedicated musical gestures
- **Arpeggio Layer (optional)**: Legacy subtle background texture (disabled when drone modulation enabled)
- **Rich Move Log**: Tabular output with EMT, compressed window seconds, expression tag, timing ticks
- **Time Compression Summary**: Shows total EMT vs condensed musical window time & compression ratio
- **Audio Export**: Convert MIDI to MP3 via Makefile helpers

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

Current script flags (`python3 c2m.py --help`):

```
usage: c2m.py [-h] [--config CONFIG] [--output OUTPUT] [--play] [--sheet]
              [--no-move-log] [--track-stats] [--trace-arps]
              pgn_file

Convert a PGN with EMT annotations into an expressive multi-track MIDI: includes drone thinking windows, expression-based scaling, and timing summary.

positional arguments:
  pgn_file              Input PGN with EMT/time comments (mainline only)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration YAML (default: config.yaml)
  --output OUTPUT, -o OUTPUT
                        Explicit output MIDI path (default: <pgn_basename>_music.mid)
  --play                After render, attempt to play the MIDI file (system opener)
  --sheet               Print first measures as ASCII sheet at end
  --no-move-log         Suppress per-move tabular log
  --track-stats         Print per-track basic statistics summary
  --trace-arps          Verbose tracing for any (legacy) arpeggio generation
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
