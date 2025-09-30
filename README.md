# Chess-to-Music: Algorithmic Composition from Chess Games

A Python toolkit that transforms chess games into electronic music compositions using subtractive synthesis. Inspired by Laurie Spiegel's work on algorithmic composition, this project analyzes chess game narratives and translates them into evolving soundscapes through a three-layer synthesis architecture.

## Overview

This system converts chess games (PGN format) into audio (WAV) through a multi-stage pipeline:

1. **Analysis**: Extract game features, narratives, and key moments
2. **Synthesis**: Generate music using subtractive synthesis with Moog-style filters
3. **Output**: Direct-to-WAV audio file

The music reflects the drama of the game through:
- **Overall narrative arcs** (defeat, masterpiece, precision)
- **Section-level tension** (tactical chaos, king hunts, quiet positions)
- **Key moments** (brilliant moves, blunders, sacrifices)

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- `python-chess` - Chess game parsing
- `numpy` - Numerical computing
- `scipy` - Signal processing

### Basic Usage

```bash
# Full pipeline: PGN → Audio
./c2m path/to/game.pgn

# This runs:
# 1. thinking_time.py    - Add elapsed move time (EMT) to PGN
# 2. feature_extractor.py - Extract game features
# 3. tagger.py           - Generate narrative tags
# 4. synth_composer.py   - Synthesize audio
```

### Individual Pipeline Steps

```bash
# Step 1: Add timing data to PGN
python3 thinking_time.py data/game.pgn

# Step 2: Extract features
python3 feature_extractor.py data/game.pgn

# Step 3: Generate narrative tags
python3 tagger.py data/game-feat.json

# Step 4: Synthesize music
python3 synth_composer.py data/game-tags.json
# → Creates: chess_synth.wav
```

## Architecture

### Pipeline Components

```
thinking_time.py
    ├── Adds EMT (Elapsed Move Time) annotations to PGN
    └── Preserves all game data and comments

feature_extractor.py
    ├── Parses annotated PGN with python-chess
    ├── Extracts tactical features (checks, captures, sacrifices)
    ├── Analyzes position evaluation changes
    └── Outputs: *-feat.json

tagger.py
    ├── Reads feature JSON
    ├── Identifies narrative patterns
    ├── Segments game into dramatic sections
    ├── Tags key moments (brilliant, blunder, development)
    └── Outputs: *-tags.json

synth_composer.py (REFACTORED)
    ├── Three-layer synthesis architecture
    ├── Uses narrative tags to drive synthesis
    └── Outputs: chess_synth.wav
```

### Synthesis Architecture (NEW)

The synthesis engine has been **completely refactored** for clarity and maintainability:

```
synth_composer.py (687 lines) - Main composition logic
    └── ChessSynthComposer
        ├── Three-layer narrative system
        ├── Section composition
        └── Audio output

synth_config.py (664 lines) - All parameters centralized
    ├── Musical scales
    ├── Envelope presets
    ├── Narrative parameters
    ├── Section modulations
    ├── Moment voice parameters
    ├── Sequencer patterns
    └── Mixing levels

synth_engine.py (354 lines) - Pure synthesis engine
    └── SubtractiveSynth
        ├── Band-limited oscillators (PolyBLEP anti-aliasing)
        ├── Moog-style 4-pole low-pass filter
        ├── ADSR envelopes
        └── Roland JP-8000 style supersaw

synth_narrative.py (199 lines) - Narrative processes
    ├── TumblingDefeatProcess - Gradual deterioration
    ├── AttackingMasterpieceProcess - Building crescendo
    └── QuietPrecisionProcess - Equilibrium-seeking
```

**Total: ~1,900 lines** (down from 1,627 in a single file)

## Three-Layer Synthesis System

The music generation uses three simultaneous layers:

### Layer 1: Overall Narrative (Base Drone)
Sets the fundamental character for the entire game:
- **Tumbling Defeat**: Bright → dark, detuned supersaws, increasing chaos
- **Attacking Masterpiece**: Closed → open filters, building momentum
- **Peaceful Draw**: Stable parameters, gentle oscillation

### Layer 2: Section Narratives (Rhythmic Patterns)
Modulates the base sound for each game phase:
- **Tactical Chaos**: Fast tempo, high resonance, many notes
- **King Hunt**: Bright filters, dramatic sweeps, frantic density
- **Quiet Positional**: Slow, clean, sparse patterns
- **Desperate Defense**: Dark, slow, minimal filter movement

### Layer 3: Key Moments (Punctuation)
Adds musical accents for specific events:
- **Brilliant moves**: Rising filter sweeps, triumphant tones
- **Blunders**: Descending crashes, dissonant harmonies
- **Development**: Rising melodic phrases
- **First Exchange**: Question-answer call-and-response
- **Mate Sequence**: Dramatic finality (victory fanfare or death knell)

## Configuration & Tweaking

All musical parameters are now in **`synth_config.py`** for easy modification:

```python
from synth_config import SynthConfig

# Create custom config
config = SynthConfig()

# Modify parameters
config.MIXING['drone_level'] = 0.8  # Louder drone
config.TIMING['section_fade_sec'] = 0.2  # Longer fades
config.NARRATIVE_BASE_PARAMS['TUMBLING_DEFEAT']['filter_end'] = 500  # Less dark

# Use in composer
composer = ChessSynthComposer(tags, config=config)
composer.save()
```

### Key Configuration Sections

- **`SCALES`**: Musical scales (minor, phrygian, dorian)
- **`ENVELOPES`**: ADSR presets (percussive, pad, stab, etc.)
- **`NARRATIVE_BASE_PARAMS`**: Overall game character settings
- **`SECTION_MODULATIONS`**: Per-section parameter adjustments
- **`MOMENT_VOICES`**: Synthesis settings for key moments
- **`SEQUENCER_PATTERNS`**: 16-step MIDI patterns for Layer 3
- **`MIXING`**: Volume levels for all layers
- **`TIMING`**: Fade times, gaps, overlaps

## Testing the Synthesizer

```bash
# Test subtractive synthesis directly
python3 simple_synth_test.py

# This exercises:
# - Oscillators (saw, pulse, triangle, sine)
# - Moog filter with resonance
# - ADSR envelopes
# - Supersaw detuning
```

## Project Structure

```
├── c2m                      # Main pipeline script
├── thinking_time.py         # EMT annotation
├── feature_extractor.py     # Game feature extraction
├── tagger.py                # Narrative tagging
├── synth_composer.py        # Main composition (REFACTORED)
├── synth_config.py          # Configuration hub (NEW)
├── synth_engine.py          # Synthesis engine (NEW)
├── synth_narrative.py       # Narrative processes (NEW)
├── simple_synth_test.py     # Synth testing tool
├── data/                    # PGN files and generated output
├── openings/                # ECO opening database
├── REFACTORING_PLAN.md      # Refactoring documentation
└── composer_architecture.md # Technical architecture docs
```

## Example Output

```
♫ CHESS TO MUSIC SYNTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Game: 1-0 | ECO: A13 | Scale: Phrygian
Overall Narrative: DEATH_SPIRAL
Base Waveform: saw | Detune: 5→18 cents

Synthesizing 3 sections with 2.0s crossfades...

SECTION 1/3: OPENING (20s)
  Narrative: COMPLEX_STRUGGLE | Tension: 0.55
  Filter: 1620Hz | Resonance: 1.77
  Key Moments: 3
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (Markov chain)
  → Layer 3: Sequencer (3 key moments)
  ↓ Crossfading to MIDDLEGAME...

SECTION 2/3: MIDDLEGAME (36s)
  Narrative: CRUSHING_ATTACK | Tension: 0.53
  Filter: 770Hz | Resonance: 3.04
  Key Moments: 4
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)
  → Layer 3: Sequencer (4 key moments)
  ↓ Crossfading to ENDGAME...

SECTION 3/3: ENDGAME (21s)
  Narrative: CRUSHING_ATTACK | Tension: 0.73
  Filter: 280Hz | Resonance: 4.00
  Key Moments: 4
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)
  → Layer 3: Sequencer (4 key moments)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Synthesis complete: 73.0 seconds
  Audio Stats: Peak -8.4dB | No clipping
  Output: chess_synth.wav (6.1 MB)
```

## Musical Features

### Subtractive Synthesis
- **Band-limited oscillators**: PolyBLEP anti-aliasing for clean waveforms
- **Moog-style filter**: 4-pole ladder with resonance and soft clipping
- **Supersaw**: Roland JP-8000 style detuned saw ensemble
- **ADSR envelopes**: Exponential curves for natural sound

### Musical Mapping
- **Scales**: A minor, Phrygian (dark), Dorian (bright minor)
- **Waveforms**: Saw, pulse, triangle, square, sine
- **Filter ranges**: 20Hz - 20kHz with envelope modulation
- **Resonance**: 0.0 - 4.0 (up to self-oscillation)
- **Tempo**: Adaptive based on game narrative (0.7x - 1.3x)

### Spiegel-Inspired Processes
- **Entropy-based decay**: Mistakes accumulate, system becomes unstable
- **Momentum building**: Brilliant moves create positive feedback
- **Equilibrium-seeking**: Quiet games maintain balance with gentle breathing

## Benefits of Refactored Architecture

✓ **Clarity**: Clean separation of synthesis, configuration, and composition
✓ **Tweakability**: Change parameters in seconds without code search
✓ **Testability**: Import config, modify values, test instantly
✓ **Maintainability**: Each module has single responsibility
✓ **AI-Friendly**: Clear structure for experimentation

## Dependencies

### Core Requirements
- `python-chess >= 1.9.0` - Chess game parsing and analysis
- `numpy >= 1.20.0` - Array operations and DSP
- `scipy >= 1.7.0` - Signal processing (filters, waveforms)

### Python Version
- Python 3.8 or higher

## Technical Details

### Audio Specifications
- **Sample Rate**: 44.1 kHz
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Format**: WAV (uncompressed PCM)

### Synthesis Parameters
- **Filter**: State-variable 4-pole Moog ladder
- **Anti-aliasing**: PolyBLEP (Polynomial Band-Limited Edge Pulse)
- **Envelope curves**: Exponential (configurable curve factor)
- **Supersaw voices**: 7 detuned oscillators (configurable)
- **Pattern length**: 16 steps (sequencer layer)

## Inspired By

**Laurie Spiegel** - Pioneer of algorithmic composition and creator of Music Mouse, an intelligent instrument that understands musical context and responds to player intent.

> "I automate whatever can be automated to be freer to focus on those aspects of music that can't be automated. The challenge is to figure out which is which." - Laurie Spiegel

## License

See LICENSE file for details.

## Contributing

The refactored architecture makes contributions much easier:
- **Add new narrative types**: Edit `synth_config.py` → `NARRATIVE_BASE_PARAMS`
- **Create new processes**: Subclass `NarrativeProcess` in `synth_narrative.py`
- **Add moment types**: Edit `MOMENT_VOICES` in `synth_config.py`
- **Tweak synthesis**: Modify `SubtractiveSynth` in `synth_engine.py`

## Troubleshooting

### Import Errors
Ensure all modules are in the same directory:
```bash
ls synth_*.py
# Should show: synth_composer.py, synth_config.py, synth_engine.py, synth_narrative.py
```

### Audio Quality Issues
Adjust parameters in `synth_config.py`:
```python
config.MIXING['master_limiter'] = 0.8  # Reduce if clipping
config.MIXING['soft_clip_pre'] = 0.7   # More headroom
```

### Performance
For faster rendering, reduce:
- `note_density` in section modulations
- `filter_chunk_size_samples` in timing config
- Section durations in tagger output
