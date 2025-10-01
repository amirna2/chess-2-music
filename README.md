# Chess-to-Music: Algorithmic Composition from Chess Games

A Python toolkit that transforms chess games into electronic music compositions using subtractive synthesis. Inspired by **Laurie Spiegel's work on algorithmic composition and information theory**, this project analyzes chess game narratives and translates them into evolving soundscapes through a three-layer synthesis architecture with **entropy-driven continuous evolution**.

## Overview

This system converts chess games (PGN format) into audio (WAV) through a multi-stage pipeline:

1. **Feature Extraction**: Parse PGN and extract move-by-move data with evaluations
2. **Narrative Analysis**: Identify game structure, narratives, and key moments
3. **Entropy Calculation**: Measure position complexity using information theory
4. **Synthesis**: Generate music using subtractive synthesis with entropy-driven evolution
5. **Output**: Direct-to-WAV audio file

The music reflects the drama of the game through:
- **Overall narrative arcs** (defeat, masterpiece, precision)
- **Section-level tension** (tactical chaos, king hunts, quiet positions)
- **Key moments** (brilliant moves, blunders, sacrifices)
- **Continuous entropy curves** (position complexity drives predictability)

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
# Step 1: Add timing data to PGN (if PGN has clock data)
python3 thinking_time.py data/game.pgn
# → Converts [%clk] annotations to [%emt] elapsed move times
# → This feeds into entropy calculation (thinking time = decision complexity)

# Step 2: Extract features (includes eval data for entropy)
python3 feature_extractor.py data/game.pgn --json > feat-game.json
# → Extracts eval_cp, is_capture, is_check, emt_seconds, etc.

# Step 3: Generate narrative tags (preserves moves for entropy calculation)
python3 tagger.py feat-game.json --output tags-game.json
# → Creates narrative structure + preserves move data automatically

# Step 4: Synthesize music with entropy-driven Layer 3
python3 synth_composer.py tags-game.json -o output.wav
# → Entropy curve controls note selection, rhythm, filters, harmonies
```

**Note**: `tagger.py` now automatically preserves move data from feature extraction, enabling entropy calculation during synthesis. No manual merge needed!

## Architecture

### Pipeline Components

```
thinking_time.py
    ├── Adds EMT (Elapsed Move Time) annotations to PGN
    ├── Converts [%clk] to [%emt] (thinking time per move)
    └── Preserves all game data and comments

feature_extractor.py
    ├── Parses annotated PGN with python-chess
    ├── Extracts tactical features (checks, captures, sacrifices)
    ├── Analyzes position evaluation changes (eval_cp, eval_mate)
    ├── Parses timing data (emt_seconds, clock_after_seconds)
    └── Outputs: *-feat.json (141 moves with full data)

tagger.py
    ├── Reads feature JSON
    ├── Identifies narrative patterns (DEATH_SPIRAL, ATTACKING_MASTERPIECE, etc.)
    ├── Segments game into dramatic sections (OPENING, MIDDLEGAME, ENDGAME)
    ├── Tags key moments (brilliant, blunder, development)
    ├── **Preserves move data for entropy calculation** (NEW!)
    └── Outputs: *-tags.json (narrative + moves)

entropy_calculator.py (NEW!)
    ├── Calculates informational entropy from position complexity
    ├── Eval volatility: Rolling std dev of eval_cp
    ├── Tactical density: Captures + checks frequency
    ├── Time pressure: Thinking time patterns
    └── Returns: entropy curve (0-1 per ply)

synth_composer.py (REFACTORED + ENTROPY)
    ├── Three-layer synthesis architecture
    ├── Calculates entropy curves per section
    ├── Uses narrative tags + entropy to drive synthesis
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

### Layer 3: Key Moments + Entropy (Punctuation & Evolution)
Adds musical accents for specific events **PLUS continuous entropy-driven evolution**:

**Key Moments (Discrete Events):**
- **Brilliant moves**: Rising filter sweeps, triumphant tones
- **Blunders**: Descending crashes, dissonant harmonies
- **Development**: Rising melodic phrases
- **First Exchange**: Question-answer call-and-response
- **Mate Sequence**: Dramatic finality (victory fanfare or death knell)

**Entropy Modulation (Continuous, Laurie Spiegel-inspired):**
- **Note Selection**: Entropy controls available note pool
  - Low entropy (< 0.3): Root-fifth only → predictable, stable
  - Medium (0.3-0.7): Diatonic scale → developing
  - High (> 0.7): Full chromatic → tense, unpredictable
- **Rhythm Variation**: High entropy adds ±50% timing irregularity
- **Portamento Control**: Low entropy = smooth glides, high = jumpy
- **Filter Modulation**: Entropy controls sweep speed
- **Harmonic Density**: High entropy adds random harmony notes

**Result**: Layer 3 now **breathes with position complexity** instead of just switching patterns at discrete events.

## Entropy-Driven Composition (Laurie Spiegel Approach)

> "The moment to moment variation of level of predictability that is embodied in an entropy curve arouses in the listener feelings of expectation, anticipation, satisfaction, disappointment, surprise, tension, frustration and other emotions." — Laurie Spiegel

### What is Entropy in This Context?

**Informational entropy** measures the **uncertainty/complexity** of a chess position:
- **High entropy**: Position is unclear, many possibilities, eval swinging wildly
- **Low entropy**: Position is simple, forced moves, eval stable

### How Entropy is Calculated

```python
# Three components, weighted combination:
entropy = eval_volatility * 0.5 +      # Rolling std dev of eval_cp
          tactical_density * 0.4 +     # Captures + checks frequency
          thinking_time * 0.1           # Long thinks = difficult position
```

### Musical Mapping

Entropy **directly controls predictability** in Layer 3:

| Position Type | Entropy | Musical Result |
|--------------|---------|----------------|
| **Theory moves** | 0.05-0.15 | Simple root-fifth drone, regular rhythm |
| **Quiet maneuvering** | 0.2-0.4 | Diatonic melody, slight variation |
| **Complex struggle** | 0.5-0.7 | Active patterns, moderate chromaticism |
| **Tactical chaos** | 0.8-1.0 | Full chromatic, irregular rhythm, dense harmonies |

### Example: Fischer-Taimanov Game 4

```
Opening (plies 1-20):   entropy = 0.168
  → Starts 0.08 (theory) → rises to 0.27 (exchanges) → drops to 0.08 (settled)
  → Music: Simple → Active → Simple

Middlegame (plies 29-45): entropy = 0.269
  → King hunt tactics, position unclear
  → Music: Chromatic, dense, tense

Endgame (plies 46-141):   entropy = 0.207
  → Technical conversion, some complexity remains
  → Music: Moderate activity, gradually simplifying
```

The eval volatility literally measures "how hard is this to understand" = perfect mapping to **musical uncertainty**!

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
├── thinking_time.py         # EMT annotation (clock → thinking time)
├── feature_extractor.py     # Game feature extraction
├── tagger.py                # Narrative tagging (preserves moves)
├── entropy_calculator.py    # Entropy calculation (NEW!)
├── synth_composer.py        # Main composition (REFACTORED + ENTROPY)
├── synth_config.py          # Configuration hub (includes ENTROPY_CONFIG)
├── synth_engine.py          # Synthesis engine
├── synth_narrative.py       # Narrative processes (Spiegel-inspired)
├── simple_synth_test.py     # Synth testing tool
├── pgn-examples/            # PGN files and generated output
├── openings/                # ECO opening database
├── ENTROPY_INTEGRATION.md   # Entropy system documentation (NEW!)
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
  Narrative: POSITIONAL_THEORY | Tension: 0.44
  Filter: 1980Hz | Resonance: 1.37
  Key Moments: 3
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (Positional opening: Methodical build)
  Entropy: mean=0.163, range=[0.063, 0.237]
  → Layer 3: Sequencer (3 key moments, entropy-driven)
  ↓ Crossfading to MIDDLEGAME...

SECTION 2/3: MIDDLEGAME (36s)
  Narrative: CRUSHING_ATTACK | Tension: 0.42
  Filter: 770Hz | Resonance: 2.88
  Key Moments: 4
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)
  Entropy: mean=0.130, range=[0.062, 0.268]
  → Layer 3: Sequencer (4 key moments, entropy-driven)
  ↓ Crossfading to ENDGAME...

SECTION 3/3: ENDGAME (21s)
  Narrative: CRUSHING_ATTACK | Tension: 0.73
  Filter: 280Hz | Resonance: 4.00
  Key Moments: 4
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)
  Entropy: mean=0.327, range=[0.201, 0.704]
  → Layer 3: Sequencer (4 key moments, entropy-driven)

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
