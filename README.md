# Chess-to-Music: Algorithmic Composition from Chess Games

A Python toolkit that transforms chess games into electronic music compositions using subtractive synthesis. Inspired by **Laurie Spiegel's work on algorithmic composition and information theory**, this project analyzes chess game narratives and translates them into evolving soundscapes through a four-layer synthesis architecture with **gesture archetypes** and **entropy-driven continuous evolution**.

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

### Synthesis Architecture

The synthesis engine uses a modular architecture:

```
synth_composer.py - Main composition orchestrator
    └── ChessSynthComposer
        ├── Four-layer synthesis system
        ├── Section composition
        └── Audio output

synth_composer/ - Modular pattern generation package
    ├── coordinator.py - Pattern coordination
    ├── core/
    │   ├── note_event.py - Note event representation
    │   ├── timing_engine.py - Timing and rhythm
    │   ├── audio_buffer.py - Audio buffer management
    │   └── synthesizer.py - Pattern synthesis
    └── patterns/
        ├── base.py - Base pattern classes
        ├── markov.py - Markov chain patterns
        ├── state_machine.py - State machine patterns
        ├── theory.py - Music theory patterns
        ├── conversion.py - Pattern conversion utilities
        └── outro.py - Outro pattern generation

synth_config.py - All parameters centralized
    ├── Musical scales
    ├── Envelope presets
    ├── Narrative parameters
    ├── Section modulations
    ├── Moment voice parameters
    └── Mixing levels

synth_engine.py - Pure synthesis engine (shared by all layers)
    └── SubtractiveSynth
        ├── Band-limited oscillators (PolyBLEP anti-aliasing)
        ├── Moog-style 4-pole low-pass filter
        ├── ADSR envelopes
        ├── Time-varying filter support
        ├── Noise generators
        └── Roland JP-8000 style supersaw

layer3b/ - Gesture archetype system
    ├── archetype_configs.py - All archetype definitions
    ├── base.py - Base gesture generator
    ├── coordinator.py - Gesture coordinator
    ├── curve_generators.py - Pitch/harmony/filter/envelope curves
    ├── particle_system.py - Particle-based gestures
    ├── synthesizer.py - GestureSynthesizer (wraps SubtractiveSynth)
    └── utils.py - Utility functions

synth_narrative.py - Legacy narrative processes
    ├── TumblingDefeatProcess - Gradual deterioration
    ├── AttackingMasterpieceProcess - Building crescendo
    └── QuietPrecisionProcess - Equilibrium-seeking

Note: NoteSynthesizer (Layer 2) and GestureSynthesizer (Layer 3b) are
high-level wrappers around SubtractiveSynth. All DSP is implemented once
in SubtractiveSynth and shared across all layers.
```

## Four-Layer Synthesis System

The music generation uses four simultaneous layers:

### Layer 1: Overall Narrative (Base Drone)
Sets the fundamental character for the entire game:
- **Tumbling Defeat**: Bright → dark, detuned supersaws, increasing chaos
- **Attacking Masterpiece**: Closed → open filters, building momentum
- **Peaceful Draw**: Stable parameters, gentle oscillation

### Layer 2: Section Narratives (Rhythmic Patterns)
Modulates the base sound for each game phase using adaptive pattern generation:
- **Markov Chains**: Probabilistic note sequences trained on section characteristics
- **State Machines**: Multi-state patterns (e.g., ADVANCE → STRIKE → OVERWHELM)
- **Theory Patterns**: Opening theory, methodical development, conversion
- **Outro Patterns**: Final resolution and fadeout

Section narratives include:
- **Tactical Chaos**: Fast tempo, high resonance, dense note patterns
- **King Hunt**: Bright filters, dramatic sweeps, frantic density
- **Quiet Positional**: Slow, clean, sparse patterns
- **Desperate Defense**: Dark, slow, minimal filter movement

### Layer 3a: Heartbeat (Rhythmic Pulse)
Rhythmic heartbeat motif tied to game tension:
- Pulse rate varies with section tension
- Adaptive tempo and intensity
- Provides rhythmic foundation

### Layer 3b: Gestures (Key Moment Archetypes)
Spectromorphological gestures for chess moments using two approaches:

#### Curve-Based Gestures
Deterministic, expressive gestures with full parameter control:
- **Pitch curves**: Stable, glissando, tremor, drift, leap trajectories
- **Harmony**: Unison, clusters, chords, converging voices
- **Filter curves**: Sweeps, focus, opening/closing spectral shapes
- **Envelope**: Attack-Decay, Graduated Continuant, etc.
- **Texture**: Noise ratio, shimmer, waveform selection
- Examples: MOVE, GAME_CHANGING, BRILLIANT, BLUNDER, CHECKMATE

#### Particle-Based Gestures
Stochastic, polyphonic gestures with emergent behavior:
- **Emission curves**: Control particle spawning density over time
- **Particle properties**: Independent pitch, velocity, lifetime, detune
- **Emission patterns**: Gusts, bursts, drifts, rhythmic clusters, dissolves
- Examples: INACCURACY, FIRST_EXCHANGE, TACTICAL_SEQUENCE, FINAL_RESOLUTION

All gestures:
- Use spectromorphological classifications (Attack-Decay, Graduated Continuant, etc.)
- Have phase structure: pre_shadow, impact, bloom, decay, residue
- Respond to section context (tension, entropy scaling)

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

Entropy influences multiple aspects of the composition:

**Layer 2 (Pattern Generation):**
- Controls note pool selection and pattern density
- Influences rhythm regularity and timing variations
- Modulates filter movement speed

**Layer 3b (Gestures):**
- Scales gesture intensity and complexity
- Influences particle emission rates
- Controls spectral evolution characteristics

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

All musical parameters are centralized in **`synth_config.py`** for easy modification:

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
- **`MIXING`**: Volume levels for all layers
- **`TIMING`**: Fade times, gaps, overlaps

### Layer 3b Gesture Configuration

Gesture archetypes are defined in **`layer3b/archetype_configs.py`**:

```python
from layer3b.archetype_configs import CURVE_ARCHETYPES, PARTICLE_ARCHETYPES

# All curve-based gesture archetypes
print(CURVE_ARCHETYPES.keys())

# All particle-based gesture archetypes
print(PARTICLE_ARCHETYPES.keys())
```

Each archetype defines:
- Spectromorphological classification
- Phase structure (pre_shadow, impact, bloom, decay, residue)
- Pitch, harmony, filter, envelope, and texture curves
- Context-responsive scaling based on tension and entropy

## Testing the Synthesizer

### Basic Synthesis Tests
```bash
# Test subtractive synthesis directly
python3 tools/simple_synth_test.py

# This exercises:
# - Oscillators (saw, pulse, triangle, sine)
# - Moog filter with resonance
# - ADSR envelopes
# - Supersaw detuning
```

### Layer 3b Gesture Tests

#### Curve-Based Gestures
```bash
# List all available curve-based gesture archetypes
python3 tools/gesture_test.py --list

# Test specific archetype
python3 tools/gesture_test.py BRILLIANT

# Test with custom tension
python3 tools/gesture_test.py CHECKMATE --tension 0.9

# Test with custom entropy
python3 tools/gesture_test.py GAME_CHANGING --entropy 0.7
```

#### Particle-Based Gestures
```bash
# List all available particle-based gesture archetypes
python3 tools/particle_test.py --list

# Generate and visualize particle emission
python3 tools/particle_test.py INACCURACY

# Generate audio output
python3 tools/particle_test.py FIRST_EXCHANGE --audio

# Save to custom file
python3 tools/particle_test.py TACTICAL_SEQUENCE --audio -o test.wav
```

### Other Tools
```bash
# Design ADSR envelopes
python3 tools/adsr_gen.py

# Design heartbeat rhythms
python3 tools/heartbeat_designer.py
```

## Project Structure

```
├── c2m                          # Main pipeline script
├── thinking_time.py             # EMT annotation (clock → thinking time)
├── feature_extractor.py         # Game feature extraction
├── tagger.py                    # Narrative tagging (preserves moves)
├── entropy_calculator.py        # Entropy calculation
├── synth_composer.py            # Main composition orchestrator
├── synth_composer/              # Modular pattern generation package
│   ├── __init__.py
│   ├── coordinator.py           # Pattern coordination
│   ├── core/                    # Core synthesis components
│   │   ├── note_event.py        # Note event representation
│   │   ├── timing_engine.py     # Timing and rhythm engine
│   │   ├── audio_buffer.py      # Audio buffer management
│   │   └── synthesizer.py       # Pattern synthesis
│   └── patterns/                # Pattern generation modules
│       ├── base.py              # Base pattern classes
│       ├── markov.py            # Markov chain patterns
│       ├── state_machine.py     # State machine patterns
│       ├── theory.py            # Music theory patterns
│       ├── conversion.py        # Pattern conversion utilities
│       └── outro.py             # Outro generation
├── synth_config.py              # Configuration hub
├── synth_engine.py              # Subtractive synthesis engine
├── synth_narrative.py           # Legacy narrative processes
├── layer3b/                     # Gesture archetype system
│   ├── __init__.py
│   ├── archetype_configs.py     # All gesture archetype definitions
│   ├── base.py                  # Base gesture generator
│   ├── coordinator.py           # Gesture coordinator
│   ├── curve_generators.py      # Pitch/harmony/filter/envelope curves
│   ├── particle_system.py       # Particle-based gesture system
│   ├── synthesizer.py           # GestureSynthesizer (wraps SubtractiveSynth)
│   └── utils.py                 # Utility functions
├── tools/                       # Testing and development utilities
│   ├── gesture_test.py          # Test curve-based gestures
│   ├── particle_test.py         # Test particle-based gestures
│   ├── adsr_gen.py              # ADSR envelope designer
│   ├── heartbeat_designer.py    # Heartbeat motif designer
│   └── simple_synth_test.py     # Basic synthesis tests
├── tests/                       # Unit and integration tests
├── data/                        # PGN files and generated audio/tags
├── openings/                    # ECO opening database (Lichess TSV)
└── docs/                        # Documentation
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
  → Layer 3a: Heartbeat (adaptive pulse)
  → Layer 3b: Gestures (3 key moments, entropy-scaled)
  ↓ Crossfading to MIDDLEGAME...

SECTION 2/3: MIDDLEGAME (36s)
  Narrative: CRUSHING_ATTACK | Tension: 0.42
  Filter: 770Hz | Resonance: 2.88
  Key Moments: 4
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)
  Entropy: mean=0.130, range=[0.062, 0.268]
  → Layer 3a: Heartbeat (adaptive pulse)
  → Layer 3b: Gestures (4 key moments, entropy-scaled)
  ↓ Crossfading to ENDGAME...

SECTION 3/3: ENDGAME (21s)
  Narrative: CRUSHING_ATTACK | Tension: 0.73
  Filter: 280Hz | Resonance: 4.00
  Key Moments: 4
  → Layer 1: Evolving drone (55.0Hz base)
  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)
  Entropy: mean=0.327, range=[0.201, 0.704]
  → Layer 3a: Heartbeat (adaptive pulse)
  → Layer 3b: Gestures (4 key moments, entropy-scaled)

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

## Architecture Benefits

✓ **Modularity**: Clean separation of synthesis, patterns, gestures, and configuration
✓ **Extensibility**: Easy to add new pattern types and gesture archetypes
✓ **Tweakability**: Change parameters in seconds without code search
✓ **Testability**: Individual components can be tested in isolation
✓ **Maintainability**: Each module has single responsibility
✓ **Spectromorphological**: Gestures use established electroacoustic music theory

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
- **Gesture phases**: pre_shadow, impact, bloom, decay, residue

## Inspired By

**Laurie Spiegel** - Pioneer of algorithmic composition and creator of Music Mouse, an intelligent instrument that understands musical context and responds to player intent.

> "I automate whatever can be automated to be freer to focus on those aspects of music that can't be automated. The challenge is to figure out which is which." - Laurie Spiegel

## License

See LICENSE file for details.

## Contributing

The modular architecture makes contributions straightforward:
- **Add new gesture archetypes**: Edit `layer3b/archetype_configs.py`
- **Create new pattern types**: Add to `synth_composer/patterns/`
- **Add narrative types**: Edit `synth_config.py` → `NARRATIVE_BASE_PARAMS`
- **Create new processes**: Subclass `NarrativeProcess` in `synth_narrative.py`
- **Tweak synthesis**: Modify `SubtractiveSynth` in `synth_engine.py`
- **Add particle emission patterns**: Extend `layer3b/particle_system.py`

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
