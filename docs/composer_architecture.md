# SYNTH_COMPOSER ARCHITECTURE

## Overview

The synthesis system uses a modular architecture with clear separation of concerns:
- **Pattern generation** (Layer 2) via synth_composer/ package
- **Gesture archetypes** (Layer 3b) via layer3b/ package
- **Centralized configuration** via synth_config.py
- **Pure synthesis engine** via synth_engine.py

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ChessSynthComposer                             │
│                    (synth_composer.py)                              │
│                                                                     │
│  Four-layer composition system orchestration:                       │
│  • Layer 1: Evolving drone (overall narrative)                      │
│  • Layer 2: Generative patterns (section narrative)                 │
│  • Layer 3a: Heartbeat (rhythmic pulse)                             │
│  • Layer 3b: Gestures (key moment archetypes)                       │
│                                                                     │
│  ECO-seeded randomness for reproducibility                          │
│  Section crossfading for smooth transitions                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┬───────────────┐
              │               │               │               │
              ▼               ▼               ▼               ▼
┌──────────────────┐ ┌─────────────────┐ ┌───────────────┐ ┌─────────────────┐
│  SynthConfig     │ │ SubtractiveSynth│ │ synth_composer│ │ layer3b/        │
│  (config.py)     │ │  (engine.py)    │ │ (patterns/)   │ │ (gestures/)     │
│                  │ │                 │ │               │ │                 │
│ Config-driven:   │ │ Pure DSP:       │ │ Layer 2:      │ │ Layer 3b:       │
│ • Scales         │ │ • Oscillators   │ │ • Markov      │ │ • Curve-based   │
│ • Envelopes      │ │ • Moog filters  │ │ • State mach  │ │ • Particle-based│
│ • Narratives     │ │ • ADSR          │ │ • Theory      │ │ • Archetypes    │
│ • Archetypes     │ │ • Supersaw      │ │ • Outro       │ │ • Phase struct  │
│ • Mixing levels  │ │ • Anti-aliasing │ │ • NoteSynth*  │ │ • GestureSynth* │
└──────────────────┘ └─────────────────┘ └───────────────┘ └─────────────────┘
                              ▲                   │                   │
                              │                   │                   │
                              └───────────────────┴───────────────────┘
                                  * Both wrap SubtractiveSynth
```

## Module Breakdown

### 1. synth_composer.py
**Main composition logic - orchestrates everything**

```python
class ChessSynthComposer:
    def __init__(self, chess_tags, config=None)
        # Accepts optional config for parameter overrides
        # Initializes PatternCoordinator and GestureCoordinator

    def compose_section(self, section, section_index, total_sections)
        # Four-layer composition:
        # Layer 1: Base drone (overall narrative)
        # Layer 2: Generative patterns (section narrative via PatternCoordinator)
        # Layer 3a: Heartbeat (rhythmic pulse)
        # Layer 3b: Gestures (key moments via GestureCoordinator)

    def compose()
        # Iterate all sections, crossfade, normalize, return audio

    def save(filename)
        # Write to WAV file
```

**Key Responsibilities:**
- Parse narrative tags from JSON
- Coordinate four synthesis layers
- Apply narrative processes
- Interface with PatternCoordinator and GestureCoordinator
- Mix and normalize final output

**Configuration-Driven** - All parameters through `self.config` and archetype configs

---

### 2. synth_config.py (664 lines)
**Configuration hub - all parameters in one place**

```python
@dataclass
class SynthConfig:
    # === MUSICAL SCALES ===
    SCALES: Dict[str, List[float]]

    # === ENVELOPE PRESETS ===
    ENVELOPES: Dict[str, Tuple[float, float, float, float]]
    FILTER_ENVELOPES: Dict[str, Tuple[float, float, float, float]]

    # === NARRATIVE BASE PARAMETERS ===
    NARRATIVE_BASE_PARAMS: Dict[str, Dict]
    # Overall game character (defeat, masterpiece, draw)

    # === SECTION MODULATION PARAMETERS ===
    SECTION_MODULATIONS: Dict[str, Dict]
    # Per-section adjustments (tactical chaos, king hunt, etc.)

    # === MOMENT VOICE PARAMETERS ===
    MOMENT_VOICES: Dict[str, Dict]
    # Key moment synthesis settings

    # === SEQUENCER PATTERNS ===
    SEQUENCER_PATTERNS: Dict
    # 16-step MIDI patterns for Layer 3

    # === TIMING & MIXING ===
    TIMING: Dict[str, float]
    MIXING: Dict[str, float]
```

**Key Features:**
- **Easy to find**: All parameters organized by category
- **Easy to modify**: Change values without touching code
- **Type hints**: Clear parameter types and structure
- **Helper functions**: `get_envelope()`, `get_narrative_params()`, etc.

**Example Usage:**
```python
# Create custom config
config = SynthConfig()
config.MIXING['drone_level'] = 0.8
config.NARRATIVE_BASE_PARAMS['TUMBLING_DEFEAT']['filter_end'] = 500

# Use in composer
composer = ChessSynthComposer(tags, config=config)
```

---

### 3. synth_engine.py (354 lines)
**Pure synthesis engine - no musical decisions**

```python
class SubtractiveSynth:
    def __init__(self, sample_rate=44100)
        # Initialize filter state

    def oscillator(self, freq, duration, waveform='saw')
        # Band-limited waveform generation
        # PolyBLEP anti-aliasing for clean output

    def moog_filter(self, signal, cutoff_hz, resonance=0.0)
        # 4-pole Moog-style low-pass ladder
        # Soft saturation, controlled resonance

    def adsr_envelope(self, num_samples, attack, decay, sustain, release)
        # Exponential ADSR with configurable curve

    def supersaw(self, freq, duration, detune_cents, ...)
        # Roland JP-8000 style detuned saw ensemble
        # Time-varying filter, phase randomization

    def create_synth_note(self, freq, duration, waveform, ...)
        # Complete subtractive synthesis voice
        # Oscillator → Filter (envelope modulation) → Amp envelope
```

**Key Features:**
- **No hard-coded parameters**: All values passed as arguments
- **Stateful filter**: Maintains filter state for continuity
- **Anti-aliased**: PolyBLEP prevents aliasing artifacts
- **Analog modeling**: Soft clipping, exponential curves

**Design Philosophy:**
- Pure DSP engine
- No knowledge of chess or narratives
- Reusable for any synthesis task

---

### 4. synth_narrative.py (199 lines)
**Narrative process classes - Spiegel-inspired evolution (Legacy)**

```python
class NarrativeProcess(ABC):
    """Abstract base for process transformations"""
    def update(self, current_time, key_moment=None) -> dict
        # Returns transformation parameters

class TumblingDefeatProcess(NarrativeProcess):
    """Gradual deterioration through accumulated mistakes"""
    # Stability decays, tempo drifts, pitch becomes unstable
    # Based on entropy and error accumulation

class AttackingMasterpieceProcess(NarrativeProcess):
    """Building momentum toward triumph"""
    # Positive feedback loops, crescendo
    # Brilliant moves build momentum

class QuietPrecisionProcess(NarrativeProcess):
    """Equilibrium-seeking with gentle breathing"""
    # Homeostasis, returns to balance
    # Slow oscillation, minimal drift
```

**Key Features:**
- **Stateful evolution**: Each process maintains internal state
- **Time-aware**: Transforms based on progress through piece
- **Event-reactive**: Responds to key moments (blunders, brilliant moves)
- **Modular**: Easy to add new process types

**Factory Pattern:**
```python
def create_narrative_process(narrative, duration, plies):
    # Maps narrative string to appropriate process class
    return ProcessClass(duration, plies)
```

---

### 5. synth_composer/ Package
**Modular pattern generation system for Layer 2**

```
synth_composer/
├── __init__.py              # Package exports
├── coordinator.py           # PatternCoordinator (main interface)
├── core/                    # Core synthesis components
│   ├── note_event.py        # NoteEvent dataclass
│   ├── timing_engine.py     # Rhythm and timing calculations
│   ├── audio_buffer.py      # Audio buffer management
│   └── synthesizer.py       # NoteSynthesizer (wraps SubtractiveSynth)
└── patterns/                # Pattern generators
    ├── base.py              # PatternGenerator ABC
    ├── markov.py            # MarkovPatternGenerator
    ├── state_machine.py     # StateMachinePatternGenerator
    ├── theory.py            # TheoryPatternGenerator
    ├── outro.py             # OutroPatternGenerator
    └── conversion.py        # Helper utilities
```

**Key Components:**

**PatternCoordinator:**
```python
class PatternCoordinator:
    """Main interface for generating Layer 2 patterns"""

    def generate_pattern(self, section, context) -> np.ndarray:
        # 1. Select pattern generator based on section narrative
        # 2. Generate note events
        # 3. Synthesize audio via NoteSynthesizer
        # 4. Return mixed audio array
```

**NoteSynthesizer (synth_composer/core/synthesizer.py):**
```python
class NoteSynthesizer:
    """
    Wrapper around SubtractiveSynth for NoteEvent objects.

    Converts NoteEvent → SubtractiveSynth parameters
    Does NOT implement its own DSP - delegates to SubtractiveSynth
    """
    def synthesize(self, event: NoteEvent) -> np.ndarray:
        return self.synth.create_synth_note(
            freq=event.freq,
            duration=event.duration,
            waveform=event.waveform,
            # ... all other synth parameters
        )
```

**Pattern Generators:**
- **MarkovPatternGenerator**: Probabilistic note sequences with transition matrices
- **StateMachinePatternGenerator**: Multi-state patterns (ADVANCE → STRIKE → OVERWHELM)
- **TheoryPatternGenerator**: Opening theory, methodical development
- **OutroPatternGenerator**: Final resolution and fadeout

**Key Features:**
- **Modular**: Each pattern type is independent
- **Extensible**: Easy to add new pattern generators
- **ECO-seeded**: Reproducible random generation
- **Context-aware**: Responds to entropy, tension, section characteristics
- **Shared synthesis**: Uses SubtractiveSynth for all DSP

---

### 6. layer3b/ Package
**Gesture archetype system for Layer 3b**

**See:** `docs/LAYER3B_COMPLETE_REFERENCE.md` and `docs/layer3b_implementation.md`

```
layer3b/
├── __init__.py              # Package exports
├── archetype_configs.py     # CURVE_ARCHETYPES and PARTICLE_ARCHETYPES
├── base.py                  # GestureGenerator base class
├── coordinator.py           # GestureCoordinator (archetype registry)
├── curve_generators.py      # Pitch/harmony/filter/envelope curve functions
├── particle_system.py       # ParticleEmitter for stochastic gestures
├── synthesizer.py           # GestureSynthesizer (wraps SubtractiveSynth)
└── utils.py                 # Phase computation, normalization
```

**GestureSynthesizer (layer3b/synthesizer.py):**
```python
class GestureSynthesizer:
    """
    High-level wrapper around SubtractiveSynth for gesture synthesis.

    Orchestrates time-varying multi-voice synthesis:
    - Time-varying pitch (via SubtractiveSynth oscillators)
    - Time-varying filters (via SubtractiveSynth.moog_filter_timevarying)
    - Amplitude envelopes (direct multiplication)
    - Noise texture (via SubtractiveSynth.generate_noise)
    - Shimmer effects (amplitude modulation)

    Does NOT implement its own DSP - delegates to SubtractiveSynth
    """
    def synthesize(self, pitch_voices, filter_curve, envelope, texture_curve):
        # Orchestrates time-varying synthesis using SubtractiveSynth primitives
```

**Two Gesture Approaches:**

**Curve-Based Gestures** (Deterministic, expressive):
- Full parameter control via curves
- Examples: MOVE, BRILLIANT, BLUNDER, CHECKMATE, GAME_CHANGING
- Phase structure: pre_shadow → impact → bloom → decay → residue
- Pitch/harmony/filter/envelope/texture curves

**Particle-Based Gestures** (Stochastic, emergent):
- Emission-controlled polyphonic gestures
- Examples: INACCURACY, FIRST_EXCHANGE, TACTICAL_SEQUENCE, FINAL_RESOLUTION
- Particle properties: independent pitch, velocity, lifetime, detune
- Emission patterns: gusts, bursts, drifts, clusters, dissolves

**Key Features:**
- **Spectromorphological**: Uses electroacoustic music theory classifications
- **Context-responsive**: Scales with tension and entropy
- **Configuration-driven**: All archetypes defined in archetype_configs.py
- **Unified pipeline**: Both approaches share same synthesis infrastructure
- **Shared synthesis**: Uses SubtractiveSynth for all DSP primitives

---

## Synthesizer Architecture

### Single Source of DSP Truth: SubtractiveSynth

All audio synthesis in the system uses **synth_engine.SubtractiveSynth** as the foundation. There are no duplicate DSP implementations.

**SubtractiveSynth (synth_engine.py):**
- Pure DSP primitives
- Oscillators (sine, triangle, saw, square, supersaw)
- Moog-style 4-pole low-pass filter
- Time-varying filter support
- ADSR envelopes
- Noise generators (white, pink, brown)
- PolyBLEP anti-aliasing

### High-Level Wrappers

Different layers use **wrapper classes** that delegate to SubtractiveSynth:

**Layer 2 → NoteSynthesizer (synth_composer/core/synthesizer.py):**
```python
class NoteSynthesizer:
    """Wraps SubtractiveSynth for discrete note events"""
    def __init__(self, synth_engine):
        self.synth = synth_engine  # SubtractiveSynth instance

    def synthesize(self, event: NoteEvent) -> np.ndarray:
        # Converts NoteEvent to SubtractiveSynth parameters
        return self.synth.create_synth_note(...)
```

**Layer 3b → GestureSynthesizer (layer3b/synthesizer.py):**
```python
class GestureSynthesizer:
    """Wraps SubtractiveSynth for time-varying gestures"""
    def __init__(self, synth_engine):
        self.synth = synth_engine  # SubtractiveSynth instance

    def synthesize(self, pitch_voices, filter_curve, envelope, texture_curve):
        # Orchestrates time-varying synthesis using SubtractiveSynth primitives
        # - Multi-voice oscillators
        # - Time-varying filters
        # - Noise textures
```

**Layer 1 & 3a:**
- Call SubtractiveSynth directly (no wrapper needed)
- Layer 1: Continuous drones
- Layer 3a: Heartbeat synthesis

### Design Benefits

✓ **No code duplication**: All DSP in one place
✓ **Consistent sound**: All layers use same synthesis engine
✓ **Easy maintenance**: Bug fixes benefit all layers
✓ **Shared optimizations**: Performance improvements benefit all layers
✓ **Clear architecture**: Wrappers coordinate, SubtractiveSynth synthesizes

---

## Data Flow

### Input: Tagged Chess Game JSON
```json
{
  "overall_narrative": "TUMBLING_DEFEAT",
  "eco": "B90",
  "total_plies": 78,
  "duration_seconds": 65,
  "sections": [
    {
      "name": "OPENING",
      "duration": "0:20",
      "narrative": "COMPLEX_POSITION",
      "tension": 0.45,
      "key_moments": [
        {"type": "DEVELOPMENT", "ply": 8, "second": 8},
        {"type": "BLUNDER", "ply": 16, "second": 18}
      ]
    },
    ...
  ]
}
```

### Processing Pipeline

```
1. INITIALIZATION
   ChessSynthComposer.__init__()
   ├── Load config (or use default)
   ├── Create SubtractiveSynth engine
   ├── Seed randomness with ECO code (reproducibility)
   ├── Get narrative base parameters from config
   └── Create narrative process

2. COMPOSITION (per section)
   compose_section()
   ├── Calculate progress (0.0 - 1.0)
   ├── Interpolate base parameters
   ├── Get section modulation from config
   ├── Update narrative process
   │
   ├── LAYER 1: Base Drone
   │   ├── Get drone frequency from scale
   │   ├── Generate continuous drone (supersaw or basic waveform)
   │   └── Apply LFO modulation
   │
   ├── LAYER 2: Generative Patterns (via PatternCoordinator)
   │   ├── Select pattern generator based on section narrative
   │   ├── Execute generative algorithm (Markov/state machine/theory/outro)
   │   ├── Generate note events with context (entropy, tension)
   │   └── Synthesize pattern layer
   │
   ├── LAYER 3a: Heartbeat
   │   ├── Generate LUB-dub cardiac rhythm
   │   ├── Apply entropy-driven BPM modulation
   │   ├── Add pitch variation and timing jitter
   │   └── Low-pass filter for sub-bass character
   │
   ├── LAYER 3b: Gestures (via GestureCoordinator)
   │   ├── Identify key moments in section
   │   ├── Select archetype for each moment
   │   ├── Generate gesture audio (curve-based or particle-based)
   │   └── Apply context scaling (tension, entropy)
   │
   └── Mix all four layers with appropriate levels

3. FINALIZATION
   compose()
   ├── Generate all sections
   ├── Crossfade between sections (2 second overlap)
   ├── Normalize to prevent clipping
   └── Return audio array

4. OUTPUT
   save()
   └── Write WAV file (16-bit PCM, 44.1kHz, stereo/mono)
```

### Output: WAV Audio File
- 44.1 kHz sample rate
- 16-bit depth
- Stereo channels (configurable)
- Direct PCM (no compression)

---

## Four-Layer Synthesis System

### Layer 1: Overall Narrative (Base Character)
**Sets fundamental parameters for entire game**

```python
# Example: TUMBLING_DEFEAT
{
    'base_waveform': 'supersaw',    # Rich, detuned
    'filter_start': 2500,           # Starts bright
    'filter_end': 300,              # Ends dark
    'resonance_start': 0.8,
    'resonance_end': 3.5,           # Near self-oscillation
    'tempo_start': 1.0,
    'tempo_end': 0.7,               # Slows down
    'detune_start': 3,              # Tight
    'detune_end': 20,               # Very dissonant
    'scale': 'phrygian'             # Dark mode
}
```

**Audio Result:**
- Continuous drone throughout section
- Evolves gradually based on game progress
- Provides harmonic foundation

### Layer 2: Section Narrative (Generative Patterns)
**Modular pattern generation system via synth_composer/ package**

**Implementation Approach:**
- Pattern generators selected based on section narrative
- Uses probabilistic algorithms (Markov chains, state machines, theory patterns)
- Controlled randomness seeded by ECO code for reproducibility
- Non-repetitive, evolving patterns inspired by Laurie Spiegel

**Pattern Generators:**

**MarkovPatternGenerator:**
- Transition matrix-based note selection
- Weighted toward tonic for stability
- Random durations and velocity variations
- Context-aware: responds to entropy and tension

**StateMachinePatternGenerator:**
- Multi-state patterns (e.g., ADVANCE → STRIKE → OVERWHELM)
- State-dependent behavior (attack/retreat dynamics)
- Evolving aggression and density
- Directional bias (upward attacks, downward crushing)

**TheoryPatternGenerator:**
- Opening theory: methodical development
- Quiet positional: sparse, contemplative
- Conversion: systematic simplification

**OutroPatternGenerator:**
- Final resolution patterns
- Fadeout and closure
- Harmonic resolution to tonic

**Audio Result:**
- Each narrative sounds completely different
- Non-repetitive, truly generative
- Same game produces identical output (ECO seeding)
- Different openings produce different patterns
- Modular: easy to add new pattern types

### Layer 3a: Heartbeat (Rhythmic Pulse)
**Biological rhythmic foundation**

**Implementation:**
- Sine wave synthesis with precise LUB-dub cardiac pattern
- Entropy-driven variation (BPM, pitch, timing jitter)
- Low-pass filtered for sub-bass/thump character
- Click-free using linear ADSR and dedicated filter

**Heartbeat Pattern:**
```python
# LUB-dub cycle structure
lub_beat    # Main "LUB" (attack + decay + release)
gap         # Short silence (~0.15s)
dub_beat    # Quieter "dub" (80% volume, lower pitch)
pause       # Rest to complete BPM cycle
```

**Entropy-Driven Variation:**
- **BPM modulation**: Low entropy = slow/steady (60 BPM), high = fast/anxious (100 BPM)
- **Pitch variation**: ±2 semitones around root based on entropy
- **Timing jitter**: High entropy adds slight randomness to gaps/pauses

**Audio Result:**
- Biological, organic pulse underlying the composition
- Continuous presence (not event-triggered)
- Provides temporal/rhythmic foundation independent of chess events
- Centered in stereo field (biological constant)

---

### Layer 3b: Gestures (Key Moment Archetypes)
**Spectromorphological gesture system via layer3b/ package**

**See:** `docs/LAYER3B_COMPLETE_REFERENCE.md` and `docs/layer3b_implementation.md` for complete documentation

**Two Gesture Approaches:**

#### Curve-Based Gestures (Deterministic)
**Full parameter control via multi-dimensional curves**

**Architecture:**
- **Phase structure**: pre_shadow → impact → bloom → decay → residue
- **Curve generators**: Pitch, harmony, filter, envelope, texture
- **Spectromorphological**: Uses electroacoustic music theory classifications

**Pitch Curve Types:**
- Stable: Held frequency
- Glissando: Smooth continuous pitch change
- Tremor: Oscillating pitch variations
- Drift: Slow wandering
- Leap: Discrete pitch jumps

**Harmony Types:**
- Unison: Single voice
- Dyad: Two-voice intervals
- Cluster: Dense semitone groupings
- Chord: Harmonic intervals
- Converging: Multiple voices resolving

**Filter Curve Types:**
- Sweep: Continuous cutoff movement
- Focus: Narrowing bandwidth
- Opening/Closing: Spectral evolution
- Morphing: Filter type transitions

**Examples:** MOVE, BRILLIANT, BLUNDER, CHECKMATE, GAME_CHANGING

#### Particle-Based Gestures (Stochastic)
**Emission-controlled polyphonic gestures with emergent behavior**

**Architecture:**
- **Particle system**: Independent particles with lifetime, pitch, velocity, detune
- **Emission curves**: Control spawning density over time
- **Emergent behavior**: Complex textures from simple rules

**Emission Patterns:**
- **Gust**: Quick burst of particles
- **Burst**: Explosive emission
- **Drift**: Gradual sustained emission
- **Rhythmic clusters**: Pulsed groups
- **Dissolve**: Gradual fadeout

**Particle Properties:**
- Independent pitch evolution
- Individual velocity envelopes
- Variable lifetime durations
- Detune for spectral richness
- Exponential decay curves

**Examples:** INACCURACY, FIRST_EXCHANGE, TACTICAL_SEQUENCE, FINAL_RESOLUTION

#### Unified Features
**Both approaches share:**
- **Context-responsive**: Scale with section tension and entropy
- **Configuration-driven**: All archetypes defined in `archetype_configs.py`
- **Spectromorphological**: Based on electroacoustic music theory
- **Synthesis integration**: Wraps SubtractiveSynth for DSP primitives

**Audio Result:**
- **Event-reactive**: Triggered by key chess moments
- **Emotionally expressive**: Each archetype has distinct character
- **Context-aware**: Responds to section tension and entropy
- **Spectrally rich**: Complex timbral evolution
- **Non-repetitive**: Stochastic elements provide variation

**Layer 3 Design Philosophy:**
The four-layer system addresses different temporal scales and narrative functions:
- **3a (heartbeat)**: Continuous, biological, rhythmic foundation
- **3b (gestures)**: Discrete, expressive, event-driven punctuation

This creates musical depth with clear separation between continuous pulse and discrete moments.

---

## Reproducibility: ECO Code Seeding

### The Challenge
Layer 2 uses probabilistic generative algorithms that rely on random number generation. Without control, the same chess game would produce different music on each run.

### The Solution: ECO Code Seeding
```python
def _seed_from_eco(self, eco_code):
    """Convert ECO code to integer seed for reproducible randomness"""
    # ECO format: Letter (A-E) + two digits (00-99)
    # Convert to integer: A00=0, A01=1, ..., E99=599
    if len(eco_code) >= 3:
        letter_value = ord(eco_code[0].upper()) - ord('A')  # 0-4
        number_value = int(eco_code[1:3])  # 0-99
        seed = letter_value * 100 + number_value
    else:
        seed = 0  # Fallback
    np.random.seed(seed)
```

### Benefits
✓ **Reproducible**: Same game → same ECO → identical audio output
✓ **Chess-meaningful**: Opening character influences musical character
✓ **Deterministic**: Can reliably regenerate exact audio
✓ **Verifiable**: MD5 checksum validation proves reproducibility

### Example
```bash
# Game 13: ECO C11 (French Defense)
# Seed: 2*100 + 11 = 211
python3 synth_composer.py game13.json  # MD5: eb3de004729dcd5e80318c4642f59a1a
python3 synth_composer.py game13.json  # MD5: eb3de004729dcd5e80318c4642f59a1a (identical!)

# Game 12: ECO A13 (English Opening)
# Seed: 0*100 + 13 = 13
# Different seed → different generative patterns
```

---

## Section Transitions: Crossfading

### The Problem
Hard cuts between sections created audible gaps and discontinuities.

### The Solution: 2-Second Crossfades
```python
# Fade out end of current section
fade_out = np.linspace(1.0, 0.0, crossfade_samples)
composition[-crossfade_samples:] *= fade_out

# Fade in start of next section
fade_in = np.linspace(0.0, 1.0, crossfade_samples)
next_section[:crossfade_samples] *= fade_in

# Overlap and sum
composition = np.concatenate([
    composition[:-crossfade_samples],
    composition[-crossfade_samples:] + next_section[:crossfade_samples],
    next_section[crossfade_samples:]
])
```

### Benefits
✓ **Smooth transitions**: No audible gaps or clicks
✓ **Natural evolution**: Sections blend organically
✓ **Shorter output**: 2s overlap per transition saves time
✓ **Configurable**: `TIMING['section_crossfade_sec']` in config

---

## Entropy-Driven Layer 3: Continuous Musical Evolution

### The Problem
Traditional event-driven sequencers switch patterns only at discrete key moments (captures, blunders), creating static repetition between events and missing gradual complexity changes.

### The Solution: Evaluation Volatility as Compositional Parameter
Following Laurie Spiegel's principle of using information entropy to drive musical evolution, Layer 3 uses **computer evaluation volatility** to continuously control musical parameters.

> "The moment to moment variation of level of predictability that is embodied in an entropy curve arouses in the listener feelings of expectation, anticipation, satisfaction, disappointment, surprise, tension, frustration and other emotions."
> — Laurie Spiegel

### Entropy Calculation
```python
from entropy_calculator import ChessEntropyCalculator

# Initialize with move data including eval_cp (centipawn evaluation)
entropy_calculator = ChessEntropyCalculator(moves)

# Calculate volatility-based entropy for section
raw_entropy = entropy_calculator.calculate_combined_entropy(
    start_ply, end_ply,
    weights={'eval': 0.5, 'tactical': 0.4, 'time': 0.1}
)

# Smooth to avoid sudden jumps
entropy_curve = gaussian_filter1d(raw_entropy, sigma=2)
```

**Entropy Sources:**
1. **Evaluation volatility** (50%): Rolling std dev of centipawn evaluation
   - High volatility = unclear position = high entropy
   - Low volatility = clear position = low entropy
2. **Tactical density** (40%): Frequency of captures/checks/promotions
3. **Thinking time** (10%): Time spent on move (if available)

### Musical Parameter Mappings

**1. Note Selection Pool (Harmonic Complexity)**
```python
# Low entropy (0.0-0.3): Simple, predictable
if entropy < 0.3:
    available_intervals = [0, 4]  # Root-fifth only (C, G)

# Medium entropy (0.3-0.7): Moderate
elif entropy < 0.7:
    available_intervals = [0, 2, 4, 5, 7]  # Diatonic scale

# High entropy (0.7-1.0): Complex, chromatic
else:
    available_intervals = [0, 1, 2, 3, 4, 5, 6, 7]  # Full chromatic
```

**2. Rhythmic Variation (Temporal Predictability)**
```python
# High entropy = irregular timing (±50% variation)
rhythm_var = entropy * 0.5
duration_multiplier = 1.0 + random.uniform(-rhythm_var, rhythm_var)
actual_duration = base_duration * duration_multiplier
```

**3. Filter Modulation Speed (Spectral Activity)**
```python
# Low entropy = slow filter sweep (0.02 Hz = 50s cycle)
# High entropy = fast filter sweep (0.07 Hz = 14s cycle)
filter_lfo_speed = 0.02 + entropy * 0.05
```

**4. Portamento/Glide Amount (Note Connectivity)**
```python
# Low entropy = long smooth glides (flowing, connected)
# High entropy = short/no glides (jumpy, nervous)
glide_reduction = entropy * 0.5  # Up to 50% reduction
glide_time = base_glide_time * (1.0 - glide_reduction)
```

**5. Harmonic Density (Vertical Complexity)**
```python
# High entropy: Add random harmony notes (cluster effect)
if entropy > 0.7 and random.random() < (entropy - 0.7):
    harmony_interval = random.choice([3, 4, 7])  # Third, fourth, fifth
    add_harmony_note(harmony_interval)
```

### Configuration
```python
ENTROPY_CONFIG = {
    # Calculation weights
    'weights': {
        'eval': 0.5,      # Evaluation volatility (primary)
        'tactical': 0.4,  # Tactical density
        'time': 0.1,      # Thinking time
    },

    # Smoothing
    'smoothing_sigma': 2,  # Gaussian filter sigma

    # Musical thresholds
    'low_threshold': 0.3,   # Below = simple
    'high_threshold': 0.7,  # Above = complex

    # Note pools
    'note_pools': {
        'low': [0, 4],                    # Simple: root-fifth
        'medium': [0, 2, 4, 5, 7],        # Moderate: diatonic
        'high': [0, 1, 2, 3, 4, 5, 6, 7], # Complex: chromatic
    },

    # Parameter ranges
    'rhythm_variation_max': 0.5,        # Max ±50% timing variation
    'filter_lfo_range': (0.02, 0.07),   # 50s to 14s cycles
    'glide_reduction_max': 0.5,         # Max 50% portamento reduction
    'harmony_probability_threshold': 0.7,  # Start harmonies above this
}
```

### Musical Benefits
✓ **Organic evolution**: Music continuously adapts to position complexity
✓ **Anticipatory**: Tension builds *before* critical moments (not just reacting)
✓ **Emotional arcs**: Rising entropy = anticipation, falling = resolution
✓ **Non-repetitive**: Same pattern sounds different based on position
✓ **Chess-meaningful**: Eval volatility directly correlates to position uncertainty

### Example: Opening Section
```
Plies 1-3:   Entropy 0.06 (book theory - very predictable)
             → Notes: C-G-C-G (root-fifth only)
             → Rhythm: Regular quarter notes
             → Filter: Stable, slow sweep
             → Glides: Long, smooth

Plies 4-9:   Entropy rising 0.19→0.27 (exchanges opening position)
             → Notes: Adding C-D-E-G-A (diatonic)
             → Rhythm: Starting to vary
             → Filter: Opening up faster
             → Glides: Getting shorter

Plies 13-18: Entropy 0.73 (tactical complexity)
             → Notes: Chromatic C-Db-Eb-E-G-Ab
             → Rhythm: Irregular triplets
             → Filter: Fast sweeping
             → Glides: Jumpy, nervous
             → Harmony: Added cluster notes
```

**Result**: You *hear* the position getting more complex before the blunder happens, creating anticipation rather than just reaction.

---

## Stereo Output: Entropy-Driven Spatial Dynamics

### The Problem
Mono output lacks spatial depth and doesn't leverage the full stereo field to express the dynamic complexity of chess positions.

### The Solution: Layer-Specific Stereo Treatment
```python
# Layer 1+2 (Drone + Generative Patterns): Static stereo width
# Width varies with section tension (0.0 = mono, 0.8 = wide)
width_12 = min_width + (avg_tension * 0.3)
stereo_12 = stereo_width(layers_1_2, width=width_12, center_pan=0.0)

# Layer 3 (Sequencer): Dynamic entropy-driven panning
# Position entropy controls left/right panning
entropy_resampled = resample(section['position_entropy'], len(layer_3))
pan_curve = (entropy_resampled - 0.5) * 2.0 * entropy_pan_amount
stereo_3 = apply_dynamic_pan(layer_3, pan_curve)

# Mix stereo layers
composition = mix_stereo([stereo_12, stereo_3])
```

### Implementation Details

**Stereo Width (Layers 1+2):**
- Uses Haas effect (slight delay) plus equal-power panning
- Width controlled by section tension
- Remains relatively static/centered
- Creates ambient spatial foundation

**Entropy-Driven Panning (Layer 3):**
- **Evaluation volatility entropy** from computer analysis (standard deviation of eval in centipawns)
- High entropy (volatile/unclear position) → pans to extremes
- Low entropy (stable/clear position) → stays center
- Sample-accurate interpolation for smooth movement
- Constant-power panning law maintains perceived loudness
- **Combines with slow L/R oscillation** for directional movement

**Equal-Power Panning Law:**
```python
# Ensures constant perceived energy across stereo field
left_gain = np.cos(pan_angle)
right_gain = np.sin(pan_angle)
# Where pan_angle ranges from 0 (left) to π/2 (right)
```

### Configuration
```python
STEREO_CONFIG = {
    'enabled': True,                # Enable stereo output
    'entropy_pan_amount': 1.5,      # How much entropy affects panning
    'min_width': 0.0,               # Narrow stereo at low tension
    'max_width': 0.8,               # Wide stereo at high tension
}

MIXING = {
    'channels': 2,  # Stereo output
}
```

### Musical Benefits
✓ **Spatial depth**: Music occupies full stereo field
✓ **Position complexity visualization**: Hear position entropy as spatial movement
✓ **Layered spatial treatment**: Static foundation, dynamic foreground
✓ **Laurie Spiegel principle**: Data-driven spatial movement (entropy → position)
✓ **Smooth transitions**: Constant-power panning prevents loudness jumps
✓ **Tension-responsive width**: Tactical sections expand stereo field

### Technical Benefits
✓ **Configurable**: Enable/disable stereo, adjust panning amounts
✓ **Backward compatible**: Falls back to mono if disabled
✓ **Sample-accurate**: Entropy data resampled to match audio length
✓ **Phase-coherent**: Haas effect maintains mono compatibility

---

## Configuration Examples

### Tweaking Overall Narratives
```python
# Make defeat less dark
config.NARRATIVE_BASE_PARAMS['TUMBLING_DEFEAT']['filter_end'] = 500  # Was 300

# Make masterpiece more dramatic
config.NARRATIVE_BASE_PARAMS['ATTACKING_MASTERPIECE']['filter_end'] = 8000  # Was 5000
```

### Adjusting Section Modulations
```python
# Make tactical chaos even more chaotic
config.SECTION_MODULATIONS['TACTICAL_CHAOS']['note_density'] = 3.0  # Was 2.0
config.SECTION_MODULATIONS['TACTICAL_CHAOS']['resonance_add'] = 3.0  # Was 2.0
```

### Customizing Moment Voices
```python
# Make blunders more dramatic
config.MOMENT_VOICES['BLUNDER_IN_DEFEAT']['freq'] = 27.5  # Was 55 (octave lower)
config.MOMENT_VOICES['BLUNDER_IN_DEFEAT']['duration'] = 2.0  # Was 1.0 (longer)
```

### Mixing Adjustments
```python
# More drone, less sequencer
config.LAYER_MIXING['drone_in_supersaw'] = 0.8  # Was 0.6
config.LAYER_MIXING['sequencer_note_level'] = 0.3  # Was 0.4

# Overall louder
config.MIXING['section_level'] = 0.4  # Was 0.3
```

### Stereo Configuration
```python
# Disable stereo (mono output)
config.STEREO_CONFIG['enabled'] = False

# More dramatic panning
config.STEREO_CONFIG['entropy_pan_amount'] = 2.5  # Was 1.5 (wider movement)

# Wider stereo field for layers 1+2
config.STEREO_CONFIG['max_width'] = 1.0  # Was 0.8 (maximum width)

# Keep layers 1+2 more centered
config.STEREO_CONFIG['min_width'] = 0.2  # Was 0.0 (never fully mono)
```

### Entropy Configuration
```python
# Adjust entropy weights
config.ENTROPY_CONFIG['weights']['eval'] = 0.7      # Was 0.5 (more eval influence)
config.ENTROPY_CONFIG['weights']['tactical'] = 0.2  # Was 0.4 (less tactical)

# More aggressive thresholds (more chromatic earlier)
config.ENTROPY_CONFIG['low_threshold'] = 0.2   # Was 0.3
config.ENTROPY_CONFIG['high_threshold'] = 0.6  # Was 0.7

# More rhythmic chaos
config.ENTROPY_CONFIG['rhythm_variation_max'] = 0.8  # Was 0.5 (±80% variation)

# Less portamento reduction (keep glides even at high entropy)
config.ENTROPY_CONFIG['glide_reduction_max'] = 0.3  # Was 0.5

# More smoothing (less jumpy entropy curve)
config.ENTROPY_CONFIG['smoothing_sigma'] = 3  # Was 2
```

---

## Benefits of Modular Architecture

### For Developers
✓ **Find parameters instantly**: All in `synth_config.py` and `archetype_configs.py`
✓ **No code search**: Change values without grep
✓ **Safe modifications**: Parameters isolated from logic
✓ **Clear responsibilities**: Each module has one job
✓ **Easy extension**: Add new patterns or archetypes via configuration

### For Musicians/Sound Designers
✓ **Tweak without coding**: Edit configuration files
✓ **Immediate experimentation**: Change → save → run
✓ **Understand structure**: Clear parameter organization
✓ **Create presets**: Multiple config instances
✓ **Gesture design**: Test individual archetypes with gesture_test.py and particle_test.py

### For AI Assistants
✓ **Clear entry points**: Import config, modify, test
✓ **Isolated changes**: Modify parameters without touching synthesis
✓ **Testable**: Easy to generate variations
✓ **Documented**: Type hints and comprehensive documentation
✓ **Modular**: Add features without breaking existing code

---

## Testing Strategy

### Unit Testing Synthesis Engine
```python
from synth_engine import SubtractiveSynth

synth = SubtractiveSynth(44100)

# Test oscillator
signal = synth.oscillator(440, 1.0, 'saw')
assert len(signal) == 44100
assert -1.0 <= signal.min() <= signal.max() <= 1.0

# Test filter
filtered = synth.moog_filter(signal, 1000, 0.5)
assert len(filtered) == len(signal)
```

### Integration Testing Composer
```python
from synth_composer import ChessSynthComposer
from synth_config import SynthConfig

# Create test tags
tags = {
    'overall_narrative': 'TUMBLING_DEFEAT',
    'total_plies': 40,
    'duration_seconds': 30,
    'sections': [...]
}

# Test with custom config
config = SynthConfig()
config.MIXING['section_level'] = 0.5

composer = ChessSynthComposer(tags, config)
audio = composer.compose()

assert len(audio) > 0
assert audio.dtype == np.float64
```

### A/B Testing Parameters
```python
# Test different filter ranges
configs = []
for filter_end in [300, 500, 800, 1200]:
    cfg = SynthConfig()
    cfg.NARRATIVE_BASE_PARAMS['TUMBLING_DEFEAT']['filter_end'] = filter_end
    configs.append((filter_end, cfg))

for filter_end, cfg in configs:
    composer = ChessSynthComposer(tags, cfg)
    composer.save(f'test_filter_{filter_end}.wav')
```

---

## Performance Characteristics

### Synthesis Times (approximate)
- **Short game (30s audio)**: ~15-30 seconds
- **Medium game (60s audio)**: ~30-60 seconds
- **Long game (90s audio)**: ~60-120 seconds

### Memory Usage
- **Config**: ~1 MB (Python dictionaries)
- **Engine state**: Minimal (~1 KB filter state)
- **Audio buffer**: ~176 KB per second (44.1kHz float64)

### Optimization Opportunities
- **Chunk processing**: Filters processed in 512-sample chunks
- **Pattern caching**: Sequencer patterns pre-generated
- **Envelope reuse**: ADSR cached for common durations
- **Filter state**: Continuous across notes (no resets)

---

## Future Extensions

### Easy to Add
- **New overall narratives (Layer 1)**: Add entry to `NARRATIVE_BASE_PARAMS`
- **New section patterns (Layer 2)**: Subclass `PatternGenerator` in synth_composer/patterns/
- **New gesture archetypes (Layer 3b)**: Add configuration to `archetype_configs.py`
- **New curve types**: Add to `curve_generators.py`
- **New particle emissions**: Extend `particle_system.py`
- **New processes**: Subclass `NarrativeProcess`

### Recently Implemented (2025)
✓ **Four-layer system**: Split Layer 3 into heartbeat (3a) and gestures (3b)
✓ **Modular pattern generation**: synth_composer/ package with multiple generators
✓ **Gesture archetype system**: layer3b/ package with curve and particle approaches
✓ **Spectromorphological design**: Electroacoustic music theory classifications
✓ **ECO code seeding**: Reproducible randomness
✓ **Section crossfading**: Smooth 2-second transitions
✓ **Entropy-driven composition**: Evaluation volatility controls multiple parameters
✓ **Stereo output with entropy panning**: Spatial position driven by evaluation entropy
✓ **Testing tools**: gesture_test.py and particle_test.py for archetype development

### Possible Enhancements
- **More pattern generators**: Additional Layer 2 pattern types
- **More gesture archetypes**: Expand Layer 3b moment library
- **Multi-band processing**: Frequency-specific effects
- **Granular synthesis**: Micro-sound textures
- **Spectral processing**: FFT-based effects
- **Real-time mode**: Streaming synthesis

---

## Comparison: Before vs After

### Before (Monolithic)
```
synth_composer.py (1,627 lines)
├── NarrativeProcess classes (160 lines)
├── SubtractiveSynth class (343 lines)
├── Hard-coded parameters (scattered)
├── Pattern generation (inline)
└── ChessSynthComposer class (1,100+ lines)
```

**Problems:**
- Parameters scattered throughout
- Difficult to find and modify values
- Tight coupling between components
- Hard to test individual parts
- Code duplication
- No gesture system
- Limited pattern variety

### After (Modular)
```
synth_composer.py
    └── ChessSynthComposer (main orchestrator)

synth_config.py
    └── Centralized configuration

synth_engine.py
    └── SubtractiveSynth (pure DSP)

synth_narrative.py
    └── NarrativeProcess classes (legacy)

synth_composer/
    ├── coordinator.py (PatternCoordinator)
    ├── core/ (timing, events, audio, synthesis)
    └── patterns/ (markov, state machine, theory, outro)

layer3b/
    ├── coordinator.py (GestureCoordinator)
    ├── archetype_configs.py (all archetypes)
    ├── base.py (GestureGenerator)
    ├── curve_generators.py (pitch/harmony/filter/envelope)
    ├── particle_system.py (stochastic gestures)
    ├── synthesizer.py (GestureSynthesizer)
    └── utils.py
```

**Benefits:**
- Clear separation of concerns
- Easy to find and modify parameters
- Loosely coupled, testable modules
- No duplication, DRY principles
- Extensible architecture
- Comprehensive gesture system
- Multiple pattern generators
- Spectromorphological design principles

---

## Conclusion

The modular architecture creates a maintainable system where:

1. **Configuration is centralized** - Parameters in synth_config.py and archetype_configs.py
2. **Synthesis is isolated** - Pure DSP with no musical decisions
3. **Patterns are modular** - Easy to add new Layer 2 generators
4. **Gestures are expressive** - Comprehensive Layer 3b archetype system
5. **Composition is clear** - Orchestrates four layers without buried logic
6. **Testing is built-in** - Tools for development and validation

This makes the system **easy to read, easy to modify, easy to extend, and easy to test** - providing a solid foundation for algorithmic music composition.
