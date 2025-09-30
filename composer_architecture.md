# SYNTH_COMPOSER ARCHITECTURE (REFACTORED)

## Overview

The synthesis system has been **completely refactored** from a 1,627-line monolith into a clean, modular architecture with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ChessSynthComposer                             │
│                    (synth_composer.py)                              │
│                                                                     │
│  Three-layer composition system orchestration:                      │
│  • Layer 1: Evolving drone (overall narrative)                      │
│  • Layer 2: Generative patterns (section narrative)                 │
│  • Layer 3: Sequencer (key moments)                                 │
│                                                                     │
│  ECO-seeded randomness for reproducibility                          │
│  Section crossfading for smooth transitions                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌─────────────────┐ ┌───────────────────────┐
│  SynthConfig     │ │ SubtractiveSynth│ │ Narrative Processes   │
│  (config.py)     │ │  (engine.py)    │ │  (narrative.py)       │
│                  │ │                 │ │                       │
│ Config-driven:   │ │ Pure DSP:       │ │ Generative:           │
│ • Scales         │ │ • Oscillators   │ │ • Markov chains       │
│ • Envelopes      │ │ • Moog filters  │ │ • State machines      │
│ • Narratives     │ │ • ADSR          │ │ • Evolution rules     │
│ • Patterns       │ │ • Supersaw      │ │ • Spiegel-inspired    │
│ • Mixing levels  │ │ • Anti-aliasing │ │ • Probabilistic       │
└──────────────────┘ └─────────────────┘ └───────────────────────┘
```

## Module Breakdown

### 1. synth_composer.py (687 lines)
**Main composition logic - orchestrates everything**

```python
class ChessSynthComposer:
    def __init__(self, chess_tags, config=None)
        # Accepts optional config for parameter overrides

    def compose_section(self, section, section_index, total_sections)
        # Three-layer composition:
        # Layer 1: Base drone (overall narrative)
        # Layer 2: Rhythmic patterns (section narrative)
        # Layer 3: Sequencer (key moments)

    def create_moment_voice(self, moment, current_params, progress)
        # Context-aware moment synthesis
        # Looks up parameters from config

    def compose()
        # Iterate all sections, normalize, return audio

    def save(filename)
        # Write to WAV file
```

**Key Responsibilities:**
- Parse narrative tags from JSON
- Coordinate three synthesis layers
- Apply narrative processes
- Mix and normalize final output

**No Hard-Coded Parameters** - All lookups go through `self.config`

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
**Narrative process classes - Spiegel-inspired evolution**

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
   ├── LAYER 2: Generative Patterns
   │   ├── Dispatch to narrative-specific generator
   │   ├── Execute generative algorithm (Markov/state machine)
   │   ├── Generate probabilistic note sequence
   │   └── Build pattern layer with random variations
   │
   ├── LAYER 3: Sequencer
   │   ├── Initialize pattern from config
   │   ├── Process evolution points (key moments)
   │   ├── Generate MIDI sequence
   │   ├── Synthesize with supersaw
   │   └── Apply global filter sweep
   │
   └── Mix layers with sidechain compression

3. FINALIZATION
   compose()
   ├── Generate all sections
   ├── Crossfade between sections (2 second overlap)
   ├── Normalize to prevent clipping
   └── Return audio array

4. OUTPUT
   save()
   └── Write WAV file (16-bit PCM, 44.1kHz, mono)
```

### Output: WAV Audio File
- 44.1 kHz sample rate
- 16-bit depth
- Mono channel
- Direct PCM (no compression)

---

## Three-Layer Synthesis System

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
**Narrative-specific generative algorithms for each section type**

**Implementation Approach:**
- Each section narrative has a dedicated generative pattern generator
- Uses probabilistic algorithms (Markov chains, state machines)
- Controlled randomness seeded by ECO code for reproducibility
- Non-repetitive, evolving patterns inspired by Laurie Spiegel

**Implemented Patterns:**

```python
# COMPLEX_STRUGGLE: Markov chain random walk
- 8x8 transition matrix weighted toward tonic
- Random durations (longer on tonic = thinking/hesitation)
- Random velocity and filter variations
- Increasing pauses as tension builds
- Waveform: pulse (hollow, thoughtful)

# KING_HUNT: 3-state machine (ATTACK/RETREAT/PAUSE)
- 70% upward bias during attack state
- Random octave jumps with increasing probability
- Evolving aggression (faster, brighter, louder)
- State-dependent timing and velocity
- Waveform: saw (aggressive, bright)

# CRUSHING_ATTACK: 3-state machine (ADVANCE/STRIKE/OVERWHELM)
- Downward bias (crushing from above)
- Hammer blow strikes on low notes
- Random chord stabs (60% probability)
- Accelerating rhythm, building power
- Waveform: saw with power chords
```

**Audio Result:**
- Each narrative sounds completely different
- Non-repetitive, truly generative
- Same game produces identical output (ECO seeding)
- Different openings produce different patterns

### Layer 3: Key Moments (Sequencer)
**Jarre-esque arpeggiated sequences for key events**

**Implementation:**
- 16-step MIDI-style sequencer patterns
- Triggered by key moments (blunders, brilliant moves, tactical sequences)
- Supersaw synthesis with global filter sweeps
- Classic electronic music aesthetic

```python
# Example: BRILLIANT move in MASTERPIECE context
{
    'freq': 220 * (1 + progress),   # Higher as game progresses
    'duration': 0.5,
    'waveform': 'pulse',
    'filter_base': 500,
    'filter_env_amount': 4000 * (1 + progress),  # Bigger sweeps later
    'resonance': 2.0,
    'amp_env': 'stab',
    'filter_env': 'sweep'
}
```

**Audio Result:**
- Context-aware moment voices
- Same event sounds different in different narratives
- Dramatic punctuation of key moments
- Repetitive arpeggios (intentional - classic electronic style)

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

---

## Benefits of Refactored Architecture

### For Developers
✓ **Find parameters instantly**: All in `synth_config.py`
✓ **No code search**: Change values without grep
✓ **Safe modifications**: Parameters isolated from logic
✓ **Clear responsibilities**: Each file has one job

### For Musicians/Sound Designers
✓ **Tweak without coding**: Edit `SynthConfig` dataclass
✓ **Immediate experimentation**: Change → save → run
✓ **Understand structure**: Clear parameter organization
✓ **Create presets**: Multiple config instances

### For AI Assistants
✓ **Clear entry points**: Import config, modify, test
✓ **Isolated changes**: Modify parameters without touching synthesis
✓ **Testable**: Easy to generate variations
✓ **Documented**: Type hints and docstrings throughout

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
- **New section narratives (Layer 2)**: Implement new generative pattern function
- **New moment types (Layer 3)**: Add entry to `MOMENT_VOICES`
- **New processes**: Subclass `NarrativeProcess`

### Recently Implemented (2025)
✓ **Multi-timescale evolving drones**: MACRO (section sweep) + MESO (slow LFO) + MICRO (fast LFO)
✓ **Generative Layer 2 patterns**: Markov chains and state machines
✓ **ECO code seeding**: Reproducible randomness
✓ **Section crossfading**: Smooth 2-second transitions
✓ **DEATH_SPIRAL narrative**: Dark saw wave with increasing chaos

### Possible Enhancements
- **Stereo output**: Separate left/right synthesis
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
└── ChessSynthComposer class (1,100+ lines)
```

**Problems:**
- Parameters scattered throughout
- Difficult to find and modify values
- Tight coupling between components
- Hard to test individual parts
- Code duplication

### After (Modular)
```
synth_composer.py (687 lines)
    └── ChessSynthComposer only

synth_config.py (664 lines)
    └── All parameters organized

synth_engine.py (354 lines)
    └── SubtractiveSynth only

synth_narrative.py (199 lines)
    └── NarrativeProcess classes
```

**Benefits:**
- Clear separation of concerns
- Easy to find and modify parameters
- Loosely coupled, testable modules
- No duplication, DRY principles
- Future-proof architecture

---

## Conclusion

The refactored architecture transforms a monolithic codebase into a clean, maintainable system where:

1. **Configuration is centralized** - One place for all parameters
2. **Synthesis is isolated** - Pure DSP with no musical decisions
3. **Processes are modular** - Easy to add new evolution types
4. **Composition is clear** - Orchestrates layers without buried logic

This makes the system **easy to read, easy to modify, and easy to extend** - fulfilling the original goals of the refactoring.
