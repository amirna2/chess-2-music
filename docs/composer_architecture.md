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
   └── Write WAV file (16-bit PCM, 44.1kHz, stereo/mono)
```

### Output: WAV Audio File
- 44.1 kHz sample rate
- 16-bit depth
- Stereo channels (configurable)
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

### Layer 3: Key Moments (Split into Two Sub-Layers)
**Redesigned as dual-layer system: biological heartbeat foundation + event-driven moments**

**Architecture:**
Layer 3 is split into two independent sub-layers with different stereo treatment:

#### Layer 3a: Heartbeat Sub-Drone (Biological Foundation)
**LUB-dub cardiac rhythm providing organic temporal anchor**

**Implementation:**
- Sine wave synthesis with precise LUB-dub pattern (from `heartbeat_designer.py`)
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

#### Layer 3b: Moment Sequencer (Event-Driven)
**Supersaw arpeggios triggered by key chess moments**

**Implementation:**
- Triggered ONLY by key moments (captures, blunders, brilliant moves, etc.)
- Supersaw synthesis with entropy-driven note selection
- Global filter sweeps for dramatic effect
- No continuous base pattern (heartbeat layer handles that role)

**Entropy-Driven Parameters:**
- **Note selection**: Low entropy = root-fifth, high = chromatic
- **Rhythm variation**: Low entropy = metronomic, high = irregular
- **Filter modulation**: Low entropy = slow sweeps, high = fast restless
- **Portamento**: Low entropy = smooth glides, high = jumpy nervous
- **Harmonic density**: High entropy = add cluster harmonies

```python
# Moment-triggered synthesis only
if entropy < 0.3:
    available_notes = [0, 4]  # Simple: root-fifth
elif entropy < 0.7:
    available_notes = [0, 2, 4, 5, 7]  # Moderate: diatonic
else:
    available_notes = [0, 1, 2, 3, 4, 5, 6, 7]  # Complex: chromatic

note = random.choice(available_notes)
```

**Audio Result:**
- **Event-reactive**: Only plays during key chess moments
- **Context-aware**: Entropy shapes how moments sound
- **Spatial movement**: Entropy-driven panning (L/R position)
- **Dramatic emphasis**: Supersaw richness highlights critical events

**Layer 3 Design Philosophy:**
The split addresses the original problem: Layer 3 needed both continuous presence (temporal foundation) and event emphasis (key moments). By separating these roles:
- **3a (heartbeat)**: Continuous, biological, centered, entropy-modulated tempo
- **3b (moments)**: Discrete, dramatic, spatially mobile, event-triggered

This creates musical depth while maintaining clarity of both continuous and discrete narrative elements.

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
✓ **Entropy-driven Layer 3**: Evaluation volatility controls note selection, rhythm, filters, portamento, harmony
✓ **Stereo output with entropy panning**: Layer 3 spatial position driven by evaluation entropy

### Possible Enhancements
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
