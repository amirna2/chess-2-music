# Layer3b System Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Chess Moment Event                        │
│  { type: "BRILLIANT", timestamp: 45.2, score: 8, ... }      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   GestureCoordinator                         │
│  Routes moment to appropriate generator based on config     │
└───────────┬───────────────────────────────────┬─────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌───────────────────────────┐
│  GestureGenerator     │         │  ParticleGestureGenerator │
│  (curve-based)        │         │  (particle-based)         │
│                       │         │                           │
│  • CurveGenerators    │         │  • ParticleEmitter        │
│    - Pitch            │         │  • Emission patterns      │
│    - Harmony          │         │  • Stochastic spawning    │
│    - Filter           │         │                           │
│    - Envelope         │         │                           │
│  • Phase system       │         │                           │
│  • Context scaling    │         │                           │
└──────────┬────────────┘         └───────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌───────────────────────┐         ┌───────────────────────────┐
│  GestureSynthesizer   │         │   ParticleRenderer        │
│  (multi-voice synth)  │         │   (per-particle synth)    │
└──────────┬────────────┘         └───────────┬───────────────┘
           │                                  │
           └──────────┬───────────────────────┘
                      ▼
           ┌──────────────────────┐
           │  SubtractiveSynth    │
           │  (DSP primitives)    │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Mono Audio Buffer   │
           │  (numpy array)       │
           └──────────────────────┘
```

## Components

### 1. GestureCoordinator

**File:** `coordinator.py`

**Purpose:** Routes chess moment events to the appropriate gesture generator.

**Key Methods:**
- `generate_gesture(archetype_name, moment_event, section_context, sample_rate)` → audio

**Logic:**
```python
system_type = archetype_config['system']  # Explicit field: "curve" or "particle"
if system_type == 'particle':
    return ParticleGestureGenerator(...)
elif system_type == 'curve':
    return GestureGenerator(...)
else:
    raise ValueError(f"Unknown system type: {system_type}")
```

**Responsibilities:**
- Archetype lookup
- Generator instantiation
- Delegation to appropriate generator type

---

### 2. GestureGenerator (Curve-Based)

**File:** `base.py`

**Purpose:** Generates curve-based gestures using time-varying parameter curves.

**Data Flow:**
```
Archetype Config
      ↓
Duration Calculation (base + tension/entropy scaling)
      ↓
Phase Calculation (pre_shadow, impact, bloom, decay, residue)
      ↓
Curve Generation (pitch, harmony, filter, envelope, texture)
      ↓
GestureSynthesizer.synthesize()
      ↓
Audio
```

**Key Methods:**
- `generate_gesture(moment_event, section_context, sample_rate)` → audio
- `_compute_duration(section_context)` → duration_seconds
- `_compute_phase_boundaries(total_samples)` → phase_dict

**Phase System:**

Gestures are divided into 5 temporal phases:
- **pre_shadow** (0-15%): Anticipatory buildup
- **impact** (10-20%): Initial attack/onset
- **bloom** (20-50%): Peak activity/expansion
- **decay** (15-40%): Energy dissipation
- **residue** (10-20%): Trailing sustain/echo

Each phase ratio is configurable per archetype. Curve generators use phase boundaries to shape their trajectories.

---

### 3. ParticleGestureGenerator

**File:** `particle_system.py`

**Purpose:** Generates particle-based gestures using stochastic spawning.

**Data Flow:**
```
Particle Config
      ↓
Duration Calculation
      ↓
Emission Curve Generation (temporal density)
      ↓
ParticleEmitter.emit_particles() (stochastic spawning)
      ↓
Particle List (birth_sample, pitch_hz, lifetime, velocity, ...)
      ↓
ParticleRenderer.render() (per-particle synthesis)
      ↓
Audio
```

**Key Components:**

#### Particle
Individual sound event:
```python
@dataclass
class Particle:
    birth_sample: int         # When it spawns
    lifetime_samples: int     # How long it rings
    pitch_hz: float           # Base frequency
    velocity: float           # Amplitude (0-1)
    detune_cents: float       # Pitch micro-variation
    decay_rate: float         # Exponential decay (negative)
    waveform: str             # 'sine', 'triangle', etc.
```

#### ParticleEmitter
Spawns particles using density curve + randomness:
- **Emission curve**: Temporal density [0-1] controlling spawn probability
- **Poisson-style timing**: Random per-sample spawn decisions
- **Result**: Artistic control (when) + natural chaos (exact timing)

#### ParticleRenderer
Synthesizes and mixes independent particles:
- Each particle rendered separately with SubtractiveSynth
- Equal-power mixing: `sum(particles) / sqrt(N)` prevents volume increase
- Independent overlapping lifetimes

---

### 4. GestureSynthesizer (Core Audio Engine)

**File:** `synthesizer.py`

**Purpose:** Multi-voice time-varying synthesis engine. Wraps `SubtractiveSynth` primitives to coordinate gesture audio rendering.

**Design Pattern:** Thin wrapper around SubtractiveSynth
- Delegates all DSP primitives to synth engine
- Provides gesture-level coordination logic
- Maintains architectural consistency with other layers

**Main Method:**
```python
def synthesize(
    pitch_voices: List[np.ndarray],  # List of pitch curves (Hz)
    filter_curve: Dict,               # Cutoff/resonance curves
    envelope: np.ndarray,             # Amplitude envelope (0-1)
    texture_curve: Dict,              # Noise ratio/type
    sample_rate: int
) -> np.ndarray
```

**Synthesis Pipeline:**

```
1. Validation
   ├─ Sample rate matches
   ├─ Non-empty pitch_voices
   └─ All curves same length
           ↓
2. Oscillator Generation
   ├─ For each pitch voice:
   │  └─ SubtractiveSynth.oscillator_timevarying_pitch()
   └─ Returns: List[np.ndarray]
           ↓
3. Equal-Power Voice Mixing
   ├─ Sum all voices
   └─ Normalize: sum / sqrt(num_voices)
           ↓
4. Noise Texture Blending
   ├─ Generate noise: SubtractiveSynth.generate_noise()
   └─ Mix: (1 - ratio) * oscillators + ratio * noise
           ↓
5. Time-Varying Filter
   └─ SubtractiveSynth.moog_filter_timevarying()
           ↓
6. Amplitude Envelope
   └─ Element-wise multiply: audio *= envelope
           ↓
7. Shimmer Effect (optional)
   └─ Amplitude modulation: 0.5 + 0.5 * sin(2π * rate * t)
           ↓
8. Return Audio
```

**Input Structures:**

```python
filter_curve = {
    'cutoff': np.ndarray,      # Cutoff frequency (Hz)
    'resonance': np.ndarray,   # Resonance (0-4)
    'type': str                # Documentation only
}

texture_curve = {
    'noise_ratio': float,      # Mix ratio (0-1)
    'noise_type': str,         # 'white' | 'pink' | 'brown'
    'shimmer_enable': bool,    # Optional
    'shimmer_rate_hz': float   # Optional
}
```

**Key Design Decisions:**

1. **Multi-voice support**: List of pitch curves for polyphonic gestures
2. **Equal-power mixing**: Prevents volume increase with voice count
3. **Time-varying everything**: All parameters can change over time
4. **Thin wrapper**: All DSP delegated to SubtractiveSynth
5. **Reusable**: Same engine for all curve-based archetypes

---

### 5. CurveGenerators

**File:** `curve_generators.py`

**Purpose:** Generate time-varying parameter curves from archetype configs.

**Architecture:**
```python
generate_pitch_curve(config, total_samples, phases, section_context)
    ├─ Routes to _pitch_<type>() based on config['pitch']['type']
    └─ Returns: List[np.ndarray] (one per voice)

generate_harmony_curve(config, total_samples, phases, section_context)
    ├─ Routes to _harmony_<type>()
    └─ Returns: List[np.ndarray] (pitch curves for all voices)

generate_filter_curve(config, total_samples, phases, section_context)
    ├─ Routes to _filter_<type>()
    └─ Returns: Dict with 'cutoff' and 'resonance' curves

generate_envelope_curve(config, total_samples, phases, section_context)
    ├─ Routes to _envelope_<type>()
    └─ Returns: np.ndarray (amplitude 0-1)
```

**Curve Types:**
- **Pitch**: ~29 types (9 implemented)
- **Harmony**: ~25 types (8 implemented)
- **Filter**: ~25 types (8 implemented)
- **Envelope**: ~7 types (7 implemented)

See [`CURVE_SYSTEM.md`](CURVE_SYSTEM.md) for implementation details.

---

### 6. SubtractiveSynth (DSP Primitives)

**File:** `../synth_engine.py` (project root)

**Purpose:** Core synthesis primitives shared across all layers.

**Key Methods Used by Layer3b:**
- `oscillator_timevarying_pitch(pitch_curve, waveform)` → audio
- `generate_noise(duration_samples, noise_type)` → audio
- `moog_filter_timevarying(audio, cutoff_curve, resonance_curve, filter_type)` → audio

**Features:**
- Anti-aliasing (polyBLEP)
- Click prevention
- Multiple waveforms (sine, triangle, sawtooth, square)
- Noise types (white, pink, brown)
- Moog-style resonant filters

---

## Data Flow Example: BRILLIANT Moment

```
1. Chess Event
   { type: "BRILLIANT", timestamp: 45.2, score: 8 }

2. GestureCoordinator
   ├─ Lookup: ARCHETYPES['BRILLIANT']
   ├─ Detect: curve-based (no 'particle' key)
   └─ Create: GestureGenerator(config)

3. GestureGenerator
   ├─ Duration: 3.3s base + (0.8 × tension) - (0.5 × entropy)
   │           = 3.3 + 0.56 - 0.25 = 3.61s
   │
   ├─ Phases:  pre_shadow=0.16, impact=0.10, bloom=0.48, decay=0.16, residue=0.10
   │           = [0s-0.58s, 0.58s-0.94s, 0.94s-2.68s, 2.68s-3.26s, 3.26s-3.61s]
   │
   ├─ Pitch:   ascending_spread (220Hz → 880Hz, exponential rise in bloom phase)
   │
   ├─ Harmony: unison_to_chord (1 voice → 4 voices major_seventh in bloom)
   │
   ├─ Filter:  lowpass_to_highpass_open (300Hz LP → 3800Hz HP in bloom)
   │
   └─ Envelope: gradual_sustained (40ms attack, 54% sustain, linear decay)

4. GestureSynthesizer
   ├─ Generate 4 oscillators (ascending pitch curves)
   ├─ Mix with equal-power normalization
   ├─ Add 6% white noise
   ├─ Apply filter sweep (LP → HP)
   ├─ Apply envelope
   └─ Add shimmer modulation (7.5 Hz)

5. Audio Output
   3.61s mono audio buffer (318,042 samples @ 88.2kHz)
```

---

## Design Principles

### 1. Configuration Over Code
All gesture archetypes defined declaratively in `archetype_configs.py`. No hardcoded synthesis logic per archetype.

### 2. Context Responsiveness
Gestures adapt to chess game state via tension/entropy scaling:
```python
duration = duration_base + (tension_scale × tension) + (entropy_scale × entropy)
start_freq = start_freq_base + (entropy_scale × entropy)
```

### 3. Separation of Concerns
- **Coordinator**: Routing logic
- **Generator**: Curve/particle generation
- **Synthesizer**: Audio rendering
- **CurveGenerators**: Parameter trajectory math
- **SubtractiveSynth**: DSP primitives

### 4. Reusability
- Single `GestureSynthesizer` for all curve-based archetypes
- Single `ParticleRenderer` for all particle archetypes
- Shared `SubtractiveSynth` across all project layers

### 5. Phase-Based Temporal Structure
All curve-based gestures use common phase structure for coherent temporal organization.

---

## File Organization

```
layer3b/
├── coordinator.py           # GestureCoordinator (routing)
├── base.py                  # GestureGenerator (curve-based)
├── curve_generators.py      # Pitch/harmony/filter/envelope generators
├── particle_system.py       # Particle system (stochastic)
├── synthesizer.py           # GestureSynthesizer (audio rendering)
└── utils.py                 # Shared utilities
```

All components follow the same pattern:
1. Read archetype config
2. Apply context (tension/entropy)
3. Generate time-domain data (curves or particles)
4. Synthesize audio via SubtractiveSynth primitives
5. Return mono audio buffer
