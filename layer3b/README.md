# Layer3b: Chess Moment Gesture System

## Overview

Layer3b translates chess game moments into expressive sound gestures. Each significant chess event (brilliant move, blunder, tactical sequence, checkmate, etc.) is mapped to a sonic gesture archetype that captures its musical and narrative character.

**Core Concept:** Chess moments → Gesture archetypes → Synthesized audio

## Architecture

```
Chess Moment Event
       ↓
GestureCoordinator
       ↓
GestureGenerator (curve-based or particle-based)
       ↓
Audio Curves / Particles
       ↓
GestureSynthesizer / ParticleRenderer
       ↓
Synthesized Audio
```

For architectural details, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Gesture Types

Layer3b supports two fundamentally different gesture synthesis approaches:

### 1. Curve-Based Gestures (25 archetypes)

Traditional sound design using time-varying curves:
- **Pitch curves**: Glissandi, drones, tremors, arcs, etc.
- **Harmony**: Multi-voice clusters, chords, intervals
- **Filter curves**: Spectral shaping (sweeps, focus, opening/closing)
- **Envelope**: Amplitude shaping over time
- **Texture**: Noise blending, shimmer effects

**Examples:** MOVE, BRILLIANT, BLUNDER, CHECKMATE, CASTLING

See [`docs/CURVE_SYSTEM.md`](docs/CURVE_SYSTEM.md) for details.

### 2. Particle-Based Gestures (5 archetypes)

Stochastic polyphonic textures using independent sound particles:
- **Particles**: Independent sound events with pitch, velocity, lifetime, decay
- **Emission patterns**: Gusts, bursts, rhythmic clusters, dissolves
- **Stochastic timing**: Density curves + Poisson randomness

**Examples:** INACCURACY, FIRST_EXCHANGE, TACTICAL_SEQUENCE, FINAL_RESOLUTION

See [`docs/PARTICLE_SYSTEM.md`](docs/PARTICLE_SYSTEM.md) for details.

## Archetype Configuration

All archetypes are defined in `archetype_configs.py` using declarative configuration (not code):

```python
"BRILLIANT": {
    "system": "curve",  # Explicit system type: "curve" or "particle"
    "duration_base": 3.3,
    "duration_tension_scale": 0.8,
    "duration_entropy_scale": -0.5,
    "phases": {
        "pre_shadow": 0.16,
        "impact": 0.10,
        "bloom": 0.48,
        "decay": 0.16,
        "residue": 0.10
    },
    "pitch": {
        "type": "ascending_spread",
        "start_freq_base": 220,
        "octave_rise": 2.2,
        ...
    },
    "harmony": {...},
    "filter": {...},
    "envelope": {...},
    "texture": {...},
    "peak_limit": 0.86,
    "rms_target": -15.5
}
```

**System Field**: Each archetype explicitly declares its gesture system type:
- `"system": "curve"` - Uses curve-based synthesis (25 archetypes)
- `"system": "particle"` - Uses particle-based synthesis (5 archetypes)

### Context Responsiveness

Gestures adapt to chess game context:
- **Tension** (0-1): Section-level dramatic intensity
- **Entropy** (0-1): Positional complexity

Parameters can scale with context:
```python
"duration_tension_scale": 0.8  # Duration increases with tension
"start_freq_entropy_scale": 110  # Pitch varies with entropy
```

## Testing Individual Archetypes

### Curve-Based Gestures

```bash
# Test a specific archetype
python3 tools/gesture_test.py BRILLIANT

# With custom context
python3 tools/gesture_test.py CHECKMATE --tension 0.9 --entropy 0.3

# List all curve-based archetypes
python3 tools/gesture_test.py --list
```

Generates WAV file: `gesture_test_<archetype>.wav`

### Particle-Based Gestures

```bash
# Analyze particle behavior
python3 tools/particle_test.py INACCURACY

# Analyze and generate audio
python3 tools/particle_test.py FIRST_EXCHANGE --audio

# Custom output filename
python3 tools/particle_test.py INACCURACY --audio -o test.wav

# List all particle archetypes
python3 tools/particle_test.py --list
```

Shows particle spawning timeline, emission curves, and musical characteristics.

## Creating a New Archetype

1. **Choose gesture type**: Curve-based or particle-based?
   - Use curves for continuous, evolving sounds
   - Use particles for sparse, stochastic textures

2. **Add configuration** to `archetype_configs.py`:
   ```python
   "MY_ARCHETYPE": {
       "duration_base": 2.5,
       "duration_tension_scale": 0.5,
       "duration_entropy_scale": 0.0,
       # ... rest of config
   }
   ```

3. **Test it**:
   ```bash
   python3 tools/gesture_test.py MY_ARCHETYPE
   # or
   python3 tools/particle_test.py MY_ARCHETYPE --audio
   ```

4. **Iterate** on parameters until it sounds right

## Implementation Status

### ✅ Fully Implemented
- Particle system (all 5 archetypes work)
- 9 pitch curve types
- 8 harmony types
- 8 filter types
- 7 envelope types

### ⚠️ Missing Implementations

**20 pitch curve types** used by archetypes are not yet implemented:
- `stable`, `sustained_drone`, `parabolic`, `converging_steps`
- `reciprocal_pair`, `dual_iteration`, `focused_center`, `impact_transient`
- `divergent_split`, `gentle_descent`, `chaotic_iteration`, `dual_descent`
- `final_descent`, `stable_rise`, `aggressive_rise`, `ascending_burst`
- `low_linear_rise`, `stable_center`, `parabolic_arc`, `gradual_emergence`
- `forceful_rise`

**Harmony types**: Several referenced types need implementation
**Filter types**: Several referenced types need implementation

This means **most curve-based archetypes cannot be synthesized yet**.

See [`docs/CURVE_SYSTEM.md`](docs/CURVE_SYSTEM.md) for complete implementation status.

## Project Structure

```
layer3b/
├── README.md                    # This file
├── __init__.py                  # Package exports
├── archetype_configs.py         # All 30 archetype definitions
├── base.py                      # Base GestureGenerator class
├── coordinator.py               # GestureCoordinator (routes to generators)
├── curve_generators.py          # Pitch/harmony/filter/envelope generators
├── particle_system.py           # Particle system implementation
├── synthesizer.py               # GestureSynthesizer (audio rendering)
├── utils.py                     # Utility functions
└── docs/
    ├── ARCHITECTURE.md          # System architecture details
    ├── CURVE_SYSTEM.md          # Curve-based gesture design
    └── PARTICLE_SYSTEM.md       # Particle-based gesture design
```

## References

- Spectromorphology: Denis Smalley's gesture theory
- Algorithmic composition: Laurie Spiegel, Jean-Michel Jarre
- Particle systems: Game engine particle systems adapted for audio
