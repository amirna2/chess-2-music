# Particle System for Sound Gestures

## Overview

The particle system enables stochastic, polyphonic sound textures like wind chimes, rain, cricket swarms, and other natural phenomena that traditional monolithic gestures cannot simulate.

## Architecture

### Key Components

1. **`Particle`** - Individual sound event with independent properties:
   - `birth_sample`: When particle spawns
   - `lifetime_samples`: How long it rings
   - `pitch_hz`: Base frequency
   - `velocity`: Amplitude (0-1)
   - `detune_cents`: Pitch micro-variation (±50¢ typical)
   - `decay_rate`: Exponential decay speed (negative, e.g., -2.5)
   - `waveform`: Oscillator type ('sine', 'triangle')

2. **`ParticleEmitter`** - Spawns particles using density curves + Poisson randomness:
   - Density curve controls spawn probability over time [0-1]
   - Random Poisson-style timing within probability field
   - Result: Artistic control (when events happen) + natural chaos (exact timing)

3. **`ParticleRenderer`** - Synthesizes and mixes independent particle audio:
   - Each particle rendered independently with SubtractiveSynth
   - Equal-power mixing prevents volume increase with particle count
   - Particles overlap naturally with independent lifetimes

4. **`ParticleGestureGenerator`** - High-level API matching `GestureGenerator`:
   - Uses same archetype config pattern
   - Compatible with existing Layer3b coordinator
   - Mono output (stereo panning handled upstream)

## Emission Patterns

### 1. Gusts (Wind Chimes)
```python
"emission": {
    "type": "gusts",
    "num_gusts": 2,           # Number of gust cycles
    "base_density": 0.05,     # Calm density
    "peak_density": 0.3       # Gust peak density
}
```
Creates wind-like bursts of activity with calm periods between.

### 2. Constant (Rain)
```python
"emission": {
    "type": "constant",
    "density": 0.2            # Constant spawn rate
}
```
Steady, uniform particle spawning (like rainfall).

### 3. Swell (Crescendo)
```python
"emission": {
    "type": "swell",
    "start_density": 0.05,    # Initial density
    "end_density": 0.9        # Final density
}
```
Gradual increase in particle density over time.

### 4. Decay Scatter (Falling Debris)
```python
"emission": {
    "type": "decay_scatter",
    "start_density": 0.9,     # Initial burst
    "decay_rate": -2.5        # Exponential decay rate
}
```
High initial density that exponentially decays (like objects settling).

## Creating Particle Archetypes

Add to `archetype_configs.py`:

```python
"MY_PARTICLE_GESTURE": {
    "duration_base": 4.0,
    "duration_tension_scale": 0.5,
    "duration_entropy_scale": 0.0,

    # Particle-specific config
    "particle": {
        # Emission pattern
        "emission": {
            "type": "gusts",
            "num_gusts": 2,
            "base_density": 0.05,
            "peak_density": 0.3
        },

        # Spawning parameters
        "base_spawn_rate": 0.001,          # Base probability multiplier

        # Particle property ranges (randomized per particle)
        "pitch_range_hz": [440, 880],      # Pitch range
        "lifetime_range_s": [1.0, 2.5],    # Lifetime range (seconds)
        "velocity_range": [0.2, 0.8],      # Amplitude range
        "detune_range_cents": [-30, 30],   # Pitch variation (cents)
        "decay_rate_range": [-3.0, -1.0],  # Decay speed range
        "waveform": "triangle"             # Oscillator waveform
    },

    # Audio finalization
    "peak_limit": 0.4,
    "rms_target": -28.0,

    # Metadata
    "morphology": {
        "spectromorphological_archetype": "Particle System",
        "gesture_class": "Stochastic / Polyphonic",
        "motion_type": "Independent discrete events"
    }
}
```

## Usage Examples

### Basic Usage
```python
from layer3b.particle_system import ParticleGestureGenerator
from layer3b.archetype_configs import ARCHETYPES
from synth_engine import SubtractiveSynth
import numpy as np

# Setup
sample_rate = 88200
rng = np.random.default_rng(seed=42)
synth = SubtractiveSynth(sample_rate=sample_rate)

# Create generator
archetype_config = ARCHETYPES['WIND_CHIMES']
generator = ParticleGestureGenerator(archetype_config, rng, synth)

# Generate gesture
moment_event = {'event_type': 'WIND_CHIMES', 'timestamp': 0.0}
section_context = {'tension': 0.5, 'entropy': 0.5}
audio = generator.generate_gesture(moment_event, section_context, sample_rate)
```

### Testing Different Patterns
See `demo_particle_patterns.py` for examples of all emission types.

### Visualizing Behavior
Use `visualize_particles.py` to see particle spawning timeline and emission curves.

## Design Philosophy

### Why Particle System?

Traditional gesture generation creates a **single monolithic sound** with pre-computed parameters. This works for continuous gestures (sweeps, drones, swooshes) but fails for:

- **Stochastic textures**: Wind chimes, rain, crickets
- **Polyphonic independence**: Each sound event needs its own lifetime
- **Natural randomness**: Real-world phenomena have unpredictable timing
- **Temporal evolution**: Density changes over time (gusts, swells)

### Inspiration

Inspired by game engine particle systems (Unity, Unreal) adapted for audio:
- Video game particles: Visual effects (fire, smoke, sparks)
- Audio particles: Sound events (chimes, raindrops, chirps)

### Key Differences from Traditional Gestures

| Aspect | Traditional Gesture | Particle System |
|--------|-------------------|-----------------|
| Structure | Monolithic | Independent events |
| Timing | Pre-computed | Stochastic spawning |
| Parameters | Global curves | Per-particle random |
| Polyphony | Harmony voices | Overlapping lifetimes |
| Control | Deterministic | Curve-based + random |

## Technical Details

### Spawning Algorithm
```python
for sample_idx in range(total_samples):
    current_density = emission_curve[sample_idx]  # 0.0 to 1.0
    spawn_probability = current_density * base_spawn_rate

    if rng.random() < spawn_probability:
        spawn_particle(birth_time=sample_idx, ...)
```

### Rendering Pipeline
1. For each particle:
   - Generate oscillator audio (constant pitch, no glide)
   - Apply exponential decay envelope
   - Apply velocity scaling
2. Place particle audio at birth_sample position
3. Mix all overlapping particles (equal-power summing)
4. Finalize with peak limiting

### Performance Considerations
- Particles rendered sequentially (not real-time)
- Memory scales with particle count (typical: 10-100 particles)
- CPU scales linearly with total particle lifetime
- Equal-power normalization prevents clipping with many particles

## Future Enhancements

### Potential Additions
1. **LFO Modulation**: Per-particle pitch/amplitude wobble
2. **FM Synthesis**: Metallic timbres for realistic chimes
3. **Spatial Panning**: Per-particle stereo positioning
4. **Physical Behaviors**: Gravity, damping, collision metaphors
5. **Burst Events**: Manual particle bursts at specific times
6. **Scale Quantization**: Snap particles to musical scales

### Adding New Emission Types

Add to `ParticleGestureGenerator._generate_emission_curve()`:

```python
elif emission_type == 'my_custom_pattern':
    # Your custom emission logic here
    # Return np.ndarray of shape (total_samples,) with values [0-1]
    curve = ...
    return curve.astype(np.float32)
```

## Troubleshooting

### Too Many/Few Particles
- Adjust `base_spawn_rate` (higher = more particles)
- Adjust emission curve density values

### Particles Too Loud/Quiet
- Adjust `velocity_range`
- Adjust `peak_limit` and `rms_target`

### Particles Sound Unnatural
- Increase `detune_range_cents` for more variation
- Vary `lifetime_range_s` for irregular decay timing
- Add more randomness to emission curve

### No Particles Spawning
- Check `base_spawn_rate` is not too low
- Check emission curve has non-zero density
- Verify duration is sufficient for spawning window

## References

- **Spectromorphology**: Denis Smalley's gesture theory
- **Particle Systems**: William T. Reeves (1983) - Original computer graphics paper
- **Algorithmic Composition**: Laurie Spiegel's work on generative music
- **Game Engine Particles**: Unity/Unreal particle system documentation
