# Particle-Based Gesture System

## Overview

Particle-based gestures use stochastic spawning of independent sound events to create polyphonic textures that traditional curve-based gestures cannot achieve. Think wind chimes, rain, cricket swarms, or metallic collisions.

**5 archetypes** use this system: INACCURACY, FIRST_EXCHANGE, TACTICAL_SEQUENCE, SIGNIFICANT_SHIFT, FINAL_RESOLUTION

## When to Use Particles vs Curves

| Use Particles When... | Use Curves When... |
|-----------------------|--------------------|
| Sparse, discrete events | Continuous, evolving sound |
| Stochastic/unpredictable timing | Precise temporal control |
| Polyphonic independence needed | Monophonic or synchronized voices |
| Natural/organic textures | Musical/harmonic structures |
| Wind chimes, rain, clicks | Glissandi, drones, sweeps |

## Architecture

```
Archetype Config
       ↓
Emission Pattern Config
  (gusts, bursts, rhythmic_clusters, dissolve, etc.)
       ↓
ParticleEmitter
  ├─ Generate emission curve (temporal density [0-1])
  └─ Stochastic spawning (Poisson-style per-sample)
       ↓
List[Particle]
  Each particle:
    - birth_sample: when it spawns
    - lifetime_samples: how long it rings
    - pitch_hz: frequency
    - velocity: amplitude (0-1)
    - detune_cents: pitch variation
    - decay_rate: exponential decay speed
    - waveform: 'sine', 'triangle', etc.
       ↓
ParticleRenderer
  ├─ Synthesize each particle independently (SubtractiveSynth)
  └─ Equal-power mixing: sum / sqrt(N)
       ↓
Audio
```

## Components

### 1. Particle

Individual sound event (dataclass):

```python
@dataclass
class Particle:
    birth_sample: int         # When particle spawns (sample index)
    lifetime_samples: int     # How long particle rings (samples)
    pitch_hz: float           # Base frequency
    velocity: float           # Amplitude (0-1)
    detune_cents: float       # Pitch micro-variation (±50¢ typical)
    decay_rate: float         # Exponential decay (negative, e.g., -2.5)
    waveform: str             # 'sine', 'triangle', 'sawtooth'

    @property
    def death_sample(self) -> int:
        return self.birth_sample + self.lifetime_samples
```

Particles are independent - they don't know about each other and have completely separate lifetimes.

### 2. ParticleEmitter

Spawns particles using **density curves + Poisson randomness**:

**Key Concept:** Artistic control (when events happen) + natural chaos (exact timing)

```python
for sample_idx in range(total_samples):
    current_density = emission_curve[sample_idx]  # [0-1]
    spawn_probability = current_density * base_spawn_rate

    if random() < spawn_probability:
        spawn_particle_at(sample_idx)
```

This creates:
- **Predictable patterns** (emission curve shape)
- **Unpredictable micro-timing** (Poisson randomness)
- **Natural feel** (like real physical phenomena)

### 3. ParticleRenderer

Synthesizes and mixes all particles:

1. For each particle:
   - Generate envelope: exponential decay from birth to death
   - Generate oscillator: time-varying pitch (SubtractiveSynth)
   - Apply envelope
   - Insert into audio buffer at birth_sample

2. Equal-power mixing:
   ```python
   final_audio = sum(all_particles) / sqrt(num_particles)
   ```
   This prevents volume increase when many particles overlap.

### 4. ParticleGestureGenerator

High-level API matching `GestureGenerator`:
- Uses same archetype config pattern
- Compatible with GestureCoordinator
- Returns mono audio buffer

## Emission Patterns

Emission patterns define the **temporal density curve** that controls spawning probability over time.

### 1. Gusts (Wind Chimes)

```python
"emission": {
    "type": "gusts",
    "num_gusts": 2,           # Number of gust cycles
    "base_density": 0.015,    # Calm density (sparse)
    "peak_density": 0.06      # Gust peak density
}
```

**Behavior:**
- Starts at base_density
- Rises to peak_density
- Falls back to base
- Repeats for num_gusts cycles
- Creates wind-like bursts with calm between

**Used by:** INACCURACY

**Sound:** Sparse wind chime strikes with occasional gusts of activity

---

### 2. Impact Burst (Metallic Collision)

```python
"emission": {
    "type": "impact_burst",
    "impact_time_ratio": 0.15,     # When impact occurs (15% into gesture)
    "burst_density": 0.95,         # Very dense burst
    "burst_duration_ratio": 0.08,  # Short burst (8% of duration)
    "tail_density": 0.05           # Sparse tail after
}
```

**Behavior:**
- Low density at start
- Sharp spike to burst_density at impact_time
- Rapid decay to tail_density
- Sparse tail continues

**Used by:** FIRST_EXCHANGE

**Sound:** Sudden metallic collision scatter with long tail

---

### 3. Rhythmic Clusters (Calculation Clicks)

```python
"emission": {
    "type": "rhythmic_clusters",
    "num_clusters": 4,             # Number of click clusters
    "cluster_duration_ratio": 0.15, # Each cluster is 15% of duration
    "cluster_density": 0.7,        # Dense during cluster
    "gap_density": 0.05            # Sparse between clusters
}
```

**Behavior:**
- Alternates between high-density clusters and low-density gaps
- Regular rhythm: cluster → gap → cluster → gap → ...
- num_clusters determines rhythm subdivision

**Used by:** TACTICAL_SEQUENCE

**Sound:** Rhythmic bursts of clicks (like abacus or calculation)

---

### 4. Drift Scatter (Sparse Wandering)

```python
"emission": {
    "type": "drift_scatter",
    "start_density": 0.08,
    "end_density": 0.15,
    "drift_rate": 0.3
}
```

**Behavior:**
- Slowly increases density from start to end
- Gradual accumulation of events
- Smooth transition

**Used by:** SIGNIFICANT_SHIFT

**Sound:** Sparse particles gradually increasing in frequency

---

### 5. Dissolve (Fading Away)

```python
"emission": {
    "type": "dissolve",
    "start_density": 0.6,
    "decay_rate": -1.8
}
```

**Behavior:**
- Starts with high density
- Exponential decay toward zero
- Fading/dissolving effect

**Used by:** FINAL_RESOLUTION

**Sound:** Dense texture gradually dissolving to silence

---

## Particle Configuration

Each archetype specifies particle properties (sampled randomly within ranges):

```python
"particle": {
    "emission": {...},                  # Emission pattern (see above)
    "base_spawn_rate": 0.001,          # Spawn probability multiplier
    "pitch_range_hz": [880, 1760],     # Random pitch range
    "lifetime_range_s": [1.2, 2.5],    # Random lifetime range
    "velocity_range": [0.35, 0.55],    # Random amplitude range
    "detune_range_cents": [-30, 30],   # Random pitch detune
    "decay_rate_range": [-2.5, -1.5],  # Random decay speed
    "waveform": "triangle"             # Oscillator waveform
}
```

Each spawned particle randomly samples within these ranges, creating natural variation.

## Example: INACCURACY Archetype

```python
"INACCURACY": {
    "duration_base": 4.5,
    "duration_tension_scale": 0.0,
    "duration_entropy_scale": 0.0,
    "particle": {
        "emission": {
            "type": "gusts",
            "num_gusts": 2,
            "base_density": 0.015,
            "peak_density": 0.06
        },
        "base_spawn_rate": 0.001,
        "pitch_range_hz": [880, 1760],      # High metallic range
        "lifetime_range_s": [1.2, 2.5],     # Long ringing
        "velocity_range": [0.35, 0.55],     # Moderate volume
        "detune_range_cents": [-30, 30],    # Subtle pitch variation
        "decay_rate_range": [-2.5, -1.5],   # Medium decay
        "waveform": "triangle"              # Warm timbre
    },
    "peak_limit": 0.6,
    "rms_target": -20.0
}
```

**Result:**
- 4.5 second duration
- ~9 particles total (sparse)
- Two gusts of activity with calm between
- High-pitched metallic strikes (880-1760 Hz)
- Long ringing tails (1.2-2.5s)
- Natural pitch micro-variation (±30¢)
- Triangle wave for warm timbre

## Testing Particle Archetypes

```bash
# Analyze particle behavior (no audio)
python3 tools/particle_test.py INACCURACY

# Analyze and generate audio
python3 tools/particle_test.py FIRST_EXCHANGE --audio

# Custom output
python3 tools/particle_test.py TACTICAL_SEQUENCE --audio -o test.wav
```

Output shows:
- Total particles spawned
- Spawning timeline
- Pitch/velocity/lifetime statistics
- Emission curve visualization
- Temporal distribution histogram

## Creating New Emission Patterns

### 1. Add Pattern Generator

File: `particle_system.py`

```python
def _generate_emission_my_pattern(
    config: Dict[str, Any],
    total_samples: int,
    section_context: Dict[str, Any]
) -> np.ndarray:
    """
    My custom emission pattern.

    Returns:
        Emission curve [0-1] of length total_samples
    """
    # Extract config
    start_density = config.get('start_density', 0.1)
    end_density = config.get('end_density', 0.5)

    # Generate curve
    t = np.linspace(0, 1, total_samples)
    emission_curve = start_density + (end_density - start_density) * t

    return emission_curve
```

### 2. Register in Generator

```python
def _generate_emission_curve(self, emission_config, total_samples, section_context):
    emission_type = emission_config['type']

    if emission_type == "my_pattern":
        return _generate_emission_my_pattern(emission_config, total_samples, section_context)
    # ... other patterns
```

### 3. Use in Archetype

```python
"MY_ARCHETYPE": {
    "duration_base": 3.0,
    "particle": {
        "emission": {
            "type": "my_pattern",
            "start_density": 0.1,
            "end_density": 0.5
        },
        "base_spawn_rate": 0.002,
        "pitch_range_hz": [440, 880],
        "lifetime_range_s": [0.5, 1.5],
        "velocity_range": [0.3, 0.7],
        "detune_range_cents": [-20, 20],
        "decay_rate_range": [-3.0, -1.5],
        "waveform": "sine"
    },
    "peak_limit": 0.7,
    "rms_target": -18.0
}
```

## Design Considerations

### Density Tuning

Emission curve values [0-1] are **multiplied** by `base_spawn_rate` to get per-sample spawn probability:

```python
spawn_probability = emission_curve[i] * base_spawn_rate
```

Typical values:
- **base_spawn_rate**: 0.0005 - 0.005
- **emission peak**: 0.05 - 0.95
- **Result**: ~0.0001 - 0.005 per-sample probability

Higher values → denser textures (more particles)

### Pitch Range

Consider timbre and register:
- **Low (110-440 Hz)**: Warm, mellow, drone-like
- **Mid (440-880 Hz)**: Balanced, musical
- **High (880-2200 Hz)**: Bright, metallic, bell-like
- **Very high (2200+ Hz)**: Harsh, crystalline, glassy

### Lifetime vs Decay

Two parameters control how long particles are audible:

1. **lifetime_range_s**: Physical duration
2. **decay_rate_range**: Exponential decay speed (-5.0 = fast, -1.0 = slow)

Short lifetime + slow decay = truncated sustain
Long lifetime + fast decay = long ringing tail

Typical combinations:
- **Chimes**: Long lifetime (1-3s) + medium decay (-2 to -1.5)
- **Clicks**: Short lifetime (0.2-0.5s) + fast decay (-5 to -3)
- **Rain**: Medium lifetime (0.5-1s) + medium decay (-3 to -2)

### Waveform Choice

- **sine**: Pure, mellow, drone-like
- **triangle**: Warm, soft harmonics
- **sawtooth**: Bright, buzzy, more harmonics
- **square**: Hollow, reedy, odd harmonics

## Implementation Status

✅ **All 5 particle archetypes work fully**

Particle system is complete and production-ready. All emission patterns implemented:
- gusts (INACCURACY)
- impact_burst (FIRST_EXCHANGE)
- rhythmic_clusters (TACTICAL_SEQUENCE)
- drift_scatter (SIGNIFICANT_SHIFT)
- dissolve (FINAL_RESOLUTION)

Unlike curve-based system, **no missing implementations**.
