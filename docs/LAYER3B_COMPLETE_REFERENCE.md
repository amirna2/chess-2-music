# Layer 3b: Gesture Archetype System
## Configuration Structure Reference

**This document describes the ACTUAL structure in `layer3b/archetype_configs.py`.**

For complete archetype definitions, see the source file directly.

---

## Two Distinct Systems

### Curve-Based Gestures (`"system": "curve"`)
Deterministic, phase-structured, full parameter control.

### Particle-Based Gestures (`"system": "particle"`)
Stochastic, emission-controlled, polyphonic.

**These have COMPLETELY DIFFERENT configuration structures!**

---

## Curve-Based Configuration Structure

### Root-Level Parameters

```python
"ARCHETYPE_NAME": {
    "system": "curve",  # Always "curve"
    "duration_base": float,  # Base duration in seconds
    "duration_tension_scale": float,  # Tension scaling
    "duration_entropy_scale": float,  # Entropy scaling
    "phases": { ... },
    "pitch": { ... },
    "harmony": { ... },
    "filter": { ... },
    "envelope": { ... },
    "texture": { ... },
    "peak_limit": float,  # Max amplitude (0-1)
    "rms_target": float   # Target RMS in dB
}
```

---

### Phases (Curve-Based Only)

Five phases that MUST sum to 1.0:

```python
"phases": {
    "pre_shadow": 0.10,  # Anticipation
    "impact": 0.15,      # Attack
    "bloom": 0.45,       # Expansion
    "decay": 0.20,       # Fall
    "residue": 0.10      # Afterglow
}
```

---

### Pitch Section (Curve-Based)

**Structure**:
```python
"pitch": {
    "type": "TYPE_NAME",
    # ... type-specific parameters
}
```

**Example Types**:
- `"stable"`: Fixed pitch with optional entropy scaling
- `"ascending_spread"`: Rising glissando with phase control
- `"parabolic"`: Arc-shaped pitch curve
- `"converging_steps"`: Discrete step sequence
- `"exponential_gliss"`: Exponential pitch slide
- `"oscillating_tremor"`: Pitch vibrato

**See actual archetypes for complete list.**

---

### Harmony Section (Curve-Based)

**Structure**:
```python
"harmony": {
    "type": "TYPE_NAME",
    # ... type-specific parameters
}
```

**Example Types**:
- `"simple_unison"`: Single or doubled voice
- `"unison_to_chord"`: Expanding to chord structure
- `"cluster_to_interval"`: Cluster resolving to interval
- `"dense_cluster"`: Tight semitone groupings
- `"converging_cluster"`: Multiple voices converging

**See actual archetypes for complete list.**

---

### Filter Section (Curve-Based)

**Structure**:
```python
"filter": {
    "type": "TYPE_NAME",
    # ... type-specific parameters
}
```

**Example Types**:
- `"simple_lowpass"`: Static lowpass filter
- `"lowpass_to_highpass_open"`: Filter morph with phase control
- `"bandpass_sweep"`: Moving bandpass center
- `"bandpass_to_lowpass_choke"`: Closing spectral focus
- `"focusing_narrowband"`: Narrowing bandwidth

**See actual archetypes for complete list.**

---

### Envelope Section (Curve-Based)

**Three envelope types exist**:

#### 1. sudden_short_tail
```python
"envelope": {
    "type": "sudden_short_tail",
    "attack_ms_base": float,           # Base attack time
    "attack_ms_entropy_scale": float,  # Entropy influence
    "sustain_phase_ratio": float,      # Sustain portion
    "decay_curve": str,                # "exponential" or "linear"
    "decay_coefficient": float         # Decay steepness (-6.0 to -1.0)
}
```

#### 2. gradual_sustained
```python
"envelope": {
    "type": "gradual_sustained",
    "attack_ms": float,
    "sustain_phase_ratio": float,
    "decay_curve": str  # "linear", "exponential", "sigmoid"
}
```

#### 3. gated_pulse
```python
"envelope": {
    "type": "gated_pulse",
    "attack_ms": float,
    "gate_duration_ms": float,
    "release_ms": float,
    "pulse_rate_hz": float
}
```

---

### Texture Section (Curve-Based)

**IMPORTANT**: Curve-based textures do NOT have `waveform` field!

```python
"texture": {
    "noise_ratio_base": float,           # 0.0-1.0
    "noise_ratio_entropy_scale": float,  # 0.0-0.5
    "noise_type": str,                   # "white" or "pink"
    # Optional:
    "shimmer_enable": bool,              # Enable shimmer
    "shimmer_rate_hz": float             # Shimmer rate (2.0-12.0)
}
```

---

## Particle-Based Configuration Structure

**COMPLETELY DIFFERENT** from curve-based!

### Root-Level Parameters

```python
"ARCHETYPE_NAME": {
    "system": "particle",  # Always "particle"
    "duration_base": float,
    "duration_tension_scale": float,
    "duration_entropy_scale": float,
    "particle": { ... },  # Particle config (NOT phases/pitch/harmony/filter/envelope/texture!)
    "peak_limit": float,
    "rms_target": float
}
```

---

### Particle Section (Particle-Based Only)

```python
"particle": {
    "emission": {
        "type": "EMISSION_TYPE",
        # ... emission-type-specific parameters
    },
    "base_spawn_rate": float,              # Particles per sample
    "pitch_range_hz": [min_hz, max_hz],    # Frequency range
    "lifetime_range_s": [min_s, max_s],    # Particle lifetime
    "velocity_range": [min_vel, max_vel],  # Amplitude range
    "detune_range_cents": [min, max],      # Pitch detune range
    "decay_rate_range": [min, max],        # Decay coefficient range
    "waveform": str                        # "sine" or "triangle"
}
```

---

### Emission Types (Particle-Based Only)

**Each emission type has unique parameters!**

#### drift_scatter
```python
"emission": {
    "type": "drift_scatter",
    "start_density": float,
    "end_density": float,
    "drift_rate": float
}
```

#### gusts
```python
"emission": {
    "type": "gusts",
    "num_gusts": int,
    "base_density": float,
    "peak_density": float
}
```

#### rhythmic_clusters
```python
"emission": {
    "type": "rhythmic_clusters",
    "cluster_rate_hz": float,
    "cluster_width_s": float,
    "cluster_density": float
}
```

#### dissolve
```python
"emission": {
    "type": "dissolve",
    "start_density": float,
    "end_density": float,
    "dissolve_curve": str  # e.g., "exponential"
}
```

#### impact_burst
```python
"emission": {
    "type": "impact_burst",
    "burst_density": float,
    "burst_duration_s": float
}
```

**See actual archetypes for complete emission types.**

---

## Key Differences: Curve vs. Particle

| Feature | Curve-Based | Particle-Based |
|---------|-------------|----------------|
| **Phases** | ✓ Yes (5 phases) | ✗ No |
| **Pitch** | ✓ Complex curve types | ✗ Range only |
| **Harmony** | ✓ Voice control | ✗ No |
| **Filter** | ✓ Time-varying | ✗ No |
| **Envelope** | ✓ 3 types | ✗ No |
| **Texture** | ✓ (no waveform) | ✗ No |
| **Particle** | ✗ No | ✓ Emission control |
| **Waveform** | ✗ Not in texture | ✓ In particle section |

---

## How to Explore Archetypes

### List All Archetypes

```python
from layer3b.archetype_configs import ARCHETYPES

# See all names
print(list(ARCHETYPES.keys()))

# Check if curve or particle
for name, config in ARCHETYPES.items():
    print(f"{name}: {config['system']}")
```

### Inspect Specific Archetype

```python
# See full configuration
import pprint
pprint.pprint(ARCHETYPES['BRILLIANT'])
```

### Test Archetypes

```bash
# Curve-based
python3 tools/gesture_test.py BRILLIANT --tension 0.8

# Particle-based
python3 tools/particle_test.py INACCURACY --audio
```

---

## Adding New Archetypes

1. **Choose system**: curve or particle
2. **Copy similar archetype** as template
3. **Modify parameters** to match desired sound
4. **Test with tools** until it sounds right
5. **No code changes needed** - just configuration!

---

## Documentation Philosophy

**Why this doc doesn't list everything:**

- Layer 3b has **dozens of archetypes**
- Each has **unique parameter combinations**
- **76+ distinct types** across pitch/harmony/filter
- Listing everything would create **1000+ line document**
- **Would be outdated instantly** when archetypes change

**Instead, this doc shows:**
- ✓ Overall structure (curve vs particle)
- ✓ Required vs optional fields
- ✓ Parameter types and purposes
- ✓ How to explore the source file yourself

**Source of truth**: `layer3b/archetype_configs.py` - read it directly!

---

## Further Reading

- **layer3b/archetype_configs.py**: Complete source of truth
- **ENTROPY_INTEGRATION.md**: Why entropy matters
- **composer_architecture.md**: System architecture
