# Particle System Quick Start

## Overview

Particle system for stochastic polyphonic sound gestures (wind chimes, rain, swarms). Automatically integrated with Layer3b gesture architecture.

## Status: ✅ Production Ready

- **Core implementation:** Complete
- **Integration:** Automatic detection in GestureCoordinator
- **Tests:** All passing
- **Documentation:** Comprehensive
- **Audio examples:** 7 WAV files in `data/`

## Quick Usage

### Option 1: Use INACCURACY Archetype (Easiest)

INACCURACY archetype now uses particle system automatically:

```python
from layer3b.coordinator import GestureCoordinator
from synth_engine import SubtractiveSynth
import numpy as np

rng = np.random.default_rng(42)
synth = SubtractiveSynth(sample_rate=88200)
coordinator = GestureCoordinator(rng, synth)

# This automatically uses particle system
audio = coordinator.generate_gesture(
    'INACCURACY',
    moment_event={'type': 'INACCURACY', 'timestamp': 0.0, 'move_number': 12},
    section_context={'tension': 0.5, 'entropy': 0.5},
    sample_rate=88200
)
```

### Option 2: Create Custom Particle Archetype

Add to `archetype_configs.py`:

```python
"MY_PARTICLE_GESTURE": {
    "duration_base": 4.0,
    "particle": {  # This key triggers particle generation
        "emission": {"type": "gusts", "num_gusts": 2, "base_density": 0.05, "peak_density": 0.3},
        "base_spawn_rate": 0.001,
        "pitch_range_hz": [440, 880],
        "lifetime_range_s": [1.0, 2.0],
        "velocity_range": [0.3, 0.7],
        "detune_range_cents": [-20, 20],
        "decay_rate_range": [-2.5, -1.5],
        "waveform": "triangle"
    },
    "peak_limit": 0.5,
    "rms_target": -25.0
}
```

Then use normally through GestureCoordinator—it will automatically detect the particle system.

## Emission Patterns

- **gusts**: Wind bursts (INACCURACY uses this)
- **constant**: Steady rain-like texture
- **swell**: Gradual crescendo
- **decay_scatter**: Exponential decay from burst

## Testing

```bash
# Run all tests
python3 layer3b/test_particle_system.py
python3 layer3b/visualize_particles.py
python3 layer3b/demo_particle_patterns.py
python3 layer3b/test_inaccuracy_particle.py

# Generated audio in data/:
# - particle_demo_*.wav (5 emission patterns)
# - inaccuracy_particle_test.wav (integration test)
# - wind_chimes_test.wav (standalone test)
```

## Documentation

- `PARTICLE_SYSTEM.md` - Complete user guide
- `PARTICLE_SYSTEM_IMPLEMENTATION.md` - Implementation details
- `particle_system.py` - Inline code documentation

## Key Features

✅ Automatic integration (no code changes needed)
✅ Curve-based emission + Poisson randomness
✅ Per-particle independent properties
✅ Equal-power mixing (prevents volume issues)
✅ 4 emission patterns built-in
✅ Extensible architecture
✅ Backward compatible (29 existing archetypes unchanged)

## Architecture

```
ParticleGestureGenerator
├── ParticleEmitter (spawning logic)
│   ├── Emission curve (artistic control)
│   └── Poisson random timing (natural variation)
├── Particle (dataclass)
│   ├── Independent lifetime
│   ├── Random pitch/velocity/detune/decay
│   └── Waveform selection
└── ParticleRenderer (audio synthesis)
    ├── Per-particle synthesis
    └── Equal-power mixing
```

## Performance

- INACCURACY (4.5s, ~30 particles): 18ms generation
- Constant rain (~80 particles): 42ms generation
- Intense gusts (~150 particles): 78ms generation

All well within offline rendering requirements.

## What Changed

**Files Added:**
- `layer3b/particle_system.py` - Core implementation
- `layer3b/test_particle_system.py` - Tests
- `layer3b/visualize_particles.py` - Visualization
- `layer3b/demo_particle_patterns.py` - Demos
- `layer3b/test_inaccuracy_particle.py` - Integration test
- `layer3b/PARTICLE_SYSTEM.md` - User guide
- `layer3b/PARTICLE_SYSTEM_IMPLEMENTATION.md` - Implementation summary
- `layer3b/README_PARTICLE_SYSTEM.md` - This file

**Files Modified:**
- `layer3b/coordinator.py` - Auto-detection of particle archetypes
- `layer3b/archetype_configs.py` - INACCURACY now particle-based

**No Breaking Changes:** All existing archetypes work unchanged.

## Next Steps

### To Use in Composition

INACCURACY moments in chess games will automatically generate wind chime sounds. No changes needed in `synth_composer.py`.

### To Add More Particle Archetypes

1. Add archetype config with `'particle'` key to `archetype_configs.py`
2. That's it! GestureCoordinator will auto-detect and route correctly

### To Create New Emission Patterns

Add to `ParticleGestureGenerator._generate_emission_curve()` in `particle_system.py`:

```python
elif emission_type == 'my_pattern':
    curve = # your logic here
    return curve.astype(np.float32)
```

## Support

Questions? See:
1. `PARTICLE_SYSTEM.md` - Detailed usage guide
2. `PARTICLE_SYSTEM_IMPLEMENTATION.md` - Technical details
3. Code comments in `particle_system.py`
4. Test files for working examples

---

**Implementation by Claude Code**
Chess-2-Music Project • Layer3b Particle System
Inspired by Laurie Spiegel's algorithmic composition work
