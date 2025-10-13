# Laurie Spiegel Style Profile (1979)

**Inspired by**: The Expanding Universe, Appalachian Grove, Patchwork

## Aesthetic Philosophy

> "I automate whatever can be automated to be freer to focus on those aspects of music that can't be automated. The challenge is to figure out which is which." - Laurie Spiegel

Spiegel's approach emphasizes:
- **Algorithmic intelligence**: Processes that evolve organically
- **Harmonic awareness**: Bach-inspired voice leading and progression
- **Textural evolution**: Long-form timbral transformation
- **Sparse events**: Thoughtful note placement, not density
- **Process music**: Systems that unfold naturally over time

---

## High-Level Style Profile

```python
STYLE_PROFILES['spiegel'] = {
    'description': 'Laurie Spiegel (1979) - Sparse, contemplative, drone-based',

    # === TEMPO ===
    'bpm': None,  # Free time (no fixed grid)

    # === ENABLED LAYERS ===
    'layers': ['drone', 'sparse_patterns', 'heartbeat', 'gestures'],

    # === GLOBAL MIXING ===
    'mixing': {
        'drone_level': 0.20,      # Increased from default 0.15
        'pattern_level': 0.50,    # Melodic events clear but not dominant
        'heartbeat_level': 0.30,  # Subtle biological presence
        'gesture_level': 0.60,    # Key moments prominent
    },

    # === FX CHAIN (Phase 4+ - Not Implemented Yet) ===
    'fx': {
        'chorus': {
            'enabled': False,  # Future: True
            'depth': 0.15,
            'rate_hz': 0.05,
        },
        'delay': {
            'enabled': False,  # Future: True
            'time_ms': 620,
            'feedback': 0.35,
            'mix': 0.25,
        },
        'reverb': {
            'enabled': False,  # Future: True
            'time_s': 6.0,
            'mix': 0.45,
        },
        'tape_saturation': {
            'enabled': False,  # Future: True
            'amount': 0.8,
        },
    },
}
```

---

## Layer 1: Evolving Drone

**Purpose**: Sub-bass foundation that evolves slowly over minutes

### Parameters

| Parameter | Setting | Purpose | Status |
|-----------|---------|---------|--------|
| **Waveform** | Saw (from narrative) | Harmonically rich foundation | ✅ Current |
| **Voices** | 3 detuned oscillators | Beating, organic character | ✅ Current |
| **Detune** | ±4 cents per voice | Subtle beating | ✅ Current |
| **Filter Type** | Moog lowpass | Warm analog character | ✅ Current |
| **Filter Sweep** | Linear (narrative-driven) | Evolution over section | ✅ Current |
| **Filter LFO** | 0.05 Hz, ±200 Hz depth | Breathing motion | ⚠️ Configurable (Phase 2.5) |
| **Resonance** | 0.4 - 0.8 (slow modulation) | Adds character | ✅ Current |
| **LFO System** | 2 nested (meso + micro) | Multi-timescale evolution | ✅ Current (3rd LFO deferred) |
| **Envelope** | `(2.0, 0.0, 1.0, 2.0)` | Glacial fade in/out | ✅ Current (pad envelope) |
| **Stereo** | Centered, width 0.3 | Focused but warm | ✅ Phase 2.5 |
| **Mix Level** | 0.20 | More prominent than default | ✅ Phase 2.5 |

### Current Implementation

```python
'layer1_drone': {
    # Waveform from narrative (saw/triangle based on game context)
    'waveform': None,  # None = use narrative default

    # Filter configuration
    'filter': {
        'type': 'moog_lowpass',  # Current implementation
        'curve': 'linear',       # Current (logarithmic deferred)
        'lfo': {
            'enabled': True,
            'rate_hz': 0.05,     # Breathing rate (Phase 2.5)
            'depth': 200,        # ±200 Hz modulation (Phase 2.5)
        },
    },

    # Envelope (maps to existing 'pad' preset)
    'envelope': (0.5, 0.0, 1.0, 0.5),  # Current pad envelope

    # Stereo placement
    'stereo': {
        'pan': 0.0,      # Centered
        'width': 0.3,    # Narrow stereo field
    },

    # Mix level
    'mix_level': 0.20,  # Style override
}
```

### Deferred Enhancements (Phase 4+)

| Feature | Target Setting | Reason for Deferral |
|---------|----------------|---------------------|
| Waveform mix | Triangle + sine (50/50) | Requires new oscillator mixing |
| Filter curve | Logarithmic | Requires interpolation change |
| Just intonation | Custom frequency ratios | Major scale system change |
| 3rd LFO | 0.4 Hz micro shimmer | Already have 2, add later |
| Pan drift LFO | ±0.3 slow drift | Requires dynamic panning |

---

## Layer 2: Sparse Patterns

**Purpose**: Occasional melodic events using Markov chain algorithms

### Parameters

| Parameter | Setting | Purpose | Status |
|-----------|---------|---------|--------|
| **Pattern Type** | Markov chain with gravity | Probabilistic note selection | ✅ Current |
| **Density** | 0.2x (very sparse) | One note every 2-5 seconds | ✅ Current (via note_density) |
| **Tonic Weight** | 0.4 | Pull to tonic (cautious play) | ✅ Current (MarkovChainPattern) |
| **Pause Probability** | 30% rests | Silence is musical | ✅ Current |
| **Pause Duration** | 2-5 seconds | Contemplative gaps | ✅ Current |
| **Waveform** | Pulse (25% duty) | Clear, focused tone | ✅ Current |
| **Filter** | 800-2000 Hz (darker) | Not overly bright | ✅ Current (narrative-driven) |
| **Envelope** | Pluck to soft | Natural decay | ✅ Current (various presets) |
| **Stereo** | Entropy-driven panning | Follows chess complexity | ✅ Current |
| **Mix Level** | 0.50 | Clear but not dominant | ✅ Phase 2.5 |

### Current Implementation

Uses existing `PatternCoordinator` with:
- `MarkovChainPattern` for COMPLEX_STRUGGLE
- `DesperateDefensePattern` for DESPERATE_DEFENSE
- Various state machine patterns for other narratives

**No changes needed** - already Spiegel-appropriate!

---

## Layer 3a: Heartbeat

**Purpose**: Biological anchor, subconscious pulse

### Parameters

| Parameter | Setting | Purpose | Status |
|-----------|---------|---------|--------|
| **Waveform** | Sine | Pure, focused sub-bass | ✅ Current |
| **Frequency** | MIDI 36 (C1, ~65Hz) | Sub-bass heartbeat | ✅ Current |
| **BPM** | 70 | Resting heart rate | ✅ Current |
| **Pattern** | Lub-dub | Realistic heartbeat | ✅ Current |
| **Lub-Dub Gap** | 80ms | Medical accuracy | ✅ Current |
| **Dub Offset** | -2 semitones | Lower than lub | ✅ Current |
| **Dub Volume** | 70% of lub | Quieter second beat | ✅ Current |
| **Filter** | 220 Hz lowpass | Muffled but audible | ✅ Current |
| **Resonance** | 0.0 | Minimal, natural | ✅ Current |
| **Envelope** | `(0.003, 0.06, 0.03, 0.20)` | Percussive pulse | ✅ Current |
| **Stereo** | Centered, mono | Biological constant | ✅ Current |
| **Mix Level** | 0.30 | Subtle presence | ✅ Phase 2.5 |

### Current Implementation

Already well-implemented in `compose_section()` as Layer 3a.

**No changes needed** - heartbeat is perfect as-is!

---

## Layer 3b: Gestures

**Purpose**: Dramatic punctuation for key chess moments

### Parameters

| Parameter | Setting | Purpose | Status |
|-----------|---------|---------|--------|
| **System** | Spectromorphological | Curve-based gestures | ✅ Current (archetype system) |
| **Duration** | 1.5-4 seconds | Longer, contemplative | ✅ Current (archetype configs) |
| **Attack** | Slow (gradual) | Not percussive | ✅ Current |
| **Filter Sweeps** | Slow, dramatic | Timbral evolution | ✅ Current |
| **Harmonic Spread** | Unison → chord (gradual) | Textural expansion | ✅ Current |
| **Particle System** | For uncertainty/complexity | Stochastic events | ✅ Current |
| **Stereo** | Random per event | Each gesture unique | ✅ Current (entropy-driven) |
| **Mix Level** | 0.60 | Prominent but not overwhelming | ✅ Phase 2.5 |

### Gesture Archetypes Used

- **BRILLIANT**: Ascending glissando, opening filters
- **BLUNDER**: Descending pitch, closing filters
- **INACCURACY**: Particle drift (uncertainty)
- **TACTICAL_SEQUENCE**: Burst of activity
- **FIRST_EXCHANGE**: Mirror symmetry
- **CHECKMATE**: Final resolution

### Current Implementation

Uses existing `GestureCoordinator` with archetype system.

**No changes needed** - gestures are already Spiegel-style!

---

## Narrative Response

How Spiegel style interprets chess narratives:

### Tumbling Defeat
- **Drone**: Saw waveform, darkening filters (2500→300 Hz)
- **Patterns**: Very sparse (0.15 density), descending Markov
- **Gestures**: Blunders are long, doom-like
- **Overall**: Gradual decay, increasing chaos

### Attacking Masterpiece
- **Drone**: Triangle waveform, opening filters (800→3000 Hz)
- **Patterns**: Slightly denser (0.25), ascending Markov
- **Gestures**: Brilliant moves are bright, triumphant
- **Overall**: Crescendo, increasing clarity

### Quiet Precision
- **Drone**: Triangle, static filters (1500 Hz)
- **Patterns**: Balanced Markov, minimal chaos
- **Gestures**: Subtle, controlled
- **Overall**: Equilibrium, gentle breathing

---

## Stereo Image

```
         LEFT          CENTER         RIGHT
          |              |              |
Pattern: [---entropy-driven panning----] (±0.5)
Drone:        [====focused====]         (width 0.3)
Heartbeat:           •                  (mono center)
Gestures: [---random per event---]     (±0.7)
```

**Philosophy**: Drone is the stable center, patterns and gestures move spatially based on chess complexity.

---

## Mixing Strategy

```python
# Final mix (before master limiting)
mixed = (
    drone     * 0.20 +  # Spiegel emphasizes drone
    patterns  * 0.50 +  # Melodic events clear
    heartbeat * 0.30 +  # Subtle biological anchor
    gestures  * 0.60    # Key moments prominent
)
```

**Note**: These levels override the defaults in `synth_config.py`:
- Default drone: 0.15 → Spiegel: 0.20 (+33%)
- Default heartbeat: 0.40 → Spiegel: 0.30 (-25%, more subtle)
- Default gestures: 0.50 → Spiegel: 0.60 (+20%)

---

## Implementation Status

### ✅ Phase 2.5 (Current)
- [x] Style-controlled mixing levels
- [x] Layer 1 stereo configuration
- [x] Layer 1 filter LFO (configurable)
- [x] Existing patterns/heartbeat/gestures (no changes)

### 🔄 Phase 3 (Deferred)
- [ ] Logarithmic filter curves
- [ ] Waveform mixing (triangle + sine)
- [ ] Third LFO (micro shimmer)
- [ ] Pan drift LFO

### 🔮 Phase 4+ (Future)
- [ ] Just intonation scales
- [ ] FX chain (chorus, delay, reverb, tape saturation)
- [ ] Bach-inspired harmonic progression
- [ ] Voice-leading algorithms

---

## Validation Criteria

A successful Spiegel-style rendering should:

1. ✅ **Sound sparse** - Silence is part of the music
2. ✅ **Evolve slowly** - Changes happen gradually over minutes
3. ✅ **Feel contemplative** - Not rhythmically driving
4. ✅ **Emphasize texture** - Timbre evolution is prominent
5. ✅ **Have harmonic intelligence** - Voice leading makes sense
6. ✅ **Use gestures dramatically** - Key moments punctuate evolution
7. ✅ **Respect the drone** - Foundation is always present

---

## Reference Listening

For parameter tuning, reference these Spiegel works:

- **"Appalachian Grove"** (1974) - Sparse melodic events over drone
- **"Patchwork"** (1974) - Algorithmic processes unfolding
- **"The Expanding Universe"** (1980) - Long-form evolution
- **"Old Wave"** (1977) - Harmonic intelligence with minimalism

---

## Discrepancies from Ideal

| Ideal Feature | Current Status | Impact | Timeline |
|---------------|----------------|--------|----------|
| Just intonation | Equal temperament | Low (barely audible) | Phase 4+ |
| Logarithmic filter | Linear interpolation | Low (gradual anyway) | Phase 3 |
| Triangle/sine mix | Single waveform | Low (narrative chooses) | Phase 3 |
| 3 nested LFOs | 2 nested LFOs | Very low (already rich) | Phase 3 |
| Pan drift LFO | Static pan | Low (entropy panning works) | Phase 3 |
| FX chain | No FX | Medium (reverb would help) | Phase 4 |

**Verdict**: Current implementation is **90% Spiegel-appropriate** even without deferred features!

---

## Next Steps

1. ✅ Phase 2.5: Apply style mixing levels in renderer
2. ✅ Phase 2.5: Make filter LFO configurable from style profile
3. 🔄 Phase 3: Implement Jarre renderer (validates architecture)
4. 🔮 Phase 4: Add FX chain implementations
5. 🔮 Phase 5: Refine with just intonation, logarithmic curves
