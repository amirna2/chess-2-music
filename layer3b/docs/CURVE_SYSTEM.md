# Curve-Based Gesture System

## Overview

Curve-based gestures use time-varying parameter curves to create expressive sound shapes. This is the traditional sound design approach adapted for chess moment gestures.

**25 archetypes** use this system: MOVE, BRILLIANT, BLUNDER, CHECKMATE, CASTLING, etc.

## Architecture

```
Archetype Config
       ↓
CurveGenerators
  ├─ generate_pitch_curve()
  ├─ generate_harmony_curve()
  ├─ generate_filter_curve()
  └─ generate_envelope_curve()
       ↓
Time-Varying Curves
  ├─ Pitch voices: List[np.ndarray]
  ├─ Filter: {cutoff, resonance}
  ├─ Envelope: np.ndarray
  └─ Texture: {noise_ratio, noise_type, shimmer}
       ↓
GestureSynthesizer
       ↓
Audio
```

## Phase System

All curve-based gestures divide time into 5 phases:

| Phase | Typical % | Purpose |
|-------|-----------|---------|
| **pre_shadow** | 10-25% | Anticipatory buildup before main event |
| **impact** | 5-20% | Initial attack/onset moment |
| **bloom** | 20-50% | Peak activity/expansion phase |
| **decay** | 15-40% | Energy dissipation |
| **residue** | 10-20% | Trailing sustain/echo |

Configured per archetype:
```python
"phases": {
    "pre_shadow": 0.16,
    "impact": 0.10,
    "bloom": 0.48,
    "decay": 0.16,
    "residue": 0.10
}
```

Curve generators use phase boundaries to shape their trajectories. For example, a glissando might occur only during the bloom phase, or a filter might sweep from impact through decay.

## Curve Generators

### Pitch Curves

**Purpose:** Define frequency trajectory over time

**Output:** `List[np.ndarray]` - List of pitch curves in Hz (one per voice)

**Types (23 total, 9 implemented):**

#### ✅ Implemented

1. **exponential_gliss** (2 archetypes: BLUNDER, MISTAKE)
   - Exponential pitch drop (falling glissando)
   - Configurable octave drop, start frequency
   - Can scale with tension/entropy

2. **ascending_spread** (2 archetypes: GAME_CHANGING, BRILLIANT)
   - Rising pitch trajectory with voice spreading
   - Octave rise configurable
   - Voices diverge during bloom phase

3. **oscillating_tremor** (1 archetype: TIME_PRESSURE)
   - Rapid oscillation around center frequency
   - Tremor rate and depth configurable
   - Accelerates during specified phase

4. **cellular_sequence** (status unclear)
   - Discrete pitch steps in pattern
   - Used for iterative/sequential gestures

5. **weak_parabolic** (status unclear)
   - Gentle parabolic arc (rise → fall)
   - Subtle pitch motion

6. **slow_drift** (status unclear)
   - Gradual pitch drift (ascending/descending)
   - Configurable semitone range

7. **impact_transient** (status unclear)
   - Sharp attack transient
   - Strike-like pitch behavior

8. **final_descent** (status unclear)
   - Terminal descending gesture
   - Logarithmic/exponential curve

9. **discrete_chimes** (status unclear)
   - Discrete pitch events
   - Wind-chime-like behavior

#### ❌ Missing Implementations

10. **stable** (1 archetype: MOVE)
    - Stable center frequency with minimal drift

11. **sustained_drone** (1 archetype: DEEP_THINK)
    - Static drone with micro-drift

12. **parabolic** (1 archetype: CRITICAL_SWING)
    - Parabolic arc trajectory

13. **converging_steps** (1 archetype: MATE_SEQUENCE)
    - Discrete steps converging to target

14. **reciprocal_pair** (1 archetype: CASTLING)
    - Two frequencies crossing/exchanging

15. **dual_iteration** (1 archetype: ROOKS_DOUBLED)
    - Dual iterative pattern

16. **focused_center** (1 archetype: QUEEN_CENTRALIZED)
    - Focused centric fixation

17. **divergent_split** (1 archetype: ASYMMETRY)
    - Voices diverging from center

18. **gentle_descent** (1 archetype: TIME_MILESTONE)
    - Smooth descending trajectory

19. **chaotic_iteration** (1 archetype: TIME_SCRAMBLE)
    - Chaotic rapid iteration

20. **dual_descent** (1 archetype: QUEENS_TRADED)
    - Two voices descending together

21. **stable_rise** (1 archetype: STRONG)
    - Gradual stable ascent

22. **aggressive_rise** (1 archetype: KING_ATTACK)
    - Forceful ascending gesture

23. **ascending_burst** (1 archetype: PROMOTION)
    - Burst-like ascent

Plus several more single-use types...

---

### Harmony Curves

**Purpose:** Define multi-voice pitch relationships

**Output:** `List[np.ndarray]` - Pitch curves for all voices combined

**Types (21 total, 8 implemented):**

#### ✅ Implemented

1. **cluster_to_interval** (3 archetypes: CRITICAL_SWING, BLUNDER, MISTAKE)
   - Dense cluster → simple interval
   - Microtonal spacing at start
   - Resolves during decay phase

2. **unison_to_chord** (2 archetypes: GAME_CHANGING, BRILLIANT)
   - Single pitch → chord formation
   - Chord type configurable (major_seventh, etc.)
   - Spreads during bloom

3. **dense_cluster** (2 archetypes: DEEP_THINK, TIME_PRESSURE)
   - Stable dense microtonal cluster
   - 5-8 voices
   - Semitone spacing configurable

4. **harmonic_stack** (status unclear)
   - Harmonic series stack
   - Integer frequency ratios

5. **minimal_dyad** (status unclear)
   - Two-voice interval
   - Simple harmonic relationship

6. **collision_cluster** (status unclear)
   - Impact-like dense cluster
   - High density, short duration

7. **shifting_voices** (status unclear)
   - Voices shifting in pitch space
   - Independent motion

8. **resolving_to_root** (status unclear)
   - Complex → simple resolution
   - Tonal center convergence

#### ❌ Missing Implementations

9. **simple_unison** (1 archetype: MOVE)
10. **converging_cluster** (1 archetype: MATE_SEQUENCE)
11. **controlled_expansion** (1 archetype: STRONG)
12. **dense_agglomeration** (1 archetype: KING_ATTACK)
13. **dual_voices** (1 archetype: CASTLING)
14. **balanced_spectrum** (1 archetype: CENTER_CONTROL)
15. **flowing_voices** (1 archetype: PIECE_MANEUVER)
16. **expanding_formation** (1 archetype: DEVELOPMENT)
17. **forceful_stack** (1 archetype: ROOK_ACTIVATION)
18. **reinforced_pair** (1 archetype: ROOKS_DOUBLED)
19. **centered_cluster** (1 archetype: QUEEN_CENTRALIZED)
20. **fragmented_voices** (1 archetype: ASYMMETRY)
21. **resolving_dyad** (1 archetype: TIME_MILESTONE)

Plus more...

---

### Filter Curves

**Purpose:** Define spectral shaping over time

**Output:** `Dict` with `'cutoff'` and `'resonance'` curves (both np.ndarray)

**Types (22 total, 8 implemented):**

#### ✅ Implemented

1. **bandpass_to_lowpass_choke** (2 archetypes: BLUNDER, MISTAKE)
   - Bandpass → narrowing lowpass
   - "Choking" spectral closure

2. **lowpass_to_highpass_open** (2 archetypes: GAME_CHANGING, BRILLIANT)
   - Lowpass → highpass transition
   - Spectral "opening"

3. **bandpass_sweep** (2 archetypes: CRITICAL_SWING, TIME_PRESSURE)
   - Bandpass center sweeps across spectrum
   - Configurable sweep curve

4. **rhythmic_gate** (status unclear)
   - Rhythmic gating pattern
   - Pulse-like filtering

5. **gentle_bandpass** (status unclear)
   - Subtle bandpass filtering
   - Low resonance

6. **gradual_sweep** (status unclear)
   - Slow filter sweep
   - Linear/sigmoid curve

7. **impact_spike** (status unclear)
   - Sharp resonance spike
   - Attack emphasis

8. **closing_focus** (status unclear)
   - Narrowing spectral focus
   - Convergent filtering

#### ❌ Missing Implementations

9. **simple_lowpass** (1 archetype: MOVE)
10. **focusing_narrowband** (1 archetype: MATE_SEQUENCE)
11. **static_bandpass** (1 archetype: DEEP_THINK)
12. **controlled_opening** (1 archetype: STRONG)
13. **aggressive_open** (1 archetype: KING_ATTACK)
14. **symmetric_sweep** (1 archetype: CASTLING)
15. **bass_emphasis** (1 archetype: PAWN_ADVANCE)
16. **broadband_stable** (1 archetype: CENTER_CONTROL)
17. **curved_sweep** (1 archetype: PIECE_MANEUVER)
18. **gradual_opening** (1 archetype: DEVELOPMENT)
19. **forceful_open** (1 archetype: ROOK_ACTIVATION)
20. **dual_resonance** (1 archetype: ROOKS_DOUBLED)
21. **focused_bandpass** (1 archetype: QUEEN_CENTRALIZED)
22. **split_trajectories** (1 archetype: ASYMMETRY)

Plus more...

---

### Envelope Curves

**Purpose:** Define amplitude shaping over time

**Output:** `np.ndarray` - Amplitude envelope (0-1)

**Types (7 total, 7 implemented):**

#### ✅ All Implemented

1. **sudden_short_tail** (8 archetypes)
   - Fast attack (1-20ms)
   - Short sustain
   - Exponential decay
   - Used for: MOVE, CRITICAL_SWING, BLUNDER, MISTAKE, etc.

2. **gradual_sustained** (11 archetypes)
   - Gradual attack (40-85ms)
   - Long sustain (40-56%)
   - Linear/sigmoid decay
   - Used for: GAME_CHANGING, BRILLIANT, STRONG, etc.

3. **plateau_sustained** (1 archetype: DEEP_THINK)
   - Very gradual attack (250ms)
   - Long plateau sustain (68%)

4. **stepped_convergence** (1 archetype: MATE_SEQUENCE)
   - Discrete amplitude steps
   - Final accent multiplier
   - Converging pattern

5. **symmetric_dual** (1 archetype: CASTLING)
   - Balanced dual-peak envelope
   - Mirror symmetry

6. **terminal_strike** (1 archetype: CHECKMATE)
   - Sharp attack
   - Final accent emphasis

7. **gated_pulse** (2 archetypes: TIME_SCRAMBLE, ROOKS_DOUBLED)
   - Rhythmic gating pattern
   - Pulse rate configurable

---

## Context Responsiveness

Curves adapt to chess game context through scaling parameters:

### Tension Scaling (0-1)
Reflects section-level dramatic intensity.

```python
"duration_tension_scale": 0.8  # Duration increases with tension
"octave_drop_tension_scale": 1  # Deeper drop when tense
```

### Entropy Scaling (0-1)
Reflects positional complexity/chaos.

```python
"duration_entropy_scale": 0.3  # Duration increases with entropy
"start_freq_entropy_scale": 110  # Pitch varies with entropy
```

### Example Calculation
```python
duration = duration_base + (tension_scale × tension) + (entropy_scale × entropy)
         = 3.3 + (0.8 × 0.7) + (-0.5 × 0.5)
         = 3.3 + 0.56 - 0.25
         = 3.61 seconds
```

---

## Creating New Curve Types

### 1. Add Generator Function

File: `curve_generators.py`

```python
def _pitch_my_new_type(
    config: Dict[str, Any],
    total_samples: int,
    phases: Dict[str, Any],
    section_context: Dict[str, Any]
) -> List[np.ndarray]:
    """
    My new pitch trajectory type.

    Args:
        config: Pitch config from archetype
        total_samples: Total gesture length in samples
        phases: Phase boundaries dict
        section_context: Tension/entropy values

    Returns:
        List of pitch curves (one per voice)
    """
    # Extract phase boundaries
    impact_start = phases['impact']['start']
    bloom_end = phases['bloom']['end']

    # Extract config parameters
    start_freq = config.get('start_freq_base', 440.0)
    end_freq = config.get('end_freq_base', 880.0)

    # Apply context scaling
    tension = section_context.get('tension', 0.5)
    start_freq += config.get('start_freq_tension_scale', 0.0) * tension

    # Generate curve
    t = np.linspace(0, 1, total_samples)
    pitch_curve = start_freq + (end_freq - start_freq) * t

    return [pitch_curve]  # Single voice
```

### 2. Register in Dispatcher

```python
def generate_pitch_curve(...):
    trajectory_type = config['type']

    if trajectory_type == "my_new_type":
        return _pitch_my_new_type(config, total_samples, phases, section_context)
    # ... other types
```

### 3. Use in Archetype

```python
"MY_ARCHETYPE": {
    "pitch": {
        "type": "my_new_type",
        "start_freq_base": 440,
        "end_freq_base": 880,
        "start_freq_tension_scale": 110
    },
    # ... rest of config
}
```

---

## Implementation Status Summary

| Component | Total Types | Implemented | Missing | % Complete |
|-----------|-------------|-------------|---------|------------|
| **Pitch** | 23+ | 9 | 14+ | 39% |
| **Harmony** | 21+ | 8 | 13+ | 38% |
| **Filter** | 22+ | 8 | 14+ | 36% |
| **Envelope** | 7 | 7 | 0 | 100% |

**Critical:** ~80% of curve-based archetypes cannot be fully synthesized due to missing implementations.

### Priority Missing Implementations

High-priority based on usage:
1. **simple_unison** (harmony) - used by MOVE
2. **stable** (pitch) - used by MOVE
3. **simple_lowpass** (filter) - used by MOVE
4. **sustained_drone** (pitch) - used by DEEP_THINK
5. **parabolic** (pitch) - used by CRITICAL_SWING

Most other types are single-use and can be implemented as needed.

---

## Design Patterns

### 1. Phase-Aware Trajectories
Curves should respect phase boundaries:
```python
bloom_start = phases['bloom']['start']
bloom_end = phases['bloom']['end']

# Apply effect only during bloom
curve[bloom_start:bloom_end] = modified_values
```

### 2. Context Scaling
Always support tension/entropy scaling:
```python
param = base_value + (tension_scale × tension) + (entropy_scale × entropy)
```

### 3. Multi-Voice Coordination
Return list of curves for polyphonic gestures:
```python
return [voice1_curve, voice2_curve, voice3_curve]
```

### 4. Smooth Transitions
Avoid discontinuities with interpolation:
```python
curve = np.interp(t, [0, 0.5, 1], [start, peak, end])
```

### 5. Natural Variation
Add subtle randomness for organic feel:
```python
noise = rng.normal(0, 0.02, len(curve))
curve += noise
```
