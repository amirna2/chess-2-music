# Entropy Integration for Layer 3
## Laurie Spiegel-Inspired Continuous Musical Evolution

### Core Concept

**Current Problem**: Layer 3 is **event-driven** - it switches patterns only when key moments occur, creating static repetition between events.

**Spiegel's Solution**: Use **informational entropy** to create continuous evolution based on position complexity. The entropy curve itself becomes a compositional parameter that controls moment-to-moment predictability.

> "The moment to moment variation of level of predictability that is embodied in an entropy curve arouses in the listener feelings of expectation, anticipation, satisfaction, disappointment, surprise, tension, frustration and other emotions."
> — Laurie Spiegel

---

## What Changes: Before vs. After

### CURRENT Layer 3 (Event-Driven)

```python
# Pattern switches ONLY at key moments
for i in range(total_steps):
    current_pattern = SEQUENCER_PATTERNS['PULSE']  # Static!

    # Only changes when hitting a discrete event
    if key_moment_type == 'BLUNDER':
        current_pattern = SEQUENCER_PATTERNS['BLUNDER']  # Sudden switch!
```

**Sound**: Binary - either "calm pattern" OR "crisis pattern"
- Sounds like: Video game with discrete states
- Misses: Gradual build-up before critical moments
- Result: `beep-beep-beep-beep [BLUNDER!] CRASH beep-beep-beep-beep`

---

### NEW Layer 3 (Entropy-Driven)

```python
# Pattern evolves CONTINUOUSLY based on position complexity
for i in range(total_steps):
    current_ply = get_ply_from_step(i)
    entropy = entropy_curve[current_ply - start_ply]  # 0.0 to 1.0

    # CONTINUOUS evolution based on entropy
    note_pool = get_notes_for_entropy(entropy)      # Wider range = more options
    rhythm_var = entropy * 0.5                       # More variation = less predictable
    filter_speed = 0.02 + entropy * 0.10            # Faster = more active
    harmonic_density = entropy                       # More voices = richer
```

**Sound**: Analog curve - continuously evolving complexity
- Sounds like: Organic, breathing music
- Captures: Gradual intensification as eval volatility increases
- Result: `simple → building → peak tension → resolving → simple`

---

## Data Sources for Entropy

### 1. Evaluation Volatility (Primary)
**From**: `feat-r4.json` → `eval_cp` field
**Calculation**: Rolling standard deviation of eval in centipawns
```python
# High volatility = high entropy (position unclear)
# Low volatility = low entropy (position clear)
volatility = np.std(evals_window)
entropy = min(1.0, volatility / 200.0)  # Normalize 0-200cp → 0-1
```

**Why it works**: Eval swings directly measure "how hard is this to evaluate" = computational entropy

### 2. Tactical Density (Secondary)
**From**: `is_capture`, `is_check`, `is_promotion` fields
**Calculation**: Frequency of tactical moves in local window
```python
tactical_count = captures + checks*1.5 + promotions*2
tactical_density = tactical_count / window_size
```

**Why it works**: More tactical options = more complexity = higher entropy

### 3. Thinking Time (Tertiary, if available)
**From**: `emt_seconds` field
**Calculation**: Normalize thinking time
```python
# Quick move (< 10s) = low entropy (obvious)
# Long think (60s+) = high entropy (difficult)
if emt < 10:
    time_entropy = 0.1 + (emt/10) * 0.3
elif emt < 60:
    time_entropy = 0.4 + ((emt-10)/50) * 0.4
else:
    time_entropy = min(1.0, 0.8 + (emt-60)/300)
```

### Combined Entropy
```python
entropy = eval_entropy * 0.5 + tactical_entropy * 0.4 + time_entropy * 0.1
```

---

## Concrete Musical Mappings

### 1. Note Selection Pool

**Low Entropy (0.0 - 0.3)**: Predictable, simple
```python
if entropy < 0.3:
    note_pool = [0, 4]  # Just tonic and fifth (C, G)
```
**Sound**: Drone-like, stable, resolved

**Medium Entropy (0.3 - 0.7)**: Developing, interesting
```python
elif entropy < 0.7:
    note_pool = [0, 2, 4, 5, 7]  # Most of diatonic scale
```
**Sound**: Melodic, exploring, building

**High Entropy (0.7 - 1.0)**: Unpredictable, complex
```python
else:
    note_pool = [0, 1, 2, 3, 4, 5, 6, 7]  # Full chromatic within octave
```
**Sound**: Chromatic, tense, chaotic

---

### 2. Rhythmic Variation

**Low Entropy**: Metronomic, regular
```python
if entropy < 0.3:
    duration = base_duration  # Exactly on grid
```

**High Entropy**: Irregular, loose
```python
else:
    rhythm_var = entropy * 0.5
    duration = base_duration * (1.0 + np.random.uniform(-rhythm_var, rhythm_var))
    # ±50% variation = feels chaotic
```

---

### 3. Filter Sweep Rate

**Low Entropy**: Slow, stable (50-second cycle)
```python
filter_lfo_speed = 0.02  # Hz
```

**High Entropy**: Fast, restless (8-second cycle)
```python
filter_lfo_speed = 0.02 + entropy * 0.10  # Up to 0.12 Hz
```

---

### 4. Portamento/Glide Amount

**Low Entropy**: Long smooth glides (flowing)
```python
glide_time = sixteenth_duration * 0.3  # 30% of note duration
```

**High Entropy**: Short/no glides (jumpy, nervous)
```python
glide_time = sixteenth_duration * 0.3 * (1.0 - entropy * 0.5)
# Up to 50% reduction = shorter, jumpier
```

---

### 5. Harmonic Density

**Low Entropy**: Single note melody (sparse)
```python
if entropy < 0.3:
    # Just play the note, no harmony
```

**High Entropy**: Add harmony notes (dense, cluster-like)
```python
if entropy > 0.7 and np.random.random() < entropy:
    # Add random harmony
    chord_interval = np.random.choice([3, 4, 7])
    harmony_note = (note_interval + chord_interval) % len(scale)
```

---

## Example: Fischer-Taimanov Opening (Plies 1-20)

### Entropy Profile
```
Ply  Entropy  Meaning
1-3   0.06    Standard Sicilian theory - very predictable
4-9   0.27    First exchanges - position opening up
10-12 0.07    Normal development - back to theory
13-18 0.25    Tactical ideas emerging - more complex
20    0.05    Position clarified - simple again
```

### Musical Result (Current vs. Entropy-Driven)

**CURRENT (Event-Driven)**:
```
Plies 1-4:   PULSE pattern [0, None, 4, None] repeating
             beep _ beep _ beep _ beep _
Ply 5:       Capture detected → switch to DEVELOPMENT
Plies 5-20:  DEVELOPMENT [0, 3, 5, 7] repeating
             beep-beep-beep-beep-beep-beep
```
- Sudden jump at ply 5
- Static before and after
- Doesn't reflect the gradual complexity changes

**WITH ENTROPY**:
```
Plies 1-3:   Entropy 0.06 (theory)
             Notes: C-G-C-G (just root-fifth)
             Rhythm: Regular quarters ♩ ♩ ♩ ♩
             Filter: Stable, dark

Plies 4-9:   Entropy rising 0.19→0.27 (exchanges opening position)
             Notes: Gradually adding C-D-E-G-A
             Rhythm: Starting to vary ♩ ♪♪ ♩. ♪
             Filter: Opening up slowly (500→1000Hz)
             Portamento: Getting shorter (flowing→jumpy)

Plies 10-12: Entropy drops to 0.07 (normal development)
             Notes: Back to C-G-C-G
             Rhythm: Regular again ♩ ♩ ♩ ♩
             Filter: Settling back (800Hz stable)

Plies 13-18: Entropy rises 0.17→0.25 (tactical complexity)
             Notes: Adding chromaticism C-Db-Eb-E-G-Ab
             Rhythm: More varied, some triplets ♩ ♪♪♪ ♩. ♪
             Filter: Sweeping actively (600→1500Hz cycle)
             Harmony: Occasional added notes

Ply 20:      Entropy drops to 0.05 (clarified)
             Notes: Just C (held)
             Rhythm: Settling
             Filter: Dark again (400Hz)
```

- **Gradual build-up** matches eval volatility
- **Natural ebb and flow** of complexity
- You **hear tension building** before key moments
- **Organic evolution**, not discrete jumps

---

## Why This Creates Better Musicality

### Current Approach: "What Happened"
Layer 3 tells you **WHEN something important occurred** (after the fact)
- Key moment happens → pattern switches
- Static between events
- Reactive, not anticipatory

### Entropy Approach: "How Uncertain Is This"
Layer 3 tells you **HOW COMPLEX THE POSITION IS** (in real-time)
- Complexity building → music intensifies
- Complexity resolving → music simplifies
- Proactive, anticipatory

### Emotional Arc (Spiegel's Goal)

**Anticipation**: Rising entropy before critical moment
→ "Something's building, I can feel it"

**Surprise**: Sudden entropy spike
→ "Whoa! What just happened?"

**Satisfaction**: Entropy dropping after resolution
→ "Ahh, that makes sense now"

**Tension**: Sustained high entropy
→ "This is uncomfortable, unclear"

**Inevitability**: Steadily dropping entropy (mate sequences, conversions)
→ "This is inexorable, unstoppable"

---

## Implementation Plan

### Phase 1: Calculate Entropy Curves
```python
# In compose_section() before Layer 3 generation
from entropy_calculator import ChessEntropyCalculator

entropy_calc = ChessEntropyCalculator(self.tags['moves'])
section_entropy = entropy_calc.calculate_combined_entropy(
    section['start_ply'],
    section['end_ply']
)
```

### Phase 2: Sample Entropy at Each Step
```python
# In Layer 3 sequencer loop (line ~1350)
for i in range(total_steps):
    # Map step to ply
    current_time = i * sixteenth_duration
    current_ply = section['start_ply'] + int(current_time)

    # Get entropy for this ply
    entropy = entropy_calc.get_entropy_at_ply(
        current_ply,
        section['start_ply'],
        section_entropy
    )
```

### Phase 3: Apply Entropy to Parameters
```python
    # Note selection
    if entropy < 0.3:
        available_intervals = [0, 4]  # Simple
    elif entropy < 0.7:
        available_intervals = [0, 2, 4, 5, 7]  # Moderate
    else:
        available_intervals = list(range(8))  # Complex

    note_interval = np.random.choice(available_intervals)

    # Rhythm variation
    rhythm_var = entropy * 0.5
    actual_duration = sixteenth_duration * (1.0 + np.random.uniform(-rhythm_var, rhythm_var))

    # Filter modulation speed
    filter_lfo_freq = 0.02 + entropy * 0.10

    # Portamento amount
    glide_factor = 1.0 - entropy * 0.5
    glide_time = sixteenth_duration * 0.3 * glide_factor

    # Harmonic additions
    if entropy > 0.7 and np.random.random() < entropy:
        # Add harmony note
        harmony_interval = np.random.choice([3, 4, 7])
```

### Phase 4: Smooth Transitions
```python
# Apply smoothing to avoid sudden jumps in entropy curve
from scipy.ndimage import gaussian_filter1d
smoothed_entropy = gaussian_filter1d(section_entropy, sigma=2)
```

---

## Configuration Parameters

Add to `synth_config.py`:

```python
ENTROPY_CONFIG = {
    # Calculation weights
    'weights': {
        'eval': 0.5,      # Evaluation volatility (primary)
        'tactical': 0.4,  # Tactical density
        'time': 0.1,      # Thinking time (if available)
    },

    # Window sizes for local calculations
    'eval_window': 5,      # Plies for eval volatility
    'tactical_window': 5,  # Plies for tactical density

    # Smoothing
    'smoothing_sigma': 2,  # Gaussian filter sigma

    # Musical thresholds
    'low_threshold': 0.3,   # Below = simple
    'high_threshold': 0.7,  # Above = complex

    # Parameter ranges
    'note_pools': {
        'low': [0, 4],                    # Simple: root-fifth
        'medium': [0, 2, 4, 5, 7],        # Moderate: diatonic
        'high': [0, 1, 2, 3, 4, 5, 6, 7], # Complex: chromatic
    },

    'rhythm_variation_max': 0.5,  # Max ±50% timing variation
    'filter_lfo_range': (0.02, 0.12),  # Hz
    'glide_reduction_max': 0.5,  # Max 50% reduction at high entropy
    'harmony_probability_min': 0.7,  # Start adding harmonies above this entropy
}
```

---

## Testing & Validation

### Qualitative Tests

1. **Listen to opening** (should start simple, build to first exchanges, settle)
2. **Listen to tactical chaos section** (should sound unpredictable, chromatic)
3. **Listen to conversion endgame** (should sound inexorable, simplifying)

### Quantitative Metrics

```python
# Verify entropy correlates with eval volatility
assert np.corrcoef(eval_stdev, entropy_values)[0,1] > 0.7

# Verify entropy range is used
assert 0.0 <= np.min(entropy_values) < 0.2
assert 0.8 < np.max(entropy_values) <= 1.0

# Verify smoothness (no sudden jumps)
entropy_gradient = np.gradient(entropy_values)
assert np.max(np.abs(entropy_gradient)) < 0.3  # Max 0.3 change per ply
```

---

## Future Extensions

### 1. Narrative-Specific Entropy Curves
Different overall narratives could shape the entropy mapping:

**TUMBLING_DEFEAT**: Entropy steadily increases (order → chaos)
```python
entropy_adjusted = base_entropy * (1 + progress * 0.5)
```

**ATTACKING_MASTERPIECE**: Entropy decreases (complex tactics → forced mate)
```python
entropy_adjusted = base_entropy * (1 - progress * 0.5)
```

### 2. Multi-Timescale Entropy
Layer different entropy timescales (Spiegel's approach):
- **Micro** (per note): Tactical density
- **Meso** (per phrase): Eval volatility
- **Macro** (per section): Overall narrative arc

### 3. Entropy as Compositional Meta-Parameter
Use entropy to control ALL layers:
- **Layer 1** (Drone): Detune amount, filter width
- **Layer 2** (Patterns): Pattern complexity, note density
- **Layer 3** (Sequencer): As described above

---

## References

**Laurie Spiegel** (1980s-1990s):
- "The Expanding Universe" - algorithmic composition using process-based systems
- Music Mouse software - real-time generative music with entropy control
- Writings on information theory and musical structure

**Shannon, C.E.** (1948):
- "A Mathematical Theory of Communication" - foundation of information entropy

**Pierce, J.R.** (1961):
- "Symbols, Signals and Noise" - information theory for general audiences

---

## Summary

**What**: Use eval volatility + tactical density to calculate continuous entropy curve

**Why**: Creates organic musical evolution instead of discrete state switches

**How**: Sample entropy at each sequencer step, use it to control note selection, rhythm, filters, harmonies

**Result**: Music that **anticipates** critical moments, **breathes** with position complexity, and creates **emotional arcs** through predictability variation

This is Laurie Spiegel's vision applied to chess: **the information content of the position directly drives the information content of the music**.
