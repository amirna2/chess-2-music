# Melodic Layer Design Document

## Overview

The **Melodic Layer** is a new alternative to Layer 2 that generates deterministic, musically coherent melodies using Laurie Spiegel's compositional principles. Unlike the current Layer 2 (which uses Markov chains and state machines for random note selection), the Melodic Layer uses **curve-based pitch generation** and **rule-based voice leading** to create phrases with clear contours, stepwise motion, and harmonic structure.

## Design Principles (Based on Laurie Spiegel Research)

1. **DETERMINISTIC, NOT RANDOM**: No Markov chains or probabilistic note selection
2. **CONTINUOUS FUNCTIONS**: Pitch evolves as smooth curves over time, not discrete random events
3. **RULE-BASED VOICE LEADING**: Follows counterpoint rules (stepwise motion, parallel/contrary motion)
4. **HARMONIC AUTOMATION**: Automatic doubling at intervals (3rds, 5ths, octaves)
5. **PHRASE-ORIENTED**: Focus on melodic shapes and contours, not individual notes
6. **SINGLE RESPONSIBILITY**: Each phrase generator creates ONE specific melodic gesture

## Architecture

### System Structure

```
Melodic Layer
├── melodic_patterns/           # Melody generators (like synth_composer/patterns/)
│   ├── base.py                # Base class with common curve logic
│   ├── opening_melodies.py    # Melody generators for opening sections
│   ├── middlegame_melodies.py # Melody generators for middlegame sections
│   └── endgame_melodies.py    # Melody generators for endgame sections
├── melodic_config.py          # Config loader (like synth_config.py)
└── melodic_coordinator.py     # Main coordinator (like PatternCoordinator)
```

### Integration with Existing System

The Melodic Layer **replaces** Layer 2 (section-based patterns) when enabled in config:

```yaml
# config.yaml
composition:
  layer_mode: "melodic"  # Options: "pattern" (current layer2) or "melodic" (new)
```

Layer structure remains:
- **Layer 1 (Drone)**: Long-form game narrative arc
- **Layer 2 (MELODIC)**: Section-based melodic phrases ← NEW
- **Layer 3a (Heartbeat)**: Rhythmic tension motif
- **Layer 3b (Gestures)**: Key moment gestures

## Melodic Layer Configuration

### Config Structure

```yaml
composition:
  layer_mode: "melodic"  # Enable melodic layer

  melodic_layer:
    # Global settings
    waveform: "triangle"  # SAME waveform for all sections
    base_register: 220    # A3 - melodic register (not bass)

    # Voice leading rules
    voice_leading:
      max_leap_semitones: 7         # Prefer stepwise motion
      parallel_motion_prob: 0.6     # 60% parallel, 40% contrary
      stepwise_motion_bias: 0.8     # Strong bias toward steps

    # Harmonic doubling
    harmony:
      doubling_intervals: [0, 7, 12]  # Unison, 5th, Octave
      num_voices: 2                    # Default voice count
      voice_spacing: "close"           # Close voicing

    # Section-specific melody generators
    # Each generator fills the ENTIRE section duration (30-90+ seconds)
    sections:
      opening:
        melody_types:
          - sharp_theory_melody      # Fast, energetic (Sicilian)
          - positional_melody        # Calm, contemplative (French)
          - solid_melody            # Grounded, stable (QGD)

      middlegame:
        melody_types:
          - complex_struggle_melody  # Tense, cautious
          - king_hunt_melody        # Aggressive pursuit
          - building_pressure_melody # Gradual intensification

      endgame:
        melody_types:
          - flawless_conversion_melody  # Methodical, inevitable
          - drawn_ending_melody         # Peaceful, unresolved
          - decisive_ending_melody      # Strong resolution
```

## Melody Generators

### Base Melody Generator

All melody generators inherit from `MelodicGenerator`:

```python
class MelodicGenerator:
    """Base class for deterministic continuous melody generation."""

    def generate_section(self, section_duration, section_context, params):
        """Generate continuous melody for entire section duration.

        Args:
            section_duration: FULL section duration in seconds (e.g., 45s)
            section_context: {tension, entropy, narrative, scale}
            params: {waveform, filter, resonance, config, sample_rate}

        Returns:
            Continuous pitch curve spanning entire section (no gaps)
        """
        # 1. Generate pitch curve for FULL section (deterministic function)
        pitch_curve = self.generate_pitch_curve(section_duration, section_context, params)

        # 2. Apply voice leading rules
        voices = self.apply_voice_leading(pitch_curve, params)

        # 3. Add harmonic doubling
        harmony_voices = self.add_harmonic_doubling(voices, params)

        # 4. Create continuous note events (fills entire section)
        events = self.create_continuous_events(harmony_voices, section_duration, params)

        return events
```

### Curve-Based Pitch Generation

Unlike Layer 2's random note selection, melodies use **deterministic pitch curves** that span the **entire section duration**:

#### Example: Ascending Melody

```python
def _pitch_ascending_melody(config, section_duration, scale, section_context):
    """Generate ascending melodic curve spanning entire section (stepwise with goal)."""

    # Start on tonic, end on fifth (or octave for longer sections)
    start_note = 0
    # Goal changes based on section length
    end_note = 4 if section_duration < 30 else 7  # Fifth or octave

    # Calculate number of notes to fill section
    note_rate = 2.0  # notes per second (adjustable)
    num_notes = int(section_duration * note_rate)

    # Generate curve: stepwise ascent over ENTIRE section
    curve = []
    current_degree = start_note

    for i in range(num_notes):
        progress = i / num_notes

        # Goal-directed: gradually move toward end_note
        if current_degree < end_note:
            # Step up (90% of time) or stay (10% for rhythmic variation)
            if progress < 0.9:  # Deterministic threshold
                current_degree += 1

        # Convert scale degree to frequency
        freq = scale[current_degree % len(scale)]
        curve.append(freq)

    return np.array(curve)  # Array length = num_notes (fills section)
```

#### Example: Arch Melody

```python
def _pitch_arch_melody(config, section_duration, scale, section_context):
    """Generate arch contour spanning entire section (up then down)."""

    # Calculate number of notes to fill entire section
    note_rate = 1.5  # notes per second
    num_notes = int(section_duration * note_rate)
    apex = num_notes // 2  # Peak at midpoint of section

    curve = []
    current_degree = 0  # Start on tonic

    for i in range(num_notes):
        if i < apex:
            # Ascending: step up (first half of section)
            if i % 2 == 0:  # Every other note
                current_degree += 1
        else:
            # Descending: step down (second half of section)
            if i % 2 == 0:
                current_degree -= 1

        current_degree = np.clip(current_degree, 0, len(scale) - 1)
        freq = scale[current_degree]
        curve.append(freq)

    return np.array(curve)  # Fills entire section
```

### Voice Leading Rules

Applies traditional counterpoint rules:

```python
def apply_voice_leading(self, base_curve, params):
    """Apply voice leading rules to generate second voice."""

    rules = params['config'].MELODIC_VOICE_LEADING

    # Start with base voice
    voice1 = base_curve.copy()
    voice2 = np.zeros_like(voice1)

    # Initialize second voice at interval (3rd or 5th)
    initial_interval = 4  # Major 3rd
    voice2[0] = voice1[0] * (2 ** (initial_interval / 12))

    # Apply motion rules
    for i in range(1, len(voice1)):
        # Calculate interval to maintain
        target_interval = initial_interval

        # Motion type decision (deterministic based on position)
        if i % 8 < 5:  # Parallel motion 62.5% of time
            # Parallel: maintain interval
            voice2[i] = voice1[i] * (2 ** (target_interval / 12))
        else:  # Contrary motion 37.5%
            # Contrary: move opposite direction
            v1_motion = voice1[i] - voice1[i-1]
            voice2[i] = voice2[i-1] - v1_motion * 0.5

        # Enforce stepwise motion (max leap)
        max_leap_ratio = 2 ** (rules['max_leap_semitones'] / 12)
        leap = abs(voice2[i] / voice2[i-1])
        if leap > max_leap_ratio:
            # Reduce leap to stepwise
            voice2[i] = voice2[i-1] * 1.05946  # Minor 2nd

    return [voice1, voice2]
```

### Harmonic Doubling

Automatic doubling at consonant intervals (Spiegel's "harmonic automation"):

```python
def add_harmonic_doubling(self, voices, params):
    """Add harmonic doubling at configured intervals."""

    harmony_config = params['config'].MELODIC_HARMONY
    intervals = harmony_config['doubling_intervals']  # [0, 7, 12]

    doubled_voices = []

    for voice in voices:
        for interval in intervals:
            if interval == 0:
                # Unison (original voice)
                doubled_voices.append(voice.copy())
            else:
                # Transpose by interval
                ratio = 2 ** (interval / 12)
                doubled_voices.append(voice * ratio)

    return doubled_voices
```

## Melody Types by Section

### Opening Melodies

Match opening theory narratives and fill entire opening section:

#### Sharp Theory Melody
- **Narrative**: SHARP_THEORY (Sicilian, Najdorf)
- **Contour**: Fast ascending runs with leaps
- **Motion**: Rapid stepwise ascent, occasional jumps to upper register
- **Character**: Energetic, tactical, unpredictable

```python
class SharpTheoryMelody(MelodicGenerator):
    """Fast, aggressive continuous melody for sharp openings."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE opening section
        # Rapid ascending phrases with jumps
        # No randomness - deterministic pattern
        # Example: 1-2-3-5-6-8 (skipping 4th, 7th)
        # Repeats/varies pattern to fill section_duration
        pass
```

#### Positional Theory Melody
- **Narrative**: POSITIONAL_THEORY (French, English)
- **Contour**: Slow, contemplative melody with long notes
- **Motion**: Stepwise motion around tonic/third/fifth
- **Character**: Calm, strategic, patient

```python
class PositionalTheoryMelody(MelodicGenerator):
    """Calm, strategic continuous melody for positional openings."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE opening section
        # Slow tempo, long notes
        # Emphasizing stable intervals (tonic, third, fifth)
        # Minimal motion, lots of repetition
        pass
```

#### Solid Theory Melody
- **Narrative**: SOLID_THEORY (QGD, Slav)
- **Contour**: Grounded bass-register melodies
- **Motion**: Repetitive patterns (tonic-fifth-third-fifth)
- **Character**: Stable, solid, predictable

```python
class SolidTheoryMelody(MelodicGenerator):
    """Grounded, stable continuous melody for solid openings."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE opening section
        # Lower register (drop octave)
        # Repetitive harmonic pattern
        # Predictable, building-block structure
        pass
```

### Middlegame Melodies

Fill entire middlegame section:

#### Complex Struggle Melody
- **Narrative**: COMPLEX_STRUGGLE
- **Contour**: Tense, hovering around tonic with cautious steps
- **Motion**: Bias toward returning to tonic (gravity)
- **Character**: Hesitant, calculating, tense

```python
class ComplexStruggleMelody(MelodicGenerator):
    """Tense continuous melody with gravitational pull to tonic."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE middlegame section
        # Start on tonic
        # Explore nearby scale degrees (2nd, 3rd, 4th)
        # Always return to tonic (70% of section time on tonic)
        # Longer notes on tonic = thinking
        pass
```

#### King Hunt Melody
- **Narrative**: KING_HUNT
- **Contour**: Aggressive ascending melody with driving rhythm
- **Motion**: Forceful leaps upward, relentless ascent
- **Character**: Pursuing, aggressive, inevitable

```python
class KingHuntMelody(MelodicGenerator):
    """Aggressive ascending continuous melody for attacks."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE middlegame section
        # Start low, drive upward throughout section
        # Larger leaps acceptable (4th, 5th, octave)
        # Relentless ascent spanning full section
        pass
```

#### Building Pressure Melody
- **Narrative**: BUILDING_PRESSURE
- **Contour**: Gradual ascent with increasing density
- **Motion**: Start slow/sparse, accelerate and fill in
- **Character**: Growing tension, accumulation

```python
class BuildingPressureMelody(MelodicGenerator):
    """Gradually intensifying continuous melody."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE middlegame section
        # Start with long notes, sparse (beginning of section)
        # Gradually shorten notes (acceleration over section)
        # Gradually fill in scale degrees (more notes toward end)
        # End with rapid figures in upper register
        pass
```

### Endgame Melodies

Fill entire endgame section:

#### Flawless Conversion Melody
- **Narrative**: FLAWLESS_CONVERSION
- **Contour**: Methodical stepwise melody with clear direction
- **Motion**: Precise, calculated steps (Fischer technique)
- **Character**: Inevitable, methodical, controlled

```python
class FlawlessConversionMelody(MelodicGenerator):
    """Methodical continuous melody for precise endgame technique."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE endgame section
        # Stepwise motion ONLY (no leaps)
        # Clear goal direction (ascend OR descend, not both)
        # Even note durations (mechanical precision)
        # Ends on target note (resolution)
        pass
```

#### Drawn Ending Melody
- **Narrative**: DRAWN_ENDING
- **Contour**: Circular melody (return to starting point)
- **Motion**: Tonic ↔ fourth (peaceful rocking)
- **Character**: Unresolved, balanced, peaceful

```python
class DrawnEndingMelody(MelodicGenerator):
    """Circular continuous melody for drawn games."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE endgame section
        # Circular motion: tonic → fourth → tonic (repeating)
        # No strong resolution (avoid dominant)
        # Equal weight on both notes
        # Fades out without finality
        pass
```

#### Decisive Ending Melody
- **Narrative**: DECISIVE_ENDING
- **Contour**: Strong resolution to tonic
- **Motion**: Descending (defeat) or ascending (victory)
- **Character**: Definitive, resolved, final

```python
class DecisiveEndingMelody(MelodicGenerator):
    """Resolving continuous melody for decisive endings."""

    def generate_pitch_curve(self, section_duration, context, params):
        # Generate continuous melody for ENTIRE endgame section
        # Based on overall_narrative (DEFEAT or VICTORY)
        # Descending: 7 → 5 → 4 → 2 → 1 (somber resolution)
        # Ascending: 1 → 2 → 4 → 5 → 7 → 1 (triumphant)
        # Strong cadence on final tonic at END of section
        pass
```

## Technical Implementation

### Note Event Structure

Unlike Layer 2's discrete note events, Melodic Layer uses **pitch curves**:

```python
class MelodicNoteEvent(NoteEvent):
    """Extended NoteEvent with pitch curve support."""

    def __init__(self, pitch_curve, duration, timestamp, velocity, waveform,
                 filter_base, filter_env_amount, resonance,
                 amp_env, filter_env, extra_context):
        """
        Args:
            pitch_curve: np.ndarray of frequencies over time (not single freq)
            duration: Total curve duration in seconds
            ... (other params same as NoteEvent)
        """
        # Store curve instead of single frequency
        self.pitch_curve = pitch_curve
        self.duration = duration
        # ... rest of initialization
```

### Synthesis Integration

The synth engine must be updated to handle pitch curves:

```python
# In SubtractiveSynth
def render_melodic_event(self, event):
    """Render note event with time-varying pitch curve."""

    num_samples = int(event.duration * self.sample_rate)

    # Resample pitch curve to match sample count
    pitch_curve_samples = np.interp(
        np.linspace(0, len(event.pitch_curve), num_samples),
        np.arange(len(event.pitch_curve)),
        event.pitch_curve
    )

    # Generate oscillator with varying frequency
    phase = 0
    signal = np.zeros(num_samples)

    for i in range(num_samples):
        freq = pitch_curve_samples[i]
        phase += 2 * np.pi * freq / self.sample_rate
        signal[i] = self.oscillator(phase, event.waveform)

    # Apply filter and envelope (same as before)
    # ...

    return signal
```

## Migration Path

### Phase 1: Core Infrastructure
1. Create `melodic_patterns/base.py` with `MelodicGenerator`
2. Create `melodic_config.py` for config loading
3. Update `synth_engine.py` to support pitch curves
4. Add `layer_mode` config switch

### Phase 2: Opening Melodies
1. Implement `SharpTheoryMelody` (fills entire opening section)
2. Implement `PositionalTheoryMelody` (fills entire opening section)
3. Implement `SolidTheoryMelody` (fills entire opening section)
4. Test with opening-heavy games (verify full section coverage)

### Phase 3: Middlegame Melodies
1. Implement `ComplexStruggleMelody` (fills entire middlegame section)
2. Implement `KingHuntMelody` (fills entire middlegame section)
3. Implement `BuildingPressureMelody` (fills entire middlegame section)
4. Test with tactical games (verify full section coverage)

### Phase 4: Endgame Melodies
1. Implement `FlawlessConversionMelody` (fills entire endgame section)
2. Implement `DrawnEndingMelody` (fills entire endgame section)
3. Implement `DecisiveEndingMelody` (fills entire endgame section)
4. Test with endgame-focused games (verify full section coverage)

### Phase 5: Integration
1. Create `melodic_coordinator.py` (replaces `PatternCoordinator`)
2. Wire into main `synth_composer.py`
3. Full system testing (verify no gaps between sections)
4. Documentation and examples

## Testing Strategy

### Unit Tests
- Test each phrase generator produces deterministic output
- Test voice leading rules (no excessive leaps)
- Test harmonic doubling intervals
- Test pitch curve generation

### Integration Tests
- Test melodic layer with each section narrative
- Test layer switching (pattern vs melodic mode)
- Test section transitions and crossfades
- Test waveform consistency across sections
- **Verify melodies fill entire section** (no gaps, no silence)

### Musical Tests
- **Section Coverage**: Verify melody spans full section duration (30-90+ seconds)
- **Stepwise Motion**: Verify >80% of intervals are steps (≤2 semitones)
- **Clear Contours**: Verify melodies have identifiable shapes (arch, ascent, descent)
- **Goal Direction**: Verify melodies reach target notes by end of section
- **Voice Leading**: Verify parallel/contrary motion ratios
- **Harmonic Clarity**: Verify doubling creates consonance
- **No Silence**: Verify continuous sound throughout section

## Success Criteria

The Melodic Layer is successful if:

1. ✓ **Fills entire section** (no gaps, continuous melody for full 30-90+ seconds)
2. ✓ Output is **deterministic** (same input → same output)
3. ✓ Melodies have **clear contours** (arch, ascent, descent, circular)
4. ✓ Uses **stepwise motion** (>80% steps, <20% leaps)
5. ✓ **No random note selection** (no Markov chains, no probabilistic choices)
6. ✓ Voice leading follows **counterpoint rules** (parallel/contrary motion)
7. ✓ Harmony uses **automatic doubling** at consonant intervals
8. ✓ Each melody has **structure** (beginning, middle, end over section)
9. ✓ Melodies are **memorable** and **singable** (not algorithmic noise)
10. ✓ Same **waveform** used across all sections
11. ✓ Ties into **section narratives** (opening/middlegame/endgame)

## Comparison: Layer 2 vs Melodic Layer

| Aspect | Layer 2 (Pattern) | Melodic Layer |
|--------|-------------------|---------------|
| **Duration** | Fills entire section | Fills entire section |
| **Pitch Selection** | Random (Markov chains, state machines) | Deterministic (pitch curves) |
| **Motion** | Discrete random jumps | Continuous smooth curves |
| **Voice Leading** | None | Counterpoint rules |
| **Harmony** | Random or fixed intervals | Automatic doubling |
| **Melodic Structure** | Random walk, no goal | Clear contours, goal-directed |
| **Contour** | Unpredictable (random walk) | Defined shapes (arch, ascent) |
| **Inspiration** | Generic algorithmic | Laurie Spiegel techniques |
| **Memorability** | Low (sounds random) | High (singable melodies) |
| **Determinism** | Non-deterministic (random seed) | Fully deterministic |

## References

- Laurie Spiegel: "Music Mouse - An Intelligent Instrument"
- Laurie Spiegel: "Random, Algorithmic, and Intelligent Music"
- Traditional counterpoint and voice leading rules
- Layer 3b curve generation system (existing codebase)
- Spectromorphology: phrase shapes and contours

## Next Steps

1. Review this design document
2. Get approval on architecture
3. Begin Phase 1 implementation
4. Iterate based on musical results
