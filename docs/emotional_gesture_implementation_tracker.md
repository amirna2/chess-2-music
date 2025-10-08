# Emotional Gesture Implementation Tracker

**Based on:** `emotional_gesture_generator.md`
**Approach:** Pragmatic, minimal patch (single file implementation)
**Status:** In Progress
**Started:** 2025-10-07

---

## Implementation Checklist

### Phase 1: Core Implementation (moment_generators.py)

- [ ] **1.1 Create `moment_generators.py`**
  - Location: `/Users/nathoo/dev/chess-2-music/moment_generators.py`
  - Dependencies: `numpy`, `dataclasses`

- [ ] **1.2 Implement `GestureSpec` dataclass**
  - Fields: `duration`, `sample_rate`, `pitch_curve`, `amp_curve`, `filter_curve`, `width_curve`, `noise_curve`, `base_freq`, `waveform`, `blend`

- [ ] **1.3 Implement `MomentGestureGenerator` class skeleton**
  - `__init__(self, synth, rng, config)`
  - Store references to synth engine, RNG, config

- [ ] **1.4 Implement helper methods**
  - [ ] `_ease(self, n)` - S-curve easing (smoothstep)
  - [ ] `_gliss_curve(self, n, start, end, shape='exp')` - Pitch glissando
  - [ ] `_amp_shape(self, n, archetype)` - Amplitude envelopes per archetype
  - [ ] `_filter_shape(self, n, archetype)` - Filter trajectories per archetype
  - [ ] `_width_shape(self, n, archetype)` - Stereo width curves per archetype
  - [ ] `_noise_curve(self, n, archetype)` - Noise mix curves per archetype

- [ ] **1.5 Implement archetype-specific amplitude shapes**
  - [ ] BLUNDER: brief suspend → plunge → muffled tail
  - [ ] BRILLIANT: smooth arch (sin^0.7)
  - [ ] TACTICAL_SEQUENCE: even pulses with subtle modulation
  - [ ] TIME_PRESSURE: accelerating tremor with rising amplitude
  - [ ] INACCURACY: quick flicker with asymmetrical decay

- [ ] **1.6 Implement archetype-specific filter shapes**
  - [ ] BLUNDER: 4000Hz → 400Hz (closing choke)
  - [ ] BRILLIANT: 1200Hz → 7200Hz (opening bloom)
  - [ ] TACTICAL_SEQUENCE: 1800Hz ± 800Hz oscillation
  - [ ] TIME_PRESSURE: 900Hz → 3900Hz (accelerating rise)
  - [ ] INACCURACY: 1500Hz with decaying oscillation

- [ ] **1.7 Implement `build_spec()` method**
  - [ ] Pitch motion per archetype:
    - BLUNDER: high → low exponential gliss (2× → 0.5×)
    - BRILLIANT: low → high exponential gliss (0.5× → 4×)
    - TIME_PRESSURE: tremolo with accelerating rate
    - TACTICAL_SEQUENCE: discrete pitch cells (harmonic series)
    - INACCURACY: subtle wobble with downward drift
  - [ ] Apply entropy/tension scaling to curves
  - [ ] Return populated `GestureSpec`

- [ ] **1.8 Implement `render()` method**
  - [ ] Phase accumulation oscillator (sample-accurate pitch tracking)
  - [ ] Block-based filter processing (128-sample chunks)
  - [ ] 1-pole lowpass filter (exponential smoothing)
  - [ ] Noise generation (gaussian) with curve blending
  - [ ] Amplitude envelope application
  - [ ] Soft clipping (tanh) for peak limiting
  - [ ] Return rendered audio buffer

---

### Phase 2: Integration (synth_composer.py)

- [ ] **2.1 Import MomentGestureGenerator**
  - Add import at top of `synth_composer.py`

- [ ] **2.2 Initialize generator in `ChessSynthComposer.__init__()`**
  - Create `self.moment_generator = MomentGestureGenerator(self.synth_layer3, self.rng, self.config)`
  - Location: After `self.synth_layer3` initialization

- [ ] **2.3 Replace Layer 3b sequencer logic in `compose_section()`**
  - [ ] Find Layer 3b moment rendering section (after heartbeat generation)
  - [ ] Replace step-sequencer loop with gesture-based rendering:
    - Create `gesture_buffer = np.zeros_like(sequencer_layer)`
    - Loop over `moment_events`
    - Extract entropy/tension snapshot per event
    - Map event type to archetype
    - Call `moment_generator.build_spec()`
    - Call `moment_generator.render()`
    - Mix gesture into buffer with gain control
  - [ ] Assign `sequencer_layer = gesture_buffer`
  - [ ] Comment out or remove old note-by-note sequencer code

- [ ] **2.4 Update archetype mapping**
  - [ ] Map chess event types to gesture archetypes:
    - `BLUNDER` → BLUNDER
    - `BRILLIANT` / `GREAT` → BRILLIANT
    - `CAPTURE` / `TACTICAL` → TACTICAL_SEQUENCE
    - `TIME_TROUBLE` → TIME_PRESSURE
    - `INACCURACY` / `MISTAKE` → INACCURACY
    - Default → INACCURACY

---

### Phase 3: Configuration (synth_config.py)

- [ ] **3.1 Add GESTURE_LIMITS config (optional)**
  - [ ] `max_peak`: 0.38
  - [ ] `max_rms`: -20.0 dB
  - [ ] `post_sum_normalize`: True
  - Location: In `SynthConfig` dataclass

- [ ] **3.2 Add LAYER_ENABLE toggle for moments**
  - [ ] `'moments': True` in `LAYER_ENABLE` dict
  - Allows disabling gestures without code changes

---

### Phase 4: Testing & Validation

- [ ] **4.1 Unit test gesture rendering**
  - [ ] Test each archetype renders without errors
  - [ ] Verify output length matches duration
  - [ ] Check amplitude stays within bounds
  - [ ] Validate no NaN/Inf values

- [ ] **4.2 Integration test with real game**
  - [ ] Select test game with varied moments (captures, blunders, brilliant moves)
  - [ ] Run full composition pipeline
  - [ ] Listen to Layer 3b in isolation
  - [ ] Listen to full mix

- [ ] **4.3 A/B comparison**
  - [ ] Render same game with OLD sequencer (backup current implementation)
  - [ ] Render same game with NEW gestures
  - [ ] Compare:
    - Musical clarity (do moments punctuate effectively?)
    - Mix balance (do gestures dominate or blend?)
    - Emotional impact (do archetypes convey chess events?)

- [ ] **4.4 Parameter tuning**
  - [ ] Adjust `max_target` level if gestures too loud/quiet
  - [ ] Tweak archetype durations (min 0.25s, max 3.5s)
  - [ ] Fine-tune filter ranges per archetype
  - [ ] Adjust noise blend ratios

---

## Implementation Notes

### Key Differences from Current System

| Aspect | Current (Step Sequencer) | New (Emotional Gestures) |
|--------|-------------------------|--------------------------|
| **Structure** | Grid-based 16-step patterns | Event-triggered sculpted audio |
| **Timing** | Quantized to beat grid | Sample-accurate placement |
| **Shape** | Uniform note envelopes | Archetype-specific curves |
| **Pitch** | Static frequency per step | Time-varying glissandi, tremolo |
| **Character** | Mechanical, repetitive | Emotional, expressive |
| **Duration** | Plays until next event | Fixed duration per gesture (0.25-3.5s) |

### Technical Design Decisions

1. **Simple oscillator**: Uses direct phase accumulation (not SubtractiveSynth) for simplicity
2. **1-pole filter**: Exponential smoothing (faster than Moog ladder, good enough for gestures)
3. **Block-based processing**: 128-sample chunks balance accuracy vs performance
4. **Soft clipping**: `tanh()` prevents harsh peaks from gesture overlaps
5. **No pattern evolution**: Each gesture is self-contained (unlike Layer 2 patterns)

### Files Modified

- ✅ **NEW:** `moment_generators.py` (~250 lines)
- ⚠️ **MODIFIED:** `synth_composer.py` (~30 lines changed in Layer 3b section)
- ⚠️ **MODIFIED:** `synth_config.py` (~5 lines added, optional)

### Rollback Plan

If gestures don't work well:
1. Comment out `gesture_buffer` logic in `synth_composer.py`
2. Uncomment old sequencer logic
3. Set `LAYER_ENABLE['moments'] = False`
4. Keep `moment_generators.py` for future refinement

---

## Success Criteria

✅ **Must Have:**
- Gestures render without crashes
- Audio stays within [-1, 1] range
- Each archetype sounds distinct
- Gestures don't dominate mix (< -18 dBFS RMS)

✅ **Should Have:**
- BLUNDER feels like "falling" (descending gliss, closing filter)
- BRILLIANT feels "ascending" (rising gliss, opening filter)
- TIME_PRESSURE feels "anxious" (tremolo, accelerating)
- TACTICAL_SEQUENCE feels "mechanical" (discrete pitches, even rhythm)
- INACCURACY feels "brief stumble" (short, flickering)

✅ **Nice to Have:**
- Gestures blend smoothly with heartbeat (Layer 3a)
- Entropy/tension modulation is audible
- Professional listeners can identify archetypes blind

---

## Next Session Tasks

**Immediate next steps:**
1. Create `moment_generators.py` file
2. Implement `GestureSpec` dataclass
3. Implement `MomentGestureGenerator` skeleton
4. Implement helper methods (`_ease`, `_gliss_curve`)
5. Implement first archetype (BLUNDER) as proof of concept

**Estimated time:** 2-3 hours for core implementation, 1 hour for integration, 1-2 hours for testing

---

## References

- **Design Doc:** `emotional_gesture_generator.md`
- **Architecture Doc:** `layer3b_implementation.md` (future evolution)
- **Synth Engine:** `synth_engine.py` (not modified in this phase)
- **Current Layer 3b:** `synth_composer.py` lines 1975-2257

---

**Status Legend:**
- ⏳ Not started
- 🔄 In progress
- ✅ Complete
- ⚠️ Blocked / needs decision
- ❌ Abandoned

**Last Updated:** 2025-10-07
