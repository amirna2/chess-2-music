# Synth Engine Architecture & Technical Reference

**A comprehensive guide to the subtractive synthesis engine powering chess-to-music conversion**

This document explains the core audio synthesis techniques, anti-aliasing measures, and architectural decisions that evolved from a simple oscillator to a synthesis engine with sophisticated click prevention and state management. Use this as a reference for understanding the codebase or building your own algorithmic music systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Anti-Click & Anti-Aliasing Techniques](#anti-click--anti-aliasing-techniques)
4. [Phase Continuity System](#phase-continuity-system)
5. [Equal-Power Crossfading](#equal-power-crossfading)
6. [Per-Layer Synth Isolation](#per-layer-synth-isolation)
7. [Moog Filter Implementation](#moog-filter-implementation)
8. [Envelope Generation](#envelope-generation)
9. [Denormal Protection](#denormal-protection)
10. [Lessons Learned](#lessons-learned)

---

## Overview

The `SubtractiveSynth` class provides band-limited waveform generation with time-varying filtering and sophisticated anti-click measures. It evolved through careful debugging of audible artifacts (clicks, pops, phase discontinuities) that plague many software synthesizers.

### Key Features

- **PolyBLEP anti-aliasing** for band-limited waveforms
- **Phase-safe retriggering** to eliminate clicks between notes
- **Equal-power crossfading** for smooth note transitions
- **Per-layer state isolation** to prevent cross-contamination
- **Denormal protection** to avoid CPU spikes
- **Moog-style ladder filter** with state preservation

---

## Core Architecture

### Synthesis Chain

```
Oscillator → Filter (time-varying) → Envelope → Crossfade → Output
    ↓            ↓                      ↓           ↓
PolyBLEP    Moog Ladder           ADSR      Equal-power
            + State                      + Phase continuity
```

### Class Structure

```python
class SubtractiveSynth:
    def __init__(self, sample_rate=44100):
        # Filter state variables (preserved between calls)
        self.filter_z1 = 0.0
        self.filter_z2 = 0.0
        self.filter_z3 = 0.0
        self.filter_z4 = 0.0

        # Phase continuity for click-free retriggering
        self.phase = 0.0
        self.last_env_value = 0.0
        self.last_signal_tail = None
        self.crossfade_samples = 64  # ~1.45ms at 44.1kHz
```

**Design principle**: State persistence enables continuity across note boundaries.

---

## Anti-Click & Anti-Aliasing Techniques

### 1. PolyBLEP (Polynomial Band-Limited Edge Pulse)

**Problem**: Naive waveforms (sawtooth, square) contain infinite harmonics that alias above Nyquist frequency, creating harsh digital artifacts.

**Solution**: PolyBLEP smooths discontinuities at waveform edges using polynomial interpolation.

#### Implementation

```python
def poly_blep(self, dt, phase):
    """
    PolyBLEP anti-aliasing
    dt: normalized frequency (freq / sample_rate)
    phase: phase position (0.0 to 1.0)
    """
    if phase < dt:
        # Beginning of period - rising edge
        t = phase / dt
        return t + t - t * t - 1.0
    elif phase > 1.0 - dt:
        # End of period - falling edge
        t = (phase - 1.0) / dt
        return t * t + t + t + 1.0
    else:
        return 0.0
```

#### Usage in Sawtooth

```python
# Band-limited sawtooth
signal[i] = 2.0 * phase - 1.0
signal[i] -= self.poly_blep(dt, phase)  # Subtract correction
```

**Key insight**: PolyBLEP corrects discontinuities *within* a waveform cycle, but doesn't handle retriggering between notes.

### 2. Envelope Attack Guard

**Problem**: Ultra-short attack times (< 1ms) create near-instantaneous amplitude jumps that generate audible clicks.

**Solution**: Enforce minimum envelope times to ensure smooth transitions.

```python
def adsr_envelope(self, num_samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    # Guard against ultra-short attacks that cause clicks
    attack = max(attack, 0.001)   # Minimum 1ms attack (~44 samples)
    decay = max(decay, 0.001)
    release = max(release, 0.001)
```

**Rationale**: 1ms = 44 samples at 44.1kHz provides enough ramp to avoid perceivable clicks while still sounding "instant" musically.

---

## Phase Continuity System

### The Problem: Phase Discontinuities

**Scenario**: A note retriggers before the previous note's envelope has decayed to zero.

**Before fix**:
```python
# ALWAYS resets phase to 0
def oscillator(self, freq, duration):
    phase = 0.0  # ❌ Discontinuity if previous note hasn't decayed!
```

**Result**: Audible click from sudden waveform jump.

### The Solution: Phase-Safe Retriggering

**Strategy**: Only reset phase when the previous note has fully decayed. Otherwise, continue from the last phase position.

```python
# Phase-safe retrigger: only reset if previous note fully decayed
if self.last_env_value < 1e-3:
    phase = 0.0  # Safe reset - previous note silent
else:
    phase = self.phase  # Continue from last position - smooth!
```

### State Tracking

```python
# Save final phase for next retrigger
self.phase = phase

# Track envelope value if provided
if apply_envelope is not None:
    self.last_env_value = apply_envelope[-1]
```

**Key insight**: The envelope value tells us whether it's safe to reset phase. Below 0.001 = inaudible, safe to reset.

---

## Equal-Power Crossfading

### The Problem: Volume Dip in Linear Crossfades

When crossfading two signals, linear blending causes a perceived volume drop at the midpoint.

**Linear crossfade** (naive):
```python
x = np.linspace(0, 1, xfade_len)
out = old * (1 - x) + new * x
```

At midpoint (x=0.5):
- Amplitude: `0.5 + 0.5 = 1.0` ✓
- **Power**: `0.5² + 0.5² = 0.5` ❌ (50% volume dip!)

### The Solution: Equal-Power Crossfade

Use square root curves to maintain constant perceived loudness.

```python
x = np.linspace(0, 1, xfade_len)
a = np.sqrt(1 - x)  # Equal-power old
b = np.sqrt(x)      # Equal-power new
signal[:xfade_len] = (
    self.last_signal_tail[:xfade_len] * a +
    signal[:xfade_len] * b
)
```

At midpoint (x=0.5):
- Amplitude: `√0.5 + √0.5 ≈ 1.414`
- **Power**: `(√0.5)² + (√0.5)² = 1.0` ✓ (constant!)

### Mathematical Proof

For equal-power:
```
a² + b² = 1
(√(1-x))² + (√x)² = (1-x) + x = 1 ✓
```

**Key insight**: Equal-power maintains perceptual loudness, making crossfades transparent. When implemented correctly, the listener hears seamless transitions rather than the crossfade itself - the technique becomes invisible.

---

## Per-Layer Synth Isolation

### The Problem: Shared State Pollution

**Original architecture**: Single `SubtractiveSynth` instance shared across all 3 layers (drone, patterns, sequencer).

**Issue**: State variables (`phase`, `last_signal_tail`, `filter_z1-4`) got overwritten by whichever layer rendered last, causing:
- Phase discontinuities between layers
- Crossfade domain mismatch (Layer 1's tail mixed with Layer 2's signal)
- Destructive interference → quieter mix (RMS: -25.9 dBFS)

### The Solution: Per-Layer Instances

```python
# Separate synth instances per layer for state isolation
self.synth_layer1 = SubtractiveSynth(self.sample_rate)  # Drone
self.synth_layer2 = SubtractiveSynth(self.sample_rate)  # Patterns
self.synth_layer3 = SubtractiveSynth(self.sample_rate)  # Sequencer
```

### Results

**Before** (shared synth):
```
Pre-normalization peak: -11.6 dBFS
Final RMS: -25.9 dBFS
Crest factor: 22.9 dB
```

**After** (per-layer synths):
```
Pre-normalization peak: -18.0 dBFS
Final RMS: -19.5 dBFS  (+6.4 dB louder!)
Crest factor: 16.5 dB  (tighter, more consistent)
```

**Key insight**: State isolation eliminated destructive interference, increasing RMS by 6.4 dB and improving mix coherence.

---

## Moog Filter Implementation

### 4-Pole Ladder Filter

A digital emulation of the classic Moog synthesizer filter, using 4 cascaded one-pole filters with feedback.

```python
def moog_filter(self, signal, cutoff_hz, resonance=0.0):
    f = 2.0 * np.sin(np.pi * cutoff_hz / self.sample_rate)
    f = np.clip(f, 0.0001, 1.0)
    resonance = np.clip(resonance, 0.0, 4.0)

    y1, y2, y3, y4 = self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4

    for i, x in enumerate(signal):
        # Feedback from output
        x -= resonance * y4

        # Input soft clipping (analog saturation)
        x = np.tanh(x)

        # 4 cascaded one-pole filters
        y1 += f * (x - y1)
        y2 += f * (y1 - y2)
        y3 += f * (y2 - y3)
        y4 += f * (y3 - y4)

        # Output soft clipping
        out[i] = np.tanh(y4)

    # Save states for next call
    self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4 = y1, y2, y3, y4
```

### Key Features

1. **State preservation**: Filter states persist between calls for smooth continuity
2. **Soft clipping**: `tanh()` provides analog-style saturation, prevents harsh digital distortion
3. **Stable frequency mapping**: `2.0 * sin(π * fc / fs)` ensures stability at high cutoffs
4. **Resonance control**: Feedback from output (y4) creates characteristic peak

### Time-Varying Filter

Filters are typically called in chunks with envelope modulation:

```python
chunk_size = 64
for i in range(0, len(signal), chunk_size):
    # Envelope modulates cutoff over time
    env_val = filt_env[i]
    current_cutoff = filter_base + (filter_env_amount * env_val)

    # Apply filter to chunk (state carries through)
    filtered[i:end] = self.moog_filter(chunk, current_cutoff, resonance)
```

**Design choice**: Chunk-based processing balances time-varying behavior with computational efficiency.

---

## Envelope Generation

### ADSR with Exponential Curves

```python
def adsr_envelope(self, num_samples, attack=0.01, decay=0.1,
                  sustain=0.7, release=0.2, curve=0.3):
    # Attack - exponential rise (slow start, fast finish)
    t = np.linspace(0, 1, attack_samples)
    envelope[current:current+attack_samples] = np.power(t, 1.0 - curve)

    # Decay - exponential fall (fast start, slow finish)
    t = np.linspace(0, 1, decay_samples)
    decay_curve = 1.0 - np.power(t, curve)
    envelope[current:end] = 1.0 - decay_curve * (1.0 - sustain)

    # Sustain - constant level
    envelope[current:end] = sustain

    # Release - exponential fall to zero
    t = np.linspace(0, 1, release_samples)
    release_curve = np.power(t, curve)
    envelope[current:] = sustain * (1.0 - release_curve)
```

### Curve Parameter

- `curve = 0.1`: Gentle, linear-like
- `curve = 0.3`: Musical (default)
- `curve = 1.0`: Aggressive, sharp transitions

**Exponential curves** sound more natural than linear ramps because they match how acoustic instruments behave.

---

## Denormal Protection

### The Problem: CPU Spikes from Tiny Values

**Denormal numbers** are extremely small floating-point values (< ~10⁻³⁰⁸) that force the CPU into slow subnormal arithmetic paths, causing:
- Processing delays → audible ticks/pops
- Performance spikes during envelope decay tails
- Filter state accumulation of tiny errors

### The Solution: Zero Tiny Values

```python
# Denormal protection - prevent CPU spikes from tiny values
signal[np.abs(signal) < 1e-20] = 0.0
```

Applied after:
- Oscillator generation
- Envelope application
- Filter processing

**Threshold**: `1e-20` ≈ -400 dBFS (completely inaudible) but well above denormal range.

**Key insight**: Force CPU to stay in fast normal arithmetic by zeroing values that are inaudible anyway.

---

## Lessons Learned

### 1. Crossfade in the Right Domain

**Wrong**: Crossfade raw oscillator output
```
Osc → [Crossfade] → Filter → Envelope → Output
```
Result: Click! Filter and envelope changes create discontinuities.

**Right**: Crossfade post-VCA (after filter + envelope)
```
Osc → Filter → Envelope → [Crossfade] → Output
```
Result: Smooth! Crossfading the actual audible signal.

### 2. State Isolation Matters

When multiple voices/layers share a synth instance:
- Phase state pollution
- Crossfade tail mismatch
- Destructive interference

**Solution**: One synth instance per voice/layer.

### 3. Simple Optimizations Can Backfire

**Case study**: Intelligent oversampling for filter sweeps.

**Attempt**: Detect steep cutoff changes → oversample at 2x → prevent aliasing.

**Result**: Created MORE artifacts due to:
- **Cutoff modulation noise**: Coefficient recalculation at 2x rate introduced numerical jitter, producing crackle
- **Incorrect coefficient scaling**: Resonance/feedback terms not properly scaled for 2x rate → doubled resonance → distortion
- **Naive upsampling/downsampling**: Linear interpolation and truncation artifacts
- **State continuity issues**: Transitions between 1x and 2x rates caused discontinuities

**Root cause**: Oversampling helps when aliasing exists, but applying it to an already-stable filter can exaggerate feedback noise and quantization artifacts.

**Better alternatives** (if filter aliasing is audible):
1. **Smooth cutoff transitions**: `self.cutoff += 0.05 * (target_cutoff - self.cutoff)`
2. **Clamp resonance**: `self.resonance = min(self.resonance, 0.9)`
3. **Limit cutoff range**: Keep `cutoff <= Nyquist * 0.45` (already done: `0.95` in code)
4. Only add oversampling for specific fast modulations where aliasing is proven audible

**Lesson**: In the context of an experimental algorithmic composition system, the stable single-rate Moog ladder is cleaner. Don't add complexity that introduces new problems when the simpler solution works well.

### 4. Envelope Guards Are Essential

1ms minimum attack/decay/release prevents:
- Amplitude discontinuities
- Perceived "clicks" on fast notes
- Harsh digital artifacts

**Musical impact**: Negligible (1ms feels instant), but audio quality improves dramatically.

### 5. Equal-Power > Linear

Linear crossfades have a perceivable volume dip. Always use equal-power curves (`√x`) for transparent transitions.

---

## Implementation Checklist

Building your own synthesis engine? Use this checklist:

### Anti-Aliasing
- [ ] PolyBLEP or similar for discontinuous waveforms
- [ ] Sine/triangle don't need it (naturally band-limited)

### Click Prevention
- [ ] Phase continuity tracking
- [ ] Envelope value monitoring (`< 1e-3` = safe reset)
- [ ] Minimum 1ms envelope times

### Crossfading
- [ ] Equal-power curves (`√(1-x)` and `√x`)
- [ ] Crossfade post-VCA (after filter + envelope)
- [ ] 64-128 samples (~1.5-3ms) crossfade window

### State Management
- [ ] Per-voice synth instances (or voice ID parameter)
- [ ] Filter state preservation between calls
- [ ] Phase tracking across note boundaries

### CPU Protection
- [ ] Denormal protection (`< 1e-20` → 0)
- [ ] Soft clipping with `tanh()` instead of hard clipping
- [ ] Chunk-based processing for time-varying filters

---

## References

### PolyBLEP
- Välimäki, V., & Huovilainen, A. (2007). "Antialiasing Oscillators in Subtractive Synthesis"
- https://www.kvraudio.com/forum/viewtopic.php?t=375517

### Moog Filter
- Huovilainen, A. (2004). "Non-linear digital implementation of the Moog ladder filter"
- https://www.musicdsp.org/en/latest/Filters/

### Equal-Power Panning
- https://www.cs.cmu.edu/~music/icm-online/readings/panlaws/

### Denormal Numbers
- https://en.wikipedia.org/wiki/Denormal_number
- Intel optimization guides on DAZ (Denormals Are Zero) mode

---

## Conclusion

This synthesis engine evolved from debugging real-world artifacts in a chess-to-music conversion system. Each technique addresses a specific audible problem:

- **PolyBLEP**: Harsh aliasing on sawtooth/square waves
- **Phase continuity**: Clicks on fast note retriggering
- **Equal-power crossfade**: Volume dips during transitions
- **Per-layer isolation**: State pollution in multi-layer mixes
- **Denormal protection**: CPU-induced audio glitches
- **Envelope guards**: Ultra-short attack clicks

The result: Click-free synthesis suitable for algorithmic composition systems.

**For questions or improvements**: This is a living document. If you fork this repo and discover new techniques, please contribute back!

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Synth Engine Version**: See `synth_engine.py` for implementation details
