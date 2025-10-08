# SubtractiveSynth Enhancement Analysis

**Date:** 2025-10-07
**Purpose:** Identify primitives needed in `synth_engine.py` to support high-level synthesizers (NoteSynthesizer, GestureSynthesizer)

---

## Current SubtractiveSynth Methods (13 total)

### Core Primitives
1. `oscillator(freq, duration, waveform)` - Band-limited waveform generation
2. `moog_filter(signal, cutoff_hz, resonance)` - 4-pole ladder filter
3. `adsr_envelope(num_samples, A, D, S, R)` - Exponential ADSR
4. `filter_envelope(num_samples, A, D, S, R)` - Filter modulation envelope
5. `poly_blep(dt, phase)` - Anti-aliasing helper

### High-Level Methods
6. `create_synth_note(...)` - Complete subtractive synth voice
7. `supersaw(...)` - Detuned saw ensemble (Layer 3b)
8. `pitch_sweep_note(...)` - Frequency glissando (R2D2 beeps)
9. `create_heartbeat_cycle(...)` - LUB-dub pattern (Layer 3a)

### Internal Helpers (heartbeat-specific)
10. `_create_linear_adsr(...)` - Linear envelope (vs exponential)
11. `_apply_lowpass_filter(...)` - Simple scipy butterworth
12. `_generate_beat_waveform(...)` - Sine wave generator

---

## GestureSynthesizer Requirements

Based on `layer3b_implementation.md`, gestures need:

### 1. ✅ **Time-Varying Pitch** (SUPPORTED)
```python
# Current approach (from pitch_sweep_note):
phase = np.cumsum(freq_curve / sample_rate)
signal = np.sin(2 * np.pi * phase)
```
**Status:** ✅ Already implemented in `pitch_sweep_note()`
**Recommendation:** Extract as primitive `oscillator_timevarying(freq_curve, waveform)`

### 2. ⚠️ **Multi-Voice Rendering** (PARTIALLY SUPPORTED)
```python
# Currently: Must call oscillator() multiple times and mix manually
voice1 = synth.oscillator(440, 2.0)
voice2 = synth.oscillator(550, 2.0)
mixed = (voice1 + voice2) / 2
```
**Status:** ⚠️ Works but inefficient (creates intermediate arrays)
**Recommendation:** Add `oscillator_multivoice(freq_list, duration, waveform)`

### 3. ❌ **Time-Varying Filter with Arbitrary Curves** (NOT SUPPORTED)
```python
# Needed for gestures:
filter_curve = {
    'cutoff': [1000, 1200, 800, ..., 150],  # Per-sample or per-chunk
    'resonance': [0.4, 0.5, 0.7, ..., 0.9],
    'type': 'bandpass->lowpass'  # Filter morphing
}
audio = synth.moog_filter_timevarying(signal, filter_curve)
```
**Status:** ❌ Current `moog_filter()` only accepts scalar cutoff
**Recommendation:** Add `moog_filter_timevarying(signal, cutoff_curve, resonance_curve)`

### 4. ❌ **Filter Type Switching** (NOT SUPPORTED)
```python
# Gesture needs:
- Lowpass (current: ✅ moog_filter)
- Highpass (❌ missing)
- Bandpass (❌ missing)
- Morphing between types (❌ missing)
```
**Status:** ❌ Only lowpass implemented
**Recommendation:** Add filter type parameter or separate methods

### 5. ❌ **Noise Generation** (NOT SUPPORTED)
```python
# Gestures need noise blending:
white_noise = synth.generate_noise(num_samples, 'white')
pink_noise = synth.generate_noise(num_samples, 'pink')
```
**Status:** ❌ Not implemented
**Recommendation:** Add `generate_noise(num_samples, noise_type)`

### 6. ⚠️ **Envelope Variety** (LIMITED)
```python
# Current: Only exponential ADSR
# Gestures need:
- Linear ADSR (✅ exists as _create_linear_adsr, but private)
- S-curve (sigmoid) envelopes (❌ missing)
- Gated pulses (❌ missing)
```
**Status:** ⚠️ Linear exists but is private/heartbeat-specific
**Recommendation:** Expose `linear_adsr()` publicly, add `sigmoid_envelope()`

---

## Recommended Enhancements

### Priority 1: Essential for GestureSynthesizer

#### 1.1 **Time-Varying Filter** (HIGH PRIORITY)
```python
def moog_filter_timevarying(self, signal, cutoff_curve, resonance_curve=None):
    """
    Apply Moog filter with time-varying parameters.

    Args:
        signal: Input audio
        cutoff_curve: Cutoff in Hz (scalar, array, or None for static)
        resonance_curve: Resonance 0-4 (scalar, array, or None)

    Returns:
        Filtered audio
    """
    # If scalar, convert to array
    if np.isscalar(cutoff_curve):
        cutoff_curve = np.full(len(signal), cutoff_curve)

    if resonance_curve is None or np.isscalar(resonance_curve):
        resonance_curve = np.full(len(signal), resonance_curve or 0.5)

    # Block-based processing (balance between accuracy and speed)
    chunk_size = 128  # ~1.5ms at 88.2kHz
    filtered = np.zeros_like(signal)

    for i in range(0, len(signal), chunk_size):
        end = min(i + chunk_size, len(signal))
        chunk = signal[i:end]

        # Average parameters over chunk
        cutoff = np.mean(cutoff_curve[i:end])
        resonance = np.mean(resonance_curve[i:end])

        # Apply filter (state carries through)
        filtered[i:end] = self.moog_filter(chunk, cutoff, resonance)

    return filtered
```

#### 1.2 **Multi-Voice Oscillator** (MEDIUM PRIORITY)
```python
def oscillator_multivoice(self, freq_list, duration, waveform='saw'):
    """
    Generate multiple oscillator voices efficiently.

    Args:
        freq_list: List of frequencies (Hz)
        duration: Duration in seconds
        waveform: 'saw', 'square', 'sine', 'triangle', 'pulse'

    Returns:
        Mixed audio (equal-power summing)
    """
    num_samples = int(duration * self.sample_rate)
    mixed = np.zeros(num_samples)

    for freq in freq_list:
        voice = self.oscillator(freq, duration, waveform)
        mixed += voice

    # Equal-power normalization
    mixed /= np.sqrt(len(freq_list))

    return mixed
```

#### 1.3 **Time-Varying Pitch Oscillator** (MEDIUM PRIORITY)
Extract from `pitch_sweep_note()`:
```python
def oscillator_timevarying_pitch(self, freq_curve, waveform='sine'):
    """
    Generate oscillator with per-sample frequency modulation.

    Args:
        freq_curve: Array of frequencies (Hz), one per sample
        waveform: Currently only 'sine' (others need PolyBLEP per-sample)

    Returns:
        Audio signal
    """
    # Phase accumulation (integral of frequency)
    phase = np.cumsum(freq_curve / self.sample_rate)

    if waveform == 'sine':
        signal = np.sin(2 * np.pi * phase)
    elif waveform == 'triangle':
        # Triangle from phase
        phase_frac = phase % 1.0
        signal = np.where(phase_frac < 0.5,
                         4.0 * phase_frac - 1.0,
                         3.0 - 4.0 * phase_frac)
    else:
        # Saw/square with time-varying pitch need per-sample PolyBLEP
        # TODO: This is complex, start with sine/triangle only
        raise NotImplementedError(f"Time-varying {waveform} requires per-sample PolyBLEP")

    return signal
```

#### 1.4 **Noise Generator** (LOW PRIORITY)
```python
def generate_noise(self, num_samples, noise_type='white'):
    """
    Generate noise signal.

    Args:
        num_samples: Length of noise buffer
        noise_type: 'white', 'pink', 'brown'

    Returns:
        Noise signal (normalized to [-1, 1])
    """
    if noise_type == 'white':
        return self.rng.standard_normal(num_samples)

    elif noise_type == 'pink':
        # Simple pink noise: low-pass filtered white
        from scipy.signal import butter, filtfilt
        white = self.rng.standard_normal(num_samples)
        b, a = butter(1, 0.1)  # 1st order LP at 0.1 * Nyquist
        pink = filtfilt(b, a, white)
        return pink / np.std(pink)  # Normalize

    elif noise_type == 'brown':
        # Brownian noise: cumsum of white
        white = self.rng.standard_normal(num_samples)
        brown = np.cumsum(white)
        return brown / np.std(brown)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
```

---

### Priority 2: Nice-to-Have for Advanced Gestures

#### 2.1 **Additional Filter Types**
```python
def bandpass_filter(self, signal, center_freq, bandwidth, resonance=0.5):
    """Bandpass filter (for TIME_PRESSURE gesture)"""
    # TODO: Implement using scipy.signal or state-variable filter
    pass

def highpass_filter(self, signal, cutoff_hz, resonance=0.5):
    """Highpass filter (for BRILLIANT gesture opening)"""
    # TODO: Derive from moog_filter by inverting response
    pass
```

#### 2.2 **S-Curve Envelope**
```python
def sigmoid_envelope(self, num_samples, attack_samples, release_samples, steepness=4.0):
    """
    S-curve (sigmoid) envelope for gradual attacks.

    Args:
        num_samples: Total envelope length
        attack_samples: Attack duration
        release_samples: Release duration
        steepness: Sigmoid steepness (higher = sharper transition)

    Returns:
        Envelope (0-1)
    """
    envelope = np.zeros(num_samples)
    sustain_samples = num_samples - attack_samples - release_samples

    # Attack (sigmoid rise)
    t = np.linspace(0, 1, attack_samples)
    envelope[:attack_samples] = 0.5 * (1 + np.tanh(steepness * (t - 0.5)))

    # Sustain
    envelope[attack_samples:attack_samples + sustain_samples] = 1.0

    # Release (sigmoid fall)
    t = np.linspace(0, 1, release_samples)
    envelope[attack_samples + sustain_samples:] = 0.5 * (1 + np.tanh(steepness * (0.5 - t)))

    return envelope
```

#### 2.3 **Public Linear ADSR**
```python
def linear_adsr(self, num_samples, attack, decay, sustain, release):
    """
    Linear ADSR envelope (vs exponential).

    Public version of _create_linear_adsr for general use.
    """
    return self._create_linear_adsr(
        duration_sec=num_samples / self.sample_rate,
        attack=attack,
        decay=decay,
        sustain=sustain,
        release=release
    )
```

---

## Implementation Strategy

### Phase 1: Critical Enhancements (Week 1)
1. ✅ Add `moog_filter_timevarying()` - enables arbitrary filter curves
2. ✅ Add `oscillator_timevarying_pitch()` - enables glissandi/tremolo
3. ✅ Add `generate_noise()` - enables texture blending

### Phase 2: Convenience Methods (Week 2)
4. ✅ Add `oscillator_multivoice()` - efficient multi-voice rendering
5. ✅ Make `linear_adsr()` public - expose existing functionality
6. ✅ Add `sigmoid_envelope()` - gradual attack/release

### Phase 3: Advanced Filters (Week 3, Optional)
7. ⚠️ Add `bandpass_filter()` - BP sweep for TIME_PRESSURE
8. ⚠️ Add `highpass_filter()` - HP opening for BRILLIANT

---

## Testing Strategy

### Unit Tests for New Primitives
```python
# test_synth_engine_enhancements.py

def test_timevarying_filter():
    synth = SubtractiveSynth(88200)
    signal = synth.oscillator(440, 1.0, 'saw')

    # Sweep filter from 200Hz to 2000Hz
    cutoff_curve = np.linspace(200, 2000, len(signal))
    filtered = synth.moog_filter_timevarying(signal, cutoff_curve)

    assert len(filtered) == len(signal)
    assert np.max(np.abs(filtered)) <= 1.5  # With gain compensation

def test_timevarying_pitch():
    synth = SubtractiveSynth(88200)

    # Glissando from 880Hz to 220Hz (2 octaves down)
    freq_curve = np.exp(np.linspace(np.log(880), np.log(220), 88200))
    signal = synth.oscillator_timevarying_pitch(freq_curve, 'sine')

    assert len(signal) == len(freq_curve)
    assert -1.0 <= signal.min() <= signal.max() <= 1.0

def test_noise_generation():
    synth = SubtractiveSynth(88200)

    white = synth.generate_noise(88200, 'white')
    pink = synth.generate_noise(88200, 'pink')

    # White noise should have flat spectrum (roughly equal energy at all freqs)
    # Pink noise should have 1/f spectrum (less high-freq energy)

    assert len(white) == 88200
    assert len(pink) == 88200
    assert np.std(white) > np.std(pink)  # Pink is low-pass filtered
```

---

## Breaking Changes / Compatibility

### ✅ All additions are BACKWARDS COMPATIBLE
- New methods, no changes to existing signatures
- Existing code (Layers 1, 2, 3a, 3b current) unaffected
- Only enhancement, no refactoring

### ⚠️ Performance Considerations
- `moog_filter_timevarying()` is slower than static filter (per-chunk parameter updates)
- `oscillator_multivoice()` creates N oscillators (memory overhead)
- Both are acceptable for offline synthesis (not realtime)

---

## Summary

### Current Capabilities
✅ Fixed-pitch oscillators with anti-aliasing
✅ Static Moog lowpass filter with state preservation
✅ Exponential ADSR envelopes
✅ Supersaw ensemble (Layer 3b current)
✅ Heartbeat patterns (Layer 3a)

### Missing for GestureSynthesizer
❌ Time-varying filter curves
❌ Multi-voice efficient rendering
❌ Noise generation
❌ Bandpass/highpass filters
❌ Public linear/sigmoid envelopes

### Recommendation
**Implement Phase 1 enhancements** (time-varying filter, pitch, noise) to enable `GestureSynthesizer` to use `SubtractiveSynth` as foundation, avoiding code duplication and inheriting all click-prevention/anti-aliasing benefits.

**Total new code:** ~150 lines (3 methods + helpers)
**Testing effort:** ~100 lines (unit tests)
**Integration effort:** Update `layer3b_implementation.md` design doc

This keeps `SubtractiveSynth` as the **single source of truth** for synthesis primitives.
