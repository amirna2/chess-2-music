# GestureSynthesizer Implementation

## Overview

`synthesizer.py` implements the **GestureSynthesizer** class - a high-level wrapper around `synth_engine.SubtractiveSynth` that coordinates multi-voice gesture synthesis with time-varying parameters.

## Architecture

**Design Pattern:** Mirrors `synth_composer/core/synthesizer.NoteSynthesizer`
- Thin wrapper around SubtractiveSynth
- Delegates all DSP primitives to the synth engine
- Provides higher-level coordination logic for gesture synthesis

**Key Principle:** Reuse SubtractiveSynth for all low-level operations to:
- Avoid code duplication
- Inherit anti-aliasing and click-prevention benefits
- Maintain architectural consistency across layers
- Leverage existing tested synthesis primitives

## Class: GestureSynthesizer

### Constructor

```python
def __init__(self, synth_engine)
```

**Args:**
- `synth_engine`: SubtractiveSynth instance (shared with other layers)

**Stores:**
- `self.synth`: Reference to synth engine
- `self.sample_rate`: Sample rate from synth engine

### Main Method: synthesize()

```python
def synthesize(pitch_voices, filter_curve, envelope, texture_curve, sample_rate) -> np.ndarray
```

**Synthesis Pipeline:**

1. **Validation:**
   - Verify sample rate matches synth engine
   - Check pitch_voices is non-empty
   - Verify all curves have matching lengths

2. **Oscillator Generation:**
   - Generate sine oscillator for each pitch voice
   - Delegates to `SubtractiveSynth.oscillator_timevarying_pitch()`

3. **Equal-Power Voice Mixing:**
   - Sum all voices
   - Normalize by `sqrt(num_voices)` to prevent volume increase

4. **Noise Texture Blending:**
   - If `noise_ratio > 0`, generate noise via `SubtractiveSynth.generate_noise()`
   - Mix: `(1 - ratio) * oscillators + ratio * noise`

5. **Time-Varying Filter:**
   - Apply filter with time-varying cutoff and resonance
   - Delegates to `SubtractiveSynth.moog_filter_timevarying()`

6. **Amplitude Envelope:**
   - Apply envelope via direct multiplication: `audio *= envelope`

7. **Shimmer Effect (Optional):**
   - If enabled, apply amplitude modulation
   - LFO at specified rate: `0.5 + 0.5 * sin(2π * shimmer_rate * t)`

**Args:**
- `pitch_voices`: List of pitch curves in Hz (List[np.ndarray])
- `filter_curve`: Dict with keys:
  - `'cutoff'`: Cutoff frequency curve in Hz (np.ndarray)
  - `'resonance'`: Resonance curve 0-4 (np.ndarray)
  - `'type'`: Filter type string (for documentation)
- `envelope`: Amplitude envelope 0-1 (np.ndarray)
- `texture_curve`: Dict with keys:
  - `'noise_ratio'`: Noise mix ratio 0-1 (float)
  - `'noise_type'`: 'white' | 'pink' (str)
  - `'shimmer_enable'`: Enable shimmer effect (bool, optional)
  - `'shimmer_rate_hz'`: Shimmer LFO rate in Hz (float, optional)
- `sample_rate`: Audio sample rate in Hz (must match synth_engine)

**Returns:**
- Mono audio buffer (np.ndarray)

**Raises:**
- `ValueError`: If sample_rate mismatch or invalid parameters

### Private Methods

#### _generate_oscillator()
```python
def _generate_oscillator(pitch_curve) -> np.ndarray
```
Delegates to `SubtractiveSynth.oscillator_timevarying_pitch()` with sine waveform.

**Note:** Saw/square waves require per-sample PolyBLEP which is complex for time-varying pitch. Sine is used for simplicity and spectral purity.

#### _generate_noise()
```python
def _generate_noise(num_samples, noise_type) -> np.ndarray
```
Delegates to `SubtractiveSynth.generate_noise()`.

Supported types:
- `'white'`: White noise (flat spectrum)
- `'pink'`: Pink noise (1/f spectrum)

#### _apply_timevarying_filter()
```python
def _apply_timevarying_filter(audio, filter_curve) -> np.ndarray
```
Delegates to `SubtractiveSynth.moog_filter_timevarying()`.

Processes audio in chunks with averaged parameters for efficiency.

#### _apply_shimmer()
```python
def _apply_shimmer(audio, shimmer_rate_hz) -> np.ndarray
```
Implements LFO-based amplitude modulation.

Creates shimmering texture by modulating amplitude at specified rate.

## Usage Example

```python
from synth_engine import SubtractiveSynth
from layer3b.synthesizer import GestureSynthesizer
import numpy as np

# Initialize synth engine
sample_rate = 88200
rng = np.random.default_rng(seed=42)
synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)

# Create gesture synthesizer
gesture_synth = GestureSynthesizer(synth_engine)

# Define parameter curves (1 second)
duration = 1.0
num_samples = int(duration * sample_rate)

# Pitch: glissando from A4 to A3 with harmonic voices
base_pitch = np.exp(np.linspace(np.log(440), np.log(220), num_samples))
pitch_voices = [
    base_pitch,
    base_pitch * (2 ** (4/12)),  # Major third
    base_pitch * (2 ** (7/12))   # Perfect fifth
]

# Filter: sweep from bright to dark
filter_curve = {
    'cutoff': np.exp(np.linspace(np.log(2000), np.log(200), num_samples)),
    'resonance': np.full(num_samples, 0.7),
    'type': 'lowpass'
}

# Envelope: sudden attack with exponential decay
envelope = np.zeros(num_samples)
attack = int(0.01 * sample_rate)  # 10ms
envelope[:attack] = np.linspace(0, 1, attack)
envelope[attack:] = np.exp(np.linspace(0, -4, num_samples - attack))

# Texture: 30% pink noise with shimmer
texture_curve = {
    'noise_ratio': 0.3,
    'noise_type': 'pink',
    'shimmer_enable': True,
    'shimmer_rate_hz': 6.0
}

# Synthesize gesture
audio = gesture_synth.synthesize(
    pitch_voices=pitch_voices,
    filter_curve=filter_curve,
    envelope=envelope,
    texture_curve=texture_curve,
    sample_rate=sample_rate
)

# audio is now a mono numpy array ready for mixing
```

## Integration with Gesture Generation

The GestureSynthesizer is designed to receive parameter curves from the curve generation system:

```python
from layer3b.base import GestureGenerator
from layer3b.archetype_configs import ARCHETYPES

# Initialize generator (uses curve_generators to produce parameter curves)
archetype_config = ARCHETYPES['BLUNDER']
rng = np.random.default_rng(seed=42)
generator = GestureGenerator(archetype_config, rng)

# Generate gesture (internally uses GestureSynthesizer)
moment_event = {'event_type': 'BLUNDER', 'timestamp': 10.5}
section_context = {'tension': 0.8, 'entropy': 0.6, 'scale': [0, 2, 4, 5, 7, 9, 11]}

audio = generator.generate_gesture(
    moment_event=moment_event,
    section_context=section_context,
    sample_rate=88200
)
```

## Performance Characteristics

**Sample Rate:** 88200 Hz (2x 44.1kHz) for anti-aliasing headroom

**Typical Gesture Duration:** 0.5 - 10 seconds

**Voice Count:** 1-5 voices typical (equal-power summing prevents clipping)

**Processing Time:** O(n) where n = total_samples
- Oscillator generation: O(n * num_voices)
- Filter processing: O(n) chunked processing
- Envelope/shimmer: O(n)

**Memory:** ~8 bytes/sample/voice (float64)
- 1 second, 3 voices @ 88.2kHz = ~2MB

## Error Handling

The synthesizer validates inputs and provides clear error messages:

1. **Sample rate mismatch:** Detects if requested rate != synth_engine rate
2. **Empty pitch voices:** Prevents synthesis with no voices
3. **Length mismatches:** Ensures all curves have same length as envelope
4. **Empty envelope:** Detects zero-length envelope

## Testing

Run the test suite:
```bash
python3 test_gesture_synthesizer.py
```

Tests verify:
- Basic multi-voice synthesis
- Equal-power summing behavior
- Noise texture mixing
- Shimmer effect amplitude modulation
- Error handling and validation

## Mathematical Foundations

### Equal-Power Summing
When mixing N voices, divide by `sqrt(N)` instead of `N`:
- Preserves RMS power (perceptual loudness)
- Prevents excessive volume reduction
- Standard practice in audio mixing

### Shimmer LFO
Amplitude modulation formula:
```
amplitude = 0.5 + 0.5 * sin(2π * rate * t)
```
- Range: [0, 1]
- Rate: typically 4-8 Hz for subtle shimmer

### Noise Mixing
Linear crossfade:
```
output = (1 - ratio) * oscillators + ratio * noise
```
- ratio = 0: Pure oscillators
- ratio = 1: Pure noise
- ratio = 0.3: 30% noise, 70% oscillators

## Dependencies

**Required:**
- `numpy`: Numerical computation
- `synth_engine.SubtractiveSynth`: Low-level synthesis primitives

**Required SubtractiveSynth Methods:**
- `oscillator_timevarying_pitch(freq_curve, waveform)`: Time-varying oscillator
- `moog_filter_timevarying(signal, cutoff_curve, resonance_curve)`: Time-varying filter
- `generate_noise(num_samples, noise_type)`: Noise generation

These primitives were added to SubtractiveSynth specifically to support Layer 3b gesture synthesis while maintaining architectural consistency.

## Future Enhancements

**Potential additions:**
1. **Multi-waveform support:** Add triangle waveform option (already supported by oscillator_timevarying_pitch)
2. **Stereo width:** Support stereo positioning via width curves
3. **Velocity sensitivity:** Scale parameters based on moment intensity
4. **Filter morphing:** Support bandpass/highpass filter types beyond lowpass
5. **Ring modulation:** Add carrier modulation for metallic textures

**Performance optimizations:**
1. **Vectorization:** Batch process multiple voices in parallel
2. **Caching:** Cache noise buffers for repeated calls
3. **Adaptive chunk size:** Adjust filter chunk size based on curve variation

## See Also

- `/Users/nathoo/dev/chess-2-music/docs/layer3b_implementation.md`: Full architecture spec
- `/Users/nathoo/dev/chess-2-music/synth_engine.py`: SubtractiveSynth implementation
- `/Users/nathoo/dev/chess-2-music/synth_composer/core/synthesizer.py`: NoteSynthesizer (parallel pattern)
- `/Users/nathoo/dev/chess-2-music/layer3b/curve_generators.py`: Parameter curve generation
