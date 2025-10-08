"""
EMOTIONAL GESTURE GENERATOR FOR LAYER 3B

Generates self-contained audio gestures for chess moment events.
Each gesture has archetype-specific shape (amplitude, pitch, filter, noise curves).

Based on: emotional_gesture_generator.md
Architecture: Simple, pragmatic implementation (single file)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GestureSpec:
    """
    Complete specification for rendering a musical gesture.

    All curves are numpy arrays of same length (num_samples).
    """
    duration: float              # Seconds
    sample_rate: int             # Hz
    pitch_curve: np.ndarray      # Frequency (Hz) per sample
    amp_curve: np.ndarray        # Amplitude (0-1) per sample
    filter_curve: np.ndarray     # Cutoff frequency (Hz) per sample
    width_curve: np.ndarray      # Stereo width (0-1) per sample (reserved)
    noise_curve: np.ndarray      # Noise mix ratio (0-1) per sample
    base_freq: float             # Reference frequency (Hz)
    waveform: str                # 'saw' or 'triangle'
    blend: float                 # Overall mix amount (0-1)


class MomentGestureGenerator:
    """
    Generates emotional gestures for chess moments.

    Five archetype shapes:
    - BLUNDER: Suspended high → fast downward gliss, closing filter
    - BRILLIANT: Rising gliss, opening filter, shimmer
    - TACTICAL_SEQUENCE: Discrete pitch cells, precise pulses
    - TIME_PRESSURE: Accelerating tremor, rising tension
    - INACCURACY: Brief flicker, asymmetrical stumble
    """

    def __init__(self, synth, rng, config):
        """
        Initialize gesture generator.

        Args:
            synth: SubtractiveSynth instance (for sample_rate reference)
            rng: NumPy random generator (for reproducible noise)
            config: SynthConfig instance (for parameters)
        """
        self.synth = synth
        self.rng = rng
        self.config = config

    # =========================================================================
    # HELPER METHODS: Curve Generation
    # =========================================================================

    def _ease(self, n):
        """
        Smoothstep easing function (S-curve).

        Args:
            n: Number of samples

        Returns:
            Eased curve (0-1)
        """
        t = np.linspace(0, 1, n)
        return t * t * (3 - 2 * t)

    def _gliss_curve(self, n, start, end, shape='exp'):
        """
        Generate pitch glissando curve.

        Args:
            n: Number of samples
            start: Start frequency (Hz)
            end: End frequency (Hz)
            shape: 'exp' (exponential) or 'lin' (linear)

        Returns:
            Frequency curve (Hz per sample)
        """
        t = np.linspace(0, 1, n)
        if shape == 'exp':
            # Exponential feels more musical (equal steps in log space)
            return start * (end / start) ** t
        else:
            # Linear
            return start + (end - start) * t

    # =========================================================================
    # AMPLITUDE SHAPES (per archetype)
    # =========================================================================

    def _amp_shape(self, n, archetype):
        """
        Generate amplitude envelope for archetype.

        Args:
            n: Number of samples
            archetype: Archetype name string

        Returns:
            Amplitude curve (0-1)
        """
        t = np.linspace(0, 1, n)

        if archetype == 'BLUNDER':
            # Brief suspend → plunge → muffled tail
            a = np.zeros(n)
            hold = int(n * 0.15)      # 15% suspended
            fall = int(n * 0.35)      # 35% falling

            # Suspended phase (pre-echo)
            a[:hold] = 0.4 + 0.3 * self._ease(hold)

            # Plunge (impact)
            a[hold:hold + fall] = np.linspace(a[hold - 1], 1.0, fall)

            # Tail (decay + residue)
            tail = n - (hold + fall)
            if tail > 0:
                a[hold + fall:] = np.linspace(1.0, 0.1, tail)

            return a

        elif archetype == 'BRILLIANT':
            # Smooth arch (ascending bloom)
            return np.sin(np.pi * t) ** 0.7

        elif archetype == 'TACTICAL_SEQUENCE':
            # Even pulses with subtle modulation
            pulses = np.sin(2 * np.pi * (6 + 2 * t) * t)
            base = 0.3 + 0.7 * np.clip(pulses, 0, 1)
            return base * (0.95 + 0.05 * np.cos(8 * np.pi * t))

        elif archetype == 'TIME_PRESSURE':
            # Accelerating tremor with rising amplitude
            accel = t ** 0.35
            trem = 0.5 + 0.5 * np.sin(2 * np.pi * (4 + 20 * accel) * t)
            return (0.2 + 0.8 * accel) * trem

        elif archetype == 'INACCURACY':
            # Quick flicker, asymmetrical
            a = np.zeros(n)
            peak = int(n * 0.25)
            a[:peak] = np.linspace(0, 0.6, peak)

            mid = int(n * 0.15)
            a[peak:peak + mid] = np.linspace(0.6, 0.3, mid)

            a[peak + mid:] = np.linspace(0.3, 0.0, n - (peak + mid))
            return a

        else:
            # Default: simple ease
            return self._ease(n)

    # =========================================================================
    # FILTER SHAPES (per archetype)
    # =========================================================================

    def _filter_shape(self, n, archetype):
        """
        Generate filter cutoff curve for archetype.

        Args:
            n: Number of samples
            archetype: Archetype name string

        Returns:
            Cutoff frequency curve (Hz per sample)
        """
        t = np.linspace(0, 1, n)

        if archetype == 'BLUNDER':
            # Closing choke: 4000Hz → 400Hz
            return np.linspace(4000, 400, n)

        elif archetype == 'BRILLIANT':
            # Opening bloom: 1200Hz → 7200Hz
            return 1200 + 6000 * (t ** 0.8)

        elif archetype == 'TACTICAL_SEQUENCE':
            # Rhythmic oscillation around 1800Hz
            return 1800 + 800 * np.sin(2 * np.pi * 8 * t)

        elif archetype == 'TIME_PRESSURE':
            # Accelerating rise: 900Hz → 3900Hz
            return 900 + 3000 * (t ** 1.4)

        elif archetype == 'INACCURACY':
            # Brief wobble with decay
            return 1500 + 500 * np.sin(2 * np.pi * 3 * t) * np.exp(-3 * t)

        else:
            # Default: gentle sweep
            return np.linspace(800, 1400, n)

    # =========================================================================
    # WIDTH SHAPES (stereo, reserved for future)
    # =========================================================================

    def _width_shape(self, n, archetype):
        """
        Generate stereo width curve for archetype.

        Args:
            n: Number of samples
            archetype: Archetype name string

        Returns:
            Width curve (0-1, reserved for stereo implementation)
        """
        t = np.linspace(0, 1, n)

        if archetype == 'BRILLIANT':
            # Expanding width
            return t ** 0.6

        elif archetype == 'BLUNDER':
            # Collapsing width
            return np.concatenate([
                np.linspace(0.6, 0.1, int(n * 0.7)),
                np.linspace(0.1, 0.0, n - int(n * 0.7))
            ])

        elif archetype == 'TIME_PRESSURE':
            # Rising width (anxious spread)
            return 0.2 + 0.5 * (t ** 1.2)

        elif archetype == 'TACTICAL_SEQUENCE':
            # Oscillating width
            return 0.3 + 0.2 * np.sin(2 * np.pi * 6 * t)

        elif archetype == 'INACCURACY':
            # Narrowing
            return np.linspace(0.15, 0.0, n)

        else:
            # Default: moderate width
            return np.linspace(0.2, 0.4, n)

    # =========================================================================
    # NOISE CURVES (per archetype)
    # =========================================================================

    def _noise_curve(self, n, archetype):
        """
        Generate noise mix ratio curve for archetype.

        Args:
            n: Number of samples
            archetype: Archetype name string

        Returns:
            Noise ratio curve (0-1)
        """
        t = np.linspace(0, 1, n)

        if archetype == 'BLUNDER':
            # High noise during impact
            return np.concatenate([
                np.linspace(0, 0.5, int(n * 0.2)),
                np.linspace(0.5, 0.05, n - int(n * 0.2))
            ])

        elif archetype == 'BRILLIANT':
            # Rising shimmer
            return (t ** 2) * 0.4

        elif archetype == 'TACTICAL_SEQUENCE':
            # Moderate, rhythmic
            return 0.15 + 0.1 * np.sin(2 * np.pi * 10 * t)

        elif archetype == 'TIME_PRESSURE':
            # Increasing noise (tension)
            return 0.2 + 0.5 * (t ** 1.3)

        elif archetype == 'INACCURACY':
            # Brief noise spike
            return np.linspace(0.25, 0.0, n)

        else:
            # Default: low noise
            return np.linspace(0.1, 0.2, n)

    # =========================================================================
    # SPECIFICATION BUILDER
    # =========================================================================

    def build_spec(self, archetype, base_freq, duration, entropy, tension):
        """
        Build complete GestureSpec for archetype.

        Args:
            archetype: Archetype name ('BLUNDER', 'BRILLIANT', etc.)
            base_freq: Base frequency (Hz)
            duration: Duration (seconds)
            entropy: Position complexity (0-1)
            tension: Section tension (0-1)

        Returns:
            GestureSpec with all curves populated
        """
        n = int(duration * self.synth.sample_rate)
        if n < 32:
            n = 32  # Minimum 32 samples

        # =====================================================================
        # PITCH MOTION (archetype-specific)
        # =====================================================================

        if archetype == 'BLUNDER':
            # Suspended high → fast downward gliss
            pitch_curve = self._gliss_curve(n, base_freq * 2, base_freq * 0.5, 'exp')

        elif archetype == 'BRILLIANT':
            # Rising 2-3 octave glide
            pitch_curve = self._gliss_curve(n, base_freq * 0.5, base_freq * 4, 'exp')

        elif archetype == 'TIME_PRESSURE':
            # Narrow oscillation rising microtonally (tremolo)
            t = np.linspace(0, 1, n)
            tremolo_rate = 4 + entropy * 10  # 4-14 Hz based on complexity
            pitch_curve = base_freq * (1 + 0.015 * np.sin(2 * np.pi * tremolo_rate * t))

        elif archetype == 'TACTICAL_SEQUENCE':
            # Fixed pitch cells (harmonic series)
            cells = [1, 4/3, 3/2, 2]  # Root, 4th, 5th, octave
            seq = np.array([base_freq * cells[i % 4] for i in range(n)])
            pitch_curve = seq

        elif archetype == 'INACCURACY':
            # Tiny upward wobble then slip
            t = np.linspace(0, 1, n)
            pitch_curve = base_freq * (1 + 0.03 * np.sin(2 * np.pi * 3 * t) - 0.05 * t)

        else:
            # Default: constant pitch
            pitch_curve = np.full(n, base_freq)

        # =====================================================================
        # GENERATE CURVES
        # =====================================================================

        amp_curve = self._amp_shape(n, archetype)
        filter_curve = self._filter_shape(n, archetype)
        width_curve = self._width_shape(n, archetype)
        noise_curve = self._noise_curve(n, archetype)

        # =====================================================================
        # ENTROPY/TENSION SCALING
        # =====================================================================

        # Entropy modulates amplitude (0.7-1.3x)
        amp_curve *= (0.7 + 0.6 * entropy)

        # Tension modulates filter brightness (0.8-1.2x)
        filter_curve *= (0.8 + 0.4 * tension)

        # =====================================================================
        # WAVEFORM SELECTION
        # =====================================================================

        # Bright archetypes use saw, dark ones use triangle
        if archetype in ['BRILLIANT', 'TACTICAL_SEQUENCE', 'TIME_PRESSURE']:
            waveform = 'saw'
        else:
            waveform = 'triangle'

        return GestureSpec(
            duration=duration,
            sample_rate=self.synth.sample_rate,
            pitch_curve=pitch_curve,
            amp_curve=amp_curve,
            filter_curve=filter_curve,
            width_curve=width_curve,
            noise_curve=noise_curve,
            base_freq=base_freq,
            waveform=waveform,
            blend=0.8
        )

    # =========================================================================
    # AUDIO RENDERER
    # =========================================================================

    def render(self, spec: GestureSpec):
        """
        Render GestureSpec to audio.

        Args:
            spec: GestureSpec with all curves

        Returns:
            Audio buffer (mono, numpy array)
        """
        n = len(spec.amp_curve)
        audio = np.zeros(n)
        sr = spec.sample_rate

        # =====================================================================
        # TONAL CORE: Oscillator with time-varying pitch + filter
        # =====================================================================

        # Phase accumulation for sample-accurate pitch tracking
        phase = 0.0
        block_size = 128  # Process in chunks for efficiency

        for i in range(0, n, block_size):
            end = min(i + block_size, n)
            freq_block = spec.pitch_curve[i:end]

            # Generate oscillator for this block
            # Simple sine for now (could extend to saw/triangle)
            local = np.sin(2 * np.pi * np.cumsum(freq_block) / sr + phase)
            phase = (phase + 2 * np.pi * np.sum(freq_block) / sr) % (2 * np.pi)

            # Apply 1-pole lowpass filter (exponential smoothing)
            cutoff_block = spec.filter_curve[i:end]
            alpha = np.clip(cutoff_block / (sr * 0.5), 0.001, 0.99)

            # Filter state
            y = 0
            filt_out = np.zeros_like(local)
            for k in range(len(local)):
                y = y + alpha[k] * (local[k] - y)
                filt_out[k] = y

            audio[i:end] = filt_out

        # =====================================================================
        # NOISE COMPONENT
        # =====================================================================

        noise = self.rng.standard_normal(n) * spec.noise_curve
        audio += noise * 0.7  # Blend noise at 70% of curve value

        # =====================================================================
        # AMPLITUDE ENVELOPE
        # =====================================================================

        audio *= spec.amp_curve

        # =====================================================================
        # SOFT CLIPPING (prevent harsh peaks)
        # =====================================================================

        audio = np.tanh(audio * 1.8) * 0.55

        return audio
