"""
Gesture audio synthesis engine.

Renders parameter curves to audio using SubtractiveSynth as foundation.

ARCHITECTURE: This is a HIGH-LEVEL wrapper around synth_engine.SubtractiveSynth,
similar to how synth_composer/core/synthesizer.NoteSynthesizer wraps it for Layer 2.

Coordinates multi-voice synthesis with time-varying parameters:
- Multi-voice oscillators with time-varying pitch (via SubtractiveSynth)
- Time-varying filter (via SubtractiveSynth.moog_filter_timevarying)
- Amplitude envelope (direct multiplication)
- Noise texture (via SubtractiveSynth.generate_noise)
- Shimmer effect (amplitude modulation)
"""

import numpy as np
from typing import List, Dict, Any


class GestureSynthesizer:
    """
    High-level gesture synthesis coordinator.

    Uses SubtractiveSynth for all low-level DSP (oscillators, filters, noise).
    This wrapper orchestrates time-varying parameter curves and multi-voice rendering.
    """

    def __init__(self, synth_engine):
        """
        Initialize with existing SubtractiveSynth engine.

        Args:
            synth_engine: SubtractiveSynth instance (shared with other layers)
        """
        self.synth = synth_engine
        self.sample_rate = synth_engine.sample_rate

    def synthesize(self,
                  pitch_voices: List[np.ndarray],
                  filter_curve: Dict[str, np.ndarray],
                  envelope: np.ndarray,
                  texture_curve: Dict[str, Any],
                  sample_rate: int) -> np.ndarray:
        """
        Synthesize gesture audio from parameter curves.

        Args:
            pitch_voices: List of pitch curves (Hz) for each voice
            filter_curve: Dict with 'cutoff', 'resonance', 'type'
            envelope: Amplitude envelope curve (0-1)
            texture_curve: Dict with noise parameters
            sample_rate: Audio sample rate (must match synth_engine)

        Returns:
            Mono audio buffer (numpy array)

        Raises:
            ValueError: If sample_rate mismatch or invalid parameters
        """
        # Validate sample rate
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: synthesizer={self.sample_rate}, "
                f"requested={sample_rate}"
            )

        # Validate inputs
        if not pitch_voices:
            raise ValueError("No pitch voices provided")

        total_samples = len(envelope)
        if total_samples == 0:
            raise ValueError("Envelope is empty")

        # Validate all pitch curves have same length
        for i, pitch_curve in enumerate(pitch_voices):
            if len(pitch_curve) != total_samples:
                raise ValueError(
                    f"Pitch curve {i} length {len(pitch_curve)} != "
                    f"envelope length {total_samples}"
                )

        # Generate and mix oscillators for all voices
        audio = np.zeros(total_samples)

        for pitch_curve in pitch_voices:
            voice_audio = self._generate_oscillator(pitch_curve)
            audio += voice_audio

        # Normalize multi-voice sum using equal-power summing
        # This prevents volume increase with more voices
        audio /= np.sqrt(len(pitch_voices))

        # Blend noise texture if requested
        noise_ratio = texture_curve.get('noise_ratio', 0.0)
        if noise_ratio > 0:
            noise = self._generate_noise(total_samples, texture_curve.get('noise_type', 'white'))
            # Mix: (1 - ratio) * oscillators + ratio * noise
            audio = (1 - noise_ratio) * audio + noise_ratio * noise

        # Apply time-varying filter
        audio = self._apply_timevarying_filter(audio, filter_curve)

        # Apply amplitude envelope
        audio *= envelope

        # Optional shimmer effect (amplitude modulation)
        if texture_curve.get('shimmer_enable', False):
            shimmer_rate = texture_curve.get('shimmer_rate_hz', 6.0)
            audio = self._apply_shimmer(audio, shimmer_rate)

        return audio

    def _generate_oscillator(self, pitch_curve: np.ndarray) -> np.ndarray:
        """
        Generate oscillator with time-varying pitch.

        Delegates to SubtractiveSynth.oscillator_timevarying_pitch().

        Args:
            pitch_curve: Frequency curve in Hz (numpy array)

        Returns:
            Audio signal (same length as pitch_curve)
        """
        # Use sine wave (saw/square need per-sample PolyBLEP which is complex)
        return self.synth.oscillator_timevarying_pitch(pitch_curve, waveform='sine')

    def _generate_noise(self, num_samples: int, noise_type: str) -> np.ndarray:
        """
        Generate noise signal.

        Delegates to SubtractiveSynth.generate_noise().

        Args:
            num_samples: Length of noise buffer
            noise_type: 'white' | 'pink'

        Returns:
            Noise signal (normalized to ±1.0)
        """
        return self.synth.generate_noise(num_samples, noise_type)

    def _apply_timevarying_filter(self,
                                  audio: np.ndarray,
                                  filter_curve: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply time-varying filter.

        Delegates to SubtractiveSynth.moog_filter_timevarying().

        Args:
            audio: Input audio signal
            filter_curve: Dict with 'cutoff', 'resonance', 'type' keys

        Returns:
            Filtered audio
        """
        cutoff_curve = filter_curve['cutoff']
        resonance_curve = filter_curve['resonance']

        # Use SubtractiveSynth time-varying filter
        return self.synth.moog_filter_timevarying(audio, cutoff_curve, resonance_curve)

    def _apply_shimmer(self, audio: np.ndarray, shimmer_rate_hz: float) -> np.ndarray:
        """
        Apply shimmer effect (amplitude modulation).

        Implements LFO-based amplitude modulation for shimmering texture.

        Args:
            audio: Input audio signal
            shimmer_rate_hz: LFO frequency in Hz

        Returns:
            Shimmered audio
        """
        t = np.arange(len(audio)) / self.sample_rate
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * shimmer_rate_hz * t)
        return audio * lfo
