"""
Base class for unified gesture generation across all archetypes.

All gesture archetypes (BLUNDER, BRILLIANT, TIME_PRESSURE, etc.) use this
single class with different configuration parameters. The generation pipeline
is identical for all gestures - only the parameter curves differ.

This mirrors the pattern system architecture used in synth_composer/patterns/.
"""

import numpy as np
from typing import Dict, Any

from .curve_generators import (
    generate_pitch_curve,
    generate_harmony,
    generate_filter_curve,
    generate_envelope,
    generate_texture_curve
)
from .synthesizer import GestureSynthesizer
from .utils import compute_phases, finalize_audio


class GestureGenerator:
    """
    Unified gesture generator for all archetypes.

    All gestures follow the same synthesis pipeline:
    1. Compute duration based on archetype and section context
    2. Compute phase timeline (pre-shadow → impact → bloom → decay → residue)
    3. Generate parameter curves (pitch, harmony, filter, envelope, texture)
    4. Synthesize audio using shared synthesis engine
    5. Finalize (normalize, safety clip)

    Archetypes differ only in configuration parameters, not code.
    This enables easy addition of new gesture types without modifying the pipeline.
    """

    def __init__(self, archetype_config: Dict[str, Any], rng: np.random.Generator,
                 synth_engine=None):
        """
        Initialize gesture generator.

        Args:
            archetype_config: Configuration dict defining gesture characteristics
                            (duration, phases, pitch, harmony, filter, envelope, texture)
            rng: NumPy random generator for reproducible randomness
            synth_engine: SubtractiveSynth instance (optional, for GestureSynthesizer)

        Raises:
            ValueError: If archetype_config is missing required keys
        """
        # Validate configuration
        required_keys = ['duration_base', 'phases', 'pitch', 'harmony',
                        'filter', 'envelope', 'texture']
        missing_keys = [k for k in required_keys if k not in archetype_config]
        if missing_keys:
            raise ValueError(
                f"Archetype config missing required keys: {missing_keys}"
            )

        self.config = archetype_config
        self.rng = rng

        # Initialize synthesizer with synth engine if provided
        if synth_engine is not None:
            self.synthesizer = GestureSynthesizer(synth_engine)
        else:
            # Create default synthesizer (will fail if SubtractiveSynth not available)
            # This allows testing of generator logic without synthesis
            self.synthesizer = None

    def generate_gesture(self,
                        moment_event: Dict[str, Any],
                        section_context: Dict[str, Any],
                        sample_rate: int) -> np.ndarray:
        """
        Generate complete audio gesture for a moment event.

        Pipeline:
        1. Compute duration → samples
        2. Compute phase timeline
        3. Generate pitch curve
        4. Generate harmony voices from pitch curve
        5. Generate filter curve
        6. Generate amplitude envelope
        7. Generate texture parameters
        8. Synthesize audio (oscillators → filter → envelope)
        9. Finalize (normalize, safety clip)

        Args:
            moment_event: Moment metadata dict with keys:
                         - event_type: str (e.g., 'BLUNDER', 'BRILLIANT')
                         - timestamp: float (seconds)
                         - move_number: int
                         - (other event-specific metadata)
            section_context: Section-level parameters dict with keys:
                            - tension: float [0-1] (game tension)
                            - entropy: float [0-1] (game chaos/complexity)
                            - scale: str (musical scale name)
                            - key: str (musical key)
            sample_rate: Audio sample rate in Hz (e.g., 88200)

        Returns:
            Mono audio buffer (numpy array, float32, normalized to ±peak_limit)

        Raises:
            ValueError: If synthesizer not initialized or invalid parameters
        """
        if self.synthesizer is None:
            raise ValueError(
                "Cannot synthesize audio: GestureGenerator initialized without synth_engine. "
                "Provide synth_engine parameter to __init__() for synthesis capability."
            )

        # Validate sample_rate
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {sample_rate}. Must be positive.")

        # Validate section_context has required keys
        if 'tension' not in section_context or 'entropy' not in section_context:
            raise ValueError(
                "section_context must contain 'tension' and 'entropy' keys"
            )

        # Step 1: Compute duration
        duration = self._compute_duration(section_context)
        total_samples = int(duration * sample_rate)

        if total_samples <= 0:
            raise ValueError(
                f"Computed duration {duration}s results in zero samples at {sample_rate} Hz"
            )

        # Step 2: Compute phase timeline
        phases = compute_phases(
            self.config['phases'],
            total_samples,
            section_context,
            self.rng
        )

        # Step 3: Generate pitch curve (fundamental frequency trajectory)
        pitch_curve = generate_pitch_curve(
            self.config['pitch'],
            phases,
            section_context,
            total_samples,
            self.rng,
            sample_rate
        )

        # Step 4: Generate harmony voices (pitch curve → multiple detuned/harmonized voices)
        harmony_voices = generate_harmony(
            self.config['harmony'],
            pitch_curve,
            phases,
            section_context,
            self.rng
        )

        # Step 5: Generate filter curve (cutoff, resonance over time)
        filter_curve = generate_filter_curve(
            self.config['filter'],
            phases,
            section_context,
            total_samples,
            self.rng,
            sample_rate
        )

        # Step 6: Generate amplitude envelope
        envelope = generate_envelope(
            self.config['envelope'],
            phases,
            total_samples,
            self.rng,
            sample_rate
        )

        # Step 7: Generate texture parameters (noise, shimmer)
        texture_curve = generate_texture_curve(
            self.config['texture'],
            phases,
            section_context,
            total_samples,
            self.rng
        )

        # Step 8: Synthesize audio
        # GestureSynthesizer orchestrates: oscillators → filter → envelope → texture
        audio = self.synthesizer.synthesize(
            pitch_voices=harmony_voices,
            filter_curve=filter_curve,
            envelope=envelope,
            texture_curve=texture_curve,
            sample_rate=sample_rate
        )

        # Step 9: Finalize (normalize to target RMS, apply safety clipping)
        audio = finalize_audio(
            audio,
            peak_limit=self.config.get('peak_limit', 0.8),
            rms_target=self.config.get('rms_target', -18.0)
        )

        return audio

    def _compute_duration(self, section_context: Dict[str, Any]) -> float:
        """
        Compute gesture duration based on archetype and section context.

        Duration formula:
            duration = base + (tension × tension_scale) + (entropy × entropy_scale)

        This allows gestures to adapt to game state:
        - High tension → longer/shorter gestures (archetype-dependent)
        - High entropy → more/less extended gestures (archetype-dependent)

        Args:
            section_context: Section-level parameters with keys:
                           - tension: float [0-1]
                           - entropy: float [0-1]

        Returns:
            Duration in seconds, clamped to [0.5, 10.0]

        Mathematical grounding:
            Linear scaling provides predictable, musically coherent duration mapping.
            Clamping ensures gestures remain in perceptually useful range:
            - Min 0.5s: Enough time for attack/decay phases
            - Max 10.0s: Avoids excessively long gestures that lose narrative impact
        """
        base = self.config['duration_base']
        tension_scale = self.config.get('duration_tension_scale', 0.0)
        entropy_scale = self.config.get('duration_entropy_scale', 0.0)

        tension = section_context.get('tension', 0.5)
        entropy = section_context.get('entropy', 0.5)

        # Linear duration scaling
        duration = base + (tension * tension_scale) + (entropy * entropy_scale)

        # Clamp to reasonable range (0.5s to 10.0s)
        # This prevents both too-short gestures (incomplete envelopes) and
        # excessively long gestures (loss of moment character)
        return np.clip(duration, 0.5, 10.0)
