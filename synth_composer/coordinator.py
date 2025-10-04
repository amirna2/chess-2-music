"""
PatternCoordinator - Main orchestrator for pattern generation.

Manages pattern registry and generates audio from narrative names.
"""

import numpy as np
from typing import List, Dict, Any
from .core.note_event import NoteEvent
from .core.audio_buffer import AudioBuffer
from .core.synthesizer import NoteSynthesizer
from .patterns.markov import MarkovChainPattern
from .patterns.state_machine import (
    KingHuntPattern,
    DesperateDefensePattern,
    TacticalChaosPattern,
    CrushingAttackPattern
)
from .patterns.theory import (
    SharpTheoryPattern,
    PositionalTheoryPattern,
    SolidTheoryPattern
)
from .patterns.conversion import FlawlessConversionPattern
from .patterns.outro import DecisiveOutroPattern, DrawOutroPattern


class PatternCoordinator:
    """
    Coordinates pattern generation and audio synthesis.

    Manages:
    - Pattern registry (narrative name -> generator)
    - Audio synthesis (events -> audio buffer)
    - Timeline management (section-relative timestamps)
    """

    def __init__(self,
                 sample_rate: int,
                 config,
                 synth_engine,
                 rng: np.random.Generator):
        """
        Initialize pattern coordinator.

        Args:
            sample_rate: Audio sample rate
            config: SynthConfig instance (can be None for testing)
            synth_engine: SubtractiveSynth instance
            rng: NumPy random generator for reproducible randomness
        """
        self.sample_rate = sample_rate
        self.config = config
        self.rng = rng
        self.synthesizer = NoteSynthesizer(synth_engine, sample_rate)

        # Register all patterns
        self.patterns = self._build_pattern_registry()

    def _build_pattern_registry(self) -> Dict[str, Any]:
        """Build pattern registry mapping narratives to generators."""
        decisive_outro = DecisiveOutroPattern(self.rng)
        draw_outro = DrawOutroPattern(self.rng)

        return {
            'COMPLEX_STRUGGLE': MarkovChainPattern(self.rng),
            'KING_HUNT': KingHuntPattern(self.rng),
            'DESPERATE_DEFENSE': DesperateDefensePattern(self.rng),
            'TACTICAL_CHAOS': TacticalChaosPattern(self.rng),
            'CRUSHING_ATTACK': CrushingAttackPattern(self.rng),
            'SHARP_THEORY': SharpTheoryPattern(self.rng),
            'POSITIONAL_THEORY': PositionalTheoryPattern(self.rng),
            'SOLID_THEORY': SolidTheoryPattern(self.rng),
            'FLAWLESS_CONVERSION': FlawlessConversionPattern(self.rng),
            'DECISIVE_OUTRO': decisive_outro,
            'DECISIVE_ENDING': decisive_outro,  # Alias
            'DRAW_OUTRO': draw_outro,
            'DRAWN_ENDING': draw_outro,  # Alias
        }

    def register_pattern(self, name: str, pattern):
        """Register custom pattern generator."""
        self.patterns[name] = pattern

    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern names."""
        return list(self.patterns.keys())

    def generate_pattern(self,
                        narrative: str,
                        duration: float,
                        scale: List[float],
                        params: Dict[str, Any]) -> np.ndarray:
        """
        Generate complete audio pattern for narrative.

        Args:
            narrative: Narrative type (e.g., 'COMPLEX_STRUGGLE')
            duration: Pattern duration in seconds
            scale: Musical scale frequencies (Hz)
            params: Pattern parameters dict

        Returns:
            Audio buffer (numpy array, mono)

        Raises:
            ValueError: If narrative type is unknown
        """
        # Validate narrative
        if narrative not in self.patterns:
            raise ValueError(
                f"Unknown narrative: {narrative}. "
                f"Available patterns: {list(self.patterns.keys())}"
            )

        # Get pattern generator
        generator = self.patterns[narrative]

        # Generate note events
        events = generator.generate_events(duration, scale, params)

        # Allocate audio buffer
        total_samples = int(duration * self.sample_rate)
        buffer = AudioBuffer(total_samples)

        # Synthesize each event and mix into buffer
        mix_level = params.get('mix_level', 1.0)
        section_start_time = params.get('section_start_time', 0.0)

        # Apply pattern note level from config (same as original)
        pattern_note_level = self.config.LAYER_MIXING['pattern_note_level'] if self.config else 0.2

        # Check if any event requests decay curve (for outro patterns)
        decay_curve_type = None
        decay_amount = 0.0

        for event in events:
            # Render event to audio
            audio = self.synthesizer.synthesize(event)

            # Calculate buffer position (section-relative)
            # Subtract section_start_time because buffer is section-local
            start_sample = int((event.timestamp - section_start_time) * self.sample_rate)

            # Get mix level from event's extra_context or use default
            if event.extra_context and 'level' in event.extra_context:
                # Pattern explicitly set level (e.g., CRUSHING_ATTACK with voice_scale)
                gain = event.extra_context['level'] * mix_level
            elif event.extra_context and 'mix_level' in event.extra_context:
                # Pattern set mix_level (e.g., outro patterns)
                gain = event.extra_context['mix_level'] * mix_level
            else:
                # Default: velocity * pattern_note_level
                gain = event.velocity * pattern_note_level * mix_level
            buffer.add_audio(audio, start_sample, gain)

            # Check for decay curve request
            if event.extra_context and 'decay_curve' in event.extra_context:
                decay_curve_type = event.extra_context['decay_curve']
                decay_amount = event.extra_context.get('decay_amount', -2.5)

        # Apply decay curve if requested (for outro patterns)
        result = buffer.get_buffer()
        if decay_curve_type == 'exp':
            decay_curve = np.exp(np.linspace(0, decay_amount, len(result)))
            result *= decay_curve

        return result
