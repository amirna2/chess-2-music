"""
Gesture coordinator - manages archetype registry and moment dispatching.

Parallel to PatternCoordinator in synth_composer/patterns/.
Provides centralized registry of all gesture generators and dispatches
moment events to appropriate archetypes.
"""

import numpy as np
from typing import Dict, Any

from .base import GestureGenerator
from .archetype_configs import ARCHETYPES


class GestureCoordinator:
    """
    Coordinates gesture generation and archetype registry.

    This class serves as the primary interface for Layer 3b gesture generation.
    It maintains a registry of all available gesture archetypes and dispatches
    moment events to the appropriate generator.

    Usage:
        # Initialize coordinator with synth engine
        rng = np.random.default_rng(seed=42)
        synth_engine = SubtractiveSynth(sample_rate=88200, rng=rng)
        coordinator = GestureCoordinator(rng, synth_engine=synth_engine)

        # Generate gesture for a moment
        audio = coordinator.generate_gesture(
            'BLUNDER',
            moment_event={'event_type': 'BLUNDER', 'timestamp': 1.0, 'move_number': 5},
            section_context={'tension': 0.7, 'entropy': 0.5, 'scale': 'C_MAJOR'},
            sample_rate=88200
        )
    """

    def __init__(self, rng: np.random.Generator, synth_engine=None):
        """
        Initialize coordinator with archetype registry.

        Args:
            rng: NumPy random generator for reproducible randomness
            synth_engine: SubtractiveSynth instance (optional)
                         If None, generators will be created but cannot synthesize.
                         Must be provided before calling generate_gesture().

        Note:
            The synth_engine parameter allows lazy initialization of synthesis
            capabilities. This is useful for testing generator logic without
            requiring a full synthesis engine.
        """
        self.rng = rng
        self.synth_engine = synth_engine
        self.gestures = self._build_gesture_registry()

    def _build_gesture_registry(self) -> Dict[str, GestureGenerator]:
        """
        Build gesture registry mapping archetype names to generators.

        Creates a GestureGenerator instance for each archetype defined in
        ARCHETYPES configuration. All generators share the same RNG and
        synth_engine for consistency.

        Returns:
            Dict mapping archetype name → GestureGenerator instance
        """
        registry = {}

        for archetype_name, archetype_config in ARCHETYPES.items():
            # Create generator with shared synth_engine
            # Generators will validate synth_engine presence when generate_gesture() called
            registry[archetype_name] = GestureGenerator(
                archetype_config,
                self.rng,
                synth_engine=self.synth_engine
            )

        return registry

    def get_available_archetypes(self) -> list:
        """
        Get list of available archetype names.

        Returns:
            List of archetype name strings (e.g., ['BLUNDER', 'BRILLIANT', ...])
        """
        return list(self.gestures.keys())

    def generate_gesture(self,
                        archetype_name: str,
                        moment_event: Dict[str, Any],
                        section_context: Dict[str, Any],
                        sample_rate: int) -> np.ndarray:
        """
        Generate gesture audio for a moment event.

        This is the primary public interface for gesture generation.
        Routes the request to the appropriate archetype generator.

        Args:
            archetype_name: Archetype name (e.g., 'BLUNDER', 'BRILLIANT', 'TIME_PRESSURE')
            moment_event: Moment metadata dict with keys:
                         - event_type: str (should match archetype_name)
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
            Mono audio buffer (numpy array, float32, normalized)

        Raises:
            ValueError: If archetype_name is unknown
            ValueError: If synth_engine not provided (either in __init__ or via set_synth_engine)
            ValueError: If section_context missing required keys
            ValueError: If sample_rate invalid

        Example:
            audio = coordinator.generate_gesture(
                'BLUNDER',
                moment_event={
                    'event_type': 'BLUNDER',
                    'timestamp': 12.5,
                    'move_number': 8,
                    'quality': 'poor',
                    'eval_drop': 3.5
                },
                section_context={
                    'tension': 0.8,
                    'entropy': 0.6,
                    'scale': 'D_DORIAN',
                    'key': 'D'
                },
                sample_rate=88200
            )
        """
        # Validate archetype exists
        if archetype_name not in self.gestures:
            available = ', '.join(self.get_available_archetypes())
            raise ValueError(
                f"Unknown archetype: '{archetype_name}'. "
                f"Available archetypes: {available}"
            )

        # Get generator and delegate
        generator = self.gestures[archetype_name]
        return generator.generate_gesture(moment_event, section_context, sample_rate)

    def set_synth_engine(self, synth_engine):
        """
        Set or update synthesis engine for all generators.

        This allows lazy initialization of synthesis capabilities after
        coordinator creation. Useful when synth_engine initialization is
        expensive or deferred.

        Args:
            synth_engine: SubtractiveSynth instance

        Example:
            # Create coordinator without synthesis
            coordinator = GestureCoordinator(rng)

            # Later, add synthesis capability
            synth = SubtractiveSynth(sample_rate=88200, rng=rng)
            coordinator.set_synth_engine(synth)

            # Now can generate
            audio = coordinator.generate_gesture(...)
        """
        self.synth_engine = synth_engine

        # Update all existing generators
        for generator in self.gestures.values():
            from .synthesizer import GestureSynthesizer
            generator.synthesizer = GestureSynthesizer(synth_engine)

    def get_archetype_config(self, archetype_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific archetype.

        Useful for introspection, debugging, or UI display of archetype parameters.

        Args:
            archetype_name: Archetype name

        Returns:
            Configuration dict for archetype

        Raises:
            ValueError: If archetype_name unknown
        """
        if archetype_name not in ARCHETYPES:
            available = ', '.join(self.get_available_archetypes())
            raise ValueError(
                f"Unknown archetype: '{archetype_name}'. "
                f"Available archetypes: {available}"
            )

        return ARCHETYPES[archetype_name]

    def compute_archetype_duration(self,
                                   archetype_name: str,
                                   section_context: Dict[str, Any]) -> float:
        """
        Compute expected duration for an archetype in given context.

        Useful for timeline planning and audio buffer allocation.

        Args:
            archetype_name: Archetype name
            section_context: Section-level parameters (tension, entropy)

        Returns:
            Expected duration in seconds (clamped to [0.5, 10.0])

        Raises:
            ValueError: If archetype_name unknown
        """
        if archetype_name not in self.gestures:
            available = ', '.join(self.get_available_archetypes())
            raise ValueError(
                f"Unknown archetype: '{archetype_name}'. "
                f"Available archetypes: {available}"
            )

        generator = self.gestures[archetype_name]
        return generator._compute_duration(section_context)
