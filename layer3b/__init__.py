"""
Layer 3b: Unified Moment Gesture System

This module provides algorithmic generation of emotional gesture moments that
punctuate the musical narrative in chess-to-music compositions.

Architecture:
- All gesture archetypes share the same synthesis pipeline
- Archetypes differ only in configuration parameters, not code
- Mirrors the pattern system architecture used in synth_composer/patterns/

Main components:
- GestureCoordinator: Primary interface for gesture generation
- GestureGenerator: Unified generator for all archetypes
- GestureSynthesizer: Audio synthesis wrapper around SubtractiveSynth

Usage:
    from layer3b import GestureCoordinator
    from synth_engine import SubtractiveSynth
    import numpy as np

    # Initialize
    rng = np.random.default_rng(seed=42)
    synth = SubtractiveSynth(sample_rate=88200, rng=rng)
    coordinator = GestureCoordinator(rng, synth_engine=synth)

    # Generate gesture
    audio = coordinator.generate_gesture(
        'BLUNDER',
        moment_event={'event_type': 'BLUNDER', 'timestamp': 1.0, 'move_number': 5},
        section_context={'tension': 0.7, 'entropy': 0.5, 'scale': 'C_MAJOR', 'key': 'C'},
        sample_rate=88200
    )
"""

from .base import GestureGenerator
from .coordinator import GestureCoordinator
from .synthesizer import GestureSynthesizer
from .archetype_configs import ARCHETYPES

__all__ = [
    'GestureGenerator',
    'GestureCoordinator',
    'GestureSynthesizer',
    'ARCHETYPES',
]
