"""
PatternGenerator base class.

Abstract base for all narrative pattern generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class PatternGenerator(ABC):
    """
    Abstract base class for pattern generators.

    All pattern generators follow same interface:
    - Accept RNG for reproducible randomness
    - generate_events() returns list of NoteEvent objects
    - No audio generation (handled by coordinator)
    """

    def __init__(self, rng: np.random.Generator):
        """
        Initialize pattern generator.

        Args:
            rng: NumPy random generator for reproducible randomness
        """
        self.rng = rng

    @abstractmethod
    def generate_events(self,
                       duration: float,
                       scale: List[float],
                       params: Dict[str, Any]) -> List:
        """
        Generate note events for this pattern.

        Args:
            duration: Pattern duration in seconds
            scale: Musical scale frequencies (Hz)
            params: Pattern parameters dict containing:
                - sample_rate: Audio sample rate
                - section_start_time: Global timeline offset
                - filter: Base filter frequency
                - filter_env: Filter envelope amount
                - resonance: Filter resonance
                - note_duration: Base note duration
                - tension: Section tension (0-1)
                - config: SynthConfig object
                - mix_level: Audio mix level

        Returns:
            List of NoteEvent objects with timestamps
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"{self.__class__.__name__}()"
