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

    def get_waveform(self, params: Dict[str, Any]) -> str:
        """
        Extract waveform from layer2_config (Spiegel-style: constant timbre).

        Args:
            params: Pattern parameters dict containing layer2_config

        Returns:
            Waveform name (saw, triangle, square, sine, pulse)
        """
        return params.get('layer2_config', {}).get('waveform', 'triangle')

    def get_articulation(self, params: Dict[str, Any]) -> str:
        """
        Extract articulation style from layer2_config.

        Args:
            params: Pattern parameters dict containing layer2_config

        Returns:
            Articulation style (staccato, legato, mixed)
        """
        return params.get('layer2_config', {}).get('articulation', 'mixed')

    def get_phrase_style(self, params: Dict[str, Any]) -> str:
        """
        Extract phrase style from layer2_config.

        Args:
            params: Pattern parameters dict containing layer2_config

        Returns:
            Phrase style (connected, detached, mixed)
        """
        return params.get('layer2_config', {}).get('phrase_style', 'mixed')

    def get_melodic_bias(self, params: Dict[str, Any]) -> str:
        """
        Extract melodic bias from layer2_config.

        Args:
            params: Pattern parameters dict containing layer2_config

        Returns:
            Melodic bias (ascending, descending, arch, neutral)
        """
        return params.get('layer2_config', {}).get('melodic_bias', 'neutral')

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

    def print_debug_summary(self, events: List, extra_stats: Dict[str, Any] = None) -> None:
        """
        Print debug summary of generated events.

        Args:
            events: List of NoteEvent objects
            extra_stats: Optional dict of additional stats to display (e.g., {'states': state_counts})
        """
        if not events:
            return

        # Count waveforms
        wave_counts = {}
        for e in events:
            wave_counts[e.waveform] = wave_counts.get(e.waveform, 0) + 1

        wave_str = ', '.join([f"{w}:{c}" for w, c in wave_counts.items()])

        # Get amp/filter envelope names (assume consistent across events)
        amp_env_name = events[0].amp_env_name if hasattr(events[0], 'amp_env_name') else 'unknown'
        filter_env_name = events[0].filter_env_name if hasattr(events[0], 'filter_env_name') else 'unknown'

        # Calculate averages
        avg_vel = sum(e.velocity for e in events) / len(events)
        avg_dur = sum(e.duration for e in events) / len(events)

        # Build output string
        output = (f"      {len(events)} events | wave: {wave_str} | "
                 f"amp: {amp_env_name} | filter: {filter_env_name} | "
                 f"vel: {avg_vel:.2f} | dur: {avg_dur:.3f}s")

        # Add extra stats if provided
        if extra_stats:
            for key, value in extra_stats.items():
                if isinstance(value, dict):
                    stat_str = ', '.join([f"{k}:{v}" for k, v in value.items()])
                    output += f" | {key}: {stat_str}"
                else:
                    output += f" | {key}: {value}"

        print(output)

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"{self.__class__.__name__}()"
