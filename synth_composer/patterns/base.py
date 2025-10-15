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

    def get_articulation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get full articulation configuration for current layer2 config.

        Returns complete articulation settings including envelope choices,
        timing multipliers, and overlap amount.

        Args:
            params: Pattern parameters dict containing:
                - layer2_config: Dict with 'articulation' key
                - config: SynthConfig object with ARTICULATION_PARAMS

        Returns:
            Dict containing:
                - amp_envelopes: List[str] - envelope names to choose from
                - filter_envelopes: List[str] - filter envelope names
                - note_duration_mult: float - duration multiplier
                - gap_mult: float - gap between notes multiplier
                - overlap: float - note overlap amount (0.0-1.0)
        """
        articulation = self.get_articulation(params)
        return params['config'].ARTICULATION_PARAMS[articulation]

    def get_articulation_multipliers(
            self,
            params: Dict[str, Any]) -> tuple[float, float]:
        """
        Get timing multipliers for current articulation.

        Convenience method that extracts just the timing multipliers
        from full articulation configuration.

        Args:
            params: Pattern parameters dict

        Returns:
            Tuple of (note_duration_mult, gap_mult)
            - note_duration_mult: Multiplier for note durations
            - gap_mult: Multiplier for gaps between notes
        """
        art_params = self.get_articulation_params(params)
        return (
            art_params['note_duration_mult'],
            art_params['gap_mult']
        )

    def select_envelope(
            self,
            params: Dict[str, Any],
            envelope_type: str = 'amp') -> str:
        """
        Select appropriate envelope based on articulation.

        Randomly selects from the envelope pool defined by the
        current articulation style.

        Args:
            params: Pattern parameters dict
            envelope_type: 'amp' or 'filter'

        Returns:
            Envelope name (str)
        """
        art_params = self.get_articulation_params(params)

        if envelope_type == 'amp':
            choices = art_params['amp_envelopes']
        else:  # filter
            choices = art_params['filter_envelopes']

        return self.rng.choice(choices)

    def calculate_advance_with_overlap(
            self,
            note_samples: int,
            params: Dict[str, Any]) -> int:
        """
        Calculate timeline advance amount considering note overlap.

        For legato articulation, notes should overlap (blend together).
        For staccato, notes are fully separated (no overlap).

        Args:
            note_samples: Full duration of the note in samples
            params: Pattern parameters dict

        Returns:
            Number of samples to advance timeline
            - If overlap=0.0 (staccato): returns note_samples (full separation)
            - If overlap=0.3 (legato): returns 70% of note_samples (30% overlap)
        """
        art_params = self.get_articulation_params(params)
        overlap = art_params['overlap']

        # Advance by (1 - overlap) of the note duration
        # overlap=0.0 → advance 100% (no overlap)
        # overlap=0.3 → advance 70% (30% overlap with next note)
        # overlap=1.0 → advance 0% (complete overlap, notes stack)
        advance_samples = int(note_samples * (1.0 - overlap))

        return max(1, advance_samples)  # At least 1 sample to prevent infinite loop

    def apply_melodic_bias(
            self,
            current_idx: int,
            scale_len: int,
            progress: float,
            params: Dict[str, Any]) -> int:
        """
        Apply melodic bias to note selection.

        Biases note selection toward ascending, descending, or arch
        contours based on layer2_config melodic_bias setting.

        Research-based: Arch contour is most common in melodies,
        followed by descending, ascending, and concave shapes.

        Args:
            current_idx: Current scale degree index (0-7)
            scale_len: Length of scale (usually 8)
            progress: Progress through pattern (0.0-1.0)
            params: Pattern parameters dict

        Returns:
            Adjusted scale degree index (0 to scale_len-1)

        Note:
            This is a probabilistic nudge (30% probability),
            not a hard constraint. Pattern's own logic still dominates.
        """
        bias = self.get_melodic_bias(params)

        # Only apply bias 30% of the time (Spiegel-style: gentle influence)
        if self.rng.random() > 0.3:
            return current_idx

        if bias == 'ascending':
            # Bias upward - 60% chance to increase index
            if current_idx < scale_len - 1 and self.rng.random() < 0.6:
                return min(current_idx + 1, scale_len - 1)

        elif bias == 'descending':
            # Bias downward - 60% chance to decrease index
            if current_idx > 0 and self.rng.random() < 0.6:
                return max(current_idx - 1, 0)

        elif bias == 'arch':
            # Rise in first half, fall in second half
            if progress < 0.5:
                # First half: bias upward
                if current_idx < scale_len - 1 and self.rng.random() < 0.6:
                    return min(current_idx + 1, scale_len - 1)
            else:
                # Second half: bias downward
                if current_idx > 0 and self.rng.random() < 0.6:
                    return max(current_idx - 1, 0)

        # 'neutral' or no bias applied
        return current_idx

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
