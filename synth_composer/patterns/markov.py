"""
MarkovChainPattern - COMPLEX_STRUGGLE narrative pattern generator.

Implements Markov chain random walk with gravity toward tonic.
"""

import numpy as np
from typing import List, Dict, Any
from .base import PatternGenerator
from ..core.note_event import NoteEvent
from ..core.timing_engine import TimingEngine

try:
    from synth_config import get_envelope, get_filter_envelope
except ImportError:
    def get_envelope(name, config):
        presets = {
            'soft': (0.05, 0.2, 0.8, 0.3),
            'stab': (0.01, 0.1, 0.3, 0.1),
            'pluck': (0.001, 0.05, 0.0, 0.05),
        }
        return presets.get(name, (0.01, 0.1, 0.7, 0.2))

    def get_filter_envelope(name, config):
        presets = {
            'gentle': (0.1, 0.3, 0.4, 0.3),
            'sweep': (0.01, 0.15, 0.3, 0.2),
            'closing': (0.3, 0.2, 0.3, 0.5),
        }
        return presets.get(name, (0.01, 0.15, 0.3, 0.2))


class MarkovChainPattern(PatternGenerator):
    """
    COMPLEX_STRUGGLE: Markov chain with gravitational pull to tonic.

    Musical characteristics:
    - Probabilistic note selection
    - Higher probability of returning to tonic (cautious play)
    - Irregular durations based on scale position
    - Evolving hesitation (pauses increase with tension)
    - Darker filter on tonic, brighter when exploring scale
    """

    def __init__(self, rng: np.random.Generator):
        super().__init__(rng)
        self.transition_matrix = self._build_transition_matrix()

    def generate_events(self,
                       duration: float,
                       scale: List[float],
                       params: Dict[str, Any]) -> List[NoteEvent]:
        """Generate note events using Markov chain random walk."""
        events = []

        timing = TimingEngine(
            sample_rate=params['sample_rate'],
            section_start_time=params.get('section_start_time', 0.0)
        )

        total_samples = int(duration * params['sample_rate'])
        current_note_idx = 0  # Start on tonic
        base_note_dur = params['note_duration'] * 1.5

        while not timing.is_finished(total_samples):
            # CRITICAL: progress calculated at loop start (match original line 279)
            # Used for pause calculation later in this iteration
            progress = timing.current_sample / total_samples

            note_freq = scale[current_note_idx] if current_note_idx < len(scale) else scale[0]

            # Calculate duration (longer on tonic = thinking)
            note_dur = self._calculate_note_duration(
                current_note_idx,
                base_note_dur,
                params['tension']
            )

            # Quantize to samples (CRITICAL: match original behavior exactly)
            note_samples = int(note_dur * params['sample_rate'])
            note_samples = min(note_samples, timing.remaining_samples(total_samples))

            if note_samples > 0:
                # Convert back to seconds (quantized duration)
                note_dur_quantized = note_samples / params['sample_rate']

                filter_mult = 0.7 + (current_note_idx / len(scale)) * 0.8
                velocity = self.rng.uniform(0.6, 1.0)

                events.append(NoteEvent(
                    freq=note_freq,
                    duration=note_dur_quantized,  # Use quantized duration
                    timestamp=timing.get_timestamp(),
                    velocity=velocity,
                    waveform='pulse',
                    filter_base=params['filter'] * filter_mult,
                    filter_env_amount=params['filter_env'] * self.rng.uniform(0.4, 0.8),
                    resonance=params['resonance'] * self.rng.uniform(0.7, 0.9),
                    amp_env=get_envelope('soft', params['config']),
                    filter_env=get_filter_envelope('gentle', params['config']),
                    amp_env_name='soft',
                    filter_env_name='gentle',
                    extra_context={
                        'state': 'markov_walk',
                        'scale_degree': current_note_idx,
                        'filter_mult': filter_mult,
                        'velocity': velocity
                    }
                ))

            # Advance timeline (CRITICAL: outside if block, match original line 321)
            # Original does: current_sample += note_samples + pause_samples
            timing.advance(note_samples)

            # Add pause (more hesitation as tension builds)
            # Uses progress from START of this iteration (matches original)
            pause_dur = self._calculate_pause_duration(
                base_note_dur,
                progress,
                params['tension']
            )
            timing.add_pause(pause_dur)

            # Markov transition
            current_note_idx = self._next_note(current_note_idx)

        # Debug output
        self.print_debug_summary(events)

        return events

    def _build_transition_matrix(self) -> np.ndarray:
        """Build Markov transition matrix with gravity to tonic."""
        matrix = np.array([
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0],
            [0.4, 0.2, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0, 0.0],
            [0.2, 0.0, 0.1, 0.2, 0.1, 0.2, 0.2, 0.0],
            [0.1, 0.0, 0.0, 0.1, 0.3, 0.2, 0.2, 0.1],
            [0.2, 0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 0.2],
            [0.3, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2],
        ], dtype=np.float64)

        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        return matrix

    def _calculate_note_duration(self, note_idx: int, base_duration: float, tension: float) -> float:
        """Calculate note duration based on scale position."""
        if note_idx == 0:  # Tonic = thinking
            return base_duration * self.rng.uniform(1.2, 2.5 + tension)
        else:
            return base_duration * self.rng.uniform(0.6, 1.2)

    def _calculate_pause_duration(self, base_duration: float, progress: float, tension: float) -> float:
        """Calculate pause duration between notes."""
        return base_duration * self.rng.uniform(0.2, 0.8 + progress * tension)

    def _next_note(self, current_idx: int) -> int:
        """Choose next note using Markov transition probabilities."""
        if current_idx < len(self.transition_matrix):
            probabilities = self.transition_matrix[current_idx]
            return self.rng.choice(len(probabilities), p=probabilities)
        else:
            return 0

    def __repr__(self) -> str:
        return "MarkovChainPattern(COMPLEX_STRUGGLE)"
