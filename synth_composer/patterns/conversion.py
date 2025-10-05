"""Conversion pattern for endgame phase."""

import numpy as np
from .base import PatternGenerator
from ..core.note_event import NoteEvent
from ..core.timing_engine import TimingEngine
from synth_config import get_envelope, get_filter_envelope


class FlawlessConversionPattern(PatternGenerator):
    """
    FLAWLESS_CONVERSION: State machine for precise endgame technique
    - States: ADVANCE (build pressure), CONSOLIDATE (hold position), BREAKTHROUGH (push advantage)
    - Methodical, controlled, inevitable like Fischer's technique
    """

    # State constants
    STATE_ADVANCE = 0
    STATE_CONSOLIDATE = 1
    STATE_BREAKTHROUGH = 2

    def generate_events(self, duration, scale, params):
        """Generate flawless conversion pattern events."""
        total_samples = int(duration * params['sample_rate'])
        timing = TimingEngine(params['sample_rate'])
        events = []

        base_note_dur = params['note_duration'] * 1.3
        current_state = self.STATE_ADVANCE
        current_note_idx = 0
        advance_progress = 0  # Track how far we've advanced

        final_filter = params['filter']
        filter_env_amount = params['filter_env']
        final_resonance = params['resonance']
        config = params['config']

        while not timing.is_finished(total_samples):
            progress = timing.current_sample / total_samples

            # State transitions
            if current_state == self.STATE_ADVANCE:
                advance_progress += 1
                if advance_progress > 5 and self.rng.random() < 0.3:
                    current_state = self.STATE_CONSOLIDATE
                    advance_progress = 0
                elif progress > 0.7 and self.rng.random() < 0.2:
                    current_state = self.STATE_BREAKTHROUGH
            elif current_state == self.STATE_CONSOLIDATE:
                if self.rng.random() < 0.5:
                    current_state = self.STATE_ADVANCE
                elif progress > 0.7 and self.rng.random() < 0.3:
                    current_state = self.STATE_BREAKTHROUGH
            elif current_state == self.STATE_BREAKTHROUGH:
                if self.rng.random() < 0.1:
                    current_state = self.STATE_ADVANCE

            # Note selection by state
            if current_state == self.STATE_ADVANCE:
                # Build pressure - favor dominant and upper notes, but randomly
                weights = [0.1, 0.05, 0.15, 0.1, 0.25, 0.15, 0.1, 0.1]  # Favor 5th (index 4)
                current_note_idx = self.rng.choice(range(len(scale)), p=weights)
            elif current_state == self.STATE_CONSOLIDATE:
                # Hold stable harmonic notes - tonic, third, fifth
                current_note_idx = self.rng.choice([0, 2, 4], p=[0.4, 0.3, 0.3])
            elif current_state == self.STATE_BREAKTHROUGH:
                # Decisive moves - octave relationships
                current_note_idx = self.rng.choice([0, 4, 7], p=[0.3, 0.4, 0.3])  # Tonic, fifth, seventh

            note_freq = scale[current_note_idx]
            duration_float = base_note_dur * self.rng.uniform(0.96, 1.04)
            velocity = 0.55 + progress * 0.15

            note_samples = int(duration_float * params['sample_rate'])
            note_samples = min(note_samples, total_samples - timing.current_sample)

            if note_samples > 0:
                duration_quantized = note_samples / params['sample_rate']

                event = NoteEvent(
                    freq=note_freq,
                    duration=duration_quantized,
                    timestamp=timing.current_time,
                    velocity=velocity,
                    waveform='triangle',
                    filter_base=final_filter * (0.7 + progress * 0.6),
                    filter_env_amount=filter_env_amount * 0.7,
                    resonance=final_resonance * 0.6,
                    amp_env=get_envelope('sustained', config),
                    filter_env=get_filter_envelope('closing', config),
                    amp_env_name='sustained',
                    filter_env_name='closing',
                    extra_context={'mix_level': config.LAYER_MIXING['pattern_note_level'] * velocity * 0.85}
                )
                events.append(event)

            pause_samples = int(base_note_dur * 0.18 * params['sample_rate'])
            timing.advance(note_samples + pause_samples)

        # Debug output
        self.print_debug_summary(events)

        return events
