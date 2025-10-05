"""Theory patterns for chess openings."""

import numpy as np
from .base import PatternGenerator
from ..core.note_event import NoteEvent
from ..core.timing_engine import TimingEngine
from synth_config import get_envelope, get_filter_envelope


class SharpTheoryPattern(PatternGenerator):
    """
    SHARP_THEORY: State machine for aggressive tactical openings
    - States: ATTACK (rapid ascent), DART (quick jumps), SETTLE (brief pause)
    - Fast, energetic, unpredictable like Sicilian tactics
    """

    # State constants
    STATE_ATTACK = 0
    STATE_DART = 1
    STATE_SETTLE = 2

    def generate_events(self, duration, scale, params):
        """Generate sharp theory pattern events."""
        total_samples = int(duration * params['sample_rate'])
        timing = TimingEngine(params['sample_rate'])
        events = []

        base_note_dur = params['note_duration'] * 0.4
        current_state = self.STATE_ATTACK
        current_note_idx = 0

        tension = params['tension']
        final_filter = params['filter']
        filter_env_amount = params['filter_env']
        final_resonance = params['resonance']
        config = params['config']

        while not timing.is_finished(total_samples):
            progress = timing.current_sample / total_samples

            # State transitions
            if current_state == self.STATE_ATTACK:
                if self.rng.random() < 0.2:
                    current_state = self.STATE_DART
                elif self.rng.random() < 0.1:
                    current_state = self.STATE_SETTLE
            elif current_state == self.STATE_DART:
                if self.rng.random() < 0.6:
                    current_state = self.STATE_ATTACK
                elif self.rng.random() < 0.2:
                    current_state = self.STATE_SETTLE
            elif current_state == self.STATE_SETTLE:
                if self.rng.random() < 0.8:
                    current_state = self.STATE_ATTACK

            # Note selection by state
            if current_state == self.STATE_ATTACK:
                # Aggressive attacks - favor upper register, random leaps
                if self.rng.random() < 0.5:
                    current_note_idx = self.rng.choice([4, 5, 6, 7])  # Upper half
                else:
                    current_note_idx = self.rng.choice([2, 3, 4])  # Middle
            elif current_state == self.STATE_DART:
                # Random tactical jumps anywhere
                current_note_idx = self.rng.integers(0, len(scale))
            elif current_state == self.STATE_SETTLE:
                # Gravitate to stable notes
                if self.rng.random() < 0.6:
                    current_note_idx = 0  # Tonic
                else:
                    current_note_idx = 4  # Dominant

            note_freq = scale[current_note_idx]
            duration_float = base_note_dur * self.rng.uniform(0.6, 1.0)
            velocity = self.rng.uniform(0.75, 1.0)

            note_samples = int(duration_float * params['sample_rate'])
            note_samples = min(note_samples, total_samples - timing.current_sample)

            if note_samples > 0:
                duration_quantized = note_samples / params['sample_rate']

                event = NoteEvent(
                    freq=note_freq,
                    duration=duration_quantized,
                    timestamp=timing.current_time,
                    velocity=velocity,
                    waveform='saw',
                    filter_base=final_filter * (1.5 + tension * 0.5),
                    filter_env_amount=filter_env_amount,
                    resonance=final_resonance * 0.8,
                    amp_env=get_envelope('pluck', config),
                    filter_env=get_filter_envelope('sweep', config),
                    amp_env_name='pluck',
                    filter_env_name='sweep',
                    extra_context={'mix_level': config.LAYER_MIXING['pattern_note_level'] * velocity}
                )
                events.append(event)

            pause_samples = int(base_note_dur * 0.05 * params['sample_rate'])
            timing.advance(note_samples + pause_samples)

        # Debug output
        if events:
            self.print_debug_summary(events, extra_stats={
                'states': {'attack': sum(1 for _ in events if 'attack' in str(_.timestamp)),
                          'dart': sum(1 for _ in events if 'dart' in str(_.timestamp)),
                          'settle': sum(1 for _ in events if 'settle' in str(_.timestamp))}
            })

        return events


class PositionalTheoryPattern(PatternGenerator):
    """
    POSITIONAL_THEORY: Markov chain for strategic maneuvering
    - Weighted toward stable harmonic intervals (tonic, third, fifth)
    - Deliberate, controlled transitions
    - Patient, strategic like French/English openings
    - Calm, contemplative, with ample breathing room
    """

    def generate_events(self, duration, scale, params):
        """Generate positional theory pattern events."""
        total_samples = int(duration * params['sample_rate'])
        timing = TimingEngine(params['sample_rate'])
        events = []

        base_note_dur = params['note_duration'] * 1.2  # Slower notes for calm, patient play

        final_filter = params['filter']
        filter_env_amount = params['filter_env']
        final_resonance = params['resonance']
        config = params['config']

        # Markov chain: favor stable harmonic intervals
        transition_matrix = np.array([
            [0.3, 0.1, 0.3, 0.1, 0.2, 0.0, 0.0, 0.0],  # From tonic: to tonic/third/fifth
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0],  # From 1
            [0.4, 0.1, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0],  # From 2 (third): to tonic/fifth
            [0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0],  # From 3
            [0.3, 0.0, 0.2, 0.1, 0.2, 0.1, 0.1, 0.0],  # From 4 (fifth): to tonic/third
            [0.1, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0],  # From 5
            [0.1, 0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1],  # From 6
            [0.2, 0.0, 0.1, 0.0, 0.2, 0.2, 0.2, 0.1],  # From 7: return toward tonic/fifth
        ])

        # Normalize
        for i in range(len(transition_matrix)):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum

        current_note_idx = 0  # Start on tonic

        while not timing.is_finished(total_samples):
            note_freq = scale[current_note_idx]
            duration_float = base_note_dur * self.rng.uniform(0.8, 1.0)  # Shorter notes, more space
            velocity = 0.35 + self.rng.uniform(-0.1, 0.15)  # Much quieter (25-50% vs 60-80%)

            note_samples = int(duration_float * params['sample_rate'])
            note_samples = min(note_samples, total_samples - timing.current_sample)

            if note_samples > 0:
                duration_quantized = note_samples / params['sample_rate']

                event = NoteEvent(
                    freq=note_freq,
                    duration=duration_quantized,
                    timestamp=timing.current_time,
                    velocity=velocity,
                    waveform='triangle',  # Warmer, less nasal than pulse
                    filter_base=final_filter * 0.7,  # Darker, less bright
                    filter_env_amount=filter_env_amount * 0.5,  # Less filter movement
                    resonance=final_resonance * 0.5,  # Softer resonance
                    amp_env=get_envelope('soft', config),
                    filter_env=get_filter_envelope('gentle', config),
                    amp_env_name='soft',
                    filter_env_name='gentle',
                    extra_context={'mix_level': config.LAYER_MIXING['pattern_note_level'] * velocity}
                )
                events.append(event)

            # Much longer pause for contemplative, strategic feel
            pause_samples = int(base_note_dur * 0.6 * params['sample_rate'])  # 60% pause vs 15%
            timing.advance(note_samples + pause_samples)

            # Markov transition
            if current_note_idx < len(transition_matrix):
                probabilities = transition_matrix[current_note_idx]
                current_note_idx = self.rng.choice(len(probabilities), p=probabilities)
            else:
                current_note_idx = 0

        # Debug output
        if events:
            self.print_debug_summary(events)

        return events


class SolidTheoryPattern(PatternGenerator):
    """
    SOLID_THEORY: Grounded bass patterns with stable rhythms
    - Safe, solid character for Queen's Gambit Declined, Slav, solid openings
    - Lower register emphasis
    - Predictable, repetitive patterns (building blocks)
    - Steady rhythm with minimal variation
    """

    def generate_events(self, duration, scale, params):
        """Generate solid theory pattern events."""
        total_samples = int(duration * params['sample_rate'])
        timing = TimingEngine(params['sample_rate'])
        events = []

        base_note_dur = params['note_duration'] * 1.2  # Slower, grounded

        final_filter = params['filter']
        filter_env_amount = params['filter_env']
        final_resonance = params['resonance']
        config = params['config']

        # Build a simple repeating pattern (tonic, fifth, third, fifth)
        pattern_sequence = [0, 4, 2, 4]  # Scale degrees
        pattern_idx = 0

        while not timing.is_finished(total_samples):
            progress = timing.current_sample / total_samples

            # Get note from pattern (lower octave for grounded feel)
            scale_idx = pattern_sequence[pattern_idx % len(pattern_sequence)]
            note_freq = scale[scale_idx] * 0.75  # Lower by perfect fourth

            # Duration: very consistent for stability
            duration_float = base_note_dur * self.rng.uniform(0.95, 1.05)

            # Generate note
            note_samples = int(duration_float * params['sample_rate'])
            note_samples = min(note_samples, total_samples - timing.current_sample)

            if note_samples > 0:
                # Filter: darker, grounded
                filter_mult = 0.6 + progress * 0.2  # Stays dark

                # Velocity: very stable
                velocity = 0.75 + self.rng.uniform(-0.05, 0.05)

                duration_quantized = note_samples / params['sample_rate']

                event = NoteEvent(
                    freq=note_freq,
                    duration=duration_quantized,
                    timestamp=timing.current_time,
                    velocity=velocity,
                    waveform='sine',  # Pure, smooth, solid bass
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * 0.6,
                    resonance=final_resonance * 0.5,  # Low resonance for solid feel
                    amp_env=get_envelope('soft', config),
                    filter_env=get_filter_envelope('gentle', config),
                    amp_env_name='soft',
                    filter_env_name='gentle',
                    extra_context={'mix_level': config.LAYER_MIXING['pattern_note_level'] * velocity}
                )
                events.append(event)

            # Consistent pause
            pause_samples = int(base_note_dur * 0.2 * params['sample_rate'])
            timing.advance(note_samples + pause_samples)

            # Advance pattern
            pattern_idx += 1

        # Debug output
        if events:
            self.print_debug_summary(events)

        return events
