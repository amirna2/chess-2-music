"""Outro patterns for game endings."""

import numpy as np
from .base import PatternGenerator
from ..core.note_event import NoteEvent
from ..core.timing_engine import TimingEngine
from synth_config import get_envelope, get_filter_envelope


class DecisiveOutroPattern(PatternGenerator):
    """
    DECISIVE_ENDING: Strong resolution for decisive games (1-0 or 0-1)
    - Clear, definitive phrases
    - Resolves to tonic with authority
    - Descending/ascending gesture based on overall narrative
    """

    def generate_events(self, duration, scale, params):
        """Generate decisive outro pattern events."""
        # Get overall_narrative from params (if available)
        overall_narrative = params.get('overall_narrative', '')

        # Determine direction based on overall narrative
        is_defeat = 'DEFEAT' in overall_narrative

        # Create final resolving phrase
        phrase_notes = []
        if is_defeat:
            # Descending resolution - somber but resolved
            phrase_notes = [7, 5, 4, 2, 0]  # 7th -> 5th -> 4th -> 2nd -> tonic
        else:
            # Ascending resolution - triumphant
            phrase_notes = [0, 2, 4, 5, 7, 0]  # tonic -> ... -> tonic (octave)

        final_filter = params['filter']
        filter_env_amount = params['filter_env']
        final_resonance = params['resonance']
        config = params['config']
        sample_rate = params['sample_rate']

        # Get articulation multipliers from config
        note_mult, pause_mult = self.get_articulation_multipliers(params)

        # Play the phrase
        timing = TimingEngine(sample_rate)
        events = []
        base_note_dur = duration / (len(phrase_notes) + 1)  # Leave space at end

        for i, note_idx in enumerate(phrase_notes):
            if note_idx >= len(scale):
                note_idx = 0

            # Apply melodic bias
            note_idx = self.apply_melodic_bias(note_idx, len(scale), i / len(phrase_notes), params)

            note_freq = scale[note_idx]

            # Fade out over the phrase
            progress = i / len(phrase_notes)
            velocity = 0.7 * (1.0 - progress * 0.5)  # Gentle fadeout

            note_dur = base_note_dur * note_mult
            note_samples = int(note_dur * sample_rate)
            if note_samples > 0:
                duration_quantized = note_samples / sample_rate

                # Select envelopes based on articulation
                amp_env_name = self.select_envelope(params, 'amp')
                filter_env_name = self.select_envelope(params, 'filter')

                event = NoteEvent(
                    freq=note_freq,
                    duration=duration_quantized,
                    timestamp=timing.current_time,
                    velocity=velocity,
                    waveform=self.get_waveform(params),
                    filter_base=final_filter * (1.0 - progress * 0.3),  # Close filter
                    filter_env_amount=filter_env_amount * 0.5,
                    resonance=final_resonance * 0.7,
                    amp_env=get_envelope(amp_env_name, config),
                    filter_env=get_filter_envelope(filter_env_name, config),
                    amp_env_name=amp_env_name,
                    filter_env_name=filter_env_name,
                    extra_context={
                        'mix_level': velocity,
                        'decay_curve': 'exp',  # Signal that we need exponential decay
                        'decay_amount': -2.5
                    }
                )
                events.append(event)

            # Advance with overlap for smooth outro
            advance_samples = self.calculate_advance_with_overlap(note_samples, params)
            timing.advance(advance_samples)

        # Debug output
        if events:
            direction = 'descending' if is_defeat else 'ascending'
            self.print_debug_summary(events, extra_stats={'direction': direction})

        return events


class DrawOutroPattern(PatternGenerator):
    """
    DRAWN_ENDING: Balanced, unresolved ending for drawn games (1/2-1/2)
    - Circular motion
    - Returns to tonic but without strong resolution
    - Peaceful but incomplete feeling
    """

    def generate_events(self, duration, scale, params):
        """Generate draw outro pattern events."""
        # Create circular phrase - goes around and returns without strong cadence
        # Use perfect fourth intervals for stability without finality
        phrase_notes = [0, 3, 0, 3, 0]  # Tonic <-> fourth (peaceful rocking)

        final_filter = params['filter']
        filter_env_amount = params['filter_env']
        final_resonance = params['resonance']
        config = params['config']
        sample_rate = params['sample_rate']

        # Get articulation multipliers from config
        note_mult, pause_mult = self.get_articulation_multipliers(params)

        timing = TimingEngine(sample_rate)
        events = []
        base_note_dur = duration / (len(phrase_notes) + 2)  # Extra space

        for i, note_idx in enumerate(phrase_notes):
            if note_idx >= len(scale):
                note_idx = 0

            # Apply melodic bias
            note_idx = self.apply_melodic_bias(note_idx, len(scale), i / len(phrase_notes), params)

            note_freq = scale[note_idx]

            # Gradual fadeout
            progress = i / len(phrase_notes)
            velocity = 0.6 * (1.0 - progress * 0.6)  # Faster fadeout

            note_dur = base_note_dur * note_mult
            note_samples = int(note_dur * sample_rate)
            if note_samples > 0:
                duration_quantized = note_samples / sample_rate

                # Select envelopes based on articulation
                amp_env_name = self.select_envelope(params, 'amp')
                filter_env_name = self.select_envelope(params, 'filter')

                event = NoteEvent(
                    freq=note_freq,
                    duration=duration_quantized,
                    timestamp=timing.current_time,
                    velocity=velocity,
                    waveform=self.get_waveform(params),
                    filter_base=final_filter * (1.0 - progress * 0.4),
                    filter_env_amount=filter_env_amount * 0.3,
                    resonance=final_resonance * 0.5,
                    amp_env=get_envelope(amp_env_name, config),
                    filter_env=get_filter_envelope(filter_env_name, config),
                    amp_env_name=amp_env_name,
                    filter_env_name=filter_env_name,
                    extra_context={
                        'mix_level': velocity,
                        'decay_curve': 'exp',  # Signal that we need exponential decay
                        'decay_amount': -3.0
                    }
                )
                events.append(event)

            # Advance with overlap for smooth outro
            advance_samples = self.calculate_advance_with_overlap(note_samples, params)
            timing.advance(advance_samples)

        # Debug output
        if events:
            self.print_debug_summary(events)

        return events
