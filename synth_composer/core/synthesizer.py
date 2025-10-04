"""
NoteSynthesizer - Wrapper around SubtractiveSynth engine.

Converts NoteEvent objects to audio samples.
"""

import numpy as np
from .note_event import NoteEvent


class NoteSynthesizer:
    """
    Wrapper around SubtractiveSynth that accepts NoteEvent objects.

    IMPORTANT: Only passes parameters that synth_engine.create_synth_note() accepts.
    Does NOT pass timestamp, velocity, env names, or extra_context to synth engine.
    """

    def __init__(self, synth_engine, sample_rate: int):
        """
        Initialize synthesizer wrapper.

        Args:
            synth_engine: SubtractiveSynth instance
            sample_rate: Audio sample rate
        """
        self.synth = synth_engine
        self.sample_rate = sample_rate

    def synthesize(self, event: NoteEvent) -> np.ndarray:
        """
        Render NoteEvent to audio samples.

        Extracts only the parameters that create_synth_note() accepts.
        Ignores timestamp, velocity, env_names, and extra_context.

        Args:
            event: NoteEvent with all synthesis parameters

        Returns:
            Audio samples (mono, 1D numpy array)
        """
        return self.synth.create_synth_note(
            freq=event.freq,
            duration=event.duration,
            waveform=event.waveform,
            filter_base=event.filter_base,
            filter_env_amount=event.filter_env_amount,
            resonance=event.resonance,
            amp_env=event.amp_env,
            filter_env=event.filter_env
        )

    def synthesize_batch(self, events: list) -> list:
        """
        Synthesize multiple events.

        Args:
            events: List of NoteEvent objects

        Returns:
            List of audio numpy arrays
        """
        return [self.synthesize(event) for event in events]
