"""
Comprehensive tests for all pattern generators.
"""

import pytest
import numpy as np
from synth_composer.patterns.markov import MarkovChainPattern
from synth_composer.patterns.state_machine import StateMachinePattern
from synth_composer.patterns.theory import TheoryPattern
from synth_composer.patterns.outro import OutroPattern
from synth_composer.patterns.conversion import ConversionPattern


@pytest.fixture
def rng():
    """Deterministic RNG"""
    return np.random.default_rng(seed=42)


@pytest.fixture
def scale():
    """A minor scale"""
    return [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.0]


@pytest.fixture
def params():
    """Standard parameters"""
    return {
        'sample_rate': 44100,
        'section_start_time': 0.0,
        'filter': 1500.0,
        'filter_env': 2000.0,
        'resonance': 0.8,
        'note_duration': 0.25,
        'tension': 0.5,
        'config': None,
    }


class TestAllPatterns:
    """Test all pattern generators"""

    def test_markov_chain_pattern(self, rng, scale, params):
        """Test COMPLEX_STRUGGLE pattern"""
        pattern = MarkovChainPattern(rng)
        events = pattern.generate_events(2.0, scale, params)

        assert len(events) > 0
        assert all(0 <= e.timestamp <= 2.1 for e in events)
        assert all(e.freq in scale for e in events)

    @pytest.mark.parametrize("pattern_type", [
        'KING_HUNT',
        'DESPERATE_DEFENSE',
        'TACTICAL_CHAOS',
        'CRUSHING_ATTACK'
    ])
    def test_state_machine_patterns(self, rng, scale, params, pattern_type):
        """Test all state machine patterns"""
        pattern = StateMachinePattern(rng, pattern_type)
        events = pattern.generate_events(2.0, scale, params)

        assert len(events) > 0
        assert all(0 <= e.timestamp <= 2.1 for e in events)
        assert all(e.duration > 0 for e in events)
        assert all('state' in e.extra_context for e in events)

    @pytest.mark.parametrize("theory_type", ['SHARP', 'POSITIONAL', 'SOLID'])
    def test_theory_patterns(self, rng, scale, params, theory_type):
        """Test all theory patterns"""
        pattern = TheoryPattern(rng, theory_type)
        events = pattern.generate_events(2.0, scale, params)

        assert len(events) > 0
        assert all(0 <= e.timestamp <= 2.1 for e in events)
        assert all(e.freq in scale for e in events)

    @pytest.mark.parametrize("outro_type", ['DECISIVE', 'DRAW'])
    def test_outro_patterns(self, rng, scale, params, outro_type):
        """Test outro patterns"""
        pattern = OutroPattern(rng, outro_type)
        events = pattern.generate_events(2.0, scale, params)

        assert len(events) > 0
        assert all(0 <= e.timestamp <= 2.1 for e in events)
        # Verify fadeout (velocity decreases)
        if len(events) > 1:
            assert events[-1].velocity < events[0].velocity

    def test_conversion_pattern(self, rng, scale, params):
        """Test FLAWLESS_CONVERSION pattern"""
        pattern = ConversionPattern(rng)
        events = pattern.generate_events(2.0, scale, params)

        assert len(events) > 0
        assert all(0 <= e.timestamp <= 2.1 for e in events)
        assert all(e.freq in scale for e in events)
        assert all(e.waveform == 'triangle' for e in events)

    def test_all_patterns_generate_valid_events(self, scale, params):
        """Ensure all patterns generate valid NoteEvent objects"""
        rng = np.random.default_rng(seed=123)

        patterns = [
            MarkovChainPattern(rng),
            StateMachinePattern(rng, 'KING_HUNT'),
            StateMachinePattern(rng, 'DESPERATE_DEFENSE'),
            StateMachinePattern(rng, 'TACTICAL_CHAOS'),
            StateMachinePattern(rng, 'CRUSHING_ATTACK'),
            TheoryPattern(rng, 'SHARP'),
            TheoryPattern(rng, 'POSITIONAL'),
            TheoryPattern(rng, 'SOLID'),
            OutroPattern(rng, 'DECISIVE'),
            OutroPattern(rng, 'DRAW'),
            ConversionPattern(rng),
        ]

        for pattern in patterns:
            events = pattern.generate_events(1.0, scale, params)

            # Basic validation
            assert len(events) > 0, f"{pattern} generated no events"

            for event in events:
                # Validate NoteEvent fields
                assert event.freq > 0
                assert event.duration > 0
                assert event.timestamp >= 0
                assert 0 <= event.velocity <= 1.0
                assert event.waveform in ('saw', 'pulse', 'triangle', 'sine', 'square', 'supersaw')
                assert event.amp_env_name != ''
                assert event.filter_env_name != ''
                assert isinstance(event.extra_context, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
