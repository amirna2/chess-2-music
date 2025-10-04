"""
Test MarkovChainPattern to verify it generates valid events.
"""

import pytest
import numpy as np
from synth_composer.patterns.markov import MarkovChainPattern


class TestMarkovChainPattern:
    """Tests for MarkovChainPattern"""

    @pytest.fixture
    def rng(self):
        """Create deterministic RNG for testing"""
        return np.random.default_rng(seed=42)

    @pytest.fixture
    def pattern(self, rng):
        """Create MarkovChainPattern instance"""
        return MarkovChainPattern(rng)

    @pytest.fixture
    def scale(self):
        """A minor scale for testing"""
        return [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.0]

    @pytest.fixture
    def params(self):
        """Standard parameters for pattern generation"""
        return {
            'sample_rate': 44100,
            'section_start_time': 0.0,
            'filter': 1500.0,
            'filter_env': 2000.0,
            'resonance': 0.8,
            'note_duration': 0.25,
            'tension': 0.5,
            'config': None,  # Mock config
        }

    def test_initialization(self, pattern):
        """Test pattern initializes correctly"""
        assert pattern is not None
        assert pattern.transition_matrix.shape == (8, 8)
        # Verify probability distribution (each row sums to 1.0)
        row_sums = pattern.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_generate_events_basic(self, pattern, scale, params):
        """Test basic event generation"""
        duration = 2.0  # 2 seconds
        events = pattern.generate_events(duration, scale, params)

        # Should generate multiple events
        assert len(events) > 0
        assert len(events) < 100  # Sanity check (not infinite)

    def test_event_timestamps(self, pattern, scale, params):
        """Test that events have valid, increasing timestamps"""
        duration = 2.0
        events = pattern.generate_events(duration, scale, params)

        # All timestamps should be within section duration
        for event in events:
            assert 0.0 <= event.timestamp <= duration + 0.1  # Small tolerance

        # Timestamps should be monotonically increasing
        for i in range(len(events) - 1):
            assert events[i].timestamp <= events[i+1].timestamp

    def test_event_frequencies(self, pattern, scale, params):
        """Test that all frequencies come from the scale"""
        duration = 2.0
        events = pattern.generate_events(duration, scale, params)

        for event in events:
            assert event.freq in scale

    def test_event_parameters(self, pattern, scale, params):
        """Test that events have correct synthesis parameters"""
        duration = 1.0
        events = pattern.generate_events(duration, scale, params)

        for event in events:
            # Basic validations
            assert event.duration > 0
            assert 0.6 <= event.velocity <= 1.0  # As per pattern logic
            assert event.waveform == 'pulse'
            assert event.amp_env_name == 'soft'
            assert event.filter_env_name == 'gentle'
            # Extra context
            assert 'state' in event.extra_context
            assert event.extra_context['state'] == 'markov_walk'
            assert 'scale_degree' in event.extra_context

    def test_reproducibility(self, scale, params):
        """Test that same seed produces same events"""
        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)

        pattern1 = MarkovChainPattern(rng1)
        pattern2 = MarkovChainPattern(rng2)

        events1 = pattern1.generate_events(1.0, scale, params)
        events2 = pattern2.generate_events(1.0, scale, params)

        assert len(events1) == len(events2)

        for e1, e2 in zip(events1, events2):
            assert e1.freq == e2.freq
            assert np.isclose(e1.timestamp, e2.timestamp)
            assert np.isclose(e1.duration, e2.duration)
            assert np.isclose(e1.velocity, e2.velocity)

    def test_with_section_offset(self, pattern, scale, params):
        """Test that section_start_time offset works"""
        params['section_start_time'] = 10.0  # Start at 10 seconds

        events = pattern.generate_events(2.0, scale, params)

        # All timestamps should be >= 10.0
        for event in events:
            assert event.timestamp >= 10.0
            assert event.timestamp <= 12.1  # 10 + 2 + tolerance

    def test_markov_starts_on_tonic(self, pattern, scale, params):
        """Test that pattern always starts on tonic (index 0)"""
        events = pattern.generate_events(2.0, scale, params)

        # First event should be tonic
        assert events[0].freq == scale[0]

    def test_filter_modulation(self, pattern, scale, params):
        """Test that filter changes based on scale position"""
        events = pattern.generate_events(2.0, scale, params)

        base_filter = params['filter']

        # Events should have varying filter frequencies
        filter_values = [e.filter_base for e in events]
        assert len(set(filter_values)) > 1  # Not all the same

        # Filter multiplier should be in expected range (0.7 to 1.5)
        for event in events:
            filter_mult = event.filter_base / base_filter
            assert 0.7 <= filter_mult <= 1.5

    def test_velocity_variation(self, pattern, scale, params):
        """Test that velocity varies randomly"""
        events = pattern.generate_events(2.0, scale, params)

        velocities = [e.velocity for e in events]
        # Should have variety
        assert len(set([round(v, 2) for v in velocities])) > 1

        # All within expected range
        for v in velocities:
            assert 0.6 <= v <= 1.0

    def test_short_duration(self, pattern, scale, params):
        """Test with very short duration"""
        events = pattern.generate_events(0.1, scale, params)  # 100ms

        # Should handle gracefully
        assert len(events) >= 1  # At least one note
        for event in events:
            assert event.timestamp < 0.15  # Within tolerance

    def test_long_duration(self, pattern, scale, params):
        """Test with longer duration"""
        events = pattern.generate_events(10.0, scale, params)  # 10 seconds

        # Should generate many events
        assert len(events) > 10

        # Last event should be near end
        assert events[-1].timestamp <= 10.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
