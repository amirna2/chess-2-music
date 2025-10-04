"""
Test SequencerLogger integration with synthesis pipeline.
"""

import pytest
import numpy as np
from synth_composer import PatternCoordinator
from synth_engine import SubtractiveSynth
from synth_logger import SequencerLogger


@pytest.fixture
def rng():
    """Deterministic RNG"""
    return np.random.default_rng(seed=42)


@pytest.fixture
def synth_engine(rng):
    """Create SubtractiveSynth instance"""
    return SubtractiveSynth(sample_rate=44100, rng=rng)


@pytest.fixture
def coordinator(rng, synth_engine):
    """Create PatternCoordinator"""
    return PatternCoordinator(
        sample_rate=44100,
        config=None,
        synth_engine=synth_engine,
        rng=rng
    )


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
        'mix_level': 1.0,
    }


class TestSequencerLogger:
    """Tests for SequencerLogger integration"""

    def test_logger_initialization(self):
        """Test logger initializes correctly"""
        logger = SequencerLogger()
        assert len(logger) == 0
        assert len(logger.events_by_timestamp) == 0

    def test_logger_captures_events(self, coordinator, scale, params):
        """Test that logger captures synthesis events"""
        logger = SequencerLogger()

        # Generate pattern with logging
        audio = coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=1.0,
            scale=scale,
            params=params,
            logger=logger.log_event
        )

        # Should have captured events
        assert len(logger) > 0
        assert len(logger.events_by_timestamp) > 0

        # Audio should still be generated correctly
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 44100

    def test_logger_timeline_report(self, coordinator, scale, params):
        """Test timeline report generation"""
        logger = SequencerLogger()

        # Generate short pattern
        coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=0.5,
            scale=scale,
            params=params,
            logger=logger.log_event
        )

        # Get report
        report = logger.get_timeline_report()

        # Should contain header
        assert "SEQUENCER TIMELINE VIEW" in report
        assert "Total events:" in report

        # Should contain event data
        assert "Hz" in report
        assert "dur=" in report
        assert "vel=" in report

    def test_logger_groups_by_timestamp(self, coordinator, scale, params):
        """Test that events are properly grouped by timestamp"""
        logger = SequencerLogger()

        coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=1.0,
            scale=scale,
            params=params,
            logger=logger.log_event
        )

        # Check that timestamps are sorted
        timestamps = sorted(logger.events_by_timestamp.keys())
        assert timestamps == sorted(timestamps)

        # Each timestamp should have events
        for ts in timestamps:
            assert len(logger.events_by_timestamp[ts]) > 0

    def test_logger_clears(self, coordinator, scale, params):
        """Test logger clear functionality"""
        logger = SequencerLogger()

        coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=0.5,
            scale=scale,
            params=params,
            logger=logger.log_event
        )

        assert len(logger) > 0

        logger.clear()

        assert len(logger) == 0
        assert len(logger.events_by_timestamp) == 0

    def test_logger_with_multiple_patterns(self, coordinator, scale, params):
        """Test logger accumulates events across multiple patterns"""
        logger = SequencerLogger()

        # Generate two different patterns
        coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=0.5,
            scale=scale,
            params=params,
            logger=logger.log_event
        )

        events_after_first = len(logger)

        coordinator.generate_pattern(
            narrative='KING_HUNT',
            duration=0.5,
            scale=scale,
            params=params,
            logger=logger.log_event
        )

        # Should have more events after second pattern
        assert len(logger) > events_after_first

    def test_logger_without_logger_parameter(self, coordinator, scale, params):
        """Test that synthesis works without logger (backward compatibility)"""
        # Should not raise error
        audio = coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=0.5,
            scale=scale,
            params=params
            # No logger parameter
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) == int(0.5 * 44100)

    def test_logger_repr(self):
        """Test logger string representation"""
        logger = SequencerLogger()
        assert "SequencerLogger" in repr(logger)
        assert "0 events" in repr(logger)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
