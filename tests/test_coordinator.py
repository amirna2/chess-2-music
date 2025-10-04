"""
Test PatternCoordinator integration.
"""

import pytest
import numpy as np
from synth_composer import PatternCoordinator
from synth_engine import SubtractiveSynth


@pytest.fixture
def rng():
    """Deterministic RNG"""
    return np.random.default_rng(seed=42)


@pytest.fixture
def synth_engine():
    """Create SubtractiveSynth instance"""
    return SubtractiveSynth(sample_rate=44100)


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


class TestPatternCoordinator:
    """Tests for PatternCoordinator"""

    def test_initialization(self, coordinator):
        """Test coordinator initializes correctly"""
        assert coordinator is not None
        assert coordinator.sample_rate == 44100
        assert len(coordinator.patterns) == 11  # All patterns registered

    def test_pattern_registry(self, coordinator):
        """Test all patterns are registered"""
        expected_patterns = [
            'COMPLEX_STRUGGLE',
            'KING_HUNT',
            'DESPERATE_DEFENSE',
            'TACTICAL_CHAOS',
            'CRUSHING_ATTACK',
            'SHARP_THEORY',
            'POSITIONAL_THEORY',
            'SOLID_THEORY',
            'FLAWLESS_CONVERSION',
            'DECISIVE_OUTRO',
            'DRAW_OUTRO',
        ]

        available = coordinator.get_available_patterns()
        assert len(available) == len(expected_patterns)
        for pattern in expected_patterns:
            assert pattern in available

    def test_generate_pattern_basic(self, coordinator, scale, params):
        """Test basic pattern generation"""
        audio = coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=1.0,
            scale=scale,
            params=params
        )

        # Should return audio buffer
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 44100  # 1 second at 44.1kHz
        assert audio.dtype == np.float32

    def test_generate_pattern_invalid_narrative(self, coordinator, scale, params):
        """Test that invalid narrative raises error"""
        with pytest.raises(ValueError, match="Unknown narrative"):
            coordinator.generate_pattern(
                narrative='INVALID_PATTERN',
                duration=1.0,
                scale=scale,
                params=params
            )

    @pytest.mark.parametrize("narrative", [
        'COMPLEX_STRUGGLE',
        'KING_HUNT',
        'DESPERATE_DEFENSE',
        'TACTICAL_CHAOS',
        'CRUSHING_ATTACK',
        'SHARP_THEORY',
        'POSITIONAL_THEORY',
        'SOLID_THEORY',
        'FLAWLESS_CONVERSION',
        'DECISIVE_OUTRO',
        'DRAW_OUTRO',
    ])
    def test_all_patterns_generate_audio(self, coordinator, scale, params, narrative):
        """Test that all patterns generate valid audio"""
        audio = coordinator.generate_pattern(
            narrative=narrative,
            duration=1.0,
            scale=scale,
            params=params
        )

        # Basic validation
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 44100
        assert not np.all(audio == 0)  # Should have some sound

    def test_different_durations(self, coordinator, scale, params):
        """Test pattern generation with different durations"""
        durations = [0.5, 1.0, 2.0, 5.0]

        for duration in durations:
            audio = coordinator.generate_pattern(
                narrative='COMPLEX_STRUGGLE',
                duration=duration,
                scale=scale,
                params=params
            )

            expected_samples = int(duration * 44100)
            assert len(audio) == expected_samples

    def test_section_start_time_offset(self, coordinator, scale, params):
        """Test that section_start_time parameter works"""
        params['section_start_time'] = 10.0  # Start at 10 seconds

        audio = coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=1.0,
            scale=scale,
            params=params
        )

        # Audio buffer should still be correct length (section-local)
        assert len(audio) == 44100

    def test_mix_level(self, coordinator, scale, params):
        """Test that mix_level affects output amplitude"""
        # Generate with full level
        params['mix_level'] = 1.0
        audio_full = coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=1.0,
            scale=scale,
            params=params
        )

        # Generate with half level
        params['mix_level'] = 0.5
        audio_half = coordinator.generate_pattern(
            narrative='COMPLEX_STRUGGLE',
            duration=1.0,
            scale=scale,
            params=params
        )

        # Half level should have lower RMS
        rms_full = np.sqrt(np.mean(audio_full ** 2))
        rms_half = np.sqrt(np.mean(audio_half ** 2))
        assert rms_half < rms_full

    def test_reproducibility(self):
        """Test that same seed produces same output"""
        scale = [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.0]
        params = {
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

        # Create two coordinators with same seed (including synth engine RNG)
        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)
        synth_rng1 = np.random.default_rng(seed=456)
        synth_rng2 = np.random.default_rng(seed=456)

        synth_engine1 = SubtractiveSynth(sample_rate=44100, rng=synth_rng1)
        synth_engine2 = SubtractiveSynth(sample_rate=44100, rng=synth_rng2)

        coordinator1 = PatternCoordinator(44100, None, synth_engine1, rng1)
        coordinator2 = PatternCoordinator(44100, None, synth_engine2, rng2)

        audio1 = coordinator1.generate_pattern('COMPLEX_STRUGGLE', 1.0, scale, params)
        audio2 = coordinator2.generate_pattern('COMPLEX_STRUGGLE', 1.0, scale, params)

        # Should be identical (or very close due to floating point)
        assert np.allclose(audio1, audio2, atol=1e-6)

    def test_register_custom_pattern(self, coordinator, scale, params):
        """Test registering custom pattern"""
        from synth_composer.patterns.base import PatternGenerator
        from synth_composer.core.note_event import NoteEvent

        class CustomPattern(PatternGenerator):
            def generate_events(self, duration, scale, params):
                return [NoteEvent(
                    freq=scale[0],
                    duration=0.1,
                    timestamp=0.0,
                    velocity=1.0,
                )]

        custom = CustomPattern(np.random.default_rng())
        coordinator.register_pattern('CUSTOM', custom)

        assert 'CUSTOM' in coordinator.get_available_patterns()

        # Should be able to generate
        audio = coordinator.generate_pattern('CUSTOM', 1.0, scale, params)
        assert isinstance(audio, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
