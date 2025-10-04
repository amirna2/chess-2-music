"""
Unit tests for synth_composer core modules.

Tests NoteEvent, TimingEngine, AudioBuffer, and NoteSynthesizer.
"""

import pytest
import numpy as np
from synth_composer.core.note_event import NoteEvent
from synth_composer.core.timing_engine import TimingEngine
from synth_composer.core.audio_buffer import AudioBuffer


class TestNoteEvent:
    """Tests for NoteEvent data structure"""

    def test_basic_creation(self):
        """Test creating NoteEvent with minimal parameters"""
        event = NoteEvent(freq=440.0, duration=0.5, timestamp=1.0)
        assert event.freq == 440.0
        assert event.duration == 0.5
        assert event.timestamp == 1.0
        assert event.velocity == 1.0  # default
        assert event.waveform == 'saw'  # default

    def test_full_parameters(self):
        """Test creating NoteEvent with all parameters"""
        event = NoteEvent(
            freq=880.0,
            duration=0.25,
            timestamp=2.5,
            velocity=0.8,
            waveform='pulse',
            filter_base=1500,
            filter_env_amount=2500,
            resonance=1.2,
            amp_env=(0.01, 0.2, 0.8, 0.3),
            filter_env=(0.02, 0.3, 0.5, 0.4),
            amp_env_name='stab',
            filter_env_name='sweep',
            extra_context={'state': 'ATTACK', 'scale_degree': 4}
        )
        assert event.freq == 880.0
        assert event.waveform == 'pulse'
        assert event.amp_env_name == 'stab'
        assert event.extra_context['state'] == 'ATTACK'

    def test_immutability(self):
        """Test that NoteEvent is immutable (frozen dataclass)"""
        event = NoteEvent(freq=440.0, duration=0.5, timestamp=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            event.freq = 880.0

    def test_invalid_frequency(self):
        """Test validation: frequency must be positive"""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            NoteEvent(freq=0.0, duration=0.5, timestamp=1.0)

        with pytest.raises(ValueError, match="Frequency must be positive"):
            NoteEvent(freq=-440.0, duration=0.5, timestamp=1.0)

    def test_invalid_duration(self):
        """Test validation: duration must be positive"""
        with pytest.raises(ValueError, match="Duration must be positive"):
            NoteEvent(freq=440.0, duration=0.0, timestamp=1.0)

        with pytest.raises(ValueError, match="Duration must be positive"):
            NoteEvent(freq=440.0, duration=-0.5, timestamp=1.0)

    def test_invalid_velocity(self):
        """Test validation: velocity must be in [0, 1]"""
        with pytest.raises(ValueError, match="Velocity must be in"):
            NoteEvent(freq=440.0, duration=0.5, timestamp=1.0, velocity=1.5)

        with pytest.raises(ValueError, match="Velocity must be in"):
            NoteEvent(freq=440.0, duration=0.5, timestamp=1.0, velocity=-0.1)

    def test_invalid_waveform(self):
        """Test validation: waveform must be valid type"""
        with pytest.raises(ValueError, match="Invalid waveform"):
            NoteEvent(freq=440.0, duration=0.5, timestamp=1.0, waveform='invalid')

    def test_invalid_envelope(self):
        """Test validation: envelopes must have 4 values (ADSR)"""
        with pytest.raises(ValueError, match="must have 4 values"):
            NoteEvent(freq=440.0, duration=0.5, timestamp=1.0, amp_env=(0.1, 0.2, 0.3))


class TestTimingEngine:
    """Tests for TimingEngine"""

    def test_initialization(self):
        """Test TimingEngine initialization"""
        timing = TimingEngine(sample_rate=44100)
        assert timing.sample_rate == 44100
        assert timing.current_sample == 0
        assert timing.section_start_time == 0.0

    def test_initialization_with_offset(self):
        """Test TimingEngine with section start time offset"""
        timing = TimingEngine(sample_rate=44100, section_start_time=10.5)
        assert timing.section_start_time == 10.5

    def test_invalid_sample_rate(self):
        """Test validation: sample rate must be positive"""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            TimingEngine(sample_rate=0)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            TimingEngine(sample_rate=-44100)

    def test_get_timestamp(self):
        """Test timestamp calculation"""
        timing = TimingEngine(sample_rate=44100, section_start_time=10.0)
        assert timing.get_timestamp() == 10.0  # At start

        timing.current_sample = 44100  # +1 second
        assert timing.get_timestamp() == 11.0

        timing.current_sample = 88200  # +2 seconds
        assert timing.get_timestamp() == 12.0

    def test_get_section_relative_time(self):
        """Test section-relative time (ignores offset)"""
        timing = TimingEngine(sample_rate=44100, section_start_time=10.0)
        timing.current_sample = 44100  # +1 second
        assert timing.get_section_relative_time() == 1.0

    def test_advance_samples(self):
        """Test advancing by sample count"""
        timing = TimingEngine(sample_rate=44100)
        timing.advance(1000)
        assert timing.current_sample == 1000

        timing.advance(500)
        assert timing.current_sample == 1500

    def test_advance_negative_samples(self):
        """Test that negative advance raises error"""
        timing = TimingEngine(sample_rate=44100)
        with pytest.raises(ValueError, match="Cannot advance by negative"):
            timing.advance(-100)

    def test_advance_seconds(self):
        """Test advancing by duration in seconds"""
        timing = TimingEngine(sample_rate=44100)
        timing.advance_seconds(1.0)
        assert timing.current_sample == 44100

        timing.advance_seconds(0.5)
        assert timing.current_sample == int(44100 * 1.5)

    def test_advance_negative_seconds(self):
        """Test that negative duration raises error"""
        timing = TimingEngine(sample_rate=44100)
        with pytest.raises(ValueError, match="Cannot advance by negative"):
            timing.advance_seconds(-1.0)

    def test_add_pause(self):
        """Test add_pause (alias for advance_seconds)"""
        timing = TimingEngine(sample_rate=44100)
        timing.add_pause(0.5)
        assert timing.current_sample == int(44100 * 0.5)

    def test_remaining_samples(self):
        """Test remaining sample calculation"""
        timing = TimingEngine(sample_rate=44100)
        total = 44100 * 2  # 2 seconds

        assert timing.remaining_samples(total) == total  # At start

        timing.advance(44100)  # +1 second
        assert timing.remaining_samples(total) == 44100  # 1 second left

        timing.advance(44100)  # +1 second (at end)
        assert timing.remaining_samples(total) == 0

        timing.advance(1000)  # Past end
        assert timing.remaining_samples(total) == 0  # Never negative

    def test_remaining_seconds(self):
        """Test remaining time in seconds"""
        timing = TimingEngine(sample_rate=44100)
        total = 44100 * 2  # 2 seconds

        assert timing.remaining_seconds(total) == 2.0

        timing.advance(44100)
        assert timing.remaining_seconds(total) == 1.0

    def test_is_finished(self):
        """Test section end detection"""
        timing = TimingEngine(sample_rate=44100)
        total = 44100

        assert not timing.is_finished(total)

        timing.advance(44100)
        assert timing.is_finished(total)

        timing.advance(100)  # Past end
        assert timing.is_finished(total)

    def test_reset(self):
        """Test reset to beginning"""
        timing = TimingEngine(sample_rate=44100, section_start_time=5.0)
        timing.advance(10000)

        timing.reset()
        assert timing.current_sample == 0
        assert timing.section_start_time == 5.0  # Offset preserved


class TestAudioBuffer:
    """Tests for AudioBuffer"""

    def test_initialization(self):
        """Test AudioBuffer initialization"""
        buffer = AudioBuffer(1000)
        assert len(buffer) == 1000
        assert buffer.total_samples == 1000
        assert np.all(buffer.data == 0.0)

    def test_invalid_size(self):
        """Test validation: buffer size must be positive"""
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            AudioBuffer(0)

        with pytest.raises(ValueError, match="Buffer size must be positive"):
            AudioBuffer(-1000)

    def test_add_audio_basic(self):
        """Test basic audio mixing"""
        buffer = AudioBuffer(100)
        audio = np.ones(10) * 0.5

        buffer.add_audio(audio, start_sample=0)
        assert buffer.data[0] == 0.5
        assert buffer.data[9] == 0.5
        assert buffer.data[10] == 0.0  # Unaffected

    def test_add_audio_with_gain(self):
        """Test audio mixing with gain"""
        buffer = AudioBuffer(100)
        audio = np.ones(10)

        buffer.add_audio(audio, start_sample=0, gain=0.5)
        assert buffer.data[0] == 0.5

    def test_add_audio_mixing(self):
        """Test that add_audio mixes (+=) not replaces"""
        buffer = AudioBuffer(100)
        audio1 = np.ones(10) * 0.3
        audio2 = np.ones(10) * 0.4

        buffer.add_audio(audio1, start_sample=0)
        buffer.add_audio(audio2, start_sample=0)  # Same position
        assert np.isclose(buffer.data[0], 0.7)

    def test_add_audio_bounds_truncation(self):
        """Test that audio is truncated at buffer end"""
        buffer = AudioBuffer(100)
        audio = np.ones(20)

        buffer.add_audio(audio, start_sample=90)
        # Only 10 samples fit
        assert buffer.data[90] == 1.0
        assert buffer.data[99] == 1.0

    def test_add_audio_past_end(self):
        """Test that audio past buffer end is ignored"""
        buffer = AudioBuffer(100)
        audio = np.ones(10)

        buffer.add_audio(audio, start_sample=100)  # At end
        assert np.all(buffer.data == 0.0)  # Nothing added

        buffer.add_audio(audio, start_sample=200)  # Past end
        assert np.all(buffer.data == 0.0)  # Nothing added

    def test_add_audio_negative_start(self):
        """Test that negative start is ignored"""
        buffer = AudioBuffer(100)
        audio = np.ones(10)

        buffer.add_audio(audio, start_sample=-10)
        assert np.all(buffer.data == 0.0)  # Nothing added

    def test_add_audio_empty(self):
        """Test that empty audio is handled gracefully"""
        buffer = AudioBuffer(100)
        audio = np.array([])

        buffer.add_audio(audio, start_sample=0)  # Should not crash
        assert np.all(buffer.data == 0.0)

    def test_get_buffer(self):
        """Test getting final buffer"""
        buffer = AudioBuffer(100)
        audio = np.ones(10) * 0.5

        buffer.add_audio(audio, start_sample=0)
        result = buffer.get_buffer()
        assert len(result) == 100
        assert result[0] == 0.5

    def test_clear(self):
        """Test buffer clearing"""
        buffer = AudioBuffer(100)
        buffer.add_audio(np.ones(10), start_sample=0)

        buffer.clear()
        assert np.all(buffer.data == 0.0)

    def test_get_rms(self):
        """Test RMS calculation"""
        buffer = AudioBuffer(100)
        buffer.data[:] = 1.0  # All ones

        rms = buffer.get_rms()
        assert np.isclose(rms, 1.0)

    def test_get_peak(self):
        """Test peak calculation"""
        buffer = AudioBuffer(100)
        buffer.data[50] = 0.8
        buffer.data[75] = -0.9

        peak = buffer.get_peak()
        assert np.isclose(peak, 0.9)  # Absolute value

    def test_normalize(self):
        """Test buffer normalization"""
        buffer = AudioBuffer(100)
        buffer.data[:] = 0.5  # All 0.5

        buffer.normalize(target_peak=1.0)
        assert np.isclose(buffer.get_peak(), 1.0)

    def test_apply_gain(self):
        """Test gain application"""
        buffer = AudioBuffer(100)
        buffer.data[:] = 0.5

        buffer.apply_gain(2.0)
        assert np.all(buffer.data == 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
