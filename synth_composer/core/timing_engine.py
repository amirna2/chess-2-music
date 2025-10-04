"""
TimingEngine - Single location for all timing logic.

Manages timeline position within a section and calculates timestamps.
"""


class TimingEngine:
    """
    Manages timeline position within a section.
    Automatically calculates timestamps for note events.
    """

    def __init__(self, sample_rate: int, section_start_time: float = 0.0):
        """
        Initialize timing engine.

        Args:
            sample_rate: Audio sample rate (Hz)
            section_start_time: Global timeline offset (seconds)
        """
        self.sample_rate = sample_rate
        self.current_sample = 0  # Section-local sample position
        self.section_start_time = section_start_time  # Global timeline offset

    def get_timestamp(self) -> float:
        """Get current absolute timeline position in seconds"""
        return self.section_start_time + (self.current_sample / self.sample_rate)

    def advance(self, samples: int):
        """Advance playhead by sample count"""
        self.current_sample += samples

    def advance_seconds(self, duration: float):
        """Advance playhead by duration in seconds"""
        self.current_sample += int(duration * self.sample_rate)

    def add_pause(self, duration: float):
        """Add pause/silence (same as advance_seconds)"""
        self.advance_seconds(duration)

    def remaining_samples(self, total_samples: int) -> int:
        """Get remaining samples in section"""
        return max(0, total_samples - self.current_sample)

    def remaining_seconds(self, total_samples: int) -> float:
        """Get remaining time in section (seconds)"""
        return self.remaining_samples(total_samples) / self.sample_rate

    def is_finished(self, total_samples: int) -> bool:
        """Check if we've reached end of section"""
        return self.current_sample >= total_samples

    @property
    def current_time(self) -> float:
        """Get current section-local time in seconds"""
        return self.current_sample / self.sample_rate
