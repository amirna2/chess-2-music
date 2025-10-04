"""
AudioBuffer - Single location for buffer management.

Manages audio sample buffer with safe mixing operations.
"""

import numpy as np


class AudioBuffer:
    """
    Manages audio sample buffer with safe mixing operations.
    Handles bounds checking, prevents buffer overruns.
    """

    def __init__(self, total_samples: int):
        """
        Initialize audio buffer.

        Args:
            total_samples: Buffer size in samples
        """
        self.data = np.zeros(total_samples, dtype=np.float32)
        self.total_samples = total_samples

    def add_audio(self, audio: np.ndarray, start_sample: int, gain: float = 1.0):
        """
        Mix audio into buffer with automatic bounds checking.

        Args:
            audio: Audio samples to add
            start_sample: Starting position in buffer
            gain: Amplitude multiplier (velocity * mix_level)
        """
        if start_sample >= self.total_samples:
            return  # Past end of buffer

        end_sample = min(start_sample + len(audio), self.total_samples)
        if end_sample > start_sample:
            audio_len = end_sample - start_sample
            self.data[start_sample:end_sample] += audio[:audio_len] * gain

    def get_buffer(self) -> np.ndarray:
        """Return final audio buffer"""
        return self.data

    def __len__(self) -> int:
        """Return buffer length in samples"""
        return self.total_samples
