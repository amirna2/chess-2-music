"""
NoteEvent data structure.

Value object representing a single synthesized note with all parameters.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class NoteEvent:
    """
    Immutable note event with timestamp and synth parameters.

    Encapsulates ALL parameters needed to synthesize a single note.
    Timestamp is included at creation for automatic timeline tracking.
    """
    # Required parameters
    freq: float
    duration: float
    timestamp: float  # Absolute timeline position in seconds

    # Optional synthesis parameters with defaults
    velocity: float = 1.0
    waveform: str = 'saw'
    filter_base: float = 2000
    filter_env_amount: float = 3000
    resonance: float = 0.5
    amp_env: Tuple[float, float, float, float] = (0.01, 0.1, 0.7, 0.2)
    filter_env: Tuple[float, float, float, float] = (0.01, 0.15, 0.3, 0.2)
    amp_env_name: str = ''
    filter_env_name: str = ''
    extra_context: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Concise string representation for debugging"""
        return (f"NoteEvent({self.freq:.1f}Hz @ {self.timestamp:.3f}s, "
                f"dur={self.duration:.3f}s, vel={self.velocity:.2f})")
