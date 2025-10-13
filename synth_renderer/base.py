"""
StyleRenderer - Abstract base class for all style renderers

Defines the interface that all style renderers must implement.
Renderers translate chess narratives into style-specific synthesis parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class StyleRenderer(ABC):
    """
    Abstract base class for style renderers.

    A style renderer coordinates the generation of multiple audio layers
    according to a specific musical aesthetic (e.g., Spiegel, Jarre).

    Responsibilities:
    1. Translate narrative → layer decisions (which layers to generate)
    2. Translate narrative → synthesis parameters (waveforms, filters, etc.)
    3. Define mixing strategy (layer weights, stereo placement)
    4. Coordinate pattern generators
    """

    def __init__(self,
                 config,
                 synth_engines: Dict[str, Any],
                 pattern_coordinator: Optional[Any] = None,
                 gesture_coordinator: Optional[Any] = None,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize style renderer.

        Args:
            config: SynthConfig instance with all parameters
            synth_engines: Dict of SubtractiveSynth instances by name
                          (e.g., {'drone': synth, 'pattern': synth})
            pattern_coordinator: PatternCoordinator instance (for Layer 2)
            gesture_coordinator: GestureCoordinator instance (for Layer 3b)
            rng: NumPy random generator for reproducibility
        """
        self.config = config
        self.synth_engines = synth_engines
        self.pattern_coordinator = pattern_coordinator
        self.gesture_coordinator = gesture_coordinator
        self.rng = rng if rng is not None else np.random.default_rng()

        # Style profile - set by subclass
        self.style_profile = None

        # Sample rate from config
        self.sample_rate = config.SAMPLE_RATE if config else 88200

    @abstractmethod
    def render_section(self,
                      section: Dict[str, Any],
                      context: Dict[str, Any]) -> np.ndarray:
        """
        Main entry point: Render a chess section to stereo audio.

        This method orchestrates all layer generation and mixing according
        to the style's aesthetic.

        Args:
            section: Section data from tags JSON
                - 'narrative': Section narrative (e.g., 'DESPERATE_DEFENSE')
                - 'tension': Tension value (0.0-1.0)
                - 'key_moments': List of key moment dicts
                - 'start_ply': Starting ply number
                - 'end_ply': Ending ply number

            context: Overall context
                - 'section_index': Current section index
                - 'total_sections': Total number of sections
                - 'overall_narrative': Overall game narrative
                - 'scale': Musical scale (list of frequencies)
                - 'scale_name': Scale name (e.g., 'minor', 'phrygian')
                - 'sample_rate': Audio sample rate
                - 'section_duration': Duration in seconds

        Returns:
            Stereo audio array (N, 2) where N is sample count
        """
        pass

    # === Parameter Translation Methods ===
    # Subclasses implement these to map narrative → synth parameters

    @abstractmethod
    def get_layer_config(self, narrative: str, tension: float) -> Dict[str, bool]:
        """
        Decide which layers to generate for this narrative.

        Args:
            narrative: Section narrative
            tension: Tension value (0.0-1.0)

        Returns:
            Dict of layer_name -> enabled (bool)
            Example: {'drone': True, 'bass_seq': False, 'drums': False}
        """
        pass

    # === Utility Methods ===

    def mix_layers(self, layers: Dict[str, np.ndarray],
                   weights: Dict[str, float]) -> np.ndarray:
        """
        Mix multiple mono or stereo layers with specified weights.

        Args:
            layers: Dict of layer_name -> audio array
                   Arrays can be mono (N,) or stereo (N, 2)
            weights: Dict of layer_name -> mix weight (0.0-1.0)

        Returns:
            Mixed stereo audio (N, 2)
        """
        if not layers:
            return np.zeros((0, 2))

        # Find maximum length
        max_len = max(len(audio) for audio in layers.values())

        # Initialize stereo output
        mixed = np.zeros((max_len, 2))

        for layer_name, audio in layers.items():
            weight = weights.get(layer_name, 1.0)

            # Convert mono to stereo if needed
            if audio.ndim == 1:
                # Mono - duplicate to stereo
                stereo = np.zeros((len(audio), 2))
                stereo[:, 0] = audio
                stereo[:, 1] = audio
            else:
                stereo = audio

            # Add weighted layer to mix
            mixed[:len(stereo)] += stereo * weight

        return mixed

    def apply_stereo_pan(self,
                        mono_signal: np.ndarray,
                        pan: float,
                        width: float = 0.0) -> np.ndarray:
        """
        Apply stereo panning to mono signal.

        Args:
            mono_signal: Mono audio (N,)
            pan: Pan position (-1.0 = left, 0.0 = center, 1.0 = right)
            width: Stereo width (0.0 = mono, 1.0 = wide)

        Returns:
            Stereo audio (N, 2)
        """
        # Constant power panning
        pan_angle = (pan + 1.0) * np.pi / 4
        left_gain = np.cos(pan_angle)
        right_gain = np.sin(pan_angle)

        stereo = np.zeros((len(mono_signal), 2))

        if width == 0.0:
            # Simple panning
            stereo[:, 0] = mono_signal * left_gain
            stereo[:, 1] = mono_signal * right_gain
        else:
            # Add width using Haas effect
            delay_samples = int(width * 40)  # Up to 40 samples delay

            if delay_samples > 0:
                stereo[:-delay_samples, 0] = mono_signal[delay_samples:] * left_gain
                stereo[:, 1] = mono_signal * right_gain
            else:
                stereo[:, 0] = mono_signal * left_gain
                stereo[:, 1] = mono_signal * right_gain

        return stereo

    def get_style_name(self) -> str:
        """Get the name of this style renderer."""
        return self.__class__.__name__.replace('Renderer', '').lower()
