"""
SpiegelRenderer - Laurie Spiegel style (1979)

Sparse drone-based aesthetic with algorithmic patterns and dramatic gestures.
Inspired by: The Expanding Universe, Appalachian Grove, Patchwork

Characteristics:
- Evolving drones (sub-bass foundation)
- Sparse melodic events (Markov chain with gravity)
- Minimal percussion (heartbeat pulse only)
- Prominent gestures (spectromorphological events)
- Textural complexity from entropy

This renderer maps existing components in synth_composer.py to maintain
backward compatibility while enabling the new style architecture.
"""

from typing import Dict, Any
import numpy as np
from .base import StyleRenderer


class SpiegelRenderer(StyleRenderer):
    """
    Spiegel-style renderer using existing ChessSynthComposer infrastructure.

    This is a thin wrapper that delegates to the existing compose_section logic
    to ensure backward compatibility during Phase 2 validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get style profile from config
        self.style_profile = self.config.STYLE_PROFILES['spiegel']

        # Store reference to composer (will be set during integration)
        self.composer = None

    def set_composer(self, composer):
        """
        Set reference to ChessSynthComposer for delegation.

        During Phase 2, we delegate to existing compose_section logic.
        Later, we can extract this into pure renderer methods.

        Args:
            composer: ChessSynthComposer instance
        """
        self.composer = composer

    def render_section(self,
                      section: Dict[str, Any],
                      context: Dict[str, Any]) -> np.ndarray:
        """
        Render section using Spiegel aesthetic.

        Applies style-specific mixing levels to layer outputs.

        Args:
            section: Section data from tags
            context: Overall context

        Returns:
            Stereo audio (N, 2)
        """
        if self.composer is None:
            raise RuntimeError(
                "SpiegelRenderer requires composer to be set. "
                "Call set_composer() before rendering."
            )

        # Temporarily override config with Spiegel style parameters
        original_values = {}

        # === MIXING LEVELS ===
        style_mixing = self.style_profile['mixing']
        for key in ['drone_level', 'pattern_level', 'sequencer_level', 'moment_level']:
            original_values[f'MIXING.{key}'] = self.config.MIXING.get(key)
            self.config.MIXING[key] = style_mixing[key]
            print(f"  [SPIEGEL] Set MIXING['{key}'] = {style_mixing[key]} (was {original_values[f'MIXING.{key}']})")

        # === LAYER 1 DRONE PARAMETERS ===
        if 'layer1_drone' in self.style_profile:
            layer1 = self.style_profile['layer1_drone']

            # Override narrative base params for current section narrative
            narrative = section.get('narrative', 'QUIET_PRECISION')
            if narrative in self.config.NARRATIVE_BASE_PARAMS:
                base_params = self.config.NARRATIVE_BASE_PARAMS[narrative]

                # Save originals
                for key in ['base_waveform', 'filter_start', 'filter_end', 'resonance_start', 'resonance_end', 'drone_voices']:
                    if key in base_params:
                        original_values[f'NARRATIVE.{narrative}.{key}'] = base_params.get(key)

                # Apply style overrides
                if 'waveform' in layer1:
                    base_params['base_waveform'] = layer1['waveform']
                if 'filter_base_hz' in layer1:
                    base_params['filter_start'] = layer1['filter_base_hz']
                if 'filter_range_hz' in layer1:
                    # filter_range_hz can be negative (sweep down)
                    base_params['filter_end'] = layer1['filter_base_hz'] + layer1['filter_range_hz']
                if 'resonance' in layer1:
                    base_params['resonance_start'] = layer1['resonance']
                    base_params['resonance_end'] = layer1['resonance']
                if 'voices' in layer1:
                    base_params['drone_voices'] = layer1['voices']

        try:
            # Delegate to existing compose_section logic
            section_index = context.get('section_index', 0)
            total_sections = context.get('total_sections', 1)

            return self.composer.compose_section(
                section,
                section_index,
                total_sections
            )
        finally:
            # Restore narrative params but NOT mixing levels
            # (mixing levels need to persist for final mix in compose())
            for key, value in original_values.items():
                if key.startswith('NARRATIVE.'):
                    parts = key.split('.')
                    narrative_name = parts[1]
                    param_name = parts[2]
                    if narrative_name in self.config.NARRATIVE_BASE_PARAMS:
                        self.config.NARRATIVE_BASE_PARAMS[narrative_name][param_name] = value

    def get_layer_config(self, narrative: str, tension: float) -> Dict[str, bool]:
        """
        Spiegel style uses: drone, sparse patterns, heartbeat, gestures.

        Args:
            narrative: Section narrative
            tension: Tension value

        Returns:
            Layer enable flags
        """
        return {
            'drone': True,              # Layer 1: Evolving drone
            'sparse_patterns': True,    # Layer 2: Sparse Markov patterns
            'heartbeat': True,          # Layer 3a: Heartbeat pulse
            'gestures': True,           # Layer 3b: Key moment gestures
            'sequenced_bass': False,    # Not used in Spiegel style
            'drums': False,             # Not used in Spiegel style
            'arpeggios': False,         # Not used in Spiegel style
            'pads': False,              # Not used in Spiegel style
        }
