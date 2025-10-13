"""
Synth Renderer Module - Style-based music rendering

This module provides style renderers that translate chess game narratives
into specific musical aesthetics (e.g., Spiegel, Jarre).

Key Components:
- StyleRenderer: Abstract base class for all renderers
- StyleRendererFactory: Factory for creating renderer instances
- SpiegelRenderer: Laurie Spiegel style (sparse, drone-based)
- JarreRenderer: Jean-Michel Jarre style (sequenced, rhythmic)
"""

from .base import StyleRenderer
from .factory import StyleRendererFactory

__all__ = [
    'StyleRenderer',
    'StyleRendererFactory',
]
