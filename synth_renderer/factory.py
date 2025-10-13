"""
StyleRendererFactory - Factory for creating style renderer instances

Provides centralized creation of style renderers with proper initialization.
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import StyleRenderer


class StyleRendererFactory:
    """
    Factory for creating style renderer instances.

    Handles registration and instantiation of style renderers.
    """

    # Registry of available renderers
    _renderers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, renderer_class: type):
        """
        Register a renderer class.

        Args:
            name: Style name (e.g., 'spiegel', 'jarre')
            renderer_class: Renderer class (must extend StyleRenderer)
        """
        if not issubclass(renderer_class, StyleRenderer):
            raise TypeError(f"{renderer_class} must extend StyleRenderer")

        cls._renderers[name.lower()] = renderer_class

    @classmethod
    def create(cls,
               style: str,
               config: Any,
               synth_engines: Dict[str, Any],
               pattern_coordinator: Optional[Any] = None,
               gesture_coordinator: Optional[Any] = None,
               rng: Optional[np.random.Generator] = None) -> StyleRenderer:
        """
        Create a style renderer instance.

        Args:
            style: Style name (e.g., 'spiegel', 'jarre')
            config: SynthConfig instance
            synth_engines: Dict of SubtractiveSynth instances
            pattern_coordinator: PatternCoordinator instance
            gesture_coordinator: GestureCoordinator instance
            rng: NumPy random generator

        Returns:
            Initialized StyleRenderer instance

        Raises:
            ValueError: If style is not registered
        """
        style_lower = style.lower()

        # Auto-register known styles on first access
        if not cls._renderers:
            cls._auto_register()

        if style_lower not in cls._renderers:
            available = ', '.join(cls._renderers.keys())
            raise ValueError(
                f"Unknown style '{style}'. Available styles: {available}"
            )

        renderer_class = cls._renderers[style_lower]

        return renderer_class(
            config=config,
            synth_engines=synth_engines,
            pattern_coordinator=pattern_coordinator,
            gesture_coordinator=gesture_coordinator,
            rng=rng
        )

    @classmethod
    def _auto_register(cls):
        """
        Auto-register available renderer classes.

        Dynamically imports and registers renderer modules.
        """
        # Try to import SpiegelRenderer
        try:
            from .spiegel import SpiegelRenderer
            cls.register('spiegel', SpiegelRenderer)
        except ImportError:
            pass  # Spiegel renderer not yet implemented

        # Try to import JarreRenderer
        try:
            from .jarre import JarreRenderer
            cls.register('jarre', JarreRenderer)
        except ImportError:
            pass  # Jarre renderer not yet implemented

    @classmethod
    def list_styles(cls) -> list:
        """
        List available style names.

        Returns:
            List of registered style names
        """
        if not cls._renderers:
            cls._auto_register()

        return list(cls._renderers.keys())

    @classmethod
    def get_style_info(cls, style: str) -> Dict[str, Any]:
        """
        Get information about a style.

        Args:
            style: Style name

        Returns:
            Dict with style information (name, description, class)

        Raises:
            ValueError: If style is not registered
        """
        if not cls._renderers:
            cls._auto_register()

        style_lower = style.lower()

        if style_lower not in cls._renderers:
            available = ', '.join(cls._renderers.keys())
            raise ValueError(
                f"Unknown style '{style}'. Available styles: {available}"
            )

        renderer_class = cls._renderers[style_lower]

        return {
            'name': style_lower,
            'class': renderer_class.__name__,
            'description': renderer_class.__doc__.split('\n')[0] if renderer_class.__doc__ else 'No description'
        }
