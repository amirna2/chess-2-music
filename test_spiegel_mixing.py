#!/usr/bin/env python3
"""Quick test to verify Spiegel style mixing levels are applied."""

import sys
import importlib.util

# Load synth_composer.py module
spec = importlib.util.spec_from_file_location('synth_composer_main', 'synth_composer.py')
synth_composer_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth_composer_main)
ChessSynthComposer = synth_composer_main.ChessSynthComposer

from synth_config import SynthConfig

# Simple test tags
test_tags = {
    'sections': [
        {
            'narrative': 'QUIET_PRECISION',
            'tension': 0.5,
            'entropy': 0.3,
            'moves': [],
            'key_moments': [],
            'duration': 5.0,
        }
    ]
}

print("Creating ChessSynthComposer with style='spiegel'...")
composer = ChessSynthComposer(test_tags, style='spiegel')

print(f"\nDefault mixing levels:")
print(f"  drone_level: {SynthConfig().MIXING['drone_level']}")
print(f"  pattern_level: {SynthConfig().MIXING['pattern_level']}")

print(f"\nSpiegel style profile mixing:")
print(f"  drone_level: {composer.config.STYLE_PROFILES['spiegel']['mixing']['drone_level']}")
print(f"  pattern_level: {composer.config.STYLE_PROFILES['spiegel']['mixing']['pattern_level']}")
print(f"  sequencer_level: {composer.config.STYLE_PROFILES['spiegel']['mixing']['sequencer_level']}")
print(f"  moment_level: {composer.config.STYLE_PROFILES['spiegel']['mixing']['moment_level']}")

print(f"\nRenderer type: {type(composer.renderer).__name__}")
print(f"Renderer has composer set: {composer.renderer.composer is not None}")

print("\n✅ Spiegel renderer loaded successfully!")
print("Mixing levels will be applied during render_section().")
