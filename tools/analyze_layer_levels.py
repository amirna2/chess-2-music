#!/usr/bin/env python3
"""
Analyze and visualize audio levels for each layer in a chess composition section.
"""

import sys
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from the synth_composer.py module file directly
import importlib.util
spec = importlib.util.spec_from_file_location("synth_composer_module", parent_dir / "synth_composer.py")
synth_composer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth_composer_module)

ChessSynthComposer = synth_composer_module.ChessSynthComposer

from synth_config import SynthConfig


def rms_to_db(rms):
    """Convert RMS to dB"""
    if rms <= 0:
        return -np.inf
    return 20 * np.log10(rms)


def analyze_section(tags_file, section_name='OPENING'):
    """Analyze layer levels for a specific section"""

    # Load tags
    with open(tags_file) as f:
        tags = json.load(f)

    # Find the section
    sections = tags.get('sections', [])
    section = None
    for s in sections:
        if s.get('name') == section_name:
            section = s
            break

    if not section:
        print(f"Section {section_name} not found!")
        return

    # Create composer
    config = SynthConfig()
    composer = ChessSynthComposer(tags, config)

    # Render just this section
    print(f"\n{'='*70}")
    print(f"ANALYZING SECTION: {section_name}")
    print(f"{'='*70}\n")

    section_data = composer.compose_section(section, 0, 1)

    # Extract layers
    layers_1_2 = section_data['layers_1_2']
    layer_3a = section_data['layer_3a_heartbeat']
    layer_3b = section_data['layer_3b_moments']

    # Get individual drone and pattern BEFORE they're mixed
    base_drone = section_data.get('base_drone', np.zeros(1))
    section_pattern = section_data.get('section_pattern', np.zeros(1))

    # Pad all layers to same length
    max_len = max(len(layers_1_2), len(layer_3a), len(layer_3b))

    def pad_to_length(arr, length):
        if len(arr) < length:
            return np.concatenate([arr, np.zeros(length - len(arr))])
        return arr[:length]

    layers_1_2_padded = pad_to_length(layers_1_2, max_len)
    layer_3a_padded = pad_to_length(layer_3a, max_len)
    layer_3b_padded = pad_to_length(layer_3b, max_len)

    # Pad drone and pattern to same length for analysis
    base_drone_padded = pad_to_length(base_drone, max_len)
    section_pattern_padded = pad_to_length(section_pattern, max_len)

    # These layers are ALREADY scaled by compose_section!
    # DO NOT multiply again - just analyze as-is
    layers = {
        'Layer 1 (Drone) [PRE-MIX]': base_drone_padded,
        'Layer 2 (Patterns) [PRE-MIX]': section_pattern_padded,
        'Layer 1+2 (COMBINED)': layers_1_2_padded,
        'Layer 3a (Heartbeat)': layer_3a_padded,
        'Layer 3b (Gestures)': layer_3b_padded,
    }

    final_mix = layers_1_2_padded + layer_3a_padded + layer_3b_padded

    print(f"\n{'Layer':<35} {'RMS':<12} {'Peak':<12} {'RMS (dB)':<12} {'Peak (dB)':<12}")
    print(f"{'-'*80}")

    results = {}

    for name, audio in layers.items():
        if len(audio) == 0:
            rms = 0.0
            peak = 0.0
        else:
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))

        rms_db = rms_to_db(rms) if rms > 0 else -np.inf
        peak_db = rms_to_db(peak) if peak > 0 else -np.inf

        results[name] = {'rms': rms, 'peak': peak, 'rms_db': rms_db, 'peak_db': peak_db}

        print(f"{name:<35} {rms:<12.6f} {peak:<12.6f} {rms_db:<12.2f} {peak_db:<12.2f}")

    # Final mix
    final_rms = np.sqrt(np.mean(final_mix**2))
    final_peak = np.max(np.abs(final_mix))
    final_rms_db = rms_to_db(final_rms) if final_rms > 0 else -np.inf
    final_peak_db = rms_to_db(final_peak) if final_peak > 0 else -np.inf

    print(f"{'-'*80}")
    print(f"{'FINAL MIX':<35} {final_rms:<12.6f} {final_peak:<12.6f} {final_rms_db:<12.2f} {final_peak_db:<12.2f}")

    # Calculate ACTUAL power percentages in final mix
    power_12 = results['Layer 1+2 (COMBINED)']['rms'] ** 2
    power_3a = results['Layer 3a (Heartbeat)']['rms'] ** 2
    power_3b = results['Layer 3b (Gestures)']['rms'] ** 2
    total_power = power_12 + power_3a + power_3b

    actual_pct_12 = (power_12 / total_power * 100) if total_power > 0 else 0
    actual_pct_3a = (power_3a / total_power * 100) if total_power > 0 else 0
    actual_pct_3b = (power_3b / total_power * 100) if total_power > 0 else 0

    # Print config values vs actual
    print(f"\n{'='*70}")
    print(f"POWER BUDGET VERIFICATION")
    print(f"{'='*70}")
    print(f"{'Layer':<20} {'Target %':<12} {'Actual %':<12} {'Match':<10}")
    print(f"{'-'*54}")

    target_12 = (config.MIXING['drone_level'] + config.MIXING['patterns_level']) * 100
    target_3a = config.MIXING['sequencer_level'] * 100
    target_3b = config.MIXING['gestures_level'] * 100

    match_12 = "✓" if abs(actual_pct_12 - target_12) < 5 else "✗"
    match_3a = "✓" if abs(actual_pct_3a - target_3a) < 5 else "✗"
    match_3b = "✓" if abs(actual_pct_3b - target_3b) < 5 else "✗"

    print(f"{'Layers 1+2':<20} {target_12:<12.1f} {actual_pct_12:<12.1f} {match_12:<10}")
    print(f"{'Heartbeat':<20} {target_3a:<12.1f} {actual_pct_3a:<12.1f} {match_3a:<10}")
    print(f"{'Gestures':<20} {target_3b:<12.1f} {actual_pct_3b:<12.1f} {match_3b:<10}")

    print(f"\n{'='*70}")
    print(f"INDIVIDUAL LAYER TARGETS")
    print(f"{'='*70}")
    print(f"  Drone:      {config.MIXING['drone_level'] * 100:.0f}%")
    print(f"  Patterns:   {config.MIXING['patterns_level'] * 100:.0f}%")
    print(f"  Heartbeat:  {config.MIXING['sequencer_level'] * 100:.0f}%")
    print(f"  Gestures:   {config.MIXING['gestures_level'] * 100:.0f}%")

    # Visual bar chart
    print(f"\n{'='*70}")
    print(f"VISUAL LEVEL COMPARISON (RMS)")
    print(f"{'='*70}")

    max_rms = max([r['rms'] for r in results.values()] + [final_rms])

    for name, data in results.items():
        rms = data['rms']
        if max_rms > 0:
            bar_width = int((rms / max_rms) * 50)
        else:
            bar_width = 0
        bar = '█' * bar_width
        print(f"{name:<35} {bar} {rms:.6f}")

    print(f"{'-'*70}")
    bar_width = int((final_rms / max_rms) * 50) if max_rms > 0 else 0
    bar = '█' * bar_width
    print(f"{'FINAL MIX':<35} {bar} {final_rms:.6f}")

    print(f"\n{'='*70}")
    print(f"DIAGNOSTICS")
    print(f"{'='*70}")

    # Check for issues
    if final_peak > 1.0:
        print(f"⚠️  WARNING: Final mix is clipping! Peak = {final_peak:.3f} (should be ≤ 1.0)")
        print(f"   → Reduce layer levels to prevent distortion")
    elif final_peak < 0.1:
        print(f"⚠️  WARNING: Final mix is very quiet! Peak = {final_peak:.3f}")
        print(f"   → Increase layer levels or layer_1_plus_2_master_level")
    elif final_peak < 0.3:
        print(f"ℹ️  INFO: Final mix is somewhat quiet. Peak = {final_peak:.3f}")
        print(f"   → Consider increasing levels for more presence")
    else:
        print(f"✓  Final mix peak is good: {final_peak:.3f}")

    # Check RMS
    if final_rms < 0.05:
        print(f"⚠️  WARNING: Final mix RMS is very low! RMS = {final_rms:.6f}")
        print(f"   → This will sound very quiet. Increase layer levels.")
    elif final_rms < 0.1:
        print(f"ℹ️  INFO: Final mix RMS is low. RMS = {final_rms:.6f}")
        print(f"   → May sound quiet on some playback systems.")

    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_layer_levels.py <tags_file> [section_name]")
        print("Example: python3 analyze_layer_levels.py tags-game1.json OPENING")
        sys.exit(1)

    tags_file = sys.argv[1]
    section_name = sys.argv[2] if len(sys.argv) > 2 else 'OPENING'

    analyze_section(tags_file, section_name)
