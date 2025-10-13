#!/usr/bin/env python3
"""
Test script for configuration service.

Validates that config.yaml loads correctly and SynthConfig works.
"""

from config_service import get_config
from synth_config import DEFAULT_CONFIG, get_envelope, get_filter_envelope, get_narrative_params, get_section_modulation


def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration service...")
    print()

    # Test YAML loading
    print("=== YAML CONFIG SERVICE ===")
    cfg = get_config()
    print(f"Sample rate: {cfg.get('synthesis.sample_rate')}")
    print(f"Default BPM: {cfg.get('composition.rhythm.default_bpm')}")
    print(f"Root frequency: {cfg.get('composition.harmony.root_frequency')}")
    print()

    # Test SynthConfig dataclass
    print("=== SYNTHCONFIG DATACLASS ===")
    print(f"SAMPLE_RATE: {DEFAULT_CONFIG.SAMPLE_RATE}")
    print(f"DEFAULT_BPM: {DEFAULT_CONFIG.DEFAULT_BPM}")
    print(f"BASE_NOTE_DURATION: {DEFAULT_CONFIG.BASE_NOTE_DURATION}")
    print()

    # Test scales
    print("=== SCALES ===")
    print(f"Available scales: {list(DEFAULT_CONFIG.SCALES.keys())}")
    print(f"Minor scale (first 4 notes): {DEFAULT_CONFIG.SCALES['minor'][:4]}")
    print()

    # Test envelopes
    print("=== ENVELOPES ===")
    print(f"Available envelopes: {list(DEFAULT_CONFIG.ENVELOPES.keys())[:5]}")
    print(f"Drone envelope (ADSR): {DEFAULT_CONFIG.ENVELOPES['drone']}")
    print(f"Filter envelopes: {list(DEFAULT_CONFIG.FILTER_ENVELOPES.keys())[:5]}")
    print(f"Sweep filter envelope: {DEFAULT_CONFIG.FILTER_ENVELOPES['sweep']}")
    print()

    # Test narrative params
    print("=== NARRATIVE PARAMS ===")
    print(f"Narratives: {list(DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS.keys())[:3]}")
    tumbling = DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS['TUMBLING_DEFEAT']
    print(f"TUMBLING_DEFEAT scale: {tumbling.get('harmonic', {}).get('scale')}")
    print()

    # Test section modulation
    print("=== SECTION MODULATION ===")
    print(f"Modulations: {list(DEFAULT_CONFIG.SECTION_MODULATIONS.keys())[:3]}")
    king_hunt = DEFAULT_CONFIG.SECTION_MODULATIONS['KING_HUNT']
    print(f"KING_HUNT filter_mult: {king_hunt.get('filter_mult')}")
    print()

    # Test moment voices
    print("=== MOMENT VOICES ===")
    print(f"Has BLUNDER_IN_DEFEAT: {'BLUNDER_IN_DEFEAT' in DEFAULT_CONFIG.MOMENT_VOICES}")
    print(f"Has BRILLIANT_IN_MASTERPIECE: {'BRILLIANT_IN_MASTERPIECE' in DEFAULT_CONFIG.MOMENT_VOICES}")
    print(f"BLUNDER_IN_DEFEAT pitch: {DEFAULT_CONFIG.MOMENT_VOICES['BLUNDER_IN_DEFEAT'].get('pitch')}")
    print()

    # Test sequencer patterns
    print("=== SEQUENCER PATTERNS ===")
    print(f"Patterns: {list(DEFAULT_CONFIG.SEQUENCER_PATTERNS.keys())[:5]}")
    dev = DEFAULT_CONFIG.SEQUENCER_PATTERNS.get('DEVELOPMENT')
    if isinstance(dev, dict):
        print(f"DEVELOPMENT early pattern: {dev.get('early')[:8]}")
    print()

    # Test mixing
    print("=== MIXING ===")
    print(f"Mixing keys: {sorted(DEFAULT_CONFIG.MIXING.keys())[:8]}")
    print(f"section: {DEFAULT_CONFIG.MIXING.get('section')}")
    print(f"ducking_amount: {DEFAULT_CONFIG.MIXING.get('ducking_amount')}")
    print()

    # Test layer enable
    print("=== LAYER ENABLE ===")
    print(f"drone enabled: {DEFAULT_CONFIG.LAYER_ENABLE.get('drone')}")
    print(f"patterns enabled: {DEFAULT_CONFIG.LAYER_ENABLE.get('patterns')}")
    print()

    # Test backward compatibility helpers
    print("=== BACKWARD COMPATIBILITY HELPERS ===")
    env = get_envelope('percussive')
    print(f"get_envelope('percussive'): {env}")

    filt_env = get_filter_envelope('dramatic')
    print(f"get_filter_envelope('dramatic'): {filt_env}")

    narrative = get_narrative_params('ATTACKING_MASTERPIECE')
    print(f"get_narrative_params scale: {narrative.get('harmonic', {}).get('scale')}")

    section = get_section_modulation('DESPERATE_DEFENSE')
    print(f"get_section_modulation filter_mult: {section.get('filter_mult')}")
    print()

    print("✓ All tests passed!")


if __name__ == '__main__':
    test_config_loading()
