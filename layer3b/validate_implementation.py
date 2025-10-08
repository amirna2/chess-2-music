#!/usr/bin/env python3
"""
Validation script for Layer 3b implementation.

Verifies that all components are correctly implemented according to the
specification in docs/layer3b_implementation.md.
"""

import sys
import numpy as np
from typing import List, Tuple


def test_imports() -> Tuple[bool, str]:
    """Test that all modules can be imported."""
    try:
        from layer3b import GestureGenerator, GestureCoordinator, GestureSynthesizer, ARCHETYPES
        from layer3b.base import GestureGenerator as BaseGG
        from layer3b.coordinator import GestureCoordinator as CoordGC
        from layer3b.synthesizer import GestureSynthesizer as SynthGS
        from layer3b.archetype_configs import ARCHETYPES as ConfigAT
        from layer3b.curve_generators import (
            generate_pitch_curve,
            generate_harmony,
            generate_filter_curve,
            generate_envelope,
            generate_texture_curve
        )
        from layer3b.utils import compute_phases, finalize_audio, soft_clip
        return True, "All modules imported successfully"
    except Exception as e:
        return False, f"Import failed: {e}"


def test_archetype_configs() -> Tuple[bool, str]:
    """Test that all archetypes are properly configured."""
    from layer3b import ARCHETYPES

    required_keys = ['duration_base', 'phases', 'pitch', 'harmony', 'filter', 'envelope', 'texture']
    expected_archetypes = ['BLUNDER', 'BRILLIANT', 'TIME_PRESSURE', 'TACTICAL_SEQUENCE']

    for arch in expected_archetypes:
        if arch not in ARCHETYPES:
            return False, f"Missing archetype: {arch}"

        config = ARCHETYPES[arch]
        for key in required_keys:
            if key not in config:
                return False, f"Archetype {arch} missing key: {key}"

    return True, f"All {len(expected_archetypes)} archetypes properly configured"


def test_gesture_generator() -> Tuple[bool, str]:
    """Test GestureGenerator initialization and basic functionality."""
    from layer3b.base import GestureGenerator
    from layer3b.archetype_configs import ARCHETYPES

    rng = np.random.default_rng(42)

    try:
        # Test initialization
        generator = GestureGenerator(ARCHETYPES['BLUNDER'], rng)

        # Test duration computation
        context = {'tension': 0.7, 'entropy': 0.5}
        duration = generator._compute_duration(context)

        if not (0.5 <= duration <= 10.0):
            return False, f"Duration {duration}s outside valid range [0.5, 10.0]"

        return True, f"GestureGenerator validated (duration: {duration:.2f}s)"

    except Exception as e:
        return False, f"GestureGenerator test failed: {e}"


def test_gesture_coordinator() -> Tuple[bool, str]:
    """Test GestureCoordinator initialization and registry."""
    from layer3b import GestureCoordinator

    rng = np.random.default_rng(42)

    try:
        coordinator = GestureCoordinator(rng)

        # Test registry
        archetypes = coordinator.get_available_archetypes()
        if len(archetypes) < 4:
            return False, f"Expected at least 4 archetypes, got {len(archetypes)}"

        # Test config retrieval
        for arch in archetypes:
            config = coordinator.get_archetype_config(arch)
            if 'duration_base' not in config:
                return False, f"Config for {arch} missing duration_base"

        # Test duration computation
        context = {'tension': 0.5, 'entropy': 0.5}
        for arch in archetypes:
            duration = coordinator.compute_archetype_duration(arch, context)
            if not (0.5 <= duration <= 10.0):
                return False, f"{arch} duration {duration}s outside valid range"

        return True, f"GestureCoordinator validated ({len(archetypes)} archetypes)"

    except Exception as e:
        return False, f"GestureCoordinator test failed: {e}"


def test_synthesis_integration() -> Tuple[bool, str]:
    """Test complete synthesis pipeline."""
    from layer3b import GestureCoordinator
    from synth_engine import SubtractiveSynth

    sample_rate = 88200
    rng = np.random.default_rng(42)

    try:
        synth = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
        coordinator = GestureCoordinator(rng, synth_engine=synth)

        moment = {'event_type': 'BLUNDER', 'timestamp': 1.0, 'move_number': 1}
        context = {'tension': 0.7, 'entropy': 0.5, 'scale': 'C_MAJOR', 'key': 'C'}

        audio = coordinator.generate_gesture('BLUNDER', moment, context, sample_rate)

        # Validate audio
        if len(audio) == 0:
            return False, "Generated audio is empty"

        peak = np.max(np.abs(audio))
        if peak > 1.0:
            return False, f"Peak {peak:.4f} exceeds 1.0 (clipping)"

        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0.0:
            return False, "RMS is zero (silent audio)"

        duration = len(audio) / sample_rate

        return True, f"Synthesis validated (duration: {duration:.2f}s, peak: {peak:.4f}, RMS: {rms:.4f})"

    except Exception as e:
        return False, f"Synthesis test failed: {e}"


def test_all_archetypes() -> Tuple[bool, str]:
    """Test generation for all archetypes."""
    from layer3b import GestureCoordinator, ARCHETYPES
    from synth_engine import SubtractiveSynth

    sample_rate = 88200
    rng = np.random.default_rng(42)

    try:
        synth = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
        coordinator = GestureCoordinator(rng, synth_engine=synth)

        context = {'tension': 0.7, 'entropy': 0.5, 'scale': 'C_MAJOR', 'key': 'C'}

        results = []
        for archetype in ARCHETYPES.keys():
            moment = {'event_type': archetype, 'timestamp': 1.0, 'move_number': 1}
            audio = coordinator.generate_gesture(archetype, moment, context, sample_rate)

            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio ** 2))
            duration = len(audio) / sample_rate

            results.append(f"{archetype}: {duration:.2f}s, peak={peak:.3f}, RMS={rms:.3f}")

        return True, f"All archetypes generated:\n  " + "\n  ".join(results)

    except Exception as e:
        return False, f"Archetype test failed: {e}"


def test_error_handling() -> Tuple[bool, str]:
    """Test error handling for invalid inputs."""
    from layer3b import GestureCoordinator

    rng = np.random.default_rng(42)
    coordinator = GestureCoordinator(rng)

    errors_caught = []

    # Test 1: Unknown archetype
    try:
        moment = {'event_type': 'INVALID', 'timestamp': 1.0, 'move_number': 1}
        context = {'tension': 0.5, 'entropy': 0.5, 'scale': 'C_MAJOR', 'key': 'C'}
        coordinator.generate_gesture('INVALID', moment, context, 88200)
        return False, "Failed to catch unknown archetype"
    except ValueError:
        errors_caught.append("unknown archetype")

    # Test 2: Missing synth_engine
    try:
        moment = {'event_type': 'BLUNDER', 'timestamp': 1.0, 'move_number': 1}
        context = {'tension': 0.5, 'entropy': 0.5, 'scale': 'C_MAJOR', 'key': 'C'}
        coordinator.generate_gesture('BLUNDER', moment, context, 88200)
        return False, "Failed to catch missing synth_engine"
    except ValueError:
        errors_caught.append("missing synth_engine")

    # Test 3: Unknown config
    try:
        coordinator.get_archetype_config('INVALID')
        return False, "Failed to catch unknown config"
    except ValueError:
        errors_caught.append("unknown config")

    return True, f"Error handling validated ({len(errors_caught)} cases)"


def run_tests():
    """Run all validation tests."""
    tests = [
        ("Module Imports", test_imports),
        ("Archetype Configs", test_archetype_configs),
        ("GestureGenerator", test_gesture_generator),
        ("GestureCoordinator", test_gesture_coordinator),
        ("Synthesis Integration", test_synthesis_integration),
        ("All Archetypes", test_all_archetypes),
        ("Error Handling", test_error_handling),
    ]

    print("=" * 80)
    print("Layer 3b Implementation Validation")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"Testing {name}...", end=" ")
        try:
            success, message = test_func()
            if success:
                print(f"✓ PASS")
                print(f"  {message}")
                passed += 1
            else:
                print(f"✗ FAIL")
                print(f"  {message}")
                failed += 1
        except Exception as e:
            print(f"✗ ERROR")
            print(f"  Unhandled exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print()
        print("✓ All validation tests passed!")
        print()
        print("Layer 3b implementation is complete and ready for integration.")
        return 0
    else:
        print()
        print("✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
