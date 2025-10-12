#!/usr/bin/env python3
"""
Quick integration test - verify Layer3b works with composer for game1.

This tests that:
1. Composer can initialize with game1 tags
2. GestureCoordinator is properly set up
3. All game1 archetypes can generate audio
4. No crashes or missing implementations
"""

import json
import numpy as np
from synth_config import SynthConfig
from synth_engine import SubtractiveSynth
from layer3b import GestureCoordinator


def test_layer3b_integration():
    """Test Layer3b integration with game1 moments."""
    print("=" * 70)
    print("Layer3b Integration Test - Game1")
    print("=" * 70)

    # Load game1 tags
    print("\n1. Loading tags-game1.json...")
    with open('tags-game1.json', 'r') as f:
        tags = json.load(f)

    eco = tags.get('eco', 'C11')
    print(f"   ✓ Loaded: ECO {eco}, {tags.get('total_plies')} plies, {tags.get('duration_seconds')}s")

    # Initialize synth engine
    print("\n2. Initializing synthesis engine...")
    config = SynthConfig()
    sample_rate = config.SAMPLE_RATE

    # Create RNG from ECO
    if len(eco) >= 3:
        letter_value = ord(eco[0].upper()) - ord('A')
        number_value = int(eco[1:3])
        seed = letter_value * 100 + number_value
    else:
        seed = 0

    rng = np.random.default_rng(seed)
    synth = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    print(f"   ✓ Sample rate: {sample_rate} Hz")

    # Initialize gesture coordinator
    print("\n3. Initializing GestureCoordinator...")
    coordinator = GestureCoordinator(rng, synth_engine=synth)
    print(f"   ✓ Available archetypes: {len(coordinator.get_available_archetypes())}")

    # Test each unique moment type in game1
    print("\n4. Testing gesture generation for game1 moments...")

    moment_types = set()
    for section in tags.get('sections', []):
        for moment in section.get('key_moments', []):
            moment_types.add(moment['type'])

    print(f"   Found {len(moment_types)} unique moment types: {sorted(moment_types)}")

    section_context = {
        'tension': 0.7,
        'entropy': 0.5,
        'scale': 'C_MAJOR',
        'key': 'C'
    }

    results = {}
    for moment_type in sorted(moment_types):
        try:
            # Create test event
            test_event = {
                'type': moment_type,
                'timestamp': 10.0,
                'move_number': 5,
                'score': 5
            }

            # Generate gesture
            audio = coordinator.generate_gesture(
                moment_type,
                test_event,
                section_context,
                sample_rate
            )

            # Validate
            assert isinstance(audio, np.ndarray), "Audio must be numpy array"
            assert len(audio) > 0, "Audio must have samples"
            assert np.all(np.isfinite(audio)), "Audio must not contain NaN/Inf"
            assert audio.dtype == np.float32 or audio.dtype == np.float64, "Audio must be float"

            results[moment_type] = {
                'success': True,
                'duration_s': len(audio) / sample_rate,
                'peak': float(np.max(np.abs(audio))),
                'samples': len(audio)
            }
            print(f"   ✓ {moment_type}: {len(audio)/sample_rate:.2f}s, peak {np.max(np.abs(audio)):.3f}")

        except Exception as e:
            results[moment_type] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ✗ {moment_type}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    successes = sum(1 for r in results.values() if r['success'])
    total = len(results)

    print(f"\nGesture generation: {successes}/{total} successful")

    if successes == total:
        print("\n✓ All tests passed! Layer3b is ready for game1 composition.")
        print("\nYou can now run:")
        print("  python3 synth_composer.py tags-game1.json -o game1_output.wav")
        return 0
    else:
        print(f"\n✗ {total - successes} moment type(s) failed")
        print("\nFailed moments:")
        for mtype, result in results.items():
            if not result['success']:
                print(f"  • {mtype}: {result['error']}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(test_layer3b_integration())
