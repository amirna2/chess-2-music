#!/usr/bin/env python3
"""
Test Layer3b implementations with game1 moments.

Quick validation that all curve generators work correctly
for the specific archetypes used in game1.
"""

import numpy as np
from layer3b.archetype_configs import ARCHETYPES
from layer3b.curve_generators import (
    generate_pitch_curve,
    generate_harmony,
    generate_filter_curve,
    generate_envelope
)
from layer3b.utils import compute_phases


def test_archetype(archetype_name: str, sample_rate: int = 88200):
    """Test a single archetype's curve generation."""
    print(f"\nTesting archetype: {archetype_name}")

    if archetype_name not in ARCHETYPES:
        print(f"  ✗ Unknown archetype: {archetype_name}")
        return False

    config = ARCHETYPES[archetype_name]
    rng = np.random.default_rng(42)

    # Test context
    section_context = {
        'tension': 0.7,
        'entropy': 0.5,
        'scale': 'C_MAJOR',
        'key': 'C'
    }

    # Compute duration and phases
    duration_base = config['duration_base']
    duration_tension = config.get('duration_tension_scale', 0.0) * section_context['tension']
    duration_entropy = config.get('duration_entropy_scale', 0.0) * section_context['entropy']
    duration = max(0.5, min(10.0, duration_base + duration_tension + duration_entropy))
    total_samples = int(duration * sample_rate)

    phases = compute_phases(
        config['phases'],
        total_samples,
        section_context,
        rng
    )

    try:
        # Test pitch generation
        print(f"  Testing pitch type: {config['pitch']['type']}")
        pitch_curve = generate_pitch_curve(
            config['pitch'],
            phases,
            section_context,
            total_samples,
            rng,
            sample_rate
        )
        assert len(pitch_curve) == total_samples, f"Pitch curve length mismatch"
        assert np.all(np.isfinite(pitch_curve)), f"Pitch curve contains NaN/Inf"
        assert np.all(pitch_curve > 0), f"Pitch curve contains non-positive frequencies"
        print(f"    ✓ Pitch: {len(pitch_curve)} samples, range {pitch_curve.min():.1f}-{pitch_curve.max():.1f} Hz")

        # Test harmony generation
        print(f"  Testing harmony type: {config['harmony']['type']}")
        harmony_voices = generate_harmony(
            config['harmony'],
            pitch_curve,
            phases,
            section_context,
            rng
        )
        assert len(harmony_voices) > 0, f"No harmony voices generated"
        for i, voice in enumerate(harmony_voices):
            assert len(voice) == total_samples, f"Voice {i} length mismatch"
            assert np.all(np.isfinite(voice)), f"Voice {i} contains NaN/Inf"
        print(f"    ✓ Harmony: {len(harmony_voices)} voices")

        # Test filter generation
        print(f"  Testing filter type: {config['filter']['type']}")
        filter_curve = generate_filter_curve(
            config['filter'],
            phases,
            section_context,
            total_samples,
            rng,
            sample_rate
        )
        assert 'cutoff' in filter_curve, f"Filter missing 'cutoff' key"
        assert 'resonance' in filter_curve, f"Filter missing 'resonance' key"
        assert len(filter_curve['cutoff']) == total_samples, f"Filter cutoff length mismatch"
        assert len(filter_curve['resonance']) == total_samples, f"Filter resonance length mismatch"
        print(f"    ✓ Filter: cutoff {filter_curve['cutoff'].min():.1f}-{filter_curve['cutoff'].max():.1f} Hz")

        # Test envelope generation
        print(f"  Testing envelope type: {config['envelope']['type']}")
        envelope = generate_envelope(
            config['envelope'],
            phases,
            total_samples,
            rng,
            sample_rate
        )
        assert len(envelope) == total_samples, f"Envelope length mismatch"
        assert np.all(np.isfinite(envelope)), f"Envelope contains NaN/Inf"
        assert np.all(envelope >= 0), f"Envelope contains negative values"
        assert np.all(envelope <= 1.0), f"Envelope exceeds 1.0"
        print(f"    ✓ Envelope: peak {envelope.max():.3f}")

        print(f"  ✓ {archetype_name} passed all tests")
        return True

    except Exception as e:
        print(f"  ✗ {archetype_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all archetypes used in game1."""
    print("=" * 60)
    print("Layer3b Game1 Archetype Validation")
    print("=" * 60)

    # Archetypes used in game1
    game1_archetypes = [
        'INACCURACY',
        'FIRST_EXCHANGE',
        'TACTICAL_SEQUENCE',
        'SIGNIFICANT_SHIFT',
        'FINAL_RESOLUTION'
    ]

    results = {}
    for archetype in game1_archetypes:
        results[archetype] = test_archetype(archetype)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    for archetype, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {archetype}")

    if passed == total:
        print("\n✓ All game1 archetypes validated successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} archetype(s) failed validation")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
