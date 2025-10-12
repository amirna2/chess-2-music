"""
Test INACCURACY archetype with particle system through GestureCoordinator.

This tests the full integration path used in actual chess-to-music composition.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3b.coordinator import GestureCoordinator
from synth_engine import SubtractiveSynth


def test_inaccuracy_integration():
    """Test INACCURACY particle archetype through GestureCoordinator."""

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth = SubtractiveSynth(sample_rate=sample_rate)

    # Create coordinator (this will detect INACCURACY is particle-based)
    coordinator = GestureCoordinator(rng, synth_engine=synth)

    print("GestureCoordinator initialized")
    print(f"Available archetypes: {len(coordinator.get_available_archetypes())}")

    # Check if INACCURACY is registered
    if 'INACCURACY' in coordinator.get_available_archetypes():
        print("✓ INACCURACY archetype registered")

        # Check if it's using ParticleGestureGenerator
        generator = coordinator.gestures['INACCURACY']
        from layer3b.particle_system import ParticleGestureGenerator
        if isinstance(generator, ParticleGestureGenerator):
            print("✓ INACCURACY using ParticleGestureGenerator")
        else:
            print("✗ INACCURACY using wrong generator type")
    else:
        print("✗ INACCURACY not found in archetypes")
        return

    # Create moment event (as it would come from chess analysis)
    moment_event = {
        'type': 'INACCURACY',
        'event_type': 'INACCURACY',
        'timestamp': 0.0,
        'start_time': 0.0,
        'end_time': 5.0,
        'start_sample': 0,
        'move_number': 12,
        'eval': 0.2,
        'prev_eval': 0.5
    }

    section_context = {
        'tension': 0.4,
        'entropy': 0.3,
        'scale': 'C_MAJOR',
        'key': 'C'
    }

    print("\nGenerating INACCURACY gesture through coordinator...")
    audio = coordinator.generate_gesture(
        'INACCURACY',
        moment_event,
        section_context,
        sample_rate
    )

    print(f"✓ Generated: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
    print(f"  Peak: {np.abs(audio).max():.4f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    # Save audio
    output_path = "data/inaccuracy_particle_test.wav"
    os.makedirs("data", exist_ok=True)

    try:
        from scipy.io import wavfile
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"✓ Saved: {output_path}")
    except ImportError:
        print("Warning: scipy not available")

    # Test duration computation
    expected_duration = coordinator.compute_archetype_duration(
        'INACCURACY',
        section_context
    )
    actual_duration = len(audio) / sample_rate
    print(f"\n Duration: expected≈{expected_duration:.2f}s, actual={actual_duration:.2f}s")

    return audio


if __name__ == "__main__":
    test_inaccuracy_integration()
