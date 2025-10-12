#!/usr/bin/env python3
"""
Test individual gesture archetypes in isolation.
Quick script to generate and play single gestures for testing.
"""

import numpy as np
import sys
from layer3b.coordinator import GestureCoordinator
from synth_engine import SubtractiveSynth
import scipy.io.wavfile as wav


def test_single_gesture(archetype_name, seed=42):
    """Generate a single gesture and save to WAV."""

    print(f"\n{'='*60}")
    print(f"Testing: {archetype_name}")
    print(f"{'='*60}")

    # Create synth and coordinator
    rng = np.random.default_rng(seed=seed)
    sample_rate = 88200
    synth = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    coordinator = GestureCoordinator(rng, synth_engine=synth)

    # Get archetype config
    from layer3b.archetype_configs import ARCHETYPES
    config = ARCHETYPES.get(archetype_name)

    if not config:
        print(f"❌ Unknown archetype: {archetype_name}")
        print(f"Available: {', '.join(coordinator.get_available_archetypes())}")
        return

    # Print config
    print(f"\nConfiguration:")
    print(f"  duration_base: {config.get('duration_base', 'N/A')}s")

    if 'particle' in config:
        particle = config['particle']
        print(f"  🔊 PARTICLE SYSTEM:")
        print(f"    emission: {particle['emission']['type']}")
        print(f"    pitch_range: {particle['pitch_range_hz'][0]}-{particle['pitch_range_hz'][1]} Hz")
        print(f"    lifetime: {particle['lifetime_range_s'][0]}-{particle['lifetime_range_s'][1]}s")
        print(f"    velocity: {particle['velocity_range'][0]}-{particle['velocity_range'][1]}")
        print(f"    decay_rate: {particle['decay_rate_range'][0]} to {particle['decay_rate_range'][1]}")
        print(f"    waveform: {particle.get('waveform', 'sine')}")
    else:
        print(f"  Traditional curve-based gesture")

    # Create moment event
    moment_event = {
        'event_type': archetype_name,
        'timestamp': 0.0,
        'move_number': 5
    }

    # Create section context
    section_context = {
        'tension': 0.5,
        'entropy': 0.5,
        'scale': 'C_MAJOR',
        'key': 'C'
    }

    # Generate gesture
    print(f"\nGenerating gesture...")
    audio = coordinator.generate_gesture(
        archetype_name,
        moment_event,
        section_context,
        sample_rate
    )

    # Analyze
    duration = len(audio) / sample_rate
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    non_zero = np.count_nonzero(audio)

    print(f"\nAudio Analysis:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Peak level: {peak:.4f}")
    print(f"  RMS level: {rms:.4f}")
    print(f"  Non-zero samples: {non_zero}/{len(audio)} ({100*non_zero/len(audio):.1f}%)")

    # Save
    output_file = f"data/test_isolated_{archetype_name.lower()}.wav"
    wav.write(output_file, sample_rate, audio.astype(np.float32))
    print(f"\n✓ Saved: {output_file}")
    print(f"  Play with: afplay {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_gesture_isolated.py <ARCHETYPE_NAME> [seed]")
        print("\nExamples:")
        print("  python3 test_gesture_isolated.py INACCURACY")
        print("  python3 test_gesture_isolated.py FIRST_EXCHANGE")
        print("  python3 test_gesture_isolated.py TACTICAL_SEQUENCE")
        print("\nAvailable archetypes:")
        from layer3b.archetype_configs import ARCHETYPES
        for name in sorted(ARCHETYPES.keys()):
            print(f"  - {name}")
        sys.exit(1)

    archetype = sys.argv[1].upper()
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    test_single_gesture(archetype, seed=seed)
