"""
Test all game1 particle archetypes individually.

Generates test audio for each archetype used in tags-game1.json.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3b.coordinator import GestureCoordinator
from synth_engine import SubtractiveSynth


def test_archetype(name: str, coordinator: GestureCoordinator, sample_rate: int):
    """Generate test audio for a single archetype."""

    moment_event = {
        'type': name,
        'event_type': name,
        'timestamp': 0.0,
        'move_number': 10
    }

    section_context = {
        'tension': 0.5,
        'entropy': 0.5,
        'scale': 'C_MAJOR',
        'key': 'C'
    }

    print(f"\n{name}:")
    print("=" * 60)

    audio = coordinator.generate_gesture(name, moment_event, section_context, sample_rate)

    print(f"  Duration: {len(audio)/sample_rate:.2f}s")
    print(f"  Peak: {np.abs(audio).max():.4f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    # Save audio
    output_path = f"data/game1_test_{name.lower()}.wav"

    try:
        from scipy.io import wavfile
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"  ✓ Saved: {output_path}")
    except ImportError:
        print("  Warning: scipy not available")

    return audio


def main():
    """Test all game1 archetypes."""

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth = SubtractiveSynth(sample_rate=sample_rate)
    coordinator = GestureCoordinator(rng, synth)

    print("Testing game1 Particle Archetypes")
    print("=" * 60)

    # Test each archetype used in tags-game1.json
    archetypes = [
        'INACCURACY',
        'FIRST_EXCHANGE',
        'TACTICAL_SEQUENCE',
        'SIGNIFICANT_SHIFT',
        'FINAL_RESOLUTION'
    ]

    for archetype in archetypes:
        test_archetype(archetype, coordinator, sample_rate)

    print("\n" + "=" * 60)
    print(f"✅ All {len(archetypes)} archetypes tested successfully!")
    print("Check data/ directory for WAV files")


if __name__ == "__main__":
    main()
