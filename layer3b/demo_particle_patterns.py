"""
Demonstrate different particle emission patterns.

Creates multiple audio files showcasing various particle system behaviors:
- Wind gusts (current WIND_CHIMES)
- Constant rain
- Swell (crescendo)
- Decay scatter (falling debris)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3b.particle_system import ParticleGestureGenerator
from synth_engine import SubtractiveSynth


def create_test_archetype(emission_type: str, emission_params: dict) -> dict:
    """Create test archetype with specified emission pattern."""
    return {
        "duration_base": 4.5,
        "duration_tension_scale": 0.0,
        "duration_entropy_scale": 0.0,
        "particle": {
            "emission": {"type": emission_type, **emission_params},
            "base_spawn_rate": 0.001,
            "pitch_range_hz": [880, 1760],
            "lifetime_range_s": [1.0, 2.0],
            "velocity_range": [0.3, 0.7],
            "detune_range_cents": [-20, 20],
            "decay_rate_range": [-2.5, -1.5],
            "waveform": "triangle"
        },
        "peak_limit": 0.4,
        "rms_target": -28.0
    }


def generate_demo(name: str, archetype_config: dict):
    """Generate and save demo audio."""
    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth = SubtractiveSynth(sample_rate=sample_rate)

    generator = ParticleGestureGenerator(archetype_config, rng, synth)

    moment_event = {'event_type': 'DEMO', 'timestamp': 0.0, 'move_number': 1}
    section_context = {'tension': 0.5, 'entropy': 0.5, 'scale': 'major', 'key': 'C'}

    print(f"\nGenerating: {name}")
    audio = generator.generate_gesture(moment_event, section_context, sample_rate)

    output_path = f"data/particle_demo_{name}.wav"
    os.makedirs("data", exist_ok=True)

    try:
        from scipy.io import wavfile
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"✓ Saved: {output_path}")
        print(f"  Duration: {len(audio)/sample_rate:.2f}s | "
              f"Peak: {np.abs(audio).max():.3f} | "
              f"RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    except ImportError:
        print("Warning: scipy not available")


def main():
    """Generate all demo patterns."""
    print("Particle System Emission Pattern Demos")
    print("=" * 70)

    # 1. Wind gusts (current WIND_CHIMES)
    generate_demo("wind_gusts", create_test_archetype(
        "gusts",
        {"num_gusts": 2, "base_density": 0.05, "peak_density": 0.3}
    ))

    # 2. Constant rain
    generate_demo("constant_rain", create_test_archetype(
        "constant",
        {"density": 0.2}
    ))

    # 3. Swell (crescendo)
    generate_demo("swell", create_test_archetype(
        "swell",
        {"start_density": 0.05, "end_density": 0.5}
    ))

    # 4. Decay scatter (falling debris)
    generate_demo("decay_scatter", create_test_archetype(
        "decay_scatter",
        {"start_density": 0.6, "decay_rate": -2.5}
    ))

    # 5. Intense gusts (more aggressive)
    generate_demo("intense_gusts", create_test_archetype(
        "gusts",
        {"num_gusts": 3, "base_density": 0.1, "peak_density": 0.7}
    ))

    print("\n" + "=" * 70)
    print("All demos generated successfully!")
    print("Check the data/ directory for WAV files")


if __name__ == "__main__":
    main()
