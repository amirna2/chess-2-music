"""
Test script for particle system wind chime simulation.

Generates a sample wind chime gesture and saves it as a WAV file.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3b.particle_system import ParticleGestureGenerator
from layer3b.archetype_configs import ARCHETYPES
from synth_engine import SubtractiveSynth


def test_wind_chimes():
    """Generate and save wind chime audio."""

    # Audio settings
    sample_rate = 88200

    # Initialize random generator for reproducibility
    rng = np.random.default_rng(seed=42)

    # Initialize synth engine
    synth = SubtractiveSynth(sample_rate=sample_rate)

    # Get WIND_CHIMES archetype config
    archetype_config = ARCHETYPES['WIND_CHIMES']

    # Create particle gesture generator
    generator = ParticleGestureGenerator(
        archetype_config=archetype_config,
        rng=rng,
        synth_engine=synth
    )

    # Mock event and context
    moment_event = {
        'event_type': 'WIND_CHIMES',
        'timestamp': 0.0,
        'move_number': 1
    }

    section_context = {
        'tension': 0.3,
        'entropy': 0.5,
        'scale': 'major',
        'key': 'C'
    }

    print("Generating wind chime gesture...")
    audio = generator.generate_gesture(moment_event, section_context, sample_rate)

    print(f"Generated audio: {len(audio)} samples ({len(audio)/sample_rate:.2f} seconds)")
    print(f"Peak amplitude: {np.abs(audio).max():.4f}")
    print(f"RMS level: {np.sqrt(np.mean(audio**2)):.4f}")

    # Save as WAV file
    output_path = "data/wind_chimes_test.wav"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Save using scipy
    try:
        from scipy.io import wavfile
        # Convert to 16-bit PCM
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"✓ Saved audio to {output_path}")
    except ImportError:
        print("Warning: scipy not available, cannot save WAV file")
        print("Install with: pip install scipy")

    return audio


if __name__ == "__main__":
    test_wind_chimes()
