#!/usr/bin/env python3
"""
Quick test of GestureSynthesizer to verify it works with SubtractiveSynth.
"""

import numpy as np
from synth_engine import SubtractiveSynth
from layer3b.synthesizer import GestureSynthesizer


def test_basic_synthesis():
    """Test basic gesture synthesis."""
    print("Testing GestureSynthesizer...")

    # Create synth engine
    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)

    # Create gesture synthesizer
    gesture_synth = GestureSynthesizer(synth_engine)

    # Create test parameter curves (1 second duration)
    duration = 1.0
    num_samples = int(duration * sample_rate)

    # Pitch: simple glissando from 440 Hz to 220 Hz (exponential)
    pitch_curve = np.exp(np.linspace(np.log(440), np.log(220), num_samples))

    # 3 voices with slight detuning
    pitch_voices = [
        pitch_curve,
        pitch_curve * (2 ** (4 / 12)),   # Major third above
        pitch_curve * (2 ** (7 / 12))    # Perfect fifth above
    ]

    # Filter: sweep from 1000 Hz to 200 Hz
    filter_curve = {
        'cutoff': np.exp(np.linspace(np.log(1000), np.log(200), num_samples)),
        'resonance': np.full(num_samples, 0.5),
        'type': 'lowpass'
    }

    # Envelope: simple attack-decay
    envelope = np.zeros(num_samples)
    attack_samples = int(0.05 * sample_rate)  # 50ms attack
    decay_samples = num_samples - attack_samples
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:] = np.exp(np.linspace(0, -4, decay_samples))

    # Texture: 20% pink noise
    texture_curve = {
        'noise_ratio': 0.2,
        'noise_type': 'pink',
        'shimmer_enable': False
    }

    # Synthesize
    print(f"  Synthesizing {duration}s gesture with {len(pitch_voices)} voices...")
    audio = gesture_synth.synthesize(
        pitch_voices=pitch_voices,
        filter_curve=filter_curve,
        envelope=envelope,
        texture_curve=texture_curve,
        sample_rate=sample_rate
    )

    # Verify output
    assert len(audio) == num_samples, f"Output length mismatch: {len(audio)} != {num_samples}"
    assert audio.dtype == np.float64, f"Output dtype should be float64, got {audio.dtype}"

    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    print(f"  Peak level: {peak:.4f}")
    print(f"  RMS level: {rms:.4f}")
    print(f"  Dynamic range: {20 * np.log10(peak / (rms + 1e-10)):.2f} dB")

    # Sanity checks
    assert peak < 1.0, f"Clipping detected: peak={peak}"
    assert peak > 0.01, f"Signal too quiet: peak={peak}"
    assert rms > 0.001, f"RMS too low: rms={rms}"

    print("  ✓ Basic synthesis test passed")


def test_shimmer_effect():
    """Test shimmer effect."""
    print("\nTesting shimmer effect...")

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    gesture_synth = GestureSynthesizer(synth_engine)

    duration = 0.5
    num_samples = int(duration * sample_rate)

    # Single voice at 440 Hz
    pitch_voices = [np.full(num_samples, 440.0)]

    # No filter (passthrough)
    filter_curve = {
        'cutoff': np.full(num_samples, 20000.0),
        'resonance': np.full(num_samples, 0.1),
        'type': 'lowpass'
    }

    # Flat envelope
    envelope = np.ones(num_samples)

    # Texture with shimmer
    texture_curve = {
        'noise_ratio': 0.0,
        'noise_type': 'white',
        'shimmer_enable': True,
        'shimmer_rate_hz': 4.0  # 4 Hz shimmer
    }

    audio = gesture_synth.synthesize(
        pitch_voices=pitch_voices,
        filter_curve=filter_curve,
        envelope=envelope,
        texture_curve=texture_curve,
        sample_rate=sample_rate
    )

    # Verify shimmer creates amplitude modulation
    # Extract amplitude envelope from signal
    window_size = int(sample_rate / 100)  # 10ms window
    amplitude_curve = np.array([
        np.max(np.abs(audio[i:i+window_size]))
        for i in range(0, len(audio) - window_size, window_size)
    ])

    # Check for periodic variation
    amplitude_range = np.max(amplitude_curve) - np.min(amplitude_curve)
    print(f"  Amplitude range: {amplitude_range:.4f}")
    assert amplitude_range > 0.3, "Shimmer should create significant amplitude variation"

    print("  ✓ Shimmer effect test passed")


def test_noise_mixing():
    """Test noise texture mixing."""
    print("\nTesting noise mixing...")

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    gesture_synth = GestureSynthesizer(synth_engine)

    duration = 0.2
    num_samples = int(duration * sample_rate)

    pitch_voices = [np.full(num_samples, 440.0)]

    filter_curve = {
        'cutoff': np.full(num_samples, 20000.0),
        'resonance': np.full(num_samples, 0.1),
        'type': 'lowpass'
    }

    envelope = np.ones(num_samples)

    # Test different noise ratios
    for noise_ratio in [0.0, 0.5, 1.0]:
        texture_curve = {
            'noise_ratio': noise_ratio,
            'noise_type': 'white',
            'shimmer_enable': False
        }

        audio = gesture_synth.synthesize(
            pitch_voices=pitch_voices,
            filter_curve=filter_curve,
            envelope=envelope,
            texture_curve=texture_curve,
            sample_rate=sample_rate
        )

        print(f"  Noise ratio {noise_ratio:.1f}: RMS={np.sqrt(np.mean(audio**2)):.4f}")

    print("  ✓ Noise mixing test passed")


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    gesture_synth = GestureSynthesizer(synth_engine)

    num_samples = 1000

    # Test sample rate mismatch
    try:
        gesture_synth.synthesize(
            pitch_voices=[np.full(num_samples, 440.0)],
            filter_curve={'cutoff': np.full(num_samples, 1000.0),
                         'resonance': np.full(num_samples, 0.5),
                         'type': 'lowpass'},
            envelope=np.ones(num_samples),
            texture_curve={'noise_ratio': 0.0},
            sample_rate=44100  # Wrong!
        )
        assert False, "Should raise ValueError for sample rate mismatch"
    except ValueError as e:
        assert "Sample rate mismatch" in str(e)
        print("  ✓ Sample rate mismatch detected")

    # Test empty pitch voices
    try:
        gesture_synth.synthesize(
            pitch_voices=[],
            filter_curve={'cutoff': np.full(num_samples, 1000.0),
                         'resonance': np.full(num_samples, 0.5),
                         'type': 'lowpass'},
            envelope=np.ones(num_samples),
            texture_curve={'noise_ratio': 0.0},
            sample_rate=sample_rate
        )
        assert False, "Should raise ValueError for empty pitch voices"
    except ValueError as e:
        assert "No pitch voices" in str(e)
        print("  ✓ Empty pitch voices detected")

    # Test length mismatch
    try:
        gesture_synth.synthesize(
            pitch_voices=[np.full(num_samples, 440.0), np.full(500, 440.0)],  # Different lengths!
            filter_curve={'cutoff': np.full(num_samples, 1000.0),
                         'resonance': np.full(num_samples, 0.5),
                         'type': 'lowpass'},
            envelope=np.ones(num_samples),
            texture_curve={'noise_ratio': 0.0},
            sample_rate=sample_rate
        )
        assert False, "Should raise ValueError for length mismatch"
    except ValueError as e:
        assert "length" in str(e)
        print("  ✓ Pitch curve length mismatch detected")

    print("  ✓ Error handling test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("GestureSynthesizer Test Suite")
    print("=" * 60)

    test_basic_synthesis()
    test_shimmer_effect()
    test_noise_mixing()
    test_error_handling()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
