#!/usr/bin/env python3
"""
Layer3b Gesture Sound Tester

Interactive tool to test individual gesture archetypes and diagnose synthesis issues.
Generates isolated gesture audio with detailed parameter analysis.

Usage:
    python3 gesture_test.py INACCURACY
    python3 gesture_test.py FIRST_EXCHANGE --tension 0.8
    python3 gesture_test.py --list  # List all archetypes
"""

import sys
import os
import argparse
import numpy as np
import wave
import struct

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FORCE FRESH IMPORT - clear any cached modules
for module_name in list(sys.modules.keys()):
    if 'layer3b' in module_name:
        del sys.modules[module_name]

from layer3b import GestureCoordinator, ARCHETYPES
from synth_engine import SubtractiveSynth


def analyze_audio(audio, sample_rate):
    """Analyze audio and return diagnostic info."""
    analysis = {}

    # Level analysis
    analysis['peak_amplitude'] = float(np.max(np.abs(audio)))
    analysis['peak_db'] = 20 * np.log10(analysis['peak_amplitude']) if analysis['peak_amplitude'] > 0 else -np.inf

    rms = np.sqrt(np.mean(audio ** 2))
    analysis['rms_amplitude'] = float(rms)
    analysis['rms_db'] = 20 * np.log10(rms) if rms > 0 else -np.inf

    analysis['crest_factor_db'] = analysis['peak_db'] - analysis['rms_db']

    # Spectral analysis (simple)
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

    # Find dominant frequency
    dominant_idx = np.argmax(magnitude)
    analysis['dominant_freq_hz'] = float(freqs[dominant_idx])

    # Spectral centroid (brightness measure)
    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    analysis['spectral_centroid_hz'] = float(spectral_centroid)

    # Energy distribution
    total_energy = np.sum(magnitude ** 2)
    low_band = magnitude[freqs < 500]
    mid_band = magnitude[(freqs >= 500) & (freqs < 2000)]
    high_band = magnitude[freqs >= 2000]

    analysis['energy_low_pct'] = float(np.sum(low_band ** 2) / total_energy * 100) if total_energy > 0 else 0
    analysis['energy_mid_pct'] = float(np.sum(mid_band ** 2) / total_energy * 100) if total_energy > 0 else 0
    analysis['energy_high_pct'] = float(np.sum(high_band ** 2) / total_energy * 100) if total_energy > 0 else 0

    # Temporal analysis
    analysis['duration_s'] = len(audio) / sample_rate

    # Envelope analysis (simplified)
    envelope = np.abs(audio)
    attack_samples = int(0.05 * sample_rate)  # First 50ms
    if len(envelope) > attack_samples:
        analysis['attack_peak'] = float(np.max(envelope[:attack_samples]))
    else:
        analysis['attack_peak'] = float(np.max(envelope))

    # Decay analysis
    if len(envelope) > attack_samples:
        decay_portion = envelope[attack_samples:]
        if len(decay_portion) > 0:
            analysis['decay_final'] = float(decay_portion[-1])
            analysis['decay_ratio'] = analysis['decay_final'] / analysis['attack_peak'] if analysis['attack_peak'] > 0 else 0

    return analysis


def test_gesture(archetype_name, tension=0.7, entropy=0.5, sample_rate=88200, output_file=None):
    """Test a single gesture archetype."""

    if archetype_name not in ARCHETYPES:
        print(f"❌ Unknown archetype: {archetype_name}")
        print(f"\nAvailable archetypes: {', '.join(sorted(ARCHETYPES.keys()))}")
        return 1

    config = ARCHETYPES[archetype_name]

    # Check if this is a particle-based archetype (use explicit system field)
    system_type = config.get('system', 'curve')  # Default to curve for backward compat
    if system_type == 'particle':
        print(f"❌ {archetype_name} is a particle-based archetype.")
        print(f"   Use particle_test.py with --audio flag for particle archetypes.")
        print(f"\n   Example: python3 tools/particle_test.py {archetype_name} --audio")
        return 1

    print("=" * 80)
    print(f"LAYER3B GESTURE TEST: {archetype_name}")
    print("=" * 80)

    # Print archetype configuration
    print(f"\n⏱  DURATION")
    print(f"   Base: {config['duration_base']:.2f}s")
    print(f"   Tension scale: {config.get('duration_tension_scale', 0):.2f}")
    print(f"   Entropy scale: {config.get('duration_entropy_scale', 0):.2f}")

    print(f"\n🎵 PITCH ({config['pitch']['type']})")
    for key, value in config['pitch'].items():
        if key != 'type':
            print(f"   {key}: {value}")

    print(f"\n🎶 HARMONY ({config['harmony']['type']})")
    for key, value in config['harmony'].items():
        if key != 'type':
            print(f"   {key}: {value}")

    print(f"\n🎛  FILTER ({config['filter']['type']})")
    for key, value in config['filter'].items():
        if key != 'type':
            print(f"   {key}: {value}")

    print(f"\n📊 ENVELOPE ({config['envelope']['type']})")
    for key, value in config['envelope'].items():
        if key != 'type':
            print(f"   {key}: {value}")

    print(f"\n🌫  TEXTURE")
    print(f"   Noise ratio: {config['texture']['noise_ratio_base']:.2%}")
    print(f"   Noise type: {config['texture']['noise_type']}")
    print(f"   Shimmer: {config['texture'].get('shimmer_enable', False)}")
    print(f"   Waveform: {config['texture'].get('waveform', 'NOT SET - defaults to sine!')}")

    print(f"\n🔊 LEVELS")
    print(f"   Peak limit: {config['peak_limit']:.2f}")
    print(f"   RMS target: {config['rms_target']:.1f} dB")

    # Initialize synthesis
    print(f"\n🔧 SYNTHESIS")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Test context: tension={tension:.2f}, entropy={entropy:.2f}")

    rng = np.random.default_rng(42)
    synth = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    coordinator = GestureCoordinator(rng, synth_engine=synth)

    section_context = {
        'tension': tension,
        'entropy': entropy,
        'scale': 'C_MAJOR',
        'key': 'C'
    }

    test_event = {
        'type': archetype_name,
        'timestamp': 0.0,
        'move_number': 1,
        'score': 5
    }

    # Generate gesture with debug tracing
    print(f"   Generating audio...")

    # Monkey patch to trace waveform usage
    original_osc = synth.oscillator_timevarying_pitch
    waveforms_used = []
    def trace_osc(pitch_curve, waveform='sine'):
        waveforms_used.append(waveform)
        return original_osc(pitch_curve, waveform=waveform)
    synth.oscillator_timevarying_pitch = trace_osc

    audio = coordinator.generate_gesture(
        archetype_name,
        test_event,
        section_context,
        sample_rate
    )

    # Restore original
    synth.oscillator_timevarying_pitch = original_osc

    print(f"   🔍 DEBUG: Waveforms actually used in synthesis: {set(waveforms_used)}")

    # Analyze
    print(f"\n📈 AUDIO ANALYSIS")
    analysis = analyze_audio(audio, sample_rate)

    print(f"\n   Duration: {analysis['duration_s']:.3f}s ({len(audio)} samples)")
    print(f"\n   LEVELS:")
    print(f"      Peak amplitude: {analysis['peak_amplitude']:.4f} ({analysis['peak_db']:.1f} dBFS)")
    print(f"      RMS amplitude: {analysis['rms_amplitude']:.4f} ({analysis['rms_db']:.1f} dBFS)")
    print(f"      Crest factor: {analysis['crest_factor_db']:.1f} dB")

    print(f"\n   FREQUENCY CONTENT:")
    print(f"      Dominant frequency: {analysis['dominant_freq_hz']:.1f} Hz")
    print(f"      Spectral centroid: {analysis['spectral_centroid_hz']:.1f} Hz (brightness)")

    print(f"\n   ENERGY DISTRIBUTION:")
    print(f"      Low (<500 Hz): {analysis['energy_low_pct']:.1f}%")
    print(f"      Mid (500-2000 Hz): {analysis['energy_mid_pct']:.1f}%")
    print(f"      High (>2000 Hz): {analysis['energy_high_pct']:.1f}%")

    print(f"\n   ENVELOPE:")
    print(f"      Attack peak (first 50ms): {analysis['attack_peak']:.4f}")
    if 'decay_final' in analysis:
        print(f"      Decay final: {analysis['decay_final']:.4f}")
        print(f"      Decay ratio: {analysis['decay_ratio']:.2%}")

    # Diagnostic warnings
    print(f"\n⚠️  DIAGNOSTICS")
    warnings = []

    if analysis['peak_db'] > -6.0:
        warnings.append(f"❌ VERY LOUD! Peak {analysis['peak_db']:.1f} dBFS (should be < -6 dBFS)")
    elif analysis['peak_db'] > -12.0:
        warnings.append(f"⚠️  Loud. Peak {analysis['peak_db']:.1f} dBFS (consider reducing)")
    else:
        warnings.append(f"✓ Peak level OK ({analysis['peak_db']:.1f} dBFS)")

    if analysis['rms_db'] > -18.0:
        warnings.append(f"❌ RMS TOO HOT! {analysis['rms_db']:.1f} dBFS (should be < -18 dBFS)")
    elif analysis['rms_db'] > -24.0:
        warnings.append(f"⚠️  RMS moderately loud ({analysis['rms_db']:.1f} dBFS)")
    else:
        warnings.append(f"✓ RMS level appropriate ({analysis['rms_db']:.1f} dBFS)")

    if analysis['spectral_centroid_hz'] > 2000:
        warnings.append(f"⚠️  Very bright/harsh (centroid {analysis['spectral_centroid_hz']:.0f} Hz)")
    elif analysis['spectral_centroid_hz'] < 300:
        warnings.append(f"⚠️  Very dark/muffled (centroid {analysis['spectral_centroid_hz']:.0f} Hz)")
    else:
        warnings.append(f"✓ Spectral balance OK (centroid {analysis['spectral_centroid_hz']:.0f} Hz)")

    if analysis['energy_high_pct'] > 40:
        warnings.append(f"❌ TOO MUCH HIGH FREQUENCY! {analysis['energy_high_pct']:.0f}% (harsh/screechy)")

    if analysis['energy_low_pct'] > 60:
        warnings.append(f"⚠️  Bass-heavy: {analysis['energy_low_pct']:.0f}% low energy")

    if analysis['crest_factor_db'] < 6:
        warnings.append(f"⚠️  Low crest factor ({analysis['crest_factor_db']:.1f} dB) - may sound compressed")

    for warning in warnings:
        print(f"   {warning}")

    # Save audio
    if output_file is None:
        output_file = f"gesture_test_{archetype_name.lower()}.wav"

    print(f"\n💾 SAVING AUDIO")
    print(f"   Output: {output_file}")

    # Write WAV file
    with wave.open(output_file, 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        # Convert to 16-bit PCM
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wav.writeframes(audio_int16.tobytes())

    print(f"\n✓ Test complete! Listen to: {output_file}")
    print("=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Test individual Layer3b gesture archetypes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 gesture_test.py INACCURACY
  python3 gesture_test.py FIRST_EXCHANGE --tension 0.9
  python3 gesture_test.py BRILLIANT -o brilliant_test.wav
  python3 gesture_test.py --list
        """
    )

    parser.add_argument('archetype', nargs='?', help='Archetype name (e.g., INACCURACY)')
    parser.add_argument('--list', '-l', action='store_true', help='List all available archetypes')
    parser.add_argument('--tension', '-t', type=float, default=0.7, help='Tension value 0.0-1.0 (default: 0.7)')
    parser.add_argument('--entropy', '-e', type=float, default=0.5, help='Entropy value 0.0-1.0 (default: 0.5)')
    parser.add_argument('--sample-rate', '-r', type=int, default=88200, help='Sample rate (default: 88200)')
    parser.add_argument('--output', '-o', help='Output WAV file (default: gesture_test_<name>.wav)')

    args = parser.parse_args()

    if args.list:
        print("Available Layer3b Archetypes (Curve-based only):")
        print("=" * 80)
        for name in sorted(ARCHETYPES.keys()):
            config = ARCHETYPES[name]
            if 'particle' not in config:
                print(f"  {name}")
        print("\nNote: Particle-based archetypes excluded. Use visualize_particles.py --audio for those.")
        return 0

    if not args.archetype:
        parser.print_help()
        return 1

    return test_gesture(
        args.archetype.upper(),
        tension=args.tension,
        entropy=args.entropy,
        sample_rate=args.sample_rate,
        output_file=args.output
    )


if __name__ == '__main__':
    sys.exit(main())
