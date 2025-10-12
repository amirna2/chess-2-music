#!/usr/bin/env python3
"""
Particle System Analysis Tool

Analyzes and visualizes particle-based gesture archetypes to understand:
1. How many particles spawn (density)
2. When they spawn over time (temporal distribution)
3. Their musical characteristics (pitch, velocity, lifetime)
4. The emission curve behavior (spawning probability over time)

This helps validate archetype configurations and debug particle system behavior.

Usage:
    python particle_test.py [ARCHETYPE_NAME]

Examples:
    python particle_test.py INACCURACY
    python particle_test.py FIRST_EXCHANGE
    python particle_test.py --list
"""

import numpy as np
import sys
import os
import argparse
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3b.particle_system import ParticleGestureGenerator, ParticleEmitter
from layer3b.archetype_configs import ARCHETYPES
from synth_engine import SubtractiveSynth


def format_time(seconds):
    """Format seconds as human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def analyze_particles(archetype_name, seed=42, generate_audio=False, output_file=None):
    """
    Comprehensive particle system analysis.

    Args:
        archetype_name: Name of archetype to analyze
        seed: Random seed for reproducibility
        generate_audio: If True, generate and save audio WAV file
        output_file: Optional custom output filename
    """

    # Validate archetype
    if archetype_name not in ARCHETYPES:
        print(f"❌ Error: Unknown archetype '{archetype_name}'")
        print(f"\nAvailable archetypes:")
        for name in sorted(ARCHETYPES.keys()):
            print(f"  - {name}")
        sys.exit(1)

    archetype_config = ARCHETYPES[archetype_name]

    # Check if this is a particle archetype
    if 'particle' not in archetype_config:
        print(f"❌ Error: {archetype_name} is not a particle-based archetype.")
        print(f"   This tool only works with particle archetypes.")
        print(f"\n   Use test_gesture_sound.py for curve-based archetypes.")
        sys.exit(1)

    sample_rate = 88200
    rng = np.random.default_rng(seed=seed)
    synth = SubtractiveSynth(sample_rate=sample_rate)

    generator = ParticleGestureGenerator(archetype_config, rng, synth)

    # Generate particles
    section_context = {'tension': 0.3, 'entropy': 0.5}
    duration = generator._compute_duration(section_context)
    total_samples = int(duration * sample_rate)

    particle_config = archetype_config['particle']
    emission_curve = generator._generate_emission_curve(
        particle_config['emission'],
        total_samples,
        section_context
    )

    emitter = ParticleEmitter(
        emission_curve=emission_curve,
        base_spawn_rate=particle_config.get('base_spawn_rate', 0.001),
        pitch_range_hz=tuple(particle_config['pitch_range_hz']),
        lifetime_range_samples=(
            int(particle_config['lifetime_range_s'][0] * sample_rate),
            int(particle_config['lifetime_range_s'][1] * sample_rate)
        ),
        velocity_range=tuple(particle_config.get('velocity_range', [0.3, 0.8])),
        detune_range_cents=tuple(particle_config.get('detune_range_cents', [-20, 20])),
        decay_rate_range=tuple(particle_config.get('decay_rate_range', [-3.0, -1.5])),
        waveform=particle_config.get('waveform', 'sine'),
        rng=rng
    )

    particles = emitter.emit_particles()

    # Print analysis header
    print("\n" + "="*80)
    print(f"PARTICLE SYSTEM ANALYSIS: {archetype_name}")
    print("="*80)

    # Overview
    print(f"\n📊 OVERVIEW")
    print(f"   Duration:          {format_time(duration)}")
    print(f"   Total Particles:   {len(particles)}")
    print(f"   Spawn Density:     {len(particles)/duration:.1f} particles/second")
    print(f"   Waveform:          {particle_config.get('waveform', 'sine')}")

    # Musical characteristics
    if particles:
        pitches = [p.pitch_hz for p in particles]
        velocities = [p.velocity for p in particles]
        lifetimes = [(p.death_sample - p.birth_sample) / sample_rate for p in particles]

        print(f"\n🎵 MUSICAL CHARACTERISTICS")
        print(f"   Pitch Range:       {min(pitches):.0f} Hz - {max(pitches):.0f} Hz")
        print(f"   Mean Pitch:        {np.mean(pitches):.0f} Hz")
        print(f"   Velocity Range:    {min(velocities):.2f} - {max(velocities):.2f}")
        print(f"   Mean Velocity:     {np.mean(velocities):.2f}")
        print(f"   Lifetime Range:    {format_time(min(lifetimes))} - {format_time(max(lifetimes))}")
        print(f"   Mean Lifetime:     {format_time(np.mean(lifetimes))}")

    # Emission curve analysis
    print(f"\n📈 EMISSION CURVE (spawning probability over time)")
    print(f"   Min Density:       {emission_curve.min():.4f}")
    print(f"   Max Density:       {emission_curve.max():.4f}")
    print(f"   Mean Density:      {emission_curve.mean():.4f}")
    print(f"   Peak Location:     {format_time((np.argmax(emission_curve) / sample_rate))}")

    # Temporal distribution visualization
    print(f"\n⏱️  TEMPORAL DISTRIBUTION (when particles spawn)")
    print(f"   " + "─"*60)

    # Create histogram of spawn times
    if particles:
        time_bins = 20
        spawn_times = [p.birth_sample / sample_rate for p in particles]
        hist, bin_edges = np.histogram(spawn_times, bins=time_bins, range=(0, duration))

        max_count = hist.max()
        bar_width = 50

        for i, count in enumerate(hist):
            time_start = bin_edges[i]
            time_end = bin_edges[i+1]

            if max_count > 0:
                bar_len = int((count / max_count) * bar_width)
            else:
                bar_len = 0

            bar = "█" * bar_len
            print(f"   {format_time(time_start):>6s}-{format_time(time_end):<6s} │{bar} {count}")

        print(f"   " + "─"*60)

    # Emission curve visualization
    print(f"\n📉 EMISSION CURVE VISUALIZATION")
    print(f"   " + "─"*60)

    downsample_factor = max(1, total_samples // 60)
    downsampled = emission_curve[::downsample_factor][:60]

    # Normalize for visualization
    min_val = downsampled.min()
    max_val = downsampled.max()

    if max_val > min_val:
        normalized = (downsampled - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(downsampled)

    max_height = 10

    for height in range(max_height, 0, -1):
        line = "   "
        threshold = height / max_height
        for val in normalized:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        print(line)

    print(f"   " + "─"*60)
    print(f"   0s{' '*52}{format_time(duration)}")
    print(f"   (showing normalized curve from {min_val:.4f} to {max_val:.4f})")

    # Sample particles (first 10)
    if particles:
        print(f"\n🔍 SAMPLE PARTICLES (first 10)")
        print(f"   {'#':<4} {'Spawn':>8} {'Pitch':>9} {'Life':>8} {'Vel':>5} {'Detune':>7}")
        print(f"   " + "─"*60)

        for i, p in enumerate(particles[:10]):
            birth_time = p.birth_sample / sample_rate
            lifetime = (p.death_sample - p.birth_sample) / sample_rate
            print(f"   {i+1:<4d} {format_time(birth_time):>8s} {p.pitch_hz:>8.0f}Hz "
                  f"{format_time(lifetime):>8s} {p.velocity:>5.2f} {p.detune_cents:>+6.0f}¢")

        if len(particles) > 10:
            print(f"   ... and {len(particles) - 10} more particles")

    # Configuration summary
    print(f"\n⚙️  CONFIGURATION")
    print(f"   Base Spawn Rate:   {particle_config.get('base_spawn_rate', 0.001)}")
    print(f"   Pitch Range:       {particle_config['pitch_range_hz'][0]:.0f}-{particle_config['pitch_range_hz'][1]:.0f} Hz")
    print(f"   Lifetime Range:    {particle_config['lifetime_range_s'][0]:.2f}-{particle_config['lifetime_range_s'][1]:.2f} s")
    print(f"   Velocity Range:    {particle_config.get('velocity_range', [0.3, 0.8])}")
    print(f"   Detune Range:      {particle_config.get('detune_range_cents', [-20, 20])} cents")
    print(f"   Decay Rate Range:  {particle_config.get('decay_rate_range', [-3.0, -1.5])}")

    # Generate audio if requested
    if generate_audio:
        print("\n" + "="*80)
        print("AUDIO GENERATION")
        print("="*80)

        print(f"\n🔊 Generating audio with {len(particles)} particles...")

        # Generate full gesture audio
        moment_event = {'type': archetype_name, 'timestamp': 0.0}
        section_context_full = {'tension': 0.3, 'entropy': 0.5}
        audio = generator.generate_gesture(moment_event, section_context_full, sample_rate)

        # Audio analysis
        peak_amplitude = float(np.max(np.abs(audio)))
        peak_db = 20 * np.log10(peak_amplitude) if peak_amplitude > 0 else -np.inf
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf

        print(f"\n📈 AUDIO ANALYSIS")
        print(f"   Duration:        {format_time(len(audio) / sample_rate)}")
        print(f"   Samples:         {len(audio)}")
        print(f"   Peak level:      {peak_amplitude:.4f} ({peak_db:.1f} dBFS)")
        print(f"   RMS level:       {rms:.4f} ({rms_db:.1f} dBFS)")

        # Warnings
        if peak_db > -6.0:
            print(f"   ⚠️  WARNING: Very loud! Peak {peak_db:.1f} dBFS")
        elif peak_db > -12.0:
            print(f"   ⚠️  Loud. Consider reducing levels.")

        # Save to WAV
        if output_file is None:
            output_file = f"particle_{archetype_name.lower()}.wav"

        print(f"\n💾 SAVING AUDIO")
        print(f"   Output file: {output_file}")

        with wave.open(output_file, 'w') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)

            # Convert to 16-bit PCM
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            wav.writeframes(audio_int16.tobytes())

        print(f"\n✓ Audio saved! Listen to: {output_file}")

    print("\n" + "="*80 + "\n")


def list_archetypes():
    """List all available particle-based archetypes."""
    print("\n" + "="*80)
    print("PARTICLE-BASED ARCHETYPES")
    print("="*80 + "\n")

    particle_archetypes = []
    for name in sorted(ARCHETYPES.keys()):
        config = ARCHETYPES[name]
        if 'particle' in config:
            particle_archetypes.append(name)
            print(f"  {name}")

    if not particle_archetypes:
        print("  (No particle archetypes found)")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze and visualize particle system archetypes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s INACCURACY                      Analyze INACCURACY archetype
  %(prog)s FIRST_EXCHANGE --audio          Analyze and generate audio
  %(prog)s INACCURACY --audio -o test.wav  Custom output filename
  %(prog)s --list                          List all available archetypes
        """
    )
    parser.add_argument(
        'archetype',
        nargs='?',
        help='Archetype name to analyze'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available archetypes'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--audio', '-a',
        action='store_true',
        help='Generate audio WAV file in addition to analysis'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output WAV filename (default: particle_<archetype>.wav)'
    )

    args = parser.parse_args()

    if args.list:
        list_archetypes()
    elif args.archetype:
        analyze_particles(
            args.archetype,
            seed=args.seed,
            generate_audio=args.audio,
            output_file=args.output
        )
    else:
        parser.print_help()
        sys.exit(1)
