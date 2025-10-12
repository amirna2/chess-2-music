"""
Visualize particle system behavior.

Shows particle spawning timeline and emission curve.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3b.particle_system import ParticleGestureGenerator, ParticleEmitter
from layer3b.archetype_configs import ARCHETYPES
from synth_engine import SubtractiveSynth


def visualize_particle_spawning():
    """Generate particles and print spawning timeline."""

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth = SubtractiveSynth(sample_rate=sample_rate)

    archetype_config = ARCHETYPES['WIND_CHIMES']
    generator = ParticleGestureGenerator(archetype_config, rng, synth)

    # Compute duration
    section_context = {'tension': 0.3, 'entropy': 0.5}
    duration = generator._compute_duration(section_context)
    total_samples = int(duration * sample_rate)

    # Generate emission curve
    particle_config = archetype_config['particle']
    emission_curve = generator._generate_emission_curve(
        particle_config['emission'],
        total_samples,
        section_context
    )

    # Create emitter
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

    # Emit particles
    particles = emitter.emit_particles()

    print(f"Duration: {duration:.2f}s ({total_samples} samples)")
    print(f"Total particles spawned: {len(particles)}")
    print(f"\nParticle Timeline:")
    print("=" * 70)

    for i, p in enumerate(particles):
        birth_time = p.birth_sample / sample_rate
        death_time = p.death_sample / sample_rate
        lifetime = death_time - birth_time
        print(f"#{i+1:2d} | t={birth_time:5.2f}s | "
              f"pitch={p.pitch_hz:7.1f}Hz | "
              f"life={lifetime:.2f}s | "
              f"vel={p.velocity:.2f} | "
              f"detune={p.detune_cents:+.0f}¢")

    # Emission curve statistics
    print(f"\nEmission Curve Statistics:")
    print("=" * 70)
    print(f"Min density: {emission_curve.min():.4f}")
    print(f"Max density: {emission_curve.max():.4f}")
    print(f"Mean density: {emission_curve.mean():.4f}")

    # ASCII visualization of emission curve (downsampled)
    print(f"\nEmission Curve Visualization:")
    print("=" * 70)
    downsample_factor = total_samples // 60  # 60 characters wide
    if downsample_factor > 0:
        downsampled = emission_curve[::downsample_factor][:60]
        max_height = 10

        for height in range(max_height, 0, -1):
            line = ""
            threshold = height / max_height
            for val in downsampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            print(line)
        print("-" * 60)
        print(f"0s{' ' * 52}{duration:.1f}s")


if __name__ == "__main__":
    visualize_particle_spawning()
