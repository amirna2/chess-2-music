"""
Particle system for stochastic polyphonic sound gesture generation.

Enables simulation of natural phenomena like wind chimes, rain, cricket swarms,
and other textures requiring independent sound events over time.

Architecture inspired by game engine particle systems, adapted for audio synthesis.
Each particle is an independent sound event with its own lifetime, pitch, and decay.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Particle:
    """
    Individual sound event with independent lifetime and properties.

    Each particle represents one "chime strike", "raindrop", or similar
    discrete sound event that spawns, rings, and decays independently.
    """
    birth_sample: int           # When particle spawns (sample index)
    lifetime_samples: int       # How long particle rings (in samples)
    pitch_hz: float             # Base frequency
    velocity: float             # Amplitude/loudness (0-1)
    detune_cents: float         # Pitch variation in cents (±50 typical)
    decay_rate: float           # Exponential decay coefficient (negative, e.g., -3.0)
    waveform: str = 'sine'      # Oscillator waveform type

    @property
    def death_sample(self) -> int:
        """Sample index when particle dies."""
        return self.birth_sample + self.lifetime_samples


class ParticleEmitter:
    """
    Spawns particles over time using density curves with random variation.

    Combines artistic control (emission curves) with natural randomness
    (Poisson-style stochastic timing).
    """

    def __init__(self,
                 emission_curve: np.ndarray,
                 base_spawn_rate: float,
                 pitch_range_hz: tuple,
                 lifetime_range_samples: tuple,
                 velocity_range: tuple,
                 detune_range_cents: tuple,
                 decay_rate_range: tuple,
                 waveform: str,
                 rng: np.random.Generator):
        """
        Initialize particle emitter.

        Args:
            emission_curve: Density over time [0-1], length = total_samples
            base_spawn_rate: Base probability multiplier for spawning
            pitch_range_hz: (min_hz, max_hz) for random pitch selection
            lifetime_range_samples: (min_samples, max_samples) for particle lifetime
            velocity_range: (min_vel, max_vel) for amplitude variation
            detune_range_cents: (min_cents, max_cents) for pitch micro-variation
            decay_rate_range: (min_rate, max_rate) for decay speed (negative values)
            waveform: Oscillator waveform type ('sine', 'triangle')
            rng: NumPy random generator for reproducibility
        """
        self.emission_curve = emission_curve
        self.base_spawn_rate = base_spawn_rate
        self.pitch_range_hz = pitch_range_hz
        self.lifetime_range_samples = lifetime_range_samples
        self.velocity_range = velocity_range
        self.detune_range_cents = detune_range_cents
        self.decay_rate_range = decay_rate_range
        self.waveform = waveform
        self.rng = rng

    def emit_particles(self) -> List[Particle]:
        """
        Generate all particles for entire gesture duration.

        Uses Poisson-like stochastic spawning modulated by emission curve.

        Returns:
            List of Particle objects sorted by birth_sample
        """
        total_samples = len(self.emission_curve)
        particles = []

        # Iterate through time, spawning particles based on density curve
        for sample_idx in range(total_samples):
            current_density = self.emission_curve[sample_idx]

            # Spawn probability = density × base_rate × time_delta
            # (dt = 1 sample, so spawn_probability is per-sample probability)
            spawn_probability = current_density * self.base_spawn_rate

            # Stochastic spawning: random draw
            if self.rng.random() < spawn_probability:
                particle = self._create_particle(sample_idx)
                particles.append(particle)

        # Sort by birth time for efficient rendering
        particles.sort(key=lambda p: p.birth_sample)

        return particles

    def _create_particle(self, birth_sample: int) -> Particle:
        """
        Create a single particle with randomized properties.

        Args:
            birth_sample: Sample index when particle spawns

        Returns:
            New Particle instance
        """
        # Random pitch within range
        pitch_hz = self.rng.uniform(self.pitch_range_hz[0], self.pitch_range_hz[1])

        # Random lifetime
        lifetime_samples = int(self.rng.uniform(
            self.lifetime_range_samples[0],
            self.lifetime_range_samples[1]
        ))

        # Random velocity (amplitude)
        velocity = self.rng.uniform(self.velocity_range[0], self.velocity_range[1])

        # Random detune (pitch micro-variation)
        detune_cents = self.rng.uniform(self.detune_range_cents[0], self.detune_range_cents[1])

        # Random decay rate
        decay_rate = self.rng.uniform(self.decay_rate_range[0], self.decay_rate_range[1])

        return Particle(
            birth_sample=birth_sample,
            lifetime_samples=lifetime_samples,
            pitch_hz=pitch_hz,
            velocity=velocity,
            detune_cents=detune_cents,
            decay_rate=decay_rate,
            waveform=self.waveform
        )


class ParticleRenderer:
    """
    Synthesizes and mixes particle audio using SubtractiveSynth engine.

    Each particle is rendered independently, then mixed with equal-power summing.
    """

    def __init__(self, synth_engine, sample_rate: int):
        """
        Initialize particle renderer.

        Args:
            synth_engine: SubtractiveSynth instance for audio generation
            sample_rate: Audio sample rate (Hz)
        """
        self.synth = synth_engine
        self.sample_rate = sample_rate

    def render_particles(self,
                        particles: List[Particle],
                        total_samples: int) -> np.ndarray:
        """
        Render all particles and mix into final audio buffer.

        Args:
            particles: List of Particle objects to render
            total_samples: Total length of output buffer

        Returns:
            Mixed mono audio buffer (numpy array, float32)
        """
        if not particles:
            # No particles, return silence
            return np.zeros(total_samples, dtype=np.float32)

        # Accumulator for mixed audio
        mixed_audio = np.zeros(total_samples, dtype=np.float32)

        # Track max overlapping particles for proper normalization
        overlap_count = np.zeros(total_samples, dtype=np.int32)

        # Render each particle independently
        for particle in particles:
            particle_audio = self._render_single_particle(particle)

            # Place particle audio at correct position in timeline
            start_idx = particle.birth_sample
            end_idx = min(particle.death_sample, total_samples)
            particle_length = end_idx - start_idx

            if particle_length > 0:
                # Mix into accumulator
                mixed_audio[start_idx:end_idx] += particle_audio[:particle_length]
                # Track overlaps
                overlap_count[start_idx:end_idx] += 1

        # Dynamic normalization based on actual overlap at each sample
        # This prevents clipping when many particles overlap
        max_overlap = np.max(overlap_count)
        if max_overlap > 1:
            # Normalize by maximum overlap (conservative approach)
            mixed_audio /= np.sqrt(max_overlap)

        return mixed_audio

    def _render_single_particle(self, particle: Particle) -> np.ndarray:
        """
        Render audio for a single particle.

        Args:
            particle: Particle to render

        Returns:
            Mono audio buffer for this particle (length = lifetime_samples)
        """
        num_samples = particle.lifetime_samples

        # Apply detune to pitch
        detune_multiplier = 2 ** (particle.detune_cents / 1200)
        actual_pitch_hz = particle.pitch_hz * detune_multiplier

        # Generate constant pitch curve (struck note doesn't glide)
        pitch_curve = np.full(num_samples, actual_pitch_hz, dtype=np.float32)

        # Generate oscillator audio
        audio = self.synth.oscillator_timevarying_pitch(pitch_curve, waveform=particle.waveform)

        # Apply exponential decay envelope
        envelope = self._generate_exponential_envelope(
            num_samples,
            particle.decay_rate,
            particle.velocity
        )

        audio *= envelope

        return audio

    def _generate_exponential_envelope(self,
                                      num_samples: int,
                                      decay_rate: float,
                                      peak_amplitude: float) -> np.ndarray:
        """
        Generate exponential decay envelope for particle.

        Args:
            num_samples: Length of envelope
            decay_rate: Decay coefficient (negative, e.g., -3.0)
            peak_amplitude: Peak amplitude (0-1)

        Returns:
            Envelope curve (numpy array)
        """
        # Smooth attack to prevent clicks (longer for cleaner sound)
        attack_samples = max(1, int(0.005 * self.sample_rate))  # 5ms attack (smoother)

        # Exponential decay: exp(decay_rate * t / duration)
        t = np.arange(num_samples) / num_samples
        envelope = np.exp(decay_rate * t)

        # Smooth attack (avoid click)
        if attack_samples < num_samples:
            attack_ramp = np.linspace(0, 1, attack_samples)
            envelope[:attack_samples] *= attack_ramp

        # Apply peak amplitude
        envelope *= peak_amplitude

        return envelope


class ParticleGestureGenerator:
    """
    High-level particle-based gesture generator.

    Alternative to GestureGenerator for particle-based archetypes.
    Uses the same configuration pattern as traditional gestures.
    """

    def __init__(self,
                 archetype_config: Dict[str, Any],
                 rng: np.random.Generator,
                 synth_engine):
        """
        Initialize particle gesture generator.

        Args:
            archetype_config: Configuration dict with 'particle' section
            rng: NumPy random generator
            synth_engine: SubtractiveSynth instance
        """
        self.config = archetype_config
        self.rng = rng
        self.synth_engine = synth_engine

        # Validate config has particle section
        if 'particle' not in archetype_config:
            raise ValueError(
                "Particle archetype config must contain 'particle' section"
            )

    def generate_gesture(self,
                        moment_event: Dict[str, Any],
                        section_context: Dict[str, Any],
                        sample_rate: int) -> np.ndarray:
        """
        Generate particle-based audio gesture.

        Args:
            moment_event: Moment metadata dict
            section_context: Section-level parameters
            sample_rate: Audio sample rate (Hz)

        Returns:
            Mono audio buffer (numpy array, float32)
        """
        from .utils import finalize_audio

        # Compute duration
        duration = self._compute_duration(section_context)
        total_samples = int(duration * sample_rate)

        # Generate emission curve
        particle_config = self.config['particle']
        emission_curve = self._generate_emission_curve(
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
            rng=self.rng
        )

        # Emit particles
        particles = emitter.emit_particles()

        # Render particles
        renderer = ParticleRenderer(self.synth_engine, sample_rate)
        audio = renderer.render_particles(particles, total_samples)

        # Finalize (peak limiting)
        audio = finalize_audio(
            audio,
            peak_limit=self.config.get('peak_limit', 0.5),
            rms_target=self.config.get('rms_target', -25.0)
        )

        return audio

    def _compute_duration(self, section_context: Dict[str, Any]) -> float:
        """Compute gesture duration (same as GestureGenerator)."""
        base = self.config['duration_base']
        tension_scale = self.config.get('duration_tension_scale', 0.0)
        entropy_scale = self.config.get('duration_entropy_scale', 0.0)

        tension = section_context.get('tension', 0.5)
        entropy = section_context.get('entropy', 0.5)

        duration = base + (tension * tension_scale) + (entropy * entropy_scale)
        return np.clip(duration, 0.5, 10.0)

    def _generate_emission_curve(self,
                                 emission_config: Dict[str, Any],
                                 total_samples: int,
                                 section_context: Dict[str, Any]) -> np.ndarray:
        """
        Generate emission density curve.

        Args:
            emission_config: Emission configuration from archetype
            total_samples: Length of curve
            section_context: Section-level parameters

        Returns:
            Emission curve [0-1] (numpy array)
        """
        emission_type = emission_config['type']

        if emission_type == 'constant':
            # Constant emission rate
            density = emission_config.get('density', 0.5)
            return np.full(total_samples, density, dtype=np.float32)

        elif emission_type == 'gusts':
            # Wind gust pattern: calm → gust → calm (multiple cycles)
            num_gusts = emission_config.get('num_gusts', 2)
            base_density = emission_config.get('base_density', 0.1)
            peak_density = emission_config.get('peak_density', 0.9)

            curve = np.ones(total_samples) * base_density
            gust_width = total_samples // (num_gusts * 2)

            for i in range(num_gusts):
                # Gust center position
                gust_center = int((i + 0.5) * total_samples / num_gusts)
                gust_start = max(0, gust_center - gust_width // 2)
                gust_end = min(total_samples, gust_center + gust_width // 2)

                # Gaussian bump for gust
                gust_t = np.linspace(-2, 2, gust_end - gust_start)
                gust_envelope = np.exp(-gust_t ** 2)
                curve[gust_start:gust_end] += (peak_density - base_density) * gust_envelope

            return np.clip(curve, 0.0, 1.0).astype(np.float32)

        elif emission_type == 'swell':
            # Gradual swell: low → high
            start_density = emission_config.get('start_density', 0.1)
            end_density = emission_config.get('end_density', 0.9)
            return np.linspace(start_density, end_density, total_samples).astype(np.float32)

        elif emission_type == 'decay_scatter':
            # High at start, decaying exponentially (like falling debris)
            start_density = emission_config.get('start_density', 0.9)
            decay_rate = emission_config.get('decay_rate', -2.0)
            t = np.arange(total_samples) / total_samples
            curve = start_density * np.exp(decay_rate * t)
            return np.clip(curve, 0.0, 1.0).astype(np.float32)

        elif emission_type == 'impact_burst':
            # Sudden cluster at specific time, then sparse tail (for collisions)
            impact_time_ratio = emission_config.get('impact_time_ratio', 0.15)
            burst_density = emission_config.get('burst_density', 0.95)
            burst_duration_ratio = emission_config.get('burst_duration_ratio', 0.08)
            tail_density = emission_config.get('tail_density', 0.05)

            impact_sample = int(impact_time_ratio * total_samples)
            burst_samples = int(burst_duration_ratio * total_samples)

            curve = np.ones(total_samples) * tail_density
            burst_start = max(0, impact_sample)
            burst_end = min(total_samples, impact_sample + burst_samples)
            curve[burst_start:burst_end] = burst_density

            return curve.astype(np.float32)

        elif emission_type == 'rhythmic_clusters':
            # Grouped bursts with gaps between (for tactical thinking)
            num_clusters = emission_config.get('num_clusters', 4)
            cluster_duration_ratio = emission_config.get('cluster_duration_ratio', 0.15)
            cluster_density = emission_config.get('cluster_density', 0.7)
            gap_density = emission_config.get('gap_density', 0.05)

            curve = np.ones(total_samples) * gap_density
            cluster_samples = int(cluster_duration_ratio * total_samples)
            gap_samples = (total_samples - num_clusters * cluster_samples) // (num_clusters + 1)

            for i in range(num_clusters):
                start = gap_samples * (i + 1) + cluster_samples * i
                end = min(start + cluster_samples, total_samples)
                if start < total_samples:
                    curve[start:end] = cluster_density

            return curve.astype(np.float32)

        elif emission_type == 'drift_scatter':
            # Sparse particles with slowly changing density (for gradual shifts)
            start_density = emission_config.get('start_density', 0.08)
            end_density = emission_config.get('end_density', 0.12)
            drift_rate = emission_config.get('drift_rate', 0.3)

            # Linear drift with slight sinusoidal modulation
            linear = np.linspace(start_density, end_density, total_samples)
            t = np.arange(total_samples) / total_samples
            modulation = drift_rate * np.sin(2 * np.pi * 2 * t) * 0.1  # Subtle variation
            curve = linear + modulation

            return np.clip(curve, 0.0, 1.0).astype(np.float32)

        elif emission_type == 'dissolve':
            # High start, exponential decay to silence (for endings)
            start_density = emission_config.get('start_density', 0.6)
            decay_rate = emission_config.get('decay_rate', -1.8)

            t = np.arange(total_samples) / total_samples
            curve = start_density * np.exp(decay_rate * t)

            return np.clip(curve, 0.0, 1.0).astype(np.float32)

        else:
            raise ValueError(f"Unknown emission type: {emission_type}")
