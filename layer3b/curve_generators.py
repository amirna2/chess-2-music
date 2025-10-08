"""
Reusable parameter curve generation functions for Layer 3b gesture system.

Each function takes configuration parameters and generates time-varying curves
for pitch, harmony, filter, envelope, and texture parameters.

All functions are PURE - no side effects, deterministic given the same inputs.
"""

import numpy as np
from typing import Dict, Any, List


def generate_pitch_curve(config: Dict[str, Any],
                        phases: Dict[str, Any],
                        section_context: Dict[str, Any],
                        total_samples: int,
                        rng: np.random.Generator,
                        sample_rate: int = 88200) -> np.ndarray:
    """
    Route to specific pitch trajectory algorithm based on configuration type.

    Args:
        config: Pitch configuration from archetype
        phases: Phase timeline dict with sample boundaries
        section_context: Section-level parameters (tension, entropy, scale)
        total_samples: Total gesture duration in samples
        rng: Random generator for reproducible randomness
        sample_rate: Audio sample rate (Hz)

    Returns:
        Pitch curve in Hz (numpy array, length = total_samples)
    """
    trajectory_type = config['type']

    if trajectory_type == "exponential_gliss":
        return _pitch_exponential_gliss(config, phases, section_context, total_samples, rng)
    elif trajectory_type == "ascending_spread":
        return _pitch_ascending_spread(config, phases, section_context, total_samples, rng)
    elif trajectory_type == "oscillating_tremor":
        return _pitch_oscillating_tremor(config, phases, section_context, total_samples, rng, sample_rate)
    elif trajectory_type == "cellular_sequence":
        return _pitch_cellular_sequence(config, phases, section_context, total_samples, rng, sample_rate)
    else:
        raise ValueError(f"Unknown pitch trajectory type: {trajectory_type}")


def _pitch_exponential_gliss(config: Dict[str, Any],
                             phases: Dict[str, Any],
                             section_context: Dict[str, Any],
                             total_samples: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Exponential glissando for BLUNDER archetype (high → low).

    Creates a suspended high pitch that dramatically falls through an
    exponential curve, mimicking the sound of failure or collapse.
    """
    tension = section_context.get('tension', 0.5)
    entropy = section_context.get('entropy', 0.5)

    # Calculate start and end frequencies based on context
    peak_freq = config['start_freq_base'] + (entropy * config['start_freq_entropy_scale'])
    octave_drop = config['octave_drop_base'] + (tension * config['octave_drop_tension_scale'])
    target_freq = peak_freq / (2 ** octave_drop)

    # Extract phase boundaries
    pre_shadow_end = phases['pre_shadow']['end_sample']
    decay_end = phases['decay']['end_sample']

    # Build curve with three distinct sections
    curve = np.zeros(total_samples)

    # Phase 1: Suspended at high pitch (creates tension)
    curve[:pre_shadow_end] = peak_freq

    # Phase 2: Exponential glissando down (the fall)
    gliss_samples = decay_end - pre_shadow_end
    if gliss_samples > 0:
        # Use logarithmic space for smooth exponential curve
        curve[pre_shadow_end:decay_end] = np.exp(
            np.linspace(np.log(peak_freq), np.log(target_freq), gliss_samples)
        )

    # Phase 3: Residue (hold at low target frequency)
    curve[decay_end:] = target_freq

    return curve


def _pitch_ascending_spread(config: Dict[str, Any],
                            phases: Dict[str, Any],
                            section_context: Dict[str, Any],
                            total_samples: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Ascending pitch spread for BRILLIANT archetype (low → high).

    Creates an upward sweeping gesture that opens and brightens,
    suggesting achievement or revelation.
    """
    entropy = section_context.get('entropy', 0.5)

    # Calculate frequency range
    start_freq = config['start_freq_base'] + (entropy * config['start_freq_entropy_scale'])
    end_freq = start_freq * (2 ** config['octave_rise'])

    # Spread occurs during impact + bloom phases
    spread_start = phases['impact']['start_sample']
    spread_end = phases['bloom']['end_sample']

    curve = np.zeros(total_samples)

    # Pre-impact: hold at starting frequency
    curve[:spread_start] = start_freq

    # Impact + bloom: exponential rise for natural pitch perception
    spread_samples = spread_end - spread_start
    if spread_samples > 0:
        curve[spread_start:spread_end] = np.exp(
            np.linspace(np.log(start_freq), np.log(end_freq), spread_samples)
        )

    # Post-bloom: sustain at target
    curve[spread_end:] = end_freq

    return curve


def _pitch_oscillating_tremor(config: Dict[str, Any],
                              phases: Dict[str, Any],
                              section_context: Dict[str, Any],
                              total_samples: int,
                              rng: np.random.Generator,
                              sample_rate: int = 88200) -> np.ndarray:
    """
    Oscillating tremor for TIME_PRESSURE archetype.

    Creates an accelerating vibrato effect that suggests nervousness,
    urgency, or instability under time pressure.
    """
    tension = section_context.get('tension', 0.5)

    # Center frequency and tremor parameters scale with tension
    center_freq = config['center_freq_base'] + (tension * config['center_freq_tension_scale'])
    tremor_rate = config['tremor_rate_base_hz'] + (tension * config['tremor_rate_tension_scale_hz'])
    tremor_depth = config['tremor_depth_semitones']

    # Tremor accelerates during bloom phase
    bloom_start = phases['bloom']['start_sample']
    bloom_end = phases['bloom']['end_sample']

    # Start with steady center frequency
    curve = np.full(total_samples, center_freq)

    # Apply accelerating tremor during bloom
    bloom_samples = bloom_end - bloom_start
    if bloom_samples > 0:
        # Time vector for bloom phase
        t = np.arange(bloom_samples) / sample_rate

        # Accelerating LFO: frequency increases linearly over time
        acceleration_factor = 1 + t / (t[-1] if t[-1] > 0 else 1)  # Double rate by end
        instantaneous_freq = tremor_rate * acceleration_factor

        # Integrate to get phase (more accurate than simple multiplication)
        phase = np.cumsum(instantaneous_freq) / sample_rate
        lfo = np.sin(2 * np.pi * phase)

        # Convert tremor depth from semitones to frequency multiplier
        tremor_mult = 2 ** ((lfo * tremor_depth) / 12)
        curve[bloom_start:bloom_end] *= tremor_mult

    return curve


def _pitch_cellular_sequence(config: Dict[str, Any],
                             phases: Dict[str, Any],
                             section_context: Dict[str, Any],
                             total_samples: int,
                             rng: np.random.Generator,
                             sample_rate: int = 88200) -> np.ndarray:
    """
    Cellular sequence for TACTICAL_SEQUENCE archetype.

    Creates discrete pitch cells that form patterns, suggesting
    methodical calculation or strategic planning.

    Note: Simple implementation using golden ratio-based pattern.
    Full cellular automaton would require more complex rules.
    """
    cells = np.array(config['cell_frequencies'])
    cell_dur_samples = int(config['cell_duration_ms'] * sample_rate / 1000)

    curve = np.zeros(total_samples)

    # Golden ratio for interesting non-repeating patterns
    golden_ratio = (1 + np.sqrt(5)) / 2

    # Generate pattern based on golden ratio sequence
    current_pos = 0
    cell_index = 0

    while current_pos < total_samples:
        end_pos = min(current_pos + cell_dur_samples, total_samples)

        # Select cell based on golden ratio pattern
        curve[current_pos:end_pos] = cells[cell_index % len(cells)]

        # Advance pattern
        current_pos = end_pos

        # Acceleration during bloom phase creates urgency
        if current_pos > phases['bloom']['start_sample']:
            cell_index += 2  # Double speed
        else:
            cell_index += 1

        # Add golden ratio offset for non-repetitive pattern
        if cell_index % 3 == 0:
            cell_index = int(cell_index * golden_ratio) % len(cells)

    return curve


def generate_harmony(config: Dict[str, Any],
                    base_pitch_curve: np.ndarray,
                    phases: Dict[str, Any],
                    section_context: Dict[str, Any],
                    rng: np.random.Generator) -> List[np.ndarray]:
    """
    Generate harmonic voices based on base pitch curve.

    Args:
        config: Harmony configuration from archetype
        base_pitch_curve: Fundamental pitch curve in Hz
        phases: Phase timeline dict
        section_context: Section-level parameters
        rng: Random generator

    Returns:
        List of pitch curves (one per voice)
    """
    harmony_type = config['type']

    if harmony_type == "cluster_to_interval":
        return _harmony_cluster_to_interval(config, base_pitch_curve, phases, section_context, rng)
    elif harmony_type == "unison_to_chord":
        return _harmony_unison_to_chord(config, base_pitch_curve, phases, section_context, rng)
    elif harmony_type == "dense_cluster":
        return _harmony_dense_cluster(config, base_pitch_curve, phases, section_context, rng)
    elif harmony_type == "harmonic_stack":
        return _harmony_harmonic_stack(config, base_pitch_curve, phases, section_context, rng)
    else:
        raise ValueError(f"Unknown harmony type: {harmony_type}")


def _harmony_cluster_to_interval(config: Dict[str, Any],
                                 base_pitch_curve: np.ndarray,
                                 phases: Dict[str, Any],
                                 section_context: Dict[str, Any],
                                 rng: np.random.Generator) -> List[np.ndarray]:
    """
    Cluster resolving to muddy interval for BLUNDER archetype.

    Creates dissonant cluster that collapses to uncomfortable interval,
    reinforcing the sense of mistake or failure.
    """
    tension = section_context.get('tension', 0.5)
    num_voices = int(config['num_voices_base'] + tension * config['num_voices_tension_scale'])
    num_voices = max(2, min(num_voices, 5))  # Clamp to reasonable range

    # Choose muddy interval based on tension
    if config['resolve_interval_type'] == "muddy":
        muddy_interval = 6 if tension > 0.5 else 1  # Tritone or minor 2nd
    else:
        muddy_interval = 6  # Default to tritone

    # Transition point where cluster resolves
    resolve_phase = config.get('resolve_phase', 'decay')
    transition_sample = phases[resolve_phase]['start_sample']

    voices = []
    for i in range(num_voices):
        voice_curve = base_pitch_curve.copy()

        # Cluster detuning (tight semitones around center)
        cluster_detune = (i - num_voices // 2) * config['cluster_semitone_spacing']

        # Before transition: apply cluster detuning
        detune_mult = 2 ** (cluster_detune / 12)
        voice_curve[:transition_sample] *= detune_mult

        # After transition: keep only 2 voices for muddy interval
        if i < 2:
            resolved_detune = muddy_interval if i == 1 else 0
            voice_curve[transition_sample:] = base_pitch_curve[transition_sample:] * (2 ** (resolved_detune / 12))
            voices.append(voice_curve)
        elif transition_sample < len(voice_curve):
            # Fade out extra voices
            voice_curve[transition_sample:] = 0

    return voices[:2] if transition_sample < len(base_pitch_curve) else voices


def _harmony_unison_to_chord(config: Dict[str, Any],
                             base_pitch_curve: np.ndarray,
                             phases: Dict[str, Any],
                             section_context: Dict[str, Any],
                             rng: np.random.Generator) -> List[np.ndarray]:
    """
    Unison spreading to chord for BRILLIANT archetype.

    Creates a blossoming effect as single note spreads into
    rich harmonic structure, suggesting success or revelation.
    """
    num_voices = config['num_voices']

    # Define chord intervals based on type
    if config['chord_type'] == "major_seventh":
        intervals = [0, 4, 7, 11]  # Root, major 3rd, 5th, major 7th
    elif config['chord_type'] == "major":
        intervals = [0, 4, 7, 12]  # Root, major 3rd, 5th, octave
    else:
        intervals = [0, 4, 7, 11]  # Default to maj7

    spread_phase = config.get('spread_phase', 'bloom')
    spread_start = phases[spread_phase]['start_sample']

    voices = []
    for i in range(min(num_voices, len(intervals))):
        voice_curve = base_pitch_curve.copy()
        interval = intervals[i]

        # Unison before spread (all voices at base pitch)
        # No change needed for first part

        # After spread point, apply chord interval
        if spread_start < len(voice_curve):
            interval_mult = 2 ** (interval / 12)
            voice_curve[spread_start:] = base_pitch_curve[spread_start:] * interval_mult

        voices.append(voice_curve)

    return voices


def _harmony_dense_cluster(config: Dict[str, Any],
                           base_pitch_curve: np.ndarray,
                           phases: Dict[str, Any],
                           section_context: Dict[str, Any],
                           rng: np.random.Generator) -> List[np.ndarray]:
    """
    Dense microtonal cluster for TIME_PRESSURE archetype.

    Creates unstable, beating cluster with microtonal intervals
    that produces anxiety and tension.
    """
    num_voices = config['num_voices']
    spacing_min = config['semitone_spacing_min']
    spacing_max = config['semitone_spacing_max']

    voices = []
    for i in range(num_voices):
        voice_curve = base_pitch_curve.copy()

        # Random microtonal detuning for each voice
        # Center voices around the base pitch
        detune = rng.uniform(spacing_min, spacing_max) * (i - num_voices // 2)

        # Apply detuning as frequency multiplier
        detune_mult = 2 ** (detune / 12)
        voice_curve *= detune_mult

        voices.append(voice_curve)

    return voices


def _harmony_harmonic_stack(config: Dict[str, Any],
                            base_pitch_curve: np.ndarray,
                            phases: Dict[str, Any],
                            section_context: Dict[str, Any],
                            rng: np.random.Generator) -> List[np.ndarray]:
    """
    Harmonic stack with just intonation for TACTICAL_SEQUENCE archetype.

    Creates precise harmonic relationships using just intonation ratios,
    producing clear, mathematical harmony.
    """
    num_voices = config['num_voices']
    ratios = config['interval_ratios']

    voices = []
    for i in range(min(num_voices, len(ratios))):
        ratio = ratios[i]
        voice_curve = base_pitch_curve * ratio

        # Optional phase offset for subtle phasing effect
        if config.get('phase_offset_enable', False) and i > 0:
            # Small time offset creates phasing
            offset_samples = int(i * 100)  # ~2.3ms at 44.1kHz
            # Use circular shift to maintain curve length
            voice_curve = np.roll(voice_curve, offset_samples)
            # Smooth the wrap-around point
            if offset_samples > 0:
                voice_curve[:offset_samples] = base_pitch_curve[:offset_samples] * ratio

        voices.append(voice_curve)

    return voices


def generate_filter_curve(config: Dict[str, Any],
                          phases: Dict[str, Any],
                          section_context: Dict[str, Any],
                          total_samples: int,
                          rng: np.random.Generator,
                          sample_rate: int = 88200) -> Dict[str, np.ndarray]:
    """
    Generate time-varying filter parameters.

    Args:
        config: Filter configuration from archetype
        phases: Phase timeline dict
        section_context: Section-level parameters
        total_samples: Total duration in samples
        rng: Random generator
        sample_rate: Audio sample rate

    Returns:
        Dict with keys: 'cutoff', 'resonance', 'type'
    """
    filter_type = config['type']

    if filter_type == "bandpass_to_lowpass_choke":
        return _filter_bp_to_lp_choke(config, phases, section_context, total_samples, rng)
    elif filter_type == "lowpass_to_highpass_open":
        return _filter_lp_to_hp_open(config, phases, section_context, total_samples, rng)
    elif filter_type == "bandpass_sweep":
        return _filter_bp_sweep(config, phases, section_context, total_samples, rng)
    elif filter_type == "rhythmic_gate":
        return _filter_rhythmic_gate(config, phases, section_context, total_samples, rng, sample_rate)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def _filter_bp_to_lp_choke(config: Dict[str, Any],
                           phases: Dict[str, Any],
                           section_context: Dict[str, Any],
                           total_samples: int,
                           rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Band-pass to low-pass choke for BLUNDER archetype.

    Filter closes down dramatically, muffling the sound to
    reinforce the sense of collapse or failure.
    """
    tension = section_context.get('tension', 0.5)

    # Calculate filter parameters based on context
    bp_center = config['bp_center_base'] + (tension * config['bp_center_tension_scale'])
    lp_cutoff = config['lp_cutoff_base'] + (tension * config['lp_cutoff_tension_scale'])
    lp_resonance = config['lp_resonance_base'] + (tension * config.get('lp_resonance_tension_scale', 0))

    morph_phase = config.get('morph_phase', 'bloom')
    morph_start = phases[morph_phase]['start_sample']

    # Cutoff curve: bandpass center → lowpass cutoff
    cutoff = np.zeros(total_samples)
    cutoff[:morph_start] = bp_center

    # Exponential morph for natural filter sweep
    morph_samples = total_samples - morph_start
    if morph_samples > 0:
        cutoff[morph_start:] = np.exp(
            np.linspace(np.log(bp_center), np.log(lp_cutoff), morph_samples)
        )

    # Resonance curve: moderate → high (emphasizes the choke)
    resonance = np.zeros(total_samples)
    resonance[:morph_start] = 0.4  # Moderate bandpass resonance

    if morph_samples > 0:
        resonance[morph_start:] = np.linspace(0.4, lp_resonance, morph_samples)

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'bandpass_to_lowpass'
    }


def _filter_lp_to_hp_open(config: Dict[str, Any],
                          phases: Dict[str, Any],
                          section_context: Dict[str, Any],
                          total_samples: int,
                          rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Low-pass to high-pass opening for BRILLIANT archetype.

    Filter opens up from dark to bright, creating sense of
    revelation or achievement.
    """
    morph_phase = config.get('morph_phase', 'bloom')
    morph_start = phases[morph_phase]['start_sample']

    lp_start = config['lp_cutoff_start']
    hp_end = config['hp_cutoff_end']

    # Cutoff curve: low → high frequency
    cutoff = np.zeros(total_samples)
    cutoff[:morph_start] = lp_start

    morph_samples = total_samples - morph_start
    if morph_samples > 0:
        # Exponential curve for perceptually linear sweep
        cutoff[morph_start:] = np.exp(
            np.linspace(np.log(lp_start), np.log(hp_end), morph_samples)
        )

    # Steady resonance throughout
    resonance = np.full(total_samples, config['resonance_base'])

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'lowpass_to_highpass'
    }


def _filter_bp_sweep(config: Dict[str, Any],
                     phases: Dict[str, Any],
                     section_context: Dict[str, Any],
                     total_samples: int,
                     rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Band-pass sweep for TIME_PRESSURE archetype.

    Creates nervous, searching filter sweep that accelerates,
    suggesting urgency and instability.
    """
    sweep_phase = config.get('sweep_phase', 'bloom')
    sweep_start = phases[sweep_phase]['start_sample']
    sweep_end = phases[sweep_phase]['end_sample']

    bp_start = config['bp_center_start']
    bp_end = config['bp_center_end']

    # Initialize with starting frequency
    cutoff = np.full(total_samples, bp_start)

    sweep_samples = sweep_end - sweep_start
    if sweep_samples > 0 and config.get('sweep_rate_modulation') == 'accelerating':
        # Quadratic curve for acceleration
        t = np.linspace(0, 1, sweep_samples)
        t_squared = t ** 2  # Accelerating curve

        # Map to frequency range
        sweep_curve = bp_start + (bp_end - bp_start) * t_squared
        cutoff[sweep_start:sweep_end] = sweep_curve

    # Hold at end frequency after sweep
    cutoff[sweep_end:] = bp_end

    # Moderate resonance for defined bandpass character
    resonance = np.full(total_samples, 0.5)

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'bandpass'
    }


def _filter_rhythmic_gate(config: Dict[str, Any],
                          phases: Dict[str, Any],
                          section_context: Dict[str, Any],
                          total_samples: int,
                          rng: np.random.Generator,
                          sample_rate: int = 88200) -> Dict[str, np.ndarray]:
    """
    Rhythmic filter gating for TACTICAL_SEQUENCE archetype.

    Creates precise, mechanical filter pulses that suggest
    calculated, methodical action.
    """
    lp_base = config['lp_cutoff_base']
    lp_pulse = lp_base * config['lp_cutoff_pulse_mult']
    pulse_hz = config['pulse_rate_hz']
    duty_cycle = config['pulse_duty_cycle']

    # Generate pulse train
    t = np.arange(total_samples) / sample_rate
    pulse_phase = (t * pulse_hz) % 1.0
    pulse_train = (pulse_phase < duty_cycle).astype(float)

    # Smooth pulse edges to prevent clicks
    # Simple linear ramp over ~1ms
    ramp_samples = int(0.001 * sample_rate)
    if ramp_samples > 0:
        # Find pulse edges
        edges = np.diff(pulse_train, prepend=pulse_train[0])
        rise_edges = np.where(edges > 0)[0]
        fall_edges = np.where(edges < 0)[0]

        # Apply ramps
        for edge in rise_edges:
            end = min(edge + ramp_samples, total_samples)
            pulse_train[edge:end] *= np.linspace(0, 1, end - edge)

        for edge in fall_edges:
            start = max(edge - ramp_samples, 0)
            pulse_train[start:edge] *= np.linspace(1, 0, edge - start)

    # Cutoff follows pulse with modulation depth
    cutoff = lp_base + (lp_pulse - lp_base) * pulse_train

    # Low resonance for clean gating
    resonance = np.full(total_samples, 0.3)

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'lowpass'
    }


def generate_envelope(config: Dict[str, Any],
                     phases: Dict[str, Any],
                     total_samples: int,
                     rng: np.random.Generator,
                     sample_rate: int = 88200) -> np.ndarray:
    """
    Generate amplitude envelope curve.

    Args:
        config: Envelope configuration from archetype
        phases: Phase timeline dict
        total_samples: Total duration in samples
        rng: Random generator
        sample_rate: Audio sample rate

    Returns:
        Envelope curve (0.0 to 1.0, length = total_samples)
    """
    envelope_type = config['type']

    if envelope_type == "sudden_short_tail":
        return _envelope_sudden_short_tail(config, phases, total_samples, rng, sample_rate)
    elif envelope_type == "gradual_sustained":
        return _envelope_gradual_sustained(config, phases, total_samples, rng, sample_rate)
    elif envelope_type == "gated_pulse":
        return _envelope_gated_pulse(config, phases, total_samples, rng, sample_rate)
    else:
        raise ValueError(f"Unknown envelope type: {envelope_type}")


def _envelope_sudden_short_tail(config: Dict[str, Any],
                                phases: Dict[str, Any],
                                total_samples: int,
                                rng: np.random.Generator,
                                sample_rate: int = 88200) -> np.ndarray:
    """
    Sudden attack with short tail for BLUNDER and TIME_PRESSURE archetypes.

    Creates sharp, immediate impact with rapid decay, suggesting
    suddenness and finality.
    """
    # Use context-based variation for attack time
    entropy_variation = rng.random()

    # Calculate attack time with variation
    attack_ms = config['attack_ms_base'] + (entropy_variation * config.get('attack_ms_entropy_scale', 0))
    attack_samples = max(1, int(attack_ms * sample_rate / 1000))

    # Calculate sustain and decay sections
    sustain_ratio = config.get('sustain_phase_ratio', 0.15)
    sustain_samples = int(sustain_ratio * total_samples)

    decay_samples = total_samples - attack_samples - sustain_samples
    decay_samples = max(0, decay_samples)

    envelope = np.zeros(total_samples)

    # Attack: rapid rise
    if attack_samples > 0:
        # Use cosine curve for smoother attack
        t = np.linspace(0, np.pi/2, attack_samples)
        envelope[:attack_samples] = np.sin(t)

    # Sustain: hold at maximum
    sustain_end = attack_samples + sustain_samples
    envelope[attack_samples:sustain_end] = 1.0

    # Decay: exponential fall
    if decay_samples > 0:
        decay_coeff = config.get('decay_coefficient', -4)
        envelope[sustain_end:] = np.exp(np.linspace(0, decay_coeff, decay_samples))

    return envelope


def _envelope_gradual_sustained(config: Dict[str, Any],
                               phases: Dict[str, Any],
                               total_samples: int,
                               rng: np.random.Generator,
                               sample_rate: int = 88200) -> np.ndarray:
    """
    Gradual attack with sustained plateau for BRILLIANT archetype.

    Creates smooth, expansive envelope that blooms and sustains,
    suggesting growth and achievement.
    """
    attack_ms = config.get('attack_ms', 50)
    attack_samples = int(attack_ms * sample_rate / 1000)

    sustain_ratio = config.get('sustain_phase_ratio', 0.5)
    sustain_samples = int(sustain_ratio * total_samples)

    decay_samples = total_samples - attack_samples - sustain_samples
    decay_samples = max(0, decay_samples)

    envelope = np.zeros(total_samples)

    # Gradual attack with S-curve for smooth onset
    if attack_samples > 0:
        t = np.linspace(-3, 3, attack_samples)
        # Sigmoid function for smooth S-curve
        envelope[:attack_samples] = 1 / (1 + np.exp(-t))

    # Sustain at full level
    sustain_end = attack_samples + sustain_samples
    envelope[attack_samples:sustain_end] = 1.0

    # Linear or exponential decay based on config
    if decay_samples > 0:
        if config.get('decay_curve', 'linear') == 'linear':
            envelope[sustain_end:] = np.linspace(1, 0, decay_samples)
        else:
            # Gentle exponential decay
            envelope[sustain_end:] = np.exp(np.linspace(0, -3, decay_samples))

    return envelope


def _envelope_gated_pulse(config: Dict[str, Any],
                         phases: Dict[str, Any],
                         total_samples: int,
                         rng: np.random.Generator,
                         sample_rate: int = 88200) -> np.ndarray:
    """
    Gated pulses for TACTICAL_SEQUENCE archetype.

    Creates rhythmic on/off gating that suggests mechanical,
    calculated action.
    """
    attack_ms = config.get('attack_ms', 5)
    attack_samples = max(1, int(attack_ms * sample_rate / 1000))

    release_ms = config.get('release_ms', 20)
    release_samples = max(1, int(release_ms * sample_rate / 1000))

    pulse_hz = config['pulse_rate_hz']

    # Generate base pulse train
    t = np.arange(total_samples) / sample_rate
    pulse_phase = (t * pulse_hz) % 1.0

    # Create gate with duty cycle of 0.5
    gate_on = pulse_phase < 0.5
    envelope = np.zeros(total_samples)
    envelope[gate_on] = 1.0

    # Apply attack/release slopes to prevent clicks
    # Find gate transitions
    transitions = np.diff(envelope, prepend=0, append=0)
    rise_edges = np.where(transitions > 0)[0]
    fall_edges = np.where(transitions < 0)[0]

    # Apply attack ramps
    for edge in rise_edges:
        ramp_end = min(edge + attack_samples, total_samples)
        ramp_len = ramp_end - edge
        if ramp_len > 0:
            # Exponential attack for punch
            envelope[edge:ramp_end] *= 1 - np.exp(-5 * np.linspace(0, 1, ramp_len))

    # Apply release ramps
    for edge in fall_edges:
        ramp_start = max(edge - release_samples, 0)
        ramp_len = edge - ramp_start
        if ramp_len > 0:
            # Exponential release
            envelope[ramp_start:edge] *= np.exp(-3 * np.linspace(0, 1, ramp_len))

    return envelope


def generate_texture_curve(config: Dict[str, Any],
                           phases: Dict[str, Any],
                           section_context: Dict[str, Any],
                           total_samples: int,
                           rng: np.random.Generator) -> Dict[str, Any]:
    """
    Generate texture parameters (noise, shimmer, etc.).

    Args:
        config: Texture configuration from archetype
        phases: Phase timeline dict
        section_context: Section-level parameters
        total_samples: Total duration in samples
        rng: Random generator

    Returns:
        Dict with texture parameters including noise_ratio, noise_type, shimmer settings
    """
    entropy = section_context.get('entropy', 0.5)

    # Calculate noise ratio based on entropy (more chaos = more noise)
    noise_ratio = config['noise_ratio_base'] + (entropy * config.get('noise_ratio_entropy_scale', 0))
    noise_ratio = np.clip(noise_ratio, 0.0, 1.0)

    # Build texture dictionary
    texture = {
        'noise_ratio': noise_ratio,
        'noise_type': config.get('noise_type', 'pink'),
        'shimmer_enable': config.get('shimmer_enable', False)
    }

    # Add shimmer parameters if enabled
    if texture['shimmer_enable']:
        texture['shimmer_rate_hz'] = config.get('shimmer_rate_hz', 6.0)
        texture['shimmer_depth'] = config.get('shimmer_depth', 0.3)

    return texture