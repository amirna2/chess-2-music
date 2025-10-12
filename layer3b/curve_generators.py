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
    elif trajectory_type == "weak_parabolic":
        return _pitch_weak_parabolic(config, phases, section_context, total_samples, rng)
    elif trajectory_type == "slow_drift":
        return _pitch_slow_drift(config, phases, section_context, total_samples, rng)
    elif trajectory_type == "impact_transient":
        return _pitch_impact_transient(config, phases, section_context, total_samples, rng)
    elif trajectory_type == "final_descent":
        return _pitch_final_descent(config, phases, section_context, total_samples, rng)
    elif trajectory_type == "discrete_chimes":
        return _pitch_discrete_chimes(config, phases, section_context, total_samples, rng, sample_rate)
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


def _pitch_weak_parabolic(config: Dict[str, Any],
                          phases: Dict[str, Any],
                          section_context: Dict[str, Any],
                          total_samples: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Weak parabolic pitch for INACCURACY archetype.

    Creates a gentle rise and fall, suggesting hesitation or uncertainty.
    Smaller pitch excursion than full parabolic gesture.
    """
    start_freq = config['start_freq_base']
    rise_semitones = config['rise_semitones']
    fall_semitones = config['fall_semitones']

    # Calculate peak and end frequencies
    peak_freq = start_freq * (2 ** (rise_semitones / 12))
    end_freq = peak_freq / (2 ** (fall_semitones / 12))

    # Phase boundaries
    impact_start = phases['impact']['start_sample']
    bloom_mid = phases['bloom']['start_sample'] + phases['bloom']['duration_samples'] // 2
    decay_end = phases['decay']['end_sample']

    curve = np.zeros(total_samples)

    # Pre-impact: hold at start
    curve[:impact_start] = start_freq

    # Impact → bloom mid: rise to peak (quadratic for smooth arc)
    rise_samples = bloom_mid - impact_start
    if rise_samples > 0:
        t = np.linspace(0, 1, rise_samples)
        # Quadratic easing for parabolic feel
        curve[impact_start:bloom_mid] = start_freq + (peak_freq - start_freq) * (t ** 2)

    # Bloom mid → decay end: fall to end (inverse quadratic)
    fall_samples = decay_end - bloom_mid
    if fall_samples > 0:
        t = np.linspace(0, 1, fall_samples)
        # Inverse quadratic for symmetric fall
        curve[bloom_mid:decay_end] = peak_freq - (peak_freq - end_freq) * (t ** 2)

    # Residue: hold at end
    curve[decay_end:] = end_freq

    return curve


def _pitch_slow_drift(config: Dict[str, Any],
                     phases: Dict[str, Any],
                     section_context: Dict[str, Any],
                     total_samples: int,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Slow drift for SIGNIFICANT_SHIFT archetype.

    Creates gradual pitch shift suggesting strategic repositioning.
    Linear or sigmoidal drift over extended duration.
    """
    start_freq = config['start_freq_base']
    drift_semitones = config['drift_semitones']
    drift_direction = config['drift_direction']

    # Calculate target frequency
    if drift_direction == 'ascending':
        end_freq = start_freq * (2 ** (drift_semitones / 12))
    else:
        end_freq = start_freq / (2 ** (drift_semitones / 12))

    # Drift occurs during bloom phase primarily
    bloom_start = phases['bloom']['start_sample']
    bloom_end = phases['bloom']['end_sample']

    curve = np.zeros(total_samples)

    # Pre-bloom: hold at start
    curve[:bloom_start] = start_freq

    # Bloom: slow drift with sigmoid curve for smooth acceleration/deceleration
    drift_samples = bloom_end - bloom_start
    if drift_samples > 0:
        t = np.linspace(-3, 3, drift_samples)  # Sigmoid range
        sigmoid = 1 / (1 + np.exp(-t))  # 0 to 1
        curve[bloom_start:bloom_end] = start_freq + (end_freq - start_freq) * sigmoid

    # Post-bloom: hold at target
    curve[bloom_end:] = end_freq

    return curve


def _pitch_impact_transient(config: Dict[str, Any],
                            phases: Dict[str, Any],
                            section_context: Dict[str, Any],
                            total_samples: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Impact transient for FIRST_EXCHANGE archetype.

    Creates sharp attack with rapid pitch decay, mimicking collision impact.
    Brief high pitch that quickly settles to lower fundamental.
    """
    strike_freq = config['strike_freq_base']
    decay_mult = config['decay_mult']
    target_freq = strike_freq * decay_mult

    # Impact occurs at impact phase
    impact_start = phases['impact']['start_sample']
    impact_end = phases['impact']['end_sample']
    bloom_end = phases['bloom']['end_sample']

    curve = np.zeros(total_samples)

    # Pre-impact: silence or very low (we'll use low fundamental)
    curve[:impact_start] = target_freq * 0.5

    # Impact: sharp strike at high pitch
    curve[impact_start:impact_end] = strike_freq

    # Bloom: exponential decay to target
    decay_samples = bloom_end - impact_end
    if decay_samples > 0:
        # Exponential decay curve
        decay_curve = np.exp(np.linspace(0, -3, decay_samples))
        curve[impact_end:bloom_end] = target_freq + (strike_freq - target_freq) * decay_curve

    # Post-bloom: settle at target
    curve[bloom_end:] = target_freq

    return curve


def _pitch_final_descent(config: Dict[str, Any],
                        phases: Dict[str, Any],
                        section_context: Dict[str, Any],
                        total_samples: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Final descent for FINAL_RESOLUTION archetype.

    Creates long, smooth descent to very low pitch, suggesting conclusion/resolution.
    Logarithmic or exponential curve for natural pitch perception.
    """
    start_freq = config['start_freq']
    end_freq = config['end_freq']
    descent_curve_type = config.get('descent_curve', 'logarithmic')

    # Descent spans bloom + decay phases
    bloom_start = phases['bloom']['start_sample']
    decay_end = phases['decay']['end_sample']

    curve = np.zeros(total_samples)

    # Pre-bloom: hold at start
    curve[:bloom_start] = start_freq

    # Bloom + decay: gradual descent
    descent_samples = decay_end - bloom_start
    if descent_samples > 0:
        if descent_curve_type == 'logarithmic':
            # Logarithmic descent (perceptually linear in pitch)
            curve[bloom_start:decay_end] = np.exp(
                np.linspace(np.log(start_freq), np.log(end_freq), descent_samples)
            )
        else:  # exponential
            # Exponential descent (accelerating fall)
            t = np.linspace(0, 1, descent_samples)
            curve[bloom_start:decay_end] = start_freq - (start_freq - end_freq) * (t ** 3)

    # Residue: hold at end (very low)
    curve[decay_end:] = end_freq

    return curve


def _pitch_discrete_chimes(config: Dict[str, Any],
                           phases: Dict[str, Any],
                           section_context: Dict[str, Any],
                           total_samples: int,
                           rng: np.random.Generator,
                           sample_rate: int) -> np.ndarray:
    """
    Discrete chime notes for INACCURACY archetype.

    Creates separate struck notes like wind chimes or xylophone,
    not continuous tones. Each note has instant attack with
    sustain at constant pitch (no glide).

    Like: "ting... ting... ting..." (discrete strikes)
    Not: "wooooop" (continuous glide)
    """
    num_notes = config.get('num_notes', 3)
    base_freq = config['start_freq_base']
    pitch_variation_semitones = config.get('pitch_variation_semitones', 3)

    curve = np.zeros(total_samples)
    velocity_curve = np.ones(total_samples)  # Velocity multiplier for each sample

    # Divide gesture into note segments
    bloom_start = phases['bloom']['start_sample']
    decay_end = phases['decay']['end_sample']
    available_duration = decay_end - bloom_start

    if available_duration <= 0 or num_notes == 0:
        curve[:] = base_freq
        return curve

    # Calculate note timing
    note_duration_samples = available_duration // num_notes
    gap_ratio = 0.3  # 30% gap between notes

    # Generate discrete notes
    # Default to silence during bloom for clear gaps; notes will set non-zero velocity
    velocity_curve[bloom_start:decay_end] = 0.0
    for i in range(num_notes):
        # Base start with slight jitter for wind variability (±20% of note duration)
        base_start = bloom_start + i * note_duration_samples
        jitter = int((note_duration_samples * 0.2) * (rng.random() - 0.5) * 2)
        note_start = max(bloom_start, min(base_start + jitter, decay_end - 1))
        # Ensure first note does not start exactly at bloom boundary to avoid abrupt onset
        if i == 0:
            min_offset = int(0.005 * sample_rate)  # 5ms
            note_start = max(note_start, bloom_start + min_offset)
        note_length = int(note_duration_samples * (1 - gap_ratio))
        note_end = min(note_start + note_length, total_samples)
        # Allow ringing to extend into the gap up to the next slot boundary
        ring_end = min(note_start + note_duration_samples, total_samples)

        if note_start >= total_samples:
            break

        # Pitch for this note (parabolic arc across sequence)
        # First note: base
        # Middle notes: rise
        # Last note: fall back
        progress = i / max(1, num_notes - 1)
        if progress < 0.5:
            # Rising phase
            pitch_offset = pitch_variation_semitones * (progress * 2)
        else:
            # Falling phase
            pitch_offset = pitch_variation_semitones * (2 - progress * 2)

        note_freq = base_freq * (2 ** (pitch_offset / 12))

        # Random velocity for this note (wind chime randomness)
        # Softer overall to avoid harshness
        velocity = 0.2 + 0.5 * rng.random()  # Range ~0.2 to 0.7

        # Constant pitch for this note including ring tail
        curve[note_start:ring_end] = note_freq

        # Per-note strike envelope: fast attack, exponential decay
        seg_len = max(1, ring_end - note_start)
        t = np.arange(seg_len) / sample_rate
        # Attack ~6ms (raised-cosine) for clickless onset
        attack_s = 0.006
        attack_samples = max(1, int(attack_s * sample_rate))
        attack_samples = min(attack_samples, seg_len)
        per_note_env = np.ones(seg_len)
        # Raised-cosine (half-Hann) attack for ultra-smooth onset
        per_note_env[:attack_samples] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, attack_samples))
        # Exponential decay over segment
        if seg_len - attack_samples > 0:
            decay_t = np.linspace(0, 1, seg_len - attack_samples)
            per_note_env[attack_samples:] = np.exp(-1.1 * decay_t)

        # Accumulate tails so multiple notes can overlap naturally
        current = velocity_curve[note_start:ring_end]
        added = velocity * per_note_env
        summed = current + added
        # Soft clip to 1.0 to avoid overs
        velocity_curve[note_start:ring_end] = np.minimum(1.0, summed)

    # Pre-bloom and post-decay: silence (or very low)
    curve[:bloom_start] = base_freq * 0.5
    curve[decay_end:] = base_freq * 0.5
    velocity_curve[:bloom_start] = 0.0
    velocity_curve[decay_end:] = 0.0

    # Store velocity curve in config for later use (hacky but works)
    config['_velocity_curve'] = velocity_curve

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
    elif harmony_type == "minimal_dyad":
        return _harmony_minimal_dyad(config, base_pitch_curve, phases, section_context, rng)
    elif harmony_type == "collision_cluster":
        return _harmony_collision_cluster(config, base_pitch_curve, phases, section_context, rng)
    elif harmony_type == "shifting_voices":
        return _harmony_shifting_voices(config, base_pitch_curve, phases, section_context, rng)
    elif harmony_type == "resolving_to_root":
        return _harmony_resolving_to_root(config, base_pitch_curve, phases, section_context, rng)
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


def _harmony_minimal_dyad(config: Dict[str, Any],
                          base_pitch_curve: np.ndarray,
                          phases: Dict[str, Any],
                          section_context: Dict[str, Any],
                          rng: np.random.Generator) -> List[np.ndarray]:
    """
    Minimal dyad for INACCURACY archetype.

    Creates simple two-voice harmony with fixed interval.
    Minimal harmonic complexity, suggesting hesitation or simplicity.
    """
    num_voices = config['num_voices']
    interval_semitones = config['interval_semitones']

    voices = []

    # Voice 1: base pitch
    voices.append(base_pitch_curve.copy())

    # Voice 2: transposed by interval
    if num_voices >= 2:
        interval_mult = 2 ** (interval_semitones / 12)
        voices.append(base_pitch_curve * interval_mult)

    return voices


def _harmony_collision_cluster(config: Dict[str, Any],
                               base_pitch_curve: np.ndarray,
                               phases: Dict[str, Any],
                               section_context: Dict[str, Any],
                               rng: np.random.Generator) -> List[np.ndarray]:
    """
    Collision cluster for FIRST_EXCHANGE archetype.

    Creates tight, high-density cluster at impact moment.
    Voices converge tightly, suggesting collision or confrontation.
    """
    num_voices = config['num_voices']
    impact_density = config['impact_density']

    # During impact phase, voices cluster tightly
    impact_start = phases['impact']['start_sample']
    impact_end = phases['impact']['end_sample']

    voices = []
    for i in range(num_voices):
        voice_curve = base_pitch_curve.copy()

        # Tight clustering during impact (±1 semitone spread)
        cluster_offset = (i - num_voices // 2) * impact_density
        cluster_mult = 2 ** (cluster_offset / 12)

        # Apply clustering during impact only
        voice_curve[impact_start:impact_end] *= cluster_mult

        voices.append(voice_curve)

    return voices


def _harmony_shifting_voices(config: Dict[str, Any],
                             base_pitch_curve: np.ndarray,
                             phases: Dict[str, Any],
                             section_context: Dict[str, Any],
                             rng: np.random.Generator) -> List[np.ndarray]:
    """
    Shifting voices for SIGNIFICANT_SHIFT archetype.

    Creates voices that gradually shift position relative to each other.
    Suggests strategic repositioning or gradual change.
    """
    num_voices = config['num_voices']
    shift_rate = config['shift_rate']

    # Bloom phase is where shift occurs
    bloom_start = phases['bloom']['start_sample']
    bloom_end = phases['bloom']['end_sample']
    bloom_samples = bloom_end - bloom_start

    voices = []
    for i in range(num_voices):
        voice_curve = base_pitch_curve.copy()

        # Each voice shifts at different rate during bloom
        if bloom_samples > 0:
            # Shift amount increases with voice index
            shift_semitones = (i - num_voices // 2) * shift_rate * 7  # Up to ±3.5 semitones

            # Gradual shift over bloom phase
            shift_curve = np.linspace(0, shift_semitones, bloom_samples)
            shift_mult = 2 ** (shift_curve / 12)

            voice_curve[bloom_start:bloom_end] *= shift_mult

        voices.append(voice_curve)

    return voices


def _harmony_resolving_to_root(config: Dict[str, Any],
                               base_pitch_curve: np.ndarray,
                               phases: Dict[str, Any],
                               section_context: Dict[str, Any],
                               rng: np.random.Generator) -> List[np.ndarray]:
    """
    Resolving to root for FINAL_RESOLUTION archetype.

    Creates voices that converge from chord to single root note.
    Suggests harmonic resolution and closure.
    """
    num_voices_start = config['num_voices_start']
    resolution_chord = config.get('resolution_chord', 'tonic')

    # Define chord intervals (before resolution)
    if resolution_chord == 'tonic':
        intervals = [0, 4, 7, 12, 16]  # Major chord with extensions
    else:
        intervals = [0, 3, 7, 10, 14]  # Minor chord with extensions

    # Resolution occurs during decay phase
    decay_start = phases['decay']['start_sample']
    decay_end = phases['decay']['end_sample']
    decay_samples = decay_end - decay_start

    voices = []
    for i in range(min(num_voices_start, len(intervals))):
        voice_curve = base_pitch_curve.copy()
        interval = intervals[i]

        # Before decay: hold chord interval
        if decay_start > 0:
            interval_mult = 2 ** (interval / 12)
            voice_curve[:decay_start] *= interval_mult

        # During decay: converge to root
        if decay_samples > 0:
            # Linear convergence from interval to unison
            start_mult = 2 ** (interval / 12)
            convergence = np.linspace(start_mult, 1.0, decay_samples)
            voice_curve[decay_start:decay_end] = base_pitch_curve[decay_start:decay_end] * convergence

        # After decay: all at root (unison)
        # Already at base_pitch_curve

        voices.append(voice_curve)

    # Return only one voice at the end (others fade in envelope)
    return voices[:1] if decay_end < len(base_pitch_curve) else voices


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
    elif filter_type == "gentle_bandpass":
        return _filter_gentle_bandpass(config, phases, section_context, total_samples, rng)
    elif filter_type == "gradual_sweep":
        return _filter_gradual_sweep(config, phases, section_context, total_samples, rng)
    elif filter_type == "impact_spike":
        return _filter_impact_spike(config, phases, section_context, total_samples, rng)
    elif filter_type == "closing_focus":
        return _filter_closing_focus(config, phases, section_context, total_samples, rng)
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


def _filter_gentle_bandpass(config: Dict[str, Any],
                            phases: Dict[str, Any],
                            section_context: Dict[str, Any],
                            total_samples: int,
                            rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Gentle bandpass for INACCURACY archetype.

    Static bandpass filter with moderate Q, gentle character.
    Creates subtle tonal focus without aggressive filtering.
    """
    bp_center = config['bp_center_base']
    bp_bandwidth = config.get('bp_bandwidth', 500)
    resonance_val = config.get('resonance', 0.35)

    # Static filter - no modulation
    cutoff = np.full(total_samples, bp_center)
    resonance = np.full(total_samples, resonance_val)

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'bandpass'
    }


def _filter_gradual_sweep(config: Dict[str, Any],
                         phases: Dict[str, Any],
                         section_context: Dict[str, Any],
                         total_samples: int,
                         rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Gradual sweep for SIGNIFICANT_SHIFT archetype.

    Slow lowpass sweep with sigmoid curve for smooth transition.
    Suggests gradual opening or closing of tonal space.
    """
    lp_start = config['lp_start']
    lp_end = config['lp_end']
    sweep_curve_type = config.get('sweep_curve', 'sigmoid')

    # Sweep occurs during bloom phase
    bloom_start = phases['bloom']['start_sample']
    bloom_end = phases['bloom']['end_sample']

    cutoff = np.zeros(total_samples)

    # Pre-bloom: hold at start
    cutoff[:bloom_start] = lp_start

    # Bloom: gradual sweep
    sweep_samples = bloom_end - bloom_start
    if sweep_samples > 0:
        if sweep_curve_type == 'sigmoid':
            # Sigmoid for smooth acceleration/deceleration
            t = np.linspace(-3, 3, sweep_samples)
            sigmoid = 1 / (1 + np.exp(-t))
            cutoff[bloom_start:bloom_end] = lp_start + (lp_end - lp_start) * sigmoid
        else:  # linear
            cutoff[bloom_start:bloom_end] = np.linspace(lp_start, lp_end, sweep_samples)

    # Post-bloom: hold at end
    cutoff[bloom_end:] = lp_end

    # Moderate resonance throughout
    resonance = np.full(total_samples, 0.4)

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'lowpass'
    }


def _filter_impact_spike(config: Dict[str, Any],
                        phases: Dict[str, Any],
                        section_context: Dict[str, Any],
                        total_samples: int,
                        rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Impact spike for FIRST_EXCHANGE archetype.

    Sharp bandpass spike at impact moment with high resonance.
    Creates focused, resonant 'ping' at collision.
    """
    bp_center = config['bp_center']
    bp_bandwidth = config.get('bp_bandwidth', 900)
    impact_resonance = config['impact_resonance']

    # Impact phase is where spike occurs
    impact_start = phases['impact']['start_sample']
    impact_end = phases['impact']['end_sample']
    bloom_end = phases['bloom']['end_sample']

    # Center frequency - stable at impact point
    cutoff = np.full(total_samples, bp_center)

    # Resonance curve: spike at impact, decay after
    resonance = np.zeros(total_samples)

    # Pre-impact: low resonance
    resonance[:impact_start] = 0.2

    # Impact: high resonance spike
    resonance[impact_start:impact_end] = impact_resonance

    # Bloom: exponential decay of resonance
    decay_samples = bloom_end - impact_end
    if decay_samples > 0:
        decay_curve = np.exp(np.linspace(0, -3, decay_samples))
        resonance[impact_end:bloom_end] = 0.2 + (impact_resonance - 0.2) * decay_curve

    # Post-bloom: low resonance
    resonance[bloom_end:] = 0.2

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'bandpass'
    }


def _filter_closing_focus(config: Dict[str, Any],
                          phases: Dict[str, Any],
                          section_context: Dict[str, Any],
                          total_samples: int,
                          rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Closing focus for FINAL_RESOLUTION archetype.

    Lowpass filter that gradually closes down to very low frequency.
    Final resonance peak adds subtle emphasis to conclusion.
    """
    lp_start = config['lp_start']
    lp_end = config['lp_end']
    final_resonance = config.get('final_resonance', 0.52)

    # Closing occurs during bloom + decay
    bloom_start = phases['bloom']['start_sample']
    decay_end = phases['decay']['end_sample']

    cutoff = np.zeros(total_samples)

    # Pre-bloom: open filter
    cutoff[:bloom_start] = lp_start

    # Bloom + decay: gradual closing (logarithmic for perceptual linearity)
    closing_samples = decay_end - bloom_start
    if closing_samples > 0:
        cutoff[bloom_start:decay_end] = np.exp(
            np.linspace(np.log(lp_start), np.log(lp_end), closing_samples)
        )

    # Residue: hold at very low
    cutoff[decay_end:] = lp_end

    # Resonance: build slightly toward end for subtle emphasis
    resonance = np.zeros(total_samples)
    resonance[:bloom_start] = 0.3

    if closing_samples > 0:
        # Linear rise in resonance as filter closes
        resonance[bloom_start:decay_end] = np.linspace(0.3, final_resonance, closing_samples)

    resonance[decay_end:] = final_resonance

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

    Phase-aware envelope:
    - pre_shadow: silent (0.0)
    - impact: attack rise (0.0 → 1.0)
    - bloom: sustain (1.0)
    - decay: exponential fall (1.0 → decay_floor)
    - residue: linear fade to silence (decay_floor → 0.0)
    """
    envelope = np.zeros(total_samples)

    # Extract phase boundaries
    pre_shadow = phases['pre_shadow']
    impact = phases['impact']
    bloom = phases['bloom']
    decay = phases['decay']
    residue = phases['residue']

    # Pre-shadow: GENTLE FADE-IN (furtive entrance)
    pre_samples = pre_shadow['duration_samples']
    if pre_samples > 0:
        # Exponential fade-in for smooth, gradual entrance
        envelope[pre_shadow['start_sample']:pre_shadow['end_sample']] = (
            0.1 * (1 - np.exp(-3 * np.linspace(0, 1, pre_samples)))
        )
        pre_end_level = envelope[pre_shadow['end_sample'] - 1]
    else:
        pre_end_level = 0.0

    # Impact: SHARP ATTACK using attack_ms parameter
    impact_samples = impact['duration_samples']
    if impact_samples > 0:
        # Use attack_ms from config for percussion-style sharp attacks
        attack_ms = config.get('attack_ms_base', 5)
        attack_samples = int((attack_ms / 1000) * sample_rate)
        attack_samples = min(attack_samples, impact_samples)  # Don't exceed impact phase

        if attack_samples > 0:
            # SHARP attack: rise to 1.0 over attack_samples
            t_attack = np.linspace(0, 1, attack_samples)
            envelope[impact['start_sample']:impact['start_sample'] + attack_samples] = (
                pre_end_level + (1.0 - pre_end_level) * (t_attack ** 1.5)  # Quick rise
            )
            # Rest of impact phase: hold at 1.0
            envelope[impact['start_sample'] + attack_samples:impact['end_sample']] = 1.0
        else:
            # Instant attack if attack_ms is tiny
            envelope[impact['start_sample']:impact['end_sample']] = 1.0

    # Bloom: sustain at maximum
    envelope[bloom['start_sample']:bloom['end_sample']] = 1.0

    # Decay: exponential fall
    decay_samples = decay['duration_samples']
    if decay_samples > 0:
        decay_coeff = config.get('decay_coefficient', -4)
        decay_curve = np.exp(np.linspace(0, decay_coeff, decay_samples))
        envelope[decay['start_sample']:decay['end_sample']] = decay_curve
        decay_floor = decay_curve[-1]  # Where decay ends
    else:
        decay_floor = 1.0

    # Residue: LINEAR FADE TO SILENCE (critical for furtive exit)
    residue_samples = residue['duration_samples']
    if residue_samples > 0:
        envelope[residue['start_sample']:residue['end_sample']] = np.linspace(
            decay_floor, 0.0, residue_samples
        )

    return envelope


def _envelope_gradual_sustained(config: Dict[str, Any],
                               phases: Dict[str, Any],
                               total_samples: int,
                               rng: np.random.Generator,
                               sample_rate: int = 88200) -> np.ndarray:
    """
    Gradual attack with sustained plateau for BRILLIANT archetype.

    Phase-aware envelope:
    - pre_shadow: slow fade in (0.0 → 0.3)
    - impact: continue rise (0.3 → 0.8)
    - bloom: full sustain (0.8 → 1.0 → 1.0)
    - decay: gentle fall (1.0 → decay_floor)
    - residue: fade to silence (decay_floor → 0.0)
    """
    envelope = np.zeros(total_samples)

    # Extract phase boundaries
    pre_shadow = phases['pre_shadow']
    impact = phases['impact']
    bloom = phases['bloom']
    decay = phases['decay']
    residue = phases['residue']

    # Pre-shadow: gentle fade-in (furtive entrance)
    pre_samples = pre_shadow['duration_samples']
    if pre_samples > 0:
        envelope[pre_shadow['start_sample']:pre_shadow['end_sample']] = np.linspace(
            0.0, 0.3, pre_samples
        )

    # Impact: continue rising to near-full
    impact_samples = impact['duration_samples']
    if impact_samples > 0:
        envelope[impact['start_sample']:impact['end_sample']] = np.linspace(
            0.3, 0.8, impact_samples
        )

    # Bloom: reach full and sustain (S-curve for smooth bloom)
    bloom_samples = bloom['duration_samples']
    if bloom_samples > 0:
        bloom_third = bloom_samples // 3
        # First third: reach 1.0
        envelope[bloom['start_sample']:bloom['start_sample'] + bloom_third] = np.linspace(
            0.8, 1.0, bloom_third
        )
        # Rest: sustain
        envelope[bloom['start_sample'] + bloom_third:bloom['end_sample']] = 1.0

    # Decay: linear or exponential fall
    decay_samples = decay['duration_samples']
    if decay_samples > 0:
        if config.get('decay_curve', 'linear') == 'linear':
            decay_curve = np.linspace(1.0, 0.0, decay_samples)
        else:
            decay_curve = np.exp(np.linspace(0, -3, decay_samples))
        envelope[decay['start_sample']:decay['end_sample']] = decay_curve
        decay_floor = decay_curve[-1]
    else:
        decay_floor = 1.0

    # Residue: LINEAR FADE TO SILENCE
    residue_samples = residue['duration_samples']
    if residue_samples > 0:
        envelope[residue['start_sample']:residue['end_sample']] = np.linspace(
            decay_floor, 0.0, residue_samples
        )

    return envelope


def _envelope_gated_pulse(config: Dict[str, Any],
                         phases: Dict[str, Any],
                         total_samples: int,
                         rng: np.random.Generator,
                         sample_rate: int = 88200) -> np.ndarray:
    """
    Gated pulses for TACTICAL_SEQUENCE archetype.

    Phase-aware envelope with rhythmic gates:
    - pre_shadow: silent fade-in
    - impact/bloom/decay: rhythmic gating at pulse_rate_hz
    - residue: final fade to silence
    """
    envelope = np.zeros(total_samples)

    # Extract phase boundaries
    pre_shadow = phases['pre_shadow']
    impact = phases['impact']
    bloom = phases['bloom']
    decay = phases['decay']
    residue = phases['residue']

    attack_ms = config.get('attack_ms', 5)
    attack_samples = max(1, int(attack_ms * sample_rate / 1000))

    release_ms = config.get('release_ms', 20)
    release_samples = max(1, int(release_ms * sample_rate / 1000))

    pulse_hz = config['pulse_rate_hz']

    # Pre-shadow: gentle fade-in (no gates yet)
    pre_samples = pre_shadow['duration_samples']
    if pre_samples > 0:
        envelope[pre_shadow['start_sample']:pre_shadow['end_sample']] = np.linspace(
            0.0, 0.2, pre_samples
        )

    # Impact + Bloom + Decay: rhythmic gating
    gate_start = impact['start_sample']
    gate_end = decay['end_sample']
    gate_duration = gate_end - gate_start

    if gate_duration > 0:
        # Generate pulse train for gated region
        t = np.arange(gate_duration) / sample_rate
        pulse_phase = (t * pulse_hz) % 1.0
        gate_on = pulse_phase < 0.5
        gate_envelope = np.zeros(gate_duration)
        gate_envelope[gate_on] = 1.0

        # Apply attack/release to prevent clicks
        transitions = np.diff(gate_envelope, prepend=0, append=0)
        rise_edges = np.where(transitions > 0)[0]
        fall_edges = np.where(transitions < 0)[0]

        for edge in rise_edges:
            ramp_end = min(edge + attack_samples, gate_duration)
            ramp_len = ramp_end - edge
            if ramp_len > 0:
                gate_envelope[edge:ramp_end] *= 1 - np.exp(-5 * np.linspace(0, 1, ramp_len))

        for edge in fall_edges:
            ramp_start = max(edge - release_samples, 0)
            ramp_len = edge - ramp_start
            if ramp_len > 0:
                gate_envelope[ramp_start:edge] *= np.exp(-3 * np.linspace(0, 1, ramp_len))

        # Apply to envelope
        envelope[gate_start:gate_end] = gate_envelope

    # Residue: FADE TO SILENCE (blend from last gate state to 0.0)
    residue_samples = residue['duration_samples']
    if residue_samples > 0:
        residue_start_val = envelope[residue['start_sample'] - 1] if residue['start_sample'] > 0 else 0.0
        envelope[residue['start_sample']:residue['end_sample']] = np.linspace(
            residue_start_val, 0.0, residue_samples
        )

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
        'shimmer_enable': config.get('shimmer_enable', False),
        'waveform': config.get('waveform', 'sine')  # Pass through waveform selection
    }

    # Add shimmer parameters if enabled
    if texture['shimmer_enable']:
        texture['shimmer_rate_hz'] = config.get('shimmer_rate_hz', 6.0)
        texture['shimmer_depth'] = config.get('shimmer_depth', 0.3)

    return texture
