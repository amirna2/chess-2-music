# Layer3b Configuration Reference

Quick reference for configuration parameters and valid values.

## Contents

- [String Values](#string-values)
- [Root Parameters](#root-parameters)
- [Phases](#phases)
- [Pitch](#pitch)
- [Harmony](#harmony)
- [Filter](#filter)
- [Envelope](#envelope)
- [Texture](#texture)
- [Example](#example)

## String Values

### noise_type
```
"white"  - supported
"pink"   - supported
```

### waveform
```
"sine"     - supported
"triangle" - supported
"saw"      - NOT supported (time-varying pitch limitation)
"square"   - NOT supported (time-varying pitch limitation)
```

### decay_curve
```
sudden_short_tail:  "exponential", "linear"
gradual_sustained:  "linear", "exponential", "sigmoid" (untested)
```

## Root Parameters

| Parameter | Type | Range |
|-----------|------|-------|
| duration_base | float | 0.5 - 5.0 |
| duration_tension_scale | float | 0.0 - 2.0 |
| duration_entropy_scale | float | -1.0 - 1.0 |
| peak_limit | float | 0.1 - 1.0 |
| rms_target | float | -40 to -10 dB |

## Phases

Must sum to 1.0

| Phase | Range |
|-------|-------|
| pre_shadow | 0.05 - 0.25 |
| impact | 0.05 - 0.20 |
| bloom | 0.15 - 0.50 |
| decay | 0.20 - 0.65 |
| residue | 0.10 - 0.20 |

## Pitch

### Implemented Types

| Type | Parameters |
|------|------------|
| discrete_chimes | start_freq_base, num_notes, pitch_variation_semitones |
| weak_parabolic | start_freq_base, peak_freq_mult, end_freq_mult |
| slow_drift | start_freq_base, drift_semitones, drift_direction |
| impact_transient | strike_freq_base, decay_mult |
| final_descent | start_freq, end_freq, descent_curve |
| exponential_gliss | start_freq_base, octave_drop_base, gliss_phases |
| ascending_spread | start_freq_base, octave_rise, spread_phases |
| oscillating_tremor | center_freq_base, tremor_rate_base_hz, tremor_depth_semitones |
| cellular_sequence | parameters TBD |

### discrete_chimes

| Parameter | Type | Range |
|-----------|------|-------|
| start_freq_base | float | 110 - 1760 Hz |
| num_notes | int | 1 - 8 |
| pitch_variation_semitones | float | 0 - 12 |

## Harmony

### Implemented Types

| Type | Parameters |
|------|------------|
| minimal_dyad | num_voices (1 or 2), interval_semitones |
| collision_cluster | num_voices, impact_density |
| shifting_voices | num_voices, shift_rate |
| resolving_to_root | num_voices_start, num_voices_end, resolution_chord |
| dense_cluster | num_voices, semitone_spacing_min, semitone_spacing_max |
| cluster_to_interval | num_voices_base, cluster_semitone_spacing, resolve_interval_type |
| unison_to_chord | chord_type, num_voices, spread_phase |
| harmonic_stack | parameters TBD |

### minimal_dyad

| Parameter | Type | Range | Notes |
|-----------|------|-------|-------|
| num_voices | int | 1 or 2 | 1 = single voice |
| interval_semitones | int | 0, 7, 12 | 0=unison, 7=fifth, 12=octave |

## Filter

### Implemented Types

| Type | Parameters |
|------|------------|
| gentle_bandpass | bp_center_base, bp_bandwidth, resonance |
| gradual_sweep | lp_start, lp_end, sweep_curve |
| impact_spike | bp_center, bp_bandwidth, impact_resonance |
| closing_focus | lp_start, lp_end, final_resonance |
| bandpass_sweep | bp_center_start, bp_center_end, bp_bandwidth, sweep_phase |
| lowpass_to_highpass_open | lp_cutoff_start, hp_cutoff_end, resonance_base, morph_phase |
| bandpass_to_lowpass_choke | bp_center_base, bp_bandwidth, lp_cutoff_base, morph_phase |
| rhythmic_gate | parameters TBD |

### gentle_bandpass

| Parameter | Type | Range |
|-----------|------|-------|
| bp_center_base | float | 200 - 10000 Hz |
| bp_bandwidth | float | 500 - 20000 Hz |
| resonance | float | 0.0 - 0.9 |

## Envelope

### Implemented Types

| Type | Parameters |
|------|------------|
| sudden_short_tail | attack_ms_base, attack_ms_entropy_scale, sustain_phase_ratio, decay_curve, decay_coefficient |
| gradual_sustained | attack_ms, sustain_phase_ratio, decay_curve |
| gated_pulse | attack_ms, gate_duration_ms, release_ms, pulse_rate_hz |

### sudden_short_tail

| Parameter | Type | Range |
|-----------|------|-------|
| attack_ms_base | float | 0.5 - 100 ms |
| attack_ms_entropy_scale | float | 0 - 50 |
| sustain_phase_ratio | float | 0.05 - 0.50 |
| decay_curve | string | "exponential" or "linear" |
| decay_coefficient | float | -6.0 to -1.0 |

### gradual_sustained

| Parameter | Type | Range |
|-----------|------|-------|
| attack_ms | float | 10 - 250 ms |
| sustain_phase_ratio | float | 0.20 - 0.70 |
| decay_curve | string | "linear", "exponential", "sigmoid" |

## Texture

| Parameter | Type | Range | Valid Values |
|-----------|------|-------|--------------|
| noise_ratio_base | float | 0.0 - 1.0 | |
| noise_ratio_entropy_scale | float | 0.0 - 0.5 | |
| noise_type | string | | "white", "pink" |
| waveform | string | | "sine", "triangle" |
| shimmer_enable | bool | | true, false |
| shimmer_rate_hz | float | 2.0 - 12.0 | |

## Example

Complete configuration for INACCURACY archetype:

```python
"INACCURACY": {
    "duration_base": 1.5,
    "duration_tension_scale": 0.2,
    "duration_entropy_scale": 0.0,

    "phases": {
        "pre_shadow": 0.05,
        "impact": 0.08,
        "bloom": 0.50,
        "decay": 0.27,
        "residue": 0.10
    },

    "pitch": {
        "type": "discrete_chimes",
        "start_freq_base": 880,
        "num_notes": 3,
        "pitch_variation_semitones": 3
    },

    "harmony": {
        "type": "minimal_dyad",
        "num_voices": 1,
        "interval_semitones": 0
    },

    "filter": {
        "type": "gentle_bandpass",
        "bp_center_base": 2200,
        "bp_bandwidth": 3500,
        "resonance": 0.15
    },

    "envelope": {
        "type": "sudden_short_tail",
        "attack_ms_base": 10,
        "attack_ms_entropy_scale": 0.5,
        "sustain_phase_ratio": 0.30,
        "decay_curve": "exponential",
        "decay_coefficient": -2.0
    },

    "texture": {
        "noise_ratio_base": 0.0,
        "noise_ratio_entropy_scale": 0.0,
        "noise_type": "white",
        "waveform": "triangle"
    },

    "peak_limit": 0.25,
    "rms_target": -32.0,

    "morphology": {
        "spectromorphological_archetype": "Attack–Decay",
        "gesture_class": "Gentle / Hesitant",
        "motion_type": "Parabolic (small rise–fall)"
    }
}
```
