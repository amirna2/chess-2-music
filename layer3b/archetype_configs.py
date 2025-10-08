"""
Archetype configuration definitions for Layer 3b gestures.

Each archetype defines its characteristic sound through configuration parameters
rather than custom code. All archetypes share the same synthesis pipeline.
"""

from typing import Dict, Any

ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "BLUNDER": {
        "duration_base": 2.5,
        "duration_tension_scale": 1.2,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.05,
            "bloom": 0.25,
            "decay": 0.35,
            "residue": 0.20
        },
        "pitch": {
            "type": "exponential_gliss",
            "start_freq_base": 880,
            "start_freq_entropy_scale": 440,
            "octave_drop_base": 2,
            "octave_drop_tension_scale": 1,
            "gliss_phases": ["bloom", "decay"]
        },
        "harmony": {
            "type": "cluster_to_interval",
            "num_voices_base": 3,
            "num_voices_tension_scale": 2,
            "cluster_semitone_spacing": 1.0,
            "resolve_interval_type": "muddy",
            "resolve_phase": "decay"
        },
        "filter": {
            "type": "bandpass_to_lowpass_choke",
            "bp_center_base": 1000,
            "bp_center_tension_scale": 1000,
            "bp_bandwidth": 800,
            "lp_cutoff_base": 100,
            "lp_cutoff_tension_scale": 100,
            "lp_resonance_base": 0.7,
            "lp_resonance_tension_scale": 0.25,
            "morph_phase": "bloom"
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 1,
            "attack_ms_entropy_scale": 4,
            "sustain_phase_ratio": 0.15,
            "decay_curve": "exponential",
            "decay_coefficient": -4
        },
        "texture": {
            "noise_ratio_base": 0.3,
            "noise_ratio_entropy_scale": 0.4,
            "noise_type": "pink"
        },
        "peak_limit": 0.8,
        "rms_target": -18.0
    },
    "BRILLIANT": {
        "duration_base": 3.0,
        "duration_tension_scale": 0.8,
        "duration_entropy_scale": -0.5,
        "phases": {
            "pre_shadow": 0.20,
            "impact": 0.10,
            "bloom": 0.40,
            "decay": 0.20,
            "residue": 0.10
        },
        "pitch": {
            "type": "ascending_spread",
            "start_freq_base": 220,
            "start_freq_entropy_scale": 110,
            "octave_rise": 2,
            "spread_phases": ["impact", "bloom"]
        },
        "harmony": {
            "type": "unison_to_chord",
            "chord_type": "major_seventh",
            "num_voices": 4,
            "spread_phase": "bloom"
        },
        "filter": {
            "type": "lowpass_to_highpass_open",
            "lp_cutoff_start": 300,
            "hp_cutoff_end": 3000,
            "resonance_base": 0.4,
            "morph_phase": "bloom"
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 50,
            "sustain_phase_ratio": 0.5,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.1,
            "noise_ratio_entropy_scale": 0.2,
            "noise_type": "white",
            "shimmer_enable": True,
            "shimmer_rate_hz": 6.0
        },
        "peak_limit": 0.85,
        "rms_target": -16.0
    },
    "TIME_PRESSURE": {
        "duration_base": 1.8,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.3,
        "phases": {
            "pre_shadow": 0.10,
            "impact": 0.15,
            "bloom": 0.30,
            "decay": 0.25,
            "residue": 0.20
        },
        "pitch": {
            "type": "oscillating_tremor",
            "center_freq_base": 440,
            "center_freq_tension_scale": 220,
            "tremor_rate_base_hz": 8,
            "tremor_rate_tension_scale_hz": 8,
            "tremor_depth_semitones": 2,
            "acceleration_phase": "bloom"
        },
        "harmony": {
            "type": "dense_cluster",
            "num_voices": 5,
            "semitone_spacing_min": 0.5,
            "semitone_spacing_max": 1.5
        },
        "filter": {
            "type": "bandpass_sweep",
            "bp_center_start": 500,
            "bp_center_end": 2000,
            "bp_bandwidth": 600,
            "sweep_rate_modulation": "accelerating",
            "sweep_phase": "bloom"
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 2,
            "attack_ms_entropy_scale": 3,
            "sustain_phase_ratio": 0.20,
            "decay_curve": "exponential",
            "decay_coefficient": -5
        },
        "texture": {
            "noise_ratio_base": 0.5,
            "noise_ratio_entropy_scale": 0.3,
            "noise_type": "white"
        },
        "peak_limit": 0.75,
        "rms_target": -20.0
    },
    "TACTICAL_SEQUENCE": {
        "duration_base": 2.2,
        "duration_tension_scale": 0.6,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.12,
            "impact": 0.08,
            "bloom": 0.35,
            "decay": 0.30,
            "residue": 0.15
        },
        "pitch": {
            "type": "cellular_sequence",
            "cell_frequencies": [330, 440, 550, 660],
            "cell_duration_ms": 80,
            "transition_type": "stepped",
            "pattern_seed": "golden_ratio"
        },
        "harmony": {
            "type": "harmonic_stack",
            "num_voices": 3,
            "interval_ratios": [1.0, 1.5, 2.0],
            "phase_offset_enable": True
        },
        "filter": {
            "type": "rhythmic_gate",
            "lp_cutoff_base": 800,
            "lp_cutoff_pulse_mult": 3.0,
            "pulse_rate_hz": 12,
            "pulse_duty_cycle": 0.3
        },
        "envelope": {
            "type": "gated_pulse",
            "attack_ms": 5,
            "gate_duration_ms": 60,
            "release_ms": 20,
            "pulse_rate_hz": 12
        },
        "texture": {
            "noise_ratio_base": 0.05,
            "noise_ratio_entropy_scale": 0.1,
            "noise_type": "pink"
        },
        "peak_limit": 0.8,
        "rms_target": -17.0
    }
}
