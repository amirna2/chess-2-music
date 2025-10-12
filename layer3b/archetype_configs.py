"""
Archetype configuration definitions for Layer 3b gestures.

Each archetype defines its characteristic sound through configuration parameters
rather than custom code. All archetypes share the same synthesis pipeline.
"""

from typing import Dict, Any

ARCHETYPES: Dict[str, Dict[str, Any]] = {
    # 1. MOVE - Attack–Decay (neutral) - Simple transitional gesture
    "MOVE": {
        "duration_base": 1.2,
        "duration_tension_scale": 0.3,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.10,
            "impact": 0.15,
            "bloom": 0.20,
            "decay": 0.35,
            "residue": 0.20
        },
        "pitch": {
            "type": "stable",
            "center_freq_base": 440,
            "center_freq_entropy_scale": 110
        },
        "harmony": {
            "type": "simple_unison",
            "num_voices": 2
        },
        "filter": {
            "type": "simple_lowpass",
            "lp_cutoff_base": 1200,
            "resonance_base": 0.3
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 3,
            "attack_ms_entropy_scale": 2,
            "sustain_phase_ratio": 0.20,
            "decay_curve": "exponential",
            "decay_coefficient": -3
        },
        "texture": {
            "noise_ratio_base": 0.15,
            "noise_ratio_entropy_scale": 0.1,
            "noise_type": "pink"
        },
        "peak_limit": 0.7,
        "rms_target": -20.0,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay",
            "gesture_class": "Neutral / Transitional",
            "motion_type": "Linear motion"
        }
    },

    # 2. GAME_CHANGING - Graduated Continuant - Transformative expansion
    "GAME_CHANGING": {
        "duration_base": 3.5,
        "duration_tension_scale": 1.0,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.10,
            "impact": 0.15,
            "bloom": 0.45,
            "decay": 0.20,
            "residue": 0.10
        },
        "pitch": {
            "type": "ascending_spread",
            "start_freq_base": 220,
            "start_freq_entropy_scale": 110,
            "octave_rise": 2.5,
            "spread_phases": ["impact", "bloom"]
        },
        "harmony": {
            "type": "unison_to_chord",
            "chord_type": "major_seventh",
            "num_voices": 6,
            "spread_phase": "bloom"
        },
        "filter": {
            "type": "lowpass_to_highpass_open",
            "lp_cutoff_start": 300,
            "hp_cutoff_end": 4000,
            "resonance_base": 0.5,
            "morph_phase": "impact"
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 40,
            "sustain_phase_ratio": 0.50,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.20,
            "noise_ratio_entropy_scale": 0.3,
            "noise_type": "white",
            "shimmer_enable": True,
            "shimmer_rate_hz": 7.0
        },
        "peak_limit": 0.90,
        "rms_target": -14.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Expansive / Transformative",
            "motion_type": "Agglomeration–Divergence"
        }
    },

    # 3. CRITICAL_SWING - Attack–Decay (accented) - Dynamic parabolic gesture
    "CRITICAL_SWING": {
        "duration_base": 2.8,
        "duration_tension_scale": 0.9,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.12,
            "impact": 0.18,
            "bloom": 0.30,
            "decay": 0.28,
            "residue": 0.12
        },
        "pitch": {
            "type": "parabolic",
            "start_freq_base": 330,
            "peak_freq_mult": 2.0,
            "end_freq_mult": 1.2
        },
        "harmony": {
            "type": "cluster_to_interval",
            "num_voices_base": 4,
            "num_voices_tension_scale": 2,
            "cluster_semitone_spacing": 1.5,
            "resolve_interval_type": "fourth",
            "resolve_phase": "decay"
        },
        "filter": {
            "type": "bandpass_sweep",
            "bp_center_start": 800,
            "bp_center_end": 2200,
            "bp_bandwidth": 700,
            "sweep_rate_modulation": "parabolic",
            "sweep_phase": "bloom"
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 5,
            "attack_ms_entropy_scale": 3,
            "sustain_phase_ratio": 0.25,
            "decay_curve": "exponential",
            "decay_coefficient": -3.5
        },
        "texture": {
            "noise_ratio_base": 0.22,
            "noise_ratio_entropy_scale": 0.25,
            "noise_type": "pink"
        },
        "peak_limit": 0.85,
        "rms_target": -15.5,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay",
            "gesture_class": "Accented / Dynamic",
            "motion_type": "Parabolic rise–fall"
        }
    },

    # 4. SIGNIFICANT_SHIFT - Graduated Continuant - Gradual drift
    "SIGNIFICANT_SHIFT": {
        "duration_base": 3.2,
        "duration_tension_scale": 0.7,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.20,
            "impact": 0.12,
            "bloom": 0.38,
            "decay": 0.20,
            "residue": 0.10
        },
        "pitch": {
            "type": "slow_drift",
            "start_freq_base": 330,
            "drift_semitones": 7,
            "drift_direction": "ascending"
        },
        "harmony": {
            "type": "shifting_voices",
            "num_voices": 4,
            "shift_rate": 0.4
        },
        "filter": {
            "type": "gradual_sweep",
            "lp_start": 800,
            "lp_end": 1900,
            "sweep_curve": "sigmoid"
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 100,
            "sustain_phase_ratio": 0.48,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.18,
            "noise_ratio_entropy_scale": 0.20,
            "noise_type": "pink"
        },
        "peak_limit": 0.78,
        "rms_target": -17.5,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Gradual / Evolving",
            "motion_type": "Ascent or Descent drift"
        }
    },

    # 5. MATE_SEQUENCE - Composite Chain - Convergent resolution
    "MATE_SEQUENCE": {
        "duration_base": 3.8,
        "duration_tension_scale": 1.0,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.12,
            "bloom": 0.35,
            "decay": 0.28,
            "residue": 0.10
        },
        "pitch": {
            "type": "converging_steps",
            "start_freq_base": 440,
            "end_freq_base": 880,
            "num_steps": 6
        },
        "harmony": {
            "type": "converging_cluster",
            "num_voices_start": 6,
            "num_voices_end": 1,
            "convergence_phase": "decay"
        },
        "filter": {
            "type": "focusing_narrowband",
            "bp_start_width": 1400,
            "bp_end_width": 250,
            "center_freq": 880
        },
        "envelope": {
            "type": "stepped_convergence",
            "num_steps": 6,
            "final_accent_mult": 1.4
        },
        "texture": {
            "noise_ratio_base": 0.16,
            "noise_ratio_entropy_scale": 0.18,
            "noise_type": "white"
        },
        "peak_limit": 0.86,
        "rms_target": -15.5,
        "morphology": {
            "spectromorphological_archetype": "Composite Chain",
            "gesture_class": "Convergent / Resolving",
            "motion_type": "Convergence → Closure"
        }
    },

    # 6. DEEP_THINK - Sustained Continuant - Static plateau
    "DEEP_THINK": {
        "duration_base": 4.5,
        "duration_tension_scale": 1.8,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.25,
            "impact": 0.10,
            "bloom": 0.42,
            "decay": 0.13,
            "residue": 0.10
        },
        "pitch": {
            "type": "sustained_drone",
            "center_freq_base": 220,
            "micro_drift_amount": 0.08
        },
        "harmony": {
            "type": "dense_cluster",
            "num_voices": 8,
            "semitone_spacing_min": 0.8,
            "semitone_spacing_max": 2.0
        },
        "filter": {
            "type": "static_bandpass",
            "bp_center": 600,
            "bp_bandwidth": 450,
            "resonance": 0.65
        },
        "envelope": {
            "type": "plateau_sustained",
            "attack_ms": 250,
            "sustain_phase_ratio": 0.68
        },
        "texture": {
            "noise_ratio_base": 0.38,
            "noise_ratio_entropy_scale": 0.20,
            "noise_type": "pink"
        },
        "peak_limit": 0.68,
        "rms_target": -20.5,
        "morphology": {
            "spectromorphological_archetype": "Sustained Continuant",
            "gesture_class": "Static / Contained",
            "motion_type": "Stasis"
        }
    },

    # 7. TIME_PRESSURE - Iterative - Oscillatory tremor
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
        "rms_target": -20.0,
        "morphology": {
            "spectromorphological_archetype": "Iterative",
            "gesture_class": "Oscillatory / Tremor",
            "motion_type": "Reciprocal motion"
        }
    },

    # 8. BLUNDER - Attack–Decay - Impulsive collapse
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
        "rms_target": -18.0,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay",
            "gesture_class": "Impulsive / Collapsing",
            "motion_type": "Descent–Dissipation"
        }
    },

    # 9. BRILLIANT - Graduated Continuant - Expansive ascending gesture
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
        "rms_target": -16.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Expansive / Ascending",
            "motion_type": "Ascent–Agglomeration (Exogeny)"
        }
    },

    # 10. MISTAKE - Attack–Decay (damped) - Weak impulsive collapse
    "MISTAKE": {
        "duration_base": 2.0,
        "duration_tension_scale": 0.9,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.08,
            "bloom": 0.22,
            "decay": 0.38,
            "residue": 0.17
        },
        "pitch": {
            "type": "exponential_gliss",
            "start_freq_base": 660,
            "start_freq_entropy_scale": 220,
            "octave_drop_base": 1.5,
            "octave_drop_tension_scale": 0.6,
            "gliss_phases": ["bloom", "decay"]
        },
        "harmony": {
            "type": "cluster_to_interval",
            "num_voices_base": 2,
            "num_voices_tension_scale": 1,
            "cluster_semitone_spacing": 1.5,
            "resolve_interval_type": "muddy",
            "resolve_phase": "decay"
        },
        "filter": {
            "type": "bandpass_to_lowpass_choke",
            "bp_center_base": 850,
            "bp_center_tension_scale": 800,
            "bp_bandwidth": 700,
            "lp_cutoff_base": 180,
            "lp_cutoff_tension_scale": 120,
            "lp_resonance_base": 0.65,
            "lp_resonance_tension_scale": 0.20,
            "morph_phase": "bloom"
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 2,
            "attack_ms_entropy_scale": 3,
            "sustain_phase_ratio": 0.14,
            "decay_curve": "exponential",
            "decay_coefficient": -3.5
        },
        "texture": {
            "noise_ratio_base": 0.26,
            "noise_ratio_entropy_scale": 0.30,
            "noise_type": "pink"
        },
        "peak_limit": 0.76,
        "rms_target": -19.0,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay",
            "gesture_class": "Impulsive / Weak",
            "motion_type": "Descent–Dissipation"
        }
    },

    # 11. INACCURACY - Attack–Decay (weak) - Wind chime gesture
    "INACCURACY": {
        "duration_base": 1.5,
        "duration_tension_scale": 0.4,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.14,
            "impact": 0.12,
            "bloom": 0.26,
            "decay": 0.32,
            "residue": 0.16
        },
        "pitch": {
            "type": "weak_parabolic",
            "start_freq_base": 550,
            "rise_semitones": 3,
            "fall_semitones": 4
        },
        "harmony": {
            "type": "minimal_dyad",
            "num_voices": 2,
            "interval_semitones": 3
        },
        "filter": {
            "type": "gentle_bandpass",
            "bp_center_base": 950,
            "bp_bandwidth": 500,
            "resonance": 0.35
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 5,
            "attack_ms_entropy_scale": 2,
            "sustain_phase_ratio": 0.20,
            "decay_curve": "exponential",
            "decay_coefficient": -2.8
        },
        "texture": {
            "noise_ratio_base": 0.20,
            "noise_ratio_entropy_scale": 0.18,
            "noise_type": "pink"
        },
        "peak_limit": 0.65,
        "rms_target": -21.0,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay",
            "gesture_class": "Gentle / Hesitant",
            "motion_type": "Parabolic (small rise–fall)"
        }
    },

    # 12. INTERESTING - Graduated Continuant (asymmetric) - Wandering spiral
    "INTERESTING": {
        "duration_base": 2.6,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.35,
        "phases": {
            "pre_shadow": 0.16,
            "impact": 0.14,
            "bloom": 0.36,
            "decay": 0.24,
            "residue": 0.10
        },
        "pitch": {
            "type": "wandering_spiral",
            "center_freq_base": 440,
            "wander_semitones": 8,
            "spiral_rate_hz": 3.5
        },
        "harmony": {
            "type": "shifting_cluster",
            "num_voices": 3,
            "detune_randomness": 0.6
        },
        "filter": {
            "type": "erratic_sweep",
            "bp_center_range_low": 600,
            "bp_center_range_high": 2100,
            "sweep_irregularity": 0.7
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 55,
            "sustain_phase_ratio": 0.42,
            "decay_curve": "sigmoid"
        },
        "texture": {
            "noise_ratio_base": 0.24,
            "noise_ratio_entropy_scale": 0.32,
            "noise_type": "white"
        },
        "peak_limit": 0.72,
        "rms_target": -18.5,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Asymmetric / Curious",
            "motion_type": "Spiral or Oscillation"
        }
    },

    # 13. STRONG - Graduated Continuant (firm) - Stable controlled flow
    "STRONG": {
        "duration_base": 2.9,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": -0.25,
        "phases": {
            "pre_shadow": 0.18,
            "impact": 0.13,
            "bloom": 0.42,
            "decay": 0.17,
            "residue": 0.10
        },
        "pitch": {
            "type": "stable_rise",
            "start_freq_base": 330,
            "octave_rise": 0.9
        },
        "harmony": {
            "type": "controlled_expansion",
            "num_voices": 4,
            "interval_type": "perfect_fifths"
        },
        "filter": {
            "type": "controlled_opening",
            "lp_start": 450,
            "lp_end": 2300,
            "resonance": 0.42
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 70,
            "sustain_phase_ratio": 0.54,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.12,
            "noise_ratio_entropy_scale": 0.14,
            "noise_type": "pink"
        },
        "peak_limit": 0.83,
        "rms_target": -17.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Stable / Controlled",
            "motion_type": "Steady Flow / Endogeny"
        }
    },

    # 14. KING_ATTACK - Graduated Continuant (high-energy) - Forceful agglomeration
    "KING_ATTACK": {
        "duration_base": 3.3,
        "duration_tension_scale": 1.1,
        "duration_entropy_scale": 0.2,
        "phases": {
            "pre_shadow": 0.11,
            "impact": 0.16,
            "bloom": 0.46,
            "decay": 0.17,
            "residue": 0.10
        },
        "pitch": {
            "type": "aggressive_rise",
            "start_freq_base": 440,
            "octave_rise": 1.9
        },
        "harmony": {
            "type": "dense_agglomeration",
            "num_voices": 7,
            "cluster_density": 0.85
        },
        "filter": {
            "type": "aggressive_open",
            "lp_start": 550,
            "hp_end": 3700,
            "resonance_peak": 0.72
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 35,
            "sustain_phase_ratio": 0.52,
            "decay_curve": "exponential"
        },
        "texture": {
            "noise_ratio_base": 0.30,
            "noise_ratio_entropy_scale": 0.24,
            "noise_type": "white"
        },
        "peak_limit": 0.88,
        "rms_target": -15.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Forceful / Driven",
            "motion_type": "Ascent–Agglomeration"
        }
    },

    # 15. TACTICAL_SEQUENCE - Iterated Composite - Cellular rhythmic convolution
    "TACTICAL_SEQUENCE": {
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
        "rms_target": -20.0,
        "morphology": {
            "spectromorphological_archetype": "Iterated Composite",
            "gesture_class": "Segmented / Cellular",
            "motion_type": "Periodic iteration / rhythmic convolution"
        }
    },

    # 16. CHECKMATE - Attack–Decay Chain - Terminal resolution
    "CHECKMATE": {
        "duration_base": 3.0,
        "duration_tension_scale": 0.7,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.14,
            "impact": 0.18,
            "bloom": 0.28,
            "decay": 0.30,
            "residue": 0.10
        },
        "pitch": {
            "type": "converging_final",
            "start_freq_base": 660,
            "end_freq_base": 880,
            "num_steps": 4
        },
        "harmony": {
            "type": "resolving_cluster",
            "num_voices_start": 5,
            "num_voices_end": 1,
            "resolution_phase": "decay"
        },
        "filter": {
            "type": "terminal_focus",
            "bp_start_width": 1300,
            "bp_end_width": 200,
            "final_resonance": 0.7
        },
        "envelope": {
            "type": "terminal_strike",
            "attack_ms": 8,
            "final_accent": True
        },
        "texture": {
            "noise_ratio_base": 0.14,
            "noise_ratio_entropy_scale": 0.16,
            "noise_type": "white"
        },
        "peak_limit": 0.90,
        "rms_target": -14.5,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay Chain",
            "gesture_class": "Terminal / Resolving",
            "motion_type": "Convergence → Termination"
        }
    },

    # 17. CASTLING - Composite Dual - Balanced mirrored gesture
    "CASTLING": {
        "duration_base": 1.7,
        "duration_tension_scale": 0.25,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.20,
            "bloom": 0.30,
            "decay": 0.25,
            "residue": 0.10
        },
        "pitch": {
            "type": "reciprocal_pair",
            "freq_1": 330,
            "freq_2": 660,
            "crossover_phase": "bloom"
        },
        "harmony": {
            "type": "dual_voices",
            "num_voices": 2,
            "mirror_motion": True
        },
        "filter": {
            "type": "symmetric_sweep",
            "bp_center": 1000,
            "bp_bandwidth": 500
        },
        "envelope": {
            "type": "symmetric_dual",
            "attack_ms": 12,
            "sustain_phase_ratio": 0.32
        },
        "texture": {
            "noise_ratio_base": 0.11,
            "noise_ratio_entropy_scale": 0.10,
            "noise_type": "pink"
        },
        "peak_limit": 0.74,
        "rms_target": -19.0,
        "morphology": {
            "spectromorphological_archetype": "Composite Dual",
            "gesture_class": "Balanced / Mirrored",
            "motion_type": "Reciprocal motion"
        }
    },

    # 18. PROMOTION - Graduated Continuant - Ascending transformation
    "PROMOTION": {
        "duration_base": 2.6,
        "duration_tension_scale": 0.6,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.14,
            "bloom": 0.36,
            "decay": 0.25,
            "residue": 0.10
        },
        "pitch": {
            "type": "ascending_burst",
            "start_freq_base": 220,
            "octave_rise": 2.3,
            "burst_phase": "decay"
        },
        "harmony": {
            "type": "transforming_chord",
            "start_voices": 2,
            "end_voices": 5,
            "transformation_phase": "bloom"
        },
        "filter": {
            "type": "brightening_burst",
            "lp_start": 420,
            "hp_end": 4200,
            "burst_phase": "decay"
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 65,
            "sustain_phase_ratio": 0.40,
            "burst_accent_mult": 1.4
        },
        "texture": {
            "noise_ratio_base": 0.16,
            "noise_ratio_entropy_scale": 0.22,
            "noise_type": "white",
            "shimmer_enable": True,
            "shimmer_rate_hz": 8.0
        },
        "peak_limit": 0.87,
        "rms_target": -15.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Ascending / Transformative",
            "motion_type": "Ascent → Burst"
        }
    },

    # 19. PAWN_ADVANCE - Attack–Decay (elongated) - Grounded progressive push
    "PAWN_ADVANCE": {
        "duration_base": 1.7,
        "duration_tension_scale": 0.35,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.13,
            "impact": 0.16,
            "bloom": 0.30,
            "decay": 0.28,
            "residue": 0.13
        },
        "pitch": {
            "type": "low_linear_rise",
            "start_freq_base": 165,
            "octave_rise": 0.6
        },
        "harmony": {
            "type": "grounded_dyad",
            "num_voices": 2,
            "bass_emphasis": True
        },
        "filter": {
            "type": "bass_emphasis",
            "lp_cutoff": 900,
            "boost_low_freq": True
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 20,
            "attack_ms_entropy_scale": 5,
            "sustain_phase_ratio": 0.30,
            "decay_curve": "linear",
            "decay_coefficient": -2.5
        },
        "texture": {
            "noise_ratio_base": 0.19,
            "noise_ratio_entropy_scale": 0.13,
            "noise_type": "pink"
        },
        "peak_limit": 0.71,
        "rms_target": -19.5,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay (elongated)",
            "gesture_class": "Grounded / Progressive",
            "motion_type": "Gradual Ascent"
        }
    },

    # 20. CENTER_CONTROL - Continuant - Stable balanced centric motion
    "CENTER_CONTROL": {
        "duration_base": 2.5,
        "duration_tension_scale": 0.45,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.18,
            "impact": 0.12,
            "bloom": 0.42,
            "decay": 0.18,
            "residue": 0.10
        },
        "pitch": {
            "type": "stable_center",
            "center_freq_base": 440,
            "stability_factor": 0.96
        },
        "harmony": {
            "type": "balanced_spectrum",
            "num_voices": 5,
            "interval_balance": "equal_spacing"
        },
        "filter": {
            "type": "broadband_stable",
            "lp_cutoff": 2600,
            "hp_cutoff": 220,
            "resonance": 0.36
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 55,
            "sustain_phase_ratio": 0.52,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.17,
            "noise_ratio_entropy_scale": 0.14,
            "noise_type": "pink"
        },
        "peak_limit": 0.77,
        "rms_target": -18.0,
        "morphology": {
            "spectromorphological_archetype": "Continuant",
            "gesture_class": "Stable / Balanced",
            "motion_type": "Centric motion"
        }
    },

    # 21. PIECE_MANEUVER - Graduated Continuant - Curved graceful arc
    "PIECE_MANEUVER": {
        "duration_base": 2.1,
        "duration_tension_scale": 0.4,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.16,
            "bloom": 0.35,
            "decay": 0.24,
            "residue": 0.10
        },
        "pitch": {
            "type": "parabolic_arc",
            "start_freq": 440,
            "peak_freq": 660,
            "end_freq": 440,
            "curve_symmetry": 0.88
        },
        "harmony": {
            "type": "flowing_voices",
            "num_voices": 3,
            "voice_motion": "arced"
        },
        "filter": {
            "type": "curved_sweep",
            "bp_center_start": 850,
            "bp_center_peak": 1700,
            "bp_center_end": 850
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 45,
            "sustain_phase_ratio": 0.38,
            "decay_curve": "sigmoid"
        },
        "texture": {
            "noise_ratio_base": 0.15,
            "noise_ratio_entropy_scale": 0.17,
            "noise_type": "pink"
        },
        "peak_limit": 0.73,
        "rms_target": -19.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Curved / Graceful",
            "motion_type": "Parabolic / Spiral"
        }
    },

    # 22. DEVELOPMENT - Continuant Growth - Emergent formation
    "DEVELOPMENT": {
        "duration_base": 2.7,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.20,
            "impact": 0.15,
            "bloom": 0.36,
            "decay": 0.19,
            "residue": 0.10
        },
        "pitch": {
            "type": "gradual_emergence",
            "start_freq_base": 330,
            "octave_rise": 0.7
        },
        "harmony": {
            "type": "expanding_formation",
            "num_voices_start": 1,
            "num_voices_end": 4,
            "expansion_curve": "exponential"
        },
        "filter": {
            "type": "gradual_opening",
            "lp_start": 520,
            "lp_end": 2100,
            "opening_curve": "sigmoid"
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 85,
            "sustain_phase_ratio": 0.44,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.21,
            "noise_ratio_entropy_scale": 0.23,
            "noise_type": "pink"
        },
        "peak_limit": 0.76,
        "rms_target": -18.5,
        "morphology": {
            "spectromorphological_archetype": "Continuant Growth",
            "gesture_class": "Emergent / Forming",
            "motion_type": "Exogeny / Growth"
        }
    },

    # 23. ROOK_ACTIVATION - Attack–Continuant - Forceful directed expansion
    "ROOK_ACTIVATION": {
        "duration_base": 2.3,
        "duration_tension_scale": 0.65,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.10,
            "impact": 0.19,
            "bloom": 0.39,
            "decay": 0.22,
            "residue": 0.10
        },
        "pitch": {
            "type": "forceful_rise",
            "start_freq_base": 330,
            "octave_rise": 1.3
        },
        "harmony": {
            "type": "forceful_stack",
            "num_voices": 4,
            "power_chord": True
        },
        "filter": {
            "type": "forceful_open",
            "lp_start": 620,
            "hp_end": 2900,
            "aggressive": True
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 10,
            "attack_ms_entropy_scale": 4,
            "sustain_phase_ratio": 0.44,
            "decay_curve": "exponential",
            "decay_coefficient": -3.2
        },
        "texture": {
            "noise_ratio_base": 0.23,
            "noise_ratio_entropy_scale": 0.19,
            "noise_type": "white"
        },
        "peak_limit": 0.84,
        "rms_target": -16.5,
        "morphology": {
            "spectromorphological_archetype": "Attack–Continuant",
            "gesture_class": "Forceful / Directed",
            "motion_type": "Linear Ascent / Expansion"
        }
    },

    # 24. ROOKS_DOUBLED - Composite Dual - Coupled reinforcing resonance
    "ROOKS_DOUBLED": {
        "duration_base": 2.5,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.12,
            "impact": 0.11,
            "bloom": 0.39,
            "decay": 0.26,
            "residue": 0.12
        },
        "pitch": {
            "type": "dual_iteration",
            "freq_1": 330,
            "freq_2": 440,
            "iteration_rate_hz": 10
        },
        "harmony": {
            "type": "reinforced_pair",
            "num_voices": 4,
            "coupling_strength": 0.82
        },
        "filter": {
            "type": "dual_resonance",
            "bp_center_1": 850,
            "bp_center_2": 1250,
            "q_factor": 0.62
        },
        "envelope": {
            "type": "gated_pulse",
            "attack_ms": 8,
            "gate_duration_ms": 70,
            "release_ms": 22,
            "pulse_rate_hz": 10
        },
        "texture": {
            "noise_ratio_base": 0.09,
            "noise_ratio_entropy_scale": 0.11,
            "noise_type": "pink"
        },
        "peak_limit": 0.81,
        "rms_target": -17.5,
        "morphology": {
            "spectromorphological_archetype": "Composite Dual",
            "gesture_class": "Coupled / Reinforcing",
            "motion_type": "Centric rotation"
        }
    },

    # 25. QUEEN_CENTRALIZED - Continuant - Focused stable centric fixation
    "QUEEN_CENTRALIZED": {
        "duration_base": 2.8,
        "duration_tension_scale": 0.6,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.17,
            "impact": 0.13,
            "bloom": 0.44,
            "decay": 0.16,
            "residue": 0.10
        },
        "pitch": {
            "type": "focused_center",
            "center_freq_base": 550,
            "focus_stability": 0.94
        },
        "harmony": {
            "type": "centered_cluster",
            "num_voices": 5,
            "spectral_focus": "midrange"
        },
        "filter": {
            "type": "focused_bandpass",
            "bp_center": 1100,
            "bp_bandwidth": 400,
            "resonance": 0.58
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 60,
            "sustain_phase_ratio": 0.56,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.13,
            "noise_ratio_entropy_scale": 0.15,
            "noise_type": "pink"
        },
        "peak_limit": 0.79,
        "rms_target": -17.5,
        "morphology": {
            "spectromorphological_archetype": "Continuant",
            "gesture_class": "Focused / Stable",
            "motion_type": "Centric fixation"
        }
    },

    # 26. FIRST_EXCHANGE - Attack–Decay - Collision impact
    "FIRST_EXCHANGE": {
        "duration_base": 1.4,
        "duration_tension_scale": 0.35,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.10,
            "impact": 0.22,
            "bloom": 0.18,
            "decay": 0.32,
            "residue": 0.18
        },
        "pitch": {
            "type": "impact_transient",
            "strike_freq_base": 660,
            "decay_mult": 0.7
        },
        "harmony": {
            "type": "collision_cluster",
            "num_voices": 3,
            "impact_density": 0.9
        },
        "filter": {
            "type": "impact_spike",
            "bp_center": 1500,
            "bp_bandwidth": 900,
            "impact_resonance": 0.75
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 1,
            "attack_ms_entropy_scale": 2,
            "sustain_phase_ratio": 0.15,
            "decay_curve": "exponential",
            "decay_coefficient": -4.5
        },
        "texture": {
            "noise_ratio_base": 0.28,
            "noise_ratio_entropy_scale": 0.25,
            "noise_type": "white"
        },
        "peak_limit": 0.78,
        "rms_target": -18.5,
        "morphology": {
            "spectromorphological_archetype": "Attack–Decay",
            "gesture_class": "Collision / Impact",
            "motion_type": "Transient strike"
        }
    },

    # 27. ASYMMETRY - Continuant - Distorted divergent fragmentation
    "ASYMMETRY": {
        "duration_base": 2.4,
        "duration_tension_scale": 0.6,
        "duration_entropy_scale": 0.42,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.15,
            "bloom": 0.36,
            "decay": 0.24,
            "residue": 0.10
        },
        "pitch": {
            "type": "divergent_split",
            "center_freq": 440,
            "divergence_semitones": 13,
            "split_phase": "bloom"
        },
        "harmony": {
            "type": "fragmented_voices",
            "num_voices": 4,
            "divergence_factor": 0.72
        },
        "filter": {
            "type": "split_trajectories",
            "bp_1_path_start": 800,
            "bp_1_path_end": 420,
            "bp_2_path_start": 800,
            "bp_2_path_end": 2200
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 50,
            "sustain_phase_ratio": 0.40,
            "asymmetry_factor": 0.65,
            "decay_curve": "sigmoid"
        },
        "texture": {
            "noise_ratio_base": 0.27,
            "noise_ratio_entropy_scale": 0.28,
            "noise_type": "white"
        },
        "peak_limit": 0.78,
        "rms_target": -18.0,
        "morphology": {
            "spectromorphological_archetype": "Continuant",
            "gesture_class": "Distorted / Divergent",
            "motion_type": "Fragmentation / Divergence"
        }
    },

    # 28. TIME_MILESTONE - Continuant - Transitional checkpoint marker
    "TIME_MILESTONE": {
        "duration_base": 1.9,
        "duration_tension_scale": 0.3,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.16,
            "bloom": 0.32,
            "decay": 0.26,
            "residue": 0.11
        },
        "pitch": {
            "type": "gentle_descent",
            "start_freq": 550,
            "end_freq": 440
        },
        "harmony": {
            "type": "resolving_dyad",
            "num_voices": 2,
            "resolution_interval": 5
        },
        "filter": {
            "type": "gentle_close",
            "lp_start": 1600,
            "lp_end": 850
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 32,
            "sustain_phase_ratio": 0.34,
            "decay_curve": "sigmoid"
        },
        "texture": {
            "noise_ratio_base": 0.13,
            "noise_ratio_entropy_scale": 0.10,
            "noise_type": "pink"
        },
        "peak_limit": 0.70,
        "rms_target": -20.0,
        "morphology": {
            "spectromorphological_archetype": "Continuant",
            "gesture_class": "Marker / Transitional",
            "motion_type": "Periodic checkpoint"
        }
    },

    # 29. TIME_SCRAMBLE - Iterative - Turbulent chaotic rapid oscillation
    "TIME_SCRAMBLE": {
        "duration_base": 1.7,
        "duration_tension_scale": 0.45,
        "duration_entropy_scale": 0.55,
        "phases": {
            "pre_shadow": 0.08,
            "impact": 0.13,
            "bloom": 0.36,
            "decay": 0.28,
            "residue": 0.15
        },
        "pitch": {
            "type": "chaotic_iteration",
            "center_freq_base": 550,
            "chaos_amount": 0.85,
            "iteration_rate_hz": 17
        },
        "harmony": {
            "type": "granular_cluster",
            "num_voices": 7,
            "micro_detune": True,
            "chaos_factor": 0.8
        },
        "filter": {
            "type": "turbulent_sweep",
            "bp_range_low": 650,
            "bp_range_high": 2700,
            "chaos_modulation": 0.75
        },
        "envelope": {
            "type": "gated_pulse",
            "attack_ms": 3,
            "gate_duration_ms": 45,
            "release_ms": 15,
            "pulse_rate_hz": 17,
            "amplitude_chaos": 0.55
        },
        "texture": {
            "noise_ratio_base": 0.48,
            "noise_ratio_entropy_scale": 0.32,
            "noise_type": "white"
        },
        "peak_limit": 0.73,
        "rms_target": -19.0,
        "morphology": {
            "spectromorphological_archetype": "Iterative",
            "gesture_class": "Turbulent / Chaotic",
            "motion_type": "Rapid oscillation / turbulence"
        }
    },

    # 30. QUEENS_TRADED - Dual Attack–Decay - Symmetrical convergent dissipation
    "QUEENS_TRADED": {
        "duration_base": 2.9,
        "duration_tension_scale": 0.55,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.12,
            "impact": 0.11,
            "bloom": 0.26,
            "decay": 0.36,
            "residue": 0.15
        },
        "pitch": {
            "type": "dual_descent",
            "freq_1": 660,
            "freq_2": 440,
            "convergence_point": 330
        },
        "harmony": {
            "type": "thinning_voices",
            "num_voices_start": 6,
            "num_voices_end": 2,
            "thinning_curve": "exponential"
        },
        "filter": {
            "type": "narrowing_spectrum",
            "bp_start_width": 1600,
            "bp_end_width": 450
        },
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 8,
            "attack_ms_entropy_scale": 4,
            "sustain_phase_ratio": 0.22,
            "decay_curve": "exponential",
            "decay_coefficient": -3.0
        },
        "texture": {
            "noise_ratio_base": 0.15,
            "noise_ratio_entropy_scale": 0.14,
            "noise_type": "pink"
        },
        "peak_limit": 0.72,
        "rms_target": -19.5,
        "morphology": {
            "spectromorphological_archetype": "Dual Attack–Decay",
            "gesture_class": "Symmetrical / Resolving",
            "motion_type": "Convergence–Dissipation"
        }
    },

    # 31. FINAL_RESOLUTION - Graduated Continuant - Fading dissolving descent
    "FINAL_RESOLUTION": {
        "duration_base": 4.2,
        "duration_tension_scale": 0.9,
        "duration_entropy_scale": 0.0,
        "phases": {
            "pre_shadow": 0.15,
            "impact": 0.10,
            "bloom": 0.32,
            "decay": 0.33,
            "residue": 0.10
        },
        "pitch": {
            "type": "final_descent",
            "start_freq": 440,
            "end_freq": 110,
            "descent_curve": "logarithmic"
        },
        "harmony": {
            "type": "resolving_to_root",
            "num_voices_start": 5,
            "num_voices_end": 1,
            "resolution_chord": "tonic"
        },
        "filter": {
            "type": "closing_focus",
            "lp_start": 2100,
            "lp_end": 320,
            "final_resonance": 0.52
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 120,
            "sustain_phase_ratio": 0.28,
            "decay_curve": "exponential",
            "final_fade_curve": "sigmoid"
        },
        "texture": {
            "noise_ratio_base": 0.19,
            "noise_ratio_entropy_scale": 0.18,
            "noise_type": "pink"
        },
        "peak_limit": 0.75,
        "rms_target": -18.0,
        "morphology": {
            "spectromorphological_archetype": "Graduated Continuant",
            "gesture_class": "Fading / Dissolving",
            "motion_type": "Descent–Resolution"
        }
    },

}
