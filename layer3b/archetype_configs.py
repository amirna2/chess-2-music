"""
Archetype configuration definitions for Layer 3b gestures.

Each archetype defines its characteristic sound through configuration parameters
rather than custom code. All archetypes share the same synthesis pipeline.
"""

from typing import Dict, Any

ARCHETYPES: Dict[str, Dict[str, Any]] = {
    # 1. MOVE - Attack–Decay (neutral) - Simple transitional gesture
    "MOVE": {
        "system": "curve",
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
        "rms_target": -20.0
    },

    # 2. GAME_CHANGING - Graduated Continuant - Transformative expansion
    "GAME_CHANGING": {
        "system": "curve",
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
        "rms_target": -14.0
    },

    # 3. CRITICAL_SWING - Attack–Decay (accented) - Dynamic parabolic gesture
    "CRITICAL_SWING": {
        "system": "curve",
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
        "rms_target": -15.5
    },

    # 4. SIGNIFICANT_SHIFT - Particle System - Sparse drifting particles
    "SIGNIFICANT_SHIFT": {
        "system": "particle",
        "duration_base": 3.6,
        "duration_tension_scale": 0.3,
        "duration_entropy_scale": 0.0,
        "particle": {
            "emission": {
                "type": "drift_scatter",
                "start_density": 0.08,
                "end_density": 0.15,
                "drift_rate": 0.3
            },
            "base_spawn_rate": 0.0008,
            "pitch_range_hz": [330, 880],
            "lifetime_range_s": [1.5, 3.0],
            "velocity_range": [0.2, 0.45],
            "detune_range_cents": [-25, 25],
            "decay_rate_range": [-3.0, -1.5],
            "waveform": "sine"
        },
        "peak_limit": 0.5,
        "rms_target": -22.0
    },

    # 5. MATE_SEQUENCE - Composite Chain - Convergent resolution
    "MATE_SEQUENCE": {
        "system": "curve",
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
        "rms_target": -15.5
    },

    # 6. DEEP_THINK - Sustained Continuant - Static plateau
    "DEEP_THINK": {
        "system": "curve",
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
        "rms_target": -20.5
    },

    # 7. TIME_PRESSURE - Iterative - Oscillatory tremor
    "TIME_PRESSURE": {
        "system": "curve",
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

    # 8. BLUNDER - Attack–Decay - Impulsive collapse
    "BLUNDER": {
        "system": "curve",
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

    # 9. BRILLIANT - Graduated Continuant - Expansive ascending gesture
    "BRILLIANT": {
        "system": "curve",
        "duration_base": 3.3,
        "duration_tension_scale": 0.8,
        "duration_entropy_scale": -0.5,
        "phases": {
            "pre_shadow": 0.16,
            "impact": 0.10,
            "bloom": 0.48,
            "decay": 0.16,
            "residue": 0.10
        },
        "pitch": {
            "type": "ascending_spread",
            "start_freq_base": 220,
            "start_freq_entropy_scale": 110,
            "octave_rise": 2.2,
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
            "hp_cutoff_end": 3800,
            "resonance_base": 0.48,
            "morph_phase": "bloom"
        },
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 40,
            "sustain_phase_ratio": 0.54,
            "decay_curve": "linear"
        },
        "texture": {
            "noise_ratio_base": 0.06,
            "noise_ratio_entropy_scale": 0.16,
            "noise_type": "white",
            "shimmer_enable": True,
            "shimmer_rate_hz": 7.5
        },
        "peak_limit": 0.86,
        "rms_target": -15.5
    },

    # 10. MISTAKE - Attack–Decay (damped) - Weak impulsive collapse
    "MISTAKE": {
        "system": "curve",
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
        "rms_target": -19.0
    },

    # 11. INACCURACY - Particle System - Sparse stochastic wind chime strikes
    "INACCURACY": {
        "system": "particle",
        "duration_base": 4.5,
        "duration_tension_scale": 0.0,
        "duration_entropy_scale": 0.0,
        "particle": {
            "emission": {
                "type": "gusts",
                "num_gusts": 2,
                "base_density": 0.015,
                "peak_density": 0.06
            },
            "base_spawn_rate": 0.001,
            "pitch_range_hz": [880, 1760],
            "lifetime_range_s": [1.2, 2.5],
            "velocity_range": [0.35, 0.55],
            "detune_range_cents": [-30, 30],
            "decay_rate_range": [-2.5, -1.5],
            "waveform": "triangle"
        },
        "peak_limit": 0.6,
        "rms_target": -20.0
    },
    # 13. STRONG - Graduated Continuant (firm) - Stable controlled flow
    "STRONG": {
        "system": "curve",
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
        "rms_target": -17.0
    },

    # 14. KING_ATTACK - Graduated Continuant (high-energy) - Forceful agglomeration
    "KING_ATTACK": {
        "system": "curve",
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
        "rms_target": -15.0
    },

    # 15. TACTICAL_SEQUENCE - Particle System - Calculation clicks in rhythmic clusters
    "TACTICAL_SEQUENCE": {
        "system": "particle",
        "duration_base": 2.2,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.3,
        "particle": {
            "emission": {
                "type": "rhythmic_clusters",
                "num_clusters": 4,
                "cluster_duration_ratio": 0.15,
                "cluster_density": 0.7,
                "gap_density": 0.05
            },
            "base_spawn_rate": 0.003,
            "pitch_range_hz": [660, 1320],
            "lifetime_range_s": [0.3, 0.7],
            "velocity_range": [0.5, 0.7],
            "detune_range_cents": [-15, 15],
            "decay_rate_range": [-4.0, -2.5],
            "waveform": "sine"
        },
        "peak_limit": 0.6,
        "rms_target": -20.0
    },

    # 16. CHECKMATE - Attack–Decay Chain - Terminal resolution
    "CHECKMATE": {
        "system": "curve",
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
        "rms_target": -14.5
    },

    # 17. CASTLING - Composite Dual - Balanced mirrored gesture
    "CASTLING": {
        "system": "curve",
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
        "rms_target": -19.0
    },

    # 18. PROMOTION - Graduated Continuant - Ascending transformation
    "PROMOTION": {
        "system": "curve",
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
        "rms_target": -15.0
    },

    # 19. PAWN_ADVANCE - Attack–Decay (elongated) - Grounded progressive push
    "PAWN_ADVANCE": {
        "system": "curve",
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
        "rms_target": -19.5
    },

    # 20. CENTER_CONTROL - Continuant - Stable balanced centric motion
    "CENTER_CONTROL": {
        "system": "curve",
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
        "rms_target": -18.0
    },

    # 21. PIECE_MANEUVER - Graduated Continuant - Curved graceful arc
    "PIECE_MANEUVER": {
        "system": "curve",
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
        "rms_target": -19.0
    },

    # 22. DEVELOPMENT - Continuant Growth - Emergent formation
    "DEVELOPMENT": {
        "system": "curve",
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
        "rms_target": -18.5
    },

    # 23. ROOK_ACTIVATION - Attack–Continuant - Forceful directed expansion
    "ROOK_ACTIVATION": {
        "system": "curve",
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
        "rms_target": -16.5
    },

    # 24. ROOKS_DOUBLED - Composite Dual - Coupled reinforcing resonance
    "ROOKS_DOUBLED": {
        "system": "curve",
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
        "rms_target": -17.5
    },

    # 25. QUEEN_CENTRALIZED - Continuant - Focused stable centric fixation
    "QUEEN_CENTRALIZED": {
        "system": "curve",
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
        "rms_target": -17.5
    },

    # 26. FIRST_EXCHANGE - Particle System - Metallic collision scatter
    "FIRST_EXCHANGE": {
        "system": "particle",
        "duration_base": 1.4,
        "duration_tension_scale": 0.35,
        "duration_entropy_scale": 0.0,
        "particle": {
            "emission": {
                "type": "impact_burst",
                "impact_time_ratio": 0.15,
                "burst_density": 0.95,
                "burst_duration_ratio": 0.08,
                "tail_density": 0.05
            },
            "base_spawn_rate": 0.005,
            "pitch_range_hz": [800, 2200],
            "lifetime_range_s": [0.4, 1.2],
            "velocity_range": [0.5, 0.7],
            "detune_range_cents": [-60, 60],
            "decay_rate_range": [-5.0, -2.5],
            "waveform": "triangle"
        },
        "peak_limit": 0.65,
        "rms_target": -18.0
    },

    # 27. ASYMMETRY - Continuant - Distorted divergent fragmentation
    "ASYMMETRY": {
        "system": "curve",
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
        "rms_target": -18.0
    },

    # 28. TIME_MILESTONE - Continuant - Transitional checkpoint marker
    "TIME_MILESTONE": {
        "system": "curve",
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
        "rms_target": -20.0
    },

    # 29. TIME_SCRAMBLE - Iterative - Turbulent chaotic rapid oscillation
    "TIME_SCRAMBLE": {
        "system": "curve",
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
        "rms_target": -19.0
    },

    # 30. QUEENS_TRADED - Dual Attack–Decay - Symmetrical convergent dissipation
    "QUEENS_TRADED": {
        "system": "curve",
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
        "rms_target": -19.5
    },

    # 31. FINAL_RESOLUTION - Particle System - Dissolving particles fade to silence
    "FINAL_RESOLUTION": {
        "system": "particle",
        "duration_base": 4.7,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.0,
        "particle": {
            "emission": {
                "type": "dissolve",
                "start_density": 0.6,
                "decay_rate": -1.8
            },
            "base_spawn_rate": 0.002,
            "pitch_range_hz": [220, 660],
            "lifetime_range_s": [2.0, 4.0],
            "velocity_range": [0.15, 0.4],
            "detune_range_cents": [-20, 20],
            "decay_rate_range": [-1.5, -0.8],
            "waveform": "sine"
        },
        "peak_limit": 0.35,
        "rms_target": -30.0
    },

}
