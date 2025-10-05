"""
SYNTH_CONFIG - Centralized Configuration for Chess Music Synthesis
All musical parameters, presets, and mappings in one place

IMPORTANT: Tagger Moment Types vs. Internal Voice Mappings
-----------------------------------------------------------
The tagger (tagger.py) produces simple moment types like:
    - BLUNDER, BRILLIANT, MISTAKE, INACCURACY, STRONG
    - MATE_SEQUENCE, TACTICAL_SEQUENCE, KING_ATTACK
    - DEVELOPMENT, FIRST_EXCHANGE
    - CHECKMATE, CASTLING, PROMOTION, etc.

The composer (synth_composer.py) uses context-aware mapping:
    - Takes tagger type (e.g., "BLUNDER")
    - Combines with overall_narrative (e.g., "TUMBLING_DEFEAT")
    - Looks up voice (e.g., "BLUNDER_IN_DEFEAT")

This allows the SAME chess event to sound different in different game contexts!
Example: A blunder in a defeat sounds like doom, but in a masterpiece it's a brief disturbance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class SynthConfig:
    """
    Centralized configuration for all synthesis parameters.
    Modify values here to change the musical output without touching code logic.
    """

    # === MUSICAL SCALES ===
    SCALES: Dict[str, List[float]] = field(default_factory=lambda: {
        'minor': [110, 123.47, 130.81, 146.83, 164.81, 174.61, 196, 220],
        'phrygian': [110, 116.54, 130.81, 146.83, 164.81, 174.61, 196, 220],
        'dorian': [110, 123.47, 130.81, 146.83, 164.81, 185, 196, 220],
    })

    # === ENVELOPE PRESETS (attack, decay, sustain, release) ===
    ENVELOPES: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        'percussive': (0.001, 0.02, 0.3, 0.05),
        'pluck': (0.001, 0.05, 0.4, 0.1),
        'pad': (0.5, 0.0, 1.0, 0.5),
        'stab': (0.01, 0.05, 0.7, 0.1),
        'drone': (0.5, 0.0, 1.0, 0.5),
        'laser': (0.02, 0.1, 0.9, 0.8),
        'doom': (0.5, 0.0, 1.0, 1.0),
        'short_pluck': (0.001, 0.01, 0.5, 0.2),
        'medium_stab': (0.001, 0.01, 0.5, 0.4),
        'gentle': (0.02, 0.05, 0.7, 0.1),
        'soft': (0.02, 0.06, 0.6, 0.1),
        'brief': (0.001, 0.01, 0.3, 0.1),
        'quick': (0.001, 0.02, 0.5, 0.05),
        'standard': (0.001, 0.01, 0.8, 0.05),
        'sustained': (0.001, 0.1, 0.8, 0.5),
    })

    FILTER_ENVELOPES: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        'gentle': (0.01, 0.15, 0.3, 0.2),
        'sweep': (0.001, 0.2, 0.5, 0.1),
        'minimal': (0.5, 0.0, 1.0, 0.5),
        'dramatic': (0.01, 0.25, 0.4, 0.4),
        'closing': (0.001, 0.5, 0.0, 0.4),
        'standard': (0.01, 0.15, 0.3, 0.2),
        'smooth': (0.001, 0.1, 0.2, 0.1),
        'sharp': (0.001, 0.2, 0.7, 0.3),
    })

    # === OVERALL NARRATIVE BASE PARAMETERS ===
    # These define the fundamental character of the entire piece
    NARRATIVE_BASE_PARAMS: Dict[str, Dict] = field(default_factory=lambda: {
        'TUMBLING_DEFEAT': {
            'base_waveform': 'supersaw',
            'filter_start': 2500,
            'filter_end': 300,
            'resonance_start': 0.8,
            'resonance_end': 3.5,
            'tempo_start': 1.0,
            'tempo_end': 0.7,
            'detune_start': 3,
            'detune_end': 20,
            'scale': 'phrygian',
        },
        'FIGHTING_DEFEAT': {
            'base_waveform': 'supersaw',
            'filter_start': 2500,
            'filter_end': 300,
            'resonance_start': 0.8,
            'resonance_end': 3.5,
            'tempo_start': 1.0,
            'tempo_end': 0.7,
            'detune_start': 3,
            'detune_end': 20,
            'scale': 'phrygian',
        },
        'ATTACKING_MASTERPIECE': {
            'base_waveform': 'pulse',
            'filter_start': 500,
            'filter_end': 5000,
            'resonance_start': 0.5,
            'resonance_end': 1.8,
            'tempo_start': 0.8,
            'tempo_end': 1.2,
            'detune_start': 0,
            'detune_end': 7,
            'scale': 'dorian',
        },
        'TACTICAL_MASTERPIECE': {
            'base_waveform': 'pulse',
            'filter_start': 800,
            'filter_end': 3000,
            'resonance_start': 0.5,
            'resonance_end': 1.3,
            'tempo_start': 0.8,
            'tempo_end': 1.2,
            'detune_start': 0,
            'detune_end': 7,
            'scale': 'dorian',
            'drone_voices': 3
        },
        'PEACEFUL_DRAW': {
            'base_waveform': 'triangle',
            'filter_start': 1500,
            'filter_end': 1500,
            'resonance_start': 0.3,
            'resonance_end': 0.3,
            'tempo_start': 1.0,
            'tempo_end': 1.0,
            'detune_start': 0,
            'detune_end': 0,
            'scale': 'dorian',
        },
        'QUIET_PRECISION': {
            'base_waveform': 'triangle',
            'filter_start': 600,
            'filter_end': 2500,
            'resonance_start': 0.8,
            'resonance_end': 1.2,
            'tempo_start': 0.9,
            'tempo_end': 1.0,
            'detune_start': 2,
            'detune_end': 5,
            'scale': 'dorian',
        },
        'DEATH_SPIRAL': {
            'base_waveform': 'saw',
            'filter_start': 1800,
            'filter_end': 400,
            'resonance_start': 1.5,
            'resonance_end': 3.0,
            'tempo_start': 1.0,
            'tempo_end': 0.6,
            'detune_start': 5,
            'detune_end': 18,
            'scale': 'phrygian',
        },
        'DEFAULT': {
            'base_waveform': 'saw',
            'filter_start': 1500,
            'filter_end': 2000,
            'resonance_start': 1.0,
            'resonance_end': 1.5,
            'tempo_start': 1.0,
            'tempo_end': 1.0,
            'detune_start': 0,
            'detune_end': 5,
            'scale': 'minor',
        },
    })

    # === SECTION NARRATIVE MODULATION PARAMETERS ===
    # These modulate the base parameters for specific game sections
    SECTION_MODULATIONS: Dict[str, Dict] = field(default_factory=lambda: {
        'DESPERATE_DEFENSE': {
            'filter_mult': 1.1,
            'resonance_add': 1.5,
            'tempo_mult': 0.85,
            'note_density': 0.7,
            'filter_env_amount': 2000,
        },
        'KING_HUNT': {
            'filter_mult': 1.3,
            'resonance_add': 1.0,
            'tempo_mult': 1.2,
            'note_density': 1.5,
            'filter_env_amount': 2000,
        },
        'MATING_ATTACK': {
            'filter_mult': 1.3,
            'resonance_add': 1.0,
            'tempo_mult': 1.2,
            'note_density': 1.5,
            'filter_env_amount': 6000,
        },
        'TACTICAL_CHAOS': {
            'filter_mult': 0.8,
            'resonance_add': 2.0,
            'tempo_mult': 1.3,
            'note_density': 2.0,
            'filter_env_amount': 5000,
        },
        'TACTICAL_BATTLE': {
            'filter_mult': 0.8,
            'resonance_add': 2.0,
            'tempo_mult': 1.3,
            'note_density': 2.0,
            'filter_env_amount': 5000,
        },
        'QUIET': {
            'filter_mult': 1.1,
            'resonance_add': -0.3,
            'tempo_mult': 0.9,
            'note_density': 0.5,
            'filter_env_amount': 1500,
        },
        'POSITIONAL': {
            'filter_mult': 1.1,
            'resonance_add': -0.3,
            'tempo_mult': 0.9,
            'note_density': 0.5,
            'filter_env_amount': 1500,
        },
        'SACRIFICIAL_ATTACK': {
            'filter_mult': 0.7,
            'resonance_add': 1.5,
            'tempo_mult': 1.1,
            'note_density': 1.3,
            'filter_env_amount': 8000,
        },
        'CRUSHING_ATTACK': {
            'filter_mult': 0.7,
            'resonance_add': 1.5,
            'tempo_mult': 1.1,
            'note_density': 1.3,
            'filter_env_amount': 8000,
        },
        'ENDGAME_PRECISION': {
            'filter_mult': 1.0,
            'resonance_add': 0.3,
            'tempo_mult': 0.8,
            'note_density': 0.8,
            'filter_env_amount': 2000,
        },
        'COMPLEX_STRUGGLE': {
            'filter_mult': 0.9,
            'resonance_add': 0.5,
            'tempo_mult': 1.0,
            'note_density': 1.0,
            'filter_env_amount': 1200,
        },
        'FLAWLESS_CONVERSION': {
            'filter_mult': 1.0,
            'resonance_add': 0.3,
            'tempo_mult': 0.85,
            'note_density': 0.9,
            'filter_env_amount': 1800,
        },
        'DECISIVE_ENDING': {
            'filter_mult': 0.6,
            'resonance_add': 2.5,
            'tempo_mult': 0.7,
            'note_density': 0.5,
            'filter_env_amount': 3000,
        },
        'DEFAULT': {
            'filter_mult': 1.0,
            'resonance_add': 0.0,
            'tempo_mult': 1.0,
            'note_density': 1.0,
            'filter_env_amount': 2500,
        },
    })

    # === KEY MOMENT VOICE PARAMETERS ===
    # NOTE: These are INTERNAL mappings used by the composer
    # The tagger produces simple types (BLUNDER, BRILLIANT, etc.)
    # The composer maps them to context-aware voices based on overall_narrative
    # Example: BLUNDER + TUMBLING_DEFEAT context → BLUNDER_IN_DEFEAT voice
    MOMENT_VOICES: Dict[str, Dict] = field(default_factory=lambda: {
        'BLUNDER_IN_DEFEAT': {
            'freq': 55,
            'duration': 1.0,
            'waveform': 'saw',
            'filter_base': 200,
            'filter_env_amount': -150,
            'resonance': 4.0,
            'amp_env': 'doom',
            'filter_env': 'closing',
        },
        'BLUNDER_IN_MASTERPIECE': {
            'freq': 110,
            'duration': 0.3,
            'waveform': 'pulse',
            'filter_base': 3000,
            'filter_env_amount': -2500,
            'resonance': 2.0,
            'amp_env': 'brief',
        },
        'BLUNDER_NEUTRAL': {
            'freq': 82.5,
            'duration': 0.5,
            'waveform': 'saw',
            'filter_base': 1000,
            'filter_env_amount': -800,
            'resonance': 3.0,
            'amp_env': 'short_pluck',
        },
        'MISTAKE_IN_DEFEAT': {
            'freq': 55,
            'duration': 1.0,
            'waveform': 'saw',
            'filter_base': 200,
            'filter_env_amount': -150,
            'resonance': 4.0,
            'amp_env': 'doom',
            'filter_env': 'closing',
        },
        'MISTAKE_IN_MASTERPIECE': {
            'freq': 110,
            'duration': 0.3,
            'waveform': 'pulse',
            'filter_base': 3000,
            'filter_env_amount': -2500,
            'resonance': 2.0,
            'amp_env': 'brief',
        },
        'MISTAKE_NEUTRAL': {
            'freq': 82.5,
            'duration': 0.5,
            'waveform': 'saw',
            'filter_base': 1000,
            'filter_env_amount': -800,
            'resonance': 3.0,
            'amp_env': 'short_pluck',
        },
        'BRILLIANT_IN_MASTERPIECE': {
            'freq': 220,  # Modified by progress in code
            'duration': 0.5,
            'waveform': 'pulse',
            'filter_base': 500,
            'filter_env_amount': 4000,  # Modified by progress in code
            'resonance': 2.0,
            'amp_env': 'stab',
            'filter_env': 'sweep',
        },
        'BRILLIANT_IN_DEFEAT': {
            'freq': 220,
            'duration': 0.2,
            'waveform': 'triangle',
            'filter_base': 2000,
            'filter_env_amount': 500,
            'resonance': 0.5,
            'amp_env': 'brief',
        },
        'BRILLIANT_NEUTRAL': {
            'freq': 330,
            'duration': 0.3,
            'waveform': 'square',
            'filter_base': 1500,
            'filter_env_amount': 2000,
            'resonance': 1.5,
            'amp_env': 'standard',
        },
        'STRONG_IN_MASTERPIECE': {
            'freq': 220,
            'duration': 0.5,
            'waveform': 'pulse',
            'filter_base': 500,
            'filter_env_amount': 4000,
            'resonance': 2.0,
            'amp_env': 'stab',
            'filter_env': 'sweep',
        },
        'STRONG_IN_DEFEAT': {
            'freq': 220,
            'duration': 0.2,
            'waveform': 'triangle',
            'filter_base': 2000,
            'filter_env_amount': 500,
            'resonance': 0.5,
            'amp_env': 'brief',
        },
        'STRONG_NEUTRAL': {
            'freq': 330,
            'duration': 0.3,
            'waveform': 'square',
            'filter_base': 1500,
            'filter_env_amount': 2000,
            'resonance': 1.5,
            'amp_env': 'standard',
        },
        'TACTICAL_SEQUENCE': {
            'freqs': [220, 275, 330, 275],
            'total_duration': 1.2,
            'note_duration': 0.2,
            'waveform': 'square',
            'filter_base': 1500,
            'filter_env_amount': 1500,
            'resonance': 1.5,
            'amp_env': 'quick',
            'overlap_factor': 0.15,
            'volume': 0.7,
        },
        'MATE_IN_DEFEAT': {
            'freq': 27.5,
            'duration': 2.0,
            'waveform': 'saw',
            'filter_base': 100,
            'filter_env_amount': 0,
            'resonance': 4.0,
            'amp_env': 'doom',
        },
        'MATE_IN_MASTERPIECE': {
            'freq': 440,
            'duration': 1.0,
            'waveform': 'pulse',
            'filter_base': 300,
            'filter_env_amount': 5000,
            'resonance': 2.5,
            'amp_env': 'sustained',
        },
        'DEFAULT_MOMENT': {
            'freq': 330,
            'duration': 0.3,
            'waveform': 'triangle',
            'filter_base': 1000,
            'filter_env_amount': 500,
            'resonance': 0.5,
            'amp_env': 'gentle',
        },
        'FINAL_RESOLUTION': {
            # Uses Layer 3 sequencer settings - will inherit from last section
            # This is a placeholder that tells the system to use sequencer envelope
            'use_sequencer_envelope': True,
        },
    })

    # === DEVELOPMENT MOMENT PARAMETERS ===
    DEVELOPMENT_PARAMS: Dict[str, Dict] = field(default_factory=lambda: {
        'IN_DEFEAT': {
            'melody_indices': [0, 1, 2, 1],
            'note_duration': 0.3,
            'waveform': 'triangle',
            'filter_mult': 0.8,
            'base_filter_env': 800,
            'filter_env_step': 200,
            'resonance': 0.8,
            'amp_env': 'pluck',
            'volume': 0.6,
        },
        'DEFAULT': {
            'melody_indices': [0, 1, 2, 4],
            'note_duration': 0.25,
            'waveform': 'pulse',
            'filter_mult': 0.8,
            'base_filter_env': 800,
            'filter_env_step': 200,
            'resonance': 0.8,
            'amp_env': 'pluck',
            'volume': 0.6,
        },
    })

    # === FIRST EXCHANGE MOMENT PARAMETERS ===
    FIRST_EXCHANGE_PARAMS: Dict[str, Dict] = field(default_factory=lambda: {
        'IN_DEFEAT': {
            'question_indices': [0, 2, 4],
            'answer_indices': [3, 1, 0],
            'question_waveform': 'pulse',
            'answer_waveform': 'triangle',
            'note_duration': 0.65,
            'filter_mult': 0.9,
            'question_filter_env_base': 600,
            'answer_filter_env_base': 500,
            'question_resonance': 1.0,
            'answer_resonance': 0.8,
            'answer_brightness': 0.7,
            'question_volume': 0.7,
            'answer_volume': 0.6,
        },
        'DEFAULT': {
            'question_indices': [0, 2, 4],
            'answer_indices': [4, 2, 0],
            'question_waveform': 'square',
            'answer_waveform': 'pulse',
            'note_duration': 0.22,
            'filter_mult': 0.9,
            'question_filter_env_base': 600,
            'answer_filter_env_base': 500,
            'question_resonance': 1.0,
            'answer_resonance': 1.0,
            'answer_brightness': 0.9,
            'question_volume': 0.7,
            'answer_volume': 0.7,
        },
    })

    # === SEQUENCER PATTERNS ===
    SEQUENCER_PATTERNS: Dict = field(default_factory=lambda: {
        'DEVELOPMENT': {
            'early': [0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0],
            'mid': [0, 3, 7, 0, 3, 7, 12, 0, 3, 7, 12, 15, 0, 7, 12, 15],
            'full': [0, 3, 5, 7, 10, 12, 15, 17, 19, 17, 15, 12, 10, 7, 5, 3],
        },
        'ASYMMETRY': [0, None, 2, None, 0, None, -2, None, 0, 3, None, -3, 0, None, None, None],
        'CRITICAL_SWING': [0, 7, 12, 19, 12, 7, 0, -5, 0, 5, 12, 7, 0, -7, -12, None],
        'GAME_CHANGING': [0, 12, 0, 12, 24, 12, 0, 12, 24, 36, 24, 12, 0, None, None, None],
        'TACTICAL_SEQUENCE': [0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12],
        'KING_ATTACK': [0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12],
        'BLUNDER': [24, 24, 12, 12, 0, 0, -12, -12, -24, None, None, -36, None, None, None, -48],
        'MISTAKE': [12, 10, 8, 7, 5, 3, 2, 0, -2, -3, -5, -7, -8, -10, -12, -15],
        'INACCURACY': [0, None, 3, 2, None, 7, 5, None, 3, None, 0, None, -2, None, 0, None],
        'FIRST_EXCHANGE': [0, 12, 19, 24, -24, -19, -12, 0, 0, 12, 19, 24, -24, -19, -12, 0],
        'MATE_SEQUENCE': [0, None, None, None, -12, None, None, None, 0, None, None, None, -24, None, None, None],
        'BRILLIANT': [0, 7, 12, 19, 24, 31, 36, 43, 48, 43, 36, 31, 24, 19, 12, 7],
        'TIME_PRESSURE': [0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12],
        'TIME_SCRAMBLE': [0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12],
        'SIGNIFICANT_SHIFT': [0, 7, 14, 7, 0, -7, -14, -7, 0, 5, 10, 5, 0, -5, -10, -5],  # Major positional shift: wave motion
        'PULSE': [0, None, -2, None, None, None, None, None, 0, None, -2, None, None, None, None, None],  # Heartbeat: LUB-dub (first loud/low, second quiet/lower)
    })

    # === SUPERSAW DETUNE PRESETS ===
    SUPERSAW_DETUNE: Dict[str, List[float]] = field(default_factory=lambda: {
        'tight': [-3, -1.5, -0.75, 0.75, 1.5, 3],
        'standard': [-7, -3.5, -1.75, 1.75, 3.5, 7],
        'wide': [-12, -7, -3, 3, 7, 12],
        'laser': [-15, -9, -4.5, 4.5, 9, 15],
    })

    # === TIMING PARAMETERS (in seconds or milliseconds) ===
    TIMING: Dict[str, float] = field(default_factory=lambda: {
        'section_fade_sec': 0.1,
        'section_gap_sec': 0.0,
        'section_crossfade_sec': 2.0,
        'note_gap_sec': 0.02,
        'phrase_pause_sec': 0.15,
        'chunk_size_samples': 512,
        'filter_chunk_size_samples': 64,
        'sequencer_overlap': 0.98,
        'lfo_frequency': 0.1,
        'sequencer_lfo_frequency': 0.25,
    })

    # === LAYER MUTING (True = enabled, False = muted) ===
    LAYER_ENABLE: Dict[str, bool] = field(default_factory=lambda: {
        'drone': True,      # Layer 1: Base drone (overall narrative)
        'patterns': True,  # Layer 2: Rhythmic patterns (section narrative)
        'sequencer': True,  # Layer 3: Continuous sequencer (heartbeat)
        'moments': True,    # Key moment punctuation
    })

    # === MIXING LEVELS ===
    MIXING: Dict[str, float] = field(default_factory=lambda: {
        'drone_level': 0.3,
        'pattern_level': 0.4,
        'sequencer_level': 0.5,
        'moment_level': 0.8,
        'section_level': 0.7,
        'master_limiter': 0.9,
        'sidechain_amount': 0.3,
        'supersaw_compression': 0.8,
        'supersaw_gain': 1.25,
        'filtered_sequence_level': 0.5,
        'ducking_amount': 0.3,
        'soft_clip_pre': 0.9,
        'soft_clip_post': 0.95,
    })

    # === LAYER MIXING ===
    LAYER_MIXING: Dict[str, float] = field(default_factory=lambda: {
        'drone_in_supersaw': 0.3,
        'pattern_in_supersaw': 0.7,
        'pattern_note_level': 0.2,
        'sequencer_note_level': 0.4,
        'lfo_modulation_depth': 0.1,
    })

    # === MELODIC PATTERN PARAMETERS ===
    MELODIC_PATTERNS: Dict[str, Dict] = field(default_factory=lambda: {
        'DEFEAT_HIGH_TENSION': {
            'indices': [7, 5, 6, 4, 5, 3, 4, 2],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
        'DEFEAT_LOW_TENSION': {
            'indices': [4, 3, 3, 2, 2, 1, 1, 0],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
        'MASTERPIECE_HIGH_TENSION': {
            'indices': [0, 2, 1, 3, 2, 5, 4, 7],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
        'MASTERPIECE_LOW_TENSION': {
            'indices': [0, 1, 2, 3, 3, 4, 5, 6],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
        'NEUTRAL_HIGH_TENSION': {
            'indices': [0, 4, 2, 5, 1, 4, 3, 7],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
        'NEUTRAL_MEDIUM_TENSION': {
            'indices': [0, 2, 3, 2, 4, 3, 2, 1],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
        'NEUTRAL_LOW_TENSION': {
            'indices': [0, 1, 2, 1, 3, 2, 1, 0],
            'octave_up_mod': 4,
            'octave_down_mod': 7,
        },
    })

    # === SEQUENCER SYNTHESIS PARAMETERS ===
    SEQUENCER_SYNTH: Dict = field(default_factory=lambda: {
        'detune_cents': [-15, -9, -4.5, 4.5, 9, 15],
        'filter_base_start': 800,
        'filter_increment_per_step': 150,
        'filter_env_amount': 2000,
        'resonance': 1.2,
        'amp_env': (0.02, 0.1, 0.9, 0.8),      # MOMENTS envelope (restored from 8343de4)
        'filter_env': (0.02, 0.2, 0.7, 0.3),   # MOMENTS filter envelope (restored from 8343de4)
        'global_filter_base': 2000,
        'global_filter_lfo_amount': 1500,
        'global_filter_sweep_amount': 1000,
        'global_filter_resonance': 2.0,
        'smoothing_window_sec': 0.005,
        # Heartbeat-specific (from heartbeat_designer.py testing)
        'heartbeat_filter': 220,                      # Tested value - muffled but audible
        'heartbeat_resonance': 0.0,                   # Minimal, natural
        'heartbeat_bpm': 70,                          # Heartbeat rate
        'heartbeat_lub_dub_gap': 0.080,               # 80ms gap between LUB and dub
        'heartbeat_root_midi': 36,                    # Low C (~65Hz sub-bass)
        'heartbeat_dub_offset': -2,                   # dub is 2 semitones lower than LUB
        'heartbeat_dub_volume': 0.7,                  # dub is 70% volume of LUB
        'heartbeat_amp_env': (0.003, 0.06, 0.03, 0.20),  # Heartbeat-only ADSR
        'heartbeat_filter_env': (0.01, 0.1, 0.0, 0.1),   # Heartbeat-only filter envelope
    })

    # === MOMENT EVENT PARAMETERS ===
    # Controls how key moments are rendered as events with duration and emphasis
    MOMENT_EVENT_PARAMS: Dict = field(default_factory=lambda: {
        'base_duration_sec': 2.5,          # Base duration for a moment (shorter to leave heartbeat room)
        'min_duration_sec': 1.5,           # Minimum audible duration
        'max_duration_sec': 4.0,           # Maximum duration (reduced to prevent overlap)
        'score_duration_mult': 0.3,        # Reduced multiplier to keep moments shorter
        'base_mix_amount': 0.3,            # Base mix amount for moment pattern
        'max_mix_amount': 0.9,             # Maximum mix for highest score moments
        'score_mix_mult': 0.06,            # Score multiplier for mix: mix = base + (score * mult)
        'filter_mod_base': 500,            # Base filter modulation amount
        'filter_mod_per_score': 200,       # Additional filter modulation per score point
        'crossfade_duration_sec': 0.5,     # Duration of crossfade between patterns
        'base_pattern_level': 0.4,         # Volume of base PULSE pattern during moments
        'moment_spacing_sec': 1.0,         # Minimum gap between moments for heartbeat breathing
    })

    # === PROCESS TRANSFORMATION PARAMETERS ===
    PROCESS_PARAMS: Dict[str, Dict] = field(default_factory=lambda: {
        'TUMBLING_DEFEAT': {
            'mistake_weights': {
                'INACCURACY': 0.05,
                'MISTAKE': 0.1,
                'BLUNDER': 0.2,
            },
            'base_decay': 0.3,
            'chaos_factor': 0.02,
            'tempo_drift_clamp': 0.3,
            'pitch_drift_multiplier': 20,
            'volume_decay_rate': 0.3,
        },
        'ATTACKING_MASTERPIECE': {
            'brilliance_weights': {
                'STRONG': 0.15,
                'BRILLIANT': 0.25,
            },
            'crescendo_exponent': 1.5,
            'max_momentum': 1.2,
            'tempo_base': 0.8,
            'tempo_range': 0.5,
            'filter_brightness_base': 0.3,
            'filter_brightness_range': 0.7,
        },
        'QUIET_PRECISION': {
            'disturbance': 0.05,
            'balance_decay': 0.95,
            'breathing_increment': 0.1,
            'breathing_amplitude': 0.08,
        },
    })

    # === BASE NOTE DURATION ===
    BASE_NOTE_DURATION: float = 0.5

    # === DEFAULT BPM ===
    DEFAULT_BPM: int = 120

    # === SAMPLE RATE ===
    SAMPLE_RATE: int = 88200  # 2x 44.1kHz - reduces PolyBLEP aliasing at high frequencies

    # === WAV OUTPUT PARAMETERS ===
    WAV_OUTPUT: Dict = field(default_factory=lambda: {
        'sample_width': 2,
        'channels': 2,  # Stereo output
        'amplitude_multiplier': 30000,
        'clamp_min': -32000,
        'clamp_max': 32000,
        'normalization_threshold': 0.9,
    })

    # === STEREO PANNING ===
    STEREO_CONFIG: Dict = field(default_factory=lambda: {
        'white_pan': -0.7,  # White pieces pan left (-1.0 = full left)
        'black_pan': 0.7,   # Black pieces pan right (1.0 = full right)
        'drone_pan': 0.0,   # Drone centered
        'min_width': 0.0,   # Low tension = narrow stereo
        'max_width': 0.8,   # High tension = wide stereo
        'entropy_pan_amount': 0.6,  # Max pan deviation for entropy
    })

    # === ENTROPY CONFIGURATION (Laurie Spiegel-inspired) ===
    ENTROPY_CONFIG: Dict = field(default_factory=lambda: {
        # Calculation weights
        'weights': {
            'eval': 0.5,      # Evaluation volatility (primary)
            'tactical': 0.4,  # Tactical density
            'time': 0.1,      # Thinking time (if available)
        },

        # Window sizes for local calculations
        'eval_window': 5,      # Plies for eval volatility
        'tactical_window': 5,  # Plies for tactical density

        # Smoothing
        'smoothing_sigma': 2.0,  # Gaussian filter sigma

        # Musical thresholds
        'low_threshold': 0.3,   # Below = simple
        'high_threshold': 0.7,  # Above = complex

        # Parameter ranges for note pools
        'note_pools': {
            'low': [0, 4],                    # Simple: root-fifth
            'medium': [0, 2, 4, 5, 7],        # Moderate: diatonic
            'high': [0, 1, 2, 3, 4, 5, 6, 7], # Complex: chromatic
        },

        # Musical parameter modulation ranges
        'rhythm_variation_max': 0.5,  # Max ±50% timing variation at high entropy
        'filter_lfo_range': (0.02, 0.12),  # Hz (slow to fast)
        'glide_reduction_max': 0.5,  # Max 50% reduction of portamento at high entropy
        'harmony_probability_threshold': 0.7,  # Start adding harmonies above this entropy
    })


# Global default config instance
DEFAULT_CONFIG = SynthConfig()


def get_narrative_params(narrative: str, config: SynthConfig = None) -> Dict:
    """Helper to get narrative parameters with fallback"""
    if config is None:
        config = DEFAULT_CONFIG

    # Check for exact match
    if narrative in config.NARRATIVE_BASE_PARAMS:
        return config.NARRATIVE_BASE_PARAMS[narrative]

    # Check for partial matches
    for key in config.NARRATIVE_BASE_PARAMS:
        if key in narrative:
            return config.NARRATIVE_BASE_PARAMS[key]

    # Return default
    return config.NARRATIVE_BASE_PARAMS['DEFAULT']


def get_section_modulation(narrative: str, config: SynthConfig = None) -> Dict:
    """Helper to get section modulation with fallback"""
    if config is None:
        config = DEFAULT_CONFIG

    # Check for exact match
    if narrative in config.SECTION_MODULATIONS:
        return config.SECTION_MODULATIONS[narrative]

    # Check for partial matches
    for key in config.SECTION_MODULATIONS:
        if key in narrative:
            return config.SECTION_MODULATIONS[key]

    # Return default
    return config.SECTION_MODULATIONS['DEFAULT']


def get_envelope(name: str, config: SynthConfig = None) -> Tuple[float, float, float, float]:
    """Helper to get envelope by name"""
    if config is None:
        config = DEFAULT_CONFIG

    return config.ENVELOPES.get(name, config.ENVELOPES['stab'])


def get_filter_envelope(name: str, config: SynthConfig = None) -> Tuple[float, float, float, float]:
    """Helper to get filter envelope by name"""
    if config is None:
        config = DEFAULT_CONFIG

    return config.FILTER_ENVELOPES.get(name, config.FILTER_ENVELOPES['gentle'])
