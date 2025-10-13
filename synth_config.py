"""
SYNTH_CONFIG - Backward Compatibility Layer for Chess Music Synthesis

DEPRECATED: This module provides backward compatibility for existing code.
New code should import from config_service instead:
    from config_service import get_config
    config = get_config()

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
from config_service import get_config


# === YAML LOADER HELPERS ===
# These functions load configuration sections from config.yaml

def _load_scales() -> Dict[str, List[float]]:
    """Load scales from YAML config"""
    cfg = get_config()
    scales = {}
    scales_config = cfg.get('composition.harmony.scales', {})
    for name, scale_data in scales_config.items():
        scales[name] = scale_data.get('frequencies', [])
    return scales


def _load_envelopes() -> Dict[str, Tuple[float, float, float, float]]:
    """Load amplitude envelopes from YAML config"""
    cfg = get_config()
    envelopes = {}
    env_config = cfg.get('synthesis.amplitude.envelopes', {})
    for name, env in env_config.items():
        envelopes[name] = (env.get('a', 0.01), env.get('d', 0.1),
                          env.get('s', 0.7), env.get('r', 0.2))
    return envelopes


def _load_filter_envelopes() -> Dict[str, Tuple[float, float, float, float]]:
    """Load filter envelopes from YAML config"""
    cfg = get_config()
    envelopes = {}
    env_config = cfg.get('synthesis.filters.envelopes', {})
    for name, env in env_config.items():
        envelopes[name] = (env.get('a', 0.01), env.get('d', 0.15),
                          env.get('s', 0.3), env.get('r', 0.2))
    return envelopes


def _load_narrative_params() -> Dict[str, Dict]:
    """
    Load macro game arc parameters from YAML.
    Returns nested structure as defined in YAML.
    """
    cfg = get_config()
    return cfg.get('composition.macro_game_arc', {})


def _load_section_modulations() -> Dict[str, Dict]:
    """Load section modulation parameters from YAML"""
    cfg = get_config()
    return cfg.get('composition.meso_section_modulation', {})


def _load_moment_voices() -> Dict[str, Dict]:
    """
    Load gesture moment voices from YAML.
    Transforms nested structure to flat keys for backward compatibility.
    YAML: BLUNDER: {in_defeat: {...}, in_masterpiece: {...}}
    Code expects: BLUNDER_IN_DEFEAT: {...}, BLUNDER_IN_MASTERPIECE: {...}
    """
    cfg = get_config()
    gestures = cfg.get('composition.micro_gestures', {})
    voices = {}

    for gesture_type, contexts in gestures.items():
        if isinstance(contexts, dict):
            # Check if this has context sub-keys
            if 'in_defeat' in contexts or 'in_masterpiece' in contexts or 'neutral' in contexts:
                # Context-aware gesture
                if 'in_defeat' in contexts:
                    voices[f'{gesture_type}_IN_DEFEAT'] = contexts['in_defeat']
                if 'in_masterpiece' in contexts:
                    voices[f'{gesture_type}_IN_MASTERPIECE'] = contexts['in_masterpiece']
                if 'neutral' in contexts:
                    voices[f'{gesture_type}_NEUTRAL'] = contexts['neutral']
            else:
                # Simple gesture
                voices[gesture_type] = contexts

    return voices


def _load_development_params() -> Dict[str, Dict]:
    """Load development moment parameters from YAML"""
    cfg = get_config()
    return cfg.get('composition.special_moments.DEVELOPMENT', {})


def _load_first_exchange_params() -> Dict[str, Dict]:
    """Load first exchange parameters from YAML"""
    cfg = get_config()
    return cfg.get('composition.special_moments.FIRST_EXCHANGE', {})


def _load_sequencer_patterns() -> Dict:
    """Load sequencer patterns from YAML"""
    cfg = get_config()
    return cfg.get('composition.sequencer.patterns', {})


def _load_supersaw_detune() -> Dict[str, List[float]]:
    """Load supersaw detune presets from YAML"""
    cfg = get_config()
    return cfg.get('synthesis.oscillators.supersaw.detune_presets', {})


def _load_timing() -> Dict[str, float]:
    """Load timing parameters from YAML"""
    cfg = get_config()
    return cfg.get('composition.rhythm.timing', {})


def _load_layer_enable() -> Dict[str, bool]:
    """Load layer enable flags from YAML"""
    cfg = get_config()
    return cfg.get('mixing.layer_enable', {})


def _load_mixing() -> Dict[str, float]:
    """
    Load mixing levels from YAML.
    Combines all mixing parameters and adds _level suffix for layer levels.
    """
    cfg = get_config()
    mixing = {}

    # Layer levels - add _level suffix for backward compatibility
    layers = cfg.get('mixing.layers', {})
    for key, value in layers.items():
        mixing[f'{key}_level'] = value

    # Dynamics
    mixing.update(cfg.get('mixing.dynamics', {}))

    # Compression
    mixing.update(cfg.get('mixing.compression', {}))

    # Special
    filtered_seq_level = cfg.get('mixing.filtered_sequence_level')
    if filtered_seq_level is not None:
        mixing['filtered_sequence_level'] = filtered_seq_level

    return mixing


def _load_layer_mixing() -> Dict[str, float]:
    """Load layer interaction parameters from YAML"""
    cfg = get_config()
    # Load layer_interaction section
    return cfg.get('mixing.layer_interaction', {})


def _load_melodic_patterns() -> Dict[str, Dict]:
    """Load melodic patterns from YAML"""
    cfg = get_config()
    return cfg.get('composition.melodic_patterns', {})


def _load_sequencer_synth() -> Dict:
    """Load sequencer synthesis parameters from YAML"""
    cfg = get_config()
    return cfg.get('composition.sequencer.synth', {})


def _load_moment_event_params() -> Dict:
    """Load gesture event parameters from YAML"""
    cfg = get_config()
    return cfg.get('composition.gesture_events', {})


def _load_process_params() -> Dict[str, Dict]:
    """Load process transformation parameters from YAML"""
    cfg = get_config()
    return cfg.get('process_transformation', {})


def _load_wav_output() -> Dict:
    """
    Load WAV output parameters from YAML.
    Expands clamp_range to clamp_min/clamp_max for backward compatibility.
    """
    cfg = get_config()
    wav = cfg.get('mixing.output', {}).copy()

    # Expand clamp_range to clamp_min/clamp_max
    if 'clamp_range' in wav:
        clamp_range = wav.pop('clamp_range')
        if isinstance(clamp_range, list) and len(clamp_range) >= 2:
            wav['clamp_min'] = clamp_range[0]
            wav['clamp_max'] = clamp_range[1]

    return wav


def _load_stereo_config() -> Dict:
    """
    Load stereo configuration from YAML.
    Expands width_range array to min_width/max_width for backward compatibility.
    """
    cfg = get_config()
    stereo = cfg.get('mixing.stereo', {}).copy()

    # Expand width_range to min_width/max_width
    if 'width_range' in stereo:
        width_range = stereo.pop('width_range')
        if isinstance(width_range, list) and len(width_range) >= 2:
            stereo['min_width'] = width_range[0]
            stereo['max_width'] = width_range[1]

    return stereo


def _load_entropy_config() -> Dict:
    """Load entropy configuration from YAML"""
    cfg = get_config()
    return cfg.get('entropy', {})


@dataclass
class SynthConfig:
    """
    Centralized configuration for all synthesis parameters.
    Loads from config.yaml via config_service.
    """

    # === MUSICAL SCALES ===
    SCALES: Dict[str, List[float]] = field(default_factory=_load_scales)

    # === ENVELOPE PRESETS (attack, decay, sustain, release) ===
    ENVELOPES: Dict[str, Tuple[float, float, float, float]] = field(default_factory=_load_envelopes)

    FILTER_ENVELOPES: Dict[str, Tuple[float, float, float, float]] = field(default_factory=_load_filter_envelopes)

    # === OVERALL NARRATIVE BASE PARAMETERS ===
    # These define the fundamental character of the entire piece
    NARRATIVE_BASE_PARAMS: Dict[str, Dict] = field(default_factory=_load_narrative_params)

    # === SECTION NARRATIVE MODULATION PARAMETERS ===
    # These modulate the base parameters for specific game sections
    SECTION_MODULATIONS: Dict[str, Dict] = field(default_factory=_load_section_modulations)

    # === KEY MOMENT VOICE PARAMETERS ===
    # NOTE: These are INTERNAL mappings used by the composer
    # The tagger produces simple types (BLUNDER, BRILLIANT, etc.)
    # The composer maps them to context-aware voices based on overall_narrative
    # Example: BLUNDER + TUMBLING_DEFEAT context → BLUNDER_IN_DEFEAT voice
    MOMENT_VOICES: Dict[str, Dict] = field(default_factory=_load_moment_voices)

    # === DEVELOPMENT MOMENT PARAMETERS ===
    DEVELOPMENT_PARAMS: Dict[str, Dict] = field(default_factory=_load_development_params)

    # === FIRST EXCHANGE MOMENT PARAMETERS ===
    FIRST_EXCHANGE_PARAMS: Dict[str, Dict] = field(default_factory=_load_first_exchange_params)

    # === SEQUENCER PATTERNS ===
    SEQUENCER_PATTERNS: Dict = field(default_factory=_load_sequencer_patterns)

    # === SUPERSAW DETUNE PRESETS ===
    SUPERSAW_DETUNE: Dict[str, List[float]] = field(default_factory=_load_supersaw_detune)

    # === TIMING PARAMETERS (in seconds or milliseconds) ===
    TIMING: Dict[str, float] = field(default_factory=_load_timing)

    # === LAYER MUTING (True = enabled, False = muted) ===
    LAYER_ENABLE: Dict[str, bool] = field(default_factory=_load_layer_enable)

    # === MIXING LEVELS ===
    MIXING: Dict[str, float] = field(default_factory=_load_mixing)

    # === LAYER MIXING ===
    LAYER_MIXING: Dict[str, float] = field(default_factory=_load_layer_mixing)

    # === MELODIC PATTERN PARAMETERS ===
    MELODIC_PATTERNS: Dict[str, Dict] = field(default_factory=_load_melodic_patterns)

    # === SEQUENCER SYNTHESIS PARAMETERS ===
    SEQUENCER_SYNTH: Dict = field(default_factory=_load_sequencer_synth)

    # === MOMENT EVENT PARAMETERS ===
    # Controls how key moments are rendered as events with duration and emphasis
    MOMENT_EVENT_PARAMS: Dict = field(default_factory=_load_moment_event_params)

    # === PROCESS TRANSFORMATION PARAMETERS ===
    PROCESS_PARAMS: Dict[str, Dict] = field(default_factory=_load_process_params)

    # === BASE NOTE DURATION ===
    BASE_NOTE_DURATION: float = field(default_factory=lambda: get_config().get('composition.rhythm.base_duration', 0.5))

    # === DEFAULT BPM ===
    DEFAULT_BPM: int = field(default_factory=lambda: get_config().get('composition.rhythm.default_bpm', 120))

    # === SAMPLE RATE ===
    SAMPLE_RATE: int = field(default_factory=lambda: get_config().get('synthesis.sample_rate', 88200))

    # === WAV OUTPUT PARAMETERS ===
    WAV_OUTPUT: Dict = field(default_factory=_load_wav_output)

    # === STEREO PANNING ===
    STEREO_CONFIG: Dict = field(default_factory=_load_stereo_config)

    # === ENTROPY CONFIGURATION (Laurie Spiegel-inspired) ===
    ENTROPY_CONFIG: Dict = field(default_factory=_load_entropy_config)


# Global default config instance
DEFAULT_CONFIG = SynthConfig()


def get_narrative_params(narrative: str, config: Optional[SynthConfig] = None) -> Dict:
    """
    Helper to get narrative parameters with fallback.

    NOTE: Config parameter is deprecated and ignored.
    Now reads from YAML via config_service.

    Args:
        narrative: Narrative type (e.g., 'TUMBLING_DEFEAT')
        config: DEPRECATED - ignored for backward compatibility

    Returns:
        Dictionary of narrative parameters
    """
    # Try exact match
    if narrative in DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS:
        return DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS[narrative]

    # Try partial match
    for key in DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS:
        if key in narrative:
            return DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS[key]

    # Return default
    return DEFAULT_CONFIG.NARRATIVE_BASE_PARAMS['DEFAULT']


def get_section_modulation(narrative: str, config: Optional[SynthConfig] = None) -> Dict:
    """
    Helper to get section modulation with fallback.

    NOTE: Config parameter is deprecated and ignored.
    Now reads from YAML via config_service.

    Args:
        narrative: Section narrative (e.g., 'KING_HUNT')
        config: DEPRECATED - ignored for backward compatibility

    Returns:
        Dictionary of modulation parameters
    """
    # Try exact match
    if narrative in DEFAULT_CONFIG.SECTION_MODULATIONS:
        return DEFAULT_CONFIG.SECTION_MODULATIONS[narrative]

    # Try partial match
    for key in DEFAULT_CONFIG.SECTION_MODULATIONS:
        if key in narrative:
            return DEFAULT_CONFIG.SECTION_MODULATIONS[key]

    # Return default
    return DEFAULT_CONFIG.SECTION_MODULATIONS['DEFAULT']


def get_envelope(name: str, config: Optional[SynthConfig] = None) -> Tuple[float, float, float, float]:
    """
    Helper to get envelope by name.

    NOTE: Config parameter is deprecated and ignored.
    Now reads from YAML via config_service.

    Args:
        name: Envelope name (e.g., 'percussive', 'drone')
        config: DEPRECATED - ignored for backward compatibility

    Returns:
        ADSR tuple (attack, decay, sustain, release)
    """
    return DEFAULT_CONFIG.ENVELOPES.get(name, DEFAULT_CONFIG.ENVELOPES['stab'])


def get_filter_envelope(name: str, config: Optional[SynthConfig] = None) -> Tuple[float, float, float, float]:
    """
    Helper to get filter envelope by name.

    NOTE: Config parameter is deprecated and ignored.
    Now reads from YAML via config_service.

    Args:
        name: Filter envelope name (e.g., 'gentle', 'sweep')
        config: DEPRECATED - ignored for backward compatibility

    Returns:
        ADSR tuple (attack, decay, sustain, release)
    """
    return DEFAULT_CONFIG.FILTER_ENVELOPES.get(name, DEFAULT_CONFIG.FILTER_ENVELOPES['gentle'])
