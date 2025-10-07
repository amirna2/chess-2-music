# Layer 3b Implementation: Unified Moment Gesture System

**Status:** Design Document
**Date:** 2025-10-05
**Replaces:** Scattered Layer 3b implementation

## Overview

Layer 3b generates **emotional gesture moments** that punctuate the musical narrative. This document specifies a **unified blueprint architecture** where all gesture archetypes (BLUNDER, BRILLIANT, TIME_PRESSURE, etc.) share the same synthesis pipeline and differ only in configuration parameters.

This architecture **mirrors** the existing `synth_composer/patterns/` system used for Layer 2.

---

## Architecture Principles

### 1. Unified Blueprint Pattern

**All gesture archetypes use the same generation pipeline:**

```
Moment Event + Section Context
    ↓
Duration Calculation
    ↓
Phase Timeline Computation (pre-shadow → impact → bloom → decay → residue)
    ↓
Parameter Curve Generation (pitch, harmony, filter, envelope, texture)
    ↓
Audio Synthesis (oscillators → filters → envelopes → mix)
    ↓
Finalization (normalization, safety clipping)
    ↓
Rendered Audio (np.ndarray)
```

**Archetypes differ only in:**
- Phase duration ratios
- Curve shape types (exponential vs linear vs oscillating)
- Harmonic structure (cluster vs chord vs unison)
- Filter trajectory (opening vs closing vs morphing)
- Texture parameters (noise ratio, shimmer, etc.)

### 2. Parallel to Pattern System

| **Layer 2 (Patterns)** | **Layer 3b (Gestures)** |
|------------------------|-------------------------|
| `PatternGenerator` base class | `GestureGenerator` base class |
| `generate_events()` → List[NoteEvent] | `generate_gesture()` → np.ndarray |
| State machines (ATTACK/RETREAT) | Phase timelines (pre-shadow/bloom/decay) |
| Note selection logic | Pitch/harmony curve generation |
| `NoteSynthesizer` (shared) | `GestureSynthesizer` (shared) |
| Pattern registry in coordinator | Archetype registry in coordinator |

### 3. Configuration-Driven

Each archetype is defined by a **configuration dictionary** (not a custom class file). This makes adding new archetypes trivial.

---

## File Structure

```
layer3b/
├── __init__.py
├── base.py                    # GestureGenerator base class
├── archetype_configs.py       # Archetype parameter definitions
├── curve_generators.py        # Reusable curve generation functions
├── synthesizer.py             # Gesture audio renderer
├── coordinator.py             # Archetype registry and moment dispatcher
└── utils.py                   # Phase computation, normalization, safety
```

---

## Core Components

### 1. Base Class (`base.py`)

```python
"""
Base class for all gesture generators.
"""

import numpy as np
from typing import Dict, Any
from .curve_generators import (
    generate_pitch_curve,
    generate_harmony,
    generate_filter_curve,
    generate_envelope,
    generate_texture_curve
)
from .synthesizer import GestureSynthesizer
from .utils import compute_phases, finalize_audio


class GestureGenerator:
    """
    Unified gesture generator for all archetypes.

    All gestures follow the same pipeline:
    1. Compute duration based on archetype and section context
    2. Compute phase timeline (pre-shadow → impact → bloom → decay → residue)
    3. Generate parameter curves (pitch, harmony, filter, envelope, texture)
    4. Synthesize audio using shared synthesis engine
    5. Finalize (normalize, safety clip)

    Archetypes differ only in configuration parameters.
    """

    def __init__(self, archetype_config: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize gesture generator.

        Args:
            archetype_config: Configuration dict defining gesture characteristics
            rng: NumPy random generator for reproducible randomness
        """
        self.config = archetype_config
        self.rng = rng
        self.synthesizer = GestureSynthesizer()

    def generate_gesture(self,
                        moment_event: Dict[str, Any],
                        section_context: Dict[str, Any],
                        sample_rate: int) -> np.ndarray:
        """
        Generate complete audio gesture for a moment event.

        Args:
            moment_event: Moment metadata (event_type, timestamp, move_number, etc.)
            section_context: Section-level parameters (tension, entropy, scale, etc.)
            sample_rate: Audio sample rate (Hz)

        Returns:
            Mono audio buffer (np.ndarray)
        """
        # 1. Compute duration
        duration = self._compute_duration(section_context)
        total_samples = int(duration * sample_rate)

        # 2. Compute phase timeline
        phases = compute_phases(
            self.config['phases'],
            total_samples,
            section_context,
            self.rng
        )

        # 3. Generate parameter curves
        pitch_curve = generate_pitch_curve(
            self.config['pitch'],
            phases,
            section_context,
            total_samples,
            self.rng,
            sample_rate
        )

        harmony_voices = generate_harmony(
            self.config['harmony'],
            pitch_curve,
            phases,
            section_context,
            self.rng
        )

        filter_curve = generate_filter_curve(
            self.config['filter'],
            phases,
            section_context,
            total_samples,
            self.rng,
            sample_rate
        )

        envelope = generate_envelope(
            self.config['envelope'],
            phases,
            total_samples,
            self.rng,
            sample_rate
        )

        texture_curve = generate_texture_curve(
            self.config['texture'],
            phases,
            section_context,
            total_samples,
            self.rng
        )

        # 4. Synthesize audio
        audio = self.synthesizer.synthesize(
            pitch_voices=harmony_voices,
            filter_curve=filter_curve,
            envelope=envelope,
            texture_curve=texture_curve,
            sample_rate=sample_rate
        )

        # 5. Finalize (normalize, safety clip)
        audio = finalize_audio(
            audio,
            peak_limit=self.config.get('peak_limit', 0.8),
            rms_target=self.config.get('rms_target', -18.0)
        )

        return audio

    def _compute_duration(self, section_context: Dict[str, Any]) -> float:
        """
        Compute gesture duration based on archetype and section context.

        Args:
            section_context: Section-level parameters (tension, entropy, etc.)

        Returns:
            Duration in seconds
        """
        base = self.config['duration_base']
        tension_scale = self.config.get('duration_tension_scale', 0.0)
        entropy_scale = self.config.get('duration_entropy_scale', 0.0)

        tension = section_context.get('tension', 0.5)
        entropy = section_context.get('entropy', 0.5)

        duration = base + (tension * tension_scale) + (entropy * entropy_scale)

        # Clamp to reasonable range
        return np.clip(duration, 0.5, 10.0)
```

---

### 2. Archetype Configurations (`archetype_configs.py`)

```python
"""
Archetype configuration definitions.

Each archetype is a dictionary specifying:
- Duration scaling
- Phase ratios
- Pitch trajectory type and parameters
- Harmony structure
- Filter path
- Envelope shape
- Texture characteristics
"""

ARCHETYPES = {
    "BLUNDER": {
        # Duration
        "duration_base": 2.5,
        "duration_tension_scale": 1.2,
        "duration_entropy_scale": 0.0,

        # Phase ratios (must sum to ~1.0)
        "phases": {
            "pre_shadow": 0.15,    # Suspended high
            "impact": 0.05,         # Onset
            "bloom": 0.25,          # Gliss down
            "decay": 0.35,          # Fall continues
            "residue": 0.20         # Tail
        },

        # Pitch trajectory
        "pitch": {
            "type": "exponential_gliss",
            "start_freq_base": 880,              # A5
            "start_freq_entropy_scale": 440,     # Up to E6 with entropy
            "octave_drop_base": 2,               # 2 octaves
            "octave_drop_tension_scale": 1,      # +1 octave with tension
            "gliss_phases": ["bloom", "decay"]   # Gliss during these phases
        },

        # Harmony
        "harmony": {
            "type": "cluster_to_interval",
            "num_voices_base": 3,
            "num_voices_tension_scale": 2,       # 3-5 voices
            "cluster_semitone_spacing": 1.0,     # Tight cluster
            "resolve_interval_type": "muddy",    # Tritone or minor 2nd
            "resolve_phase": "decay"
        },

        # Filter
        "filter": {
            "type": "bandpass_to_lowpass_choke",
            "bp_center_base": 1000,              # 1 kHz
            "bp_center_tension_scale": 1000,     # Up to 2 kHz
            "bp_bandwidth": 800,
            "lp_cutoff_base": 100,               # Choke to 100 Hz
            "lp_cutoff_tension_scale": 100,      # Up to 200 Hz
            "lp_resonance_base": 0.7,            # High Q
            "lp_resonance_tension_scale": 0.25,
            "morph_phase": "bloom"
        },

        # Envelope
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 1,
            "attack_ms_entropy_scale": 4,        # 1-5ms attack
            "sustain_phase_ratio": 0.15,         # 15% sustain
            "decay_curve": "exponential",
            "decay_coefficient": -4              # Fast decay
        },

        # Texture
        "texture": {
            "noise_ratio_base": 0.3,
            "noise_ratio_entropy_scale": 0.4,    # 30-70% noise
            "noise_type": "pink"
        },

        # Safety
        "peak_limit": 0.8,
        "rms_target": -18.0  # dB
    },

    "BRILLIANT": {
        # Duration (longer, more expansive)
        "duration_base": 3.0,
        "duration_tension_scale": 0.8,
        "duration_entropy_scale": -0.5,  # Shorter with chaos

        # Phase ratios (longer bloom)
        "phases": {
            "pre_shadow": 0.20,
            "impact": 0.10,
            "bloom": 0.40,     # Extended bloom
            "decay": 0.20,
            "residue": 0.10
        },

        # Pitch trajectory (ascending)
        "pitch": {
            "type": "ascending_spread",
            "start_freq_base": 220,              # A3
            "start_freq_entropy_scale": 110,
            "octave_rise": 2,                    # Rise 2 octaves
            "spread_phases": ["impact", "bloom"]
        },

        # Harmony (unison → major chord)
        "harmony": {
            "type": "unison_to_chord",
            "chord_type": "major_seventh",
            "num_voices": 4,
            "spread_phase": "bloom"
        },

        # Filter (opening)
        "filter": {
            "type": "lowpass_to_highpass_open",
            "lp_cutoff_start": 300,
            "hp_cutoff_end": 3000,
            "resonance_base": 0.4,
            "morph_phase": "bloom"
        },

        # Envelope (gradual, sustained)
        "envelope": {
            "type": "gradual_sustained",
            "attack_ms": 50,
            "sustain_phase_ratio": 0.5,
            "decay_curve": "linear"
        },

        # Texture (less noise, add shimmer)
        "texture": {
            "noise_ratio_base": 0.1,
            "noise_ratio_entropy_scale": 0.2,
            "noise_type": "white",
            "shimmer_enable": True,
            "shimmer_rate_hz": 6.0
        },

        # Safety
        "peak_limit": 0.85,
        "rms_target": -16.0
    },

    "TIME_PRESSURE": {
        # Duration (shorter, urgent)
        "duration_base": 1.8,
        "duration_tension_scale": 0.5,
        "duration_entropy_scale": 0.3,

        # Phase ratios (fast bloom)
        "phases": {
            "pre_shadow": 0.10,
            "impact": 0.15,
            "bloom": 0.30,
            "decay": 0.25,
            "residue": 0.20
        },

        # Pitch (oscillating tremor)
        "pitch": {
            "type": "oscillating_tremor",
            "center_freq_base": 440,
            "center_freq_tension_scale": 220,
            "tremor_rate_base_hz": 8,            # 8 Hz tremor
            "tremor_rate_tension_scale_hz": 8,   # Up to 16 Hz
            "tremor_depth_semitones": 2,         # ±2 semitones
            "acceleration_phase": "bloom"
        },

        # Harmony (dense, unstable)
        "harmony": {
            "type": "dense_cluster",
            "num_voices": 5,
            "semitone_spacing_min": 0.5,         # Microtonal
            "semitone_spacing_max": 1.5
        },

        # Filter (nervous sweep)
        "filter": {
            "type": "bandpass_sweep",
            "bp_center_start": 500,
            "bp_center_end": 2000,
            "bp_bandwidth": 600,
            "sweep_rate_modulation": "accelerating",
            "sweep_phase": "bloom"
        },

        # Envelope (rapid attack, short)
        "envelope": {
            "type": "sudden_short_tail",
            "attack_ms_base": 2,
            "attack_ms_entropy_scale": 3,
            "sustain_phase_ratio": 0.20,
            "decay_curve": "exponential",
            "decay_coefficient": -5
        },

        # Texture (high noise)
        "texture": {
            "noise_ratio_base": 0.5,
            "noise_ratio_entropy_scale": 0.3,
            "noise_type": "white"  # Harsher
        },

        # Safety
        "peak_limit": 0.75,
        "rms_target": -20.0
    },

    "TACTICAL_SEQUENCE": {
        # Duration (moderate)
        "duration_base": 2.2,
        "duration_tension_scale": 0.6,
        "duration_entropy_scale": 0.0,

        # Phase ratios
        "phases": {
            "pre_shadow": 0.12,
            "impact": 0.08,
            "bloom": 0.35,
            "decay": 0.30,
            "residue": 0.15
        },

        # Pitch (interlocking cellular automaton)
        "pitch": {
            "type": "cellular_sequence",
            "cell_frequencies": [330, 440, 550, 660],  # Harmonic series
            "cell_duration_ms": 80,                    # Fast cells
            "transition_type": "stepped",              # Quantized jumps
            "pattern_seed": "golden_ratio"             # Deterministic chaos
        },

        # Harmony (precise intervals)
        "harmony": {
            "type": "harmonic_stack",
            "num_voices": 3,
            "interval_ratios": [1.0, 1.5, 2.0],  # Just intonation
            "phase_offset_enable": True           # Phasing effect
        },

        # Filter (rhythmic pulse)
        "filter": {
            "type": "rhythmic_gate",
            "lp_cutoff_base": 800,
            "lp_cutoff_pulse_mult": 3.0,          # 3x cutoff on pulse
            "pulse_rate_hz": 12,                  # 12 Hz pulse
            "pulse_duty_cycle": 0.3               # 30% open
        },

        # Envelope (gated)
        "envelope": {
            "type": "gated_pulse",
            "attack_ms": 5,
            "gate_duration_ms": 60,
            "release_ms": 20,
            "pulse_rate_hz": 12                   # Match filter pulse
        },

        # Texture (clean, mechanical)
        "texture": {
            "noise_ratio_base": 0.05,             # Minimal noise
            "noise_ratio_entropy_scale": 0.1,
            "noise_type": "pink"
        },

        # Safety
        "peak_limit": 0.8,
        "rms_target": -17.0
    }
}
```

---

### 3. Curve Generators (`curve_generators.py`)

```python
"""
Reusable parameter curve generation functions.

Each function takes:
- config: Parameter subset from archetype config
- phases: Phase timeline dict with sample boundaries
- section_context: Section-level parameters (tension, entropy, scale)
- total_samples: Total gesture duration in samples
- rng: Random generator

Returns:
- Curve as numpy array (length = total_samples)
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
    Route to specific pitch trajectory algorithm.

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
    Exponential glissando (BLUNDER: high → low).

    Phases:
    - pre_shadow: Suspended at high pitch
    - bloom/decay: Exponential downward gliss
    - residue: Hold at target
    """
    tension = section_context.get('tension', 0.5)
    entropy = section_context.get('entropy', 0.5)

    # Compute frequencies
    peak_freq = config['start_freq_base'] + (entropy * config['start_freq_entropy_scale'])
    octave_drop = config['octave_drop_base'] + (tension * config['octave_drop_tension_scale'])
    target_freq = peak_freq / (2 ** octave_drop)

    # Extract phase boundaries
    pre_shadow_end = phases['pre_shadow']['end_sample']
    decay_end = phases['decay']['end_sample']

    # Build curve
    curve = np.zeros(total_samples)

    # Phase 1: Suspended (constant high pitch)
    curve[:pre_shadow_end] = peak_freq

    # Phase 2: Exponential gliss
    gliss_samples = decay_end - pre_shadow_end
    curve[pre_shadow_end:decay_end] = np.exp(
        np.linspace(np.log(peak_freq), np.log(target_freq), gliss_samples)
    )

    # Phase 3: Residue (hold at target)
    curve[decay_end:] = target_freq

    return curve


def _pitch_ascending_spread(config: Dict[str, Any],
                            phases: Dict[str, Any],
                            section_context: Dict[str, Any],
                            total_samples: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Ascending pitch spread (BRILLIANT: low → high).
    """
    entropy = section_context.get('entropy', 0.5)

    start_freq = config['start_freq_base'] + (entropy * config['start_freq_entropy_scale'])
    end_freq = start_freq * (2 ** config['octave_rise'])

    # Spread during impact + bloom phases
    spread_start = phases['impact']['start_sample']
    spread_end = phases['bloom']['end_sample']

    curve = np.zeros(total_samples)
    curve[:spread_start] = start_freq

    # Exponential rise
    spread_samples = spread_end - spread_start
    curve[spread_start:spread_end] = np.exp(
        np.linspace(np.log(start_freq), np.log(end_freq), spread_samples)
    )

    curve[spread_end:] = end_freq

    return curve


def _pitch_oscillating_tremor(config: Dict[str, Any],
                              phases: Dict[str, Any],
                              section_context: Dict[str, Any],
                              total_samples: int,
                              rng: np.random.Generator,
                              sample_rate: int = 88200) -> np.ndarray:
    """
    Oscillating tremor (TIME_PRESSURE: accelerating vibrato).
    """
    tension = section_context.get('tension', 0.5)

    center_freq = config['center_freq_base'] + (tension * config['center_freq_tension_scale'])
    tremor_rate = config['tremor_rate_base_hz'] + (tension * config['tremor_rate_tension_scale_hz'])
    tremor_depth = config['tremor_depth_semitones']

    # Accelerating tremor during bloom phase
    bloom_start = phases['bloom']['start_sample']
    bloom_end = phases['bloom']['end_sample']

    curve = np.full(total_samples, center_freq)

    # Generate accelerating LFO
    t = np.arange(bloom_end - bloom_start) / sample_rate
    phase_acceleration = np.cumsum(tremor_rate * (1 + t / t[-1]))  # Double rate by end
    lfo = np.sin(2 * np.pi * phase_acceleration / sample_rate)

    # Apply tremor (±semitones)
    tremor_mult = 2 ** ((lfo * tremor_depth) / 12)
    curve[bloom_start:bloom_end] = center_freq * tremor_mult

    return curve


def _pitch_cellular_sequence(config: Dict[str, Any],
                             phases: Dict[str, Any],
                             section_context: Dict[str, Any],
                             total_samples: int,
                             rng: np.random.Generator,
                             sample_rate: int = 88200) -> np.ndarray:
    """
    Cellular automaton sequence (TACTICAL_SEQUENCE: discrete pitch cells).

    WARNING: Original doc noted this creates aliasing with per-sample cell changes.
    TODO: Use block-based or interpolated transitions.
    """
    cells = config['cell_frequencies']
    cell_dur_samples = int(config['cell_duration_ms'] * sample_rate / 1000)

    curve = np.zeros(total_samples)

    # Simple deterministic pattern (golden ratio seed)
    # TODO: Implement proper cellular automaton
    cell_idx = 0
    for i in range(0, total_samples, cell_dur_samples):
        end_idx = min(i + cell_dur_samples, total_samples)
        curve[i:end_idx] = cells[cell_idx % len(cells)]
        cell_idx += 1
        if i > phases['bloom']['start_sample']:
            cell_idx += 1  # Accelerate during bloom

    return curve


def generate_harmony(config: Dict[str, Any],
                    base_pitch_curve: np.ndarray,
                    phases: Dict[str, Any],
                    section_context: Dict[str, Any],
                    rng: np.random.Generator) -> List[np.ndarray]:
    """
    Generate harmonic voices based on base pitch curve.

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
    Cluster (tight semitones) resolving to muddy interval (BLUNDER).
    """
    tension = section_context.get('tension', 0.5)
    num_voices = int(config['num_voices_base'] + tension * config['num_voices_tension_scale'])

    # Determine muddy interval (tritone or minor 2nd)
    if config['resolve_interval_type'] == "muddy":
        muddy_interval = 6 if tension > 0.5 else 1  # 6=tritone, 1=minor2nd

    # Transition point
    transition_sample = phases[config['resolve_phase']]['start_sample']

    voices = []
    for i in range(num_voices):
        voice_curve = base_pitch_curve.copy()

        # Cluster detuning (tight semitones)
        cluster_detune = (i - num_voices // 2) * config['cluster_semitone_spacing']

        # Before transition: cluster
        voice_curve[:transition_sample] *= (2 ** (cluster_detune / 12))

        # After transition: resolve to muddy interval (keep only 2 voices)
        if i < 2:
            resolved_detune = muddy_interval if i == 1 else 0
            voice_curve[transition_sample:] *= (2 ** (resolved_detune / 12))
            voices.append(voice_curve)

    return voices


def _harmony_unison_to_chord(config: Dict[str, Any],
                             base_pitch_curve: np.ndarray,
                             phases: Dict[str, Any],
                             section_context: Dict[str, Any],
                             rng: np.random.Generator) -> List[np.ndarray]:
    """
    Unison spreading to chord (BRILLIANT).
    """
    num_voices = config['num_voices']

    # Chord intervals (major seventh: 0, 4, 7, 11 semitones)
    if config['chord_type'] == "major_seventh":
        intervals = [0, 4, 7, 11]

    spread_start = phases[config['spread_phase']]['start_sample']

    voices = []
    for i, interval in enumerate(intervals[:num_voices]):
        voice_curve = base_pitch_curve.copy()

        # Unison before spread
        voice_curve[:spread_start] = base_pitch_curve[:spread_start]

        # Spread to chord
        voice_curve[spread_start:] *= (2 ** (interval / 12))
        voices.append(voice_curve)

    return voices


def _harmony_dense_cluster(config: Dict[str, Any],
                           base_pitch_curve: np.ndarray,
                           phases: Dict[str, Any],
                           section_context: Dict[str, Any],
                           rng: np.random.Generator) -> List[np.ndarray]:
    """
    Dense microtonal cluster (TIME_PRESSURE).
    """
    num_voices = config['num_voices']

    voices = []
    for i in range(num_voices):
        voice_curve = base_pitch_curve.copy()

        # Random microtonal detuning
        detune = rng.uniform(
            config['semitone_spacing_min'],
            config['semitone_spacing_max']
        ) * (i - num_voices // 2)

        voice_curve *= (2 ** (detune / 12))
        voices.append(voice_curve)

    return voices


def _harmony_harmonic_stack(config: Dict[str, Any],
                            base_pitch_curve: np.ndarray,
                            phases: Dict[str, Any],
                            section_context: Dict[str, Any],
                            rng: np.random.Generator) -> List[np.ndarray]:
    """
    Harmonic stack with just intonation (TACTICAL_SEQUENCE).
    """
    num_voices = config['num_voices']
    ratios = config['interval_ratios']

    voices = []
    for i, ratio in enumerate(ratios[:num_voices]):
        voice_curve = base_pitch_curve * ratio

        # Optional phase offset (phasing effect)
        if config.get('phase_offset_enable', False):
            offset_samples = int(i * 100)  # Small time offset
            voice_curve = np.roll(voice_curve, offset_samples)

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
    Band-pass to low-pass choke (BLUNDER).
    """
    tension = section_context.get('tension', 0.5)

    bp_center = config['bp_center_base'] + (tension * config['bp_center_tension_scale'])
    lp_cutoff = config['lp_cutoff_base'] + (tension * config['lp_cutoff_tension_scale'])
    lp_resonance = config['lp_resonance_base'] + (tension * config.get('lp_resonance_tension_scale', 0))

    morph_start = phases[config['morph_phase']]['start_sample']

    # Cutoff curve
    cutoff = np.zeros(total_samples)
    cutoff[:morph_start] = bp_center
    cutoff[morph_start:] = np.exp(
        np.linspace(np.log(bp_center), np.log(lp_cutoff), total_samples - morph_start)
    )

    # Resonance curve
    resonance = np.zeros(total_samples)
    resonance[:morph_start] = 0.4  # Moderate BP resonance
    resonance[morph_start:] = np.linspace(0.4, lp_resonance, total_samples - morph_start)

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'bandpass->lowpass'
    }


def _filter_lp_to_hp_open(config: Dict[str, Any],
                         phases: Dict[str, Any],
                         section_context: Dict[str, Any],
                         total_samples: int,
                         rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Low-pass to high-pass opening (BRILLIANT).
    """
    morph_start = phases[config['morph_phase']]['start_sample']

    lp_start = config['lp_cutoff_start']
    hp_end = config['hp_cutoff_end']

    cutoff = np.zeros(total_samples)
    cutoff[:morph_start] = lp_start
    cutoff[morph_start:] = np.exp(
        np.linspace(np.log(lp_start), np.log(hp_end), total_samples - morph_start)
    )

    resonance = np.full(total_samples, config['resonance_base'])

    return {
        'cutoff': cutoff,
        'resonance': resonance,
        'type': 'lowpass->highpass'
    }


def _filter_bp_sweep(config: Dict[str, Any],
                    phases: Dict[str, Any],
                    section_context: Dict[str, Any],
                    total_samples: int,
                    rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Band-pass sweep (TIME_PRESSURE).
    """
    sweep_start = phases[config['sweep_phase']]['start_sample']
    sweep_end = phases[config['sweep_phase']]['end_sample']

    bp_start = config['bp_center_start']
    bp_end = config['bp_center_end']

    cutoff = np.full(total_samples, bp_start)

    # Accelerating sweep (quadratic)
    if config.get('sweep_rate_modulation') == 'accelerating':
        t = np.linspace(0, 1, sweep_end - sweep_start)
        sweep_curve = bp_start + (bp_end - bp_start) * (t ** 2)
        cutoff[sweep_start:sweep_end] = sweep_curve

    cutoff[sweep_end:] = bp_end

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
    Rhythmic filter gating (TACTICAL_SEQUENCE).
    """
    lp_base = config['lp_cutoff_base']
    lp_pulse = lp_base * config['lp_cutoff_pulse_mult']
    pulse_hz = config['pulse_rate_hz']
    duty_cycle = config['pulse_duty_cycle']

    # Generate pulse train
    t = np.arange(total_samples) / sample_rate
    pulse_phase = (t * pulse_hz) % 1.0
    pulse_train = (pulse_phase < duty_cycle).astype(float)

    # Cutoff follows pulse
    cutoff = lp_base + (lp_pulse - lp_base) * pulse_train
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
    Generate amplitude envelope.

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
    Sudden attack with short tail (BLUNDER, TIME_PRESSURE).
    """
    entropy = rng.random()  # Use RNG for attack variation

    # Attack
    attack_ms = config['attack_ms_base'] + (entropy * config['attack_ms_entropy_scale'])
    attack_samples = int(attack_ms * sample_rate / 1000)

    # Sustain
    sustain_samples = int(config['sustain_phase_ratio'] * total_samples)

    # Decay
    decay_coeff = config['decay_coefficient']
    decay_samples = total_samples - attack_samples - sustain_samples

    envelope = np.zeros(total_samples)

    # Attack ramp
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Sustain
    envelope[attack_samples:attack_samples + sustain_samples] = 1.0

    # Decay (exponential)
    envelope[attack_samples + sustain_samples:] = np.exp(
        np.linspace(0, decay_coeff, decay_samples)
    )

    return envelope


def _envelope_gradual_sustained(config: Dict[str, Any],
                               phases: Dict[str, Any],
                               total_samples: int,
                               rng: np.random.Generator,
                               sample_rate: int = 88200) -> np.ndarray:
    """
    Gradual attack with sustained plateau (BRILLIANT).
    """
    attack_samples = int(config['attack_ms'] * sample_rate / 1000)
    sustain_samples = int(config['sustain_phase_ratio'] * total_samples)
    decay_samples = total_samples - attack_samples - sustain_samples

    envelope = np.zeros(total_samples)

    # Gradual attack (s-curve)
    t = np.linspace(0, 1, attack_samples)
    envelope[:attack_samples] = 0.5 * (1 + np.tanh(4 * (t - 0.5)))

    # Sustain
    envelope[attack_samples:attack_samples + sustain_samples] = 1.0

    # Linear decay
    envelope[attack_samples + sustain_samples:] = np.linspace(1, 0, decay_samples)

    return envelope


def _envelope_gated_pulse(config: Dict[str, Any],
                         phases: Dict[str, Any],
                         total_samples: int,
                         rng: np.random.Generator,
                         sample_rate: int = 88200) -> np.ndarray:
    """
    Gated pulses (TACTICAL_SEQUENCE).
    """
    attack_samples = int(config['attack_ms'] * sample_rate / 1000)
    gate_samples = int(config['gate_duration_ms'] * sample_rate / 1000)
    release_samples = int(config['release_ms'] * sample_rate / 1000)
    pulse_hz = config['pulse_rate_hz']

    # Generate pulse train
    t = np.arange(total_samples) / sample_rate
    pulse_phase = (t * pulse_hz) % 1.0

    envelope = np.zeros(total_samples)

    # Simple gate (on/off)
    gate_on = pulse_phase < 0.5
    envelope[gate_on] = 1.0

    # TODO: Add proper attack/release slopes

    return envelope


def generate_texture_curve(config: Dict[str, Any],
                           phases: Dict[str, Any],
                           section_context: Dict[str, Any],
                           total_samples: int,
                           rng: np.random.Generator) -> Dict[str, Any]:
    """
    Generate texture parameters (noise, shimmer, etc.).

    Returns:
        Dict with keys: 'noise_ratio', 'noise_type', 'shimmer_enable', etc.
    """
    entropy = section_context.get('entropy', 0.5)

    noise_ratio = config['noise_ratio_base'] + (entropy * config['noise_ratio_entropy_scale'])
    noise_ratio = np.clip(noise_ratio, 0.0, 1.0)

    texture = {
        'noise_ratio': noise_ratio,
        'noise_type': config['noise_type'],
        'shimmer_enable': config.get('shimmer_enable', False)
    }

    if texture['shimmer_enable']:
        texture['shimmer_rate_hz'] = config.get('shimmer_rate_hz', 6.0)

    return texture
```

---

### 4. Synthesizer (`synthesizer.py`)

```python
"""
Gesture audio synthesis engine.

Renders parameter curves to audio using:
- Multi-voice oscillators with time-varying pitch
- Time-varying filter
- Amplitude envelope
- Noise texture
"""

import numpy as np
from typing import List, Dict, Any
from scipy import signal


class GestureSynthesizer:
    """
    Shared synthesis engine for all gesture archetypes.
    """

    def __init__(self):
        pass

    def synthesize(self,
                  pitch_voices: List[np.ndarray],
                  filter_curve: Dict[str, np.ndarray],
                  envelope: np.ndarray,
                  texture_curve: Dict[str, Any],
                  sample_rate: int) -> np.ndarray:
        """
        Synthesize gesture audio.

        Args:
            pitch_voices: List of pitch curves (Hz) for each voice
            filter_curve: Dict with 'cutoff', 'resonance', 'type'
            envelope: Amplitude envelope curve (0-1)
            texture_curve: Dict with noise parameters
            sample_rate: Audio sample rate

        Returns:
            Mono audio buffer (numpy array)
        """
        total_samples = len(envelope)
        audio = np.zeros(total_samples)

        # Synthesize each voice
        for pitch_curve in pitch_voices:
            voice_audio = self._generate_oscillator(pitch_curve, sample_rate)
            audio += voice_audio

        # Normalize multi-voice sum
        audio /= np.sqrt(len(pitch_voices))

        # Add noise texture
        if texture_curve['noise_ratio'] > 0:
            noise = self._generate_noise(
                total_samples,
                texture_curve['noise_type'],
                sample_rate
            )
            audio = (1 - texture_curve['noise_ratio']) * audio + \
                    texture_curve['noise_ratio'] * noise

        # Apply time-varying filter
        audio = self._apply_timevarying_filter(audio, filter_curve, sample_rate)

        # Apply envelope
        audio *= envelope

        # Optional shimmer effect
        if texture_curve.get('shimmer_enable', False):
            audio = self._apply_shimmer(audio, texture_curve['shimmer_rate_hz'], sample_rate)

        return audio

    def _generate_oscillator(self,
                            pitch_curve: np.ndarray,
                            sample_rate: int) -> np.ndarray:
        """
        Generate oscillator with time-varying pitch.

        Uses phase accumulation for smooth frequency modulation.
        """
        # Accumulate phase (integral of frequency)
        phase = np.cumsum(pitch_curve / sample_rate)

        # Generate sine wave
        # TODO: Add waveform selection (saw, square, etc.)
        audio = np.sin(2 * np.pi * phase)

        return audio

    def _generate_noise(self,
                       num_samples: int,
                       noise_type: str,
                       sample_rate: int) -> np.ndarray:
        """
        Generate noise signal.
        """
        if noise_type == 'white':
            return np.random.randn(num_samples)

        elif noise_type == 'pink':
            # Simple pink noise approximation (1/f spectrum)
            white = np.random.randn(num_samples)
            # Low-pass filter white noise
            b, a = signal.butter(1, 0.1)
            pink = signal.filtfilt(b, a, white)
            return pink / np.std(pink)  # Normalize

        else:
            return np.zeros(num_samples)

    def _apply_timevarying_filter(self,
                                  audio: np.ndarray,
                                  filter_curve: Dict[str, np.ndarray],
                                  sample_rate: int) -> np.ndarray:
        """
        Apply time-varying filter.

        WARNING: Original doc noted this is CPU-intensive.
        TODO: Optimize with block-based processing or scipy.signal.lfilter
        """
        # Simple block-based approach (divide into 100ms blocks)
        block_size = int(0.1 * sample_rate)
        filtered = np.zeros_like(audio)

        for i in range(0, len(audio), block_size):
            end = min(i + block_size, len(audio))

            # Get filter params at block center
            block_center = (i + end) // 2
            cutoff = filter_curve['cutoff'][block_center]
            resonance = filter_curve['resonance'][block_center]

            # Design filter (normalized cutoff)
            nyquist = sample_rate / 2
            cutoff_norm = np.clip(cutoff / nyquist, 0.01, 0.99)

            # TODO: Handle filter type switching (BP/LP/HP)
            b, a = signal.butter(2, cutoff_norm, btype='low')

            # Apply to block
            filtered[i:end] = signal.filtfilt(b, a, audio[i:end])

        return filtered

    def _apply_shimmer(self,
                      audio: np.ndarray,
                      shimmer_rate_hz: float,
                      sample_rate: int) -> np.ndarray:
        """
        Apply shimmer effect (amplitude modulation).
        """
        t = np.arange(len(audio)) / sample_rate
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * shimmer_rate_hz * t)
        return audio * lfo
```

---

### 5. Utilities (`utils.py`)

```python
"""
Utility functions for gesture generation.
"""

import numpy as np
from typing import Dict, Any


def compute_phases(phase_config: Dict[str, float],
                  total_samples: int,
                  section_context: Dict[str, Any],
                  rng: np.random.Generator) -> Dict[str, Dict[str, int]]:
    """
    Compute phase timeline with sample boundaries.

    Args:
        phase_config: Phase ratio dict (e.g., {'pre_shadow': 0.15, ...})
        total_samples: Total gesture duration in samples
        section_context: Section-level parameters (for randomization)
        rng: Random generator

    Returns:
        Dict with phase names and their start/end samples
    """
    phases = {}
    current_sample = 0

    for phase_name, ratio in phase_config.items():
        # Add small randomization (±5%)
        randomized_ratio = ratio * rng.uniform(0.95, 1.05)
        phase_samples = int(randomized_ratio * total_samples)

        phases[phase_name] = {
            'start_sample': current_sample,
            'end_sample': current_sample + phase_samples,
            'duration_samples': phase_samples
        }

        current_sample += phase_samples

    # Adjust last phase to fill remaining samples
    last_phase = list(phases.keys())[-1]
    phases[last_phase]['end_sample'] = total_samples
    phases[last_phase]['duration_samples'] = total_samples - phases[last_phase]['start_sample']

    return phases


def finalize_audio(audio: np.ndarray,
                  peak_limit: float,
                  rms_target: float) -> np.ndarray:
    """
    Finalize audio with normalization and safety clipping.

    Args:
        audio: Raw audio buffer
        peak_limit: Maximum peak level (e.g., 0.8)
        rms_target: Target RMS level in dB (e.g., -18.0)

    Returns:
        Finalized audio buffer
    """
    # Soft clip to prevent harsh distortion
    audio = soft_clip(audio, threshold=peak_limit)

    # RMS normalization
    current_rms = np.sqrt(np.mean(audio ** 2))
    target_rms_linear = 10 ** (rms_target / 20)

    if current_rms > 0:
        audio *= (target_rms_linear / current_rms)

    # Hard clip as final safety
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def soft_clip(audio: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Apply soft clipping (tanh-based).

    Args:
        audio: Input audio
        threshold: Clipping threshold (0-1)

    Returns:
        Soft-clipped audio
    """
    return threshold * np.tanh(audio / threshold)
```

---

### 6. Coordinator (`coordinator.py`)

```python
"""
Gesture coordinator - manages archetype registry and moment dispatching.

Parallel to PatternCoordinator in synth_composer.
"""

import numpy as np
from typing import Dict, Any
from .base import GestureGenerator
from .archetype_configs import ARCHETYPES


class GestureCoordinator:
    """
    Coordinates gesture generation and archetype registry.

    Usage:
        coordinator = GestureCoordinator(rng)
        audio = coordinator.generate_gesture('BLUNDER', moment_event, section_context, sample_rate)
    """

    def __init__(self, rng: np.random.Generator):
        """
        Initialize coordinator.

        Args:
            rng: NumPy random generator for reproducible randomness
        """
        self.rng = rng
        self.gestures = self._build_gesture_registry()

    def _build_gesture_registry(self) -> Dict[str, GestureGenerator]:
        """
        Build gesture registry mapping event types to generators.
        """
        registry = {}

        for archetype_name, archetype_config in ARCHETYPES.items():
            registry[archetype_name] = GestureGenerator(archetype_config, self.rng)

        return registry

    def get_available_archetypes(self) -> list:
        """
        Get list of available archetype names.
        """
        return list(self.gestures.keys())

    def generate_gesture(self,
                        archetype_name: str,
                        moment_event: Dict[str, Any],
                        section_context: Dict[str, Any],
                        sample_rate: int) -> np.ndarray:
        """
        Generate gesture audio for a moment event.

        Args:
            archetype_name: Archetype name (e.g., 'BLUNDER', 'BRILLIANT')
            moment_event: Moment metadata (event_type, timestamp, move_number)
            section_context: Section-level parameters (tension, entropy, scale)
            sample_rate: Audio sample rate

        Returns:
            Mono audio buffer (numpy array)

        Raises:
            ValueError: If archetype is unknown
        """
        if archetype_name not in self.gestures:
            raise ValueError(
                f"Unknown archetype: {archetype_name}. "
                f"Available: {list(self.gestures.keys())}"
            )

        generator = self.gestures[archetype_name]
        return generator.generate_gesture(moment_event, section_context, sample_rate)
```

---

## Integration Example

```python
# In Layer 3b renderer

from layer3b.coordinator import GestureCoordinator

# Initialize coordinator
rng = np.random.default_rng(seed=42)
gesture_coordinator = GestureCoordinator(rng)

# Load tagged moments from file
with open('section_tags.json') as f:
    tags = json.load(f)

# Generate gestures for each moment
for section in tags['sections']:
    section_context = {
        'tension': section['tension'],
        'entropy': section['entropy'],
        'scale': section['scale']
    }

    for moment in section['moments']:
        # Generate gesture audio
        gesture_audio = gesture_coordinator.generate_gesture(
            archetype_name=moment['event_type'],  # 'BLUNDER', 'BRILLIANT', etc.
            moment_event=moment,
            section_context=section_context,
            sample_rate=88200  # Match project default (2x 44.1kHz for anti-aliasing)
        )

        # Place in timeline
        timestamp = moment['timestamp']
        # ... add to mix buffer at timestamp
```

---

## Benefits of This Architecture

1. **Unified codebase**: All archetypes share synthesis engine (no duplication)
2. **Easy to extend**: Add archetype = add config dict (no new class file)
3. **Testable**: Each curve generator is a pure function
4. **Debuggable**: Can inspect/plot curves independently
5. **Consistent with existing patterns**: Mirrors `synth_composer/` structure
6. **Modular**: Swap curve generators without touching synthesis
7. **Configuration-driven**: Non-programmers can tweak archetypes via JSON

---

## Open Questions / TODOs

1. **Sample rate**: ✅ **RESOLVED** - All functions now properly use `sample_rate` parameter (default: 88200 Hz = 2x 44.1kHz for anti-aliasing per synth_config.py)
2. **Filter optimization**: Block-based filtering is simple but crude. Consider `scipy.signal.sosfilt` or pre-computed filter banks.
3. **Cellular automaton**: `_pitch_cellular_sequence` is stub. Needs proper CA implementation.
4. **Waveform selection**: Currently only sine. Add saw/square/triangle options.
5. **Stereo width**: Original doc mentions width curves, not implemented yet.
6. **Performance**: Python loops in curve generators could be vectorized.

---

## Next Steps

1. Implement core files (base.py, archetype_configs.py, curve_generators.py, synthesizer.py, utils.py, coordinator.py)
2. Write unit tests for each curve generator
3. Create test harness to render individual gestures (for auditioning)
4. Integrate with existing Layer 3b renderer
5. Profile and optimize bottlenecks
6. Add remaining archetypes from `emotional_gesture_generator.md`
