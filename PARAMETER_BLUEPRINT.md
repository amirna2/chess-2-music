# COMPLETE PARAMETER BLUEPRINT
## All parameters used to compose tagger output into music

---

## SECTION 1: MUSICAL PRIMITIVES (synth_config.py)

### 1.1 SCALES
- **SCALES['minor']**: 8 frequencies (110-220Hz)
- **SCALES['phrygian']**: 8 frequencies
- **SCALES['dorian']**: 8 frequencies
**Usage**: Selected by `NARRATIVE_BASE_PARAMS[narrative]['scale']`

### 1.2 ENVELOPES (ADSR)
- **14 envelope presets**: percussive, pluck, pad, stab, drone, laser, doom, short_pluck, medium_stab, gentle, soft, brief, quick, standard, sustained
**Usage**: Referenced by name in MOMENT_VOICES, layer generation

### 1.3 FILTER_ENVELOPES
- **8 filter envelope presets**: gentle, sweep, minimal, dramatic, closing, standard, smooth, sharp
**Usage**: Referenced by name in MOMENT_VOICES

---

## SECTION 2: LAYER 1 - OVERALL NARRATIVE (Base Drone)

### 2.1 NARRATIVE_BASE_PARAMS
**Keys**: TUMBLING_DEFEAT, FIGHTING_DEFEAT, ATTACKING_MASTERPIECE, TACTICAL_MASTERPIECE, PEACEFUL_DRAW, QUIET_PRECISION, DEATH_SPIRAL, DEFAULT

**Per narrative**:
- `base_waveform`: saw/triangle/pulse/supersaw
- `filter_start`: Hz (start of game)
- `filter_end`: Hz (end of game)
- `resonance_start`: 0.1-4.0
- `resonance_end`: 0.1-4.0
- `tempo_start`: multiplier
- `tempo_end`: multiplier
- `detune_start`: cents
- `detune_end`: cents
- `scale`: minor/phrygian/dorian
- `drone_voices`: number (optional, default 3)

**Used by**: `synth_composer.py:237-239` (`_get_narrative_base_params()`)
**Interpolated**: Linear interpolation based on game progress in `interpolate_base_params()`

---

## SECTION 3: LAYER 2 - SECTION NARRATIVE (Pattern Modulation)

### 3.1 SECTION_MODULATIONS
**Keys**: DESPERATE_DEFENSE, KING_HUNT, MATING_ATTACK, TACTICAL_CHAOS, TACTICAL_BATTLE, QUIET, POSITIONAL, SACRIFICIAL_ATTACK, CRUSHING_ATTACK, ENDGAME_PRECISION, COMPLEX_STRUGGLE, FLAWLESS_CONVERSION, DECISIVE_ENDING, DEFAULT

**Per section narrative**:
- `filter_mult`: multiplier on base filter
- `resonance_add`: added to base resonance
- `tempo_mult`: multiplier on tempo
- `note_density`: multiplier on note count
- `filter_env_amount`: Hz for filter envelope modulation

**Used by**: `synth_composer.py:253-261` (`_get_section_modulation()`)
**Modulated**: By tension value

---

## SECTION 4: MELODIC PATTERNS

### 4.1 MELODIC_PATTERNS
**Keys**: DEFEAT_HIGH_TENSION, DEFEAT_LOW_TENSION, MASTERPIECE_HIGH_TENSION, MASTERPIECE_LOW_TENSION, NEUTRAL_HIGH_TENSION, NEUTRAL_MEDIUM_TENSION, NEUTRAL_LOW_TENSION

**Per pattern**:
- `indices`: list of scale indices [0-7]
- `octave_up_mod`: which notes octave up (modulo)
- `octave_down_mod`: which notes octave down (modulo)

**Used by**: Pattern generation based on overall_narrative + tension

---

## SECTION 5: SEQUENCER (Heartbeat & Continuous)

### 5.1 SEQUENCER_PATTERNS
**Keys**: DEVELOPMENT, ASYMMETRY, CRITICAL_SWING, GAME_CHANGING, TACTICAL_SEQUENCE, KING_ATTACK, BLUNDER, MISTAKE, INACCURACY, FIRST_EXCHANGE, MATE_SEQUENCE, BRILLIANT, TIME_PRESSURE, TIME_SCRAMBLE, SIGNIFICANT_SHIFT, PULSE

**Per pattern**: List of 16 semitone offsets (None = rest)

### 5.2 SEQUENCER_SYNTH
- `detune_cents`: list of 6 detune values
- `filter_base_start`: Hz
- `filter_increment_per_step`: Hz per game progression
- `filter_env_amount`: Hz
- `resonance`: 0.1-4.0
- `amp_env`: ADSR tuple
- `filter_env`: ADSR tuple
- `global_filter_base`: Hz
- `global_filter_lfo_amount`: Hz
- `global_filter_sweep_amount`: Hz
- `global_filter_resonance`: 0.1-4.0
- `smoothing_window_sec`: seconds
- **Heartbeat-specific**: filter, resonance, bpm, lub_dub_gap, root_midi, dub_offset, dub_volume, amp_env, filter_env

---

## SECTION 6: KEY MOMENTS (Layer 3b Gestures)

### 6.1 MOMENT_VOICES
**Keys**: BLUNDER_IN_DEFEAT, BLUNDER_IN_MASTERPIECE, BLUNDER_NEUTRAL, MISTAKE_IN_DEFEAT, MISTAKE_IN_MASTERPIECE, MISTAKE_NEUTRAL, BRILLIANT_IN_MASTERPIECE, BRILLIANT_IN_DEFEAT, BRILLIANT_NEUTRAL, STRONG_IN_MASTERPIECE, STRONG_IN_DEFEAT, STRONG_NEUTRAL, TACTICAL_SEQUENCE, MATE_IN_DEFEAT, MATE_IN_MASTERPIECE, DEFAULT_MOMENT, FINAL_RESOLUTION

**Per moment voice**:
- `freq` or `freqs`: Hz
- `duration`: seconds
- `waveform`: saw/triangle/pulse/square
- `filter_base`: Hz
- `filter_env_amount`: Hz
- `resonance`: 0.1-4.0
- `amp_env`: envelope name
- `filter_env`: envelope name (optional)
- `volume`: 0.0-1.0 (optional)
- Special: `overlap_factor`, `total_duration`, `note_duration` for sequences

### 6.2 DEVELOPMENT_PARAMS
**Keys**: IN_DEFEAT, DEFAULT
- `melody_indices`, `note_duration`, `waveform`, `filter_mult`, `base_filter_env`, `filter_env_step`, `resonance`, `amp_env`, `volume`

### 6.3 FIRST_EXCHANGE_PARAMS
**Keys**: IN_DEFEAT, DEFAULT
- `question_indices`, `answer_indices`, `question_waveform`, `answer_waveform`, `note_duration`, `filter_mult`, `question_filter_env_base`, `answer_filter_env_base`, `question_resonance`, `answer_resonance`, `answer_brightness`, `question_volume`, `answer_volume`

### 6.4 MOMENT_EVENT_PARAMS
- `base_duration_sec`: 2.5
- `min_duration_sec`: 1.5
- `max_duration_sec`: 4.0
- `score_duration_mult`: 0.3
- `base_mix_amount`: 0.3
- `max_mix_amount`: 0.9
- `score_mix_mult`: 0.06
- `filter_mod_base`: 500
- `filter_mod_per_score`: 200
- `crossfade_duration_sec`: 0.5
- `base_pattern_level`: 0.4
- `moment_spacing_sec`: 1.0

---

## SECTION 7: PROCESS TRANSFORMATION

### 7.1 PROCESS_PARAMS
**Keys**: TUMBLING_DEFEAT, ATTACKING_MASTERPIECE, QUIET_PRECISION

**Per process**:
- `mistake_weights`: dict of moment_type -> weight
- `base_decay`: float
- `chaos_factor`: float
- `tempo_drift_clamp`: float
- `pitch_drift_multiplier`: float
- `volume_decay_rate`: float
- `brilliance_weights`: dict
- `crescendo_exponent`: float
- `max_momentum`: float
- `tempo_base`, `tempo_range`: float
- `filter_brightness_base`, `filter_brightness_range`: float
- `disturbance`, `balance_decay`, `breathing_increment`, `breathing_amplitude`: float

**Used by**: `synth_narrative.py` - narrative process transformations

---

## SECTION 8: TIMING & MIXING

### 8.1 TIMING
- `section_fade_sec`: 0.1
- `section_gap_sec`: 0.0
- `section_crossfade_sec`: 2.0
- `note_gap_sec`: 0.02
- `phrase_pause_sec`: 0.15
- `chunk_size_samples`: 512
- `filter_chunk_size_samples`: 64
- `sequencer_overlap`: 1.5
- `lfo_frequency`: 0.1
- `sequencer_lfo_frequency`: 0.25

### 8.2 MIXING
- `drone_level`: 0.15
- `pattern_level`: 0.6
- `sequencer_level`: 0.4
- `moment_level`: 0.5
- `section_level`: 0.7
- `master_limiter`: 0.9
- `sidechain_amount`: 0.3
- `supersaw_compression`: 0.8
- `supersaw_gain`: 1.25
- `filtered_sequence_level`: 0.04
- `ducking_amount`: 0.3
- `soft_clip_pre`: 0.9
- `soft_clip_post`: 0.95

### 8.3 LAYER_MIXING
- `drone_in_supersaw`: 0.3
- `pattern_in_supersaw`: 0.7
- `pattern_note_level`: 0.2
- `sequencer_note_level`: 0.8
- `lfo_modulation_depth`: 0.1

### 8.4 LAYER_ENABLE
- `drone`: True/False
- `patterns`: True/False
- `sequencer`: True/False
- `moments`: True/False

---

## SECTION 9: STEREO & SPATIAL

### 9.1 STEREO_CONFIG
- `white_pan`: -0.7
- `black_pan`: 0.7
- `drone_pan`: 0.0
- `min_width`: 0.0
- `max_width`: 0.8
- `entropy_pan_amount`: 0.6

---

## SECTION 10: ENTROPY (Laurie Spiegel-inspired)

### 10.1 ENTROPY_CONFIG
- `weights`: eval=0.5, tactical=0.4, time=0.1
- `eval_window`: 5
- `tactical_window`: 5
- `smoothing_sigma`: 2.0
- `low_threshold`: 0.3
- `high_threshold`: 0.7
- `note_pools`: low=[0,4], medium=[0,2,4,5,7], high=[0,1,2,3,4,5,6,7]
- `rhythm_variation_max`: 0.5
- `filter_lfo_range`: (0.02, 0.12)
- `glide_reduction_max`: 0.5
- `harmony_probability_threshold`: 0.7

---

## SECTION 11: SYNTHESIS DETAILS

### 11.1 SUPERSAW_DETUNE
**Keys**: tight, standard, wide, laser
- Lists of 6 detune values in cents

### 11.2 BASE_NOTE_DURATION
- 0.5 seconds

### 11.3 DEFAULT_BPM
- 120

### 11.4 SAMPLE_RATE
- 88200 Hz

### 11.5 WAV_OUTPUT
- `sample_width`: 2
- `channels`: 2
- `amplitude_multiplier`: 30000
- `clamp_min`: -32000
- `clamp_max`: 32000
- `normalization_threshold`: 0.9

---

## SECTION 12: STYLE PROFILES (NEW)

### 12.1 STYLE_PROFILES['spiegel']
- `description`: string
- `mixing`: {drone_level, pattern_level, sequencer_level, moment_level}
- `layer1_drone`: {base_waveform, filter_range, resonance_range, tempo_range, detune_range, scale, drone_voices}
- `layer2_pattern`: {pattern_type, density, amp_env, filter_base_hz, filter_range_hz, resonance, pan_random_width}
- `layer3a_heartbeat`: {base_hz, filter_hz, thump_gain, stereo_width}
- `layer3b_gesture`: {curve_bias, reverb_send_db}
- `fx`: {chorus, delay, reverb, saturation}

---

## TOTAL PARAMETER COUNT

### By Category:
1. Scales: 3 scales × 8 notes = 24 values
2. Envelopes: 14 × 4 params = 56 values
3. Filter Envelopes: 8 × 4 params = 32 values
4. Narrative Base Params: 8 narratives × 9-10 params = ~80 values
5. Section Modulations: 13 sections × 5 params = 65 values
6. Melodic Patterns: 7 patterns × 3-10 params = ~50 values
7. Sequencer Patterns: 16 patterns × 16 values = 256 values
8. Sequencer Synth: ~20 params
9. Moment Voices: 17 voices × 8-10 params = ~150 values
10. Development/First Exchange: 4 configs × 10-13 params = ~50 values
11. Moment Event Params: 11 params
12. Process Params: 3 processes × 5-10 params = ~25 values
13. Timing: 10 params
14. Mixing: 15 params
15. Layer Mixing: 5 params
16. Layer Enable: 4 params
17. Stereo Config: 6 params
18. Entropy Config: ~15 params
19. Supersaw Detune: 4 × 6 = 24 values
20. Style Profiles: ~50 params per style

**ESTIMATED TOTAL: ~900-1000 individual parameter values**

---

## PARAMETER FLOW (How they're used in composition)

```
TAGGER OUTPUT (tags.json)
    ├── overall_narrative → NARRATIVE_BASE_PARAMS → base_params
    ├── sections[]
    │   ├── narrative → SECTION_MODULATIONS → modulation
    │   ├── tension → scales modulation values
    │   ├── key_moments[]
    │   │   ├── type + overall_narrative → MOMENT_VOICES → gesture synthesis
    │   │   └── score → MOMENT_EVENT_PARAMS → duration/mixing
    │   └── start_ply/end_ply → ENTROPY_CONFIG → note_pools/rhythm_variation
    └── eco → RNG seed

COMPOSITION FLOW:
1. base_params (from NARRATIVE_BASE_PARAMS)
2. interpolate based on progress
3. apply section modulation (from SECTION_MODULATIONS)
4. apply process transforms (from PROCESS_PARAMS)
5. select melodic pattern (from MELODIC_PATTERNS)
6. generate drone (using base_params + TIMING + MIXING)
7. generate patterns (using PatternCoordinator + MELODIC_PATTERNS + ENTROPY)
8. generate sequencer/heartbeat (using SEQUENCER_PATTERNS + SEQUENCER_SYNTH)
9. generate gestures (using MOMENT_VOICES + Layer3b archetypes)
10. mix all layers (using MIXING + LAYER_MIXING + STEREO_CONFIG)
11. apply master processing (using WAV_OUTPUT)
```

