# Configuration Service Guide

## Overview

The configuration service provides centralized YAML-based configuration management for the chess-to-music synthesis system. It replaces hardcoded Python configuration with a clean, maintainable YAML blueprint.

## Architecture

```
config.yaml                  # YAML configuration blueprint
    ↓
config_service.py           # Configuration reader and API
    ↓
synth_config.py (deprecated) # Backward compatibility layer
    ↓
Components (synth_composer, layer3b, etc.)
```

## Using the Config Service

### New Code (Recommended)

```python
from config_service import get_config

# Get config instance
config = get_config()

# Access parameters
sample_rate = config.sample_rate()
drone_env = config.amplitude_envelope('drone')
minor_scale = config.scale('minor')
```

### Backward Compatible Code

```python
# Old imports still work
from synth_config import get_envelope, get_narrative_params

# These now delegate to config_service internally
env = get_envelope('percussive')
narrative = get_narrative_params('TUMBLING_DEFEAT')
```

## API Reference

### Basic Access

```python
config = get_config()

# Dot-notation path access
value = config.get('synthesis.sample_rate')
value = config.get('composition.harmony.root_frequency')

# Get entire section
synthesis_config = config.get_section('synthesis')
```

### Synthesis Parameters

```python
# Core synthesis
sr = config.sample_rate()                    # 88200
waveforms = config.waveforms()               # ['sine', 'triangle', ...]
detune = config.supersaw_detune('standard')  # [-7.0, -3.5, ...]
```

### Envelopes

```python
# Amplitude envelopes (ADSR tuples)
env = config.amplitude_envelope('drone')     # (0.5, 0.0, 1.0, 0.5)
env = config.amplitude_envelope('percussive') # (0.001, 0.02, 0.3, 0.05)

# Filter envelopes
filt = config.filter_envelope('sweep')       # (0.001, 0.2, 0.5, 0.1)
filt = config.filter_envelope('dramatic')    # (0.01, 0.25, 0.4, 0.4)
```

### Harmony & Scales

```python
# Root frequency
root = config.root_frequency()               # 55.0

# Scale frequencies
minor = config.scale('minor')                # [110.0, 123.47, ...]
phrygian = config.scale('phrygian')

# Scale intervals (semitones)
intervals = config.scale_intervals('minor')  # [0, 2, 3, 5, 7, 8, 10, 12]
```

### Narrative & Sections

```python
# Macro game arc (entire game)
arc = config.macro_arc('TUMBLING_DEFEAT')
# Returns: {
#   'harmonic': {'scale': 'phrygian'},
#   'spectral': {'waveform': 'supersaw', 'filter': {...}},
#   'temporal': {'tempo': {'start': 1.0, 'end': 0.7}},
#   ...
# }

# Section modulation (30s-2min sections)
mod = config.section_modulation('KING_HUNT')
# Returns: {
#   'filter_mult': 1.3,
#   'resonance_add': 1.0,
#   'tempo_mult': 1.2,
#   ...
# }
```

### Gesture Events

```python
# Context-aware gesture parameters
blunder = config.gesture_event('BLUNDER', 'in_defeat')
# Returns: {
#   'pitch': 55,
#   'duration': 1.0,
#   'waveform': 'saw',
#   'filter_base': 200,
#   ...
# }

# Different context, different sound
blunder_mp = config.gesture_event('BLUNDER', 'in_masterpiece')
# Returns: {
#   'pitch': 110,
#   'duration': 0.3,
#   'waveform': 'pulse',
#   ...
# }
```

### Sequencer Patterns

```python
# Get pattern
dev_pattern = config.sequencer_pattern('DEVELOPMENT')
# Returns: {'early': [0, 0, 7, ...], 'mid': [...], 'full': [...]}

blunder_pattern = config.sequencer_pattern('BLUNDER')
# Returns: [0, -1, -3, -6, ...]  (semitone offsets, None = rest)

# Synth parameters for sequencer
synth_params = config.sequencer_synth_params()
# Returns: {
#   'detune_cents': [-15, -9, ...],
#   'filter_base_start': 800,
#   'resonance': 1.2,
#   ...
# }

# Heartbeat-specific params
hb_params = config.heartbeat_params()
# Returns: {
#   'filter': 220,
#   'bpm': 70,
#   'lub_dub_gap': 0.080,
#   ...
# }
```

### Mixing & Levels

```python
# Layer levels
drone_level = config.layer_level('drone')        # 0.15
pattern_level = config.layer_level('patterns')   # 0.6

# Layer enable/disable
enabled = config.layer_enabled('drone')          # True

# Stereo configuration
stereo = config.stereo_config()
# Returns: {
#   'white_pan': -0.7,
#   'black_pan': 0.7,
#   'drone_pan': 0.0,
#   ...
# }
```

### Process Transformation

```python
# Process parameters for narrative evolution
process = config.process_params('TUMBLING_DEFEAT')
# Returns: {
#   'mistake_weights': {'INACCURACY': 0.05, ...},
#   'base_decay': 0.3,
#   'chaos_factor': 0.02,
#   ...
# }
```

### Entropy

```python
# Entropy calculation and mapping
entropy = config.entropy_config()
# Returns: {
#   'calculation': {'weights': {...}, 'windows': {...}},
#   'thresholds': {'low': 0.3, 'high': 0.7},
#   'musical_mapping': {...},
#   ...
# }
```

## Configuration File Structure

The `config.yaml` file is organized by signal flow and temporal hierarchy:

```yaml
# 1. SYNTHESIS (signal chain: oscillators → filters → envelopes)
synthesis:
  sample_rate: 88200
  oscillators:
    waveforms: [...]
    supersaw: {...}
  filters: {...}
  amplitude: {...}

# 2. COMPOSITION (temporal hierarchy: macro → meso → micro)
composition:
  harmony: {...}
  rhythm: {...}
  macro_game_arc: {...}      # Entire game (5-60 min)
  meso_section_modulation: {...}  # Sections (30s-2min)
  melodic_patterns: {...}
  micro_gestures: {...}      # Events (0.1-3s)
  sequencer: {...}
  gesture_events: {...}

# 3. MIXING & MASTERING
mixing:
  layers: {...}
  layer_enable: {...}
  dynamics: {...}
  stereo: {...}

# 4. PROCESS TRANSFORMATION (narrative evolution)
process_transformation: {...}

# 5. ENTROPY (complexity-driven modulation)
entropy: {...}
```

## Migration Guide

### From synth_config.py to config_service

**Old way:**
```python
from synth_config import DEFAULT_CONFIG

sample_rate = DEFAULT_CONFIG.SAMPLE_RATE
scale = DEFAULT_CONFIG.SCALES['minor']
env = DEFAULT_CONFIG.ENVELOPES['drone']
```

**New way:**
```python
from config_service import get_config

config = get_config()
sample_rate = config.sample_rate()
scale = config.scale('minor')
env = config.amplitude_envelope('drone')
```

### Backward Compatibility

Old helper functions still work:
```python
from synth_config import (
    get_envelope,
    get_filter_envelope,
    get_narrative_params,
    get_section_modulation
)

# These now use config_service internally
env = get_envelope('drone')
narrative = get_narrative_params('TUMBLING_DEFEAT')
```

## Testing

Run the test script to validate configuration:

```bash
python3 test_config.py
```

This tests:
- YAML loading
- All accessor methods
- Backward compatibility
- Dot-notation access

## Benefits

1. **Separation of Concerns**: Configuration separate from code
2. **Maintainability**: Edit YAML without touching Python
3. **Readability**: Clear hierarchical structure
4. **Type Safety**: Accessor methods with proper return types
5. **Backward Compatible**: Existing code continues to work
6. **Extensibility**: Easy to add new parameters
7. **Documentation**: YAML serves as living documentation

## Files

- `config.yaml` - YAML configuration blueprint
- `config_service.py` - Configuration reader and API
- `synth_config.py` - Backward compatibility layer (deprecated)
- `test_config.py` - Validation test script
- `CONFIG_SERVICE_GUIDE.md` - This guide

## Next Steps

To migrate a component to use the new config service:

1. Import `get_config()` instead of `DEFAULT_CONFIG`
2. Replace dictionary access with method calls
3. Update parameter names to match YAML structure
4. Test thoroughly

Example:
```python
# Before
from synth_config import DEFAULT_CONFIG
sr = DEFAULT_CONFIG.SAMPLE_RATE

# After
from config_service import get_config
config = get_config()
sr = config.sample_rate()
```
