# Style Renderer Implementation Plan

## Overview

This document outlines the step-by-step implementation plan for the Style Renderer architecture, adding explicit style control to the chess-to-music synthesis system.

## Goals

1. ✅ Decouple aesthetic decisions from synthesis implementation
2. ✅ Enable multiple musical styles from same chess data
3. ✅ Maintain backward compatibility with current output
4. ✅ Create modular, extensible architecture
5. ✅ Implement two distinct styles: Spiegel (1979) and Jarre (1978)

## Implementation Strategy

**Recommendation: Start with Spiegel**

**Rationale**:
- Current system is ~70% Spiegel-style already
- Validates architecture with minimal new code
- Provides regression test (output should match current)
- Establishes patterns for Jarre implementation

## Phase 1: Core Infrastructure (Foundation)

**Goal**: Create base architecture without changing existing functionality

### 1.1 Create Style Renderer Module Structure

```bash
chess-2-music/
├── synth_renderer/              # NEW module
│   ├── __init__.py
│   ├── base.py                  # StyleRenderer abstract class
│   ├── factory.py               # StyleRendererFactory
│   ├── spiegel.py               # SpiegelRenderer (Phase 2)
│   ├── jarre.py                 # JarreRenderer (Phase 3)
│   └── generators/              # Style-specific generators
│       ├── __init__.py
│       ├── drums.py             # Drum synthesis (Phase 3)
│       ├── arpeggios.py         # Arpeggio patterns (Phase 3)
│       ├── pads.py              # Pad chords (Phase 3)
│       └── sequenced_bass.py    # Grid-locked bass (Phase 3)
```

**Tasks**:
- [ ] Create `synth_renderer/` directory
- [ ] Create `synth_renderer/__init__.py` with exports
- [ ] Create `synth_renderer/base.py` with `StyleRenderer` abstract class
- [ ] Create `synth_renderer/factory.py` with `StyleRendererFactory`
- [ ] Create `synth_renderer/generators/` directory structure

**Files to Create**:
1. `synth_renderer/base.py` - Abstract `StyleRenderer` class
2. `synth_renderer/factory.py` - Factory pattern for renderer creation

**Deliverable**: Empty infrastructure, no functional changes yet

---

### 1.2 Add Style Configuration to synth_config.py

**Tasks**:
- [ ] Add `STYLE_PROFILES` dictionary to `SynthConfig`
- [ ] Define Spiegel profile parameters
- [ ] Define Jarre profile parameters (placeholder)

**Changes to `synth_config.py`**:

```python
# Add to SynthConfig class
STYLE_PROFILES: Dict[str, Dict] = field(default_factory=lambda: {
    'spiegel': {
        'description': 'Laurie Spiegel - Evolving drones, sparse events, algorithmic',
        'note_density_multiplier': 0.2,
        'rhythm_presence': 'heartbeat_only',
        'bass_type': 'drone',
        'melody_continuity': 'sparse',
        'gesture_prominence': 'high',
        'layers': ['drone', 'sparse_patterns', 'heartbeat', 'gestures'],
        'mixing': {
            'drone_level': 0.20,
            'pattern_level': 0.50,
            'heartbeat_level': 0.30,
            'gesture_level': 0.60,
        },
    },
    'jarre': {
        'description': 'Jean-Michel Jarre - Sequenced, rhythmic, layered',
        'note_density_multiplier': 5.0,
        'rhythm_presence': 'full_drums',
        'bass_type': 'sequenced',
        'melody_continuity': 'continuous',
        'gesture_prominence': 'low',
        'layers': ['bass_seq', 'drums', 'arpeggios', 'pads', 'melody', 'gestures'],
        'mixing': {
            'bass_level': 0.35,
            'drum_level': 0.40,
            'arpeggio_level': 0.40,
            'pad_level': 0.30,
            'melody_level': 0.55,
            'gesture_level': 0.25,
        },
    },
})
```

**Deliverable**: Configuration profiles defined, not used yet

---

### 1.3 Refactor ChessSynthComposer for Renderer Support

**Goal**: Prepare existing composer to delegate to renderers

**Tasks**:
- [ ] Add `style` parameter to `ChessSynthComposer.__init__()`
- [ ] Integrate `StyleRendererFactory` in constructor
- [ ] Extract current composition logic into methods (prepare for delegation)
- [ ] Maintain backward compatibility (default to current behavior)

**Changes to `synth_composer.py`**:

```python
# Add import
from synth_renderer import StyleRendererFactory

class ChessSynthComposer:
    def __init__(self, chess_tags, config=None, style='spiegel'):
        """
        Initialize chess music composer with style renderer.

        Args:
            chess_tags: Dictionary with game data and narratives
            config: Optional SynthConfig instance
            style: Style name ('spiegel' or 'jarre') - default 'spiegel'
        """
        self.tags = chess_tags
        self.config = config or SynthConfig()
        self.sample_rate = self.config.SAMPLE_RATE

        # Create RNG for reproducibility
        self.rng = self._create_rng_from_eco(chess_tags.get('eco', 'A00'))

        # Create synth engines (existing code)
        self.drone_synth = SubtractiveSynth(self.sample_rate, self.rng)
        self.pattern_synth = SubtractiveSynth(self.sample_rate, self.rng)
        # ... etc

        # NEW: Create style renderer
        self.style = style
        self.renderer = StyleRendererFactory.create(
            style=style,
            config=self.config,
            synth_engines={
                'drone': self.drone_synth,
                'pattern': self.pattern_synth,
                'gesture': self.gesture_synth,
            },
            pattern_coordinator=self.pattern_coordinator,
            gesture_coordinator=self.gesture_coordinator,
            rng=self.rng,
        )

    # Keep existing compose_section() for Phase 1
    # Will delegate to renderer in Phase 2
```

**Deliverable**: Infrastructure in place, still uses existing code paths

---

## Phase 2: Spiegel Renderer (Validation)

**Goal**: Implement Spiegel renderer using existing components, validate architecture

### 2.1 Implement SpiegelRenderer

**Tasks**:
- [ ] Create `synth_renderer/spiegel.py`
- [ ] Implement `SpiegelRenderer` class extending `StyleRenderer`
- [ ] Map existing drone/pattern/gesture logic to renderer methods
- [ ] Implement parameter translation methods

**File**: `synth_renderer/spiegel.py`

**Key Methods**:
```python
class SpiegelRenderer(StyleRenderer):
    def render_section(self, section, context):
        """Main entry point - orchestrates layer generation"""
        pass

    def get_drone_params(self, narrative, tension):
        """Translate narrative → drone synthesis parameters"""
        pass

    def get_pattern_params(self, narrative, tension):
        """Translate narrative → sparse pattern parameters"""
        pass

    def get_heartbeat_params(self, narrative, tension):
        """Translate narrative → heartbeat parameters"""
        pass

    def get_gesture_params(self, section):
        """Translate section → gesture parameters"""
        pass

    def mix_layers(self, layers, profile):
        """Mix layers according to Spiegel profile"""
        pass
```

**Deliverable**: Working SpiegelRenderer that produces same output as current system

---

### 2.2 Integrate SpiegelRenderer into ChessSynthComposer

**Tasks**:
- [ ] Modify `compose_section()` to delegate to renderer
- [ ] Extract common context data for renderer
- [ ] Test that output matches current system

**Changes to `synth_composer.py`**:

```python
def compose_section(self, section, section_index, total_sections):
    """
    Compose one section using style renderer.
    """
    # Build context for renderer
    context = {
        'section_index': section_index,
        'total_sections': total_sections,
        'overall_narrative': self.tags['overall_narrative'],
        'scale': self.scale,
        'scale_name': self.scale_name,
        'sample_rate': self.sample_rate,
        'section_duration': self.calculate_section_duration(section),
    }

    # Delegate to style renderer
    section_audio = self.renderer.render_section(section, context)

    return section_audio
```

**Deliverable**: System produces identical output to current version (regression test passes)

---

### 2.3 Testing & Validation

**Tasks**:
- [ ] Render Ding/Gukesh Game 1 with Spiegel renderer
- [ ] Compare WAV output to current system (should be identical)
- [ ] Validate mix levels, timing, gesture placement
- [ ] Document any intentional differences

**Test Command**:
```bash
./c2m data/ding_gukesh/game1.pgn --style spiegel
# Compare to existing output
diff data/ding_gukesh/game1.wav data/ding_gukesh/game1_spiegel.wav
```

**Success Criteria**:
- Output is perceptually identical to current system
- All layers present and balanced correctly
- Gestures trigger at correct moments
- No regression in quality

**Deliverable**: Validated Spiegel renderer matching current output

---

## Phase 3: Jarre Renderer (New Components)

**Goal**: Implement Jarre renderer with new rhythmic/melodic components

### 3.1 Implement Drum Synthesizer

**Tasks**:
- [ ] Create `synth_renderer/generators/drums.py`
- [ ] Implement `DrumSynthesizer` class
- [ ] Synthesize kick, snare, hihat from scratch
- [ ] Create drum pattern library

**File**: `synth_renderer/generators/drums.py`

**Key Classes**:
```python
class DrumSynthesizer:
    def synthesize_kick(self, duration=0.3):
        """Synthesize kick drum (pitch envelope 150→50 Hz)"""
        pass

    def synthesize_snare(self, duration=0.15):
        """Synthesize snare (noise + tone at 200 Hz)"""
        pass

    def synthesize_hihat(self, duration=0.05):
        """Synthesize hihat (filtered white noise)"""
        pass

    def generate_pattern(self, pattern_name, bpm, duration):
        """Generate drum pattern audio"""
        pass
```

**Deliverable**: Working drum synthesizer with basic patterns

---

### 3.2 Implement Sequenced Bass Generator

**Tasks**:
- [ ] Create `synth_renderer/generators/sequenced_bass.py`
- [ ] Implement `SequencedBassGenerator` class
- [ ] Create bass pattern library
- [ ] Grid-lock notes to BPM

**File**: `synth_renderer/generators/sequenced_bass.py`

**Key Classes**:
```python
class SequencedBassGenerator:
    def generate(self, duration, bpm, scale, pattern_name):
        """Generate grid-locked bassline"""
        pass

    def get_pattern(self, pattern_name):
        """Retrieve bass pattern from library"""
        pass
```

**Deliverable**: Working sequenced bass generator

---

### 3.3 Implement Arpeggio Generator

**Tasks**:
- [ ] Create `synth_renderer/generators/arpeggios.py`
- [ ] Implement `ArpeggioGenerator` class
- [ ] Create arpeggio pattern library (16th notes)
- [ ] Support octave range specification

**File**: `synth_renderer/generators/arpeggios.py`

**Key Classes**:
```python
class ArpeggioGenerator:
    def generate(self, duration, bpm, scale, pattern_name, octave_range=3):
        """Generate 16th-note arpeggios"""
        pass
```

**Deliverable**: Working arpeggio generator

---

### 3.4 Implement Pad Chord Generator

**Tasks**:
- [ ] Create `synth_renderer/generators/pads.py`
- [ ] Implement `PadGenerator` class
- [ ] Support chord types (minor, major, sus, add9)
- [ ] Implement chord progression logic

**File**: `synth_renderer/generators/pads.py`

**Key Classes**:
```python
class PadGenerator:
    def generate(self, duration, bpm, scale, chord_type, changes_per_section):
        """Generate sustained pad chords"""
        pass
```

**Deliverable**: Working pad chord generator

---

### 3.5 Implement JarreRenderer

**Tasks**:
- [ ] Create `synth_renderer/jarre.py`
- [ ] Implement `JarreRenderer` class extending `StyleRenderer`
- [ ] Integrate all new generators (drums, bass, arpeggios, pads)
- [ ] Implement parameter translation methods
- [ ] Define mixing strategy

**File**: `synth_renderer/jarre.py`

**Key Methods**:
```python
class JarreRenderer(StyleRenderer):
    def render_section(self, section, context):
        """Orchestrate all Jarre layers"""
        pass

    def get_bass_params(self, narrative, tension):
        """Translate narrative → sequenced bass parameters"""
        pass

    def get_drum_params(self, narrative, tension):
        """Translate narrative → drum pattern parameters"""
        pass

    def get_arpeggio_params(self, narrative, tension):
        """Translate narrative → arpeggio parameters"""
        pass

    def get_pad_params(self, narrative, tension):
        """Translate narrative → pad chord parameters"""
        pass

    def get_melody_params(self, narrative, tension):
        """Translate narrative → dense melody parameters"""
        pass

    def mix_layers(self, layers, profile):
        """Mix layers according to Jarre profile"""
        pass
```

**Deliverable**: Working JarreRenderer producing full electronic music

---

### 3.6 Testing & Validation

**Tasks**:
- [ ] Render Ding/Gukesh Game 1 with Jarre renderer
- [ ] Validate all layers present (bass, drums, arpeggios, pads, melody)
- [ ] Test tempo calculation from tension
- [ ] Verify rhythmic drive and pulse
- [ ] A/B compare to Spiegel rendering (should sound completely different)

**Test Command**:
```bash
./c2m data/ding_gukesh/game1.pgn --style jarre
# Listen to both styles
./c2m data/ding_gukesh/game1.pgn --style spiegel  # Sparse, evolving
./c2m data/ding_gukesh/game1.pgn --style jarre    # Dense, rhythmic
```

**Success Criteria**:
- Clear rhythmic pulse (120-140 BPM)
- Drums audible and driving
- Bass prominent and sequenced
- Arpeggios provide texture
- Melody is continuous and memorable
- Perceptually distinct from Spiegel style

**Deliverable**: Validated Jarre renderer producing electronic music

---

## Phase 4: CLI Integration & User Experience

**Goal**: Make style selection easy and intuitive

### 4.1 Add Style Selection to CLI

**Tasks**:
- [ ] Add `--style` argument to `c2m` script
- [ ] Add `--list-styles` command
- [ ] Update help text
- [ ] Set default to `spiegel` (backward compatibility)

**Changes to `c2m` (main script)**:

```python
import argparse

parser = argparse.ArgumentParser(description='Chess to Music Synthesizer')
parser.add_argument('pgn_file', help='Path to PGN file')
parser.add_argument('--style', choices=['spiegel', 'jarre'], default='spiegel',
                    help='Musical style (default: spiegel)')
parser.add_argument('--list-styles', action='store_true',
                    help='List available styles and exit')

args = parser.parse_args()

if args.list_styles:
    print("Available styles:")
    for style, profile in config.STYLE_PROFILES.items():
        print(f"  {style}: {profile['description']}")
    sys.exit(0)

# Create composer with selected style
composer = ChessSynthComposer(tags, config, style=args.style)
```

**Deliverable**: User-friendly CLI for style selection

---

### 4.2 Documentation & Examples

**Tasks**:
- [ ] Update README.md with style examples
- [ ] Create audio examples (same game, both styles)
- [ ] Document style characteristics
- [ ] Provide listening guide

**Files to Update**:
- `README.md` - Add style selection documentation
- `docs/USAGE.md` - CLI examples
- `docs/STYLES.md` - Style comparison guide

**Deliverable**: Complete documentation for users

---

## Phase 5: Testing, Refinement & Polish

**Goal**: Ensure quality and robustness

### 5.1 Comprehensive Testing

**Test Games**:
1. Ding/Gukesh Game 1 (DEATH_SPIRAL, defeat)
2. Attacking masterpiece game (if available)
3. Drawn game (PEACEFUL_DRAW)
4. Short tactical game (< 30 moves)
5. Long endgame (> 80 moves)

**Test Matrix**:
```
For each game:
  - Render with Spiegel style
  - Render with Jarre style
  - Validate layer balance
  - Check for audio artifacts (clicks, clipping)
  - Verify gesture timing
  - Test stereo image
```

**Deliverable**: Test suite validating both renderers

---

### 5.2 Parameter Tuning

**Tasks**:
- [ ] Fine-tune mixing levels (A/B testing)
- [ ] Adjust narrative parameter mappings
- [ ] Optimize tempo ranges
- [ ] Balance gesture prominence
- [ ] Tune filter ranges

**Method**:
- Iterative listening tests
- Reference against original works (Spiegel albums, Jarre albums)
- User feedback

**Deliverable**: Optimized parameters for both styles

---

### 5.3 Code Quality & Documentation

**Tasks**:
- [ ] Add docstrings to all classes/methods
- [ ] Type hints for all parameters
- [ ] Code review and refactoring
- [ ] Performance profiling (ensure no slowdown)
- [ ] Error handling for edge cases

**Deliverable**: Production-quality code

---

## Phase 6: Future Extensions (Optional)

**Potential Future Styles**:
- `tangerine_dream` - Berlin school sequencing
- `kraftwerk` - Mechanical, robotic precision
- `eno` - Extreme ambient, no rhythm
- `hybrid` - Spiegel drones + Jarre rhythm

**Architecture Benefits**:
Adding new styles only requires:
1. Create new renderer class
2. Define style profile in config
3. Implement parameter translation methods
4. Register in factory

**No changes needed to**:
- Core synthesis engine
- Chess analysis
- Pattern generators
- Gesture system

---

## Risk Mitigation

### Risk 1: Jarre Components Too Complex
**Mitigation**: Start with simple drum/bass, iterate to complexity
**Fallback**: Simplify Jarre style, focus on sequenced bass + minimal drums

### Risk 2: Performance Degradation
**Mitigation**: Profile early, optimize hot paths
**Fallback**: Reduce layer count, simplify DSP

### Risk 3: Styles Sound Too Similar
**Mitigation**: A/B testing throughout development
**Fallback**: Exaggerate differences, consult reference material

### Risk 4: Breaking Changes to Existing System
**Mitigation**: Phase 2 includes regression testing
**Fallback**: Keep old composer as legacy option

---

## Success Metrics

1. ✅ SpiegelRenderer produces output matching current system
2. ✅ JarreRenderer produces perceptually distinct output
3. ✅ Same game rendered in both styles sounds appropriate
4. ✅ All layers audible and balanced
5. ✅ No audio artifacts (clicks, distortion)
6. ✅ Code is maintainable and extensible
7. ✅ Documentation is complete

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: Infrastructure | 5 tasks | 4-6 hours |
| Phase 2: Spiegel Renderer | 3 tasks | 6-8 hours |
| Phase 3: Jarre Renderer | 6 tasks | 16-20 hours |
| Phase 4: CLI Integration | 2 tasks | 2-3 hours |
| Phase 5: Testing & Polish | 3 tasks | 8-10 hours |
| **Total** | **19 tasks** | **36-47 hours** |

**Recommended Approach**: Implement in order, validate each phase before proceeding.

---

## Recommended First Style: Spiegel

**Why Spiegel First?**
1. ✅ Validates architecture with minimal new code
2. ✅ Current system is ~70% Spiegel already
3. ✅ Provides regression test (output should match)
4. ✅ Establishes patterns for future renderers
5. ✅ Lower risk than starting with Jarre

**After Spiegel works**, implementing Jarre becomes:
- A test of the architecture's flexibility
- A validation that style decisions are truly decoupled
- A demonstration of the system's power

---

## Next Steps

1. **Review this plan** with user
2. **Choose starting phase** (recommend: Phase 1 → Phase 2)
3. **Create initial infrastructure** (Phase 1)
4. **Implement SpiegelRenderer** (Phase 2)
5. **Validate with existing output** (regression test)
6. **Proceed to Jarre** (Phase 3)

**Ready to begin implementation?**
