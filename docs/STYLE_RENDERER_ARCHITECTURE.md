# Style Renderer Architecture

## Overview

The Style Renderer system decouples **aesthetic decisions** from **synthesis implementation**, allowing the same chess game to be rendered in different musical styles without changing the core synthesis engine or game analysis.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Chess Tags JSON                                          │
│ - overall_narrative (e.g., "DEATH_SPIRAL")                      │
│ - sections with narratives (e.g., "DESPERATE_DEFENSE")          │
│ - key_moments with types (e.g., "BLUNDER", "BRILLIANT")         │
│ - tension/entropy curves                                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ STYLE RENDERER (Selected by user)                               │
│                                                                 │
│ Responsibilities:                                               │
│ 1. Translate narrative → layer decisions                        │
│    (Which layers: bass? drums? pads? drones?)                   │
│                                                                 │
│ 2. Translate narrative → synthesis parameters                   │
│    (Waveforms, envelopes, filters, timing)                      │
│                                                                 │
│ 3. Define mixing strategy                                       │
│    (Layer weights, stereo placement)                            │
│                                                                 │
│ 4. Coordinate pattern generators                                │
│    (Which pattern types to use)                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ COMPOSER / PATTERN GENERATORS / GESTURE ENGINE                  │
│                                                                 │
│ Executes style renderer instructions:                          │
│ - Generates abstract note events (freq, time, duration)        │
│ - Uses specified pattern algorithms                            │
│ - Creates gesture audio for key moments                        │
│                                                                 │
│ Output: Lists of NoteEvent objects                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ SYNTH ENGINE (SubtractiveSynth)                                │
│                                                                 │
│ Pure DSP - no musical knowledge:                               │
│ - Oscillators (saw, square, pulse, triangle, sine)            │
│ - Filters (Moog lowpass, highpass, bandpass)                  │
│ - ADSR envelopes                                               │
│ - PolyBLEP anti-aliasing                                       │
│                                                                 │
│ Input: Synthesis parameters (Hz, ADSR, filter cutoff)         │
│ Output: Audio samples (numpy arrays)                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ MIX + STEREO FX (Controlled by style renderer)                 │
│                                                                 │
│ - Layer mixing with style-specific weights                     │
│ - Stereo panning (static or entropy-driven)                    │
│ - Master effects (soft clipping, limiting)                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    [Stereo WAV Output]
```

## Key Principles

### 1. Style Renderer is a Coordinator, Not a Post-Processor

The style renderer makes decisions **before** composition happens:

```python
# CORRECT: Style renderer decides → Composer executes
renderer = JarreRenderer(...)
section_params = renderer.get_section_params(section)
audio = composer.execute(section_params)

# WRONG: Can't apply style after notes are generated
notes = composer.generate_notes(...)
audio = style_renderer.apply_style(notes)  # Too late!
```

### 2. Separation of Concerns

| Component | Responsibility | Knowledge |
|-----------|---------------|-----------|
| **Style Renderer** | Aesthetic decisions | Musical styles, arrangement |
| **Pattern Generators** | Musical logic | Algorithms, theory, structure |
| **Synth Engine** | Audio generation | DSP, filters, oscillators |
| **Config** | Parameter storage | Presets, scales, envelopes |

### 3. Same Analysis, Different Output

```
Same Chess Game (Ding vs Gukesh)
           ↓
    [Chess Tags]  ← Same for all styles
           ↓
  ┌────────┴────────┐
  ↓                 ↓
[Jarre Renderer] [Spiegel Renderer]
  ↓                 ↓
Dense, rhythmic   Sparse, evolving
sequences         drones
```

## Style Renderer Interface

```python
class StyleRenderer(ABC):
    """
    Abstract base class for all style renderers.
    Subclasses define specific aesthetic approaches.
    """

    def __init__(self, config: SynthConfig, synth_engines: dict):
        self.config = config
        self.synth_engines = synth_engines
        self.style_profile = None  # Set by subclass

    @abstractmethod
    def render_section(self, section: dict, context: dict) -> np.ndarray:
        """
        Main entry point: Render a section to audio.

        Args:
            section: Section data from tags (narrative, tension, key_moments)
            context: Overall context (scale, tempo, overall_narrative)

        Returns:
            Stereo audio array (N, 2)
        """
        pass

    # === Parameter Translation Methods ===
    # Each renderer implements these to translate narrative → synth params

    @abstractmethod
    def get_bass_params(self, narrative: str, tension: float) -> dict:
        """Translate narrative → bass synthesis parameters"""
        pass

    @abstractmethod
    def get_melody_params(self, narrative: str, tension: float) -> dict:
        """Translate narrative → melody synthesis parameters"""
        pass

    @abstractmethod
    def get_rhythm_params(self, narrative: str, tension: float) -> dict:
        """Translate narrative → rhythm synthesis parameters"""
        pass

    # === Helper Methods ===

    def mix_layers(self, layers: dict, weights: dict) -> np.ndarray:
        """Mix layers with specified weights"""
        pass

    def apply_stereo(self, mono: np.ndarray, pan: float, width: float) -> np.ndarray:
        """Apply stereo panning and width"""
        pass
```

## Proposed Styles

### Style 1: JARRE_1978

**Inspiration**: Équinoxe, Oxygène, Magnetic Fields

**Characteristics**:
- Dense sequenced basslines (ARP 2600 style)
- Layered arpeggios (16th notes, 2-4 octaves)
- Analog drum machines (LinnDrum patterns)
- Lush pad chords (string-like, slow attack)
- Bright melodic leads (saw/pulse with filter sweeps)
- Constant rhythmic pulse (120-140 BPM)

**Layers**:
1. Sequenced bass (every beat)
2. Drum pattern (kick, snare, hihat)
3. Arpeggio sequence (16th notes)
4. Pad chords (4-8 beat sustain)
5. Lead melody (continuous phrases)
6. Gestures (key moments)

**Key Parameters**:
- Note density: 5.0x (very dense)
- Rhythm: Full drum machine
- Bass: Sequenced on grid
- Waveforms: Saw, pulse (analog character)
- Filter: Bright, swept (500-5000 Hz)
- Tempo: 120-140 BPM

### Style 2: LAURIE_SPIEGEL_1979

**Inspiration**: The Expanding Universe, Appalachian Grove, Patchwork

**Characteristics**:
- Evolving drones (long-form, minutes)
- Sparse melodic events (thoughtful placement)
- Algorithmic processes (Markov chains, recursive patterns)
- Harmonic intelligence (Bach-inspired progressions)
- Minimal percussion (heartbeat, if any)
- Emphasis on texture and timbre evolution
- Gestural punctuation (key moments as dramatic events)

**Layers**:
1. Evolving drone (sub-bass to midrange sweep)
2. Sparse melodic patterns (Markov/recursive)
3. Heartbeat pulse (biological anchor)
4. Harmonic gestures (key moments)
5. Textural events (filtered noise, shimmer)

**Key Parameters**:
- Note density: 0.2x (very sparse)
- Rhythm: Minimal (heartbeat only)
- Bass: Long drone (30-60s sustained)
- Waveforms: Triangle, sine (pure tones)
- Filter: Evolving sweeps (slow LFO)
- Tempo: Free time / very slow pulse

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create `StyleRenderer` base class
2. Implement `StyleRendererFactory`
3. Refactor `ChessSynthComposer` to use renderers
4. Add style selection to CLI/config

### Phase 2: Implement First Style (Spiegel)
**Rationale**: Your current system is already 70% Spiegel-style, so this validates the architecture with minimal new code.

1. Create `SpiegelRenderer` class
2. Map existing drone/pattern generation to renderer methods
3. Ensure output matches current system (regression test)
4. Add style-specific config profiles

### Phase 3: Implement Second Style (Jarre)
**Rationale**: This requires new components (drums, arpeggios, pads) and fully tests the renderer architecture.

1. Create `JarreRenderer` class
2. Implement drum synthesizer (kick, snare, hihat)
3. Implement arpeggio generator
4. Implement pad chord generator
5. Implement sequenced bassline generator
6. Create Jarre-specific config profiles

### Phase 4: Testing & Refinement
1. Render same game in both styles
2. A/B comparison
3. Parameter tuning
4. Documentation

## File Structure

```
chess-2-music/
├── synth_renderer/              # NEW module
│   ├── __init__.py
│   ├── base.py                  # StyleRenderer abstract class
│   ├── factory.py               # StyleRendererFactory
│   ├── spiegel.py               # SpiegelRenderer
│   ├── jarre.py                 # JarreRenderer
│   └── generators/              # Style-specific generators
│       ├── drums.py             # Drum synthesis
│       ├── arpeggios.py         # Arpeggio patterns
│       ├── pads.py              # Pad chords
│       └── sequenced_bass.py   # Grid-locked bass
│
├── synth_composer.py            # REFACTORED to use renderers
├── synth_config.py              # ADD style profiles
├── synth_engine.py              # UNCHANGED
├── layer3b/                     # UNCHANGED (gestures)
└── synth_composer/              # UNCHANGED (patterns)
```

## Migration Path

### Current Code (Implicit Style)
```python
class ChessSynthComposer:
    def compose_section(self, section):
        # Hard-coded style decisions scattered
        drone = self.synth.create_synth_note(...)
        pattern = self.pattern_coordinator.generate_pattern(...)
        mixed = drone * 0.15 + pattern * 0.6
        return mixed
```

### After Refactoring (Explicit Style)
```python
class ChessSynthComposer:
    def __init__(self, tags, config, style='spiegel'):
        self.renderer = StyleRendererFactory.create(
            style, config, self.create_synth_engines()
        )

    def compose_section(self, section):
        # Renderer handles all style decisions
        return self.renderer.render_section(section, self.context)
```

### CLI Usage
```bash
# Current (implicit Spiegel style)
./c2m data/game1.pgn

# After refactoring (explicit style selection)
./c2m data/game1.pgn --style spiegel    # Default
./c2m data/game1.pgn --style jarre      # New Jarre style
```

## Benefits

1. **Modularity**: Add new styles without touching core synthesis
2. **Testability**: Each style is isolated and testable
3. **Clarity**: Style decisions are explicit, not scattered
4. **Flexibility**: Users can choose aesthetic preference
5. **Maintainability**: Changes to one style don't affect others
6. **Experimentation**: Easy to create hybrid styles

## Next Steps

1. Review this architecture document
2. Choose first style to implement (recommend: Spiegel for validation)
3. Create base renderer infrastructure
4. Implement chosen style renderer
5. Test with existing games
6. Implement second style
7. Compare outputs
