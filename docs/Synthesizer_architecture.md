  ---
  LAYER 1: Low-Level Synthesis (The Engine)

  synth_engine.py (680 lines)

  Purpose: Pure DSP - generates audio waveforms
```
  class SubtractiveSynth:
      # Raw oscillators
      def oscillator(freq, duration, waveform):
          # Generates: sine, saw, square, pulse, triangle
          # Uses PolyBLEP anti-aliasing
          return audio_samples  # numpy array

      # Filters
      def moog_filter(signal, cutoff_hz, resonance):
          # Moog-style lowpass with resonance

      def highpass_filter(signal, cutoff_hz):
          # Simple highpass

      # Envelope generators
      def adsr_envelope(duration, attack, decay, sustain, release):
          # Creates amplitude/filter envelopes

      # High-level note generator
      def create_synth_note(freq, duration, waveform,
                            filter_base, filter_env_amount,
                            resonance, amp_env, filter_env):
          # Combines: oscillator + filter + envelope
          return audio_samples
```
  Key Point: This is pure DSP. No musical knowledge. Just "give me Hz, give me ADSR values, I'll give you audio."

  ---
  LAYER 2: Configuration (Musical Parameters)

  synth_config.py (812 lines)

  Purpose: All musical parameters in one place

```
  class SynthConfig:
      # Musical scales (Hz values)
      SCALES = {
          'minor': [110, 123.47, 130.81, ...],
          'phrygian': [110, 116.54, 130.81, ...],
      }

      # Envelope presets (ADSR tuples)
      ENVELOPES = {
          'pad': (0.5, 0.0, 1.0, 0.5),      # Slow attack
          'pluck': (0.001, 0.05, 0.4, 0.1),  # Fast attack
          'doom': (0.5, 0.0, 1.0, 1.0),      # Long sustain
      }

      # Narrative-specific synthesis parameters
      NARRATIVE_BASE_PARAMS = {
          'TUMBLING_DEFEAT': {
              'base_waveform': 'supersaw',
              'filter_start': 2500,
              'filter_end': 300,
              'scale': 'phrygian',
          },
          'ATTACKING_MASTERPIECE': {
              'base_waveform': 'pulse',
              'filter_start': 500,
              'filter_end': 5000,
              'scale': 'dorian',
          }
      }

      # Mixing levels
      MIXING = {
          'drone_level': 0.15,
          'pattern_level': 0.6,
          'sequencer_level': 0.4,
          'moment_level': 0.5,
      }
```
  Key Point: This is the aesthetic control center. Change a value here → changes the sound globally.

  ---
  LAYER 3: Pattern Generation (Musical Logic)

  synth_composer/ module (multiple files)

  This is a mini-framework for generating melodic patterns:
```
  synth_composer/
  ├── coordinator.py          # Main orchestrator
  ├── core/
  │   ├── note_event.py      # Data structure for notes
  │   ├── synthesizer.py     # Wrapper: NoteEvent → audio
  │   ├── audio_buffer.py    # Timeline-based audio placement
  │   └── timing_engine.py   # Beat/time calculations
  └── patterns/
      ├── markov.py          # Markov chain patterns
      ├── state_machine.py   # Complex state-based patterns
      ├── theory.py          # Theory-based patterns
      └── outro.py           # Ending patterns
```
  How it works:

  # 1. Pattern defines WHAT notes to play (abstract)
```
  class MarkovChainPattern:
      def generate_events(duration, scale, params):
          # Returns: List of NoteEvent objects
          # Each NoteEvent has: freq, duration, timestamp,
          #                     velocity, waveform, filter params
          return [
              NoteEvent(freq=220, duration=0.5, timestamp=0.0, ...),
              NoteEvent(freq=330, duration=0.3, timestamp=0.6, ...),
          ]
```
  # 2. PatternCoordinator converts events → audio
```
  class PatternCoordinator:
      def generate_pattern(narrative, duration, scale, params):
          # Looks up pattern generator for narrative
          pattern = self.patterns[narrative]  # e.g., MarkovChainPattern

          # Generate abstract note events
          events = pattern.generate_events(duration, scale, params)

          # Convert to audio using NoteSynthesizer
          audio_buffer = AudioBuffer(duration, sample_rate)
          for event in events:
              samples = self.synthesizer.synthesize(event)
              audio_buffer.add(samples, event.timestamp)

          return audio_buffer.render()  # Final audio array
```
  Key Point: Separates musical logic (what/when to play) from synthesis (how it sounds).

  ---
  LAYER 4: Gesture System (Event-Based)

  layer3b/ module (Layer 3b gestures)
```
  layer3b/
  ├── coordinator.py         # Manages gesture scheduling
  ├── archetype_configs.py   # Gesture definitions (BLUNDER, BRILLIANT, etc.)
  ├── base.py                # Base gesture generator
  ├── curve_generators.py    # Pitch/filter/envelope curves
  ├── particle_system.py     # Particle-based gestures
  ├── synthesizer.py         # Gesture → audio rendering
  └── utils.py              # Helper functions
```

  How it works:

  # 1. Archetype defines gesture characteristics
  ```
  ARCHETYPES = {
      "BRILLIANT": {
          "duration_base": 2.5,
          "pitch": {"type": "ascending_glissando", "octave_rise": 2},
          "harmony": {"type": "unison_to_chord", "num_voices": 6},
          "filter": {"type": "lowpass_sweep", "cutoff_start": 300},
          "envelope": {"type": "sudden_sustained"},
      }
  }
  ```

  # 2. GestureCoordinator schedules and renders
  ```
  class GestureCoordinator:
      def generate_gestures(key_moments, context):
          gestures_audio = []

          for moment in key_moments:
              # Get archetype config
              archetype = ARCHETYPES[moment['type']]

              # Generate curves (pitch, filter, envelope over time)
              curves = self.generate_curves(archetype, context)

              # Synthesize gesture audio
              audio = self.synthesizer.render_gesture(curves)

              gestures_audio.append((audio, moment['timestamp']))

          return gestures_audio
```
  Key Point: Event-triggered synthesis for dramatic moments.

  ---
  LAYER 5: Narrative Processes (Time-Based Transformations)

  synth_narrative.py (201 lines)
```
  class TumblingDefeatProcess(NarrativeProcess):
      """Gradual deterioration over time"""

      def update(self, current_time, key_moment):
          # Accumulate errors
          if key_moment and key_moment['type'] == 'BLUNDER':
              self.error_accumulation += 0.2

          # Return transformations to apply
          return {
              'pitch_drift_cents': self.pitch_drift * 20,
              'tempo_multiplier': 1.0 + self.tempo_drift,
              'volume_decay': 1.0 - (progress * 0.3),
          }
```
  Key Point: Spiegel-inspired algorithmic processes that evolve over time.

  ---
  LAYER 6: Main Composer (Orchestrates Everything)

  synth_composer.py (2612 lines) - This is the conductor

```
  class ChessSynthComposer:
      def __init__(self, chess_tags, config):
          # Create synthesis engines
          self.drone_synth = SubtractiveSynth(...)
          self.pattern_synth = SubtractiveSynth(...)

          # Create pattern coordinator (Layer 2)
          self.pattern_coordinator = PatternCoordinator(...)

          # Create gesture coordinator (Layer 3b)
          self.gesture_coordinator = GestureCoordinator(...)

          # Create narrative process
          self.narrative_process = create_narrative_process(
              chess_tags['overall_narrative']
          )

      def compose_section(self, section):
          # === LAYER 1: DRONE ===
          drone_audio = self.drone_synth.create_synth_note(
              freq=base_freq,
              duration=section_duration,
              waveform='saw',
              amp_env=(0.5, 0.0, 1.0, 0.5),  # From config
              ...
          )

          # === LAYER 2: PATTERNS ===
          pattern_audio = self.pattern_coordinator.generate_pattern(
              narrative=section['narrative'],  # e.g., "DESPERATE_DEFENSE"
              duration=section_duration,
              scale=self.config.SCALES['minor'],
              params={...}
          )

          # === LAYER 3a: HEARTBEAT ===
          heartbeat_audio = self.synthesize_heartbeat(...)

          # === LAYER 3b: GESTURES ===
          gesture_audio = self.gesture_coordinator.generate_gestures(
              key_moments=section['key_moments'],
              context={...}
          )

          # === MIX LAYERS ===
          mixed = (
              drone_audio * config.MIXING['drone_level'] +
              pattern_audio * config.MIXING['pattern_level'] +
              heartbeat_audio * config.MIXING['sequencer_level'] +
              gesture_audio * config.MIXING['moment_level']
          )

          return mixed

      def compose(self):
          """Full game composition"""
          all_sections = []

          for section in self.tags['sections']:
              section_audio = self.compose_section(section)
              all_sections.append(section_audio)

          # Concatenate and apply stereo panning
          final_audio = np.concatenate(all_sections)

          # Convert to WAV and save
          self.save_wav(final_audio)
```
  ---

  DATA FLOW DIAGRAM
```
  Chess PGN
      ↓
  [tagger.py] → Extracts features, creates tags JSON
      ↓
  tags = {
      'overall_narrative': 'TUMBLING_DEFEAT',
      'sections': [
          {'narrative': 'DESPERATE_DEFENSE', 'key_moments': [...]}
      ]
  }
      ↓
  ChessSynthComposer(tags, config)
      ↓
  ┌─────────────────────────────────────────────────┐
  │         FOR EACH SECTION                        │
  ├─────────────────────────────────────────────────┤
  │                                                 │
  │  Layer 1: Drone (synth_engine.py)               │
  │    SubtractiveSynth.create_synth_note()         │
  │    → Generates long sustained tone              │
  │                                                 │
  │  Layer 2: Patterns (synth_composer/ module)     │
  │    PatternCoordinator → MarkovChainPattern      │
  │    → Generates NoteEvent list                   │
  │    → NoteSynthesizer → SubtractiveSynth         │
  │    → Renders to audio                           │
  │                                                 │
  │  Layer 3a: Heartbeat (synth_composer.py)        │
  │    → Direct SubtractiveSynth calls              │
  │                                                 │
  │  Layer 3b: Gestures (layer3b/ module)           │
  │    GestureCoordinator → Archetype config        │
  │    → Curve generators                           │
  │    → GestureSynthesizer → SubtractiveSynth      │
  │    → Renders gesture audio                      │
  │                                                 │
  │  Mix layers (weights from synth_config.py)      │
  │  Apply stereo panning                           │
  │                                                 │
  └─────────────────────────────────────────────────┘
      ↓
  Concatenate sections
      ↓
  Write WAV file
```

  ---
  KEY INSIGHTS

  1. Two Synthesis Paths

  Path A: Direct synthesis (Layer 1 - Drone, Heartbeat)
  synth_composer.py → SubtractiveSynth.create_synth_note() → audio

  Path B: Pattern-based synthesis (Layer 2)
  synth_composer.py → PatternCoordinator → Pattern.generate_events()
  → NoteEvent list → NoteSynthesizer → SubtractiveSynth → audio

  Path C: Gesture-based synthesis (Layer 3b)
  synth_composer.py → GestureCoordinator → Archetype config
  → Curve generators → GestureSynthesizer → SubtractiveSynth → audio

  2. Where Parameters Come From

```
  | Parameter                     | Source                                           |
  |-------------------------------|--------------------------------------------------|
  | Waveform, filter cutoff, ADSR | synth_config.py (based on narrative)             |
  | Which notes to play           | Pattern generators (Markov, state machine, etc.) |
  | When to play gestures         | tags['key_moments'] from tagger                  |
  | How gestures sound            | layer3b/archetype_configs.py                     |
  | Mix levels                    | synth_config.py MIXING section                   |
```
