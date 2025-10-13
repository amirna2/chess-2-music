# Jean-Michel Jarre Style Profile (1978)

**Inspired by**: Oxygène (1976), Équinoxe (1978), Magnetic Fields (1981)

## Aesthetic Philosophy

Jarre's 1970s-1980s work represents the apex of **analog electronic pop music**:
- **Melodic accessibility**: Memorable, hummable themes
- **Rhythmic drive**: Constant pulse, danceable
- **Layered sequences**: Dense polyphonic texture
- **Analog warmth**: Saw/pulse waveforms with character
- **Cosmic atmosphere**: Lush pads and sweeping arpeggios
- **Dynamic range**: Quiet intros building to powerful climaxes

## Musical Characteristics

### Temporal Structure
- **Form**: Verse-like sections with identifiable themes
- **Duration**: 3-6 minute "songs" (not long-form drones)
- **Pacing**: Energetic, forward-moving
- **Transitions**: Clear sectional changes

### Harmonic Language
- **Scales**: Minor, Dorian (modal but accessible)
- **Progressions**: Simple, repetitive (i-VII-VI-V patterns)
- **Dissonance**: Minimal, consonant
- **Modulation**: Key changes for drama

### Rhythmic Character
- **Pulse**: Strong, constant (120-140 BPM)
- **Grid**: Strict 4/4 time signature
- **Accent**: Kick on 1 & 3, snare on 2 & 4
- **Percussion**: LinnDrum-style patterns (analog drum machine)

### Timbral Palette
- **Waveforms**: Saw, pulse (analog character)
- **Filters**: Swept resonant filters (Moog-style)
- **Envelopes**: Plucky attacks, staccato articulation
- **Texture**: Layered sequences, not drones

## Layer Architecture

```
Layer 1: Sequenced Bassline (Foundation)
    │
    ├─ Waveform: Saw or pulse
    ├─ Frequency: Follows scale root (55-110 Hz)
    ├─ Rhythm: On beat grid (every beat or every 2 beats)
    ├─ Filter: 300-600 Hz (tight bass)
    ├─ Envelope: Pluck (0.001s attack, 0.4s release)
    ├─ Pattern: Stepwise, octave jumps, root-fifth alternation
    └─ Articulation: 80% note length (slight gap)

Layer 2: Drum Pattern (Rhythmic backbone)
    │
    ├─ Kick: Every beat or on 1 & 3
    │   ├─ Synthesis: Pitch envelope (150→50 Hz, 0.3s)
    │   ├─ Waveform: Sine
    │   └─ Envelope: (0.001s, 0.3s, 0.0s, 0.0s)
    │
    ├─ Snare: On 2 & 4
    │   ├─ Synthesis: 70% filtered white noise + 30% tone (200 Hz)
    │   ├─ Filter: 3000 Hz highpass
    │   └─ Envelope: (0.001s, 0.15s, 0.0s, 0.0s)
    │
    └─ Hi-hat: Every beat or 8th/16th notes
        ├─ Synthesis: Filtered white noise
        ├─ Filter: 8000 Hz highpass
        └─ Envelope: (0.001s, 0.05s, 0.0s, 0.0s)

Layer 3: Arpeggio Sequence (Textural movement)
    │
    ├─ Waveform: Pulse (25% duty cycle)
    ├─ Frequency: 2-4 octave range (220-880 Hz typical)
    ├─ Rhythm: 16th notes (strict grid)
    ├─ Pattern: [0, 3, 7, 12, 7, 3] repeating
    ├─ Filter: 1500-3000 Hz (bright)
    ├─ Envelope: Pluck (0.001s, 0.1s, 0.0s, 0.05s)
    ├─ Velocity: Slight variation (0.7-1.0)
    └─ Articulation: Staccato (50% note length)

Layer 4: Pad Chords (Harmonic bed)
    │
    ├─ Waveform: Triangle or filtered saw
    ├─ Frequency: Chord tones (i, iv, v, VII)
    ├─ Rhythm: Changes every 4-8 beats
    ├─ Filter: 800-1200 Hz (warm, not bright)
    ├─ Filter LFO: 0.1 Hz sine wave (subtle movement)
    ├─ Envelope: Pad (0.5s attack, full sustain, 0.5s release)
    ├─ Voices: 3-4 note chords
    └─ Articulation: Legato (overlapping)

Layer 5: Lead Melody (Main theme)
    │
    ├─ Waveform: Saw or pulse with filter sweep
    ├─ Frequency: Melodic range (220-880 Hz)
    ├─ Rhythm: On beat grid (quarter/eighth notes)
    ├─ Pattern: Singable melodies, repeated phrases
    ├─ Filter: 500-5000 Hz sweep (dramatic)
    ├─ Filter Envelope: Opens on attack
    ├─ Envelope: Stab (0.01s, 0.1s, 0.7s, 0.2s)
    ├─ Portamento: Slight glide between notes (0.05s)
    └─ Articulation: Expressive (60-90% note length)

Layer 6: Gestural Events (Key moments)
    │
    ├─ Trigger: Chess key_moments (less frequent than Spiegel)
    ├─ Style: Dramatic sweeps and risers
    ├─ Duration: 0.5-2 seconds (shorter, more punctual)
    ├─ Integration: Mixed lower in Jarre style (don't overshadow melody)
    └─ Function: Accents, not main focus
```

## Parameter Mapping

### Narrative → Layer Parameters

```python
JARRE_NARRATIVE_PARAMS = {
    'TUMBLING_DEFEAT': {
        # Tempo slows as game deteriorates
        'bpm': 130,                      # Start faster
        'bpm_end': 100,                  # Slow down
        'bass_pattern': 'descending',    # Falling bassline
        'bass_octave_shifts': True,      # Octave drops for drama
        'drum_pattern': 'half_time_heavy',  # Slower, heavier
        'arpeggio_direction': 'descending',
        'arpeggio_density': 16,          # 16th notes
        'pad_chord_type': 'minor_dim',   # Dark chords
        'melody_scale': 'phrygian',
        'melody_density': 4.0,           # Dense
        'filter_trajectory': 'closing',  # Filters darken
    },

    'ATTACKING_MASTERPIECE': {
        'bpm': 120,
        'bpm_end': 140,                  # Speed up (excitement)
        'bass_pattern': 'driving_eighth',
        'bass_octave_shifts': False,
        'drum_pattern': 'four_on_floor', # Steady drive
        'arpeggio_direction': 'ascending',
        'arpeggio_density': 16,
        'pad_chord_type': 'major_add9',  # Bright chords
        'melody_scale': 'dorian',
        'melody_density': 5.0,           # Very active
        'filter_trajectory': 'opening',  # Filters brighten
    },

    'DESPERATE_DEFENSE': {
        'bpm': 128,
        'bpm_end': 120,                  # Slightly slower
        'bass_pattern': 'syncopated',    # Uneven, anxious
        'bass_octave_shifts': True,
        'drum_pattern': 'breakbeat',     # Complex, tense
        'arpeggio_direction': 'bidirectional',
        'arpeggio_density': 32,          # 32nd notes (frantic)
        'pad_chord_type': 'suspended',   # Unresolved tension
        'melody_scale': 'phrygian',
        'melody_density': 3.5,
        'filter_trajectory': 'unstable', # Oscillating filters
    },

    'TACTICAL_CHAOS': {
        'bpm': 135,
        'bpm_end': 140,
        'bass_pattern': 'staccato_jumping',
        'bass_octave_shifts': True,
        'drum_pattern': 'drum_n_bass',   # Complex, fast
        'arpeggio_direction': 'random_walk',
        'arpeggio_density': 32,
        'pad_chord_type': 'cluster',     # Dissonant
        'melody_scale': 'chromatic',     # All notes available
        'melody_density': 6.0,           # Maximum activity
        'filter_trajectory': 'sweeping', # Rapid sweeps
    },

    'POSITIONAL_THEORY': {
        'bpm': 110,
        'bpm_end': 110,                  # Steady
        'bass_pattern': 'walking',       # Classic walking bass
        'bass_octave_shifts': False,
        'drum_pattern': 'minimal_pulse', # Subtle rhythm
        'arpeggio_direction': 'pattern',
        'arpeggio_density': 8,           # Slower (8th notes)
        'pad_chord_type': 'major_seventh',
        'melody_scale': 'dorian',
        'melody_density': 2.5,           # Moderate
        'filter_trajectory': 'gentle_wave',
    },
}
```

### Tension → Musical Parameters

```python
def map_tension_to_params(tension: float, bpm_base: int) -> dict:
    """
    Jarre approach: Tension affects tempo, density, and filter brightness
    """
    return {
        'bpm': bpm_base + int(tension * 20),       # ±20 BPM
        'drum_hits_per_bar': 4 + int(tension * 12), # 4-16 hits
        'arpeggio_octave_range': 2 + int(tension * 2),  # 2-4 octaves
        'filter_resonance': 0.5 + (tension * 2.0),  # 0.5-2.5
        'filter_cutoff_mult': 0.7 + (tension * 0.6), # 0.7-1.3x
        'melody_note_density': 2.0 + (tension * 4.0), # 2-6 notes/bar
        'pad_voice_count': 3 + int(tension),        # 3-4 voices
    }
```

### Bass Pattern Library

```python
JARRE_BASS_PATTERNS = {
    'walking': [0, 0, 4, 5, 7, 5, 4, 0],         # Classic walking
    'descending': [0, -2, -4, -5, -7, -9, -12],  # Falling
    'driving_eighth': [0, 0, 7, 7, 0, 0, 7, 7],  # Oxygène 4 style
    'syncopated': [0, None, 0, 7, None, 5, 0, None],  # Rests for tension
    'staccato_jumping': [0, 12, 0, 12, 7, 12, 5, 12],  # Octave jumps
    'root_fifth': [0, 7, 0, 7, 0, 7, 0, 7],      # Simple alternation
}
```

### Drum Pattern Library

```python
JARRE_DRUM_PATTERNS = {
    'four_on_floor': {
        'kick': [1, 0, 1, 0, 1, 0, 1, 0],        # Every beat
        'snare': [0, 0, 1, 0, 0, 0, 1, 0],       # 2 & 4
        'hihat': [1, 1, 1, 1, 1, 1, 1, 1],       # Every 8th
    },

    'half_time_heavy': {
        'kick': [1, 0, 0, 0, 1, 0, 0, 0],
        'snare': [0, 0, 0, 0, 1, 0, 0, 0],
        'hihat': [1, 0, 1, 0, 1, 0, 1, 0],
    },

    'breakbeat': {
        'kick': [1, 0, 0, 1, 0, 0, 1, 0],
        'snare': [0, 0, 1, 0, 0, 1, 0, 0],
        'hihat': [1, 1, 1, 1, 1, 1, 1, 1],
    },

    'minimal_pulse': {
        'kick': [1, 0, 0, 0, 0, 0, 0, 0],        # Once per bar
        'snare': [0, 0, 0, 0, 0, 0, 0, 0],       # Silent
        'hihat': [1, 0, 1, 0, 1, 0, 1, 0],       # Sparse
    },
}
```

### Arpeggio Pattern Library

```python
JARRE_ARPEGGIO_PATTERNS = {
    'classic_up': [0, 3, 7, 12, 15, 19, 24, 19, 15, 12, 7, 3],  # Up and down
    'oxygene': [0, 7, 12, 19, 12, 7],                           # Oxygène 4 pattern
    'equinoxe': [0, 3, 7, 10, 12, 10, 7, 3],                    # Équinoxe style
    'bouncing': [0, 12, 7, 19, 12, 24, 19, 12],                 # Octave jumps
    'cascading': [24, 19, 15, 12, 7, 3, 0, -5],                 # Falling
}
```

## Synthesis Parameters

### Envelopes (ADSR in seconds)

```python
JARRE_ENVELOPES = {
    'bass_pluck': (0.001, 0.05, 0.4, 0.1),       # Punchy bass
    'kick': (0.001, 0.3, 0.0, 0.0),              # Drum kick
    'snare': (0.001, 0.15, 0.0, 0.0),            # Drum snare
    'hihat': (0.001, 0.05, 0.0, 0.0),            # Drum hihat
    'arp_pluck': (0.001, 0.1, 0.0, 0.05),        # Staccato arp
    'pad': (0.5, 0.0, 1.0, 0.5),                 # String-like pad
    'lead_stab': (0.01, 0.1, 0.7, 0.2),          # Lead melody
    'lead_sustained': (0.01, 0.05, 0.9, 0.3),    # Held notes
}
```

### Filter Envelopes

```python
JARRE_FILTER_ENVELOPES = {
    'bass': (0.001, 0.1, 0.2, 0.1),              # Tight bass filter
    'arp_sweep': (0.001, 0.2, 0.5, 0.1),         # Opening filter
    'lead_dramatic': (0.01, 0.25, 0.4, 0.4),     # Sweeping lead
    'pad_minimal': (0.5, 0.0, 1.0, 0.5),         # Static pad filter
}
```

### Mixing Levels

```python
JARRE_MIXING = {
    'bass_level': 0.35,           # Prominent bass
    'kick_level': 0.45,           # Strong kick
    'snare_level': 0.35,          # Clear snare
    'hihat_level': 0.20,          # Subtle hihat
    'arpeggio_level': 0.40,       # Textural arpeggios
    'pad_level': 0.30,            # Background pads
    'lead_level': 0.55,           # Lead melody prominent
    'gesture_level': 0.25,        # Gestures background (not foreground)
}
```

## Pattern Generators

### Sequenced Bass Generator

```python
class JarreSequencedBass:
    """
    Grid-locked bassline generator.
    Always on beat, follows pattern library.
    """

    def generate(self, duration, bpm, scale, pattern_name):
        beat_duration = 60.0 / bpm
        pattern = JARRE_BASS_PATTERNS[pattern_name]

        events = []
        current_time = 0.0

        while current_time < duration:
            for degree in pattern:
                if degree is None:  # Rest
                    current_time += beat_duration
                    continue

                freq = scale[degree % len(scale)]
                if degree >= len(scale):
                    freq *= 2  # Octave up
                elif degree < 0:
                    freq /= 2  # Octave down

                events.append(NoteEvent(
                    freq=freq,
                    duration=beat_duration * 0.8,  # Slight gap
                    timestamp=current_time,
                    waveform='saw',
                    filter_base=400,
                    resonance=0.5,
                    amp_env=JARRE_ENVELOPES['bass_pluck'],
                ))

                current_time += beat_duration

        return events
```

### Drum Pattern Generator

```python
class JarreDrumGenerator:
    """
    Analog drum machine synthesis.
    Kick, snare, hihat from scratch.
    """

    def synthesize_kick(self, duration=0.3):
        # Pitch envelope: 150 Hz → 50 Hz
        samples = int(duration * sample_rate)
        pitch_curve = np.linspace(150, 50, samples)

        signal = np.zeros(samples)
        phase = 0.0
        for i, freq in enumerate(pitch_curve):
            signal[i] = np.sin(2 * np.pi * phase)
            phase += freq / sample_rate
            if phase >= 1.0:
                phase -= 1.0

        # Amplitude envelope
        envelope = self.adsr_envelope(samples, *JARRE_ENVELOPES['kick'])
        return signal * envelope

    def synthesize_snare(self, duration=0.15):
        # 70% white noise + 30% tone at 200 Hz
        samples = int(duration * sample_rate)
        noise = np.random.randn(samples)
        tone = np.sin(2 * np.pi * 200 * np.arange(samples) / sample_rate)

        signal = 0.7 * noise + 0.3 * tone

        # Highpass filter at 3000 Hz
        signal = self.highpass_filter(signal, 3000)

        # Amplitude envelope
        envelope = self.adsr_envelope(samples, *JARRE_ENVELOPES['snare'])
        return signal * envelope

    def synthesize_hihat(self, duration=0.05):
        # Filtered white noise
        samples = int(duration * sample_rate)
        noise = np.random.randn(samples)

        # Highpass filter at 8000 Hz
        signal = self.highpass_filter(noise, 8000)

        # Amplitude envelope
        envelope = self.adsr_envelope(samples, *JARRE_ENVELOPES['hihat'])
        return signal * envelope
```

### Arpeggio Generator

```python
class JarreArpeggioGenerator:
    """
    16th-note arpeggio sequences.
    Classic Jarre texture.
    """

    def generate(self, duration, bpm, scale, pattern_name, octave_range=3):
        sixteenth_duration = 60.0 / (bpm * 4)  # 16th note length
        pattern = JARRE_ARPEGGIO_PATTERNS[pattern_name]

        events = []
        current_time = 0.0

        while current_time < duration:
            for degree in pattern:
                freq = scale[degree % len(scale)]

                # Octave shifts
                octave_shift = degree // len(scale)
                freq *= (2 ** octave_shift)

                events.append(NoteEvent(
                    freq=freq,
                    duration=sixteenth_duration * 0.5,  # Staccato
                    timestamp=current_time,
                    waveform='pulse',
                    filter_base=2000,
                    filter_env_amount=1000,
                    resonance=1.5,
                    amp_env=JARRE_ENVELOPES['arp_pluck'],
                    filter_env=JARRE_FILTER_ENVELOPES['arp_sweep'],
                ))

                current_time += sixteenth_duration

        return events
```

### Pad Chord Generator

```python
class JarrePadGenerator:
    """
    Sustained chord pads.
    Changes every 4-8 beats.
    """

    def generate(self, duration, bpm, scale, chord_type, chord_changes):
        chord_duration = (60.0 / bpm) * 4  # 4 beats per chord

        # Define chord voicings
        chord_voicings = {
            'minor': [0, 2, 4],        # i chord
            'major': [0, 2, 5],        # Major
            'suspended': [0, 3, 5],    # Sus4
            'minor_dim': [0, 2, 3],    # Diminished
            'major_add9': [0, 2, 5, 7],  # Add9
        }

        voicing = chord_voicings.get(chord_type, [0, 2, 4])

        events = []
        for chord_time in np.arange(0, duration, chord_duration):
            for degree in voicing:
                freq = scale[degree]

                events.append(NoteEvent(
                    freq=freq,
                    duration=chord_duration * 1.1,  # Overlap
                    timestamp=chord_time,
                    waveform='triangle',
                    filter_base=1000,
                    resonance=0.3,
                    amp_env=JARRE_ENVELOPES['pad'],
                    filter_env=JARRE_FILTER_ENVELOPES['pad_minimal'],
                ))

        return events
```

## Gesture Archetypes (Layer 3b)

Jarre style uses gestures **sparingly** and **mixed lower**:

```python
JARRE_GESTURE_BIASES = {
    'BRILLIANT': {
        'duration_multiplier': 0.7,      # Shorter (more punctual)
        'attack_multiplier': 0.5,        # Faster attack
        'filter_sweep_speed': 2.0,       # Rapid sweeps
        'mix_level': 0.25,               # Background, not foreground
    },

    'BLUNDER': {
        'duration_multiplier': 0.5,      # Brief
        'use_as_transition': True,       # Transition to darker section
        'mix_level': 0.20,               # Very subtle
    },

    # In Jarre style, most gestures are IMPLIED by arrangement
    # Rather than explicit spectromorphological events
    'prefer_arrangement_over_gestures': True,
}
```

## Stereo Treatment

```python
JARRE_STEREO_CONFIG = {
    'bass_pan': 0.0,                     # Centered (foundation)
    'bass_width': 0.0,                   # Mono

    'kick_pan': 0.0,                     # Centered
    'snare_pan': 0.0,                    # Centered
    'hihat_pan': 0.4,                    # Slightly right (classic)

    'arpeggio_pan': 'alternating',       # Ping-pong between L/R
    'arpeggio_width': 0.8,               # Wide

    'pad_pan': 0.0,                      # Centered
    'pad_width': 0.5,                    # Moderate width

    'lead_pan': -0.2,                    # Slightly left
    'lead_width': 0.3,                   # Narrow (focused)

    'gestures_pan': 0.0,                 # Centered
    'gestures_width': 0.4,               # Moderate
}
```

## Implementation Notes

### What to Build New

🆕 **Completely new components**:
- Drum synthesizer (kick, snare, hihat)
- Sequenced bass generator (grid-locked)
- Arpeggio generator (16th notes)
- Pad chord generator (sustained chords)
- Lead melody generator (phrase-based)

### What to Adapt from Current System

🔧 **Adapt/reuse**:
- Use existing pattern generators as melody sources
- Use synth_engine for all sound generation
- Use existing mixing/stereo utilities
- Use gesture system sparingly (low mix level)

### What Not to Use

❌ **Skip**:
- Long-form drones (use sequenced bass instead)
- Sparse event placement (use grid-locked sequences)
- Minimalist aesthetic (maximize density)

## Validation Criteria

A successful Jarre-style rendering should:

1. **Have a clear pulse** - You can tap your foot to it
2. **Be melodically memorable** - Hummable themes
3. **Sound dense** - Multiple layers always active
4. **Use analog timbres** - Warm saws and pulses
5. **Have rhythmic drive** - Forward momentum
6. **Feature dramatic filter sweeps** - Moog-style resonance
7. **Balance layers clearly** - Bass, drums, melody all audible

## Reference Listening

For parameter tuning, reference these Jarre works:

- **"Oxygène Part 4"**: Sequenced bass + arpeggios + melody
- **"Équinoxe Part 5"**: Rhythmic patterns and pad chords
- **"Magnetic Fields Part 2"**: Driving sequences
- **"Oxygène Part 2"**: Melodic lead with arpeggio texture

## Next Steps

1. Create `JarreRenderer` class implementing this profile
2. Implement drum synthesizer
3. Implement sequenced bass generator
4. Implement arpeggio generator
5. Test with Ding/Gukesh Game 1
6. Compare to Spiegel rendering (should sound completely different)
7. Fine-tune mixing and timing
