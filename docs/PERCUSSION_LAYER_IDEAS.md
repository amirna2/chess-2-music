# Percussion Layer for Chess-to-Music
## Rhythmic and Percussive Elements for Chess Events

---

## Why Percussion is Missing

Currently, the three-layer synthesis architecture consists of:
- **Layer 1 (Drone)**: Tonal, sustained, overall narrative
- **Layer 2 (Patterns)**: Melodic/harmonic patterns, section narrative
- **Layer 3 (Sequencer)**: Continuous tonal sequencer, key moments

**Gap:** No percussive/rhythmic layer to mark discrete chess events (captures, checks, tactical exchanges). The music lacks the **rhythmic punctuation** and **percussive impact** that would emphasize critical moments and drive energy through tactical sequences.

---

## Chess Events That Need Percussion

### High Priority
1. **Captures** - Physical impact, pieces leaving the board
2. **Checks** - Sharp, attention-grabbing accents
3. **Tactical sequences** - Rapid exchanges, rhythmic density
4. **Blunders (??)** - Disruptive, broken rhythms
5. **Brilliancies (!!)** - Triumphant crashes/accents

### Medium Priority
6. **Castling** - Unique two-piece movement sound
7. **Pawn promotion** - Transformation moment
8. **Move timing** - Rhythmic pulse based on actual game clock

### Lower Priority
9. **Piece type movement** - Different percussion timbres per piece
10. **Opening theory moves** - Established rhythmic patterns

---

## Proposed Percussion Synthesis Techniques

### 1. Kick Drum (Bass Impact)

**Use Cases:**
- Captures (especially heavy pieces: Queen, Rook)
- King moves in KING_ATTACK sections
- Emphasized moves in CRUSHING_ATTACK narratives

**Synthesis Approach:**
```python
def synthesize_kick(duration=0.3, fundamental=60, sample_rate=44100):
    """
    Synthesize kick drum using sine wave with pitch envelope.

    Classic analog kick: high pitch at attack, rapidly decays to fundamental.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # Pitch envelope: Start at 3x fundamental, decay to fundamental
    pitch_env = fundamental * (1 + 2 * np.exp(-t * 40))

    # Phase accumulation for frequency-modulated sine
    phase = np.cumsum(2 * np.pi * pitch_env / sample_rate)
    kick = np.sin(phase)

    # Amplitude envelope: Fast attack, medium decay
    amp_env = np.exp(-t * 8)

    return kick * amp_env
```

**Parameters to Map:**
- `fundamental`: Piece value (Pawn=80Hz, Queen=40Hz - heavier = deeper)
- `decay_rate`: Position tension (high tension = tighter kick)
- `velocity`: Move quality (blunder = weak, brilliant = strong)

---

### 2. Snare/Noise Burst (Sharp Accent)

**Use Cases:**
- Checks
- Tactical moments (forks, pins, skewers)
- INACCURACY (?!) moves
- Quick tactical exchanges

**Synthesis Approach:**
```python
def synthesize_snare(duration=0.15, tone_freq=200, noise_level=0.7, sample_rate=44100):
    """
    Synthesize snare using tonal component + noise burst.

    Combines pitched resonance with noise for sharp accent.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # Tonal component (resonant body)
    tone = np.sin(2 * np.pi * tone_freq * t)
    tone_env = np.exp(-t * 20)

    # Noise component (snare wires)
    noise = np.random.uniform(-1, 1, samples)
    noise_env = np.exp(-t * 15)

    # Mix
    snare = (tone * tone_env * (1 - noise_level)) + (noise * noise_env * noise_level)

    # Fast attack envelope
    amp_env = np.exp(-t * 12)

    return snare * amp_env
```

**Parameters to Map:**
- `tone_freq`: Square color (white=higher, black=lower)
- `noise_level`: Entropy/complexity (simple position = tonal, complex = noisy)
- `duration`: Move speed (time pressure = shorter)

---

### 3. Hi-Hat/Cymbal (Metallic Accent)

**Use Cases:**
- Brilliancies (!!)
- Checkmate
- Opening theory moves (crisp, precise)
- FLAWLESS_CONVERSION sections

**Synthesis Approach:**
```python
def synthesize_hihat(duration=0.1, brightness=0.5, sample_rate=44100):
    """
    Synthesize hi-hat using filtered noise with metallic character.

    Multiple bandpass filters create metallic shimmer.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # White noise source
    noise = np.random.uniform(-1, 1, samples)

    # Metallic frequencies (inharmonic)
    freqs = [3200, 4500, 5800, 7200, 9000, 11000]
    metallic = np.zeros(samples)

    for freq in freqs:
        # Bandpass filter (simplified - use proper filter in implementation)
        bw = 800 * brightness
        filtered = bandpass_filter(noise, freq, bw, sample_rate)
        metallic += filtered

    # Fast decay
    amp_env = np.exp(-t * 30 * (1 + brightness))

    return metallic * amp_env

def synthesize_crash(duration=1.5, brightness=0.8, sample_rate=44100):
    """
    Longer, richer cymbal crash for major moments.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # Similar to hi-hat but longer decay, more frequencies
    noise = np.random.uniform(-1, 1, samples)

    # Dense inharmonic spectrum
    freqs = np.arange(2000, 16000, 400)
    crash = np.zeros(samples)

    for freq in freqs:
        weight = np.random.uniform(0.3, 1.0)
        filtered = bandpass_filter(noise, freq, 1200, sample_rate)
        crash += filtered * weight

    # Medium decay
    amp_env = np.exp(-t * 3)

    return crash * amp_env
```

**Parameters to Map:**
- `brightness`: Move quality (brilliant = bright, blunder = dark/muted)
- `duration`: Narrative context (MASTERPIECE = longer sustain)

---

### 4. Glitch/Broken Percussion (Disruption)

**Use Cases:**
- Blunders (??)
- Mistakes (?)
- TUMBLING_DEFEAT moments
- Unexpected position changes

**Synthesis Approach:**
```python
def synthesize_glitch(duration=0.4, disruption=0.8, sample_rate=44100):
    """
    Synthesize glitchy, broken percussion for errors.

    Uses buffer manipulation, clicks, pops, and distortion.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # Start with a normal kick
    glitch = synthesize_kick(duration, fundamental=50, sample_rate=sample_rate)

    # Add digital artifacts
    if disruption > 0.5:
        # Buffer glitches (random repeats)
        num_glitches = int(disruption * 8)
        for _ in range(num_glitches):
            glitch_pos = np.random.randint(0, samples - 100)
            glitch_len = np.random.randint(10, 50)
            # Repeat a small section
            glitch[glitch_pos:glitch_pos+glitch_len] = glitch[glitch_pos]

    # Add clicks/pops
    num_clicks = int(disruption * 15)
    click_positions = np.random.randint(0, samples, num_clicks)
    glitch[click_positions] += np.random.uniform(-0.8, 0.8, num_clicks)

    # Distortion/bit crushing
    if disruption > 0.6:
        bit_depth = int(16 * (1 - disruption))
        quantize_levels = 2 ** bit_depth
        glitch = np.round(glitch * quantize_levels) / quantize_levels

    # Decay envelope
    amp_env = np.exp(-t * 5)

    return glitch * amp_env
```

**Parameters to Map:**
- `disruption`: Eval drop magnitude (bigger blunder = more disruption)
- `duration`: Time spent in bad position

---

### 5. Tom/Melodic Percussion (Pitched Drums)

**Use Cases:**
- Pawn advances (pitch rises with rank)
- KING_HUNT sections (building tom pattern)
- Rhythmic phrases in TACTICAL_CHAOS

**Synthesis Approach:**
```python
def synthesize_tom(duration=0.25, pitch=100, tension=0.5, sample_rate=44100):
    """
    Synthesize melodic tom drum with pitch envelope.

    Similar to kick but shorter decay, stays more pitched.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # Less dramatic pitch envelope than kick
    pitch_env = pitch * (1 + 0.5 * np.exp(-t * 15))

    phase = np.cumsum(2 * np.pi * pitch_env / sample_rate)
    tom = np.sin(phase)

    # Add harmonic for more body
    tom += 0.3 * np.sin(2 * phase)

    # Medium decay
    amp_env = np.exp(-t * 7)

    # Optional: Add slight noise for realism
    noise = np.random.uniform(-0.1, 0.1, samples) * np.exp(-t * 10)

    return (tom * amp_env) + noise
```

**Parameters to Map:**
- `pitch`: Board rank (rank 2 = 80Hz, rank 7 = 160Hz)
- `tension`: Section tension (higher = tighter tom)

---

## Percussion Layer Architecture

### Option A: Layer 4 (Discrete Events)
Add percussion as a **fourth layer** that operates independently:

```
Layer 1 (Drone)      : Sustained, overall narrative
Layer 2 (Patterns)   : Melodic/harmonic, section narrative
Layer 3 (Sequencer)  : Continuous tonal sequence, key moments
Layer 4 (Percussion) : Discrete rhythmic events, chess actions
```

**Pros:**
- Clean separation of concerns
- Easy to enable/disable
- Percussion volume independently controllable

**Cons:**
- Adds architectural complexity
- Four-layer mixing may need rebalancing

---

### Option B: Augment Existing Layers
Integrate percussion into existing layers based on context:

- **Layer 2**: Add percussion to pattern generators (TACTICAL_CHAOS gets hi-hats)
- **Layer 3**: Add percussion to key moments (BLUNDER gets glitch, BRILLIANT gets crash)

**Pros:**
- Percussion is contextually integrated
- No new layer needed
- Simpler mixing

**Cons:**
- Less flexibility
- Harder to disable percussion independently

---

### Option C: Hybrid Approach (Recommended)
Use **Layer 4 for major percussion events**, augment existing layers for subtle rhythms:

- **Layer 4**: Captures, checks, blunders, brilliancies (discrete, high-impact)
- **Layer 2/3**: Subtle rhythmic accents, hi-hats, toms (integrated with patterns)

**Pros:**
- Best of both worlds
- High impact moments stand out
- Subtle rhythms support texture
- Flexible mixing control

---

## Chess Event → Percussion Mapping

### Captures
```python
if move.is_capture():
    # Piece value determines kick depth
    captured_value = PIECE_VALUES[captured_piece]
    kick_freq = 80 - (captured_value * 5)  # Queen=40Hz, Pawn=80Hz

    # Position tension affects decay
    decay = 8 + (tension * 12)

    percussion = synthesize_kick(duration=0.3, fundamental=kick_freq, decay=decay)
```

### Checks
```python
if board.is_check():
    # Square color affects snare tone
    tone_freq = 220 if check_square_is_light else 180

    # Entropy affects noise level
    noise_level = min(0.9, entropy * 1.2)

    percussion = synthesize_snare(duration=0.12, tone_freq=tone_freq, noise_level=noise_level)
```

### Brilliancies
```python
if move_has_nag(NAG_BRILLIANT):
    # Masterpiece narrative = brighter, longer crash
    brightness = 0.9 if 'MASTERPIECE' in overall_narrative else 0.7
    duration = 1.8 if 'MASTERPIECE' in overall_narrative else 1.2

    percussion = synthesize_crash(duration=duration, brightness=brightness)
```

### Blunders
```python
if move_has_nag(NAG_BLUNDER):
    # Eval drop determines disruption level
    eval_drop = abs(eval_before - eval_after)
    disruption = min(1.0, eval_drop / 3.0)  # 3.0 eval drop = max disruption

    percussion = synthesize_glitch(duration=0.5, disruption=disruption)
```

### Tactical Sequences
```python
if narrative == 'TACTICAL_CHAOS':
    # Generate rhythmic hi-hat pattern based on move density
    move_density = len(moves_in_section) / section_duration
    hihat_rhythm = generate_tactical_hihat_pattern(density=move_density, tension=tension)

    # Layer with existing patterns
    percussion = hihat_rhythm
```

---

## Rhythmic Patterns

### Time-Based Patterns
Map actual game clock times to rhythmic grid:

```python
def generate_rhythm_from_timing(moves_with_timestamps, section_duration):
    """
    Convert chess move timestamps into rhythmic grid.

    Quantize to 16th notes, creating organic but structured rhythm.
    """
    rhythm_grid = np.zeros(64)  # 64 steps = 4 bars of 16th notes

    for move in moves_with_timestamps:
        # Map timestamp to grid position
        position = (move.timestamp / section_duration) * 64
        quantized_pos = int(round(position / 4) * 4)  # Quantize to 16th

        if quantized_pos < 64:
            # Piece type determines percussion voice
            rhythm_grid[quantized_pos] = get_percussion_voice(move.piece)

    return rhythm_grid
```

### Narrative-Driven Patterns

**KING_HUNT:**
```
Tom pattern building in intensity:
Bar 1: |K...|....|K...|....|  (sparse, ominous)
Bar 2: |K...|K...|K...|K...|  (building)
Bar 3: |K.K.|K.K.|K.K.|K.K.|  (intensifying)
Bar 4: |KKKK|KKKK|KKKK|KKKK|  (relentless)
```

**TACTICAL_CHAOS:**
```
Irregular hi-hat bursts:
|h..h|.hh.|...h|hh..|  (unpredictable)
```

**FLAWLESS_CONVERSION:**
```
Precise, metronomic:
|h.h.|h.h.|h.h.|h.h.|  (mechanical precision)
```

---

## Integration with Spectral Processing

From `SPECTRAL_PROCESSING_IDEAS.md`, these techniques could enhance percussion:

### 1. Spectral Flux Enhancement
Detect and emphasize percussion transients:
```python
# After synthesizing percussion event
percussion_enhanced = enhance_spectral_flux(percussion, flux_threshold=0.5)
```

### 2. Spectral Tilt on Crashes
Brilliant moves = bright crashes, blunders = dark crashes:
```python
crash = synthesize_crash(duration=1.5)
if move_has_nag(NAG_BRILLIANT):
    crash = apply_spectral_tilt(crash, tension=0.95)  # Bright
elif move_has_nag(NAG_BLUNDER):
    crash = apply_spectral_tilt(crash, tension=0.15)  # Dark
```

### 3. Inharmonicity on Blunders
Add spectral roughness to glitch sounds:
```python
glitch = synthesize_glitch(duration=0.5, disruption=0.8)
glitch = add_inharmonicity(glitch, inharmonicity_amount=disruption * 100, fundamental_freq=50)
```

---

## Mixing and Level Control

### Percussion Levels (Relative to Existing Layers)
```python
MIXING = {
    'drone_level': 0.3,           # Layer 1
    'pattern_level': 0.4,         # Layer 2
    'sequencer_level': 0.5,       # Layer 3
    'moment_level': 0.8,          # Existing key moments

    # NEW: Percussion levels
    'percussion_kick_level': 0.6,      # Bass impacts
    'percussion_snare_level': 0.5,     # Sharp accents
    'percussion_hihat_level': 0.3,     # Subtle rhythms
    'percussion_crash_level': 0.8,     # Major moments
    'percussion_glitch_level': 0.7,    # Disruptions
}
```

### Sidechain Compression
Duck other layers during percussion hits for clarity:
```python
def apply_sidechain(audio, percussion, amount=0.3):
    """
    Duck audio when percussion hits occur.

    Creates rhythmic pumping, gives percussion space in mix.
    """
    # Detect percussion envelope
    perc_env = np.abs(percussion)

    # Smooth envelope
    perc_env_smooth = scipy.ndimage.gaussian_filter1d(perc_env, sigma=100)

    # Invert and scale
    duck_env = 1 - (perc_env_smooth * amount)
    duck_env = np.clip(duck_env, 1 - amount, 1.0)

    return audio * duck_env
```

---

## Implementation Priority

### Phase 1: Core Percussion Events (Week 1)
1. Kick synthesis for captures
2. Snare synthesis for checks
3. Crash synthesis for brilliancies/checkmate
4. Glitch synthesis for blunders
5. Basic Layer 4 architecture and mixing

**Goal:** Discrete chess events have percussive punctuation

### Phase 2: Rhythmic Patterns (Week 2)
6. Hi-hat patterns for tactical sections
7. Tom patterns for KING_HUNT
8. Time-based rhythm generation from game clock
9. Narrative-driven pattern templates

**Goal:** Rhythmic texture supports section narratives

### Phase 3: Spectral Enhancement (Week 3)
10. Integrate spectral tilt for brightness control
11. Add spectral flux enhancement for transients
12. Implement inharmonicity for blunder glitches
13. Sidechain compression for mix clarity

**Goal:** Percussion is polished and integrated with spectral system

### Phase 4: Refinement (Week 4+)
14. Piece-specific percussion timbres
15. Castling/promotion special sounds
16. Dynamic mixing based on percussion density
17. Stereo imaging (white=left, black=right)

**Goal:** Professional-quality percussion layer

---

## Open Questions

1. **Should percussion follow exact move timing or quantized rhythm?**
   - Exact timing = organic, human feel
   - Quantized = musical, danceable groove
   - Hybrid = quantize to nearest 16th note?

2. **How dense should percussion be in tactical sequences?**
   - Every move = exhausting
   - Key moves only = sparse
   - Algorithmic thinning based on move density?

3. **Should percussion be stereo-positioned like other layers?**
   - White pieces = left percussion
   - Black pieces = right percussion
   - Center for mutual events (checkmate)?

4. **How to handle rapid captures (multiple in quick succession)?**
   - Layer percussion hits
   - Create rolls/flams
   - Synthesize as a single dense texture?

5. **Should there be a "percussion-free" mode for certain narratives?**
   - PEACEFUL_DRAW = minimal percussion
   - QUIET_PRECISION = only subtle hi-hats
   - CRUSHING_ATTACK = maximum percussion density

---

## Conclusion

Adding a percussion layer would significantly enhance the musical impact of the chess-to-music system by:

1. **Marking discrete events** (captures, checks, brilliant moves)
2. **Driving rhythmic energy** (tactical sequences, time pressure)
3. **Emphasizing emotional moments** (blunders as disruption, brilliancies as triumph)
4. **Creating structural clarity** (rhythmic patterns define sections)
5. **Integrating with spectral processing** (brightness, inharmonicity, flux)

**Recommended approach:** Start with Phase 1 (core percussion events) to validate the concept, then expand to rhythmic patterns and spectral integration.

This would bring the system closer to the musical sophistication of Laurie Spiegel's work, where **every data dimension informs the sonic result** - including the discrete, percussive nature of chess moves themselves.
