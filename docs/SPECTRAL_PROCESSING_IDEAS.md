# Spectral Processing for Chess-to-Music
## FFT-Based Effects and Applications

---

## What is Spectral Processing?

**Spectral processing** uses FFT (Fast Fourier Transform) to work in the **frequency domain** rather than the time domain. Instead of manipulating audio samples directly, you transform the audio into its frequency components, manipulate individual frequency bins, and transform back.

**Key Concept:** Every sound can be decomposed into a spectrum of frequencies. By working directly with this spectrum, you can achieve effects impossible in the time domain.

---

## Potential Applications for Chess-to-Music

### 1. Spectral Morphing Between Sections

**Problem:** Section transitions (even with crossfades) can sound abrupt when the harmonic content changes drastically. OPENING might be sparse and low, while MIDDLEGAME is dense and bright - the crossfade helps with amplitude but not with spectral character.

**Solution:** **Spectral morphing** - gradually transform the frequency spectrum from one section to the next, creating a smooth timbral transition.

**Implementation:**
```python
# At crossfade point between sections
fft_section1 = np.fft.rfft(section1_audio)
fft_section2 = np.fft.rfft(section2_audio)

# Interpolate in frequency domain
for alpha in np.linspace(0, 1, crossfade_samples):
    morphed_fft = fft_section1 * (1 - alpha) + fft_section2 * alpha
    morphed_audio[i] = np.fft.irfft(morphed_fft)
```

**Musical Result:**
- Smooth timbral evolution, not just volume crossfade
- Dark opening naturally brightens into tactical middlegame
- Spectral complexity increases/decreases smoothly

**Where to Apply:**
- Section crossfades (OPENING → MIDDLEGAME → ENDGAME)
- Overall narrative evolution (TUMBLING_DEFEAT's gradual spectral darkening)

**Priority:** ⭐⭐⭐⭐⭐ (High value, moderate complexity)

---

### 2. Tension-Driven Spectral Tilt

**Problem:** Current tension only affects filter cutoff frequency. The relationship between tension and brightness could be more nuanced and perceptually accurate.

**Solution:** Apply **spectral tilt** - high tension boosts high frequencies (bright, edgy), low tension boosts low frequencies (dark, ominous).

**Implementation:**
```python
# Calculate spectral tilt based on tension
freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
max_freq = sample_rate / 2

# Tension = 0: boost lows, Tension = 1: boost highs
tilt_exponent = (tension - 0.5) * 4  # -2 to +2 range
tilt_curve = (freqs / max_freq) ** tilt_exponent

# Apply to spectrum
fft_audio = np.fft.rfft(audio)
fft_audio *= tilt_curve
audio_tilted = np.fft.irfft(fft_audio)
```

**Musical Result:**
- Low tension sections: Warm, bass-heavy, grounded
- High tension sections: Bright, trebly, aggressive
- More natural than simple filter cutoff
- Affects entire spectral balance, not just cutoff point

**Where to Apply:**
- Layer 1 drone evolution across sections
- TACTICAL_CHAOS sections (extreme high tilt)
- QUIET_PRECISION sections (moderate low tilt)

**Priority:** ⭐⭐⭐⭐ (High value, low complexity - easy win!)

---

### 3. Position Entropy → Spectral Complexity

**Problem:** Currently entropy affects note selection and rhythm. But entropy could also affect **timbre** directly - the sonic texture itself.

**Solution:** Map position entropy to spectral complexity:
- **Low entropy (simple position)**: Pure, harmonic tones
- **High entropy (complex position)**: Add spectral noise, inharmonic partials, roughness

**Implementation:**
```python
# Add spectral noise/roughness based on entropy
fft_audio = np.fft.rfft(audio)

if entropy > 0.7:  # High complexity
    # Select random frequency bins
    num_noise_bins = int(entropy * len(fft_audio) * 0.3)
    noise_bins = np.random.choice(len(fft_audio), num_noise_bins)

    # Add noise to selected bins
    noise_amplitude = entropy * 0.5
    fft_audio[noise_bins] += np.random.randn(len(noise_bins)) * noise_amplitude

    # Add inharmonic partials
    for harmonic in range(1, 8):
        # Slightly detuned harmonics
        freq_offset = np.random.uniform(-0.1, 0.1) * harmonic
        # Add to spectrum with entropy-based amplitude

audio_complex = np.fft.irfft(fft_audio)
```

**Musical Result:**
- Simple positions: Clean, pure tones
- Complex positions: Gritty, textured, unstable timbres
- Directly hear the "computational difficulty" as sonic roughness
- Perfect implementation of Spiegel's entropy → music mapping

**Where to Apply:**
- Layer 3 sequencer (entropy-driven note synthesis)
- TACTICAL_CHAOS sections (maximum spectral noise)
- COMPLEX_STRUGGLE sections (moderate roughness)

**Priority:** ⭐⭐⭐⭐⭐ (High value, high complexity - deep Spiegel integration)

---

### 4. Resonance/Formant Shaping for Piece Types

**Problem:** Currently pieces map to different MIDI instruments, but they all use the same synthesis engine. They could have more distinct timbral identities.

**Solution:** Give each piece type a unique **spectral signature** using formant-like resonances (similar to vowel sounds in speech).

**Implementation:**
```python
# Define spectral signatures for each piece
PIECE_FORMANTS = {
    'PAWN': {
        'peaks': [200, 400],      # Simple, narrow
        'bandwidth': 50,
        'character': 'Simple, focused'
    },
    'KNIGHT': {
        'peaks': [300, 900, 2400], # Irregular harmonics
        'bandwidth': 150,
        'character': 'Jumping, unpredictable'
    },
    'BISHOP': {
        'peaks': [500, 1500, 3000], # Diagonal spread
        'bandwidth': 100,
        'character': 'Bright, diagonal'
    },
    'ROOK': {
        'peaks': [100, 200, 400, 800], # Regular harmonics
        'bandwidth': 80,
        'character': 'Powerful, linear'
    },
    'QUEEN': {
        'peaks': [200, 600, 1200, 2400, 4000], # Complex, many formants
        'bandwidth': 120,
        'character': 'Rich, powerful'
    },
    'KING': {
        'peaks': [80, 160, 320], # Deep, few harmonics
        'bandwidth': 60,
        'character': 'Grave, dignified'
    }
}

def apply_formants(audio, piece_type):
    fft_audio = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

    formants = PIECE_FORMANTS[piece_type]
    formant_response = np.ones_like(fft_audio)

    for peak_freq in formants['peaks']:
        # Gaussian resonance peak
        resonance = np.exp(-((freqs - peak_freq) ** 2) / (2 * formants['bandwidth'] ** 2))
        formant_response += resonance * 2.0  # Boost factor

    fft_audio *= formant_response
    return np.fft.irfft(fft_audio)
```

**Musical Result:**
- Each piece has a distinct "voice"
- Pawn: Simple, pure
- Queen: Rich, complex
- King: Deep, grave
- More expressive than just different instruments

**Where to Apply:**
- Could replace current piece-to-instrument mapping entirely
- Or augment it (MIDI instrument + spectral signature)

**Priority:** ⭐⭐⭐ (Interesting but lower priority - current system works)

---

### 5. Spectral Freeze for "Suspended" Moments

**Problem:** Critical decisions and time pressure need a sense of "frozen time" or "suspended animation" - the musical equivalent of a player staring at the board.

**Solution:** **Spectral freeze** - capture the spectrum at a critical moment and hold it static while time passes, creating an eerie, suspended quality.

**Implementation:**
```python
def spectral_freeze(audio, freeze_duration):
    # Capture spectrum at freeze point
    frozen_fft = np.fft.rfft(audio[:window_size])
    frozen_magnitudes = np.abs(frozen_fft)

    # Generate frozen audio with evolving phase
    frozen_audio = np.zeros(int(freeze_duration * sample_rate))

    for i in range(0, len(frozen_audio), hop_size):
        # Keep magnitudes constant, evolve phase slowly
        random_phase = np.random.uniform(-np.pi, np.pi, len(frozen_magnitudes))
        frozen_frame = frozen_magnitudes * np.exp(1j * random_phase)

        # IFFT and overlap-add
        time_frame = np.fft.irfft(frozen_frame)
        frozen_audio[i:i+len(time_frame)] += time_frame

    return frozen_audio
```

**Musical Result:**
- Time seems to stop, but sound continues
- Eerie, suspended quality
- Perfect for CRITICAL_DECISIONS moments
- Like a freeze-frame but for sound

**Where to Apply:**
- CRITICAL_DECISIONS narrative sections
- Moments with very long thinking times (>10 minutes)
- TENSE_EQUILIBRIUM sections
- Before BLUNDER moments (freeze → release)

**Priority:** ⭐⭐⭐⭐ (Unique effect, moderate complexity)

---

### 6. Harmonic/Inharmonic Balance for Position Evaluation

**Problem:** Good positions vs bad positions could have distinct timbral qualities beyond just brightness/darkness.

**Solution:** Control the **harmonic purity** of the sound:
- **Good position (stable, clear)**: Pure harmonic spectrum (perfect overtone series)
- **Bad position (unstable, unclear)**: Add inharmonic partials (stretched/compressed harmonics)

**Implementation:**
```python
def add_inharmonicity(audio, inharmonicity_amount, fundamental_freq):
    fft_audio = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

    # Generate stretched harmonic series
    # Normal harmonics: f, 2f, 3f, 4f...
    # Stretched: f, 2.1f, 3.3f, 4.6f...
    for n in range(1, 16):  # First 16 harmonics
        # Stretch factor increases with harmonic number
        stretch = 1 + (n ** 2) * inharmonicity_amount * 0.001
        inharmonic_freq = fundamental_freq * n * stretch

        # Find nearest bin
        bin_idx = int(inharmonic_freq * len(audio) / sample_rate)
        if bin_idx < len(fft_audio):
            # Add inharmonic component
            fft_audio[bin_idx] += fft_audio[bin_idx] * 0.3

    return np.fft.irfft(fft_audio)
```

**Musical Result:**
- **ATTACKING_MASTERPIECE**: Pure, clean harmonics (triumph)
- **TUMBLING_DEFEAT**: Increasingly inharmonic (disintegration)
- **PEACEFUL_DRAW**: Balanced harmonic structure
- Subtle but profound timbral evolution

**Where to Apply:**
- Overall narrative evolution
- TUMBLING_DEFEAT: Gradual increase in inharmonicity
- ATTACKING_MASTERPIECE: Maintain pure harmonics
- Layer 1 drone character

**Priority:** ⭐⭐⭐⭐⭐ (High value, moderate complexity - great for narrative)

---

### 7. Spectral Centroid Tracking for "Brightness Evolution"

**Problem:** Current filter sweeps are linear in frequency, but perceptual brightness is nonlinear. A sweep from 100Hz to 200Hz sounds more dramatic than 10kHz to 10.1kHz.

**Solution:** Control **spectral centroid** (the "center of mass" of the spectrum) which correlates directly with perceived brightness.

**Implementation:**
```python
def calculate_spectral_centroid(fft_audio, freqs):
    magnitudes = np.abs(fft_audio)
    centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
    return centroid

def shift_to_target_centroid(audio, target_centroid):
    fft_audio = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

    current_centroid = calculate_spectral_centroid(fft_audio, freqs)
    shift_factor = target_centroid / current_centroid

    # Shift all frequencies by factor (pitch shifting in frequency domain)
    # This is simplified - proper implementation uses phase vocoder
    shifted_fft = np.zeros_like(fft_audio)
    for i, freq in enumerate(freqs):
        new_freq = freq * shift_factor
        new_bin = int(new_freq * len(audio) / sample_rate)
        if new_bin < len(shifted_fft):
            shifted_fft[new_bin] += fft_audio[i]

    return np.fft.irfft(shifted_fft)
```

**Musical Result:**
- Perceptually linear brightness evolution
- ATTACKING_MASTERPIECE: Rising centroid (triumph)
- TUMBLING_DEFEAT: Falling centroid (despair)
- More musical than simple filter cutoff

**Where to Apply:**
- Overall narrative brightness evolution
- Section-to-section brightness mapping
- Alternative to current filter sweep

**Priority:** ⭐⭐⭐ (Refinement of existing system)

---

### 8. Spectral Flux for Attack Detection/Enhancement

**Problem:** Captures, checks, and tactical moments could be more pronounced and punchy.

**Solution:** Measure **spectral flux** (rate of spectral change) and enhance transients during high-flux moments.

**Implementation:**
```python
def calculate_spectral_flux(audio, window_size=2048):
    flux = []
    for i in range(0, len(audio) - window_size, window_size // 2):
        frame1 = np.fft.rfft(audio[i:i+window_size])
        frame2 = np.fft.rfft(audio[i+window_size//2:i+3*window_size//2])

        # Flux = sum of magnitude differences
        flux_value = np.sum(np.abs(np.abs(frame2) - np.abs(frame1)))
        flux.append(flux_value)

    return np.array(flux)

def enhance_attacks(audio, flux_threshold):
    flux = calculate_spectral_flux(audio)

    # Find high-flux moments (attacks)
    attacks = flux > flux_threshold

    # Enhance spectral brightness at attack points
    for attack_idx in np.where(attacks)[0]:
        sample_idx = attack_idx * window_size // 2
        # Apply brief spectral tilt boost
        enhance_region = audio[sample_idx:sample_idx+1024]
        # ... spectral enhancement
```

**Musical Result:**
- Captures and checks sound more impactful
- Automatic emphasis on tactical moments
- More "punch" in CRUSHING_ATTACK sections

**Where to Apply:**
- CRUSHING_ATTACK narrative
- KING_HUNT sections
- Capture-heavy tactical battles

**Priority:** ⭐⭐⭐ (Nice enhancement, moderate complexity)

---

## Priority Rankings

### Tier 1: High Value, Should Implement
1. **Spectral morphing for section transitions** ⭐⭐⭐⭐⭐
   - Immediate improvement to crossfades
   - Moderate complexity
   - Clear musical benefit

2. **Tension-driven spectral tilt** ⭐⭐⭐⭐⭐
   - Low complexity, high impact
   - Natural tension → brightness mapping
   - Easy to implement

3. **Harmonic/inharmonic balance** ⭐⭐⭐⭐⭐
   - Perfect for overall narrative distinction
   - DEFEAT vs MASTERPIECE clarity
   - Moderate complexity

### Tier 2: High Value, More Complex
4. **Position entropy → spectral complexity** ⭐⭐⭐⭐⭐
   - Deep Laurie Spiegel integration
   - Sophisticated but requires care
   - High musical payoff

5. **Spectral freeze for critical moments** ⭐⭐⭐⭐
   - Unique effect
   - Perfect for time pressure
   - Moderate complexity

### Tier 3: Interesting Refinements
6. **Spectral centroid tracking** ⭐⭐⭐
   - Refinement of existing filter system
   - More perceptually accurate

7. **Spectral flux enhancement** ⭐⭐⭐
   - Tactical moment emphasis
   - Nice polish

8. **Formant shaping for pieces** ⭐⭐⭐
   - Interesting but current system works
   - Lower priority

---

## Implementation Considerations

### Pros
✅ More sophisticated timbral control
✅ Effects impossible in time domain
✅ Direct perceptual parameter control
✅ Could make narratives MORE distinct
✅ Aligns with Spiegel's algorithmic approach

### Cons
❌ CPU intensive (FFT operations)
❌ Phase coherence issues require care
❌ Risk of "digital" artifacts if overused
❌ Adds complexity to already sophisticated system
❌ Longer processing time

### Technical Requirements
- `scipy.fft` or `numpy.fft` (already available)
- Windowing functions (Hann, Hamming)
- Overlap-add for synthesis
- Phase vocoder for pitch shifting
- Careful normalization to prevent clipping

---

## Recommended Implementation Path

### Phase 1: Quick Wins (Week 1-2)
Implement **Tier 1** features:
1. Start with **tension-driven spectral tilt** (easiest)
2. Add **spectral morphing for crossfades**
3. Implement **harmonic/inharmonic balance** for narratives

**Goal:** Noticeable improvement with manageable complexity

### Phase 2: Deep Integration (Week 3-4)
Add **Tier 2** features:
4. **Entropy → spectral complexity** for Layer 3
5. **Spectral freeze** for critical moments

**Goal:** Unique effects that distinguish this project

### Phase 3: Polish (Week 5+)
Consider **Tier 3** refinements if needed:
6. Centroid tracking, flux enhancement, formants

**Goal:** Final polish and optimization

---

## Example: Spectral Tilt Implementation

Here's a complete, ready-to-use implementation for **tension-driven spectral tilt**:

```python
def apply_spectral_tilt(audio, tension, sample_rate=44100):
    """
    Apply spectral tilt based on tension value.

    Args:
        audio: Input audio array
        tension: 0.0 to 1.0 (0 = dark, 1 = bright)
        sample_rate: Audio sample rate

    Returns:
        Tilted audio array
    """
    # Transform to frequency domain
    fft_audio = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    max_freq = sample_rate / 2

    # Normalize frequencies to 0-1
    normalized_freqs = freqs / max_freq

    # Calculate tilt exponent
    # tension = 0.5: neutral (exponent = 0)
    # tension = 0.0: boost lows (exponent = -2)
    # tension = 1.0: boost highs (exponent = +2)
    tilt_exponent = (tension - 0.5) * 4

    # Create tilt curve
    # Avoid division by zero at DC (freq = 0)
    tilt_curve = np.ones_like(normalized_freqs)
    nonzero_mask = normalized_freqs > 0
    tilt_curve[nonzero_mask] = normalized_freqs[nonzero_mask] ** tilt_exponent

    # Apply tilt
    fft_audio *= tilt_curve

    # Transform back to time domain
    audio_tilted = np.fft.irfft(fft_audio)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio_tilted))
    if max_val > 0:
        audio_tilted *= (np.max(np.abs(audio)) / max_val)

    return audio_tilted


# Usage example in Layer 1 drone evolution
def create_evolving_drone_with_spectral_tilt(self, ...):
    # ... existing drone synthesis ...

    # Apply spectral tilt based on section tension
    drone_audio = apply_spectral_tilt(drone_audio, tension, self.sample_rate)

    return drone_audio
```

---

## Conclusion

Spectral processing could significantly enhance the chess-to-music system by:

1. **Improving transitions** (spectral morphing)
2. **Deepening narrative distinction** (harmonic/inharmonic balance)
3. **Refining tension mapping** (spectral tilt)
4. **Integrating entropy more deeply** (spectral complexity)
5. **Adding unique effects** (spectral freeze)

**Recommended starting point:** Implement spectral tilt first (easiest, immediate impact), then spectral morphing (big quality improvement), then harmonic/inharmonic balance (narrative depth).

All of these align perfectly with Laurie Spiegel's vision of algorithmic composition driven by meaningful data (chess position → spectral characteristics).
