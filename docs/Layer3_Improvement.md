# Layer 3 Redesign: IMPLEMENTED ✓

## Final Implementation (2025)

Layer 3 was completely redesigned and split into two independent sub-layers:

### Layer 3a: Heartbeat Sub-Drone ✓ IMPLEMENTED
**Continuous biological pulse providing temporal foundation**

- **Implementation**: `create_heartbeat_cycle()` in `synth_engine.py`
- **Pattern**: LUB-dub cardiac rhythm (sine wave, linear ADSR)
- **Entropy-driven variation**:
  - BPM modulation: 60 BPM (calm) → 100 BPM (anxious)
  - Pitch variation: ±2 semitones around root
  - Timing jitter: High entropy adds subtle randomness
- **Stereo**: Centered (biological constant)
- **Location**: Lines 1885-1973 in `synth_composer.py`

### Layer 3b: Moment Sequencer ✓ IMPLEMENTED
**Event-triggered supersaw arpeggios for key chess moments**

- **Implementation**: Supersaw synthesis with entropy-driven note selection
- **Trigger**: ONLY on key moments (captures, blunders, brilliant moves)
- **No base pattern**: Removed continuous arpeggio (heartbeat handles that)
- **Entropy-driven parameters**:
  - Note selection pools (root-fifth → chromatic)
  - Rhythm variation
  - Filter modulation speed
  - Portamento amount
  - Harmonic density
- **Stereo**: Entropy-driven panning (spatial movement)
- **Location**: Lines 1975-2257 in `synth_composer.py`

## Design Philosophy

The split solved the original problem: Layer 3 needed both:
1. **Continuous presence** → Heartbeat (3a) provides this
2. **Event emphasis** → Moments (3b) provides this

By separating these roles, we achieved:
- Clear separation of concerns
- Independent stereo treatment
- Biological anchor (heartbeat) + dramatic events (moments)
- Reduced complexity in moment sequencer

---

# Original Design Notes (SUPERSEDED - Historical Reference)

The sections below describe the original event-based moment system concept. This was ultimately replaced by the dual-layer approach above.

## Original Concept (Not Implemented)

  Right now (2024 - before redesign):
  - Layer 3 has a fixed voice (supersaw synth) ✓
  - Moments have different musical phrases (patterns) ✓
  - Patterns switch instantly when moments occur ✗
  - No duration control - moments play until the next one arrives ✗
  - No emphasis weighting - score value is calculated but unused ✗

  What was proposed for event-based moments:

  1. Duration allocation (using score):
    - duration = base_duration * (score / max_score)
    - INACCURACY (score=5) → shorter emphasis
    - BLUNDER (score=8) → longer, more prominent
  2. Emphasis/intensity (using score):
    - Mix level: mix_amount = 0.3 + (score/10) * 0.7
    - Filter modulation depth: higher score = more dramatic filter sweeps
    - Volume envelope: higher score = more pronounced attack/sustain
  3. Overlap/blending:
    - Instead of binary pattern switches, crossfade between base pattern and moment pattern
    - Lower score moments blend subtly
    - Higher score moments dominate/override
  4. Minimum audible time:
    - Ensure each moment gets at least one full 16-step cycle (4 beats)
    - Even back-to-back moments must be heard

**Status**: This approach was NOT implemented. Instead, the dual-layer split (3a/3b) was chosen as a cleaner architectural solution.
