  Right now:
  - Layer 3 has a fixed voice (supersaw synth) ✓
  - Moments have different musical phrases (patterns) ✓
  - Patterns switch instantly when moments occur ✗
  - No duration control - moments play until the next one arrives ✗
  - No emphasis weighting - score value is calculated but unused ✗

  What's needed for event-based moments:

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
