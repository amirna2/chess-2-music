# Changelog

All notable changes to the chess-to-music project will be documented in this file.

## [3.3.0] - 2025-10-05

### Added
- **Comprehensive redesign of Layer3 with heartbeat + moments**
  - Layer3a: biological heartbeat with low-pass filter
  - Layer3b: moments sequencer (updated sequence)



## [3.2.0] - 2025-10-04

### Added
- **Comprehensive Debug Output**: Complete synthesis visibility for all layers
  - Layer 1: waveform and detune progression per section
  - Layer 2: Pattern generator debug with waveform distribution, envelopes, and pattern-specific stats
  - Layer 3: Entropy statistics showing complexity range per section
  - Moment events: timing, score-based emphasis, filter modulation, and pattern previews
  - Reusable `print_debug_summary()` method in PatternGenerator base class

### Changed
- Enhanced debug output formatting with hierarchical structure and consistent alignment
- Pattern previews now use zero-padded double digits (e.g., `[00,__,03,02,__,07]`)
- Entropy info moved under Layer 3 header for better organization
- Fixed missing envelope names in FLAWLESS_CONVERSION pattern

## [3.1.0] - 2025-10-04

### Added
- **Layer 3 Event-Based Moment System**: Key moments now have score-based duration and emphasis
  - Moments blend smoothly with crossfades instead of abrupt switches
  - Duration and mix amount scale with moment importance score
  - Minimum 2s audible duration guarantee for all moments
  - New `MOMENT_EVENT_PARAMS` configuration in `synth_config.py`

### Changed
- Layer 3 now maintains continuous supersaw texture with weighted moment blending
- Replaced binary pattern switching with event-based duration allocation

## [3.0.0] - Previous version
