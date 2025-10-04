# Changelog

All notable changes to the chess-to-music project will be documented in this file.

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
