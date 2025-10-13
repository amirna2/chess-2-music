# Archived Documentation

This directory contains **implementation-specific documentation** that was written during development but is not maintained as the codebase evolves.

## Why Archived?

These documents contain:
- Code snippets that quickly become outdated
- Implementation details tied to specific versions
- Patch-based proposals ("replace this with that")
- Low-level technical details better understood by reading the code

## What's Archived

- **emotional_gesture_generator.md**: Early design proposal for gesture system (superceded by layer3b implementation)
- **layer3b_implementation.md**: Implementation guide with code examples (use LAYER3B_COMPLETE_REFERENCE.md instead)
- **SYNTH_ENGINE_ARCHITECTURE.md**: Engine implementation details with code (see synth_engine.py directly)
- **synth_engine_enhancements.md**: Proposed enhancements with code (see actual implementation)
- **Layer3_Improvement.md**: Old improvement proposal
- **PERCUSSION_LAYER_IDEAS.md**: Ideas document (not implemented)

## Current Documentation

For up-to-date documentation, see the main docs directory:

- **README.md**: User-facing guide and quickstart
- **CLAUDE.md**: Project overview and implementation guidelines
- **composer_architecture.md**: High-level system architecture
- **ENTROPY_INTEGRATION.md**: Conceptual explanation of entropy-driven composition
- **LAYER3B_COMPLETE_REFERENCE.md**: Configuration parameter reference (stable)

## Documentation Philosophy

**Good documentation should be:**
- **Conceptual**, not implementation-specific
- **Stable**, not tied to current code structure
- **Explanatory**, focusing on "why" and "what" rather than "how"
- **Maintainable**, without code snippets that rot

These archived docs violated those principles and are kept for historical reference only.
