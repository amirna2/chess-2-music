"""
Utility functions for Layer 3b gesture generation.

Provides phase computation, audio finalization, and normalization utilities.
"""

import numpy as np
from typing import Dict, Any


def compute_phases(phase_config: Dict[str, float],
                  total_samples: int,
                  section_context: Dict[str, Any],
                  rng: np.random.Generator) -> Dict[str, Dict[str, int]]:
    """
    Compute phase timeline with sample boundaries.

    Args:
        phase_config: {'pre_shadow': 0.15, 'impact': 0.05, 'bloom': 0.40, ...}
        total_samples: Total gesture duration in samples
        section_context: Section-level parameters (for randomization)
        rng: Random generator

    Returns:
        {
            'pre_shadow': {'start_sample': int, 'end_sample': int, 'duration_samples': int},
            'impact': {...},
            'bloom': {...},
            'decay': {...},
            'residue': {...}
        }
    """
    # Randomize phase ratios by ±5%
    phase_names = ['pre_shadow', 'impact', 'bloom', 'decay', 'residue']
    randomized_ratios = {}

    for name in phase_names:
        base_ratio = phase_config.get(name, 0.2)
        variation = rng.uniform(-0.05, 0.05)
        randomized_ratios[name] = max(0.0, base_ratio + variation)

    # Normalize to sum to 1.0
    ratio_sum = sum(randomized_ratios.values())
    for name in phase_names:
        randomized_ratios[name] /= ratio_sum

    # Compute sample boundaries
    phases = {}
    current_sample = 0

    for i, name in enumerate(phase_names):
        duration_samples = int(randomized_ratios[name] * total_samples)

        # Last phase gets remaining samples (to avoid rounding errors)
        if i == len(phase_names) - 1:
            duration_samples = total_samples - current_sample

        phases[name] = {
            'start_sample': current_sample,
            'end_sample': current_sample + duration_samples,
            'duration_samples': duration_samples
        }

        current_sample += duration_samples

    return phases


def finalize_audio(audio: np.ndarray,
                  peak_limit: float,
                  rms_target: float) -> np.ndarray:
    """
    Finalize audio with normalization and safety clipping.

    Args:
        audio: Raw audio buffer
        peak_limit: Maximum peak level (e.g., 0.8)
        rms_target: Target RMS level in dB (e.g., -18.0)

    Returns:
        Finalized audio buffer

    Pipeline: soft_clip → RMS normalize → hard clip
    """
    # Step 1: Soft clip to prevent harsh peaks
    audio = soft_clip(audio, threshold=peak_limit * 1.2)

    # Step 2: RMS normalization
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 1e-10:  # Avoid division by zero
        target_rms_linear = 10 ** (rms_target / 20.0)
        audio *= (target_rms_linear / current_rms)

    # Step 3: Hard clip to absolute maximum
    audio = np.clip(audio, -peak_limit, peak_limit)

    return audio


def soft_clip(audio: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Apply soft clipping (tanh-based).

    Args:
        audio: Input audio
        threshold: Clipping threshold (0-1)

    Returns:
        Soft-clipped audio: threshold * tanh(audio / threshold)
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

    return threshold * np.tanh(audio / threshold)
