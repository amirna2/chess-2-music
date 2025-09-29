"""
SYNTH_NARRATIVE - Narrative Process Classes
Laurie Spiegel-inspired process transformations for chess music
"""

import random
import math
from abc import ABC, abstractmethod


class NarrativeProcess(ABC):
    """
    Abstract base class for Spiegel-style process transformations
    Each process maintains state and evolves over the duration of the piece
    """

    def __init__(self, total_duration: float, total_plies: int):
        self.total_duration = total_duration
        self.total_plies = total_plies
        self.current_time = 0.0

    @abstractmethod
    def update(self, current_time: float, key_moment=None) -> dict:
        """
        Update process state and return transformation parameters

        Args:
            current_time: Current position in the composition (seconds/plies)
            key_moment: Optional key moment occurring at this time

        Returns:
            dict: Transformation parameters to apply to synthesis
        """
        pass

    def get_progress(self) -> float:
        """Get normalized progress through the piece (0.0 to 1.0)"""
        return min(1.0, self.current_time / self.total_duration)


class DefaultProcess(NarrativeProcess):
    """Default process that applies no transformations - preserves existing behavior"""

    def update(self, current_time: float, key_moment=None) -> dict:
        self.current_time = current_time
        return {}  # No transformations


class TumblingDefeatProcess(NarrativeProcess):
    """
    Process for TUMBLING_DEFEAT: gradual deterioration through accumulated mistakes
    Inspired by Spiegel's concept of decay and entropy
    """

    def __init__(self, total_duration: float, total_plies: int):
        super().__init__(total_duration, total_plies)
        self.stability = 1.0  # Starts coherent
        self.error_accumulation = 0.0
        self.tempo_drift = 0.0
        self.pitch_drift = 0.0

    def update(self, current_time: float, key_moment=None) -> dict:
        self.current_time = current_time
        progress = self.get_progress()

        # Each mistake accelerates the decay (compound effect)
        if key_moment and key_moment.get('type') in ['MISTAKE', 'BLUNDER', 'INACCURACY']:
            mistake_weight = {
                'INACCURACY': 0.05,
                'MISTAKE': 0.1,
                'BLUNDER': 0.2
            }.get(key_moment.get('type'), 0.1)

            # Later mistakes have more impact (system already unstable)
            self.error_accumulation += mistake_weight * (1 + progress)

        # Stability decays based on accumulated errors and time
        base_decay = progress * 0.3  # Natural decay over time
        error_decay = self.error_accumulation * progress  # Mistakes accelerate decay
        self.stability = max(0.1, 1.0 - (base_decay + error_decay))

        # Tempo becomes increasingly erratic
        chaos_factor = (1 - self.stability) * 0.02
        self.tempo_drift += random.uniform(-chaos_factor, chaos_factor)
        self.tempo_drift = max(-0.3, min(0.3, self.tempo_drift))  # Clamp

        # Pitch drift increases over time
        self.pitch_drift += random.uniform(-0.5, 0.5) * (1 - self.stability)

        return {
            'pitch_drift_cents': self.pitch_drift * 20,  # Up to 20 cents drift
            'tempo_multiplier': 1.0 + self.tempo_drift,
            'filter_stability': self.stability,  # Affects filter consistency
            'resonance_chaos': (1 - self.stability) * 0.5,  # Add resonance variance
            'note_duration_variance': (1 - self.stability) * 0.2,  # Timing becomes erratic
            'volume_decay': 1.0 - (progress * 0.3)  # Gradual volume reduction
        }


class AttackingMasterpieceProcess(NarrativeProcess):
    """
    Process for ATTACKING_MASTERPIECE: building momentum toward triumph
    Based on positive feedback loops and crescendo
    """

    def __init__(self, total_duration: float, total_plies: int):
        super().__init__(total_duration, total_plies)
        self.momentum = 0.0
        self.brilliance_factor = 0.0

    def update(self, current_time: float, key_moment=None) -> dict:
        self.current_time = current_time
        progress = self.get_progress()

        # Brilliant moves build momentum
        if key_moment and key_moment.get('type') in ['BRILLIANT', 'STRONG']:
            brilliance_weight = {
                'STRONG': 0.15,
                'BRILLIANT': 0.25
            }.get(key_moment.get('type'), 0.15)

            self.momentum += brilliance_weight
            self.brilliance_factor += 0.1

        # Natural crescendo curve (slow start, powerful finish)
        natural_curve = progress ** 1.5  # Exponential growth
        total_momentum = min(1.2, natural_curve + self.momentum * 0.5)

        return {
            'tempo_multiplier': 0.8 + total_momentum * 0.5,  # Speed up
            'filter_brightness': 0.3 + total_momentum * 0.7,  # Open filters
            'resonance_boost': total_momentum * 1.5,  # More dramatic
            'harmonic_density': 0.5 + total_momentum * 0.5,  # Richer harmonies
            'volume_crescendo': 0.7 + total_momentum * 0.3,  # Build volume
            'attack_sharpness': total_momentum  # Crisper note attacks
        }


class QuietPrecisionProcess(NarrativeProcess):
    """
    Process for QUIET_PRECISION: equilibrium-seeking with gentle breathing
    Based on homeostasis and natural oscillation
    """

    def __init__(self, total_duration: float, total_plies: int):
        super().__init__(total_duration, total_plies)
        self.balance = 0.0
        self.breathing_phase = 0.0

    def update(self, current_time: float, key_moment=None) -> dict:
        self.current_time = current_time
        progress = self.get_progress()

        # Small perturbations always return to center
        if key_moment:
            disturbance = 0.05  # Very small disturbances
        else:
            disturbance = 0.0

        # Self-correcting process - always returns to balance
        self.balance = self.balance * 0.95 + disturbance

        # Gentle breathing pattern - slow oscillation
        self.breathing_phase += 0.1
        breath_cycle = math.sin(self.breathing_phase) * 0.08

        return {
            'tempo_regularity': 1.0,  # Metronomic consistency
            'filter_precision': 0.9 - abs(self.balance),  # Very stable
            'dynamic_breathing': breath_cycle,  # Gentle volume waves
            'pitch_stability': 0.95,  # Minimal drift
            'resonance_control': 0.3,  # Tight, controlled
            'harmonic_purity': 0.9  # Clean, simple harmonies
        }


def create_narrative_process(narrative: str, duration: float, plies: int) -> NarrativeProcess:
    """
    Factory function to create appropriate process based on overall narrative

    Args:
        narrative: Overall narrative type (e.g., 'TUMBLING_DEFEAT')
        duration: Total duration in seconds
        plies: Total number of plies

    Returns:
        NarrativeProcess instance
    """
    process_map = {
        'TUMBLING_DEFEAT': TumblingDefeatProcess,
        'ATTACKING_MASTERPIECE': AttackingMasterpieceProcess,
        'QUIET_PRECISION': QuietPrecisionProcess,
        # Aliases for similar behaviors
        'FIGHTING_DEFEAT': TumblingDefeatProcess,
        'TACTICAL_MASTERPIECE': AttackingMasterpieceProcess,
        'PEACEFUL_DRAW': QuietPrecisionProcess,
    }

    ProcessClass = process_map.get(narrative, DefaultProcess)
    return ProcessClass(duration, plies)