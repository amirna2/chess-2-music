#!/usr/bin/env python3
"""
SUBTRACTIVE SYNTHESIZER FOR CHESS MUSIC
Electronic music synthesis with filters and envelopes
Refactored for clarity and maintainability
"""

import json
import wave
import struct
import sys
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Import our refactored modules
from synth_config import SynthConfig, get_envelope, get_filter_envelope, get_narrative_params, get_section_modulation
from synth_engine import SubtractiveSynth
from synth_narrative import create_narrative_process
from entropy_calculator import ChessEntropyCalculator
from synth_composer import PatternCoordinator


# === STEREO UTILITIES ===

def apply_dynamic_pan(mono_signal, pan_curve, width=0.8):
    """
    Apply time-varying panning to mono signal.

    Args:
        mono_signal: 1D numpy array (mono)
        pan_curve: 1D numpy array same length as signal, values -1.0 to 1.0
        width: Stereo width amount

    Returns:
        (N, 2) stereo array with dynamic panning
    """
    stereo = np.zeros((len(mono_signal), 2))

    for i in range(len(mono_signal)):
        pan = pan_curve[i]
        # Constant power panning
        pan_angle = (pan + 1.0) * np.pi / 4
        left_gain = np.cos(pan_angle)
        right_gain = np.sin(pan_angle)

        stereo[i, 0] = mono_signal[i] * left_gain
        stereo[i, 1] = mono_signal[i] * right_gain

    return stereo


def pan_mono_to_stereo(mono_signal, pan_position):
    """
    Convert mono signal to stereo with panning.

    Args:
        mono_signal: 1D numpy array (mono)
        pan_position: -1.0 (full left) to 1.0 (full right), 0.0 = center

    Returns:
        (N, 2) numpy array with [left, right] channels
    """
    # Equal power panning law (constant energy)
    pan_angle = (pan_position + 1.0) * np.pi / 4  # Map -1..1 to 0..π/2
    left_gain = np.cos(pan_angle)
    right_gain = np.sin(pan_angle)

    stereo = np.zeros((len(mono_signal), 2))
    stereo[:, 0] = mono_signal * left_gain   # Left channel
    stereo[:, 1] = mono_signal * right_gain  # Right channel
    return stereo


def stereo_width(mono_signal, width, center_pan=0.0):
    """
    Create stereo from mono using Haas effect + panning.

    Args:
        mono_signal: 1D numpy array (mono)
        width: 0.0 (mono) to 1.0 (wide stereo)
        center_pan: -1.0 (left) to 1.0 (right)

    Returns:
        (N, 2) numpy array with [left, right] channels
    """
    stereo = np.zeros((len(mono_signal), 2))

    # Haas effect: delay one channel slightly for width
    delay_samples = int(width * 40)  # Up to 40 samples delay (~0.45ms at 88.2kHz)

    if delay_samples == 0:
        stereo[:, 0] = mono_signal
        stereo[:, 1] = mono_signal
    else:
        stereo[:, 0] = mono_signal
        stereo[delay_samples:, 1] = mono_signal[:-delay_samples]

    # Apply panning on top for L/R positioning
    # Constant power pan law
    pan_angle = (center_pan + 1.0) * np.pi / 4  # Map -1..1 to 0..π/2
    left_gain = np.cos(pan_angle)
    right_gain = np.sin(pan_angle)

    stereo[:, 0] *= left_gain
    stereo[:, 1] *= right_gain

    return stereo


def mix_stereo(layers):
    """
    Mix multiple stereo layers.

    Args:
        layers: List of (N, 2) numpy arrays

    Returns:
        (N, 2) mixed stereo array
    """
    if not layers:
        return np.zeros((0, 2))

    max_len = max(len(layer) for layer in layers)
    mixed = np.zeros((max_len, 2))

    for layer in layers:
        mixed[:len(layer)] += layer

    return mixed


class ChessSynthComposer:
    def __init__(self, chess_tags, config=None):
        """
        Initialize chess music composer

        Args:
            chess_tags: Dictionary with game data and narratives
            config: Optional SynthConfig instance for parameter overrides
        """
        self.tags = chess_tags
        self.config = config or SynthConfig()
        self.sample_rate = self.config.SAMPLE_RATE

        # Create dedicated RNG seeded with ECO code for reproducibility
        self.rng = self._create_rng_from_eco(chess_tags.get('eco', 'A00'))

        # Separate synth instances per layer for state isolation
        # Prevents phase/crossfade state pollution between layers
        # Pass shared RNG for reproducibility
        self.synth_layer1 = SubtractiveSynth(self.sample_rate, self.rng)  # Drone layer
        self.synth_layer2 = SubtractiveSynth(self.sample_rate, self.rng)  # Pattern layer
        self.synth_layer3 = SubtractiveSynth(self.sample_rate, self.rng)  # Sequencer layer

        self.total_duration = chess_tags.get('duration_seconds', 60)
        self.total_plies = chess_tags.get('total_plies', 40)

        # Store last section's Layer 3 envelope for outro continuity
        self.last_layer3_amp_env = self.config.SEQUENCER_SYNTH['amp_env']
        self.last_layer3_filter_env = self.config.SEQUENCER_SYNTH['filter_env']
        self.overall_narrative = chess_tags.get('overall_narrative', 'COMPLEX_GAME')
        self.eco = chess_tags.get('eco', 'A00')

        # LAYER 1: Overall narrative defines the BASE PATCH
        self.base_params = self._get_narrative_base_params()

        # Initialize narrative process
        self.narrative_process = create_narrative_process(
            self.overall_narrative,
            self.total_duration,
            self.total_plies,
            self.rng
        )

        # Initialize entropy calculator (Laurie Spiegel-inspired)
        # This requires move data with eval information
        moves = chess_tags.get('moves', [])
        self.entropy_calculator = ChessEntropyCalculator(moves) if moves else None

        # Initialize refactored pattern coordinator for Layer 2
        self.pattern_coordinator = PatternCoordinator(
            sample_rate=self.sample_rate,
            config=self.config,
            synth_engine=self.synth_layer2,
            rng=self.rng
        )

    def _create_rng_from_eco(self, eco_code):
        """
        Create dedicated numpy RNG seeded from ECO code for reproducible randomness.

        Uses modern numpy.random.Generator (not legacy global seed) for:
        - Isolation: Won't interfere with other code using np.random
        - Reproducibility: Same ECO code always produces same music
        - Thread-safety: Each composer has its own RNG state

        ECO format: Letter (A-E) + two digits (00-99)
        Converts to seed: A00=0, A01=1, ..., E99=599
        """
        if len(eco_code) >= 3:
            letter_value = ord(eco_code[0].upper()) - ord('A')  # 0-4
            number_value = int(eco_code[1:3])  # 0-99
            seed = letter_value * 100 + number_value
        else:
            seed = 0  # Fallback

        return np.random.default_rng(seed)

    def _get_narrative_base_params(self):
        """LAYER 1: Overall narrative sets the fundamental synth character"""
        return get_narrative_params(self.overall_narrative, self.config)

    def interpolate_base_params(self, progress):
        """Get current base parameters based on progress through game"""
        base = self.base_params
        return {
            'waveform': base['base_waveform'],
            'filter': base['filter_start'] + (base['filter_end'] - base['filter_start']) * progress,
            'resonance': base['resonance_start'] + (base['resonance_end'] - base['resonance_start']) * progress,
            'tempo': base['tempo_start'] + (base['tempo_end'] - base['tempo_start']) * progress,
            'detune': base['detune_start'] + (base['detune_end'] - base['detune_start']) * progress,
            'scale': self.config.SCALES[base['scale']],
        }

    def _get_section_modulation(self, section_narrative, tension):
        """LAYER 2: Section narrative modulates the base parameters"""
        modulation = get_section_modulation(section_narrative, self.config).copy()

        # Apply tension factor to resonance
        if 'resonance_add' in modulation:
            modulation['resonance_add'] *= tension

        return modulation

    def parse_section_duration(self, section):
        """Helper to parse section duration from string format"""
        duration_str = section.get('duration', '0:10')
        if ':' in duration_str:
            parts = duration_str.split(':')
            try:
                start_time = int(parts[0])
                end_time = int(parts[1])
                return end_time - start_time
            except:
                return 10
        else:
            return 10

    def generate_complex_struggle_pattern(self, section_duration, scale, tension,
                                          final_filter, filter_env_amount, final_resonance,
                                          note_duration, modulation, total_samples):
        """
        COMPLEX_STRUGGLE: Generative Markov chain - cautious random walk
        - Probabilistic note selection (prefers returning to tonic)
        - Random walk with gravity toward root
        - Irregular durations based on random choices
        - Evolving hesitation (more pauses as tension builds)
        """
        section_pattern = np.zeros(total_samples)

        # Markov chain: probability of next note based on current note
        # Rows = current note, columns = next note probabilities
        # Higher weight on returning to tonic (index 0)
        transition_matrix = np.array([
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],  # From tonic: likely stay nearby
            [0.5, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0],  # From 1: pull back to tonic
            [0.4, 0.2, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0],  # From 2: pull back to tonic
            [0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0, 0.0],  # From 3: can go either way
            [0.2, 0.0, 0.1, 0.2, 0.1, 0.2, 0.2, 0.0],  # From 4: mid-range
            [0.1, 0.0, 0.0, 0.1, 0.3, 0.2, 0.2, 0.1],  # From 5: upper range
            [0.2, 0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 0.2],  # From 6: pull back down
            [0.3, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2],  # From 7: strong pull to descend
        ])

        # Normalize rows to sum to 1
        for i in range(len(transition_matrix)):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum

        # State: current position in scale
        current_note_idx = 0  # Start on tonic
        base_note_dur = note_duration * 1.5
        current_sample = 0

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # Get current note
            note_freq = scale[current_note_idx] if current_note_idx < len(scale) else scale[0]

            # Random duration (longer when on tonic = thinking)
            if current_note_idx == 0:
                duration = base_note_dur * self.rng.uniform(1.2, 2.5 + tension)
            else:
                duration = base_note_dur * self.rng.uniform(0.6, 1.2)

            # Generate note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter: darker on tonic, brighter when exploring
                filter_mult = 0.7 + (current_note_idx / len(scale)) * 0.8

                # Random velocity variation
                velocity = self.rng.uniform(0.6, 1.0)

                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='pulse',
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * self.rng.uniform(0.4, 0.8),
                    resonance=final_resonance * self.rng.uniform(0.7, 0.9),
                    amp_env=get_envelope('soft', self.config),
                    filter_env=get_filter_envelope('gentle', self.config)
                )

                # Add to pattern with random velocity
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            # Random pause (more hesitation as time goes on)
            pause_duration = base_note_dur * self.rng.uniform(0.2, 0.8 + progress * tension)
            pause_samples = int(pause_duration * self.sample_rate)
            current_sample += note_samples + pause_samples

            # Choose next note using Markov chain
            if current_note_idx < len(transition_matrix):
                probabilities = transition_matrix[current_note_idx]
                current_note_idx = self.rng.choice(len(probabilities), p=probabilities)
            else:
                current_note_idx = 0  # Fallback to tonic

        return section_pattern

    def generate_king_hunt_pattern(self, section_duration, scale, tension,
                                   final_filter, filter_env_amount, final_resonance,
                                   note_duration, modulation, total_samples):
        """
        KING_HUNT: Generative aggressive pursuit algorithm
        - State machine: ATTACK (ascending), RETREAT (descending), PAUSE (regrouping)
        - Probabilistic upward bias with evolving aggression
        - Random octave jumps and velocity variations
        - Building intensity and speed over time
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 0.5  # Fast, aggressive
        current_sample = 0

        # State machine
        STATE_ATTACK = 0
        STATE_RETREAT = 1
        STATE_PAUSE = 2
        current_state = STATE_ATTACK

        # Evolving parameters
        current_note_idx = 0  # Position in scale [0-7]
        current_octave = 0  # Octave offset (0, 1, 2)
        attack_run_length = 0  # How long we've been attacking

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # State transitions (probabilistic)
            if current_state == STATE_ATTACK:
                # Stay in attack mode with increasing probability
                if self.rng.random() < 0.15 - progress * 0.1:  # Less retreat as hunt intensifies
                    current_state = STATE_RETREAT
                    attack_run_length = 0
                elif self.rng.random() < 0.05:
                    current_state = STATE_PAUSE
                    attack_run_length = 0
                else:
                    attack_run_length += 1

            elif current_state == STATE_RETREAT:
                # Quick retreat, return to attack
                if self.rng.random() < 0.6 + progress * 0.3:  # Return faster as hunt intensifies
                    current_state = STATE_ATTACK

            elif current_state == STATE_PAUSE:
                # Brief pause, then attack
                if self.rng.random() < 0.8:
                    current_state = STATE_ATTACK

            # Note selection based on state
            if current_state == STATE_ATTACK:
                # Upward bias (70% up, 20% repeat, 10% down)
                rand = self.rng.random()
                if rand < 0.7:
                    # Move up
                    current_note_idx = min(7, current_note_idx + self.rng.integers(1, 3))

                    # Octave jumps (increasing probability with progress and attack length)
                    if self.rng.random() < 0.1 + progress * 0.2 + attack_run_length * 0.05:
                        current_octave = min(2, current_octave + 1)
                        current_note_idx = self.rng.integers(0, 4)  # Reset to lower note after jump

                elif rand < 0.9:
                    # Repeat (stabbing same note)
                    pass
                else:
                    # Small step down
                    current_note_idx = max(0, current_note_idx - 1)

            elif current_state == STATE_RETREAT:
                # Downward motion
                current_note_idx = max(0, current_note_idx - self.rng.integers(1, 3))
                if current_note_idx == 0 and current_octave > 0:
                    current_octave -= 1

            elif current_state == STATE_PAUSE:
                # Hold on dominant or tonic
                if self.rng.random() < 0.5:
                    current_note_idx = 0  # Tonic
                else:
                    current_note_idx = 4  # Dominant

            # Get frequency
            note_freq = scale[current_note_idx] * (2 ** current_octave)

            # Duration varies by state and progress
            if current_state == STATE_ATTACK:
                # Faster as hunt intensifies
                duration = base_note_dur * self.rng.uniform(0.4, 0.8) * (1.0 - progress * 0.3)
            elif current_state == STATE_RETREAT:
                duration = base_note_dur * self.rng.uniform(0.5, 1.0)
            else:  # PAUSE
                duration = base_note_dur * self.rng.uniform(1.5, 2.5)

            # Velocity varies by state
            if current_state == STATE_ATTACK:
                velocity = self.rng.uniform(0.8, 1.0) * (1.0 + progress * 0.3)  # Louder as hunt intensifies
            elif current_state == STATE_RETREAT:
                velocity = self.rng.uniform(0.6, 0.8)
            else:  # PAUSE
                velocity = self.rng.uniform(0.5, 0.7)

            # Generate note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter and resonance evolve with progress and state
                filter_mult = 1.0 + progress * 2.0  # Much brighter as hunt intensifies
                if current_state == STATE_ATTACK:
                    filter_mult *= self.rng.uniform(1.2, 1.5)  # Extra brightness on attack

                resonance_mult = 1.0 + progress * 0.8

                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='saw',  # Aggressive, bright
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * self.rng.uniform(0.8, 1.2),
                    resonance=final_resonance * resonance_mult,
                    amp_env=get_envelope('stab', self.config),
                    filter_env=get_filter_envelope('sweep', self.config)
                )

                # Volume
                level = self.config.LAYER_MIXING['pattern_note_level'] * velocity
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * level

            # Pause between notes (minimal in attack, longer in pause)
            if current_state == STATE_ATTACK:
                pause_samples = int(base_note_dur * 0.05 * self.sample_rate * (1.0 - progress * 0.5))
            elif current_state == STATE_RETREAT:
                pause_samples = int(base_note_dur * 0.15 * self.sample_rate)
            else:  # PAUSE
                pause_samples = int(base_note_dur * 0.5 * self.sample_rate)

            current_sample += note_samples + pause_samples

        return section_pattern

    def generate_crushing_attack_pattern(self, section_duration, scale, tension,
                                         final_filter, filter_env_amount, final_resonance,
                                         note_duration, modulation, total_samples):
        """
        CRUSHING_ATTACK: Generative relentless assault algorithm
        - State machine: ADVANCE (building pressure), STRIKE (hammer blows), OVERWHELM (climax)
        - Downward bias (crushing down on opponent)
        - Random chord stabs (multiple simultaneous notes)
        - Accelerating rhythm and increasing volume
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 0.4  # Fast, aggressive
        current_sample = 0

        # State machine
        STATE_ADVANCE = 0
        STATE_STRIKE = 1
        STATE_OVERWHELM = 2
        current_state = STATE_ADVANCE

        # Evolving parameters
        current_note_idx = 7  # Start high (crushing down from above)
        strike_count = 0  # Consecutive strikes

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # State transitions
            if current_state == STATE_ADVANCE:
                # Build pressure, then strike
                if self.rng.random() < 0.3 + progress * 0.2:  # More strikes as attack intensifies
                    current_state = STATE_STRIKE
                    strike_count = self.rng.integers(2, 5)  # Multiple hammer blows

            elif current_state == STATE_STRIKE:
                strike_count -= 1
                if strike_count <= 0:
                    if progress > 0.6 and self.rng.random() < 0.3:
                        current_state = STATE_OVERWHELM  # Climax
                    else:
                        current_state = STATE_ADVANCE

            elif current_state == STATE_OVERWHELM:
                # Stay in overwhelm mode once reached
                pass

            # Note selection based on state
            if current_state == STATE_ADVANCE:
                # Downward movement (pressure building)
                if self.rng.random() < 0.7:
                    current_note_idx = max(0, current_note_idx - self.rng.integers(1, 3))
                else:
                    current_note_idx = min(7, current_note_idx + 1)  # Occasional upward jab

            elif current_state == STATE_STRIKE:
                # Hammer on low notes (powerful blows)
                current_note_idx = self.rng.choice([0, 1, 2])  # Low register only

            elif current_state == STATE_OVERWHELM:
                # Chaotic attacks across entire range
                current_note_idx = self.rng.integers(0, 8)

            # Get base frequency
            note_freq = scale[current_note_idx]

            # Octave variations (more powerful with wider range)
            octave_shift = 0.0  # Default no shift
            if current_state == STATE_STRIKE:
                # Strike uses lower octave (bass power)
                if self.rng.random() < 0.5:
                    note_freq *= 0.5
                    octave_shift = -1.0
            elif current_state == STATE_OVERWHELM:
                # Overwhelm uses wide octave range
                octave_shift = float(self.rng.choice([-1, 0, 0, 1]))  # Bias toward higher
                note_freq *= (2.0 ** octave_shift)

            # Duration varies by state and progress
            if current_state == STATE_ADVANCE:
                duration = base_note_dur * self.rng.uniform(0.8, 1.2) * (1.0 - progress * 0.3)
            elif current_state == STATE_STRIKE:
                # Short, sharp attacks
                duration = base_note_dur * self.rng.uniform(0.3, 0.5)
            else:  # OVERWHELM
                # Very fast, relentless
                duration = base_note_dur * self.rng.uniform(0.2, 0.4) * (1.0 - progress * 0.2)

            # Velocity increases with progress and state
            # Capped to prevent clipping when multiple notes overlap
            if current_state == STATE_ADVANCE:
                velocity = self.rng.uniform(0.7, 0.9) * (1.0 + progress * 0.3)
            elif current_state == STATE_STRIKE:
                velocity = 1.0
            else:  # OVERWHELM
                velocity = self.rng.uniform(0.9, 1.0)  # No progress multiplier - prevents clipping

            # Generate main note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter opens more as attack intensifies
                filter_mult = 1.0 + progress * 2.5
                if current_state == STATE_STRIKE or current_state == STATE_OVERWHELM:
                    filter_mult *= 1.5  # Extra brightness on strikes

                resonance_mult = 1.0 + progress * 0.5  # Reduced to prevent filter self-oscillation

                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='saw',
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * self.rng.uniform(1.0, 1.5),
                    resonance=final_resonance * resonance_mult,
                    amp_env=get_envelope('stab', self.config),
                    filter_env=get_filter_envelope('sweep', self.config)
                )

                # Reduce level for this pattern to prevent clipping when overlapping
                # CRUSHING_ATTACK can have 3+ simultaneous note+chord pairs
                base_level = self.config.LAYER_MIXING['pattern_note_level'] * 0.3
                level = base_level * velocity
                end_sample = min(current_sample + len(pattern_note), total_samples)

                # CHORD STABS: Add harmonic notes on strikes and overwhelm
                has_chord = (current_state == STATE_STRIKE or current_state == STATE_OVERWHELM) and self.rng.random() < 0.6

                # Voice normalization: scale down when multiple voices present
                num_voices = 2 if has_chord else 1
                voice_scale = 1.0 / np.sqrt(num_voices)

                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * level * voice_scale

                if has_chord:
                    # Add fifth or octave
                    chord_interval = self.rng.choice([4, 7])  # Perfect fourth or fifth
                    chord_idx = min(7, current_note_idx + chord_interval)
                    chord_freq = scale[chord_idx] * (2 ** octave_shift if current_state == STATE_OVERWHELM else 1.0)

                    chord_note = self.synth_layer2.create_synth_note(
                        freq=chord_freq,
                        duration=note_duration_sec,
                        waveform='saw',
                        filter_base=final_filter * filter_mult,
                        filter_env_amount=filter_env_amount * self.rng.uniform(1.0, 1.5),
                        resonance=final_resonance * resonance_mult,
                        amp_env=get_envelope('stab', self.config),
                        filter_env=get_filter_envelope('sweep', self.config)
                    )

                    section_pattern[current_sample:end_sample] += chord_note[:end_sample - current_sample] * level * voice_scale

            # Pause between notes (decreases as attack intensifies)
            if current_state == STATE_ADVANCE:
                pause_samples = int(base_note_dur * 0.2 * self.sample_rate * (1.0 - progress * 0.5))
            elif current_state == STATE_STRIKE:
                pause_samples = int(base_note_dur * 0.05 * self.sample_rate)  # Minimal pause
            else:  # OVERWHELM
                pause_samples = int(base_note_dur * 0.02 * self.sample_rate)  # Almost no pause

            current_sample += note_samples + pause_samples

        return section_pattern

    def generate_sharp_theory_pattern(self, section_duration, scale, tension,
                                       final_filter, filter_env_amount, final_resonance,
                                       note_duration, modulation, total_samples):
        """
        SHARP_THEORY: State machine for aggressive tactical openings
        - States: ATTACK (rapid ascent), DART (quick jumps), SETTLE (brief pause)
        - Fast, energetic, unpredictable like Sicilian tactics
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 0.4
        current_sample = 0

        # State machine
        STATE_ATTACK = 0
        STATE_DART = 1
        STATE_SETTLE = 2
        current_state = STATE_ATTACK
        current_note_idx = 0

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # State transitions
            if current_state == STATE_ATTACK:
                if self.rng.random() < 0.2:
                    current_state = STATE_DART
                elif self.rng.random() < 0.1:
                    current_state = STATE_SETTLE
            elif current_state == STATE_DART:
                if self.rng.random() < 0.6:
                    current_state = STATE_ATTACK
                elif self.rng.random() < 0.2:
                    current_state = STATE_SETTLE
            elif current_state == STATE_SETTLE:
                if self.rng.random() < 0.8:
                    current_state = STATE_ATTACK

            # Note selection by state
            if current_state == STATE_ATTACK:
                # Aggressive attacks - favor upper register, random leaps
                if self.rng.random() < 0.5:
                    current_note_idx = self.rng.choice([4, 5, 6, 7])  # Upper half
                else:
                    current_note_idx = self.rng.choice([2, 3, 4])  # Middle
            elif current_state == STATE_DART:
                # Random tactical jumps anywhere
                current_note_idx = self.rng.integers(0, len(scale))
            elif current_state == STATE_SETTLE:
                # Gravitate to stable notes
                if self.rng.random() < 0.6:
                    current_note_idx = 0  # Tonic
                else:
                    current_note_idx = 4  # Dominant

            note_freq = scale[current_note_idx]
            duration = base_note_dur * self.rng.uniform(0.6, 1.0)
            velocity = self.rng.uniform(0.75, 1.0)

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform='saw',
                    filter_base=final_filter * (1.5 + tension * 0.5),
                    filter_env_amount=filter_env_amount,
                    resonance=final_resonance * 0.8,
                    amp_env=get_envelope('pluck', self.config),
                    filter_env=get_filter_envelope('sweep', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            pause_samples = int(base_note_dur * 0.05 * self.sample_rate)
            current_sample += note_samples + pause_samples

        return section_pattern

    def generate_positional_theory_pattern(self, section_duration, scale, tension,
                                           final_filter, filter_env_amount, final_resonance,
                                           note_duration, modulation, total_samples):
        """
        POSITIONAL_THEORY: Markov chain for strategic maneuvering
        - Weighted toward stable harmonic intervals (tonic, third, fifth)
        - Deliberate, controlled transitions
        - Patient, strategic like French/English openings
        - Calm, contemplative, with ample breathing room
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 1.2  # Slower notes for calm, patient play
        current_sample = 0

        # Markov chain: favor stable harmonic intervals
        transition_matrix = np.array([
            [0.3, 0.1, 0.3, 0.1, 0.2, 0.0, 0.0, 0.0],  # From tonic: to tonic/third/fifth
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0],  # From 1
            [0.4, 0.1, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0],  # From 2 (third): to tonic/fifth
            [0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0],  # From 3
            [0.3, 0.0, 0.2, 0.1, 0.2, 0.1, 0.1, 0.0],  # From 4 (fifth): to tonic/third
            [0.1, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0],  # From 5
            [0.1, 0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1],  # From 6
            [0.2, 0.0, 0.1, 0.0, 0.2, 0.2, 0.2, 0.1],  # From 7: return toward tonic/fifth
        ])

        # Normalize
        for i in range(len(transition_matrix)):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum

        current_note_idx = 0  # Start on tonic

        while current_sample < total_samples:
            note_freq = scale[current_note_idx]
            duration = base_note_dur * self.rng.uniform(0.8, 1.0)  # Shorter notes, more space
            velocity = 0.35 + self.rng.uniform(-0.1, 0.15)  # Much quieter (25-50% vs 60-80%)

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform='pulse',
                    filter_base=final_filter * 0.7,  # Darker, less bright
                    filter_env_amount=filter_env_amount * 0.5,  # Less filter movement
                    resonance=final_resonance * 0.5,  # Softer resonance
                    amp_env=get_envelope('soft', self.config),
                    filter_env=get_filter_envelope('gentle', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            # Much longer pause for contemplative, strategic feel
            pause_samples = int(base_note_dur * 0.6 * self.sample_rate)  # 60% pause vs 15%
            current_sample += note_samples + pause_samples

            # Markov transition
            if current_note_idx < len(transition_matrix):
                probabilities = transition_matrix[current_note_idx]
                current_note_idx = self.rng.choice(len(probabilities), p=probabilities)
            else:
                current_note_idx = 0

        return section_pattern

    def generate_solid_theory_pattern(self, section_duration, scale, tension,
                                      final_filter, filter_env_amount, final_resonance,
                                      note_duration, modulation, total_samples):
        """
        SOLID_THEORY: Grounded bass patterns with stable rhythms
        - Safe, solid character for Queen's Gambit Declined, Slav, solid openings
        - Lower register emphasis
        - Predictable, repetitive patterns (building blocks)
        - Steady rhythm with minimal variation
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 1.2  # Slower, grounded
        current_sample = 0

        # Build a simple repeating pattern (tonic, fifth, third, fifth)
        pattern_sequence = [0, 4, 2, 4]  # Scale degrees
        pattern_idx = 0

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # Get note from pattern (lower octave for grounded feel)
            scale_idx = pattern_sequence[pattern_idx % len(pattern_sequence)]
            note_freq = scale[scale_idx] * 0.75  # Lower by perfect fourth

            # Duration: very consistent for stability
            duration = base_note_dur * self.rng.uniform(0.95, 1.05)

            # Generate note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter: darker, grounded
                filter_mult = 0.6 + progress * 0.2  # Stays dark

                # Velocity: very stable
                velocity = 0.75 + self.rng.uniform(-0.05, 0.05)

                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='pulse',  # Warm, solid
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * 0.6,
                    resonance=final_resonance * 0.5,  # Low resonance for solid feel
                    amp_env=get_envelope('soft', self.config),
                    filter_env=get_filter_envelope('gentle', self.config)
                )

                # Add to pattern
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            # Consistent pause
            pause_samples = int(base_note_dur * 0.2 * self.sample_rate)
            current_sample += note_samples + pause_samples

            # Advance pattern
            pattern_idx += 1

        return section_pattern

    def generate_desperate_defense_pattern(self, section_duration, scale, tension,
                                           final_filter, filter_env_amount, final_resonance,
                                           note_duration, modulation, total_samples):
        """
        DESPERATE_DEFENSE: Reactive, hesitant defensive patterns
        - Lower register, darker tone
        - Syncopated rhythms with pauses (hesitation)
        - Responds to pressure with tentative moves
        - Inspired by defender struggling under attack
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 1.4  # Slower, more deliberate
        current_sample = 0

        # Defensive states
        STATE_RETREAT = 0      # Move pieces back
        STATE_BLOCKADE = 1     # Hold defensive structure
        STATE_COUNTER = 2      # Brief counter-attack attempt
        current_state = STATE_RETREAT
        tension_accumulator = 0

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # State transitions based on accumulated tension
            tension_accumulator += self.rng.uniform(0, tension * 2)

            if current_state == STATE_RETREAT:
                if tension_accumulator > 3:
                    current_state = STATE_BLOCKADE
                    tension_accumulator = 0
            elif current_state == STATE_BLOCKADE:
                if tension_accumulator > 5 and self.rng.random() < 0.2:
                    current_state = STATE_COUNTER  # Rare counter
                    tension_accumulator = 0
                elif tension_accumulator > 4:
                    current_state = STATE_RETREAT
                    tension_accumulator = 0
            elif current_state == STATE_COUNTER:
                # Counter attempts are brief
                if self.rng.random() < 0.6:
                    current_state = STATE_RETREAT
                    tension_accumulator = 0

            # Note selection by defensive state
            if current_state == STATE_RETREAT:
                # Descending patterns, lower register
                note_idx = self.rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                octave_mult = 0.5  # Drop octave
            elif current_state == STATE_BLOCKADE:
                # Hold root and fifth (stable but tense)
                note_idx = self.rng.choice([0, 4], p=[0.6, 0.4])
                octave_mult = 1.0
            elif current_state == STATE_COUNTER:
                # Brief ascending attempt
                note_idx = self.rng.choice([4, 5, 6, 7], p=[0.25, 0.25, 0.25, 0.25])
                octave_mult = 1.0

            note_freq = scale[note_idx] * octave_mult

            # Hesitant timing - random pauses before notes
            if self.rng.random() < 0.4:  # 40% chance of hesitation
                hesitation_samples = int(base_note_dur * 0.3 * self.sample_rate * self.rng.random())
                current_sample += hesitation_samples

            if current_sample >= total_samples:
                break

            duration = base_note_dur * self.rng.uniform(0.7, 1.1)
            velocity = 0.5 - progress * 0.15  # Fade as defense crumbles

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                # Darker waveform, lower filter for defensive sound
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform='saw' if current_state == STATE_RETREAT else 'pulse',
                    filter_base=final_filter * (0.5 + progress * 0.3),  # Dark, opening slightly
                    filter_env_amount=filter_env_amount * 0.5,
                    resonance=final_resonance * (0.8 + tension * 0.4),
                    amp_env=get_envelope('pluck', self.config),
                    filter_env=get_filter_envelope('closing', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            # Variable pauses (uncertainty)
            pause_samples = int(base_note_dur * self.rng.uniform(0.15, 0.35) * self.sample_rate)
            current_sample += note_samples + pause_samples

        return section_pattern

    def generate_tactical_chaos_pattern(self, section_duration, scale, tension,
                                        final_filter, filter_env_amount, final_resonance,
                                        note_duration, modulation, total_samples):
        """
        TACTICAL_CHAOS: Rapid, unpredictable tactical exchanges
        - Wide register jumps
        - Dense overlapping bursts
        - Alternating attack/defense fragments
        - High entropy, nervous energy
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 0.6  # Fast exchanges
        current_sample = 0

        # Chaos parameters
        burst_mode = False
        burst_countdown = 0
        attack_side = 0  # 0=white, 1=black (alternating tactical blows)

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # Random burst mode activation
            if not burst_mode and self.rng.random() < 0.15:
                burst_mode = True
                burst_countdown = self.rng.integers(3, 8)  # 3-8 rapid notes

            if burst_mode:
                burst_countdown -= 1
                if burst_countdown <= 0:
                    burst_mode = False
                    attack_side = 1 - attack_side  # Switch attacker

            # Note selection - chaotic jumps
            if burst_mode:
                # Tactical burst - wide jumps
                note_idx = self.rng.choice(range(len(scale)))
                # Random octave shifts
                octave_choices = [0.5, 1.0, 2.0]
                octave_mult = self.rng.choice(octave_choices, p=[0.2, 0.5, 0.3])
            else:
                # Cautious repositioning between bursts
                note_idx = self.rng.choice([0, 2, 4, 5], p=[0.3, 0.25, 0.25, 0.2])
                octave_mult = 1.0

            note_freq = scale[note_idx] * octave_mult

            # Timing - erratic in bursts, steadier between
            if burst_mode:
                duration = base_note_dur * self.rng.uniform(0.4, 0.7)
                velocity = 0.7 + self.rng.random() * 0.2
            else:
                duration = base_note_dur * self.rng.uniform(1.0, 1.5)
                velocity = 0.5 + self.rng.random() * 0.15

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                # Aggressive waveform, wide filter sweeps
                waveform = 'square' if burst_mode else 'pulse'
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform=waveform,
                    filter_base=final_filter * (0.6 + progress * 0.8),
                    filter_env_amount=filter_env_amount * (1.5 if burst_mode else 0.8),
                    resonance=final_resonance * (1.3 if burst_mode else 0.9),
                    amp_env=get_envelope('percussive' if burst_mode else 'stab', self.config),
                    filter_env=get_filter_envelope('sharp' if burst_mode else 'smooth', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            # Minimal pause in bursts, longer between
            if burst_mode:
                pause_samples = int(base_note_dur * 0.05 * self.sample_rate)  # Almost no pause
            else:
                pause_samples = int(base_note_dur * self.rng.uniform(0.2, 0.5) * self.sample_rate)

            current_sample += note_samples + pause_samples

        return section_pattern

    def generate_flawless_conversion_pattern(self, section_duration, scale, tension,
                                             final_filter, filter_env_amount, final_resonance,
                                             note_duration, modulation, total_samples):
        """
        FLAWLESS_CONVERSION: State machine for precise endgame technique
        - States: ADVANCE (build pressure), CONSOLIDATE (hold position), BREAKTHROUGH (push advantage)
        - Methodical, controlled, inevitable like Fischer's technique
        """
        section_pattern = np.zeros(total_samples)
        base_note_dur = note_duration * 1.3
        current_sample = 0

        # State machine
        STATE_ADVANCE = 0
        STATE_CONSOLIDATE = 1
        STATE_BREAKTHROUGH = 2
        current_state = STATE_ADVANCE
        current_note_idx = 0
        advance_progress = 0  # Track how far we've advanced

        while current_sample < total_samples:
            progress = current_sample / total_samples

            # State transitions
            if current_state == STATE_ADVANCE:
                advance_progress += 1
                if advance_progress > 5 and self.rng.random() < 0.3:
                    current_state = STATE_CONSOLIDATE
                    advance_progress = 0
                elif progress > 0.7 and self.rng.random() < 0.2:
                    current_state = STATE_BREAKTHROUGH
            elif current_state == STATE_CONSOLIDATE:
                if self.rng.random() < 0.5:
                    current_state = STATE_ADVANCE
                elif progress > 0.7 and self.rng.random() < 0.3:
                    current_state = STATE_BREAKTHROUGH
            elif current_state == STATE_BREAKTHROUGH:
                if self.rng.random() < 0.1:
                    current_state = STATE_ADVANCE

            # Note selection by state
            if current_state == STATE_ADVANCE:
                # Build pressure - favor dominant and upper notes, but randomly
                weights = [0.1, 0.05, 0.15, 0.1, 0.25, 0.15, 0.1, 0.1]  # Favor 5th (index 4)
                current_note_idx = self.rng.choice(range(len(scale)), p=weights)
            elif current_state == STATE_CONSOLIDATE:
                # Hold stable harmonic notes - tonic, third, fifth
                current_note_idx = self.rng.choice([0, 2, 4], p=[0.4, 0.3, 0.3])
            elif current_state == STATE_BREAKTHROUGH:
                # Decisive moves - octave relationships
                current_note_idx = self.rng.choice([0, 4, 7], p=[0.3, 0.4, 0.3])  # Tonic, fifth, seventh

            note_freq = scale[current_note_idx]
            duration = base_note_dur * self.rng.uniform(0.96, 1.04)
            velocity = 0.55 + progress * 0.15

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform='triangle',
                    filter_base=final_filter * (0.7 + progress * 0.6),
                    filter_env_amount=filter_env_amount * 0.7,
                    resonance=final_resonance * 0.6,
                    amp_env=get_envelope('sustained', self.config),
                    filter_env=get_filter_envelope('closing', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity * 0.85

            pause_samples = int(base_note_dur * 0.18 * self.sample_rate)
            current_sample += note_samples + pause_samples

        return section_pattern

    def generate_decisive_outro_pattern(self, section_duration, scale, tension,
                                        final_filter, filter_env_amount, final_resonance,
                                        note_duration, modulation, total_samples):
        """
        DECISIVE_ENDING: Strong resolution for decisive games (1-0 or 0-1)
        - Clear, definitive phrases
        - Resolves to tonic with authority
        - Descending/ascending gesture based on overall narrative
        """
        section_pattern = np.zeros(total_samples)

        # Determine direction based on overall narrative
        is_defeat = 'DEFEAT' in self.overall_narrative

        # Create final resolving phrase
        phrase_notes = []
        if is_defeat:
            # Descending resolution - somber but resolved
            phrase_notes = [7, 5, 4, 2, 0]  # 7th -> 5th -> 4th -> 2nd -> tonic
        else:
            # Ascending resolution - triumphant
            phrase_notes = [0, 2, 4, 5, 7, 0]  # tonic -> ... -> tonic (octave)

        # Play the phrase
        current_sample = 0
        note_dur = section_duration / (len(phrase_notes) + 1)  # Leave space at end

        for i, note_idx in enumerate(phrase_notes):
            if note_idx >= len(scale):
                note_idx = 0
            note_freq = scale[note_idx]

            # Fade out over the phrase
            progress = i / len(phrase_notes)
            velocity = 0.7 * (1.0 - progress * 0.5)  # Gentle fadeout

            note_samples = int(note_dur * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform='triangle',
                    filter_base=final_filter * (1.0 - progress * 0.3),  # Close filter
                    filter_env_amount=filter_env_amount * 0.5,
                    resonance=final_resonance * 0.7,
                    amp_env=get_envelope('sustained', self.config),
                    filter_env=get_filter_envelope('closing', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * velocity

            current_sample += note_samples

        # Apply exponential decay to entire pattern
        decay_curve = np.exp(np.linspace(0, -2.5, len(section_pattern)))
        section_pattern *= decay_curve

        return section_pattern

    def generate_draw_outro_pattern(self, section_duration, scale, tension,
                                     final_filter, filter_env_amount, final_resonance,
                                     note_duration, modulation, total_samples):
        """
        DRAWN_ENDING: Balanced, unresolved ending for drawn games (1/2-1/2)
        - Circular motion
        - Returns to tonic but without strong resolution
        - Peaceful but incomplete feeling
        """
        section_pattern = np.zeros(total_samples)

        # Create circular phrase - goes around and returns without strong cadence
        # Use perfect fourth intervals for stability without finality
        phrase_notes = [0, 3, 0, 3, 0]  # Tonic <-> fourth (peaceful rocking)

        current_sample = 0
        note_dur = section_duration / (len(phrase_notes) + 2)  # Extra space

        for i, note_idx in enumerate(phrase_notes):
            if note_idx >= len(scale):
                note_idx = 0
            note_freq = scale[note_idx]

            # Gradual fadeout
            progress = i / len(phrase_notes)
            velocity = 0.6 * (1.0 - progress * 0.6)  # Faster fadeout

            note_samples = int(note_dur * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth_layer2.create_synth_note(
                    freq=note_freq,
                    duration=note_samples / self.sample_rate,
                    waveform='sine',  # Pure, simple tone for neutrality
                    filter_base=final_filter * (1.0 - progress * 0.4),
                    filter_env_amount=filter_env_amount * 0.3,
                    resonance=final_resonance * 0.5,
                    amp_env=get_envelope('pad', self.config),
                    filter_env=get_filter_envelope('slow', self.config)
                )
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * velocity

            current_sample += note_samples

        # Apply exponential decay
        decay_curve = np.exp(np.linspace(0, -3.0, len(section_pattern)))
        section_pattern *= decay_curve

        return section_pattern

    def create_evolving_drone(self, drone_freq, section_duration, waveform, current_base,
                              progress, total_sections, total_samples):
        """
        Create a multi-timescale evolving drone with:
        - Multiple detuned oscillators with time-varying detune
        - MACRO evolution: Linear filter sweep across entire section
        - MESO evolution: Slow LFO cycles (30-50 seconds)
        - MICRO evolution: Fast LFO shimmer (10 seconds)
        """
        # Generate multiple detuned oscillators with FIXED detune spread
        num_voices = current_base.get('drone_voices', 3)
        oscillators = []

        for v in range(num_voices):
            # Fixed detune per voice (in cents)
            detune_cents = np.linspace(-4, 4, num_voices)[v]
            detune_hz = drone_freq * (2 ** (detune_cents / 1200.0) - 1.0)

            osc = self.synth_layer1.oscillator(drone_freq + detune_hz, section_duration, waveform)

            # Apply time-varying amplitude modulation to create beating
            detune_lfo = np.sin(2 * np.pi * 0.03 * np.arange(len(osc)) / self.sample_rate + v * 0.5)
            osc = osc * (1.0 + detune_lfo * 0.03) # drone breathes instead of “chugging.”

            oscillators.append(osc)

        # Mix oscillators
        base_osc = np.sum(oscillators, axis=0) / num_voices

        # Create three-timescale LFO system
        section_progress = np.linspace(0, 1, len(base_osc))
        slow_lfo = np.sin(2 * np.pi * 0.02 * np.arange(len(base_osc)) / self.sample_rate)
        fast_lfo = signal.sawtooth(2 * np.pi * 0.1 * np.arange(len(base_osc)) / self.sample_rate, width=0.5)

        # Get filter parameters
        params_start = self.interpolate_base_params(progress)
        params_end = self.interpolate_base_params(min(1.0, progress + (1.0 / total_sections)))
        filter_start = params_start['filter']
        filter_end = params_end['filter']
        resonance_start = params_start['resonance']
        resonance_end = params_end['resonance']

        # Apply filter with multi-timescale modulation
        base_drone = np.zeros_like(base_osc)
        chunk_size = 512

        for i in range(0, len(base_osc), chunk_size):
            end = min(i + chunk_size, len(base_osc))
            chunk = base_osc[i:end]

            # MACRO: Linear evolution across section
            macro_progress = section_progress[i]
            base_cutoff = filter_start + (filter_end - filter_start) * macro_progress
            base_resonance = resonance_start + (resonance_end - resonance_start) * macro_progress

            # MESO: Slow LFO (30-50 second cycles)
            meso_mod = slow_lfo[i] * 500

            # MICRO: Fast LFO (10 second cycles)
            micro_mod = fast_lfo[i] * 100

            # Combine all timescales
            current_cutoff = base_cutoff + meso_mod + micro_mod
            current_resonance = base_resonance + slow_lfo[i] * 0.3

            current_cutoff = np.clip(current_cutoff, 20, self.synth_layer1.nyquist * 0.95)
            current_resonance = np.clip(current_resonance, 0.1, 4.0)

            # Apply filter
            filtered_chunk = self.synth_layer1.moog_filter(chunk, current_cutoff, current_resonance)
            base_drone[i:end] = filtered_chunk

        # Apply amplitude envelope
        amp_env = self.synth_layer1.adsr_envelope(len(base_drone), *get_envelope('pad', self.config))
        base_drone = base_drone * amp_env
        # Reduce drone level
        drone_gain = 0.3  # adjust between 0.2–0.5 as needed
        base_drone = base_drone * drone_gain

        return base_drone

    def create_moment_voice(self, moment, current_params, progress):
        """LAYER 3: Key moments as additional synth voices - context-aware"""

        moment_type = moment.get('type', 'UNKNOWN')

        # Build context key for moment voice lookup
        if moment_type in ['BLUNDER', 'MISTAKE']:
            if 'DEFEAT' in self.overall_narrative:
                voice_key = f'{moment_type}_IN_DEFEAT'
            elif 'MASTERPIECE' in self.overall_narrative:
                voice_key = f'{moment_type}_IN_MASTERPIECE'
            else:
                voice_key = f'{moment_type}_NEUTRAL'

            voice_params = self.config.MOMENT_VOICES.get(voice_key, self.config.MOMENT_VOICES['DEFAULT_MOMENT'])

            amp_env = get_envelope(voice_params['amp_env'], self.config)
            filter_env = get_filter_envelope(voice_params.get('filter_env', 'gentle'), self.config) if 'filter_env' in voice_params else None

            if filter_env:
                return self.synth_layer3.create_synth_note(
                    freq=voice_params['freq'],
                    duration=voice_params['duration'],
                    waveform=voice_params['waveform'],
                    filter_base=voice_params['filter_base'],
                    filter_env_amount=voice_params['filter_env_amount'],
                    resonance=voice_params['resonance'],
                    amp_env=amp_env,
                    filter_env=filter_env
                )
            else:
                return self.synth_layer3.create_synth_note(
                    freq=voice_params['freq'],
                    duration=voice_params['duration'],
                    waveform=voice_params['waveform'],
                    filter_base=voice_params['filter_base'],
                    filter_env_amount=voice_params['filter_env_amount'],
                    resonance=voice_params['resonance'],
                    amp_env=amp_env
                )

        elif moment_type in ['BRILLIANT', 'STRONG']:
            if 'MASTERPIECE' in self.overall_narrative:
                voice_key = f'{moment_type}_IN_MASTERPIECE'
                voice_params = self.config.MOMENT_VOICES[voice_key].copy()
                # Modify by progress
                voice_params['freq'] = voice_params['freq'] * (1 + progress)
                voice_params['filter_env_amount'] = voice_params['filter_env_amount'] * (1 + progress)
            elif 'DEFEAT' in self.overall_narrative:
                voice_key = f'{moment_type}_IN_DEFEAT'
                voice_params = self.config.MOMENT_VOICES[voice_key]
            else:
                voice_key = f'{moment_type}_NEUTRAL'
                voice_params = self.config.MOMENT_VOICES[voice_key]

            amp_env = get_envelope(voice_params['amp_env'], self.config)
            filter_env = get_filter_envelope(voice_params.get('filter_env', 'gentle'), self.config) if 'filter_env' in voice_params else None

            if filter_env:
                return self.synth_layer3.create_synth_note(
                    freq=voice_params['freq'],
                    duration=voice_params['duration'],
                    waveform=voice_params['waveform'],
                    filter_base=voice_params['filter_base'],
                    filter_env_amount=voice_params['filter_env_amount'],
                    resonance=voice_params['resonance'],
                    amp_env=amp_env,
                    filter_env=filter_env
                )
            else:
                return self.synth_layer3.create_synth_note(
                    freq=voice_params['freq'],
                    duration=voice_params['duration'],
                    waveform=voice_params['waveform'],
                    filter_base=voice_params['filter_base'],
                    filter_env_amount=voice_params['filter_env_amount'],
                    resonance=voice_params['resonance'],
                    amp_env=amp_env
                )

        elif moment_type == 'DEVELOPMENT':
            # Use development parameters from config
            dev_key = 'IN_DEFEAT' if 'DEFEAT' in self.overall_narrative else 'DEFAULT'
            dev_params = self.config.DEVELOPMENT_PARAMS[dev_key]

            base_filter = current_params.get('filter', 1000)
            scale = current_params.get('scale', self.config.SCALES['minor'])

            melody_freqs = [scale[i] for i in dev_params['melody_indices'] if i < len(scale)]

            phrase_samples = []
            for i, freq in enumerate(melody_freqs):
                note = self.synth_layer3.create_synth_note(
                    freq=freq,
                    duration=dev_params['note_duration'],
                    waveform=dev_params['waveform'],
                    filter_base=base_filter * dev_params['filter_mult'],
                    filter_env_amount=dev_params['base_filter_env'] + (i * dev_params['filter_env_step']),
                    resonance=dev_params['resonance'],
                    amp_env=get_envelope(dev_params['amp_env'], self.config)
                )
                phrase_samples.append(note * dev_params['volume'])

            # Connect notes with gaps
            gap_samples = int(self.config.TIMING['note_gap_sec'] * self.sample_rate)
            combined = []
            for i, note_samples in enumerate(phrase_samples):
                combined.append(note_samples)
                if i < len(phrase_samples) - 1:
                    combined.append(np.zeros(gap_samples))

            return np.concatenate(combined)

        elif moment_type == 'FIRST_EXCHANGE':
            # Use first exchange parameters from config
            ex_key = 'IN_DEFEAT' if 'DEFEAT' in self.overall_narrative else 'DEFAULT'
            ex_params = self.config.FIRST_EXCHANGE_PARAMS[ex_key]

            base_filter = current_params.get('filter', 1000)
            scale = current_params.get('scale', self.config.SCALES['minor'])

            # Generate question phrase
            question_freqs = [scale[i] for i in ex_params['question_indices'] if i < len(scale)]
            question_samples = []
            for i, freq in enumerate(question_freqs):
                note = self.synth_layer3.create_synth_note(
                    freq=freq,
                    duration=ex_params['note_duration'],
                    waveform=ex_params['question_waveform'],
                    filter_base=base_filter * ex_params['filter_mult'],
                    filter_env_amount=ex_params['question_filter_env_base'] + (i * 100),
                    resonance=ex_params['question_resonance'],
                    amp_env=get_envelope('pluck', self.config)
                )
                question_samples.append(note)

            # Generate answer phrase
            answer_freqs = [scale[i] for i in ex_params['answer_indices'] if i < len(scale)]
            answer_samples = []
            for i, freq in enumerate(answer_freqs):
                note = self.synth_layer3.create_synth_note(
                    freq=freq,
                    duration=ex_params['note_duration'],
                    waveform=ex_params['answer_waveform'],
                    filter_base=base_filter * ex_params['answer_brightness'],
                    filter_env_amount=ex_params['answer_filter_env_base'] + (i * 50),
                    resonance=ex_params['answer_resonance'],
                    amp_env=get_envelope('soft', self.config)
                )
                answer_samples.append(note)

            # Combine with timing
            gap_samples = int(self.config.TIMING['note_gap_sec'] * self.sample_rate)
            pause_samples = int(self.config.TIMING['phrase_pause_sec'] * self.sample_rate)

            combined = []
            for i, note in enumerate(question_samples):
                combined.append(note * ex_params['question_volume'])
                if i < len(question_samples) - 1:
                    combined.append(np.zeros(gap_samples))

            combined.append(np.zeros(pause_samples))

            for i, note in enumerate(answer_samples):
                combined.append(note * ex_params['answer_volume'])
                if i < len(answer_samples) - 1:
                    combined.append(np.zeros(gap_samples))

            return np.concatenate(combined)

        elif moment_type == 'TACTICAL_SEQUENCE':
            voice_params = self.config.MOMENT_VOICES['TACTICAL_SEQUENCE']
            combined = np.zeros(int(voice_params['total_duration'] * self.sample_rate))
            for i, freq in enumerate(voice_params['freqs']):
                note = self.synth_layer3.create_synth_note(
                    freq=freq,
                    duration=voice_params['note_duration'],
                    waveform=voice_params['waveform'],
                    filter_base=voice_params['filter_base'],
                    filter_env_amount=voice_params['filter_env_amount'],
                    resonance=voice_params['resonance'],
                    amp_env=get_envelope(voice_params['amp_env'], self.config)
                )
                start_sample = int(i * voice_params['overlap_factor'] * self.sample_rate)
                if start_sample + len(note) < len(combined):
                    combined[start_sample:start_sample+len(note)] += note * voice_params['volume']
            return combined

        elif moment_type == 'MATE_SEQUENCE':
            if 'DEFEAT' in self.overall_narrative:
                voice_key = 'MATE_IN_DEFEAT'
            elif 'MASTERPIECE' in self.overall_narrative:
                voice_key = 'MATE_IN_MASTERPIECE'
            else:
                voice_key = 'DEFAULT_MOMENT'

            voice_params = self.config.MOMENT_VOICES.get(voice_key, self.config.MOMENT_VOICES['DEFAULT_MOMENT'])
            return self.synth_layer3.create_synth_note(
                freq=voice_params['freq'],
                duration=voice_params['duration'],
                waveform=voice_params['waveform'],
                filter_base=voice_params['filter_base'],
                filter_env_amount=voice_params['filter_env_amount'],
                resonance=voice_params['resonance'],
                amp_env=get_envelope(voice_params['amp_env'], self.config)
            )

        # Default
        voice_params = self.config.MOMENT_VOICES['DEFAULT_MOMENT']
        return self.synth_layer3.create_synth_note(
            freq=voice_params['freq'],
            duration=voice_params['duration'],
            waveform=voice_params['waveform'],
            filter_base=voice_params['filter_base'],
            filter_env_amount=voice_params['filter_env_amount'],
            resonance=voice_params['resonance'],
            amp_env=get_envelope(voice_params['amp_env'], self.config)
        )

    def _get_melodic_pattern(self, tension):
        """Get melodic pattern based on overall narrative and tension"""
        if 'DEFEAT' in self.overall_narrative:
            if tension > 0.7:
                key = 'DEFEAT_HIGH_TENSION'
            else:
                key = 'DEFEAT_LOW_TENSION'
        elif 'MASTERPIECE' in self.overall_narrative:
            if tension > 0.7:
                key = 'MASTERPIECE_HIGH_TENSION'
            else:
                key = 'MASTERPIECE_LOW_TENSION'
        else:
            if tension > 0.7:
                key = 'NEUTRAL_HIGH_TENSION'
            elif tension > 0.4:
                key = 'NEUTRAL_MEDIUM_TENSION'
            else:
                key = 'NEUTRAL_LOW_TENSION'

        return self.config.MELODIC_PATTERNS[key]

    def compose_section(self, section, section_index, total_sections):
        """Compose a section using all three narrative layers"""

        section_duration = self.parse_section_duration(section)
        narrative = section.get('narrative', 'UNKNOWN')
        tension = section.get('tension', 0.5)

        # Calculate progress
        progress = section_index / max(1, total_sections - 1)

        # LAYER 1: Get base parameters
        current_base = self.interpolate_base_params(progress)

        # LAYER 2: Apply section modulation
        modulation = self._get_section_modulation(narrative, tension)

        # Calculate final synthesis parameters
        final_filter = current_base['filter'] * modulation['filter_mult']
        final_filter = np.clip(final_filter, 20, self.synth_layer3.nyquist * 0.95)

        final_resonance = current_base['resonance'] + modulation['resonance_add']
        final_resonance = np.clip(final_resonance, 0.1, 4.0)

        filter_env_amount = modulation['filter_env_amount'] * (1 + current_base['detune'] / 20)

        key_moments_count = len(section.get('key_moments', []))
        section_num = section_index + 1

        print(f"\nSECTION {section_num}/{total_sections}: {section['name']} ({section_duration}s)")
        print(f"  Narrative: {narrative} | Tension: {tension:.2f}")
        print(f"  Filter: {final_filter:.0f}Hz | Resonance: {final_resonance:.2f}")
        print(f"  Key Moments: {key_moments_count}")

        waveform = current_base['waveform']
        scale = current_base['scale']

        # Override base parameters for specific section narratives
        if narrative == 'FLAWLESS_CONVERSION':
            # Use warmer, calmer drone for technical endgames
            waveform = 'triangle'  # Warmer than saw
            # Create a modified current_base with lower detune
            current_base = current_base.copy()
            current_base['detune'] = min(current_base['detune'], 3)  # Cap at 3 cents (gentle)

        # Calculate note duration
        base_note_duration = self.config.BASE_NOTE_DURATION
        note_duration = base_note_duration * current_base['tempo'] * modulation['tempo_mult']

        # Apply narrative process transformations
        current_time = section.get('start_ply', 0)
        process_key_moment = None
        for moment in section.get('key_moments', []):
            if process_key_moment is None:
                process_key_moment = moment
                break

        transforms = self.narrative_process.update(current_time, process_key_moment)
        volume_multiplier = 1.0

        if transforms:
            if 'tempo_multiplier' in transforms:
                note_duration *= transforms['tempo_multiplier']
            if 'volume_decay' in transforms:
                volume_multiplier = transforms['volume_decay']
            if 'volume_crescendo' in transforms:
                volume_multiplier = transforms['volume_crescendo']

        # Get melodic pattern
        pattern_config = self._get_melodic_pattern(tension)
        pattern = [scale[idx] for idx in pattern_config['indices'] if idx < len(scale)]

        num_notes = int(section_duration / note_duration * modulation['note_density'])
        total_samples = int(section_duration * self.sample_rate)

        # LAYER 1: Generate BASE DRONE
        drone_freq = scale[0] / 2
        base_drone = np.zeros(total_samples)

        if self.config.LAYER_ENABLE['drone']:
            print(f"  → Layer 1: Evolving drone ({drone_freq:.1f}Hz base) | wave: {waveform} | detune: {current_base['detune']:.1f}¢")
        else:
            print(f"  → Layer 1: (muted)")

        if self.config.LAYER_ENABLE['drone']:
            base_drone = self.create_evolving_drone(
                drone_freq=drone_freq,
                section_duration=section_duration,
                waveform=waveform,
                current_base=current_base,
                progress=progress,
                total_sections=total_sections,
                total_samples=total_samples
            )

        # LAYER 2: Generate RHYTHMIC PATTERNS (using refactored PatternCoordinator)
        section_pattern = np.zeros(total_samples)

        if self.config.LAYER_ENABLE['patterns']:
            print(f"  → Layer 2: {narrative} pattern")
            # Use refactored pattern coordinator
            params = {
                'sample_rate': self.sample_rate,
                'section_start_time': 0.0,
                'filter': final_filter,
                'filter_env': filter_env_amount,
                'resonance': final_resonance,
                'note_duration': note_duration,
                'tension': tension,
                'config': self.config,
                'mix_level': 1.0,
                'overall_narrative': self.overall_narrative,
            }

            section_pattern = self.pattern_coordinator.generate_pattern(
                narrative=narrative,
                duration=section_duration,
                scale=scale,
                params=params
            )

            # Ensure correct length
            if len(section_pattern) < total_samples:
                temp = np.zeros(total_samples, dtype=np.float32)
                temp[:len(section_pattern)] = section_pattern
                section_pattern = temp
            elif len(section_pattern) > total_samples:
                section_pattern = section_pattern[:total_samples]
        else:
            print(f"  → Layer 2: (muted)")

        # Mix layers 1 and 2
        drone_contribution = base_drone * self.config.LAYER_MIXING['drone_in_supersaw']
        pattern_contribution = section_pattern * self.config.LAYER_MIXING['pattern_in_supersaw']
        mixed_signal = drone_contribution + pattern_contribution

        # Apply section envelope
        section_envelope = np.ones(total_samples)
        fade_samples = int(self.config.TIMING['section_fade_sec'] * self.sample_rate)
        section_envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        section_envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        samples = mixed_signal * section_envelope * self.config.MIXING['section_level'] * volume_multiplier

        # ENTROPY CURVE CALCULATION (Laurie Spiegel-inspired)
        # Calculate position complexity to drive Layer 3 predictability
        start_ply = section.get('start_ply', 1)
        end_ply = section.get('end_ply', start_ply + 20)
        entropy_curve = None
        if self.entropy_calculator is not None:

            try:
                # Calculate raw entropy
                raw_entropy = self.entropy_calculator.calculate_combined_entropy(
                    start_ply,
                    end_ply,
                    weights=self.config.ENTROPY_CONFIG['weights']
                )

                # Apply smoothing to avoid sudden jumps
                if len(raw_entropy) > 3:
                    sigma = self.config.ENTROPY_CONFIG['smoothing_sigma']
                    entropy_curve = gaussian_filter1d(raw_entropy, sigma=sigma)
                else:
                    entropy_curve = raw_entropy

                avg_entropy = np.mean(entropy_curve)
                # Don't print here - will be printed under Layer 3 header

            except Exception as e:
                # Don't print here - will be printed under Layer 3 header
                entropy_curve = None

        # LAYER 3: CONTINUOUS SEQUENCER
        sequencer_layer = np.zeros_like(samples)
        filtered_sequence = np.zeros_like(samples)

        if self.config.LAYER_ENABLE['sequencer']:
            amp_env = self.config.SEQUENCER_SYNTH['amp_env']
            filter_env = self.config.SEQUENCER_SYNTH['filter_env']
            print(f"  → Layer 3: Sequencer ({key_moments_count} key moments) | wave: supersaw | amp: {amp_env[0]:.3f} {amp_env[1]:.2f} {amp_env[2]:.2f} {amp_env[3]:.2f} | filter: {filter_env[0]:.3f} {filter_env[1]:.2f} {filter_env[2]:.2f} {filter_env[3]:.2f}")
            # Print entropy info if available
            if entropy_curve is not None and len(entropy_curve) > 0:
                avg_entropy = np.mean(entropy_curve)
                print(f"      entropy: mean={avg_entropy:.3f}, range=[{np.min(entropy_curve):.3f}, {np.max(entropy_curve):.3f}]")
        else:
            print(f"  → Layer 3: (muted)")

        if self.config.LAYER_ENABLE['sequencer']:
            bpm = section.get('bpm', self.config.DEFAULT_BPM)
            beat_duration = 60.0 / bpm
            sixteenth_duration = beat_duration / 4

            def midi_to_freq(midi_note):
                return 440.0 * 2**((midi_note - 69) / 12.0)

            # Initialize pattern evolution
            current_pattern = self.config.SEQUENCER_PATTERNS['PULSE']
            current_root = 60
            filter_frequency = 1000
            filter_target = 3000
            development_count = 0

            # Debug: Show base pattern
            pattern_preview = [f"{x:02d}" if x is not None else '__' for x in current_pattern[:8]]
            print(f"      base pattern: PULSE [{','.join(pattern_preview)}] at {self.config.MOMENT_EVENT_PARAMS['base_pattern_level']*100:.0f}% volume")

            # OUTRO: Seeded variation based on total plies
            if section.get('name') == 'OUTRO':
                # Use total plies as seed for variation
                variation_seed = self.total_plies % 8  # 8 different variations

                # Choose ending arpeggio pattern based on game length
                if self.total_plies < 40:  # Short game
                    # Simple descending pattern
                    current_pattern = [0, None, -2, None, -4, None, -5, None, -7, None, None, None, None, None, None, None]
                elif self.total_plies < 80:  # Normal game
                    # Classic resolution arpeggio (varies by seed)
                    if variation_seed % 2 == 0:
                        current_pattern = [0, 4, 7, 12, 7, 4, 0, None, None, None, None, None, None, None, None, None]
                    else:
                        current_pattern = [0, -5, -3, 0, None, None, None, None, None, None, None, None, None, None, None, None]
                else:  # Long game
                    # Extended elaborate phrase (varies by seed)
                    patterns = [
                        [0, 2, 4, 5, 7, 9, 7, 5, 4, 2, 0, None, None, None, None, None],  # Ascending-descending
                        [0, 7, 0, 5, 0, 4, 0, None, None, None, None, None, None, None, None, None],  # Pedal tone
                        [0, -2, -4, -5, -7, -9, -12, None, None, None, None, None, None, None, None, None],  # Pure descent
                    ]
                    current_pattern = patterns[variation_seed % 3]

                # Slow down the outro (half tempo)
                sixteenth_duration *= 2.0

                print(f"  → Layer 3: Outro arpeggio (variation {variation_seed}, game length: {self.total_plies} plies)")

            # Process key moments as events with duration and emphasis
            moment_events = []
            mep = self.config.MOMENT_EVENT_PARAMS

            for moment in section.get('key_moments', []):
                moment_time = moment.get('second', moment.get('ply', 0))
                duration_str = section.get('duration', '0:10')
                section_start = int(duration_str.split(':')[0]) if ':' in duration_str else 0
                relative_time = moment_time - section_start

                if 0 <= relative_time <= section_duration:
                    score = moment.get('score', 5)
                    moment_type = moment.get('type', 'UNKNOWN')

                    # Calculate duration based on score
                    duration_mult = 1.0 + (score / 10.0) * mep['score_duration_mult']
                    duration = mep['base_duration_sec'] * duration_mult
                    duration = max(mep['min_duration_sec'], min(duration, mep['max_duration_sec']))

                    # Calculate mix amount based on score
                    mix_amount = mep['base_mix_amount'] + (score * mep['score_mix_mult'])
                    mix_amount = min(mix_amount, mep['max_mix_amount'])

                    # Calculate filter modulation based on score
                    filter_mod = mep['filter_mod_base'] + (score * mep['filter_mod_per_score'])

                    moment_events.append({
                        'start_time': relative_time,
                        'end_time': relative_time + duration,
                        'type': moment_type,
                        'score': score,
                        'mix_amount': mix_amount,
                        'filter_mod': filter_mod,
                        'start_sample': int(relative_time * self.sample_rate),
                        'end_sample': int((relative_time + duration) * self.sample_rate),
                    })

            moment_events.sort(key=lambda x: x['start_time'])

            # Adjust overlapping moments: shorten them to fit, but NEVER skip
            # This preserves critical chess narratives while leaving some heartbeat room
            adjusted_events = []

            for i, event in enumerate(moment_events):
                if i == 0:
                    # First event - use as is
                    adjusted_events.append(event)
                else:
                    prev_event = adjusted_events[-1]

                    # If this moment starts before previous ends, adjust durations
                    if event['start_time'] < prev_event['end_time']:
                        # Calculate overlap
                        overlap = prev_event['end_time'] - event['start_time']

                        # Shorten previous event to end right when this one starts
                        prev_event['end_time'] = event['start_time']
                        prev_event['end_sample'] = int(prev_event['end_time'] * self.sample_rate)

                        # Keep current event as is (or slightly shorten if needed for next)
                        adjusted_events.append(event)
                    else:
                        # No overlap - keep as is
                        adjusted_events.append(event)

            moment_events = adjusted_events

            # Debug: Print moment event timeline with synthesis info
            if moment_events:
                amp_env = self.config.SEQUENCER_SYNTH['amp_env']
                filter_env = self.config.SEQUENCER_SYNTH['filter_env']
                print(f"  → Moment events ({len(moment_events)}) | wave: supersaw | amp: {amp_env[0]:.3f} {amp_env[1]:.2f} {amp_env[2]:.2f} {amp_env[3]:.2f} | filter: {filter_env[0]:.3f} {filter_env[1]:.2f} {filter_env[2]:.2f} {filter_env[3]:.2f}")
                for event in moment_events:
                    pattern_name = event['type']
                    pattern_preview = ""
                    if pattern_name in self.config.SEQUENCER_PATTERNS:
                        pat = self.config.SEQUENCER_PATTERNS[pattern_name]
                        if isinstance(pat, dict):  # DEVELOPMENT has sub-patterns
                            pattern_preview = "prog"
                        else:
                            # Show first 6 steps with zero-padded double digits
                            preview = [f"{x:02d}" if x is not None else '__' for x in pat[:6]]
                            pattern_preview = ','.join(preview)

                    print(f"      {event['type']:<18} t={event['start_time']:05.1f}-{event['end_time']:05.1f}s "
                          f"| dur={event['end_time']-event['start_time']:.1f}s score={event['score']} "
                          f"mix={event['mix_amount']:.2f} flt={event['filter_mod']:.0f}Hz | pat:[{pattern_preview},...]")

            # Generate sequence with event-based blending
            samples_per_step = int(sixteenth_duration * self.sample_rate)
            total_steps = int(section_duration / sixteenth_duration)

            # Helper function to get active moments at a given time
            def get_active_moments(current_time):
                active = []
                for event in moment_events:
                    if event['start_time'] <= current_time <= event['end_time']:
                        # Calculate blend factor based on crossfade
                        crossfade_dur = mep['crossfade_duration_sec']
                        time_in_event = current_time - event['start_time']
                        time_to_end = event['end_time'] - current_time

                        # Fade in at start
                        if time_in_event < crossfade_dur:
                            fade_in = time_in_event / crossfade_dur
                        else:
                            fade_in = 1.0

                        # Fade out at end
                        if time_to_end < crossfade_dur:
                            fade_out = time_to_end / crossfade_dur
                        else:
                            fade_out = 1.0

                        blend_factor = min(fade_in, fade_out)

                        active.append({
                            **event,
                            'blend_factor': blend_factor
                        })
                return active

            full_sequence = []
            for i in range(total_steps):
                step_index = i % 16
                current_time = i * sixteenth_duration
                current_sample = int(current_time * self.sample_rate)

                # ENTROPY-DRIVEN NOTE SELECTION
                current_ply = start_ply + int(current_time)
                entropy_value = 0.5

                if entropy_curve is not None and len(entropy_curve) > 0:
                    ply_offset = current_ply - start_ply
                    if 0 <= ply_offset < len(entropy_curve):
                        entropy_value = entropy_curve[ply_offset]

                # Get active moments at this time
                active_moments = get_active_moments(current_time)

                # Start with base pattern
                base_pattern = current_pattern
                base_interval = base_pattern[step_index]

                # Blend with moment patterns
                if active_moments:
                    # Use highest score moment as primary
                    primary_moment = max(active_moments, key=lambda m: m['score'])
                    moment_type = primary_moment['type']

                    # Get moment pattern
                    if moment_type in self.config.SEQUENCER_PATTERNS:
                        if moment_type == 'DEVELOPMENT':
                            # Handle DEVELOPMENT progression
                            development_count += 1
                            if development_count == 1:
                                moment_pattern = self.config.SEQUENCER_PATTERNS['DEVELOPMENT']['early']
                            elif development_count == 2:
                                moment_pattern = self.config.SEQUENCER_PATTERNS['DEVELOPMENT']['mid']
                            else:
                                moment_pattern = self.config.SEQUENCER_PATTERNS['DEVELOPMENT']['full']
                            current_root = min(current_root + 7, 72)
                            filter_target = min(filter_target + 500, 5000)
                        else:
                            moment_pattern = self.config.SEQUENCER_PATTERNS[moment_type]

                            # Adjust root and filter based on moment type
                            if moment_type in ['BLUNDER', 'MISTAKE', 'INACCURACY']:
                                current_root = max(current_root - 12, 36)
                                filter_target = 1000 + primary_moment['filter_mod']
                            elif moment_type in ['BRILLIANT', 'STRONG']:
                                current_root = min(current_root + 12, 84)
                                filter_target = 3000 + primary_moment['filter_mod']

                        moment_interval = moment_pattern[step_index]

                        # Blend intervals based on mix amount and blend factor
                        total_blend = primary_moment['mix_amount'] * primary_moment['blend_factor']

                        if base_interval is not None and moment_interval is not None:
                            # Both patterns have notes - blend them
                            note_interval = int(base_interval * (1 - total_blend) + moment_interval * total_blend)
                        elif moment_interval is not None:
                            # Only moment has note
                            note_interval = moment_interval if total_blend > 0.5 else base_interval
                        else:
                            # Use base
                            note_interval = base_interval
                    else:
                        note_interval = base_interval
                else:
                    note_interval = base_interval

                # ENTROPY-DRIVEN NOTE MODIFICATION
                # Modify note interval based on position complexity
                if note_interval is not None:
                    # Get entropy-appropriate note pool
                    ec = self.config.ENTROPY_CONFIG
                    low_thresh = ec['low_threshold']
                    high_thresh = ec['high_threshold']

                    if entropy_value < low_thresh:
                        # Low entropy: simple, predictable (root-fifth only)
                        available_intervals = ec['note_pools']['low']
                    elif entropy_value < high_thresh:
                        # Medium entropy: moderate complexity
                        available_intervals = ec['note_pools']['medium']
                    else:
                        # High entropy: full chromatic complexity
                        available_intervals = ec['note_pools']['high']

                    # If pattern note is None or outside available pool, pick random from pool
                    # Otherwise use pattern note if it's in the available pool
                    if note_interval not in available_intervals:
                        # Replace with random note from entropy-appropriate pool
                        note_interval = self.rng.choice(available_intervals)

                if note_interval is None:
                    midi_note = None
                else:
                    midi_note = current_root + note_interval

                full_sequence.append(midi_note)

                # ENTROPY-DRIVEN FILTER MODULATION
                # High entropy = faster filter changes
                filter_rate = 0.02 + entropy_value * 0.05  # 0.02-0.07
                filter_frequency += (filter_target - filter_frequency) * filter_rate

            # Generate audio from sequence with portamento
            prev_freq = None
            for i, midi_note in enumerate(full_sequence):
                if i * samples_per_step >= len(sequencer_layer):
                    break

                if midi_note is None:
                    prev_freq = None
                    continue

                target_freq = midi_to_freq(midi_note)

                # Get entropy for this note position
                note_time = i * sixteenth_duration
                note_ply = start_ply + int(note_time)
                note_entropy = 0.5

                if entropy_curve is not None and len(entropy_curve) > 0:
                    ply_offset = note_ply - start_ply
                    if 0 <= ply_offset < len(entropy_curve):
                        note_entropy = entropy_curve[ply_offset]

                # ENTROPY-DRIVEN PORTAMENTO
                # Low entropy = smooth long glides (flowing)
                # High entropy = short jumpy glides (nervous)
                # Add portamento (frequency glide) from previous note
                if prev_freq is not None:
                    # Create frequency glide from prev to target
                    ec = self.config.ENTROPY_CONFIG
                    glide_reduction = note_entropy * ec['glide_reduction_max']  # 0 to 0.5
                    glide_time = sixteenth_duration * 0.3 * (1.0 - glide_reduction)  # Reduce glide at high entropy
                    glide_samples = int(glide_time * self.sample_rate)
                    freq_curve = np.linspace(prev_freq, target_freq, glide_samples)

                    # Generate glide with changing frequency
                    t = np.arange(glide_samples) / self.sample_rate
                    phase = np.zeros(glide_samples)
                    for s in range(1, glide_samples):
                        phase[s] = phase[s-1] + 2 * np.pi * freq_curve[s] / self.sample_rate
                    glide_audio = np.sin(phase) * 0.3  # Lower volume for glide

                    # Add glide to start of note position
                    start_pos = int(i * samples_per_step * self.config.TIMING['sequencer_overlap'])
                    end_pos = min(start_pos + len(glide_audio), len(sequencer_layer))
                    if end_pos > start_pos:
                        sequencer_layer[start_pos:end_pos] += glide_audio[:end_pos-start_pos] * self.config.LAYER_MIXING['sequencer_note_level']

                # ENTROPY-DRIVEN RHYTHM VARIATION
                # High entropy = more timing variation (less predictable)
                ec = self.config.ENTROPY_CONFIG
                rhythm_var = note_entropy * ec['rhythm_variation_max']  # 0 to 0.5
                duration_multiplier = 1.0 + self.rng.uniform(-rhythm_var, rhythm_var)
                actual_duration = sixteenth_duration * duration_multiplier

                # OUTRO: Use last section's envelope for continuity
                if section.get('name') == 'OUTRO':
                    amp_env_to_use = self.last_layer3_amp_env
                    filter_env_to_use = self.last_layer3_filter_env
                else:
                    amp_env_to_use = self.config.SEQUENCER_SYNTH['amp_env']
                    filter_env_to_use = self.config.SEQUENCER_SYNTH['filter_env']
                    # Store for potential outro use
                    self.last_layer3_amp_env = amp_env_to_use
                    self.last_layer3_filter_env = filter_env_to_use

                # Generate main note
                # Check if this is base heartbeat (no active moments) and use fixed heartbeat filter
                is_heartbeat = len(active_moments) == 0 and self.config.SEQUENCER_SYNTH.get('heartbeat_use_fixed', False)

                if is_heartbeat:
                    # Use fixed muffled heartbeat sound (like stethoscope)
                    filter_to_use = self.config.SEQUENCER_SYNTH['heartbeat_filter']
                    resonance_to_use = self.config.SEQUENCER_SYNTH['heartbeat_resonance']
                else:
                    # Use normal evolving filter/resonance
                    filter_to_use = self.config.SEQUENCER_SYNTH['filter_base_start'] + (i * self.config.SEQUENCER_SYNTH['filter_increment_per_step'])
                    resonance_to_use = self.config.SEQUENCER_SYNTH['resonance']

                note_audio = self.synth_layer3.supersaw(
                    target_freq,
                    actual_duration,
                    detune_cents=self.config.SEQUENCER_SYNTH['detune_cents'],
                    filter_base=filter_to_use,
                    filter_env_amount=self.config.SEQUENCER_SYNTH['filter_env_amount'],
                    resonance=resonance_to_use,
                    amp_env=amp_env_to_use,
                    filter_env=filter_env_to_use
                )

                start_pos = int(i * samples_per_step * self.config.TIMING['sequencer_overlap'])
                end_pos = min(start_pos + len(note_audio), len(sequencer_layer))

                if end_pos > start_pos:
                    sequencer_layer[start_pos:end_pos] += note_audio[:end_pos-start_pos] * self.config.LAYER_MIXING['sequencer_note_level']

                # ENTROPY-DRIVEN HARMONIC DENSITY
                # High entropy = add random harmony notes (cluster effect)
                harmony_threshold = ec['harmony_probability_threshold']
                if note_entropy > harmony_threshold and self.rng.random() < (note_entropy - harmony_threshold):
                    # Add a harmony note (third, fourth, or fifth)
                    harmony_intervals = [3, 4, 7]  # Musical intervals in semitones
                    harmony_interval = self.rng.choice(harmony_intervals)
                    harmony_freq = target_freq * (2 ** (harmony_interval / 12.0))

                    harmony_audio = self.synth_layer3.supersaw(
                        harmony_freq,
                        actual_duration * 0.8,  # Slightly shorter
                        detune_cents=self.config.SEQUENCER_SYNTH['detune_cents'],
                        filter_base=self.config.SEQUENCER_SYNTH['filter_base_start'] + (i * self.config.SEQUENCER_SYNTH['filter_increment_per_step']),
                        filter_env_amount=self.config.SEQUENCER_SYNTH['filter_env_amount'],
                        resonance=self.config.SEQUENCER_SYNTH['resonance'],
                        amp_env=self.config.SEQUENCER_SYNTH['amp_env'],
                        filter_env=self.config.SEQUENCER_SYNTH['filter_env']
                    )

                    harmony_end = min(start_pos + len(harmony_audio), len(sequencer_layer))
                    if harmony_end > start_pos:
                        sequencer_layer[start_pos:harmony_end] += harmony_audio[:harmony_end-start_pos] * self.config.LAYER_MIXING['sequencer_note_level'] * 0.5

                prev_freq = target_freq

            # Apply global filter sweep
            sweep_length = len(sequencer_layer)
            filter_sweep = np.zeros(sweep_length)

            for i in range(sweep_length):
                progress_local = i / sweep_length
                lfo = np.sin(2 * np.pi * self.config.SEQUENCER_SYNTH['global_filter_lfo_amount'] * progress_local)
                filter_sweep[i] = self.config.SEQUENCER_SYNTH['global_filter_base'] + self.config.SEQUENCER_SYNTH['global_filter_lfo_amount'] * lfo + self.config.SEQUENCER_SYNTH['global_filter_sweep_amount'] * progress_local

            filtered_sequence = np.zeros_like(sequencer_layer)
            chunk_size = int(self.config.TIMING['chunk_size_samples'])
            for i in range(0, len(sequencer_layer), chunk_size):
                chunk_end = min(i + chunk_size, len(sequencer_layer))
                chunk = sequencer_layer[i:chunk_end]

                avg_cutoff = np.mean(filter_sweep[i:chunk_end])

                if len(chunk) > 0:
                    filtered_chunk = self.synth_layer3.moog_filter(chunk, cutoff_hz=avg_cutoff, resonance=self.config.SEQUENCER_SYNTH['global_filter_resonance'])
                    filtered_sequence[i:chunk_end] = filtered_chunk

            # Apply sidechain compression
            sequencer_envelope = np.abs(filtered_sequence)
            smoothing = int(self.config.SEQUENCER_SYNTH['smoothing_window_sec'] * self.sample_rate)
            if smoothing > 0:
                sequencer_envelope = np.convolve(sequencer_envelope, np.ones(smoothing) / smoothing, mode='same')

            max_env = np.max(sequencer_envelope)
            if max_env > 0:
                sequencer_envelope = sequencer_envelope / max_env

            ducking = 1.0 - (sequencer_envelope * self.config.MIXING['ducking_amount'])
            samples = samples * ducking

        # Return layers separately for stereo processing
        # Store layer 3 separately for dynamic panning
        layer_3 = filtered_sequence * self.config.MIXING['filtered_sequence_level'] if self.config.LAYER_ENABLE['sequencer'] else np.zeros_like(samples)

        # OUTRO: Apply decay to Layer 3 so it fades with Layers 1+2
        if section.get('name') == 'OUTRO' and len(layer_3) > 0:
            # Match the decay curve of Layers 1+2 for coherent fadeout
            decay_curve = np.exp(np.linspace(0, -3.5, len(layer_3)))
            layer_3 *= decay_curve
            print(f"  → Layer 3: Applied matching decay for coherent outro")

        # Layers 1+2 combined (will be centered/static stereo)
        layers_1_2 = samples

        # OUTRO: Apply gentle exponential decay to drone/patterns
        if section.get('name') == 'OUTRO' and len(layers_1_2) > 0:
            # Gentler decay (-3.5 over 6 seconds) so it doesn't end too abruptly
            decay_curve = np.exp(np.linspace(0, -3.5, len(layers_1_2)))
            layers_1_2 *= decay_curve
            print(f"  → Layer 1+2: Applied gentle exponential decay")

        # Return as dict for stereo processing
        return {
            'layers_1_2': np.array(layers_1_2),
            'layer_3': np.array(layer_3),
            'entropy_curve': entropy_curve if entropy_curve is not None else np.zeros(1)
        }

    def compose(self):
        """Create the full composition"""
        print("\n♫ CHESS TO MUSIC SYNTHESIS")
        print("━" * 50)
        print(f"Game: {self.tags.get('game_result', '?')} | ECO: {self.eco} | Scale: {self.base_params['scale'].title()}")
        print(f"Overall Narrative: {self.overall_narrative}")
        print(f"Base Waveform: {self.base_params['base_waveform']} | Detune: {self.base_params['detune_start']}→{self.base_params['detune_end']} cents")

        sections = self.tags.get('sections', [])
        total_sections = len(sections)
        section_audios = []

        print(f"\nSynthesizing {total_sections} sections with {self.config.TIMING['section_crossfade_sec']}s crossfades...")
        for i, section in enumerate(sections):
            section_data = self.compose_section(section, i, total_sections)
            section_audios.append(section_data)

            # Show crossfade indicator for next section
            if i < total_sections - 1:
                next_section_name = sections[i + 1]['name']
                print(f"  ↓ Crossfading to {next_section_name}...")

        # Crossfade sections together (mono mix for now)
        crossfade_samples = int(self.sample_rate * self.config.TIMING['section_crossfade_sec'])

        # Combine layers_1_2
        layers_1_2 = section_audios[0]['layers_1_2']
        for i in range(1, len(section_audios)):
            next_section = section_audios[i]['layers_1_2']

            if len(layers_1_2) > crossfade_samples and len(next_section) > crossfade_samples:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                layers_1_2[-crossfade_samples:] *= fade_out
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                next_section[:crossfade_samples] *= fade_in
                layers_1_2 = np.concatenate([
                    layers_1_2[:-crossfade_samples],
                    layers_1_2[-crossfade_samples:] + next_section[:crossfade_samples],
                    next_section[crossfade_samples:]
                ])
            else:
                layers_1_2 = np.concatenate([layers_1_2, next_section])

        # Combine layer_3
        layer_3 = section_audios[0]['layer_3']
        for i in range(1, len(section_audios)):
            next_section = section_audios[i]['layer_3']

            if len(layer_3) > crossfade_samples and len(next_section) > crossfade_samples:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                layer_3[-crossfade_samples:] *= fade_out
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                next_section[:crossfade_samples] *= fade_in
                layer_3 = np.concatenate([
                    layer_3[:-crossfade_samples],
                    layer_3[-crossfade_samples:] + next_section[:crossfade_samples],
                    next_section[crossfade_samples:]
                ])
            else:
                layer_3 = np.concatenate([layer_3, next_section])

        # Combine mono for now (will convert to stereo next)
        composition = layers_1_2 + layer_3

        # Convert to stereo if configured (BEFORE normalization)
        if self.config.WAV_OUTPUT['channels'] == 2:
            print(f"\n{'━' * 50}")
            print("STEREO CONVERSION")

            # Layer 1+2: Centered with moderate width
            avg_tension = np.mean([s.get('tension', 0.5) for s in sections])
            width_12 = self.config.STEREO_CONFIG['min_width'] + (avg_tension * 0.3)
            stereo_12 = stereo_width(layers_1_2, width=width_12, center_pan=0.0)
            print(f"  Layer 1+2: Centered, width={width_12:.2f}")

            # Layer 3: ENTROPY-DRIVEN dynamic panning (Spiegel principle)
            # Combine entropy curves from all sections
            entropy_combined = np.concatenate([s['entropy_curve'] for s in section_audios])

            # Resample entropy to match layer_3 length
            if len(entropy_combined) > 1:
                entropy_resampled = np.interp(
                    np.linspace(0, len(entropy_combined)-1, len(layer_3)),
                    np.arange(len(entropy_combined)),
                    entropy_combined
                )
            else:
                entropy_resampled = np.ones(len(layer_3)) * 0.5

            # Map entropy to pan: low=centered, high=wide movement
            # Use absolute deviation from center for pan amount
            # Add slow oscillation for direction
            pan_amount = np.abs(entropy_resampled - 0.5) * 2.0 * self.config.STEREO_CONFIG['entropy_pan_amount']
            pan_direction = np.sin(np.linspace(0, 3*np.pi, len(layer_3)))  # Slow L/R oscillation
            pan_curve = pan_amount * pan_direction
            pan_curve = np.clip(pan_curve, -1.0, 1.0)

            stereo_3 = apply_dynamic_pan(layer_3, pan_curve)
            avg_entropy = np.mean(entropy_resampled)
            print(f"  Layer 3: Entropy-driven panning (avg={avg_entropy:.3f}, complexity→position)")

            # Mix stereo layers
            composition = mix_stereo([stereo_12, stereo_3])
            print(f"  Result: Layer 3 travels across stereo field dynamically!")

        # Master bus processing - AFTER stereo conversion
        print(f"\n{'━' * 50}")
        print("MASTER BUS")

        # Measure pre-normalized levels
        pre_peak = np.max(np.abs(composition))
        pre_peak_db = 20 * np.log10(pre_peak) if pre_peak > 0 else -100
        pre_rms = np.sqrt(np.mean(composition**2))
        pre_rms_db = 20 * np.log10(pre_rms) if pre_rms > 0 else -100

        # Calculate how much gain needed to reach -3dBFS target
        target_db = -3.0
        target_linear = 10 ** (target_db / 20.0)

        if pre_peak > 0:
            normalization_gain = target_linear / pre_peak
            composition = composition * normalization_gain
        else:
            normalization_gain = 1.0

        # Final measurements
        final_peak = np.max(np.abs(composition))
        final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -100
        final_rms = np.sqrt(np.mean(composition**2))
        final_rms_db = 20 * np.log10(final_rms) if final_rms > 0 else -100
        crest_factor_db = final_peak_db - final_rms_db
        clipped_samples = np.sum(np.abs(composition) > 0.99)
        clipped_pct = (clipped_samples / len(composition)) * 100

        print(f"  Pre-normalization peak: {pre_peak_db:.1f} dBFS")
        print(f"  Normalization gain: {20*np.log10(normalization_gain):.1f} dB")
        print(f"  Final peak: {final_peak_db:.1f} dBFS (target: {target_db:.1f} dBFS)")
        print(f"  Final RMS: {final_rms_db:.1f} dBFS")
        print(f"  Crest factor: {crest_factor_db:.1f} dB")
        print(f"  Clipped samples: {clipped_samples} ({clipped_pct:.4f}%)")

        print(f"\n{'━' * 50}")
        print(f"✓ Synthesis complete: {len(composition)/self.sample_rate:.1f} seconds")
        return composition

    def save(self, filename='chess_synth.wav'):
        """Save the composition (stereo or mono based on config)"""
        composition = self.compose()

        with wave.open(filename, 'w') as wav:
            wav.setnchannels(self.config.WAV_OUTPUT['channels'])
            wav.setsampwidth(self.config.WAV_OUTPUT['sample_width'])
            wav.setframerate(self.sample_rate)

            if self.config.WAV_OUTPUT['channels'] == 2:
                # Stereo: composition is (N, 2) array, interleave L/R
                for frame in composition:
                    left_sample = int(frame[0] * self.config.WAV_OUTPUT['amplitude_multiplier'])
                    right_sample = int(frame[1] * self.config.WAV_OUTPUT['amplitude_multiplier'])

                    left_sample = max(self.config.WAV_OUTPUT['clamp_min'],
                                    min(self.config.WAV_OUTPUT['clamp_max'], left_sample))
                    right_sample = max(self.config.WAV_OUTPUT['clamp_min'],
                                     min(self.config.WAV_OUTPUT['clamp_max'], right_sample))

                    # Interleave: L, R, L, R, ...
                    wav.writeframes(struct.pack('<hh', left_sample, right_sample))
            else:
                # Mono: legacy support
                for sample in composition:
                    int_sample = int(sample * self.config.WAV_OUTPUT['amplitude_multiplier'])
                    int_sample = max(self.config.WAV_OUTPUT['clamp_min'],
                                    min(self.config.WAV_OUTPUT['clamp_max'], int_sample))
                    wav.writeframes(struct.pack('<h', int_sample))

        # Get file size
        import os
        file_size = os.path.getsize(filename)
        size_mb = file_size / (1024 * 1024)

        print(f"  Output: {filename} ({size_mb:.1f} MB, {self.config.WAV_OUTPUT['channels']} ch)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Synthesize chess game music')
    parser.add_argument('tags_file', help='JSON file with narrative tags')
    parser.add_argument('--only-section', nargs='+',
                        help='Only render specific sections by name. Example: --only-section OPENING MIDDLEGAME_1')
    parser.add_argument('-o', '--output', default='chess_synth.wav', help='Output filename')

    args = parser.parse_args()

    with open(args.tags_file, 'r') as f:
        tags = json.load(f)

    from synth_config import SynthConfig
    config = SynthConfig()

    # Handle section filtering
    if args.only_section:
        # Filter sections to only include requested ones
        section_names_upper = [s.upper() for s in args.only_section]
        original_sections = tags.get('sections', [])
        filtered_sections = [s for s in original_sections if s.get('name', '').upper() in section_names_upper]

        if not filtered_sections:
            print(f"Warning: No sections matched {args.only_section}")
            print(f"Available sections: {[s.get('name') for s in original_sections]}")
            sys.exit(1)

        tags['sections'] = filtered_sections
        print(f"Rendering {len(filtered_sections)} section(s): {[s.get('name') for s in filtered_sections]}")

    composer = ChessSynthComposer(tags, config=config)
    composer.save(args.output)


if __name__ == '__main__':
    main()
