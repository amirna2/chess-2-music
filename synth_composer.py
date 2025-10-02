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


# === STEREO UTILITIES ===

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
    delay_samples = int(width * 20)  # Up to 20 samples delay (~0.45ms at 44.1kHz)

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
        self.synth = SubtractiveSynth(self.sample_rate)

        self.total_duration = chess_tags.get('duration_seconds', 60)
        self.total_plies = chess_tags.get('total_plies', 40)
        self.overall_narrative = chess_tags.get('overall_narrative', 'COMPLEX_GAME')
        self.eco = chess_tags.get('eco', 'A00')

        # Seed randomness with ECO code for reproducibility
        self._seed_from_eco(self.eco)

        # LAYER 1: Overall narrative defines the BASE PATCH
        self.base_params = self._get_narrative_base_params()

        # Initialize narrative process
        self.narrative_process = create_narrative_process(
            self.overall_narrative,
            self.total_duration,
            self.total_plies
        )

        # Initialize entropy calculator (Laurie Spiegel-inspired)
        # This requires move data with eval information
        moves = chess_tags.get('moves', [])
        self.entropy_calculator = ChessEntropyCalculator(moves) if moves else None

    def _seed_from_eco(self, eco_code):
        """Convert ECO code to integer seed for reproducible randomness"""
        # ECO format: Letter (A-E) + two digits (00-99)
        # Convert to integer: A00=0, A01=1, ..., E99=599
        if len(eco_code) >= 3:
            letter_value = ord(eco_code[0].upper()) - ord('A')  # 0-4
            number_value = int(eco_code[1:3])  # 0-99
            seed = letter_value * 100 + number_value
        else:
            seed = 0  # Fallback
        np.random.seed(seed)

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
                duration = base_note_dur * np.random.uniform(1.2, 2.5 + tension)
            else:
                duration = base_note_dur * np.random.uniform(0.6, 1.2)

            # Generate note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter: darker on tonic, brighter when exploring
                filter_mult = 0.7 + (current_note_idx / len(scale)) * 0.8

                # Random velocity variation
                velocity = np.random.uniform(0.6, 1.0)

                pattern_note = self.synth.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='pulse',
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * np.random.uniform(0.4, 0.8),
                    resonance=final_resonance * np.random.uniform(0.7, 0.9),
                    amp_env=get_envelope('soft', self.config),
                    filter_env=get_filter_envelope('gentle', self.config)
                )

                # Add to pattern with random velocity
                end_sample = min(current_sample + len(pattern_note), total_samples)
                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * self.config.LAYER_MIXING['pattern_note_level'] * velocity

            # Random pause (more hesitation as time goes on)
            pause_duration = base_note_dur * np.random.uniform(0.2, 0.8 + progress * tension)
            pause_samples = int(pause_duration * self.sample_rate)
            current_sample += note_samples + pause_samples

            # Choose next note using Markov chain
            if current_note_idx < len(transition_matrix):
                probabilities = transition_matrix[current_note_idx]
                current_note_idx = np.random.choice(len(probabilities), p=probabilities)
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
                if np.random.random() < 0.15 - progress * 0.1:  # Less retreat as hunt intensifies
                    current_state = STATE_RETREAT
                    attack_run_length = 0
                elif np.random.random() < 0.05:
                    current_state = STATE_PAUSE
                    attack_run_length = 0
                else:
                    attack_run_length += 1

            elif current_state == STATE_RETREAT:
                # Quick retreat, return to attack
                if np.random.random() < 0.6 + progress * 0.3:  # Return faster as hunt intensifies
                    current_state = STATE_ATTACK

            elif current_state == STATE_PAUSE:
                # Brief pause, then attack
                if np.random.random() < 0.8:
                    current_state = STATE_ATTACK

            # Note selection based on state
            if current_state == STATE_ATTACK:
                # Upward bias (70% up, 20% repeat, 10% down)
                rand = np.random.random()
                if rand < 0.7:
                    # Move up
                    current_note_idx = min(7, current_note_idx + np.random.randint(1, 3))

                    # Octave jumps (increasing probability with progress and attack length)
                    if np.random.random() < 0.1 + progress * 0.2 + attack_run_length * 0.05:
                        current_octave = min(2, current_octave + 1)
                        current_note_idx = np.random.randint(0, 4)  # Reset to lower note after jump

                elif rand < 0.9:
                    # Repeat (stabbing same note)
                    pass
                else:
                    # Small step down
                    current_note_idx = max(0, current_note_idx - 1)

            elif current_state == STATE_RETREAT:
                # Downward motion
                current_note_idx = max(0, current_note_idx - np.random.randint(1, 3))
                if current_note_idx == 0 and current_octave > 0:
                    current_octave -= 1

            elif current_state == STATE_PAUSE:
                # Hold on dominant or tonic
                if np.random.random() < 0.5:
                    current_note_idx = 0  # Tonic
                else:
                    current_note_idx = 4  # Dominant

            # Get frequency
            note_freq = scale[current_note_idx] * (2 ** current_octave)

            # Duration varies by state and progress
            if current_state == STATE_ATTACK:
                # Faster as hunt intensifies
                duration = base_note_dur * np.random.uniform(0.4, 0.8) * (1.0 - progress * 0.3)
            elif current_state == STATE_RETREAT:
                duration = base_note_dur * np.random.uniform(0.5, 1.0)
            else:  # PAUSE
                duration = base_note_dur * np.random.uniform(1.5, 2.5)

            # Velocity varies by state
            if current_state == STATE_ATTACK:
                velocity = np.random.uniform(0.8, 1.0) * (1.0 + progress * 0.3)  # Louder as hunt intensifies
            elif current_state == STATE_RETREAT:
                velocity = np.random.uniform(0.6, 0.8)
            else:  # PAUSE
                velocity = np.random.uniform(0.5, 0.7)

            # Generate note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter and resonance evolve with progress and state
                filter_mult = 1.0 + progress * 2.0  # Much brighter as hunt intensifies
                if current_state == STATE_ATTACK:
                    filter_mult *= np.random.uniform(1.2, 1.5)  # Extra brightness on attack

                resonance_mult = 1.0 + progress * 0.8

                pattern_note = self.synth.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='saw',  # Aggressive, bright
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * np.random.uniform(0.8, 1.2),
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
                if np.random.random() < 0.3 + progress * 0.2:  # More strikes as attack intensifies
                    current_state = STATE_STRIKE
                    strike_count = np.random.randint(2, 5)  # Multiple hammer blows

            elif current_state == STATE_STRIKE:
                strike_count -= 1
                if strike_count <= 0:
                    if progress > 0.6 and np.random.random() < 0.3:
                        current_state = STATE_OVERWHELM  # Climax
                    else:
                        current_state = STATE_ADVANCE

            elif current_state == STATE_OVERWHELM:
                # Stay in overwhelm mode once reached
                pass

            # Note selection based on state
            if current_state == STATE_ADVANCE:
                # Downward movement (pressure building)
                if np.random.random() < 0.7:
                    current_note_idx = max(0, current_note_idx - np.random.randint(1, 3))
                else:
                    current_note_idx = min(7, current_note_idx + 1)  # Occasional upward jab

            elif current_state == STATE_STRIKE:
                # Hammer on low notes (powerful blows)
                current_note_idx = np.random.choice([0, 1, 2])  # Low register only

            elif current_state == STATE_OVERWHELM:
                # Chaotic attacks across entire range
                current_note_idx = np.random.randint(0, 8)

            # Get base frequency
            note_freq = scale[current_note_idx]

            # Octave variations (more powerful with wider range)
            octave_shift = 0.0  # Default no shift
            if current_state == STATE_STRIKE:
                # Strike uses lower octave (bass power)
                if np.random.random() < 0.5:
                    note_freq *= 0.5
                    octave_shift = -1.0
            elif current_state == STATE_OVERWHELM:
                # Overwhelm uses wide octave range
                octave_shift = float(np.random.choice([-1, 0, 0, 1]))  # Bias toward higher
                note_freq *= (2.0 ** octave_shift)

            # Duration varies by state and progress
            if current_state == STATE_ADVANCE:
                duration = base_note_dur * np.random.uniform(0.8, 1.2) * (1.0 - progress * 0.3)
            elif current_state == STATE_STRIKE:
                # Short, sharp attacks
                duration = base_note_dur * np.random.uniform(0.3, 0.5)
            else:  # OVERWHELM
                # Very fast, relentless
                duration = base_note_dur * np.random.uniform(0.2, 0.4) * (1.0 - progress * 0.2)

            # Velocity increases with progress and state
            # Capped to prevent clipping when multiple notes overlap
            if current_state == STATE_ADVANCE:
                velocity = np.random.uniform(0.7, 0.9) * (1.0 + progress * 0.3)
            elif current_state == STATE_STRIKE:
                velocity = 1.0
            else:  # OVERWHELM
                velocity = np.random.uniform(0.9, 1.0)  # No progress multiplier - prevents clipping

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

                pattern_note = self.synth.create_synth_note(
                    freq=note_freq,
                    duration=note_duration_sec,
                    waveform='saw',
                    filter_base=final_filter * filter_mult,
                    filter_env_amount=filter_env_amount * np.random.uniform(1.0, 1.5),
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
                has_chord = (current_state == STATE_STRIKE or current_state == STATE_OVERWHELM) and np.random.random() < 0.6

                # Voice normalization: scale down when multiple voices present
                num_voices = 2 if has_chord else 1
                voice_scale = 1.0 / np.sqrt(num_voices)

                section_pattern[current_sample:end_sample] += pattern_note[:end_sample - current_sample] * level * voice_scale

                if has_chord:
                    # Add fifth or octave
                    chord_interval = np.random.choice([4, 7])  # Perfect fourth or fifth
                    chord_idx = min(7, current_note_idx + chord_interval)
                    chord_freq = scale[chord_idx] * (2 ** octave_shift if current_state == STATE_OVERWHELM else 1.0)

                    chord_note = self.synth.create_synth_note(
                        freq=chord_freq,
                        duration=note_duration_sec,
                        waveform='saw',
                        filter_base=final_filter * filter_mult,
                        filter_env_amount=filter_env_amount * np.random.uniform(1.0, 1.5),
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
                if np.random.random() < 0.2:
                    current_state = STATE_DART
                elif np.random.random() < 0.1:
                    current_state = STATE_SETTLE
            elif current_state == STATE_DART:
                if np.random.random() < 0.6:
                    current_state = STATE_ATTACK
                elif np.random.random() < 0.2:
                    current_state = STATE_SETTLE
            elif current_state == STATE_SETTLE:
                if np.random.random() < 0.8:
                    current_state = STATE_ATTACK

            # Note selection by state
            if current_state == STATE_ATTACK:
                # Aggressive attacks - favor upper register, random leaps
                if np.random.random() < 0.5:
                    current_note_idx = np.random.choice([4, 5, 6, 7])  # Upper half
                else:
                    current_note_idx = np.random.choice([2, 3, 4])  # Middle
            elif current_state == STATE_DART:
                # Random tactical jumps anywhere
                current_note_idx = np.random.randint(0, len(scale))
            elif current_state == STATE_SETTLE:
                # Gravitate to stable notes
                if np.random.random() < 0.6:
                    current_note_idx = 0  # Tonic
                else:
                    current_note_idx = 4  # Dominant

            note_freq = scale[current_note_idx]
            duration = base_note_dur * np.random.uniform(0.6, 1.0)
            velocity = np.random.uniform(0.75, 1.0)

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth.create_synth_note(
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
            duration = base_note_dur * np.random.uniform(0.8, 1.0)  # Shorter notes, more space
            velocity = 0.35 + np.random.uniform(-0.1, 0.15)  # Much quieter (25-50% vs 60-80%)

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth.create_synth_note(
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
                current_note_idx = np.random.choice(len(probabilities), p=probabilities)
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
            duration = base_note_dur * np.random.uniform(0.95, 1.05)

            # Generate note
            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                note_duration_sec = note_samples / self.sample_rate

                # Filter: darker, grounded
                filter_mult = 0.6 + progress * 0.2  # Stays dark

                # Velocity: very stable
                velocity = 0.75 + np.random.uniform(-0.05, 0.05)

                pattern_note = self.synth.create_synth_note(
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
                if advance_progress > 5 and np.random.random() < 0.3:
                    current_state = STATE_CONSOLIDATE
                    advance_progress = 0
                elif progress > 0.7 and np.random.random() < 0.2:
                    current_state = STATE_BREAKTHROUGH
            elif current_state == STATE_CONSOLIDATE:
                if np.random.random() < 0.5:
                    current_state = STATE_ADVANCE
                elif progress > 0.7 and np.random.random() < 0.3:
                    current_state = STATE_BREAKTHROUGH
            elif current_state == STATE_BREAKTHROUGH:
                if np.random.random() < 0.1:
                    current_state = STATE_ADVANCE

            # Note selection by state
            if current_state == STATE_ADVANCE:
                # Build pressure - favor dominant and upper notes, but randomly
                weights = [0.1, 0.05, 0.15, 0.1, 0.25, 0.15, 0.1, 0.1]  # Favor 5th (index 4)
                current_note_idx = np.random.choice(range(len(scale)), p=weights)
            elif current_state == STATE_CONSOLIDATE:
                # Hold stable harmonic notes - tonic, third, fifth
                current_note_idx = np.random.choice([0, 2, 4], p=[0.4, 0.3, 0.3])
            elif current_state == STATE_BREAKTHROUGH:
                # Decisive moves - octave relationships
                current_note_idx = np.random.choice([0, 4, 7], p=[0.3, 0.4, 0.3])  # Tonic, fifth, seventh

            note_freq = scale[current_note_idx]
            duration = base_note_dur * np.random.uniform(0.96, 1.04)
            velocity = 0.55 + progress * 0.15

            note_samples = int(duration * self.sample_rate)
            note_samples = min(note_samples, total_samples - current_sample)

            if note_samples > 0:
                pattern_note = self.synth.create_synth_note(
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
        num_voices = 7
        oscillators = []

        for v in range(num_voices):
            # Fixed detune per voice (in cents)
            detune_cents = (v - 3) * current_base['detune'] / 3
            detune_hz = drone_freq * (2 ** (detune_cents / 1200.0) - 1.0)

            osc = self.synth.oscillator(drone_freq + detune_hz, section_duration, waveform)

            # Apply time-varying amplitude modulation to create beating
            detune_lfo = np.sin(2 * np.pi * 0.03 * np.arange(len(osc)) / self.sample_rate + v * 0.5)
            osc = osc * (1.0 + detune_lfo * 0.1)  # ±10% amplitude modulation

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

            current_cutoff = np.clip(current_cutoff, 20, self.synth.nyquist * 0.95)
            current_resonance = np.clip(current_resonance, 0.1, 4.0)

            # Apply filter
            filtered_chunk = self.synth.moog_filter(chunk, current_cutoff, current_resonance)
            base_drone[i:end] = filtered_chunk

        # Apply amplitude envelope
        amp_env = self.synth.adsr_envelope(len(base_drone), *get_envelope('pad', self.config))
        base_drone = base_drone * amp_env

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
                return self.synth.create_synth_note(
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
                return self.synth.create_synth_note(
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
                return self.synth.create_synth_note(
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
                return self.synth.create_synth_note(
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
                note = self.synth.create_synth_note(
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
                note = self.synth.create_synth_note(
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
                note = self.synth.create_synth_note(
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
                note = self.synth.create_synth_note(
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
            return self.synth.create_synth_note(
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
        return self.synth.create_synth_note(
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
        final_filter = np.clip(final_filter, 20, self.synth.nyquist * 0.95)

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
            print(f"  → Layer 1: Evolving drone ({drone_freq:.1f}Hz base)")
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

        # LAYER 2: Generate RHYTHMIC PATTERNS
        section_pattern = np.zeros(total_samples)

        if self.config.LAYER_ENABLE['patterns']:
            # NARRATIVE-SPECIFIC PATTERN GENERATION
            if narrative == 'COMPLEX_STRUGGLE':
                print(f"  → Layer 2: Generative patterns (Markov chain)")
                section_pattern = self.generate_complex_struggle_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            elif narrative == 'KING_HUNT':
                print(f"  → Layer 2: Generative patterns (State machine: ATTACK/RETREAT/PAUSE)")
                section_pattern = self.generate_king_hunt_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            elif narrative == 'CRUSHING_ATTACK':
                print(f"  → Layer 2: Generative patterns (State machine: ADVANCE/STRIKE/OVERWHELM)")
                section_pattern = self.generate_crushing_attack_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            elif narrative == 'SHARP_THEORY':
                print(f"  → Layer 2: Generative patterns (Sharp opening: Fast arpeggios)")
                section_pattern = self.generate_sharp_theory_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            elif narrative == 'POSITIONAL_THEORY':
                print(f"  → Layer 2: Generative patterns (Positional opening: Methodical build)")
                section_pattern = self.generate_positional_theory_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            elif narrative == 'SOLID_THEORY':
                print(f"  → Layer 2: Generative patterns (Solid opening: Grounded bass)")
                section_pattern = self.generate_solid_theory_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            elif narrative == 'FLAWLESS_CONVERSION':
                print(f"  → Layer 2: Generative patterns (Endgame: Relentless advance)")
                section_pattern = self.generate_flawless_conversion_pattern(
                    section_duration, scale, tension,
                    final_filter, filter_env_amount, final_resonance,
                    note_duration, modulation, total_samples
                )
            else:
                print(f"  → Layer 2: Fixed patterns (fallback)")
        else:
            print(f"  → Layer 2: (muted)")

        if self.config.LAYER_ENABLE['patterns'] and narrative not in ['COMPLEX_STRUGGLE', 'KING_HUNT', 'CRUSHING_ATTACK', 'SHARP_THEORY', 'POSITIONAL_THEORY', 'SOLID_THEORY', 'FLAWLESS_CONVERSION']:
                # DEFAULT FALLBACK (keep old logic for now)
                samples_per_note = int(note_duration * self.sample_rate)
                for i in range(num_notes):
                    start_sample = i * samples_per_note
                    end_sample = min(start_sample + samples_per_note, total_samples)

                    if start_sample < total_samples:
                        note_freq = pattern[i % len(pattern)]

                        # Octave variations
                        if i % pattern_config['octave_up_mod'] == 0:
                            note_freq *= 2
                        elif i % pattern_config['octave_down_mod'] == 0:
                            note_freq *= 0.5

                        note_samples = end_sample - start_sample
                        note_duration_sec = note_samples / self.sample_rate

                        pattern_note = self.synth.create_synth_note(
                            freq=note_freq,
                            duration=note_duration_sec,
                            waveform='saw' if tension > 0.5 else 'pulse',
                            filter_base=final_filter * 1.5,
                            filter_env_amount=filter_env_amount,
                            resonance=final_resonance * 0.7,
                            amp_env=get_envelope('percussive', self.config),
                            filter_env=get_filter_envelope('smooth', self.config)
                        )

                        if len(pattern_note) > 0:
                            actual_samples = min(len(pattern_note), end_sample - start_sample)
                            section_pattern[start_sample:start_sample + actual_samples] += pattern_note[:actual_samples] * self.config.LAYER_MIXING['pattern_note_level']

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
        entropy_curve = None
        if self.entropy_calculator is not None:
            start_ply = section.get('start_ply', 1)
            end_ply = section.get('end_ply', start_ply + 20)

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
                print(f"  Entropy: mean={avg_entropy:.3f}, range=[{np.min(entropy_curve):.3f}, {np.max(entropy_curve):.3f}]")

            except Exception as e:
                print(f"  Entropy: (calculation failed: {e})")
                entropy_curve = None

        # LAYER 3: CONTINUOUS SEQUENCER
        sequencer_layer = np.zeros_like(samples)
        filtered_sequence = np.zeros_like(samples)

        if self.config.LAYER_ENABLE['sequencer']:
            print(f"  → Layer 3: Sequencer ({key_moments_count} key moments, entropy-driven)")
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

            # Process evolution points
            evolution_points = []
            for moment in section.get('key_moments', []):
                moment_time = moment.get('second', moment.get('ply', 0))
                duration_str = section.get('duration', '0:10')
                section_start = int(duration_str.split(':')[0]) if ':' in duration_str else 0
                relative_time = moment_time - section_start

                if 0 <= relative_time <= section_duration:
                    evolution_points.append({
                        'time': relative_time,
                        'type': moment.get('type', 'UNKNOWN'),
                        'sample_pos': int(relative_time * self.sample_rate)
                    })

            evolution_points.sort(key=lambda x: x['time'])
            next_evolution_idx = 0

            # Generate sequence
            samples_per_step = int(sixteenth_duration * self.sample_rate)
            total_steps = int(section_duration / sixteenth_duration)

            full_sequence = []
            for i in range(total_steps):
                step_index = i % 16
                note_interval = current_pattern[step_index]

                current_time = i * sixteenth_duration
                current_sample = int(current_time * self.sample_rate)

                # ENTROPY-DRIVEN NOTE SELECTION (Laurie Spiegel approach)
                # Get entropy value for current position
                current_ply = start_ply + int(current_time)
                entropy_value = 0.5  # Default medium entropy

                if entropy_curve is not None and len(entropy_curve) > 0:
                    ply_offset = current_ply - start_ply
                    if 0 <= ply_offset < len(entropy_curve):
                        entropy_value = entropy_curve[ply_offset]

                # Check for pattern evolution
                if next_evolution_idx < len(evolution_points):
                    if current_sample >= evolution_points[next_evolution_idx]['sample_pos']:
                        evolution = evolution_points[next_evolution_idx]
                        moment_type = evolution['type']

                        if moment_type in self.config.SEQUENCER_PATTERNS:
                            if moment_type == 'DEVELOPMENT':
                                development_count += 1
                                if development_count == 1:
                                    current_pattern = self.config.SEQUENCER_PATTERNS['DEVELOPMENT']['early']
                                elif development_count == 2:
                                    current_pattern = self.config.SEQUENCER_PATTERNS['DEVELOPMENT']['mid']
                                else:
                                    current_pattern = self.config.SEQUENCER_PATTERNS['DEVELOPMENT']['full']
                                current_root = min(current_root + 7, 72)
                                filter_target = min(filter_target + 500, 5000)
                            else:
                                current_pattern = self.config.SEQUENCER_PATTERNS[moment_type]

                                # Adjust root and filter based on moment type
                                if moment_type in ['BLUNDER', 'MISTAKE']:
                                    current_root = max(current_root - 12, 36)
                                    filter_target = 1000
                                elif moment_type in ['BRILLIANT', 'STRONG']:
                                    current_root = min(current_root + 12, 84)
                                    filter_target = 5000

                        next_evolution_idx += 1

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
                        note_interval = np.random.choice(available_intervals)

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
                duration_multiplier = 1.0 + np.random.uniform(-rhythm_var, rhythm_var)
                actual_duration = sixteenth_duration * duration_multiplier

                # Generate main note
                note_audio = self.synth.supersaw(
                    target_freq,
                    actual_duration,
                    detune_cents=self.config.SEQUENCER_SYNTH['detune_cents'],
                    filter_base=self.config.SEQUENCER_SYNTH['filter_base_start'] + (i * self.config.SEQUENCER_SYNTH['filter_increment_per_step']),
                    filter_env_amount=self.config.SEQUENCER_SYNTH['filter_env_amount'],
                    resonance=self.config.SEQUENCER_SYNTH['resonance'],
                    amp_env=self.config.SEQUENCER_SYNTH['amp_env'],
                    filter_env=self.config.SEQUENCER_SYNTH['filter_env']
                )

                start_pos = int(i * samples_per_step * self.config.TIMING['sequencer_overlap'])
                end_pos = min(start_pos + len(note_audio), len(sequencer_layer))

                if end_pos > start_pos:
                    sequencer_layer[start_pos:end_pos] += note_audio[:end_pos-start_pos] * self.config.LAYER_MIXING['sequencer_note_level']

                # ENTROPY-DRIVEN HARMONIC DENSITY
                # High entropy = add random harmony notes (cluster effect)
                harmony_threshold = ec['harmony_probability_threshold']
                if note_entropy > harmony_threshold and np.random.random() < (note_entropy - harmony_threshold):
                    # Add a harmony note (third, fourth, or fifth)
                    harmony_intervals = [3, 4, 7]  # Musical intervals in semitones
                    harmony_interval = np.random.choice(harmony_intervals)
                    harmony_freq = target_freq * (2 ** (harmony_interval / 12.0))

                    harmony_audio = self.synth.supersaw(
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
                    filtered_chunk = self.synth.moog_filter(chunk, cutoff_hz=avg_cutoff, resonance=self.config.SEQUENCER_SYNTH['global_filter_resonance'])
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

        # Add Layer 3 (sequencer) if enabled
        if self.config.LAYER_ENABLE['sequencer']:
            samples = samples + filtered_sequence * self.config.MIXING['filtered_sequence_level']

        return np.array(samples)

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
            section_music = self.compose_section(section, i, total_sections)
            section_audios.append(np.array(section_music))

            # Show crossfade indicator for next section
            if i < total_sections - 1:
                next_section_name = sections[i + 1]['name']
                print(f"  ↓ Crossfading to {next_section_name}...")

        # Crossfade sections together
        crossfade_samples = int(self.sample_rate * self.config.TIMING['section_crossfade_sec'])

        composition = section_audios[0]
        for i in range(1, len(section_audios)):
            next_section = section_audios[i]

            # Create crossfade if sections are long enough
            if len(composition) > crossfade_samples and len(next_section) > crossfade_samples:
                # Fade out end of current composition
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                composition[-crossfade_samples:] *= fade_out

                # Fade in start of next section
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                next_section[:crossfade_samples] *= fade_in

                # Overlap: remove crossfade length from composition, add full next section
                composition = np.concatenate([
                    composition[:-crossfade_samples],
                    composition[-crossfade_samples:] + next_section[:crossfade_samples],
                    next_section[crossfade_samples:]
                ])
            else:
                # Sections too short for crossfade, just concatenate
                composition = np.concatenate([composition, next_section])

        # Master bus processing
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

        # Convert to stereo if configured
        if self.config.WAV_OUTPUT['channels'] == 2:
            print(f"\n{'━' * 50}")
            print("STEREO CONVERSION")
            print(f"  Converting mono to stereo with spatial mapping...")

            # Apply stereo width based on overall tension/narrative
            avg_tension = np.mean([s.get('tension', 0.5) for s in sections])
            width_amount = self.config.STEREO_CONFIG['min_width'] + (avg_tension * (self.config.STEREO_CONFIG['max_width'] - self.config.STEREO_CONFIG['min_width']))

            # Pan based on overall narrative for spatial interest
            # Death spirals pan slightly left (ominous)
            # Masterpieces pan slightly right (triumphant)
            if 'DEATH' in self.overall_narrative or 'DEFEAT' in self.overall_narrative:
                center_pan = -0.3  # Slightly left
            elif 'MASTERPIECE' in self.overall_narrative or 'BRILLIANCY' in self.overall_narrative:
                center_pan = 0.3   # Slightly right
            else:
                center_pan = 0.0   # Centered

            composition = stereo_width(composition, width=width_amount, center_pan=center_pan)
            print(f"  Stereo width: {width_amount:.2f} (tension: {avg_tension:.2f})")
            print(f"  Center pan: {center_pan:.2f} ({self.overall_narrative})")

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
