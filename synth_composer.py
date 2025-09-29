#!/usr/bin/env python3
"""
SUBTRACTIVE SYNTHESIZER FOR CHESS MUSIC
Electronic music synthesis with filters and envelopes
"""

import json
import math
import wave
import struct
import sys
import random
import re
import numpy as np
from abc import ABC, abstractmethod
from scipy import signal

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

class SubtractiveSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0

        # Filter state variables (for continuity between notes)
        self.filter_z1 = 0.0
        self.filter_z2 = 0.0
        self.filter_z3 = 0.0
        self.filter_z4 = 0.0

    def poly_blep(self, dt, phase):
        """
        PolyBLEP (Polynomial Band-Limited Edge Pulse) anti-aliasing
        dt: normalized frequency (freq / sample_rate)
        phase: phase position (0.0 to 1.0)
        """
        if phase < dt:
            # Beginning of period - rising edge
            t = phase / dt
            return t + t - t * t - 1.0
        elif phase > 1.0 - dt:
            # End of period - falling edge
            t = (phase - 1.0) / dt
            return t * t + t + t + 1.0
        else:
            return 0.0

    def oscillator(self, freq, duration, waveform='saw'):
        """Generate band-limited waveforms using PolyBLEP anti-aliasing"""
        num_samples = int(duration * self.sample_rate)
        dt = freq / self.sample_rate  # Normalized frequency

        # Generate sample by sample for proper PolyBLEP application
        signal = np.zeros(num_samples)
        phase = 0.0

        for i in range(num_samples):
            if waveform == 'saw':
                # Band-limited sawtooth
                signal[i] = 2.0 * phase - 1.0
                signal[i] -= self.poly_blep(dt, phase)

            elif waveform == 'square':
                # Band-limited square wave - 50% duty cycle
                if phase < 0.5:
                    signal[i] = 1.0
                else:
                    signal[i] = -1.0

                # Apply PolyBLEP at rising edge (phase = 0)
                signal[i] += self.poly_blep(dt, phase)
                # Apply PolyBLEP at falling edge (phase = 0.5)
                signal[i] -= self.poly_blep(dt, phase - 0.5 if phase >= 0.5 else phase + 0.5)

            elif waveform == 'pulse':
                # Band-limited pulse wave - 25% duty cycle
                width = 0.25
                if phase < width:
                    signal[i] = 1.0
                else:
                    signal[i] = -1.0

                # Apply PolyBLEP at rising edge (phase = 0)
                signal[i] += self.poly_blep(dt, phase)
                # Apply PolyBLEP at falling edge (phase = width)
                phase_shifted = phase - width if phase >= width else phase + (1.0 - width)
                signal[i] -= self.poly_blep(dt, phase_shifted)

            elif waveform == 'triangle':
                # Triangle wave - no PolyBLEP needed (continuous derivatives)
                if phase < 0.5:
                    signal[i] = 4.0 * phase - 1.0      # Rising: -1 to 1
                else:
                    signal[i] = 3.0 - 4.0 * phase      # Falling: 1 to -1

            else:  # sine
                # Sine wave - naturally band-limited
                signal[i] = np.sin(2 * np.pi * phase)

            # Advance phase
            phase += dt
            if phase >= 1.0:
                phase -= 1.0

        return signal

    def moog_filter(self, signal, cutoff_hz, resonance=0.0):
        """
        Stable 4-pole Moog-style low-pass ladder.
        Mild saturation and controlled resonance.
        Works for realtime and offline synthesis.
        """

        if cutoff_hz >= self.nyquist * 0.99:
            return signal

        f = 2.0 * np.sin(np.pi * cutoff_hz / self.sample_rate)  # stable frequency map
        f = np.clip(f, 0.0001, 1.0)
        resonance = np.clip(resonance, 0.0, 4.0)

        # State variables
        y1, y2, y3, y4 = self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4
        out = np.zeros_like(signal)

        for i, x in enumerate(signal):
            # Feedback
            x -= resonance * y4

            # Input soft clipping
            x = np.tanh(x)

            # 4 cascaded one-pole filters
            y1 += f * (x - y1)
            y2 += f * (y1 - y2)
            y3 += f * (y2 - y3)
            y4 += f * (y3 - y4)

            # Output soft clipping for smoother tone
            out[i] = np.tanh(y4)

        # Save states
        self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4 = y1, y2, y3, y4

        # Simple gain compensation
        return out * 1.5


    def supersaw(self, freq, duration,
                 detune_cents=None,
                 filter_base=1500, filter_env_amount=2500,
                 resonance=0.5,
                 amp_env=(0.05, 0.2, 0.9, 0.4),
                 filter_env=(0.01, 0.25, 0.4, 0.4)):
        """
        Roland JP-8000 style supersaw generator.
        Creates a rich detuned saw ensemble with analog-style filtering.
        Returns mono signal for compatibility with existing system.
        """

        # Default detune pattern if not provided
        if detune_cents is None:
            detune_cents = [-12, -7, -3, 3, 7, 12]

        # Base + detuned frequencies
        freqs = [freq * (2 ** (c / 1200.0)) for c in detune_cents] + [freq]

        # Generate saw layers
        layers = []
        for f in freqs:
            layer = self.oscillator(f, duration, 'saw')
            # Randomize phase for natural analog drift
            shift = np.random.randint(0, 100)  # Small phase shift
            layers.append(np.roll(layer, shift))

        # Mix all layers equally
        mixed = np.sum(layers, axis=0) / len(layers)

        # Get filter envelope
        num_samples = len(mixed)
        filt_env = self.filter_envelope(num_samples, *filter_env)

        # Apply time-varying filter
        filtered = np.zeros_like(mixed)
        chunk_size = 512  # Process in chunks for efficiency

        # Save original filter state
        orig_z1, orig_z2, orig_z3, orig_z4 = self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4

        for i in range(0, num_samples, chunk_size):
            end = min(i + chunk_size, num_samples)
            # Calculate cutoff for this chunk
            env_val = np.mean(filt_env[i:end])  # Average envelope value in chunk
            cutoff = filter_base + filter_env_amount * env_val
            cutoff = np.clip(cutoff, 20, self.nyquist * 0.95)

            # Apply filter to chunk (filter state carries through)
            filtered[i:end] = self.moog_filter(mixed[i:end], cutoff, resonance)

        # Restore original filter state for other uses
        self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4 = orig_z1, orig_z2, orig_z3, orig_z4

        # Apply amplitude envelope
        amp_env_signal = self.adsr_envelope(num_samples, *amp_env)
        result = filtered * amp_env_signal

        # Soft limiting to prevent clipping from multiple layers
        result = np.tanh(result * 0.8) * 1.25  # Gentle compression

        return result

    def adsr_envelope(self, num_samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2, curve=0.3):
        """
        ADSR envelope generator with exponential curves for musical sound
        Times in seconds, sustain is level (0-1)
        curve: exponential curve factor (0.1 = gentle, 1.0 = aggressive)
        """
        envelope = np.zeros(num_samples)

        # Convert times to samples
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples

        if sustain_samples < 0:
            # Note too short for full envelope
            sustain_samples = 0
            total = attack_samples + decay_samples + release_samples
            if total > num_samples:
                # Scale everything down proportionally
                scale = num_samples / total
                attack_samples = int(attack_samples * scale)
                decay_samples = int(decay_samples * scale)
                release_samples = num_samples - attack_samples - decay_samples

        current = 0

        # Attack - exponential rise (slow start, fast finish)
        if attack_samples > 0:
            t = np.linspace(0, 1, attack_samples)
            # Use exponential curve: starts slow, accelerates
            envelope[current:current+attack_samples] = np.power(t, 1.0 - curve)
            current += attack_samples

        # Decay - exponential fall (fast start, slow finish)
        if decay_samples > 0 and current < num_samples:
            end = min(current + decay_samples, num_samples)
            t = np.linspace(0, 1, end - current)
            # Exponential decay: starts fast, slows down
            decay_curve = 1.0 - np.power(t, curve)
            envelope[current:end] = 1.0 - decay_curve * (1.0 - sustain)
            current = end

        # Sustain - constant level
        if sustain_samples > 0 and current < num_samples:
            end = min(current + sustain_samples, num_samples)
            envelope[current:end] = sustain
            current = end

        # Release - exponential fall to zero
        if current < num_samples:
            t = np.linspace(0, 1, num_samples - current)
            # Exponential release: fast start, slow finish
            release_curve = np.power(t, curve)
            envelope[current:] = sustain * (1.0 - release_curve)

        return envelope

    def filter_envelope(self, num_samples, attack=0.05, decay=0.2, sustain=0.3, release=0.3):
        """
        Separate envelope for filter cutoff modulation
        This creates the classic 'wow' sound of analog synths
        """
        return self.adsr_envelope(num_samples, attack, decay, sustain, release)

    def pitch_sweep_note(self, freq_start, freq_end, duration,
                        waveform='sine',
                        filter_base=2000,
                        filter_env_amount=1000,
                        resonance=0.3,
                        amp_env=(0.01, 0.05, 0.7, 0.1)):
        """
        Create R2D2-style beep with pitch sweep from freq_start to freq_end
        """
        num_samples = int(duration * self.sample_rate)

        # Create frequency sweep curve
        freq_curve = np.linspace(freq_start, freq_end, num_samples)

        # Generate signal with sweeping frequency
        signal = np.zeros(num_samples)
        phase = 0.0

        for i in range(num_samples):
            current_freq = freq_curve[i]

            # Update phase
            phase += 2 * np.pi * current_freq / self.sample_rate

            # Generate sample based on waveform
            if waveform == 'sine':
                signal[i] = np.sin(phase)
            elif waveform == 'triangle':
                # Triangle wave
                if (phase / (2 * np.pi)) % 1.0 < 0.5:
                    signal[i] = 4.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
                else:
                    signal[i] = 3.0 - 4.0 * ((phase / (2 * np.pi)) % 1.0)
            else:  # Default to sine
                signal[i] = np.sin(phase)

        # Apply simple filter (no envelope needed for R2D2 style)
        filtered = self.moog_filter(signal, filter_base, resonance)

        # Apply amplitude envelope
        amp_envelope = self.adsr_envelope(num_samples, *amp_env)
        output = filtered * amp_envelope

        return output

    def create_synth_note(self, freq, duration,
                         waveform='saw',
                         filter_base=2000,
                         filter_env_amount=3000,
                         resonance=0.5,
                         amp_env=(0.01, 0.1, 0.7, 0.2),
                         filter_env=(0.01, 0.15, 0.3, 0.2)):
        """
        Create a complete synthesized note with filter and envelopes
        This is subtractive synthesis!
        """
        # Generate oscillator
        signal = self.oscillator(freq, duration, waveform)

        # Generate filter envelope
        filt_env = self.filter_envelope(len(signal), *filter_env)

        # Apply filter with envelope modulation
        # The filter sweeps from base to base+amount based on envelope
        filtered = np.zeros_like(signal)

        # Process in chunks for time-varying filter
        chunk_size = 64
        for i in range(0, len(signal), chunk_size):
            end = min(i + chunk_size, len(signal))
            chunk = signal[i:end]

            # Current filter cutoff (modulated by envelope)
            env_position = filt_env[i] if i < len(filt_env) else 0
            current_cutoff = filter_base + (filter_env_amount * env_position)
            current_cutoff = np.clip(current_cutoff, 20, 20000)

            # Apply filter to chunk
            filtered_chunk = self.moog_filter(chunk, current_cutoff, resonance)
            filtered[i:end] = filtered_chunk

        # Apply amplitude envelope
        amp_env = self.adsr_envelope(len(filtered), *amp_env)
        output = filtered * amp_env

        return output

class ChessSynthComposer:
    def __init__(self, chess_tags):
        self.tags = chess_tags
        self.sample_rate = 44100
        self.synth = SubtractiveSynth(self.sample_rate)

        self.total_duration = chess_tags.get('duration_seconds', 60)
        self.total_plies = chess_tags.get('total_plies', 40)
        self.overall_narrative = chess_tags.get('overall_narrative', 'COMPLEX_GAME')

        # Musical scales (frequencies in Hz)
        self.scales = {
            'minor': [110, 123.47, 130.81, 146.83, 164.81, 174.61, 196, 220],  # A minor
            'phrygian': [110, 116.54, 130.81, 146.83, 164.81, 174.61, 196, 220],  # Darker
            'dorian': [110, 123.47, 130.81, 146.83, 164.81, 185, 196, 220],  # Brighter minor
        }

        # LAYER 1: Overall narrative defines the BASE PATCH
        self.base_params = self.get_narrative_base_params()

        # Initialize narrative process
        self.narrative_process = self._create_process(
            self.overall_narrative,
            self.total_duration,
            self.total_plies
        )

    def get_narrative_base_params(self):
        """LAYER 1: Overall narrative sets the fundamental synth character"""

        if 'DEFEAT' in self.overall_narrative:
            return {
                'base_waveform': 'supersaw',  # Detuned saws that become chaotic
                'filter_start': 2500,    # Starts bright
                'filter_end': 300,       # Ends very dark
                'resonance_start': 0.8,
                'resonance_end': 3.5,    # Ends near self-oscillation
                'tempo_start': 1.0,
                'tempo_end': 0.7,        # Slows down (defeat energy loss)
                'detune_start': 3,       # Starts slightly detuned
                'detune_end': 20,        # Ends very dissonant
                'scale': 'phrygian',     # Dark mode
            }

        elif 'MASTERPIECE' in self.overall_narrative or 'VICTORY' in self.overall_narrative:
            return {
                'base_waveform': 'pulse',  # Hollow to start, will thicken
                'filter_start': 500,       # Starts closed
                'filter_end': 5000,        # Opens triumphantly
                'resonance_start': 0.5,
                'resonance_end': 1.8,      # Controlled power
                'tempo_start': 0.8,
                'tempo_end': 1.2,          # Accelerates to victory
                'detune_start': 0,
                'detune_end': 7,           # Adds richness, not chaos
                'scale': 'dorian',         # Brighter minor
            }

        elif 'DRAW' in self.overall_narrative or 'PEACEFUL' in self.overall_narrative:
            return {
                'base_waveform': 'triangle',  # Soft, pure
                'filter_start': 1500,
                'filter_end': 1500,           # Stays stable
                'resonance_start': 0.3,
                'resonance_end': 0.3,         # No change
                'tempo_start': 1.0,
                'tempo_end': 1.0,             # Consistent
                'detune_start': 0,
                'detune_end': 0,              # Perfect tuning
                'scale': 'dorian',            # Neutral mode
            }

        else:  # COMPLEX_GAME, UNKNOWN, etc
            return {
                'base_waveform': 'saw',
                'filter_start': 1500,
                'filter_end': 2000,        # Slight opening
                'resonance_start': 1.0,
                'resonance_end': 1.5,      # Mild increase
                'tempo_start': 1.0,
                'tempo_end': 1.0,
                'detune_start': 0,
                'detune_end': 5,
                'scale': 'minor',          # Standard minor
            }

    def interpolate_base_params(self, progress):
        """Get current base parameters based on progress through game"""
        base = self.base_params
        return {
            'waveform': base['base_waveform'],  # Doesn't change during piece
            'filter': base['filter_start'] + (base['filter_end'] - base['filter_start']) * progress,
            'resonance': base['resonance_start'] + (base['resonance_end'] - base['resonance_start']) * progress,
            'tempo': base['tempo_start'] + (base['tempo_end'] - base['tempo_start']) * progress,
            'detune': base['detune_start'] + (base['detune_end'] - base['detune_start']) * progress,
            'scale': self.scales[base['scale']],
        }

    def get_section_modulation(self, section_narrative, tension):
        """LAYER 2: Section narrative modulates the base parameters"""

        modulation = {
            'filter_mult': 1.0,
            'resonance_add': 0.0,
            'tempo_mult': 1.0,
            'note_density': 1.0,  # How many notes to play
            'filter_env_amount': 2500,  # Default envelope amount
        }

        # Apply tension as a general modifier
        tension_factor = tension

        if 'DESPERATE_DEFENSE' in section_narrative:
            modulation['filter_mult'] = 0.5      # Much darker
            modulation['resonance_add'] = 1.5 * tension_factor    # More desperate with tension
            modulation['tempo_mult'] = 0.85      # Slower, struggling
            modulation['note_density'] = 0.7     # Fewer notes, exhausted
            modulation['filter_env_amount'] = 800  # Minimal filter movement

        elif 'KING_HUNT' in section_narrative or 'MATING_ATTACK' in section_narrative:
            modulation['filter_mult'] = 1.3      # Brighter for intensity
            modulation['resonance_add'] = 1.0 * tension_factor    # More aggressive
            modulation['tempo_mult'] = 1.2       # Faster chase
            modulation['note_density'] = 1.5     # More notes, frantic
            modulation['filter_env_amount'] = 6000  # Dramatic sweeps

        elif 'TACTICAL_CHAOS' in section_narrative or 'TACTICAL_BATTLE' in section_narrative:
            modulation['filter_mult'] = 0.8      # Slightly darker
            modulation['resonance_add'] = 2.0 * tension_factor    # Very resonant
            modulation['tempo_mult'] = 1.3       # Fast exchanges
            modulation['note_density'] = 2.0     # Many rapid notes
            modulation['filter_env_amount'] = 5000  # Wild filter movement

        elif 'QUIET' in section_narrative or 'POSITIONAL' in section_narrative:
            modulation['filter_mult'] = 1.1      # Slightly brighter
            modulation['resonance_add'] = -0.3   # Less resonant, cleaner
            modulation['tempo_mult'] = 0.9       # Slower, thoughtful
            modulation['note_density'] = 0.5     # Sparse notes
            modulation['filter_env_amount'] = 1500  # Gentle movement

        elif 'SACRIFICIAL_ATTACK' in section_narrative or 'CRUSHING_ATTACK' in section_narrative:
            modulation['filter_mult'] = 0.7      # Start dark
            modulation['resonance_add'] = 1.5 * tension_factor
            modulation['tempo_mult'] = 1.1
            modulation['note_density'] = 1.3
            modulation['filter_env_amount'] = 8000  # Huge opening

        elif 'ENDGAME_PRECISION' in section_narrative:
            modulation['filter_mult'] = 1.0      # Clean
            modulation['resonance_add'] = 0.3
            modulation['tempo_mult'] = 0.8       # Deliberate
            modulation['note_density'] = 0.8
            modulation['filter_env_amount'] = 2000

        elif 'COMPLEX_STRUGGLE' in section_narrative:
            modulation['filter_mult'] = 0.9
            modulation['resonance_add'] = 0.5 * tension_factor
            modulation['tempo_mult'] = 1.0
            modulation['note_density'] = 1.0
            modulation['filter_env_amount'] = 3000

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

    def create_moment_voice(self, moment, current_params, progress):
        """LAYER 3: Key moments as additional synth voices - context-aware"""

        moment_type = moment.get('type', 'UNKNOWN')

        if moment_type in ['BLUNDER', 'MISTAKE']:
            # Different sound based on overall narrative context
            if 'DEFEAT' in self.overall_narrative:
                # In defeat context: deep, doomed sound (another nail in the coffin)
                return self.synth.create_synth_note(
                    freq=55,  # Very low
                    duration=1.0,
                    waveform='saw',
                    filter_base=200,  # Very dark
                    filter_env_amount=-150,  # Closing further
                    resonance=4.0,  # Self-oscillating
                    amp_env=(0.001, 0.01, 0.5, 0.4),
                    filter_env=(0.001, 0.5, 0.0, 0.4)
                )
            elif 'MASTERPIECE' in self.overall_narrative:
                # In masterpiece context: brief disturbance, quickly recovered
                return self.synth.create_synth_note(
                    freq=110,
                    duration=0.3,
                    waveform='pulse',
                    filter_base=3000,
                    filter_env_amount=-2500,
                    resonance=2.0,
                    amp_env=(0.001, 0.01, 0.3, 0.1)
                )
            else:
                # Neutral context: standard error sound
                return self.synth.create_synth_note(
                    freq=82.5,
                    duration=0.5,
                    waveform='saw',
                    filter_base=1000,
                    filter_env_amount=-800,
                    resonance=3.0,
                    amp_env=(0.001, 0.01, 0.5, 0.2)
                )

        elif moment_type in ['BRILLIANT', 'STRONG']:
            # Different based on context
            if 'MASTERPIECE' in self.overall_narrative:
                # In masterpiece: triumphant flourish building to climax
                freq = 220 * (1 + progress)  # Higher as game progresses
                return self.synth.create_synth_note(
                    freq=freq,
                    duration=0.5,
                    waveform='pulse',
                    filter_base=500,
                    filter_env_amount=4000 * (1 + progress),  # Bigger sweeps later
                    resonance=2.0,
                    amp_env=(0.001, 0.05, 0.7, 0.1),
                    filter_env=(0.001, 0.2, 0.5, 0.1)
                )
            elif 'DEFEAT' in self.overall_narrative:
                # In defeat context: brief hope, quickly extinguished
                return self.synth.create_synth_note(
                    freq=220,
                    duration=0.2,
                    waveform='triangle',
                    filter_base=2000,
                    filter_env_amount=500,  # Small opening
                    resonance=0.5,
                    amp_env=(0.001, 0.01, 0.3, 0.3)
                )
            else:
                # Neutral: standard brilliancy
                return self.synth.create_synth_note(
                    freq=330,
                    duration=0.3,
                    waveform='square',
                    filter_base=1500,
                    filter_env_amount=2000,
                    resonance=1.5,
                    amp_env=(0.001, 0.01, 0.8, 0.05)
                )

        elif moment_type == 'DEVELOPMENT':
            # Melodic phrase representing "awakening" or "coming online"
            # Rising phrase using current scale and rhythm

            # Get musical context
            base_filter = current_params.get('filter', 1000)
            scale = current_params.get('scale', [110, 123.47, 130.81, 146.83, 164.81])

            # Create rising melodic phrase: 1 -> 2 -> 3 -> 5 (awakening pattern)
            if 'DEFEAT' in self.overall_narrative:
                # In defeat context: hesitant, not reaching the top note
                melody_indices = [0, 1, 2, 1]  # Rise but fall back (foreshadowing)
                note_duration = 0.3
                waveform = 'triangle'  # Gentle
            else:
                # Normal context: confident rising
                melody_indices = [0, 1, 2, 4]  # 1 -> 2 -> 3 -> 5
                note_duration = 0.25
                waveform = 'pulse'

            melody_freqs = [scale[i] for i in melody_indices if i < len(scale)]

            # Generate melodic phrase
            phrase_samples = []
            for i, freq in enumerate(melody_freqs):
                note = self.synth.create_synth_note(
                    freq=freq,
                    duration=note_duration,
                    waveform=waveform,
                    filter_base=base_filter * 0.8,
                    filter_env_amount=800 + (i * 200),  # Rising brightness
                    resonance=0.8,
                    amp_env=(0.02, 0.05, 0.7, 0.1)
                )
                phrase_samples.append(note * 0.6)  # Integrate volume

            # Connect notes with tiny gaps
            gap_samples = int(0.02 * self.sample_rate)  # 20ms gaps
            combined = []
            for i, note_samples in enumerate(phrase_samples):
                combined.append(note_samples)
                if i < len(phrase_samples) - 1:  # No gap after last note
                    combined.append(np.zeros(gap_samples))

            return np.concatenate(combined)

        elif moment_type == 'FIRST_EXCHANGE':
            # Melodic question-answer phrase representing the piece trade
            # Get musical context
            base_filter = current_params.get('filter', 1000)
            scale = current_params.get('scale', [110, 123.47, 130.81, 146.83, 164.81])

            if 'DEFEAT' in self.overall_narrative:
                # Question: confident ascending phrase [1, 3, 5]
                # Answer: descending, weaker response [4, 2, 1] (showing future weakness)
                question_indices = [0, 2, 4]  # 1 -> 3 -> 5 (confident)
                answer_indices = [3, 1, 0]    # 4 -> 2 -> 1 (weakening)

                question_waveform = 'pulse'     # Assertive
                answer_waveform = 'triangle'    # Softer, weaker
                note_duration = 0.65

            else:
                # Balanced exchange
                question_indices = [0, 2, 4]   # 1 -> 3 -> 5
                answer_indices = [4, 2, 0]     # 5 -> 3 -> 1 (mirror)

                question_waveform = 'square'
                answer_waveform = 'pulse'
                note_duration = 0.22

            # Generate question phrase
            question_freqs = [scale[i] for i in question_indices if i < len(scale)]
            question_samples = []
            for i, freq in enumerate(question_freqs):
                note = self.synth.create_synth_note(
                    freq=freq,
                    duration=note_duration,
                    waveform=question_waveform,
                    filter_base=base_filter * 0.9,
                    filter_env_amount=600 + (i * 100),  # Slight brightness increase
                    resonance=1.0,
                    amp_env=(0.01, 0.04, 0.7, 0.08)
                )
                question_samples.append(note)

            # Generate answer phrase
            answer_freqs = [scale[i] for i in answer_indices if i < len(scale)]
            answer_samples = []
            for i, freq in enumerate(answer_freqs):
                brightness_mult = 0.7 if 'DEFEAT' in self.overall_narrative else 0.9
                note = self.synth.create_synth_note(
                    freq=freq,
                    duration=note_duration,
                    waveform=answer_waveform,
                    filter_base=base_filter * brightness_mult,
                    filter_env_amount=500 + (i * 50),  # Less dramatic than question
                    resonance=0.8 if 'DEFEAT' in self.overall_narrative else 1.0,
                    amp_env=(0.02, 0.06, 0.6, 0.1)  # Slightly softer
                )
                answer_samples.append(note)

            # Combine question and answer with musical timing
            gap_samples = int(0.02 * self.sample_rate)  # 20ms between notes
            pause_samples = int(0.15 * self.sample_rate)  # 150ms between phrases

            # Build complete phrase
            combined = []

            # Question phrase
            for i, note in enumerate(question_samples):
                combined.append(note * 0.7)  # Integrated volume
                if i < len(question_samples) - 1:
                    combined.append(np.zeros(gap_samples))

            # Pause between question and answer
            combined.append(np.zeros(pause_samples))

            # Answer phrase
            for i, note in enumerate(answer_samples):
                volume = 0.6 if 'DEFEAT' in self.overall_narrative else 0.7
                combined.append(note * volume)
                if i < len(answer_samples) - 1:
                    combined.append(np.zeros(gap_samples))

            return np.concatenate(combined)

        elif moment_type == 'TACTICAL_SEQUENCE':
            # Quick burst of activity - rapid arpeggiated notes
            freqs = [220, 275, 330, 275]  # Up and down pattern
            combined = np.zeros(int(1.2 * self.sample_rate))
            for i, freq in enumerate(freqs):
                note = self.synth.create_synth_note(
                    freq=freq,
                    duration=0.2,
                    waveform='square',
                    filter_base=1500,
                    filter_env_amount=1500,
                    resonance=1.5,
                    amp_env=(0.001, 0.02, 0.5, 0.05)
                )
                start_sample = int(i * 0.15 * self.sample_rate)  # Slight overlap
                if start_sample + len(note) < len(combined):
                    combined[start_sample:start_sample+len(note)] += note * 0.7
            return combined

        elif moment_type == 'MATE_SEQUENCE':
            # Mate sounds depend heavily on who's winning
            if 'DEFEAT' in self.overall_narrative:
                # Death knell
                return self.synth.create_synth_note(
                    freq=27.5,  # Extremely low
                    duration=2.0,
                    waveform='saw',
                    filter_base=100,
                    filter_env_amount=0,  # No movement, static doom
                    resonance=4.0,
                    amp_env=(0.5, 0.0, 1.0, 1.0)  # Sudden, sustained
                )
            elif 'MASTERPIECE' in self.overall_narrative:
                # Victory fanfare
                return self.synth.create_synth_note(
                    freq=440,
                    duration=1.0,
                    waveform='pulse',
                    filter_base=300,
                    filter_env_amount=5000,
                    resonance=2.5,
                    amp_env=(0.001, 0.1, 0.8, 0.5)
                )

        # Default: return gentle click (much improved from before)
        return self.synth.create_synth_note(
            freq=330,  # Higher, more pleasant
            duration=0.3,  # A bit longer
            waveform='triangle',  # Softer
            filter_base=1000,
            filter_env_amount=500,
            resonance=0.5,
            amp_env=(0.01, 0.05, 0.4, 0.1)
        )

    def _create_process(self, narrative: str, duration: float, plies: int) -> NarrativeProcess:
        """Create appropriate process based on overall narrative"""
        process_map = {
            'TUMBLING_DEFEAT': TumblingDefeatProcess,
            'ATTACKING_MASTERPIECE': AttackingMasterpieceProcess,
            'QUIET_PRECISION': QuietPrecisionProcess,
            # Add more mappings as we implement more processes
            'FIGHTING_DEFEAT': TumblingDefeatProcess,  # Similar behavior
            'TACTICAL_MASTERPIECE': AttackingMasterpieceProcess,  # Similar behavior
            'PEACEFUL_DRAW': QuietPrecisionProcess,  # Similar behavior
        }

        ProcessClass = process_map.get(narrative, DefaultProcess)
        return ProcessClass(duration, plies)

    def compose_section(self, section, section_index, total_sections):
        """Compose a section using all three narrative layers

        Layer 1: Continuous drone/pad from overall narrative
        Layer 2: Rhythmic/melodic patterns from section narrative
        Layer 3: Punctuation from key moments
        """

        # Parse section duration
        section_duration = self.parse_section_duration(section)

        narrative = section.get('narrative', 'UNKNOWN')
        tension = section.get('tension', 0.5)

        # Calculate progress through the piece (0.0 to 1.0)
        progress = section_index / max(1, total_sections - 1)

        # LAYER 1: Get base parameters from overall narrative at current progress
        current_base = self.interpolate_base_params(progress)

        # LAYER 2: Apply section narrative modulation
        modulation = self.get_section_modulation(narrative, tension)

        # Calculate final synthesis parameters
        final_filter = current_base['filter'] * modulation['filter_mult']
        final_filter = np.clip(final_filter, 20, self.synth.nyquist * 0.95)

        final_resonance = current_base['resonance'] + modulation['resonance_add']
        final_resonance = np.clip(final_resonance, 0.1, 4.0)

        filter_env_amount = modulation['filter_env_amount'] * (1 + current_base['detune'] / 20)  # Detune affects filter movement

        print(f"\n  === COMPOSING SECTION: {section['name']} ===")
        print(f"    Progress through game: {progress:.1%}")
        print(f"    Overall narrative: {self.overall_narrative}")
        print(f"    Section narrative: {narrative} (tension: {tension:.2f})")
        print(f"    Duration: {section_duration}s")
        print(f"    \n    LAYER 1 - Base params from '{self.overall_narrative}':")
        print(f"      Waveform: {current_base['waveform']}")
        print(f"      Filter: {current_base['filter']:.0f}Hz")
        print(f"      Resonance: {current_base['resonance']:.2f}")
        print(f"      Tempo factor: {current_base['tempo']:.2f}")
        print(f"      Detune: {current_base['detune']:.1f} cents")
        print(f"    \n    LAYER 2 - Section modulation from '{narrative}':")
        print(f"      Filter multiplier: {modulation['filter_mult']:.2f}")
        print(f"      Resonance addition: {modulation['resonance_add']:.2f}")
        print(f"      Tempo multiplier: {modulation['tempo_mult']:.2f}")
        print(f"      Note density: {modulation['note_density']:.2f}")
        print(f"    \n    Final synthesis parameters:")
        print(f"      Filter: {final_filter:.0f}Hz")
        print(f"      Resonance: {final_resonance:.2f}")
        print(f"      Filter envelope: {filter_env_amount:.0f}Hz")

        # Use the base waveform from overall narrative
        waveform = current_base['waveform']
        scale = current_base['scale']

        # Calculate note duration based on all layers
        base_note_duration = 0.5  # Base duration
        note_duration = base_note_duration * current_base['tempo'] * modulation['tempo_mult']

        print(f"      Note duration: {note_duration:.3f}s")
        print(f"      Scale: {[f'{f:.1f}Hz' for f in scale[:4]]}...")

        # Apply narrative process transformations (existing process system)
        current_time = section.get('start_ply', 0)  # Use ply as time

        # Check for key moments in this section for process update
        process_key_moment = None
        for moment in section.get('key_moments', []):
            # Use the first moment as representative for this section update
            if process_key_moment is None:
                process_key_moment = moment
                break

        # Get process transformations
        transforms = self.narrative_process.update(current_time, process_key_moment)

        if transforms:  # Only apply if process returns transformations
            print(f"      Process transformations: {transforms}")

            # Apply transformations to synthesis parameters
            if 'tempo_multiplier' in transforms:
                note_duration *= transforms['tempo_multiplier']

            if 'filter_brightness' in transforms:
                filter_envelope_amount *= transforms['filter_brightness']

            if 'resonance_boost' in transforms:
                resonance += transforms['resonance_boost']
                resonance = max(0.1, min(4.0, resonance))  # Clamp resonance

            if 'volume_decay' in transforms:
                volume_multiplier = transforms['volume_decay']
            else:
                volume_multiplier = 1.0

            if 'volume_crescendo' in transforms:
                volume_multiplier = transforms['volume_crescendo']
        else:
            volume_multiplier = 1.0  # Default if no process or no volume transform

        # Create a pattern based on the scale
        pattern = []

        # Generate pattern based on tension AND overall narrative
        if 'DEFEAT' in self.overall_narrative:
            # Descending patterns for defeat
            if tension > 0.7:
                indices = [7, 5, 6, 4, 5, 3, 4, 2]  # Falling with struggle
            else:
                indices = [4, 3, 3, 2, 2, 1, 1, 0]  # Gradual descent
        elif 'MASTERPIECE' in self.overall_narrative:
            # Ascending patterns for victory
            if tension > 0.7:
                indices = [0, 2, 1, 3, 2, 5, 4, 7]  # Rising with energy
            else:
                indices = [0, 1, 2, 3, 3, 4, 5, 6]  # Steady climb
        else:
            # Neutral patterns
            if tension > 0.7:
                indices = [0, 4, 2, 5, 1, 4, 3, 7]  # Jumping
            elif tension > 0.4:
                indices = [0, 2, 3, 2, 4, 3, 2, 1]  # Melodic
            else:
                indices = [0, 1, 2, 1, 3, 2, 1, 0]  # Stepwise

        print(f"      Pattern indices: {indices} (based on {self.overall_narrative} + tension {tension:.2f})")

        for idx in indices:
            if idx < len(scale):
                pattern.append(scale[idx])

        print(f"      Generated pattern frequencies: {[f'{f:.1f}Hz' for f in pattern]}")

        # Calculate how many times to play the pattern (affected by note density)
        num_notes = int(section_duration / note_duration * modulation['note_density'])
        print(f"      Will play {num_notes} notes over {section_duration}s")

        # Generate CONTINUOUS audio with the chosen waveform
        total_samples = int(section_duration * self.sample_rate)

        # Create smooth frequency modulation over time
        freq_modulation = np.ones(total_samples)
        samples_per_note = int(note_duration * self.sample_rate)

        for i in range(num_notes):
            start_sample = i * samples_per_note
            end_sample = min(start_sample + samples_per_note, total_samples)

            if start_sample < total_samples:
                note_freq = pattern[i % len(pattern)]

                # Vary the frequency slightly for movement
                if i % 4 == 0:
                    note_freq *= 2  # Octave up occasionally
                elif i % 7 == 0:
                    note_freq *= 0.5  # Octave down

                # Smooth frequency transition (no clicks!)
                freq_ratio = note_freq / 110  # Ratio from base frequency

                if i > 0:  # Smooth transition from previous note
                    prev_ratio = freq_modulation[start_sample-1] if start_sample > 0 else 1.0
                    transition_samples = min(1000, end_sample - start_sample)  # 23ms transition

                    for j in range(transition_samples):
                        progress = j / transition_samples
                        smooth_ratio = prev_ratio + (freq_ratio - prev_ratio) * progress
                        if start_sample + j < total_samples:
                            freq_modulation[start_sample + j] = smooth_ratio

                    # Fill rest with target frequency
                    if start_sample + transition_samples < end_sample:
                        freq_modulation[start_sample + transition_samples:end_sample] = freq_ratio
                else:
                    freq_modulation[start_sample:end_sample] = freq_ratio

        # LAYER 1: Generate CONTINUOUS BASE DRONE/PAD
        print(f"\n    === LAYER 1: BASE DRONE ===")

        # Base frequency for the drone - low and stable
        drone_freq = scale[0] / 2  # Root note, one octave down
        print(f"      Drone frequency: {drone_freq:.1f}Hz")

        if waveform == 'supersaw':
            # Generate continuous supersaw drone for entire section
            detune_spread = current_base['detune']
            detune_cents = [-detune_spread*2, -detune_spread, -detune_spread/2,
                          detune_spread/2, detune_spread, detune_spread*2]

            # Create the drone with slow filter evolution
            base_drone = self.synth.supersaw(
                freq=drone_freq,
                duration=section_duration,
                detune_cents=detune_cents,
                filter_base=final_filter,
                filter_env_amount=filter_env_amount * 0.3,  # Gentle filter movement
                resonance=final_resonance,
                amp_env=(0.5, 0.0, 1.0, 0.5),  # Slow attack, full sustain
                filter_env=(0.5, 0.0, 1.0, 0.5)  # Minimal filter envelope
            )
            print(f"      Supersaw drone with {detune_spread:.1f} cent detune spread")
        else:
            # Generate simple oscillator drone for other waveforms
            base_drone = self.synth.create_synth_note(
                freq=drone_freq,
                duration=section_duration,
                waveform=waveform,
                filter_base=final_filter,
                filter_env_amount=filter_env_amount * 0.3,
                resonance=final_resonance,
                amp_env=(0.5, 0.0, 1.0, 0.5),
                filter_env=(0.5, 0.0, 1.0, 0.5)
            )
            print(f"      {waveform.capitalize()} drone")

        # Apply slow LFO to drone for movement
        lfo_freq = 0.1  # Very slow LFO
        #lfo = np.sin(2 * np.pi * lfo_freq * np.arange(len(base_drone)) / self.sample_rate)
        lfo = signal.sawtooth(2 * np.pi * lfo_freq * np.arange(len(base_drone)) / self.sample_rate, width=0.5)
        base_drone = base_drone * (1 + lfo * 0.1)  # Subtle amplitude modulation

        # LAYER 2: Generate RHYTHMIC/MELODIC PATTERNS from section narrative
        print(f"\n    === LAYER 2: SECTION PATTERNS ===")

        if waveform == 'supersaw':
            # Generate rhythmic patterns with shorter, punchier supersaw notes
            section_pattern = np.zeros(total_samples)
            samples_per_note = int(note_duration * self.sample_rate)

            # Calculate detune amounts based on progress (starts tight, becomes chaotic)
            detune_spread = current_base['detune']  # This evolves from 3 to 20 cents in DEFEAT
            detune_cents = [-detune_spread, -detune_spread/2, -detune_spread/4, detune_spread/4, detune_spread/2, detune_spread]

            for i in range(num_notes):
                start_sample = i * samples_per_note
                end_sample = min(start_sample + samples_per_note, total_samples)

                if start_sample < total_samples:
                    note_freq = pattern[i % len(pattern)]

                    # Vary the frequency for movement
                    if i % 4 == 0:
                        note_freq *= 2  # Octave up
                    elif i % 7 == 0:
                        note_freq *= 0.5  # Octave down

                    # Generate supersaw for this note
                    note_samples = end_sample - start_sample
                    note_duration_sec = note_samples / self.sample_rate

                    # Rhythmic/melodic notes - shorter and punchier than drone
                    pattern_note = self.synth.create_synth_note(
                        freq=note_freq,
                        duration=note_duration_sec,
                        waveform='saw' if tension > 0.5 else 'pulse',  # Use simpler waveforms for clarity
                        filter_base=final_filter * 1.5,  # Brighter than drone
                        filter_env_amount=filter_env_amount,
                        resonance=final_resonance * 0.7,  # Less resonance for clarity
                        amp_env=(0.001, 0.05, 0.3, 0.1),  # Short, percussive envelope
                        filter_env=(0.001, 0.1, 0.2, 0.1)
                    )

                    # Place the note in the pattern layer
                    if len(pattern_note) > 0:
                        actual_samples = min(len(pattern_note), end_sample - start_sample)
                        section_pattern[start_sample:start_sample + actual_samples] += pattern_note[:actual_samples] * 0.5
        else:
            # Generate rhythmic patterns for other waveforms
            section_pattern = np.zeros(total_samples)
            samples_per_note = int(note_duration * self.sample_rate)

            for i in range(num_notes):
                start_sample = i * samples_per_note
                end_sample = min(start_sample + samples_per_note, total_samples)

                if start_sample < total_samples:
                    note_freq = pattern[i % len(pattern)]

                    # Vary frequency for movement
                    if i % 4 == 0:
                        note_freq *= 2
                    elif i % 7 == 0:
                        note_freq *= 0.5

                    # Generate pattern note
                    note_samples = end_sample - start_sample
                    note_duration_sec = note_samples / self.sample_rate

                    pattern_note = self.synth.create_synth_note(
                        freq=note_freq,
                        duration=note_duration_sec,
                        waveform='pulse' if tension > 0.5 else 'triangle',
                        filter_base=final_filter * 1.5,
                        filter_env_amount=filter_env_amount,
                        resonance=final_resonance * 0.7,
                        amp_env=(0.001, 0.05, 0.3, 0.1),
                        filter_env=(0.001, 0.1, 0.2, 0.1)
                    )

                    if len(pattern_note) > 0:
                        actual_samples = min(len(pattern_note), end_sample - start_sample)
                        section_pattern[start_sample:start_sample + actual_samples] += pattern_note[:actual_samples] * 0.5

        print(f"      Generated {num_notes} pattern notes")

        # MIX LAYER 1 AND LAYER 2
        # Base drone is continuous, patterns are rhythmic on top
        mixed_signal = base_drone * 0.6 + section_pattern * 0.4
        print(f"\n    Mixed drone (60%) + patterns (40%)")

        # Apply smooth amplitude envelope over entire section
        section_envelope = np.ones(total_samples)
        fade_samples = int(0.1 * self.sample_rate)  # 100ms fades

        # Fade in
        section_envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        # Fade out
        section_envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        samples = mixed_signal * section_envelope * 0.3 * volume_multiplier  # Volume level with process modulation

        # LAYER 3: CONTINUOUS SEQUENCER - RUNS ENTIRE SECTION
        print(f"\n    === LAYER 3: CONTINUOUS SEQUENCER ===")

        # Timing setup
        bpm = section.get('bpm', 120)
        beat_duration = 60.0 / bpm
        sixteenth_duration = beat_duration / 4  # 16th notes for sequencer

        # Initialize continuous sequencer output
        sequencer_layer = np.zeros_like(samples)

        # MIDI to frequency
        def midi_to_freq(midi_note):
            return 440.0 * 2**((midi_note - 69) / 12.0)

        # LAYER 3: DYNAMIC PATTERN EVOLUTION SYSTEM
        # Pattern BECOMES what it represents

        # Start with minimal seed pattern - will grow/change based on moments
        current_pattern = [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7]  # Simple pulse
        pattern_length = 16

        # Pattern evolution tracking
        pattern_state = {
            'type': 'PULSE',  # Current pattern type
            'intensity': 0.5,  # How intense/complex
            'rhythm_division': 16,  # 16 = sixteenth notes, 8 = eighth notes, etc
        }

        # MEANINGFUL PATTERN LIBRARY - Patterns that SOUND like what they are
        patterns = {
            # PATTERNS THAT ACTUALLY SOUND LIKE WHAT THEY REPRESENT

            # DEVELOPMENT - Building/Growing (starts small, adds notes)
            'DEVELOPMENT': {
                'early': [0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0],  # Sparse beginning
                'mid': [0, 3, 7, 0, 3, 7, 12, 0, 3, 7, 12, 15, 0, 7, 12, 15],  # Adding complexity
                'full': [0, 3, 5, 7, 10, 12, 15, 17, 19, 17, 15, 12, 10, 7, 5, 3],  # Full development
            },

            # TACTICAL_SEQUENCE - Rapid heartbeat/morse code calculation
            'TACTICAL_SEQUENCE': [0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12],  # Like rapid thinking

            # BLUNDER - Falling down stairs (large drops, irregular)
            'BLUNDER': [24, 24, 12, 12, 0, 0, -12, -12, -24, None, None, -36, None, None, None, -48],  # Stumbling (None = rest)

            # MISTAKE - Sliding downward
            'MISTAKE': [12, 10, 8, 7, 5, 3, 2, 0, -2, -3, -5, -7, -8, -10, -12, -15],  # Gradual descent

            # INACCURACY - Hesitation/uncertainty
            'INACCURACY': [0, None, 3, 2, None, 7, 5, None, 3, None, 0, None, -2, None, 0, None],  # Hesitant (None = rest)

            # FIRST_EXCHANGE - Call and immediate response
            'FIRST_EXCHANGE': [0, 12, 19, 24, -24, -19, -12, 0, 0, 12, 19, 24, -24, -19, -12, 0],  # Up then down

            # MATE_SEQUENCE - Funeral march doom (SLOW and HEAVY)
            'MATE_SEQUENCE': [0, None, None, None, -12, None, None, None, 0, None, None, None, -24, None, None, None],  # Doom bells

            # BRILLIANT - Ascending triumph
            'BRILLIANT': [0, 7, 12, 19, 24, 31, 36, 43, 48, 43, 36, 31, 24, 19, 12, 7],  # Rising victory

            # TIME_PRESSURE - Panic repetition
            'TIME_PRESSURE': [0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12],  # Frantic alternation
        }

        # Start with simple pulse
        current_root = 60  # Middle C
        filter_frequency = 1000
        filter_target = 3000
        development_count = 0  # Track development progression

        # Process key moments to determine evolution points
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

        print(f"      Running continuous sequencer for {section_duration:.1f}s")
        evolution_str = ", ".join([f"{e['type']} at {e['time']:.1f}s" for e in evolution_points])
        print(f"      Evolution points: {evolution_str}")

        # Generate continuous sequence for entire section
        samples_per_step = int(sixteenth_duration * self.sample_rate)
        total_steps = int(section_duration / sixteenth_duration)

        # Pre-generate the entire sequence pattern
        full_sequence = []
        for i in range(total_steps):
            step_index = i % 16
            note_interval = current_pattern[step_index]

            # Check if we hit an evolution point
            current_time = i * sixteenth_duration
            current_sample = int(current_time * self.sample_rate)

            if next_evolution_idx < len(evolution_points):
                if current_sample >= evolution_points[next_evolution_idx]['sample_pos']:
                    evolution = evolution_points[next_evolution_idx]

                    # ACTUALLY CHANGE THE PATTERN TO A NEW ONE
                    moment_type = evolution['type']
                    print(f"        {moment_type}: CHANGING to {moment_type} pattern")

                    # REPLACE THE ENTIRE PATTERN
                    if moment_type == 'DEVELOPMENT':
                        development_count += 1
                        # Progressive development patterns
                        if development_count == 1:
                            current_pattern = patterns['DEVELOPMENT']['early']
                        elif development_count == 2:
                            current_pattern = patterns['DEVELOPMENT']['mid']
                        else:
                            current_pattern = patterns['DEVELOPMENT']['full']
                        current_root = min(current_root + 7, 72)
                        filter_target = min(filter_target + 500, 5000)

                    elif moment_type == 'FIRST_EXCHANGE':
                        current_pattern = patterns['FIRST_EXCHANGE']
                        filter_target = 4000

                    elif moment_type == 'BLUNDER':
                        current_pattern = patterns['BLUNDER']
                        current_root = max(current_root - 12, 36)  # Drop an octave
                        filter_target = 1000  # Dark

                    elif moment_type == 'MISTAKE':
                        current_pattern = patterns['MISTAKE']
                        current_root = max(current_root - 5, 48)
                        filter_target = 1500

                    elif moment_type == 'INACCURACY':
                        current_pattern = patterns['INACCURACY']
                        filter_target = 2500

                    elif moment_type in ['BRILLIANT', 'STRONG']:
                        current_pattern = patterns['BRILLIANT']
                        current_root = min(current_root + 12, 84)
                        filter_target = 5000

                    elif moment_type in ['TACTICAL_SEQUENCE', 'KING_ATTACK']:
                        current_pattern = patterns['TACTICAL_SEQUENCE']
                        filter_target = 3500

                    elif moment_type == 'MATE_SEQUENCE':
                        current_pattern = patterns['MATE_SEQUENCE']
                        pattern_state['rhythm_division'] = 8  # SLOW DOWN for doom
                        filter_target = 1500

                    elif moment_type in ['TIME_PRESSURE', 'TIME_SCRAMBLE']:
                        current_pattern = patterns['TIME_PRESSURE']
                        filter_target = 4500

                    next_evolution_idx += 1

            # Use the actual note from the pattern
            # Handle None values (rests) in patterns
            if note_interval is None:
                # Rest - no note
                midi_note = None
            else:
                midi_note = current_root + note_interval

            full_sequence.append(midi_note)

            # Smooth filter movement
            filter_frequency += (filter_target - filter_frequency) * 0.02

        # Now generate the actual audio using SubtractiveSynth
        print(f"      Generating {len(full_sequence)} steps of continuous sequence")

        for i, midi_note in enumerate(full_sequence):
            if i * samples_per_step >= len(sequencer_layer):
                break

            # Skip rests (None values)
            if midi_note is None:
                continue

            # Check if we need different rhythm for certain patterns
            note_duration = sixteenth_duration
            if pattern_state.get('rhythm_division') == 8:
                # Slower rhythm for doom patterns like MATE_SEQUENCE
                note_duration = sixteenth_duration * 2  # Eighth notes

            # Generate note using the SubtractiveSynth
            freq = midi_to_freq(midi_note)

            # LASER HARP synthesis - wide supersaw with rising filter sweep
            note_audio = self.synth.supersaw(
                freq,
                note_duration,
                detune_cents=[-15, -9, -4.5, 4.5, 9, 15],  # WIDE supersaw for laser effect
                filter_base=800 + (i * 150),  # RISING filter sweep as sequence progresses
                filter_env_amount=2000,  # Strong filter modulation
                resonance=1.2,  # Moderate resonance for laser character
                amp_env=(0.02, 0.1, 0.9, 0.8),  # Laser harp envelope
                filter_env=(0.02, 0.2, 0.7, 0.3)  # Filter sweep envelope
            )


            # Place note in sequence
            start_pos = int(i * samples_per_step * 0.98)  # 2% overlap
            end_pos = min(start_pos + len(note_audio), len(sequencer_layer))

            if end_pos > start_pos:
                sequencer_layer[start_pos:end_pos] += note_audio[:end_pos-start_pos] * 0.4

        # Apply a global filter sweep to the entire sequence
        # This creates the signature Jarre filter movement
        sweep_length = len(sequencer_layer)
        filter_sweep = np.zeros(sweep_length)

        # Create smooth filter envelope over entire section
        for i in range(sweep_length):
            progress = i / sweep_length
            # Sine wave LFO for filter
            lfo = np.sin(2 * np.pi * 0.25 * progress)  # 0.25 Hz LFO
            filter_sweep[i] = 2000 + 1500 * lfo + 1000 * progress  # Rising with LFO

        # Apply the filter sweep using moog filter
        filtered_sequence = np.zeros_like(sequencer_layer)
        chunk_size = 512
        for i in range(0, len(sequencer_layer), chunk_size):
            chunk_end = min(i + chunk_size, len(sequencer_layer))
            chunk = sequencer_layer[i:chunk_end]

            # Average filter frequency for this chunk
            avg_cutoff = np.mean(filter_sweep[i:chunk_end])

            if len(chunk) > 0:
                filtered_chunk = self.synth.moog_filter(chunk, cutoff_hz=avg_cutoff, resonance=2.0)
                filtered_sequence[i:chunk_end] = filtered_chunk

        # Mix the continuous sequencer with existing layers
        # Apply sidechain compression for clarity

        # Duck the main mix when sequencer is playing
        sequencer_envelope = np.abs(filtered_sequence)
        smoothing = int(0.005 * self.sample_rate)
        if smoothing > 0:
            sequencer_envelope = np.convolve(sequencer_envelope, np.ones(smoothing) / smoothing, mode='same')

        max_env = np.max(sequencer_envelope)
        if max_env > 0:
            sequencer_envelope = sequencer_envelope / max_env

        # Duck existing layers when sequencer plays
        ducking = 1.0 - (sequencer_envelope * 0.3)  # Duck by 30%
        samples = samples * ducking

        # Add the filtered sequencer layer
        samples = samples + filtered_sequence * 0.5

        # Soft clipping to prevent distortion
        samples = np.tanh(samples * 0.9) * 0.95

        return np.array(samples)

    def compose(self):
        """Create the full composition"""
        print("\n♫ SUBTRACTIVE SYNTHESIS CHESS MUSIC - THREE LAYER COMPOSITION")
        print(f"Result: {self.tags.get('game_result', '?')}")
        print(f"Overall Narrative: {self.overall_narrative}")
        print(f"Base Synth Patch: {self.base_params['base_waveform']} wave")
        print(f"Filter Evolution: {self.base_params['filter_start']}Hz → {self.base_params['filter_end']}Hz")
        print(f"Resonance Evolution: {self.base_params['resonance_start']:.1f} → {self.base_params['resonance_end']:.1f}")

        composition = []
        sections = self.tags.get('sections', [])
        total_sections = len(sections)

        print(f"\nSynthesizing {total_sections} sections with three-layer narrative system:")
        for i, section in enumerate(sections):
            section_music = self.compose_section(section, i, total_sections)
            composition.extend(section_music)

            # Brief silence between sections
            composition.extend(np.zeros(int(self.sample_rate * 0.1)))

        # Final normalization
        composition = np.array(composition)
        max_val = np.max(np.abs(composition))
        if max_val > 0.9:
            composition = composition * (0.9 / max_val)

        print(f"\n✓ Synthesis complete: {len(composition)/self.sample_rate:.1f} seconds")
        return composition

    def save(self, filename='chess_synth.wav'):
        """Save the composition"""
        composition = self.compose()

        with wave.open(filename, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)

            for sample in composition:
                int_sample = int(sample * 30000)
                int_sample = max(-32000, min(32000, int_sample))
                wav.writeframes(struct.pack('<h', int_sample))

        print(f"Saved: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python synth_composer.py tags.json")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        tags = json.load(f)

    composer = ChessSynthComposer(tags)
    composer.save()

if __name__ == '__main__':
    main()
