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
import numpy as np
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
        Revised Moog-style 4-pole (24dB/octave) low-pass filter
        with non-linear saturation for a more authentic sound.
        Based on the Stilson/Smith model and further improvements.
        """
        if cutoff_hz >= self.nyquist * 0.99:
            return signal  # Bypass if cutoff too high

        # Stilson/Smith approximation
        f = cutoff_hz / self.nyquist

        # This is a common cutoff approximation, but others exist
        # For stability, constrain f
        f = np.clip(f, 0.001, 0.45)

        # Scale resonance (0-4), clamping for stability
        resonance = np.clip(resonance, 0.0, 3.95)

        # Use persistent state variables for filter memory
        z1, z2, z3, z4 = self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4

        filtered = np.zeros_like(signal)

        for i in range(len(signal)):
            input_sample = signal[i]

            # Feedback loop with non-linear saturation
            # The feedback is applied to the input
            input_with_feedback = input_sample - resonance * z4 * 0.1  # Scale down feedback

            # Simple 4-pole low-pass filter without excessive tanh
            z1 = z1 + f * (input_with_feedback - z1)
            z2 = z2 + f * (z1 - z2)
            z3 = z3 + f * (z2 - z3)
            z4 = z4 + f * (z3 - z4)

            filtered[i] = z4

        # Save state for next filter call
        self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4 = z1, z2, z3, z4

        # Compensate for filter attenuation
        output_gain = 12.0 + resonance * 2.0  # Proper listening level
        return filtered * output_gain

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

        # Musical scales (frequencies in Hz)
        self.scales = {
            'minor': [110, 123.47, 130.81, 146.83, 164.81, 174.61, 196, 220],  # A minor
            'phrygian': [110, 116.54, 130.81, 146.83, 164.81, 174.61, 196, 220],  # Darker
            'dorian': [110, 123.47, 130.81, 146.83, 164.81, 185, 196, 220],  # Brighter minor
        }

        # Initialize narrative process
        self.narrative_process = self._create_process(
            chess_tags.get('overall_narrative', 'COMPLEX_GAME'),
            self.total_duration,
            self.total_plies
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

    def compose_section(self, section):
        """Compose a section using the synthesizer"""
        samples = []

        # Parse section duration
        duration_str = section.get('duration', '0:10')
        if ':' in duration_str:
            parts = duration_str.split(':')
            try:
                start_time = int(parts[0])
                end_time = int(parts[1])
                section_duration = end_time - start_time
            except:
                section_duration = 10
        else:
            section_duration = 10

        narrative = section.get('narrative', 'UNKNOWN')
        tension = section.get('tension', 0.5)

        print(f"    Composing {section['name']}: {narrative} (tension: {tension:.2f}, duration: {section_duration}s)")

        # Choose synthesis parameters based on narrative
        if 'TACTICAL_CHAOS' in narrative:
            # Chaotic tactical exchanges - aggressive saw with rapid filter sweeps
            waveform = 'saw'
            filter_base = 300
            filter_envelope_amount = 5000 * tension
            resonance = 3.0  # Very resonant for intensity
            scale = self.scales['phrygian']
            note_duration = 0.2  # Very fast notes
            print(f"      TACTICAL_CHAOS: saw wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'TACTICAL_BATTLE' in narrative:
            # Tactical battle with advantage gained - pulse with strong filter
            waveform = 'pulse'
            filter_base = 350
            filter_envelope_amount = 4500 * tension
            resonance = 2.5  # Strong resonance
            scale = self.scales['minor']
            note_duration = 0.25  # Fast but controlled
            print(f"      TACTICAL_BATTLE: pulse wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'KING_HUNT' in narrative or 'KING_ATTACK' in narrative:
            # Pursuing, relentless sound - pulse wave building intensity
            waveform = 'pulse'
            filter_base = 400
            filter_envelope_amount = 6000 * tension
            resonance = 2.0
            scale = self.scales['minor']
            note_duration = 0.3  # Moderate speed but building
            print(f"      KING_HUNT/ATTACK: pulse wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'SACRIFICIAL_ATTACK' in narrative or 'CRUSHING_ATTACK' in narrative:
            # Explosive, dramatic sound - saw with massive filter opening
            waveform = 'saw'
            filter_base = 100  # Start very closed
            filter_envelope_amount = 8000 * tension  # Huge opening
            resonance = 2.5
            scale = self.scales['minor']
            note_duration = 0.4
            print(f"      SACRIFICIAL/CRUSHING_ATTACK: saw wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'DESPERATE_DEFENSE' in narrative or 'DEFENSIVE_STAND' in narrative:
            # Dark, closed, tense sound - saw with minimal filter opening
            waveform = 'saw'
            filter_base = 120  # Very closed
            filter_envelope_amount = 800  # Minimal opening
            resonance = 3.5  # Near self-oscillation for tension
            scale = self.scales['phrygian']
            note_duration = 0.6  # Slower, more deliberate
            print(f"      DESPERATE_DEFENSE: saw wave, filter {filter_base}Hz + {filter_envelope_amount}Hz sweep, resonance {resonance} (dark/tense)")

        elif 'COMPLEX_STRUGGLE' in narrative:
            # Evolving, complex sound - triangle with moderate filter movement
            waveform = 'triangle'
            filter_base = 600
            filter_envelope_amount = 3000 * tension
            resonance = 1.0
            scale = self.scales['dorian']
            note_duration = 0.5
            print(f"      COMPLEX_STRUGGLE: triangle wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'POSITIONAL_SQUEEZE' in narrative or 'POSITIONAL_MANEUVERING' in narrative or 'QUIET_MANEUVERING' in narrative:
            # Subtle, evolving sound - triangle with gentle filter movement
            waveform = 'triangle'
            filter_base = 1000
            filter_envelope_amount = 1500 * tension
            resonance = 0.3  # Very gentle
            scale = self.scales['dorian']
            note_duration = 0.8  # Slow, contemplative
            print(f"      POSITIONAL: triangle wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance} (gentle)")

        elif 'ENDGAME_PRECISION' in narrative:
            # Clean, precise sound - square wave with controlled filter
            waveform = 'square'
            filter_base = 800
            filter_envelope_amount = 2000 * tension
            resonance = 0.8
            scale = self.scales['minor']
            note_duration = 0.6
            print(f"      ENDGAME_PRECISION: square wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'OPENING_THEORY' in narrative or 'DEVELOPMENT' in narrative:
            # Bright, structured sound - triangle with rising filter
            waveform = 'triangle'
            filter_base = 1200
            filter_envelope_amount = 2500 * tension
            resonance = 0.5
            scale = self.scales['dorian']
            note_duration = 0.7
            print(f"      OPENING_THEORY: triangle wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'MATING_ATTACK' in narrative:
            # Inevitable, closing sound - saw with dramatic filter sweeps
            waveform = 'saw'
            filter_base = 150
            filter_envelope_amount = 7000 * tension
            resonance = 3.2
            scale = self.scales['phrygian']
            note_duration = 0.25
            print(f"      MATING_ATTACK: saw wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'TENSE_EQUILIBRIUM' in narrative:
            # Balanced but tense - pulse with moderate filter
            waveform = 'pulse'
            filter_base = 500
            filter_envelope_amount = 2000 * tension
            resonance = 1.5
            scale = self.scales['minor']
            note_duration = 0.6
            print(f"      TENSE_EQUILIBRIUM: pulse wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'LIQUIDATION' in narrative:
            # Simplifying, clearing out - triangle with falling filter
            waveform = 'triangle'
            filter_base = 2000
            filter_envelope_amount = -1000  # Negative for closing
            resonance = 0.4
            scale = self.scales['dorian']
            note_duration = 0.9
            print(f"      LIQUIDATION: triangle wave, filter {filter_base}Hz + {filter_envelope_amount}Hz sweep, resonance {resonance}")

        elif 'CRITICAL_DECISIONS' in narrative:
            # Important moments - square with sharp filter
            waveform = 'square'
            filter_base = 600
            filter_envelope_amount = 3500 * tension
            resonance = 2.2
            scale = self.scales['minor']
            note_duration = 0.5
            print(f"      CRITICAL_DECISIONS: square wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'COMPLEX_POSITION' in narrative:
            # Unknown evaluation - mysterious, ambient sound
            waveform = 'triangle'
            filter_base = 800
            filter_envelope_amount = 1800 * tension
            resonance = 0.8
            scale = self.scales['dorian']
            note_duration = 0.7
            print(f"      COMPLEX_POSITION: triangle wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        else:  # Default for any unmatched narratives
            # Balanced sound - saw with moderate characteristics
            waveform = 'saw'
            filter_base = 500
            filter_envelope_amount = 2500 * tension
            resonance = 1.2
            scale = self.scales['minor']
            note_duration = 0.5
            print(f"      DEFAULT ({narrative}): saw wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        print(f"      Scale: {scale}")
        print(f"      Note duration: {note_duration}s")

        # Apply narrative process transformations
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
        pattern_length = 8
        pattern = []

        # Generate an interesting pattern
        if tension > 0.7:
            # High tension - jumping intervals
            indices = [0, 4, 2, 5, 1, 4, 3, 7]
            print(f"      HIGH TENSION pattern: jumping intervals {indices}")
        elif tension > 0.4:
            # Medium - melodic
            indices = [0, 2, 3, 2, 4, 3, 2, 1]
            print(f"      MEDIUM TENSION pattern: melodic {indices}")
        else:
            # Low - stepwise
            indices = [0, 1, 2, 1, 3, 2, 1, 0]
            print(f"      LOW TENSION pattern: stepwise {indices}")

        for idx in indices:
            if idx < len(scale):
                pattern.append(scale[idx])

        print(f"      Generated pattern frequencies: {[f'{f:.1f}Hz' for f in pattern]}")

        # Calculate how many times to play the pattern
        num_notes = int(section_duration / note_duration)
        print(f"      Will play {num_notes} notes over {section_duration}s")

        # Generate CONTINUOUS audio with smooth frequency transitions (like real 1980s synths!)
        total_samples = int(section_duration * self.sample_rate)
        continuous_signal = self.synth.oscillator(110, section_duration, waveform)  # Base frequency

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

        # Apply frequency modulation to continuous signal
        phase = 0.0
        modulated_signal = np.zeros_like(continuous_signal)

        for i in range(len(continuous_signal)):
            # Current frequency
            current_freq = 110 * freq_modulation[i]

            # Update phase
            phase += 2 * np.pi * current_freq / self.sample_rate

            # Generate modulated sample
            if waveform == 'saw':
                modulated_signal[i] = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
            elif waveform == 'pulse':
                modulated_signal[i] = 1.0 if ((phase / (2 * np.pi)) % 1.0) < 0.3 else -1.0
            elif waveform == 'square':
                modulated_signal[i] = 1.0 if np.sin(phase) > 0 else -1.0
            elif waveform == 'triangle':
                modulated_signal[i] = 2.0 * np.abs(2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
            else:  # sine
                modulated_signal[i] = np.sin(phase)

        # Apply filter to the entire modulated signal
        filtered_signal = self.synth.moog_filter(modulated_signal, filter_base + filter_envelope_amount/2, resonance)
        print(f"      Applied Moog filter: {filter_base + filter_envelope_amount/2:.0f}Hz cutoff, {resonance} resonance")

        # Apply smooth amplitude envelope over entire section
        section_envelope = np.ones(total_samples)
        fade_samples = int(0.1 * self.sample_rate)  # 100ms fades

        # Fade in
        section_envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        # Fade out
        section_envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        samples = filtered_signal * section_envelope * 0.3 * volume_multiplier  # Volume level with process modulation

        # Add key moments as filter sweeps or resonant hits
        print(f"      Processing {len(section.get('key_moments', []))} key moments:")
        for moment in section.get('key_moments', []):
            moment_time = (moment['ply'] - section['start_ply']) * note_duration
            moment_sample_pos = int(moment_time * self.sample_rate)

            if moment_sample_pos < len(samples) - self.sample_rate:
                if moment['type'] in ['BLUNDER', 'MISTAKE']:
                    # Descending filter sweep
                    print(f"        {moment['type']} at ply {moment['ply']}: descending filter sweep (55Hz, 5000->500Hz)")
                    sweep = self.synth.create_synth_note(
                        freq=55,  # Low note
                        duration=0.5,
                        waveform='saw',
                        filter_base=5000,
                        filter_env_amount=-4500,  # Negative for closing
                        resonance=3.5,
                        amp_env=(0.001, 0.01, 0.5, 0.4),
                        filter_env=(0.001, 0.5, 0.0, 0.4)
                    )
                elif moment['type'] in ['BRILLIANT', 'STRONG']:
                    # Rising resonant sweep
                    print(f"        {moment['type']} at ply {moment['ply']}: rising filter sweep (220Hz, 200->8200Hz)")
                    sweep = self.synth.create_synth_note(
                        freq=220,
                        duration=0.3,
                        waveform='pulse',
                        filter_base=200,
                        filter_env_amount=8000,  # Big opening
                        resonance=3.0,
                        amp_env=(0.001, 0.05, 0.7, 0.1),
                        filter_env=(0.001, 0.2, 0.5, 0.1)
                    )
                else:
                    # Generic accent
                    print(f"        {moment['type']} at ply {moment['ply']}: generic accent ({note_freq*2:.1f}Hz)")
                    sweep = self.synth.create_synth_note(
                        freq=note_freq * 2,
                        duration=0.2,
                        waveform='square',
                        filter_base=1000,
                        filter_env_amount=2000,
                        resonance=2.0,
                        amp_env=(0.001, 0.01, 0.8, 0.05)
                    )

                # Mix in the sweep
                for i, sample in enumerate(sweep):
                    if moment_sample_pos + i < len(samples):
                        samples[moment_sample_pos + i] = samples[moment_sample_pos + i] * 0.5 + sample * 0.5

        return np.array(samples)

    def compose(self):
        """Create the full composition"""
        print("\n♫ SUBTRACTIVE SYNTHESIS CHESS MUSIC")
        print(f"Result: {self.tags.get('game_result', '?')}")
        print(f"Narrative: {self.tags.get('overall_narrative', 'UNKNOWN')}")

        composition = []

        print("\nSynthesizing sections:")
        for section in self.tags.get('sections', []):
            print(f"  {section['name']}: {section['narrative']} (tension: {section['tension']:.2f})")

            section_music = self.compose_section(section)
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
