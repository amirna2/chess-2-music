#!/usr/bin/env python3
"""
SUBTRACTIVE SYNTHESIZER FOR CHESS MUSIC
Real electronic music synthesis with filters and envelopes
"""

import json
import math
import wave
import struct
import sys
import numpy as np

class SubtractiveSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0

        # Filter state variables (for continuity between notes)
        self.filter_z1 = 0.0
        self.filter_z2 = 0.0

    def oscillator(self, freq, duration, waveform='saw'):
        """Generate basic waveforms rich in harmonics"""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        phase = freq * t

        if waveform == 'saw':
            # Sawtooth - all harmonics (1/n amplitude)
            signal = 2.0 * (phase % 1.0) - 1.0

        elif waveform == 'square':
            # Square wave - only odd harmonics
            signal = np.sign(np.sin(2 * np.pi * phase))

        elif waveform == 'pulse':
            # Pulse wave with variable width (PWM capability)
            width = 0.3  # Can be modulated later
            signal = np.where((phase % 1.0) < width, 1.0, -1.0)

        elif waveform == 'triangle':
            # Triangle - odd harmonics (1/n² amplitude)
            signal = 2.0 * np.abs(2.0 * (phase % 1.0) - 1.0) - 1.0

        else:  # sine
            signal = np.sin(2 * np.pi * phase)

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

        # Simplified input gain compensation (can be more complex)
        input_gain = 0.5 * (1.0 - resonance / 4.0)

        # State variables for 4 cascaded 1-pole filters (persistent)
        z1, z2, z3, z4 = 0.0, 0.0, 0.0, 0.0

        filtered = np.zeros_like(signal)

        for i in range(len(signal)):
            input_sample = signal[i]

            # Feedback loop with non-linear saturation
            # The feedback is applied to the input
            input_with_feedback = input_sample - resonance * z4

            # The non-linear element is applied to each stage's input
            stage1_in = np.tanh(input_with_feedback * input_gain)
            stage2_in = np.tanh(z1)
            stage3_in = np.tanh(z2)
            stage4_in = np.tanh(z3)

            # 4 cascaded 1-pole low-pass filters (Euler integration)
            # Note: A bilinear transform would be more accurate but is also more complex.
            z1 = z1 + f * (stage1_in - z1)
            z2 = z2 + f * (stage2_in - z2)
            z3 = z3 + f * (stage3_in - z3)
            z4 = z4 + f * (stage4_in - z4)

            filtered[i] = z4

        return filtered

    def adsr_envelope(self, num_samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
        """
        ADSR envelope generator - shapes the amplitude over time
        Times in seconds, sustain is level (0-1)
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

        # Attack
        if attack_samples > 0:
            envelope[current:current+attack_samples] = np.linspace(0, 1, attack_samples)
            current += attack_samples

        # Decay
        if decay_samples > 0 and current < num_samples:
            end = min(current + decay_samples, num_samples)
            envelope[current:end] = np.linspace(1, sustain, end - current)
            current = end

        # Sustain
        if sustain_samples > 0 and current < num_samples:
            end = min(current + sustain_samples, num_samples)
            envelope[current:end] = sustain
            current = end

        # Release
        if current < num_samples:
            envelope[current:] = np.linspace(sustain, 0, num_samples - current)

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
        if 'TACTICAL_BATTLE' in narrative or 'TACTICAL_THRILLER' in narrative:
            # Fast tactical exchanges - aggressive saw with rapid filter sweeps
            waveform = 'saw'
            filter_base = 300
            filter_envelope_amount = 5000 * tension
            resonance = 3.0  # Very resonant for intensity
            scale = self.scales['phrygian']
            note_duration = 0.2  # Very fast notes
            print(f"      TACTICAL_BATTLE: saw wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'KING_HUNT' in narrative:
            # Pursuing, relentless sound - pulse wave building intensity
            waveform = 'pulse'
            filter_base = 400
            filter_envelope_amount = 6000 * tension
            resonance = 2.0
            scale = self.scales['minor']
            note_duration = 0.3  # Moderate speed but building
            print(f"      KING_HUNT: pulse wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

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

        elif 'POSITIONAL_MANEUVERING' in narrative or 'QUIET_BUILDING' in narrative:
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

        samples = filtered_signal * section_envelope * 0.3  # Volume level

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
        print("Usage: python chess_synth.py tags.json")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        tags = json.load(f)

    composer = ChessSynthComposer(tags)
    composer.save()

if __name__ == '__main__':
    main()
