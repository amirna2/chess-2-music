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
        Moog-style 4-pole (24dB/octave) low-pass filter
        This is THE classic synth filter sound
        """
        if cutoff_hz >= self.nyquist * 0.95:
            return signal  # Bypass if cutoff too high

        cutoff = np.clip(cutoff_hz / self.nyquist, 0.001, 0.99)

        # Resonance (0-4, self-oscillates at 4)
        resonance = np.clip(resonance, 0.0, 3.95)

        # Stilson/Smith Moog filter approximation
        f = cutoff * 1.16
        fb = resonance * (1.0 - 0.15 * f * f)

        filtered = np.zeros_like(signal)

        # State variables for 4 cascaded 1-pole filters
        z1, z2, z3, z4 = 0.0, 0.0, 0.0, 0.0

        for i in range(len(signal)):
            input_sample = signal[i]
            input_sample -= z4 * fb  # Feedback
            input_sample *= 0.35013 * (f*f)*(f*f)  # Input gain compensation

            # Four cascaded 1-pole filters (Moog ladder)
            z1 = input_sample + 0.3 * z1 + (1 - f) * z1
            z2 = z1 + 0.3 * z2 + (1 - f) * z2
            z3 = z2 + 0.3 * z3 + (1 - f) * z3
            z4 = z3 + 0.3 * z4 + (1 - f) * z4

            filtered[i] = z4

        return filtered * 0.5  # Scale output

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
        if 'TACTICAL' in narrative or 'CHAOS' in narrative:
            # Aggressive sound - saw wave, fast filter sweeps
            waveform = 'saw'
            filter_base = 200
            filter_envelope_amount = 4000 * tension
            resonance = 2.5  # High resonance for aggressive sound
            scale = self.scales['phrygian']
            note_duration = 0.25  # Fast notes
            print(f"      TACTICAL/CHAOS: saw wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'ATTACK' in narrative or 'CRUSHING' in narrative:
            # Building attack - pulse wave, opening filter
            waveform = 'pulse'
            filter_base = 500
            filter_envelope_amount = 3000 * tension
            resonance = 1.5
            scale = self.scales['minor']
            note_duration = 0.4
            print(f"      ATTACK/CRUSHING: pulse wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance}")

        elif 'DEFENSE' in narrative or 'DESPERATE' in narrative:
            # Dark, closed sound
            waveform = 'saw'
            filter_base = 150  # Very closed filter
            filter_envelope_amount = 1000
            resonance = 3.0  # Self-oscillating almost
            scale = self.scales['phrygian']
            note_duration = 0.5
            print(f"      DEFENSE/DESPERATE: saw wave, filter {filter_base}Hz + {filter_envelope_amount}Hz sweep, resonance {resonance} (dark)")

        else:  # POSITIONAL, QUIET, etc.
            # Smooth, evolving sound
            waveform = 'triangle'
            filter_base = 800
            filter_envelope_amount = 2000 * tension
            resonance = 0.5  # Gentle
            scale = self.scales['dorian']
            note_duration = 0.6
            print(f"      POSITIONAL: triangle wave, filter {filter_base}Hz + {filter_envelope_amount:.0f}Hz sweep, resonance {resonance} (smooth)")

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

        # Play the pattern with variations
        for i in range(num_notes):
            note_freq = pattern[i % len(pattern)]

            # Vary the frequency slightly for movement
            if i % 4 == 0:
                note_freq *= 2  # Octave up occasionally
            elif i % 7 == 0:
                note_freq *= 0.5  # Octave down

            # Modulate filter based on position in section
            position_in_section = i / max(num_notes, 1)
            filter_mod = filter_base + (position_in_section * 1000 * tension)

            # Create the note
            note = self.synth.create_synth_note(
                freq=note_freq,
                duration=note_duration,
                waveform=waveform,
                filter_base=filter_mod,
                filter_env_amount=filter_envelope_amount,
                resonance=resonance,
                amp_env=(0.005, 0.05, 0.6, 0.1),  # Fast attack for punch
                filter_env=(0.01, 0.1 + tension * 0.1, 0.3, 0.2)
            )

            samples.extend(note)

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
