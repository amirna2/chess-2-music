#!/usr/bin/env python3
"""
MUSICAL CHESS BATTLE COMPOSER
Two voices (White and Black) in combat - but playing actual MUSICAL PHRASES!
Works with ANY game's JSON tags
"""

import json
import math
import wave
import struct
import sys

class MusicalChessBattleComposer:
    def __init__(self, chess_tags):
        self.tags = chess_tags
        self.sample_rate = 44100

        # Handle any game's duration/plies
        self.total_duration = chess_tags.get('duration_seconds', 60)
        self.total_plies = chess_tags.get('total_plies', 40)
        self.seconds_per_ply = self.total_duration / self.total_plies

        # Musical scale for actual melodies (A minor pentatonic - always sounds good)
        # KEEPING IT IN COMFORTABLE RANGE - no ear-piercing highs!
        self.scale = [110, 130.81, 146.83, 164.81, 196, 220, 261.63, 293.66]  # A2-D4 range

        # White voice - melodic phrases (mid-range, not too high!)
        self.white_phrases = {
            'attack': [261.63, 293.66, 329.63, 293.66],  # C4-E4 range
            'defend': [261.63, 220, 196, 220],  # Falling back
            'maneuver': [261.63, 220, 261.63, 293.66, 261.63],  # Probing
        }

        # Black voice - counter phrases (lower register)
        self.black_phrases = {
            'attack': [110, 130.81, 146.83, 164.81],  # A2-E3 range
            'defend': [146.83, 130.81, 110, 110],  # Defensive
            'maneuver': [110, 146.83, 130.81, 146.83, 110],  # Response
        }

        # Rhythm patterns (not monotonous!)
        self.rhythms = {
            'aggressive': [0.125, 0.125, 0.25, 0.5],
            'defensive': [0.5, 0.25, 0.25, 1.0],
            'tactical': [0.25, 0.125, 0.125, 0.25, 0.125],
        }

        self.current_eval = 0

    def play_phrase(self, notes, rhythm, intensity=0.5, voice='white'):
        """Play a musical PHRASE with smooth note transitions"""
        samples = []

        # Make sure we have matching rhythm for notes
        rhythm_cycle = rhythm * (len(notes) // len(rhythm) + 1)

        # Track phase across the entire phrase to prevent clicks
        phase = 0.0
        prev_freq = notes[0] if notes else 440

        for note_idx, freq in enumerate(notes):
            if note_idx < len(rhythm_cycle):
                duration = rhythm_cycle[note_idx]
            else:
                duration = 0.25

            num_samples = int(self.sample_rate * duration)

            # Crossfade between notes to eliminate clicks
            crossfade_samples = min(200, num_samples // 4)

            for i in range(num_samples):
                global_t = (len(samples) + i) / self.sample_rate

                # Smooth frequency transition using phase continuity
                if i < crossfade_samples and note_idx > 0:
                    # Crossfade from previous frequency to current
                    fade_ratio = i / crossfade_samples
                    blended_freq = prev_freq * (1 - fade_ratio) + freq * fade_ratio
                else:
                    blended_freq = freq

                # Update phase continuously
                phase += 2 * math.pi * blended_freq / self.sample_rate
                wave = math.sin(phase) * 0.6  # Scale down main wave to prevent clipping

                if voice == 'white':
                    # Brighter timbre - reduced harmonics
                    wave += 0.08 * math.sin(phase * 1.5)  # Fifth (reduced)
                    wave += 0.02 * math.sin(phase * 2)  # Octave (reduced)
                else:
                    # Darker timbre - reduced harmonics
                    wave += 0.15 * math.sin(phase * 0.5)  # Sub-octave (reduced)
                    wave += 0.05 * math.sin(phase * 0.75)  # Sub-fifth (reduced)

                # MUCH SIMPLER ENVELOPE - no complex calculations
                # Just use a basic envelope that doesn't kill volume

                # Simple note envelope only
                note_attack = min(50, num_samples // 10)  # Shorter attack
                note_release = min(100, num_samples // 8)  # Shorter release

                if i < note_attack:
                    envelope = 0.3 + 0.7 * (i / note_attack)  # Start at 0.3, not 0!
                elif i > num_samples - note_release:
                    envelope = 0.3 + 0.7 * ((num_samples - i) / note_release)
                else:
                    envelope = 1.0

                # NO complex envelope multiplication!
                final_envelope = envelope

                # REMOVE vibrato for now - might be causing periodic clicks
                vibrato = 1.0  # 1 + 0.02 * math.sin(2 * math.pi * 5 * global_t)

                samples.append(wave * final_envelope * intensity * 0.3)  # Reduced to prevent layering clipping

            prev_freq = freq

        return samples

    def create_battle_exchange(self, tension):
        """Create a musical exchange between white and black"""
        samples = []

        # Choose phrases based on tension
        if tension > 0.7:
            white_phrase = self.white_phrases['attack']
            black_phrase = self.black_phrases['defend']
            rhythm = self.rhythms['aggressive']
        elif tension > 0.4:
            white_phrase = self.white_phrases['maneuver']
            black_phrase = self.black_phrases['maneuver']
            rhythm = self.rhythms['tactical']
        else:
            white_phrase = self.white_phrases['defend']
            black_phrase = self.black_phrases['attack']
            rhythm = self.rhythms['defensive']

        # White plays a phrase
        white_music = self.play_phrase(white_phrase, rhythm, 0.5 + tension * 0.3, 'white')
        samples.extend(white_music)

        # COMPLETELY REMOVE THE PAUSE - it's causing clicks
        # NO pause, NO transition tone - just continue directly

        # Black responds with a phrase
        black_music = self.play_phrase(black_phrase, rhythm, 0.5 + tension * 0.3, 'black')
        samples.extend(black_music)

        return samples

    def create_positional_section(self, duration):
        """Both sides playing overlapping patterns"""
        samples = []
        num_samples = int(self.sample_rate * duration)

        # Create polyrhythm - 3 against 4 (comfortable frequencies)
        white_pattern = [261.63, 220, 293.66]  # C4 A3 D4
        black_pattern = [110, 130.81, 146.83, 130.81]  # A2 C3 D3 C3

        # Track phase for smooth transitions
        white_phase = 0.0
        black_phase = 0.0
        bass_phase = 0.0

        prev_white_freq = white_pattern[0]
        prev_black_freq = black_pattern[0]

        for i in range(num_samples):
            t = i / self.sample_rate
            sample = 0

            # White plays pattern - simple indexing, no interpolation
            white_idx = int(t * 2) % len(white_pattern)
            white_freq = white_pattern[white_idx]

            # Keep phase continuous WITHOUT complex adjustments
            white_phase += 2 * math.pi * white_freq / self.sample_rate
            sample += 0.08 * math.sin(white_phase)  # Reduced for no clipping
            prev_white_freq = white_freq

            # Black plays different pattern - simple indexing
            black_idx = int(t * 1.5) % len(black_pattern)
            black_freq = black_pattern[black_idx]

            # Keep phase continuous WITHOUT complex adjustments
            black_phase += 2 * math.pi * black_freq / self.sample_rate
            sample += 0.08 * math.sin(black_phase)  # Reduced for no clipping
            prev_black_freq = black_freq

            # Add subtle bass drone with phase tracking
            bass_phase += 2 * math.pi * 55 / self.sample_rate
            sample += 0.03 * math.sin(bass_phase)  # Reduced for no clipping

            samples.append(sample)

        return samples

    def create_dramatic_moment(self, moment_type, score):
        """Create musical drama for key moments"""
        samples = []

        if moment_type == 'BLUNDER':
            # Descending cascade (staying in comfortable range)
            cascade = [329.63, 293.66, 261.63, 220, 196, 164.81, 146.83, 110]  # E4 down to A2
            for freq in cascade:
                for i in range(int(self.sample_rate * 0.1)):
                    t = i / self.sample_rate
                    # Add dissonance
                    sample = math.sin(2 * math.pi * freq * t)
                    sample += 0.3 * math.sin(2 * math.pi * freq * 1.06 * t)
                    envelope = math.exp(-10 * t)
                    samples.append(sample * envelope * 0.5)

        elif moment_type in ['BRILLIANT', 'STRONG']:
            # Triumphant ascending arpeggio (not too high!)
            arpeggio = [110, 130.81, 164.81, 220, 261.63, 329.63]  # A2 to E4
            rhythm = [0.1, 0.1, 0.1, 0.2, 0.2, 0.5]
            moment_music = self.play_phrase(arpeggio, rhythm, 1.0, 'white')  # MAXIMUM
            samples.extend(moment_music)

        elif moment_type in ['INACCURACY', 'MISTAKE']:
            # Dissonant interval (comfortable range)
            for i in range(int(self.sample_rate * 0.3)):
                t = i / self.sample_rate
                # Tritone but not too high
                sample = 0.3 * math.sin(2 * math.pi * 220 * t)
                sample += 0.3 * math.sin(2 * math.pi * 311.13 * t)  # Eb4
                envelope = 1 - t/0.3
                samples.append(sample * envelope)

        else:
            # Generic accent using score
            intensity = score / 10.0 if score else 0.5
            accent_phrase = [220, 261.63, 220]  # A3 C4 A3
            accent_music = self.play_phrase(accent_phrase, [0.1, 0.1, 0.2], intensity)
            samples.extend(accent_music)

        return samples

    def compose_section(self, section):
        """Compose each section with actual music"""
        print(f"  {section['name']}: {section['narrative']} (tension: {section['tension']:.2f})")

        section_samples = []

        # Parse duration correctly
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

        section_plies = section['end_ply'] - section['start_ply'] + 1

        # Create music based on narrative (but with ACTUAL MUSIC!)
        narrative = section.get('narrative', 'UNKNOWN')
        tension = section.get('tension', 0.5)

        if 'CHAOS' in narrative or 'TACTICAL' in narrative:
            # Multiple rapid exchanges
            exchange_time = section_duration / 4
            for _ in range(4):
                exchange = self.create_battle_exchange(tension)
                section_samples.extend(exchange)

        elif 'KING_HUNT' in narrative:
            # Relentless pursuit - rapid ascending phrases targeting the king
            for _ in range(section_plies // 2):
                # Hunting phrase - aggressive ascending runs
                hunt = self.play_phrase(
                    [164.81, 196, 220, 261.63, 293.66, 329.63],  # E3 to E4 - climbing hunt
                    [0.125, 0.125, 0.125, 0.125, 0.25, 0.125],  # Fast, relentless rhythm
                    1.0, 'white'  # MAXIMUM
                )
                section_samples.extend(hunt)

                # King trying to escape - descending, panicked phrase
                escape = self.play_phrase(
                    [293.66, 261.63, 220, 196, 164.81],  # Descending escape attempt
                    [0.125, 0.125, 0.125, 0.125, 0.25],  # Frantic rhythm
                    1.0, 'black'  # MAXIMUM
                )
                section_samples.extend(escape)

        elif 'CRUSHING' in narrative or 'ATTACK' in narrative:
            # One side plays strong theme, other plays weak response
            for _ in range(section_plies // 2):
                # Strong attack phrase (comfortable range)
                attack = self.play_phrase(
                    [220, 261.63, 293.66, 329.63, 293.66],  # A3-E4 range
                    [0.25, 0.25, 0.25, 0.5, 0.25],
                    1.0, 'white'  # MAXIMUM
                )
                section_samples.extend(attack)

                # Weak defense
                defense = self.play_phrase(
                    [146.83, 130.81, 110],  # D3 C3 A2
                    [0.25, 0.25, 0.5],
                    1.0, 'black'  # MAXIMUM
                )
                section_samples.extend(defense)

        elif 'DEFENSE' in narrative:
            # Under pressure - syncopated rhythm (comfortable range)
            for _ in range(section_plies // 2):
                defense_phrase = self.play_phrase(
                    [146.83, 164.81, 146.83, 130.81, 110],  # D3 E3 D3 C3 A2
                    [0.125, 0.125, 0.25, 0.125, 0.375],
                    1.0, 'black'  # MAXIMUM
                )
                section_samples.extend(defense_phrase)

        else:  # COMPLEX_STRUGGLE, POSITIONAL, etc.
            # Overlapping patterns
            positional = self.create_positional_section(section_duration)
            section_samples.extend(positional)

        # Add key moments as musical events
        for moment in section.get('key_moments', []):
            moment_ply_offset = moment['ply'] - section['start_ply']
            moment_time = moment_ply_offset * (section_duration / section_plies)
            moment_pos = int(moment_time * self.sample_rate)

            print(f"    Key moment: {moment['type']} at ply {moment['ply']}")

            moment_music = self.create_dramatic_moment(moment['type'], moment.get('score', 5))

            # Layer dramatic moment ON TOP but PREVENT CLIPPING
            if moment_pos < len(section_samples):
                for i, moment_sample in enumerate(moment_music):
                    sample_idx = moment_pos + i
                    if sample_idx < len(section_samples):
                        # Scale down BOTH signals before mixing to prevent clipping
                        existing = section_samples[sample_idx] * 0.4  # Reduce base music more
                        dramatic = moment_sample * 0.3  # Reduce dramatic moment more
                        section_samples[sample_idx] = existing + dramatic  # Mix without clipping

        return section_samples

    def compose(self):
        """Create the complete musical chess battle"""
        print("\n♫ MUSICAL CHESS BATTLE")
        print(f"Result: {self.tags.get('game_result', '?')}")
        print(f"Narrative: {self.tags.get('overall_narrative', 'UNKNOWN')}")

        composition = []

        print("\nComposing sections:")
        for section in self.tags.get('sections', []):
            section_music = self.compose_section(section)
            composition.extend(section_music)

            # NO TRANSITION TONES - they cause tapping
            # Just add a brief silence with smooth fade from previous section
            fade_samples = int(self.sample_rate * 0.02)  # 20ms fade out only

            # Fade out end of previous section smoothly
            for i in range(min(fade_samples, len(composition))):
                fade_idx = len(composition) - fade_samples + i
                if fade_idx >= 0:
                    composition[fade_idx] *= (fade_samples - i) / fade_samples

            # Brief silence (no clicking transition tones)
            composition.extend([0.0] * int(self.sample_rate * 0.1))

        # Ending based on result
        game_result = self.tags.get('game_result', '*')
        if '1-0' in game_result:
            # White victory - triumphant major progression (comfortable range!)
            ending = self.play_phrase(
                [164.81, 196, 261.63, 293.66, 329.63, 392],  # E3 to G4
                [0.2, 0.2, 0.2, 0.3, 0.3, 1.0],
                1.0, 'white'  # MAXIMUM
            )
        elif '0-1' in game_result:
            # Black victory - dark minor progression
            ending = self.play_phrase(
                [110, 130.81, 146.83, 164.81, 196, 220],  # A2 to A3
                [0.2, 0.2, 0.2, 0.3, 0.3, 1.0],
                1.0, 'black'  # MAXIMUM
            )
        else:
            # Draw or unknown - suspended
            ending = self.play_phrase([164.81, 220, 164.81], [0.5, 0.5, 1.0], 1.0)  # MAXIMUM

        composition.extend(ending)

        # PROPER ANTI-CLIPPING using tanh soft limiter
        if composition:
            max_val = max(abs(min(composition)), abs(max(composition)))
            print(f"DEBUG: max_val BEFORE limiting = {max_val}")

            # Apply soft limiter using tanh function - prevents clipping while keeping loudness
            import math
            limited_composition = []
            for sample in composition:
                # Soft limit using tanh - scales smoothly instead of hard clipping
                # Scale input for tanh (around 1.0 works well)
                scaled_input = sample / 1.0
                # Apply tanh limiting
                limited = math.tanh(scaled_input)
                # Scale back to target amplitude (0.95 for slight headroom)
                limited_composition.append(limited * 0.95)

            composition = limited_composition
            new_max = max(abs(min(composition)), abs(max(composition)))
            print(f"DEBUG: After tanh limiting, max_val = {new_max}")

        duration = len(composition) / self.sample_rate
        print(f"\n✓ Battle complete: {duration:.1f} seconds")

        return composition

    def save(self, filename='musical_chess_battle.wav'):
        """Save the composition"""
        composition = self.compose()

        with wave.open(filename, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)

            # Debug first few samples to see what we're writing
            print(f"DEBUG: First 5 samples being written:")
            for i, sample in enumerate(composition[:5]):
                int_sample = int(sample * 32767)
                print(f"  Sample {i}: float={sample:.4f} -> int={int_sample}")

            for sample in composition:
                # Use FULL 16-bit range for MAXIMUM volume
                int_sample = int(sample * 32767)  # Full positive range
                int_sample = max(-32768, min(32767, int_sample))
                wav.writeframes(struct.pack('<h', int_sample))

        print(f"Saved: {filename}")
        print(f"Play: afplay {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python musical_chess_battle.py tags.json")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        tags = json.load(f)

    composer = MusicalChessBattleComposer(tags)
    composer.save()

if __name__ == '__main__':
    main()
