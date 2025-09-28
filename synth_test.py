#!/usr/bin/env python3
"""
INTERACTIVE SYNTHESIZER TESTER
Test synth parameters without needing tag files
"""

import sys
import time
import tempfile
import os
from synth_composer import SubtractiveSynth

# Try to import audio playback libraries
AUDIO_BACKEND = None

try:
    import pygame
    AUDIO_BACKEND = 'pygame'
except ImportError:
    try:
        import pyaudio
        import wave
        AUDIO_BACKEND = 'pyaudio'
    except ImportError:
        try:
            import subprocess
            # Check if system has afplay (macOS) or aplay (Linux)
            result = subprocess.run(['which', 'afplay'], capture_output=True)
            if result.returncode == 0:
                AUDIO_BACKEND = 'afplay'
            else:
                result = subprocess.run(['which', 'aplay'], capture_output=True)
                if result.returncode == 0:
                    AUDIO_BACKEND = 'aplay'
        except:
            pass

def play_audio(samples, sample_rate=44100):
    """Play audio using available backend"""
    if AUDIO_BACKEND == 'pygame':
        return play_with_pygame(samples, sample_rate)
    elif AUDIO_BACKEND == 'pyaudio':
        return play_with_pyaudio(samples, sample_rate)
    elif AUDIO_BACKEND in ['afplay', 'aplay']:
        return play_with_system(samples, sample_rate)
    else:
        print("No audio backend available. Install pygame, pyaudio, or ensure afplay/aplay is available.")
        return False

def play_with_pygame(samples, sample_rate):
    """Play using pygame"""
    try:
        import pygame
        import numpy as np

        pygame.mixer.pre_init(frequency=sample_rate, size=-16, channels=1, buffer=512)
        pygame.mixer.init()

        # Convert to 16-bit
        samples_16bit = (samples * 32767).astype('int16')
        sound = pygame.sndarray.make_sound(samples_16bit)
        sound.play()

        # Wait for sound to finish
        while pygame.mixer.get_busy():
            time.sleep(0.1)

        pygame.mixer.quit()
        return True
    except Exception as e:
        print(f"Pygame playback failed: {e}")
        return False

def play_with_pyaudio(samples, sample_rate):
    """Play using pyaudio"""
    try:
        import pyaudio
        import numpy as np

        p = pyaudio.PyAudio()

        # Convert to 16-bit
        samples_16bit = (samples * 32767).astype('int16')

        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=sample_rate,
                       output=True)

        stream.write(samples_16bit.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
    except Exception as e:
        print(f"PyAudio playback failed: {e}")
        return False

def play_with_system(samples, sample_rate):
    """Play using system command (afplay/aplay)"""
    try:
        import wave
        import struct

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_filename = tmp.name

            with wave.open(temp_filename, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)

                for sample in samples:
                    int_sample = int(sample * 30000)
                    int_sample = max(-32000, min(32000, int_sample))
                    wav.writeframes(struct.pack('<h', int_sample))

        # Play with system command
        if AUDIO_BACKEND == 'afplay':
            os.system(f'afplay "{temp_filename}"')
        else:  # aplay
            os.system(f'aplay "{temp_filename}" 2>/dev/null')

        # Clean up
        os.unlink(temp_filename)
        return True
    except Exception as e:
        print(f"System playback failed: {e}")
        return False

class SynthTester:
    def __init__(self):
        self.synth = SubtractiveSynth(44100)
        self.presets = {
            'TACTICAL_BATTLE': {
                'waveform': 'saw',
                'filter_base': 300,
                'filter_env_amount': 5000,
                'resonance': 3.0,
                'freq': 110
            },
            'KING_HUNT': {
                'waveform': 'pulse',
                'filter_base': 400,
                'filter_env_amount': 6000,
                'resonance': 2.0,
                'freq': 165
            },
            'DESPERATE_DEFENSE': {
                'waveform': 'saw',
                'filter_base': 120,
                'filter_env_amount': 800,
                'resonance': 3.5,
                'freq': 82
            },
            'COMPLEX_STRUGGLE': {
                'waveform': 'triangle',
                'filter_base': 600,
                'filter_env_amount': 3000,
                'resonance': 1.0,
                'freq': 131
            },
            'TUMBLING_DEFEAT': {
                'waveform': 'saw',
                'filter_base': 200,
                'filter_env_amount': 4000,
                'resonance': 2.8,
                'freq': 98
            }
        }

    def test_note(self, freq=220, duration=2.0, **kwargs):
        """Test a single note with given parameters"""
        print(f"\n♫ Testing: {freq}Hz, {duration}s")
        print(f"  Waveform: {kwargs.get('waveform', 'saw')}")
        print(f"  Filter: {kwargs.get('filter_base', 1000)}Hz + {kwargs.get('filter_env_amount', 2000)}Hz sweep")
        print(f"  Resonance: {kwargs.get('resonance', 1.0)}")

        samples = self.synth.create_synth_note(freq, duration, **kwargs)

        if play_audio(samples):
            print("  ✓ Played successfully")
        else:
            print("  ✗ Playback failed")

        return samples

    def test_preset(self, preset_name, duration=4.0):
        """Test a narrative preset"""
        if preset_name not in self.presets:
            print(f"Unknown preset: {preset_name}")
            return

        print(f"\n🎭 Testing preset: {preset_name}")
        preset = self.presets[preset_name].copy()
        freq = preset.pop('freq')

        return self.test_note(freq, duration, **preset)

    def interactive_menu(self):
        """Interactive menu for testing"""
        print("\n🎹 SYNTHESIZER INTERACTIVE TESTER")
        print(f"Audio backend: {AUDIO_BACKEND or 'None (no playback)'}")
        print("\nCommands:")
        print("  p <preset>           - Play preset (TACTICAL_BATTLE, KING_HUNT, etc.)")
        print("  n <freq>             - Play note at frequency")
        print("  c <freq> <wave> <filter> <res> - Custom note")
        print("  w <wave>             - Set waveform (saw, pulse, square, triangle, sine)")
        print("  f <base> <amount>    - Set filter (base_hz envelope_amount)")
        print("  r <resonance>        - Set resonance (0.0-4.0)")
        print("  d <duration>         - Set duration")
        print("  l                    - List presets")
        print("  s                    - Show current settings")
        print("  q                    - Quit")

        # Current settings
        current = {
            'waveform': 'saw',
            'filter_base': 1000,
            'filter_env_amount': 2000,
            'resonance': 1.0,
            'duration': 2.0
        }

        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == 'q':
                    break
                elif cmd[0] == 'l':
                    print("\nAvailable presets:")
                    for name, params in self.presets.items():
                        print(f"  {name}: {params['waveform']} @ {params['freq']}Hz")
                elif cmd[0] == 's':
                    print(f"\nCurrent settings:")
                    for key, value in current.items():
                        print(f"  {key}: {value}")
                elif cmd[0] == 'p' and len(cmd) > 1:
                    preset = cmd[1].upper()
                    self.test_preset(preset, current['duration'])
                elif cmd[0] == 'n' and len(cmd) > 1:
                    try:
                        freq = float(cmd[1])
                        self.test_note(freq, **current)
                    except ValueError:
                        print("Invalid frequency")
                elif cmd[0] == 'c' and len(cmd) >= 5:
                    try:
                        freq = float(cmd[1])
                        wave = cmd[2]
                        filter_base = float(cmd[3])
                        resonance = float(cmd[4])
                        self.test_note(freq, current['duration'],
                                     waveform=wave,
                                     filter_base=filter_base,
                                     filter_env_amount=2000,
                                     resonance=resonance)
                    except ValueError:
                        print("Invalid parameters: c <freq> <wave> <filter> <res>")
                elif cmd[0] == 'w' and len(cmd) > 1:
                    if cmd[1] in ['saw', 'pulse', 'square', 'triangle', 'sine']:
                        current['waveform'] = cmd[1]
                        print(f"Waveform set to: {cmd[1]}")
                    else:
                        print("Invalid waveform. Use: saw, pulse, square, triangle, sine")
                elif cmd[0] == 'f' and len(cmd) >= 3:
                    try:
                        current['filter_base'] = float(cmd[1])
                        current['filter_env_amount'] = float(cmd[2])
                        print(f"Filter set to: {current['filter_base']}Hz + {current['filter_env_amount']}Hz")
                    except ValueError:
                        print("Invalid filter parameters: f <base_hz> <envelope_amount>")
                elif cmd[0] == 'r' and len(cmd) > 1:
                    try:
                        current['resonance'] = float(cmd[1])
                        print(f"Resonance set to: {current['resonance']}")
                    except ValueError:
                        print("Invalid resonance")
                elif cmd[0] == 'd' and len(cmd) > 1:
                    try:
                        current['duration'] = float(cmd[1])
                        print(f"Duration set to: {current['duration']}s")
                    except ValueError:
                        print("Invalid duration")
                else:
                    print("Unknown command or missing parameters")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    if len(sys.argv) > 1:
        # Command line mode
        tester = SynthTester()

        if sys.argv[1] in tester.presets:
            tester.test_preset(sys.argv[1].upper())
        elif sys.argv[1].replace('.', '').isdigit():
            freq = float(sys.argv[1])
            tester.test_note(freq)
        else:
            print(f"Unknown preset or frequency: {sys.argv[1]}")
    else:
        # Interactive mode
        tester = SynthTester()
        tester.interactive_menu()

if __name__ == '__main__':
    main()
