#!/usr/bin/env python3
"""
SIMPLE SYNTH TESTER - Just the essentials!
Quick testing of basic synth components
"""

import sys
from synth_composer import SubtractiveSynth
import tempfile
import os
import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# Try audio backends
try:
    import pygame
    AUDIO = 'pygame'
except:
    AUDIO = 'system'

def play(samples, rate=44100):
    if AUDIO == 'pygame':
        import pygame
        import numpy as np
        pygame.mixer.pre_init(frequency=rate, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        samples_16bit = (samples * 32767).astype('int16')
        sound = pygame.sndarray.make_sound(samples_16bit)
        sound.play()
        while pygame.mixer.get_busy():
            pass
        pygame.mixer.quit()
    else:
        import wave, struct
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_file = tmp.name
        with wave.open(temp_file, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            for sample in samples:
                int_sample = int(sample * 30000)
                int_sample = max(-32000, min(32000, int_sample))
                wav.writeframes(struct.pack('<h', int_sample))
        os.system(f'afplay "{temp_file}" 2>/dev/null')
        os.unlink(temp_file)

def test_osc(wave, freq=220, dur=2, vol=0.2):
    """Test pure oscillator"""
    synth = SubtractiveSynth()
    samples = synth.oscillator(freq, dur, wave) * vol
    print(f"🎵 {wave} @ {freq}Hz, vol {vol}")
    print(f"🔍 Sample stats: min={samples.min():.3f}, max={samples.max():.3f}, mean={samples.mean():.3f}")
    play(samples)

def plot_waveform(wave, freq=220, dur=0.01, cycles=2):
    """Plot waveform to visualize quality and anti-aliasing"""
    if not HAS_PLOT:
        print("❌ matplotlib not available for plotting")
        return

    synth = SubtractiveSynth()
    samples = synth.oscillator(freq, dur, wave)

    # Calculate time axis
    sample_rate = synth.sample_rate
    t = np.linspace(0, dur, len(samples), False)

    # Find the number of samples for the specified cycles
    samples_per_cycle = int(sample_rate / freq)
    plot_samples = int(min(len(samples), samples_per_cycle * cycles))

    plt.figure(figsize=(12, 6))

    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.plot(t[:plot_samples] * 1000, samples[:plot_samples], 'b-', linewidth=1)
    plt.title(f'{wave.upper()} waveform @ {freq}Hz (PolyBLEP anti-aliased)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Plot frequency spectrum
    plt.subplot(2, 1, 2)
    fft = np.fft.fft(samples[:plot_samples])
    freqs = np.fft.fftfreq(plot_samples, 1/sample_rate)
    magnitude = np.abs(fft)

    # Only plot positive frequencies up to 20kHz
    pos_mask = (freqs > 0) & (freqs < 20000)
    plt.semilogy(freqs[pos_mask], magnitude[pos_mask], 'r-', linewidth=1)
    plt.title(f'Frequency Spectrum - {wave.upper()}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log scale)')
    plt.grid(True, alpha=0.3)

    # Mark harmonics
    for i in range(1, 11):  # Show first 10 harmonics
        harmonic_freq = freq * i
        if harmonic_freq < 20000:
            plt.axvline(x=harmonic_freq, color='g', alpha=0.5, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    print(f"📊 Plotted {wave} waveform: {plot_samples} samples, {cycles} cycles")

def plot_envelope(attack=0.01, decay=0.1, sustain=0.7, release=0.2, curve=0.3, duration=2.0):
    """Plot ADSR envelope to visualize shape and curves"""
    if not HAS_PLOT:
        print("❌ matplotlib not available for plotting")
        return

    synth = SubtractiveSynth()
    num_samples = int(duration * synth.sample_rate)
    envelope = synth.adsr_envelope(num_samples, attack, decay, sustain, release, curve)

    # Time axis in seconds
    t = np.linspace(0, duration, num_samples)

    plt.figure(figsize=(12, 6))
    plt.plot(t, envelope, 'b-', linewidth=2)
    plt.title(f'ADSR Envelope (A:{attack}s D:{decay}s S:{sustain} R:{release}s Curve:{curve})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Mark ADSR phases
    attack_time = attack
    decay_time = attack + decay
    sustain_end = duration - release

    plt.axvline(x=attack_time, color='r', alpha=0.7, linestyle='--', label='Attack→Decay')
    plt.axvline(x=decay_time, color='g', alpha=0.7, linestyle='--', label='Decay→Sustain')
    plt.axvline(x=sustain_end, color='orange', alpha=0.7, linestyle='--', label='Sustain→Release')

    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

    print(f"📊 Plotted ADSR envelope: A={attack}s D={decay}s S={sustain} R={release}s Curve={curve}")

def test_filter(wave, freq, cutoff, res, dur=2, vol=0.2):
    """Test oscillator + filter"""
    synth = SubtractiveSynth()
    samples = synth.oscillator(freq, dur, wave)
    print(f"🔍 Original: min={samples.min():.3f}, max={samples.max():.3f}, mean={samples.mean():.3f}")

    filtered = synth.moog_filter(samples, cutoff, res) * vol
    print(f"🔍 Filtered: min={filtered.min():.3f}, max={filtered.max():.3f}, mean={filtered.mean():.3f}")
    print(f"🎛️ {wave} @ {freq}Hz → filter {cutoff}Hz, res {res}, vol {vol}")
    play(filtered)

def test_note(freq, wave='saw', attack=0.01, decay=0.1, sustain=0.7, release=0.2,
              filter_base=1000, filter_env=2000, res=1.0, dur=2, vol=0.3):
    """Test complete note"""
    synth = SubtractiveSynth()
    samples = synth.create_synth_note(
        freq=freq, duration=dur, waveform=wave,
        filter_base=filter_base, filter_env_amount=filter_env, resonance=res,
        amp_env=(attack, decay, sustain, release)
    ) * vol
    print(f"🎼 {wave} @ {freq}Hz, filter {filter_base}+{filter_env}Hz, ADSR({attack},{decay},{sustain},{release}), vol {vol}")
    play(samples)

def test_supersaw(freq=220, dur=3, detune=12, filter_base=1500, filter_env=2500, res=0.5, vol=0.2):
    """Test supersaw - rich detuned saw ensemble"""
    synth = SubtractiveSynth()

    # Create detune pattern based on single detune amount
    detune_cents = [-detune, -detune*0.6, -detune*0.3, detune*0.3, detune*0.6, detune]

    samples = synth.supersaw(
        freq, dur,
        detune_cents=detune_cents,
        filter_base=filter_base,
        filter_env_amount=filter_env,
        resonance=res
    ) * vol

    print(f"🌊 Supersaw @ {freq}Hz, detune ±{detune} cents, filter {filter_base}+{filter_env}Hz, res {res}, vol {vol}")
    play(samples)

def test_arpeggio(root_freq=110, pattern='maj', wave='saw', tempo=120, vol=0.2, cycles=1):
    """Test arpeggio pattern to hear musical phrase"""
    synth = SubtractiveSynth()

    # Define arpeggio patterns (relative to root)
    patterns = {
        'maj': [1.0, 1.25, 1.5, 2.0],  # Major: root, maj3, 5th, octave
        'min': [1.0, 1.2, 1.5, 2.0],   # Minor: root, min3, 5th, octave
        'dom7': [1.0, 1.25, 1.5, 1.75], # Dom7: root, maj3, 5th, b7
        'dim': [1.0, 1.2, 1.414, 1.682], # Diminished
        'sus4': [1.0, 1.333, 1.5, 2.0],  # Sus4: root, 4th, 5th, octave
        'pent': [1.0, 1.125, 1.333, 1.5, 1.667, 2.0], # Major pentatonic: 1, 2, 3, 5, 6, 8
        'pent_min': [1.0, 1.2, 1.333, 1.5, 1.8, 2.0], # Minor pentatonic: 1, b3, 4, 5, b7, 8
    }

    if pattern not in patterns:
        pattern = 'maj'

    note_freqs = [root_freq * ratio for ratio in patterns[pattern]]
    note_duration = 60.0 / tempo / 2  # 8th notes

    # Create each note with slightly overlapping release
    all_samples = []

    # Create pattern based on cycles
    single_pattern = note_freqs + note_freqs[-2:0:-1]  # Up and down, avoiding repeat at top
    full_pattern = single_pattern * int(cycles)  # Repeat pattern for specified cycles

    for i, freq in enumerate(full_pattern):
        # Vary filter cutoff based on position in arpeggio
        filter_base = 400 + (i * 100)

        samples = synth.create_synth_note(
            freq=freq,
            duration=note_duration,  # No overlap to prevent clipping
            waveform=wave,
            filter_base=filter_base,
            filter_env_amount=2000,
            resonance=1.5,
            amp_env=(0.01, 0.05, 0.7, 0.15)  # Slightly longer release
        )

        # Add small gap between notes for clarity
        gap_samples = int(0.02 * synth.sample_rate)  # 20ms gap
        samples = np.concatenate([samples, np.zeros(gap_samples)])

        all_samples.append(samples)

    # Concatenate all notes
    final = np.concatenate(all_samples)

    # Apply soft clipping to prevent harsh distortion
    final = np.tanh(final * 0.7) * vol  # Soft clip then scale

    print(f"🎹 Arpeggio: {pattern} pattern @ {root_freq}Hz, tempo {tempo}, {cycles} cycles, vol {vol}")
    play(final)

def main():
    if len(sys.argv) < 2:
        print("SIMPLE SYNTH TESTER")
        print()
        print("Usage:")
        print("  osc <wave> <freq> [vol]                    - Pure oscillator")
        print("  plot <wave> <freq> [cycles]                - Plot waveform and spectrum")
        print("  env [attack] [decay] [sustain] [release] [curve] - Plot ADSR envelope")
        print("  filt <wave> <freq> <cutoff> <res> [vol]    - Oscillator + filter")
        print("  note <freq> [wave] [attack] [decay] [sustain] [release] [filter_base] [filter_env] [res] [vol]")
        print("  super [freq] [dur] [detune] [filter_base] [filter_env] [res] [vol] - Supersaw")
        print("  arp [root_freq] [pattern] [wave] [tempo] [vol] [cycles] - Play arpeggio")
        print()
        print("Examples:")
        print("  ./simple_synth_test.py osc saw 110 0.1")
        print("  ./simple_synth_test.py plot saw 110 2")
        print("  ./simple_synth_test.py env 0.01 0.1 0.7 0.2 0.3")
        print("  ./simple_synth_test.py filt saw 110 500 2.0 0.2")
        print("  ./simple_synth_test.py note 220 pulse 0.01 0.2 0.5 0.3 800 3000 1.5 0.3")
        print("  ./simple_synth_test.py super 110 3 15 1000 3000 0.7 0.2")
        print("  ./simple_synth_test.py arp 110 maj saw 120 0.2 2")
        print("  ./simple_synth_test.py arp 220 min pulse 140 0.15 3")
        print()
        print("Arpeggio patterns: maj, min, dom7, dim, sus4, pent, pent_min")
        return

    cmd = sys.argv[1]

    if cmd == 'osc':
        wave = sys.argv[2] if len(sys.argv) > 2 else 'saw'
        freq = float(sys.argv[3]) if len(sys.argv) > 3 else 220
        vol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
        test_osc(wave, freq, vol=vol)

    elif cmd == 'plot':
        wave = sys.argv[2] if len(sys.argv) > 2 else 'saw'
        freq = float(sys.argv[3]) if len(sys.argv) > 3 else 220
        cycles = float(sys.argv[4]) if len(sys.argv) > 4 else 2
        plot_waveform(wave, freq, cycles=cycles)

    elif cmd == 'env':
        attack = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
        decay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        sustain = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
        release = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
        curve = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
        plot_envelope(attack, decay, sustain, release, curve)

    elif cmd == 'filt':
        wave = sys.argv[2] if len(sys.argv) > 2 else 'saw'
        freq = float(sys.argv[3]) if len(sys.argv) > 3 else 220
        cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 1000
        res = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        vol = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
        test_filter(wave, freq, cutoff, res, vol=vol)

    elif cmd == 'note':
        freq = float(sys.argv[2]) if len(sys.argv) > 2 else 220
        wave = sys.argv[3] if len(sys.argv) > 3 else 'saw'
        attack = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
        decay = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
        sustain = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7
        release = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
        filter_base = float(sys.argv[8]) if len(sys.argv) > 8 else 1000
        filter_env = float(sys.argv[9]) if len(sys.argv) > 9 else 2000
        res = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
        vol = float(sys.argv[11]) if len(sys.argv) > 11 else 0.3
        test_note(freq, wave, attack, decay, sustain, release, filter_base, filter_env, res, vol=vol)

    elif cmd == 'super':
        freq = float(sys.argv[2]) if len(sys.argv) > 2 else 220
        dur = float(sys.argv[3]) if len(sys.argv) > 3 else 3
        detune = float(sys.argv[4]) if len(sys.argv) > 4 else 12
        filter_base = float(sys.argv[5]) if len(sys.argv) > 5 else 1500
        filter_env = float(sys.argv[6]) if len(sys.argv) > 6 else 2500
        res = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
        vol = float(sys.argv[8]) if len(sys.argv) > 8 else 0.2
        test_supersaw(freq, dur, detune, filter_base, filter_env, res, vol)

    elif cmd == 'arp':
        root_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 110
        pattern = sys.argv[3] if len(sys.argv) > 3 else 'maj'
        wave = sys.argv[4] if len(sys.argv) > 4 else 'saw'
        tempo = float(sys.argv[5]) if len(sys.argv) > 5 else 120
        vol = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
        cycles = int(sys.argv[7]) if len(sys.argv) > 7 else 1
        test_arpeggio(root_freq, pattern, wave, tempo, vol, cycles)

    else:
        print(f"Unknown command: {cmd}")

if __name__ == '__main__':
    main()