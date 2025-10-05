#!/usr/bin/env python3
"""
Heartbeat Sound Designer - Interactive tool to design the perfect heartbeat
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sounddevice as sd

# Global parameters
sample_rate = 44100
base_freq = 55  # Low bass note (A1)

# ADSR parameters
adsr_params = {
    'attack': 0.003,
    'decay': 0.06,
    'sustain': 0.15,
    'release': 0.20,
}

# Heartbeat pattern parameters
heartbeat_params = {
    'lub_pitch': 0,      # Semitones from base
    'dub_pitch': -2,     # Semitones from base (lower)
    'gap_ms': 80,        # Milliseconds between LUB and dub
    'pause_ms': 400,     # Milliseconds before next LUB
    'filter_cutoff': 120, # Hz
    'bpm': 70,           # Beats per minute
}

def midi_to_freq(midi_note):
    """Convert MIDI note to frequency."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def semitone_to_freq(base_freq, semitones):
    """Convert semitones offset to frequency."""
    return base_freq * (2.0 ** (semitones / 12.0))

def create_adsr_envelope(duration_sec, attack, decay, sustain, release):
    """Create ADSR envelope."""
    num_samples = int(duration_sec * sample_rate)
    envelope = np.zeros(num_samples)

    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    # Attack
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay
    decay_end = attack_samples + decay_samples
    if decay_samples > 0 and decay_end <= num_samples:
        envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)

    # Sustain
    sustain_start = decay_end
    sustain_end = num_samples - release_samples
    if sustain_end > sustain_start:
        envelope[sustain_start:sustain_end] = sustain

    # Release
    if release_samples > 0 and sustain_end < num_samples:
        envelope[sustain_end:] = np.linspace(sustain, 0, min(release_samples, num_samples - sustain_end))

    return envelope

def apply_lowpass_filter(audio, cutoff_freq, sample_rate):
    """Butterworth lowpass filter - no ringing artifacts."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    # 2nd order Butterworth for minimal ringing
    sos = butter(2, normalized_cutoff, btype='low', output='sos')

    # Zero-phase filtering (no phase distortion, no pre-ringing)
    return sosfiltfilt(sos, audio)

def generate_heartbeat():
    """Generate one complete heartbeat cycle (LUB-dub-pause)."""
    global heartbeat_params, adsr_params

    # Calculate timing
    bpm = heartbeat_params['bpm']
    cycle_duration = 60.0 / bpm  # Total cycle time in seconds

    lub_duration = adsr_params['attack'] + adsr_params['decay'] + adsr_params['release']
    dub_duration = lub_duration * 0.8  # Slightly shorter

    gap_sec = heartbeat_params['gap_ms'] / 1000.0

    # Generate LUB (first beat)
    lub_freq = semitone_to_freq(base_freq, heartbeat_params['lub_pitch'])
    t_lub = np.linspace(0, lub_duration, int(lub_duration * sample_rate))
    lub_wave = np.sin(2 * np.pi * lub_freq * t_lub)
    lub_env = create_adsr_envelope(lub_duration, **adsr_params)
    lub_audio = lub_wave * lub_env

    # Apply filter
    lub_audio = apply_lowpass_filter(lub_audio, heartbeat_params['filter_cutoff'], sample_rate)

    # Gap
    gap_samples = int(gap_sec * sample_rate)
    gap_audio = np.zeros(gap_samples)

    # Generate dub (second beat - quieter, lower)
    dub_freq = semitone_to_freq(base_freq, heartbeat_params['dub_pitch'])
    t_dub = np.linspace(0, dub_duration, int(dub_duration * sample_rate))
    dub_wave = np.sin(2 * np.pi * dub_freq * t_dub)
    dub_env = create_adsr_envelope(dub_duration, **adsr_params) * 0.7  # Quieter
    dub_audio = dub_wave * dub_env

    # Apply filter
    dub_audio = apply_lowpass_filter(dub_audio, heartbeat_params['filter_cutoff'], sample_rate)

    # Pause before next cycle
    pause_sec = cycle_duration - lub_duration - gap_sec - dub_duration
    if pause_sec < 0:
        pause_sec = 0.1
    pause_samples = int(pause_sec * sample_rate)
    pause_audio = np.zeros(pause_samples)

    # Combine
    heartbeat = np.concatenate([lub_audio, gap_audio, dub_audio, pause_audio])

    return heartbeat

def generate_heartbeat_sequence(num_cycles=3):
    """Generate multiple heartbeat cycles."""
    cycles = [generate_heartbeat() for _ in range(num_cycles)]
    return np.concatenate(cycles)

def update_display():
    """Update the waveform display."""
    global heartbeat_audio

    heartbeat_audio = generate_heartbeat_sequence(3)

    # Normalize to full scale for better speaker volume
    if np.max(np.abs(heartbeat_audio)) > 0:
        heartbeat_audio = heartbeat_audio / np.max(np.abs(heartbeat_audio)) * 0.95

    # Update plot
    ax.clear()
    t = np.linspace(0, len(heartbeat_audio) / sample_rate, len(heartbeat_audio))
    ax.plot(t, heartbeat_audio, color='darkred', linewidth=0.5)
    ax.set_title(f'Heartbeat Pattern ({heartbeat_params["bpm"]} BPM) - Filter: {heartbeat_params["filter_cutoff"]}Hz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    fig.canvas.draw_idle()

def play_heartbeat():
    """Play the current heartbeat."""
    global heartbeat_audio
    sd.play(heartbeat_audio, sample_rate)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)

# Initial heartbeat
heartbeat_audio = generate_heartbeat_sequence(3)
update_display()

# Create sliders
slider_attack = Slider(ax=fig.add_axes([0.15, 0.25, 0.7, 0.02]), label='Attack (s)', valmin=0.001, valmax=0.05, valinit=adsr_params['attack'])
slider_decay = Slider(ax=fig.add_axes([0.15, 0.22, 0.7, 0.02]), label='Decay (s)', valmin=0.01, valmax=0.2, valinit=adsr_params['decay'])
slider_sustain = Slider(ax=fig.add_axes([0.15, 0.19, 0.7, 0.02]), label='Sustain', valmin=0.0, valmax=1.0, valinit=adsr_params['sustain'])
slider_release = Slider(ax=fig.add_axes([0.15, 0.16, 0.7, 0.02]), label='Release (s)', valmin=0.01, valmax=0.5, valinit=adsr_params['release'])

slider_filter = Slider(ax=fig.add_axes([0.15, 0.12, 0.7, 0.02]), label='Filter (Hz)', valmin=30, valmax=300, valinit=heartbeat_params['filter_cutoff'])
slider_bpm = Slider(ax=fig.add_axes([0.15, 0.09, 0.7, 0.02]), label='BPM', valmin=40, valmax=120, valinit=heartbeat_params['bpm'])
slider_gap = Slider(ax=fig.add_axes([0.15, 0.06, 0.7, 0.02]), label='LUB-dub gap (ms)', valmin=20, valmax=200, valinit=heartbeat_params['gap_ms'])
slider_dub_pitch = Slider(ax=fig.add_axes([0.15, 0.03, 0.7, 0.02]), label='dub pitch (semitones)', valmin=-12, valmax=0, valinit=heartbeat_params['dub_pitch'])

def update_params(val):
    adsr_params['attack'] = slider_attack.val
    adsr_params['decay'] = slider_decay.val
    adsr_params['sustain'] = slider_sustain.val
    adsr_params['release'] = slider_release.val
    heartbeat_params['filter_cutoff'] = int(slider_filter.val)
    heartbeat_params['bpm'] = int(slider_bpm.val)
    heartbeat_params['gap_ms'] = int(slider_gap.val)
    heartbeat_params['dub_pitch'] = int(slider_dub_pitch.val)
    update_display()

# Connect sliders
slider_attack.on_changed(update_params)
slider_decay.on_changed(update_params)
slider_sustain.on_changed(update_params)
slider_release.on_changed(update_params)
slider_filter.on_changed(update_params)
slider_bpm.on_changed(update_params)
slider_gap.on_changed(update_params)
slider_dub_pitch.on_changed(update_params)

# Play button
btn_play = Button(ax=fig.add_axes([0.88, 0.03, 0.1, 0.04]), label='▶ Play')
btn_play.on_clicked(lambda event: play_heartbeat())

# Print button
btn_print = Button(ax=fig.add_axes([0.88, 0.08, 0.1, 0.04]), label='Print Values')
def print_values(event):
    print("\n=== Current Heartbeat Parameters ===")
    print(f"ADSR: ({adsr_params['attack']:.3f}, {adsr_params['decay']:.3f}, {adsr_params['sustain']:.2f}, {adsr_params['release']:.3f})")
    print(f"Filter: {heartbeat_params['filter_cutoff']}Hz")
    print(f"BPM: {heartbeat_params['bpm']}")
    print(f"LUB-dub gap: {heartbeat_params['gap_ms']}ms")
    print(f"dub pitch: {heartbeat_params['dub_pitch']} semitones")
btn_print.on_clicked(print_values)

plt.show()
