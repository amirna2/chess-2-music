#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sounddevice as sd

# Global variables for real-time updates
global_samplerate = 44100
global_carrier_frequency = 440
global_adsr_params = {
    'attack': 0.1,
    'decay': 0.2,
    'sustain': 0.7,
    'release': 0.5,
}
global_curve_type = 'linear'
global_waveform = 'sawtooth'
global_audio_signal = np.array([])

def create_envelope(samplerate, attack, decay, sustain, release, total_duration, curve_type='linear'):
    """Creates an ADSR envelope as a NumPy array."""
    num_samples = int(total_duration * samplerate)
    env = np.zeros(num_samples)

    # Define time points and values
    t_attack = int(attack * samplerate)
    t_decay = int(decay * samplerate)
    t_sustain_start = t_attack + t_decay
    t_release = int(release * samplerate)
    t_sustain_end = num_samples - t_release

    # Ensure times are valid and don't overlap
    t_attack = min(t_attack, num_samples)
    t_decay = min(t_decay, num_samples - t_attack)
    t_release = min(t_release, num_samples - t_sustain_start)
    t_sustain_end = num_samples - t_release
    t_sustain_start = t_attack + t_decay

    # Attack segment
    if curve_type == 'exponential' and t_attack > 0:
        attack_curve = np.exp(np.linspace(0, 5, t_attack)) / np.exp(5)
    elif t_attack > 0:
        attack_curve = np.linspace(0, 1, t_attack)
    else:
        attack_curve = np.array([])
    env[:t_attack] = attack_curve

    # Decay segment
    if curve_type == 'exponential' and t_decay > 0:
        decay_curve = sustain + (1 - sustain) * np.exp(np.linspace(5, 0, t_decay)) / np.exp(5)
    elif t_decay > 0:
        decay_curve = np.linspace(1, sustain, t_decay)
    else:
        decay_curve = np.array([])
    env[t_attack:t_sustain_start] = decay_curve

    # Sustain segment
    env[t_sustain_start:t_sustain_end] = sustain

    # Release segment
    if curve_type == 'exponential' and t_release > 0:
        release_curve = sustain * np.exp(np.linspace(5, 0, t_release)) / np.exp(5)
    elif t_release > 0:
        release_curve = np.linspace(sustain, 0, t_release)
    else:
        release_curve = np.array([])
    env[t_sustain_end:num_samples] = release_curve

    return env

def generate_waveform(waveform_type, frequency, duration, samplerate):
    """Generate different waveform types."""
    t = np.linspace(0., duration, int(samplerate * duration))

    if waveform_type == 'sine':
        return np.sin(2. * np.pi * frequency * t)
    elif waveform_type == 'sawtooth':
        return 2 * (t * frequency - np.floor(0.5 + t * frequency))
    elif waveform_type == 'square':
        return np.sign(np.sin(2. * np.pi * frequency * t))
    elif waveform_type == 'triangle':
        return 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
    elif waveform_type == 'pulse':
        # Pulse wave with 25% duty cycle
        phase = (t * frequency) % 1.0
        return np.where(phase < 0.25, 1.0, -1.0)
    else:
        return np.sin(2. * np.pi * frequency * t)

def generate_and_play(adsr_params, samplerate, curve_type, frequency, waveform, duration=2):
    """Generates an ADSR envelope, applies it to a waveform, and plays the sound."""
    global global_audio_signal

    # Pass the entire dictionary correctly
    envelope = create_envelope(samplerate, **adsr_params, total_duration=duration, curve_type=curve_type)

    t = np.linspace(0., duration, int(samplerate * duration))
    carrier_wave = generate_waveform(waveform, frequency, duration, samplerate)

    # Apply the envelope to the carrier wave
    audio_signal = carrier_wave * envelope

    # Normalize to prevent clipping
    if np.max(np.abs(audio_signal)) > 0:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))

    global_audio_signal = audio_signal

    # Display the envelope and the final waveform
    ax_env.clear()
    ax_env.plot(t, envelope, label=f'ADSR Envelope ({curve_type})')
    ax_env.set_title('ADSR Envelope')
    ax_env.set_xlabel('Time (s)')
    ax_env.set_ylim(-0.1, 1.1)
    ax_env.legend()
    ax_env.grid(True)

    ax_wave.clear()
    ax_wave.plot(t, audio_signal, label=f'Final Waveform ({waveform})', color='orange')
    ax_wave.set_title('Audio Waveform')
    ax_wave.set_xlabel('Time (s)')
    ax_wave.set_ylim(-1.1, 1.1)
    ax_wave.legend()
    ax_wave.grid(True)

    fig.canvas.draw_idle()

def play_sound():
    """Plays the globally stored audio signal."""
    global global_audio_signal
    if global_audio_signal.size > 0:
        sd.play(global_audio_signal, global_samplerate)

# Set up the plot
fig, (ax_env, ax_wave) = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.95)

# Initial generation and plot
generate_and_play(global_adsr_params, global_samplerate, global_curve_type, global_carrier_frequency, global_waveform)

# Create sliders for ADSR parameters
slider_attack = Slider(ax=fig.add_axes([0.15, 0.16, 0.75, 0.02]), label='Attack', valmin=0.0, valmax=1.0, valinit=global_adsr_params['attack'])
slider_decay = Slider(ax=fig.add_axes([0.15, 0.13, 0.75, 0.02]), label='Decay', valmin=0.0, valmax=1.0, valinit=global_adsr_params['decay'])
slider_sustain = Slider(ax=fig.add_axes([0.15, 0.10, 0.75, 0.02]), label='Sustain', valmin=0.0, valmax=1.0, valinit=global_adsr_params['sustain'])
slider_release = Slider(ax=fig.add_axes([0.15, 0.07, 0.75, 0.02]), label='Release', valmin=0.0, valmax=1.0, valinit=global_adsr_params['release'])

def update(val):
    global_adsr_params['attack'] = slider_attack.val
    global_adsr_params['decay'] = slider_decay.val
    global_adsr_params['sustain'] = slider_sustain.val
    global_adsr_params['release'] = slider_release.val
    generate_and_play(global_adsr_params, global_samplerate, global_curve_type, global_carrier_frequency, global_waveform)

slider_attack.on_changed(update)
slider_decay.on_changed(update)
slider_sustain.on_changed(update)
slider_release.on_changed(update)

# Create buttons for control - row 1 (waveforms)
btn_sine = Button(ax=fig.add_axes([0.10, 0.03, 0.09, 0.025]), label='Sine')
btn_saw = Button(ax=fig.add_axes([0.20, 0.03, 0.09, 0.025]), label='Saw')
btn_square = Button(ax=fig.add_axes([0.30, 0.03, 0.09, 0.025]), label='Square')
btn_tri = Button(ax=fig.add_axes([0.40, 0.03, 0.09, 0.025]), label='Tri')
btn_pulse = Button(ax=fig.add_axes([0.50, 0.03, 0.09, 0.025]), label='Pulse')

# Row 2 (curve type and play)
btn_curve_linear = Button(ax=fig.add_axes([0.10, 0.005, 0.12, 0.025]), label='Linear')
btn_curve_exponential = Button(ax=fig.add_axes([0.23, 0.005, 0.12, 0.025]), label='Exponential')
btn_play = Button(ax=fig.add_axes([0.80, 0.005, 0.1, 0.05]), label='▶ Play')

def on_play_clicked(_):
    play_sound()

def on_curve_linear_clicked(_):
    global global_curve_type
    global_curve_type = 'linear'
    generate_and_play(global_adsr_params, global_samplerate, global_curve_type, global_carrier_frequency, global_waveform)

def on_curve_exponential_clicked(_):
    global global_curve_type
    global_curve_type = 'exponential'
    generate_and_play(global_adsr_params, global_samplerate, global_curve_type, global_carrier_frequency, global_waveform)

def on_waveform_clicked(waveform):
    def handler(_):
        global global_waveform
        global_waveform = waveform
        generate_and_play(global_adsr_params, global_samplerate, global_curve_type, global_carrier_frequency, global_waveform)
    return handler

btn_play.on_clicked(on_play_clicked)
btn_curve_linear.on_clicked(on_curve_linear_clicked)
btn_curve_exponential.on_clicked(on_curve_exponential_clicked)
btn_sine.on_clicked(on_waveform_clicked('sine'))
btn_saw.on_clicked(on_waveform_clicked('sawtooth'))
btn_square.on_clicked(on_waveform_clicked('square'))
btn_tri.on_clicked(on_waveform_clicked('triangle'))
btn_pulse.on_clicked(on_waveform_clicked('pulse'))

plt.show()
