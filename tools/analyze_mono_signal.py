#!/usr/bin/env python3
"""
Analyze mono signal for discontinuities and visualize them.
"""

import sys
sys.path.insert(0, '/Users/nathoo/dev/chess-2-music')

from synth_engine import SubtractiveSynth
from synth_composer.core.audio_buffer import AudioBuffer
from synth_composer.core.synthesizer import NoteSynthesizer
from synth_composer.patterns.theory import PositionalTheoryPattern
import numpy as np
from synth_config import DEFAULT_CONFIG
import matplotlib.pyplot as plt

# Generate the EXACT pattern from game1
print("Generating POSITIONAL_THEORY pattern...")
synth = SubtractiveSynth(sample_rate=88200, rng=np.random.default_rng(seed=11))
synthesizer = NoteSynthesizer(synth, 88200)

scale = [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0]
pattern_gen = PositionalTheoryPattern(rng=np.random.default_rng(seed=11))

params = {
    'sample_rate': 88200,
    'note_duration': 0.5,
    'tension': 0.61,
    'filter': 1980,
    'filter_env': 0,
    'resonance': 1.32,
    'config': DEFAULT_CONFIG
}

events = pattern_gen.generate_events(duration=28.0, scale=scale, params=params)
buffer = AudioBuffer(int(28.0 * 88200))

for event in events:
    audio = synthesizer.synthesize(event)
    start_sample = int(event.timestamp * 88200)
    gain = event.extra_context['mix_level'] if event.extra_context and 'mix_level' in event.extra_context else 0.1
    buffer.add_audio(audio, start_sample, gain)

mono = buffer.get_buffer()

# Normalize like the composer does
peak = np.max(np.abs(mono))
if peak > 0:
    target_peak = 0.707  # -3dBFS
    gain_db = 20 * np.log10(target_peak / peak)
    gain = target_peak / peak
    mono_normalized = mono * gain
else:
    mono_normalized = mono
    gain_db = 0

print(f"Peak before normalization: {peak:.6f}")
print(f"Normalization gain: {gain_db:.1f} dB")
print(f"Peak after normalization: {np.max(np.abs(mono_normalized)):.6f}")

# Detect discontinuities
diff = np.abs(np.diff(mono_normalized))
threshold = 0.01
clicks = np.where(diff > threshold)[0]

print(f"\nDiscontinuities > {threshold}: {len(clicks)}")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Plot 1: Full waveform
time = np.arange(len(mono_normalized)) / 88200
axes[0].plot(time, mono_normalized, 'b-', linewidth=0.5)
axes[0].set_title('Full Normalized Mono Waveform')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Click threshold: {threshold}')
axes[0].axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
axes[0].legend()

# Mark clicks
if len(clicks) > 0:
    click_times = clicks / 88200
    axes[0].scatter(click_times, mono_normalized[clicks], c='red', s=10, alpha=0.5, zorder=5)

# Plot 2: First derivative (rate of change)
axes[1].plot(time[:-1], diff, 'g-', linewidth=0.5)
axes[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
axes[1].set_title('First Derivative (Rate of Change)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('|Δ Amplitude|')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Mark clicks
if len(clicks) > 0:
    axes[1].scatter(click_times, diff[clicks], c='red', s=20, zorder=5)

# Plot 3: Zoom into first click
if len(clicks) > 0:
    first_click = clicks[0]
    window = 1000  # samples around click
    start = max(0, first_click - window)
    end = min(len(mono_normalized), first_click + window)

    zoom_time = np.arange(start, end) / 88200
    axes[2].plot(zoom_time, mono_normalized[start:end], 'b-', linewidth=1)
    axes[2].axvline(x=first_click/88200, color='r', linestyle='--', label='Click')
    axes[2].set_title(f'Zoom: First Click at {first_click/88200:.3f}s (sample {first_click})')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Print details
    print(f"\nFirst click details:")
    print(f"  Position: {first_click/88200:.3f}s (sample {first_click})")
    print(f"  Jump size: {diff[first_click]:.6f}")
    print(f"  Before: {mono_normalized[first_click]:.6f}")
    print(f"  After: {mono_normalized[first_click+1]:.6f}")

# Plot 4: Click histogram
if len(clicks) > 0:
    axes[3].hist(diff[clicks], bins=50, color='red', alpha=0.7)
    axes[3].set_title('Click Magnitude Distribution')
    axes[3].set_xlabel('Jump Size')
    axes[3].set_ylabel('Count')
    axes[3].set_yscale('log')
    axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/mono_discontinuities.png', dpi=150)
print(f"\nVisualization saved to: /tmp/mono_discontinuities.png")
# plt.show()  # Don't block

# Analyze a single note in isolation
print("\n" + "="*60)
print("ANALYZING SINGLE NOTE IN ISOLATION")
print("="*60)

# Generate one note
single_note = synth.create_synth_note(
    freq=220.0,
    duration=0.6,
    waveform='triangle',
    filter_base=1980,
    filter_env_amount=0,
    resonance=1.32,
    amp_env=(0.02, 0.06, 0.6, 0.1),
    filter_env=(0.01, 0.15, 0.3, 0.2)
)

# Normalize it the same way
peak_single = np.max(np.abs(single_note))
if peak_single > 0:
    single_normalized = single_note * (0.707 / peak_single)
else:
    single_normalized = single_note

diff_single = np.abs(np.diff(single_normalized))
clicks_single = np.where(diff_single > threshold)[0]

print(f"Single note discontinuities > {threshold}: {len(clicks_single)}")
if len(clicks_single) > 0:
    print(f"Click positions: {clicks_single}")
    print(f"Click times: {clicks_single / 88200}")

    # Check if they align with envelope phases
    attack_samples = int(0.02 * 88200)
    decay_end = int((0.02 + 0.06) * 88200)
    sustain_end = int((0.6 - 0.1) * 88200)

    print(f"\nEnvelope phase boundaries:")
    print(f"  Attack end: {attack_samples} ({attack_samples/88200:.3f}s)")
    print(f"  Decay end: {decay_end} ({decay_end/88200:.3f}s)")
    print(f"  Sustain end: {sustain_end} ({sustain_end/88200:.3f}s)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
