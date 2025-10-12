#!/usr/bin/env python3
"""
Visualize pitch curve to verify discrete chimes behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from layer3b import GestureCoordinator, ARCHETYPES
from synth_engine import SubtractiveSynth

# Initialize
sample_rate = 88200
rng = np.random.default_rng(42)
synth = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
coordinator = GestureCoordinator(rng, synth_engine=synth)

section_context = {
    'tension': 0.7,
    'entropy': 0.5,
    'scale': 'C_MAJOR',
    'key': 'C'
}

test_event = {
    'type': 'INACCURACY',
    'timestamp': 0.0,
    'move_number': 1,
    'score': 5
}

# Generate gesture and get curves
config = ARCHETYPES['INACCURACY']
duration_base = config['duration_base']
tension = section_context['tension']
entropy = section_context['entropy']

# Calculate actual duration
duration_tension_scale = config.get('duration_tension_scale', 0.0)
duration_entropy_scale = config.get('duration_entropy_scale', 0.0)
duration = duration_base + duration_tension_scale * tension + duration_entropy_scale * entropy

total_samples = int(duration * sample_rate)

# Get phases
from layer3b.core import calculate_phases
phases = calculate_phases(config['phases'], total_samples)

# Generate pitch curve
from layer3b.curve_generators import generate_pitch_voices
pitch_voices = generate_pitch_voices(config['pitch'], phases, section_context, total_samples, rng, sample_rate)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

time_axis = np.arange(total_samples) / sample_rate

# Plot first voice
ax1.plot(time_axis, pitch_voices[0], linewidth=1, color='blue', label='Voice 1 (fundamental)')
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('INACCURACY Pitch Curve - Discrete Chimes (Voice 1)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 1200])

# Mark phase boundaries
phase_names = ['pre_shadow', 'impact', 'bloom', 'decay', 'residue']
colors = ['gray', 'red', 'green', 'blue', 'purple']
for i, (name, color) in enumerate(zip(phase_names, colors)):
    start_sample = phases[name]['start_sample']
    start_time = start_sample / sample_rate
    ax1.axvline(start_time, color=color, linestyle='--', alpha=0.5, label=f'{name} start')
    ax2.axvline(start_time, color=color, linestyle='--', alpha=0.5)

# Plot second voice (harmony)
if len(pitch_voices) > 1:
    ax2.plot(time_axis, pitch_voices[1], linewidth=1, color='orange', label='Voice 2 (harmony)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('INACCURACY Pitch Curve - Voice 2 (Perfect Fifth)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1800])

plt.tight_layout()
plt.savefig('inaccuracy_pitch_curves.png', dpi=150)
print("Saved visualization to: inaccuracy_pitch_curves.png")
print("\nExpected behavior:")
print("  - THREE discrete note events (not continuous)")
print("  - Each note has constant pitch (horizontal line)")
print("  - Gaps between notes (frequency drops to 0)")
print("  - Parabolic pitch variation across the 3 notes")
