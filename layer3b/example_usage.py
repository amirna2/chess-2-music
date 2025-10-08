#!/usr/bin/env python3
"""
Example usage of GestureSynthesizer with manually-created parameter curves.

This demonstrates how the synthesizer receives curves from the curve generation
system and produces gesture audio.
"""

import numpy as np
from synth_engine import SubtractiveSynth
from layer3b.synthesizer import GestureSynthesizer


def create_blunder_gesture():
    """
    Create a BLUNDER gesture: high → low glissando with cluster resolving to muddy interval.
    Emulates what curve_generators.py would produce for a BLUNDER archetype.
    """
    print("Creating BLUNDER gesture...")

    # Initialize synthesis engine
    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    gesture_synth = GestureSynthesizer(synth_engine)

    # Gesture duration: 2.5 seconds (typical BLUNDER)
    duration = 2.5
    num_samples = int(duration * sample_rate)

    # Phase boundaries (normalized)
    pre_shadow = 0.15  # 15% suspended high
    impact = 0.05      # 5% onset
    bloom = 0.25       # 25% gliss down
    decay = 0.35       # 35% fall continues
    # residue = 0.20   # 20% tail

    pre_shadow_end = int(pre_shadow * num_samples)
    impact_end = int((pre_shadow + impact) * num_samples)
    bloom_end = int((pre_shadow + impact + bloom) * num_samples)
    decay_end = int((pre_shadow + impact + bloom + decay) * num_samples)

    # PITCH CURVE: Exponential glissando from high to low
    peak_freq = 880  # A5
    octave_drop = 2.5
    target_freq = peak_freq / (2 ** octave_drop)

    pitch_curve = np.zeros(num_samples)
    pitch_curve[:pre_shadow_end] = peak_freq  # Suspended high

    # Exponential gliss during bloom + decay
    gliss_samples = decay_end - pre_shadow_end
    pitch_curve[pre_shadow_end:decay_end] = np.exp(
        np.linspace(np.log(peak_freq), np.log(target_freq), gliss_samples)
    )

    pitch_curve[decay_end:] = target_freq  # Hold at target

    # HARMONY: 4 voices in tight cluster (1 semitone spacing)
    num_voices = 4
    pitch_voices = []

    for i in range(num_voices):
        voice_curve = pitch_curve.copy()

        # Cluster detuning (±semitones around center)
        cluster_detune = (i - num_voices // 2) * 1.0  # 1 semitone spacing

        # Before decay: tight cluster
        voice_curve[:decay_end] *= (2 ** (cluster_detune / 12))

        # After decay: resolve to muddy interval (tritone = 6 semitones)
        # Keep only 2 voices
        if i < 2:
            resolved_detune = 6 if i == 1 else 0  # Tritone
            voice_curve[decay_end:] *= (2 ** (resolved_detune / 12))
            pitch_voices.append(voice_curve)

    # FILTER: Bandpass → lowpass choke
    bp_center = 1200  # 1.2 kHz bandpass center
    lp_cutoff = 150   # Choke to 150 Hz

    cutoff_curve = np.zeros(num_samples)
    cutoff_curve[:bloom_end] = bp_center

    # Morph to lowpass choke during decay
    morph_samples = num_samples - bloom_end
    cutoff_curve[bloom_end:] = np.exp(
        np.linspace(np.log(bp_center), np.log(lp_cutoff), morph_samples)
    )

    # High resonance during choke
    resonance_curve = np.zeros(num_samples)
    resonance_curve[:bloom_end] = 0.4  # Moderate BP resonance
    resonance_curve[bloom_end:] = np.linspace(0.4, 0.9, morph_samples)  # High Q

    filter_curve = {
        'cutoff': cutoff_curve,
        'resonance': resonance_curve,
        'type': 'bandpass->lowpass'
    }

    # ENVELOPE: Sudden attack with short tail
    envelope = np.zeros(num_samples)

    # Attack: 2ms (sudden)
    attack_samples = int(0.002 * sample_rate)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Sustain: 15%
    sustain_samples = int(0.15 * num_samples)
    envelope[attack_samples:attack_samples + sustain_samples] = 1.0

    # Decay: exponential
    decay_samples = num_samples - attack_samples - sustain_samples
    envelope[attack_samples + sustain_samples:] = np.exp(
        np.linspace(0, -4, decay_samples)
    )

    # TEXTURE: 40% pink noise (chaotic)
    texture_curve = {
        'noise_ratio': 0.4,
        'noise_type': 'pink',
        'shimmer_enable': False
    }

    # SYNTHESIZE
    audio = gesture_synth.synthesize(
        pitch_voices=pitch_voices,
        filter_curve=filter_curve,
        envelope=envelope,
        texture_curve=texture_curve,
        sample_rate=sample_rate
    )

    # Stats
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  Duration: {len(audio) / sample_rate:.2f}s")
    print(f"  Voices: {len(pitch_voices)}")
    print(f"  Peak: {peak:.4f}")
    print(f"  RMS: {rms:.4f} ({20 * np.log10(rms + 1e-10):.1f} dBFS)")
    print(f"  Dynamic range: {20 * np.log10(peak / (rms + 1e-10)):.1f} dB")

    return audio, sample_rate


def create_brilliant_gesture():
    """
    Create a BRILLIANT gesture: ascending spread with chord blooming.
    """
    print("\nCreating BRILLIANT gesture...")

    sample_rate = 88200
    rng = np.random.default_rng(seed=42)
    synth_engine = SubtractiveSynth(sample_rate=sample_rate, rng=rng)
    gesture_synth = GestureSynthesizer(synth_engine)

    # Duration: 3 seconds (longer, more expansive)
    duration = 3.0
    num_samples = int(duration * sample_rate)

    # Phase boundaries
    pre_shadow = 0.20
    impact = 0.10
    bloom = 0.40  # Extended bloom
    decay = 0.20
    # residue = 0.10

    impact_start = int(pre_shadow * num_samples)
    bloom_end = int((pre_shadow + impact + bloom) * num_samples)

    # PITCH: Ascending spread (low → high)
    start_freq = 220  # A3
    end_freq = start_freq * (2 ** 2)  # Rise 2 octaves to A5

    pitch_curve = np.zeros(num_samples)
    pitch_curve[:impact_start] = start_freq

    # Exponential rise during impact + bloom
    spread_samples = bloom_end - impact_start
    pitch_curve[impact_start:bloom_end] = np.exp(
        np.linspace(np.log(start_freq), np.log(end_freq), spread_samples)
    )

    pitch_curve[bloom_end:] = end_freq

    # HARMONY: Unison → major seventh chord
    # Intervals: 0, 4, 7, 11 semitones
    chord_intervals = [0, 4, 7, 11]
    pitch_voices = []

    for interval in chord_intervals:
        voice_curve = pitch_curve.copy()

        # Unison before bloom
        voice_curve[:bloom_end] = pitch_curve[:bloom_end]

        # Spread to chord after bloom
        voice_curve[bloom_end:] *= (2 ** (interval / 12))
        pitch_voices.append(voice_curve)

    # FILTER: Lowpass → highpass opening
    lp_start = 300
    hp_end = 4000

    cutoff_curve = np.zeros(num_samples)
    cutoff_curve[:bloom_end] = lp_start

    open_samples = num_samples - bloom_end
    cutoff_curve[bloom_end:] = np.exp(
        np.linspace(np.log(lp_start), np.log(hp_end), open_samples)
    )

    resonance_curve = np.full(num_samples, 0.3)

    filter_curve = {
        'cutoff': cutoff_curve,
        'resonance': resonance_curve,
        'type': 'lowpass->highpass'
    }

    # ENVELOPE: Gradual sustained
    envelope = np.zeros(num_samples)

    # Gradual attack (50ms, s-curve)
    attack_samples = int(0.05 * sample_rate)
    t = np.linspace(0, 1, attack_samples)
    envelope[:attack_samples] = 0.5 * (1 + np.tanh(4 * (t - 0.5)))

    # Sustain: 50%
    sustain_samples = int(0.5 * num_samples)
    envelope[attack_samples:attack_samples + sustain_samples] = 1.0

    # Linear decay
    decay_samples = num_samples - attack_samples - sustain_samples
    envelope[attack_samples + sustain_samples:] = np.linspace(1, 0, decay_samples)

    # TEXTURE: 15% white noise with 6 Hz shimmer
    texture_curve = {
        'noise_ratio': 0.15,
        'noise_type': 'white',
        'shimmer_enable': True,
        'shimmer_rate_hz': 6.0
    }

    # SYNTHESIZE
    audio = gesture_synth.synthesize(
        pitch_voices=pitch_voices,
        filter_curve=filter_curve,
        envelope=envelope,
        texture_curve=texture_curve,
        sample_rate=sample_rate
    )

    # Stats
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  Duration: {len(audio) / sample_rate:.2f}s")
    print(f"  Voices: {len(pitch_voices)}")
    print(f"  Peak: {peak:.4f}")
    print(f"  RMS: {rms:.4f} ({20 * np.log10(rms + 1e-10):.1f} dBFS)")
    print(f"  Dynamic range: {20 * np.log10(peak / (rms + 1e-10)):.1f} dB")

    return audio, sample_rate


def save_gesture_audio(audio, sample_rate, filename):
    """Save gesture audio to WAV file."""
    from scipy.io import wavfile

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0.8:
        audio = audio * (0.8 / peak)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    wavfile.write(filename, sample_rate, audio_int16)
    print(f"\nSaved: {filename}")


if __name__ == '__main__':
    print("=" * 60)
    print("GestureSynthesizer Usage Examples")
    print("=" * 60)

    # Create gestures
    blunder_audio, sr = create_blunder_gesture()
    brilliant_audio, sr = create_brilliant_gesture()

    # Optionally save to files (requires scipy)
    try:
        save_gesture_audio(blunder_audio, sr, '/tmp/gesture_blunder.wav')
        save_gesture_audio(brilliant_audio, sr, '/tmp/gesture_brilliant.wav')
        print("\nYou can audition these files:")
        print("  afplay /tmp/gesture_blunder.wav")
        print("  afplay /tmp/gesture_brilliant.wav")
    except ImportError:
        print("\nInstall scipy to save WAV files: pip install scipy")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
