"""
SYNTH_ENGINE - Pure Synthesis Engine
Low-level synthesis without hard-coded musical parameters
"""

import numpy as np


class SubtractiveSynth:
    """
    Pure subtractive synthesis engine.
    No hard-coded musical parameters - all values passed as arguments.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0

        # Filter state variables (for continuity between notes)
        self.filter_z1 = 0.0
        self.filter_z2 = 0.0
        self.filter_z3 = 0.0
        self.filter_z4 = 0.0

        # Phase continuity for click-free retriggering
        self.phase = 0.0
        self.last_env_value = 0.0
        self.last_signal_tail = None  # For crossfading
        self.crossfade_samples = 64  # ~1.45ms at 44.1kHz

    def poly_blep(self, dt, phase):
        """
        PolyBLEP (Polynomial Band-Limited Edge Pulse) anti-aliasing
        dt: normalized frequency (freq / sample_rate)
        phase: phase position (0.0 to 1.0)
        """
        if phase < dt:
            # Beginning of period - rising edge
            t = phase / dt
            return t + t - t * t - 1.0
        elif phase > 1.0 - dt:
            # End of period - falling edge
            t = (phase - 1.0) / dt
            return t * t + t + t + 1.0
        else:
            return 0.0

    def oscillator(self, freq, duration, waveform='saw', apply_envelope=None):
        """
        Generate band-limited waveforms using PolyBLEP anti-aliasing
        with phase-safe retriggering to eliminate clicks.

        apply_envelope: Optional envelope array to track decay state for phase continuity
        """
        num_samples = int(duration * self.sample_rate)
        dt = freq / self.sample_rate  # Normalized frequency

        # Phase-safe retrigger: only reset phase if previous note fully decayed
        if self.last_env_value < 1e-3:
            phase = 0.0  # Safe reset
        else:
            phase = self.phase  # Continue from last position

        # Generate sample by sample for proper PolyBLEP application
        signal = np.zeros(num_samples)

        for i in range(num_samples):
            if waveform == 'saw':
                # Band-limited sawtooth
                signal[i] = 2.0 * phase - 1.0
                signal[i] -= self.poly_blep(dt, phase)

            elif waveform == 'square':
                # Band-limited square wave - 50% duty cycle
                if phase < 0.5:
                    signal[i] = 1.0
                else:
                    signal[i] = -1.0

                # Apply PolyBLEP at rising edge (phase = 0)
                signal[i] += self.poly_blep(dt, phase)
                # Apply PolyBLEP at falling edge (phase = 0.5)
                signal[i] -= self.poly_blep(dt, phase - 0.5 if phase >= 0.5 else phase + 0.5)

            elif waveform == 'pulse':
                # Band-limited pulse wave - 25% duty cycle
                width = 0.25
                if phase < width:
                    signal[i] = 1.0
                else:
                    signal[i] = -1.0

                # Apply PolyBLEP at rising edge (phase = 0)
                signal[i] += self.poly_blep(dt, phase)
                # Apply PolyBLEP at falling edge (phase = width)
                phase_shifted = phase - width if phase >= width else phase + (1.0 - width)
                signal[i] -= self.poly_blep(dt, phase_shifted)

            elif waveform == 'triangle':
                # Triangle wave - no PolyBLEP needed (continuous derivatives)
                if phase < 0.5:
                    signal[i] = 4.0 * phase - 1.0      # Rising: -1 to 1
                else:
                    signal[i] = 3.0 - 4.0 * phase      # Falling: 1 to -1

            else:  # sine
                # Sine wave - naturally band-limited
                signal[i] = np.sin(2 * np.pi * phase)

            # Advance phase
            phase += dt
            if phase >= 1.0:
                phase -= 1.0

        # Save final phase for next retrigger
        self.phase = phase

        # Micro cross-fade on retrigger to eliminate click
        if self.last_signal_tail is not None and self.last_env_value >= 1e-3:
            xfade_len = min(self.crossfade_samples, len(signal), len(self.last_signal_tail))
            if xfade_len > 0:
                x = np.linspace(0, 1, xfade_len)
                signal[:xfade_len] = (
                    self.last_signal_tail[:xfade_len] * (1 - x) +
                    signal[:xfade_len] * x
                )

        # Store tail for next potential crossfade
        if len(signal) >= self.crossfade_samples:
            self.last_signal_tail = signal[-self.crossfade_samples:].copy()

        # Track envelope value if provided
        if apply_envelope is not None:
            self.last_env_value = apply_envelope[-1] if len(apply_envelope) > 0 else 0.0

        return signal

    def moog_filter(self, signal, cutoff_hz, resonance=0.0):
        """
        Stable 4-pole Moog-style low-pass ladder.
        Mild saturation and controlled resonance.
        Works for realtime and offline synthesis.
        """

        if cutoff_hz >= self.nyquist * 0.99:
            return signal

        f = 2.0 * np.sin(np.pi * cutoff_hz / self.sample_rate)  # stable frequency map
        f = np.clip(f, 0.0001, 1.0)
        resonance = np.clip(resonance, 0.0, 4.0)

        # State variables
        y1, y2, y3, y4 = self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4
        out = np.zeros_like(signal)

        for i, x in enumerate(signal):
            # Feedback
            x -= resonance * y4

            # Input soft clipping
            x = np.tanh(x)

            # 4 cascaded one-pole filters
            y1 += f * (x - y1)
            y2 += f * (y1 - y2)
            y3 += f * (y2 - y3)
            y4 += f * (y3 - y4)

            # Output soft clipping for smoother tone
            out[i] = np.tanh(y4)

        # Save states
        self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4 = y1, y2, y3, y4

        # Simple gain compensation
        return out * 1.5


    def supersaw(self, freq, duration,
                 detune_cents=None,
                 filter_base=1500, filter_env_amount=2500,
                 resonance=0.5,
                 amp_env=(0.05, 0.2, 0.9, 0.4),
                 filter_env=(0.01, 0.25, 0.4, 0.4)):
        """
        Roland JP-8000 style supersaw generator.
        Creates a rich detuned saw ensemble with analog-style filtering.
        Returns mono signal for compatibility with existing system.
        """

        # Default detune pattern if not provided
        if detune_cents is None:
            detune_cents = [-12, -7, -3, 3, 7, 12]

        # Base + detuned frequencies
        freqs = [freq * (2 ** (c / 1200.0)) for c in detune_cents] + [freq]

        # Generate saw layers
        layers = []
        for f in freqs:
            layer = self.oscillator(f, duration, 'saw')
            # Randomize phase for natural analog drift
            shift = np.random.randint(0, 100)  # Small phase shift
            layers.append(np.roll(layer, shift))

        # Mix all layers equally
        mixed = np.sum(layers, axis=0) / len(layers)

        # Get filter envelope
        num_samples = len(mixed)
        filt_env = self.filter_envelope(num_samples, *filter_env)

        # Apply time-varying filter
        filtered = np.zeros_like(mixed)
        chunk_size = 512  # Process in chunks for efficiency

        # Save original filter state
        orig_z1, orig_z2, orig_z3, orig_z4 = self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4

        for i in range(0, num_samples, chunk_size):
            end = min(i + chunk_size, num_samples)
            # Calculate cutoff for this chunk
            env_val = np.mean(filt_env[i:end])  # Average envelope value in chunk
            cutoff = filter_base + filter_env_amount * env_val
            cutoff = np.clip(cutoff, 20, self.nyquist * 0.95)

            # Apply filter to chunk (filter state carries through)
            filtered[i:end] = self.moog_filter(mixed[i:end], cutoff, resonance)

        # Restore original filter state for other uses
        self.filter_z1, self.filter_z2, self.filter_z3, self.filter_z4 = orig_z1, orig_z2, orig_z3, orig_z4

        # Apply amplitude envelope
        amp_env_signal = self.adsr_envelope(num_samples, *amp_env)
        result = filtered * amp_env_signal

        # Soft limiting to prevent clipping from multiple layers
        result = np.tanh(result * 0.8) * 1.25  # Gentle compression

        return result

    def adsr_envelope(self, num_samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2, curve=0.3):
        """
        ADSR envelope generator with exponential curves for musical sound
        Times in seconds, sustain is level (0-1)
        curve: exponential curve factor (0.1 = gentle, 1.0 = aggressive)
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

        # Attack - exponential rise (slow start, fast finish)
        if attack_samples > 0:
            t = np.linspace(0, 1, attack_samples)
            # Use exponential curve: starts slow, accelerates
            envelope[current:current+attack_samples] = np.power(t, 1.0 - curve)
            current += attack_samples

        # Decay - exponential fall (fast start, slow finish)
        if decay_samples > 0 and current < num_samples:
            end = min(current + decay_samples, num_samples)
            t = np.linspace(0, 1, end - current)
            # Exponential decay: starts fast, slows down
            decay_curve = 1.0 - np.power(t, curve)
            envelope[current:end] = 1.0 - decay_curve * (1.0 - sustain)
            current = end

        # Sustain - constant level
        if sustain_samples > 0 and current < num_samples:
            end = min(current + sustain_samples, num_samples)
            envelope[current:end] = sustain
            current = end

        # Release - exponential fall to zero
        if current < num_samples:
            t = np.linspace(0, 1, num_samples - current)
            # Exponential release: fast start, slow finish
            release_curve = np.power(t, curve)
            envelope[current:] = sustain * (1.0 - release_curve)

        return envelope

    def filter_envelope(self, num_samples, attack=0.05, decay=0.2, sustain=0.3, release=0.3):
        """
        Separate envelope for filter cutoff modulation
        This creates the classic 'wow' sound of analog synths
        """
        return self.adsr_envelope(num_samples, attack, decay, sustain, release)

    def pitch_sweep_note(self, freq_start, freq_end, duration,
                        waveform='sine',
                        filter_base=2000,
                        resonance=0.3,
                        amp_env=(0.01, 0.05, 0.7, 0.1)):
        """
        Create R2D2-style beep with pitch sweep from freq_start to freq_end
        with phase continuity for click-free retriggering
        """
        num_samples = int(duration * self.sample_rate)

        # Create frequency sweep curve
        freq_curve = np.linspace(freq_start, freq_end, num_samples)

        # Phase-safe retrigger
        if self.last_env_value < 1e-3:
            phase = 0.0
        else:
            phase = self.phase * 2 * np.pi  # Convert from normalized to radians

        # Generate signal with sweeping frequency
        signal = np.zeros(num_samples)

        for i in range(num_samples):
            current_freq = freq_curve[i]

            # Update phase
            phase += 2 * np.pi * current_freq / self.sample_rate

            # Generate sample based on waveform
            if waveform == 'sine':
                signal[i] = np.sin(phase)
            elif waveform == 'triangle':
                # Triangle wave
                if (phase / (2 * np.pi)) % 1.0 < 0.5:
                    signal[i] = 4.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
                else:
                    signal[i] = 3.0 - 4.0 * ((phase / (2 * np.pi)) % 1.0)
            else:  # Default to sine
                signal[i] = np.sin(phase)

        # Save phase (convert back to normalized)
        self.phase = (phase / (2 * np.pi)) % 1.0

        # Micro cross-fade on retrigger
        if self.last_signal_tail is not None and self.last_env_value >= 1e-3:
            xfade_len = min(self.crossfade_samples, len(signal), len(self.last_signal_tail))
            if xfade_len > 0:
                x = np.linspace(0, 1, xfade_len)
                signal[:xfade_len] = (
                    self.last_signal_tail[:xfade_len] * (1 - x) +
                    signal[:xfade_len] * x
                )

        # Apply simple filter (no envelope needed for R2D2 style)
        filtered = self.moog_filter(signal, filter_base, resonance)

        # Apply amplitude envelope
        amp_envelope = self.adsr_envelope(num_samples, *amp_env)
        output = filtered * amp_envelope

        # Store tail and envelope value
        if len(output) >= self.crossfade_samples:
            self.last_signal_tail = output[-self.crossfade_samples:].copy()
        self.last_env_value = amp_envelope[-1] if len(amp_envelope) > 0 else 0.0

        return output

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