"""
SYNTH_ENGINE - Pure Synthesis Engine
Low-level synthesis without hard-coded musical parameters
"""

import numpy as np
from scipy.signal import get_window, butter, sosfilt


class SubtractiveSynth:
    """
    Pure subtractive synthesis engine.
    No hard-coded musical parameters - all values passed as arguments.
    """

    def __init__(self, sample_rate=44100, rng=None):
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
        self.crossfade_samples = 128  # ~1.45ms at 88.2kHz (was 64 @ 44.1kHz)

        # Triangle wave integrator state
        self.triangle_integrator = 0.0

        # Dedicated RNG for reproducible randomness
        self.rng = rng if rng is not None else np.random.default_rng()

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
                # Band-limited triangle wave to prevent aliasing
                # Triangle wave via band-limited square integration
                # Generate band-limited square wave
                if phase < 0.5:
                    square = 1.0
                else:
                    square = -1.0

                # Apply PolyBLEP at rising edge (phase = 0)
                square += self.poly_blep(dt, phase)
                # Apply PolyBLEP at falling edge (phase = 0.5)
                square -= self.poly_blep(dt, phase - 0.5 if phase >= 0.5 else phase + 0.5)

                # Integrate square to get triangle (leaky integrator to prevent DC drift)
                # Scale by frequency to normalize amplitude
                self.triangle_integrator = 0.999 * self.triangle_integrator + square * dt * 4.0
                signal[i] = self.triangle_integrator

            else:  # sine
                # Sine wave - naturally band-limited
                signal[i] = np.sin(2 * np.pi * phase)

            # Advance phase
            phase += dt
            if phase >= 1.0:
                phase -= 1.0

        # Save final phase for next retrigger
        self.phase = phase

        # Denormal protection - prevent CPU spikes from tiny values
        signal[np.abs(signal) < 1e-20] = 0.0

        # Micro cross-fade on retrigger to eliminate click
        # Uses equal-power curve to prevent volume dip: √(1-x) and √x
        # maintain constant perceived power (a² + b² = 1) vs linear (0.5² + 0.5² = 0.5)
        if self.last_signal_tail is not None and self.last_env_value >= 1e-3:
            xfade_len = min(self.crossfade_samples, len(signal), len(self.last_signal_tail))
            if xfade_len > 0:
                x = np.linspace(0, 1, xfade_len)
                a = np.sqrt(1 - x)  # Equal-power old
                b = np.sqrt(x)      # Equal-power new
                signal[:xfade_len] = (
                    self.last_signal_tail[:xfade_len] * a +
                    signal[:xfade_len] * b
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
            shift = self.rng.integers(0, 100)  # Small phase shift
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

        # Denormal protection - prevent CPU spikes from tiny values
        result[np.abs(result) < 1e-20] = 0.0

        # Crossfade post-VCA output for click-free retriggering
        # Uses equal-power curve to prevent volume dip
        if self.last_signal_tail is not None and self.last_env_value >= 1e-3:
            xfade_len = min(self.crossfade_samples, len(result), len(self.last_signal_tail))
            if xfade_len > 0:
                x = np.linspace(0, 1, xfade_len)
                a = np.sqrt(1 - x)  # Equal-power old
                b = np.sqrt(x)      # Equal-power new
                result[:xfade_len] = (
                    self.last_signal_tail[:xfade_len] * a +
                    result[:xfade_len] * b
                )

        # Store post-VCA tail and envelope value for next retrigger
        if len(result) >= self.crossfade_samples:
            self.last_signal_tail = result[-self.crossfade_samples:].copy()
        self.last_env_value = amp_env_signal[-1] if len(amp_env_signal) > 0 else 0.0

        return result

    def adsr_envelope(self, num_samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2, curve=0.3):
        """
        ADSR envelope generator with exponential curves for musical sound
        Times in seconds, sustain is level (0-1)
        curve: exponential curve factor (0.1 = gentle, 1.0 = aggressive)
        """
        # Guard against ultra-short attacks that cause clicks
        attack = max(attack, 0.001)  # Minimum 1ms attack
        decay = max(decay, 0.001)
        release = max(release, 0.001)

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

        # Attack - smooth rise with C1 continuity (zero slope at start and end)
        if attack_samples > 0:
            t = np.linspace(0, 1, attack_samples)
            # Smoothstep: zero slope at t=0 and t=1 for click-free transitions
            smooth_t = 3 * t**2 - 2 * t**3
            envelope[current:current+attack_samples] = smooth_t
            current += attack_samples

        # Decay - smooth fall with C1 continuity (zero slope at end)
        if decay_samples > 0 and current < num_samples:
            end = min(current + decay_samples, num_samples)
            t = np.linspace(0, 1, end - current)
            # Smoothstep: zero slope at t=0 and t=1 for click-free transitions
            smooth_t = 3 * t**2 - 2 * t**3
            envelope[current:end] = 1.0 - (1.0 - sustain) * smooth_t
            current = end

        # Sustain - constant level
        if sustain_samples > 0 and current < num_samples:
            end = min(current + sustain_samples, num_samples)
            envelope[current:end] = sustain
            current = end

        # Release - smooth fall to zero with C1 continuity (zero slope at start)
        if current < num_samples:
            t = np.linspace(0, 1, num_samples - current)
            # Smoothstep: zero slope at t=0 and t=1 for click-free transitions
            smooth_t = 3 * t**2 - 2 * t**3
            envelope[current:] = sustain * (1.0 - smooth_t)

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

        # Denormal protection - prevent CPU spikes from tiny values
        signal[np.abs(signal) < 1e-20] = 0.0

        # Micro cross-fade on retrigger
        # Uses equal-power curve to prevent volume dip: √(1-x) and √x
        # maintain constant perceived power (a² + b² = 1) vs linear (0.5² + 0.5² = 0.5)
        if self.last_signal_tail is not None and self.last_env_value >= 1e-3:
            xfade_len = min(self.crossfade_samples, len(signal), len(self.last_signal_tail))
            if xfade_len > 0:
                x = np.linspace(0, 1, xfade_len)
                a = np.sqrt(1 - x)  # Equal-power old
                b = np.sqrt(x)      # Equal-power new
                signal[:xfade_len] = (
                    self.last_signal_tail[:xfade_len] * a +
                    signal[:xfade_len] * b
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
        # Reset filter state to prevent clicks from accumulated filter energy
        # between note retriggering. Filter state must start clean for each note.
        self.filter_z1 = 0.0
        self.filter_z2 = 0.0
        self.filter_z3 = 0.0
        self.filter_z4 = 0.0

        # Reset triangle integrator to prevent DC offset accumulation across notes
        self.triangle_integrator = 0.0

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
        amp_env_signal = self.adsr_envelope(len(filtered), *amp_env)
        output = filtered * amp_env_signal

        # Denormal protection - prevent CPU spikes from tiny values
        output[np.abs(output) < 1e-20] = 0.0

        # Crossfade post-VCA output for click-free retriggering
        # Uses equal-power curve to prevent volume dip
        if self.last_signal_tail is not None and self.last_env_value >= 1e-3:
            xfade_len = min(self.crossfade_samples, len(output), len(self.last_signal_tail))
            if xfade_len > 0:
                x = np.linspace(0, 1, xfade_len)
                a = np.sqrt(1 - x)  # Equal-power old
                b = np.sqrt(x)      # Equal-power new
                output[:xfade_len] = (
                    self.last_signal_tail[:xfade_len] * a +
                    output[:xfade_len] * b
                )

        # Store post-VCA tail and envelope value for next retrigger
        if len(output) >= self.crossfade_samples:
            self.last_signal_tail = output[-self.crossfade_samples:].copy()
        self.last_env_value = amp_env_signal[-1] if len(amp_env_signal) > 0 else 0.0

        return output


    def create_heartbeat_cycle(self, lub_freq, dub_freq, lub_duration, dub_duration,
                                    gap_sec, pause_sec, dub_volume,
                                    filter_cutoff, resonance, amp_env, filter_env):
        """
        EXACT reproduction of the standalone heartbeat_designer.py logic.
        This produces ZERO clicks and ZERO booming.

        NOTE: resonance and filter_env are ignored to match the reference implementation.
        """

        # Calculate durations EXACTLY like standalone script
        # lub_duration = attack + decay + release (no sustain time)
        actual_lub_duration = amp_env[0] + amp_env[1] + amp_env[3]  # attack + decay + release
        actual_dub_duration = actual_lub_duration * 0.8  # Slightly shorter, like standalone

        # LUB Beat - exact same logic as standalone
        t_lub = np.linspace(0, actual_lub_duration, int(actual_lub_duration * self.sample_rate))
        lub_wave = np.sin(2 * np.pi * lub_freq * t_lub)
        lub_env = self._create_linear_adsr(actual_lub_duration, *amp_env)
        lub_audio = lub_wave * lub_env

        # Apply filter to enveloped signal
        lub_audio = self._apply_lowpass_filter(lub_audio, filter_cutoff)

        # Gap (zeros)
        gap_samples = int(gap_sec * self.sample_rate)
        gap_audio = np.zeros(gap_samples)

        # DUB Beat - exact same logic as standalone
        t_dub = np.linspace(0, actual_dub_duration, int(actual_dub_duration * self.sample_rate))
        dub_wave = np.sin(2 * np.pi * dub_freq * t_dub)
        dub_env = self._create_linear_adsr(actual_dub_duration, *amp_env) * dub_volume
        dub_audio = dub_wave * dub_env

        # Apply filter to enveloped signal
        dub_audio = self._apply_lowpass_filter(dub_audio, filter_cutoff)

        # Pause (zeros)
        pause_samples = int(pause_sec * self.sample_rate)
        pause_audio = np.zeros(pause_samples)

        # Combine exactly like standalone: [lub, gap, dub, pause]
        heartbeat = np.concatenate([lub_audio, gap_audio, dub_audio, pause_audio])

        # Reset for phase continuity tracking
        self.last_env_value = 0.0

        return heartbeat

    def _create_linear_adsr(self, duration_sec, attack, decay, sustain, release):
        """Linear ADSR - exact copy from heartbeat_designer.py"""
        num_samples = int(duration_sec * self.sample_rate)
        envelope = np.zeros(num_samples)

        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)

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

    def _apply_lowpass_filter(self, audio, cutoff_freq):
        """Butterworth lowpass filter - no ringing artifacts"""
        from scipy.signal import butter, sosfiltfilt

        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # 2nd order Butterworth for minimal ringing
        sos = butter(2, normalized_cutoff, btype='low', output='sos')

        # Zero-phase filtering (no phase distortion, no pre-ringing)
        return sosfiltfilt(sos, audio)


    def _generate_beat_waveform(self, freq, duration):
        """
        Generates a single beat waveform with a phase that starts at zero.
        """
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate
        phase = freq * t
        return np.sin(2 * np.pi * phase)

    # =========================================================================
    # LAYER 3B GESTURE PRIMITIVES
    # =========================================================================

    def oscillator_timevarying_pitch(self, freq_curve, waveform='sine'):
        """
        Generate oscillator with time-varying pitch curve.

        Args:
            freq_curve: Frequency curve in Hz (numpy array)
            waveform: 'sine' | 'triangle' (saw/square need PolyBLEP)

        Returns:
            Audio signal (same length as freq_curve)
        """
        n = len(freq_curve)

        # Phase accumulation for sample-accurate pitch tracking
        phase = np.cumsum(freq_curve) / self.sample_rate

        if waveform == 'sine':
            return np.sin(2 * np.pi * phase)
        elif waveform == 'triangle':
            # Triangle wave from phase
            phase_frac = phase % 1.0
            return np.where(phase_frac < 0.5,
                          4.0 * phase_frac - 1.0,
                          3.0 - 4.0 * phase_frac)
        else:
            raise NotImplementedError(f"Waveform '{waveform}' not supported for time-varying pitch")

    def moog_filter_timevarying(self, signal, cutoff_curve, resonance_curve=None):
        """
        Apply Moog filter with time-varying parameters.

        Args:
            signal: Input audio
            cutoff_curve: Cutoff in Hz (scalar or array)
            resonance_curve: Resonance 0-4 (scalar or array)

        Returns:
            Filtered audio
        """
        n = len(signal)

        # Handle scalar inputs
        if np.isscalar(cutoff_curve):
            cutoff_curve = np.full(n, cutoff_curve)
        if resonance_curve is None or np.isscalar(resonance_curve):
            resonance_curve = np.full(n, resonance_curve if resonance_curve is not None else 0.5)

        # Process in chunks for efficiency
        chunk_size = 128
        filtered = np.zeros_like(signal)

        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = signal[i:end]

            # Average parameters over chunk
            cutoff = np.mean(cutoff_curve[i:end])
            resonance = np.mean(resonance_curve[i:end])

            # Apply filter (state preserved across chunks)
            filtered[i:end] = self.moog_filter(chunk, cutoff, resonance)

        return filtered

    def generate_noise(self, num_samples, noise_type='white'):
        """
        Generate noise signal.

        Args:
            num_samples: Length of noise buffer
            noise_type: 'white' | 'pink'

        Returns:
            Noise signal (normalized to ±1.0)
        """
        if noise_type == 'white':
            return self.rng.standard_normal(num_samples)

        elif noise_type == 'pink':
            # Simple pink noise: low-pass filtered white noise
            white = self.rng.standard_normal(num_samples)
            # 1-pole LP filter
            pink = np.zeros_like(white)
            alpha = 0.1
            pink[0] = white[0] * alpha
            for i in range(1, num_samples):
                pink[i] = pink[i-1] * (1 - alpha) + white[i] * alpha
            # Normalize
            return pink / (np.std(pink) + 1e-10)

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
