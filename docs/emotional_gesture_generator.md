Below is a focused plan + minimal patch to turn Layer 3b into EMOTIONAL GESTURES (punctuations) instead of a step sequencer.
---

## Core Problems
1. Current logic = timeline step loop → uniform, mechanical.
2. Moments linger too long; compete with Layers 1+2 / heartbeat.
3. No archetypal shape per event (all rendered similarly).
4. Lacks psycho-acoustic staging (pre-echo, impact, bloom, decay, residue).

---

## New Model: Gesture-Based Moment Generator
Each moment becomes a self-contained GESTURE with five optional phases:

1. Pre-shadow (anticipatory faint harmonic / filtered noise)
2. Impact (energy spike / plunge / attack chord / spectral burst)
3. Bloom (expansion / shimmer / widening / pitch spread)
4. Decay (damped fall / narrowing / filtering down)
5. Residue (brief aura, faint modulation, heartbeat-tied pulse)

All parameters derived from moment archetype + entropy snapshot + local tension.

---

## Event Archetype Mapping (examples)

| Type | Shape | Pitch Motion | Harmony | Spectral Motion | Temporal Feel |
|------|-------|--------------|---------|-----------------|---------------|
| BLUNDER | Drop-collapse | Suspended high → fast downward gliss | Cluster resolving to muddy interval | Band-pass → low-pass choke | Sudden + short tail |
| TACTICAL_SEQUENCE | Interlocking clockwork | Fixed pitch cells | Perfect intervals + augmentations | Narrow → comb-like partial highlights | Even, precise pulses |
| TIME_PRESSURE | Accelerating tremor | Narrow oscillation rising microtonally | Sparse, thinned | High-pass swell + tremolo | Tempo acceleration curve |
| BRILLIANT | Ascending bloom | Rising 2–3 octave glide | Expanding chord (add 9, 13) | Opening filter + shimmer noise | Smooth arch |
| INACCURACY | Stumble flicker | Tiny upward then slip | Dyad w/ slight detune | Brief notch sweep | Very short, asymmetrical |

---

## Core Algorithm (per moment)
1. Capture context:
   - t_start, duration budget (short caps!).
   - entropy_at_time, section tension, overall narrative bias.
2. Build gesture spec (dataclass):
   - envelopes: amp_env_curve (custom, not ADSR), pitch_curve, filter_curve, width_curve, noise_mix_curve.
3. Render layers (mix at low gain):
   - base tonal core (supersaw or triangle depending on type)
   - transient (filtered noise / impulse)
   - motion overlay (gliss, tremolo, shimmer)
   - optional residue (very low level)
4. Apply psycho-acoustic shaping:
   - soft clip to tame peaks
   - early DC block
   - equal-power fade window
5. Insert into master moment buffer (no per-step grid).

---

## Safeguards (so they never dominate)
- Hard cap peak gain per gesture (e.g. <= 0.35 of full-scale pre-normalization).
- Global Layer3b mix ceiling after summation (normalize to target RMS -18dBFS before final mix).
- Automatic ducking disabled for gestures; instead subtract 5–8% only from drone mid band during impact frame only (optional future).

---

## Integration Steps
1. Add moment_generators.py with MomentGestureGenerator.
2. Replace current "full_sequence" logic with:
   - Build empty buffer length=section_samples.
   - For each adjusted_event → generate gesture → mix (add).
3. Remove note-by-note entropy-based pool mapping here (keep entropy use inside gesture spec).
4. Keep heartbeat (3a) unchanged.

---

## New File

````python
import numpy as np
from dataclasses import dataclass

@dataclass
class GestureSpec:
    duration: float
    sample_rate: int
    pitch_curve: np.ndarray
    amp_curve: np.ndarray
    filter_curve: np.ndarray
    width_curve: np.ndarray
    noise_curve: np.ndarray
    base_freq: float
    waveform: str
    blend: float

class MomentGestureGenerator:
    def __init__(self, synth, rng, config):
        self.synth = synth
        self.rng = rng
        self.config = config

    def _ease(self, n):
        t = np.linspace(0,1,n)
        return t*t*(3-2*t)

    def _gliss_curve(self, n, start, end, shape='exp'):
        t = np.linspace(0,1,n)
        if shape=='exp':
            return start * (end/start)**t
        return start + (end-start)*t

    def _amp_shape(self, n, archetype):
        t = np.linspace(0,1,n)
        if archetype=='BLUNDER':
            # brief suspend → plunge → muffled tail
            a = np.zeros(n)
            hold = int(n*0.15)
            fall = int(n*0.35)
            a[:hold] = 0.4 + 0.3*self._ease(hold)
            a[hold:hold+fall] = np.linspace(a[hold-1],1.0,fall)
            tail = n-(hold+fall)
            if tail>0:
                a[hold+fall:] = np.linspace(1.0,0.1,tail)
            return a
        if archetype=='BRILLIANT':
            return np.sin(np.pi * t) ** 0.7
        if archetype=='TACTICAL_SEQUENCE':
            pulses = np.sin(2*np.pi*(6+2*t)*t)
            base = 0.3 + 0.7*np.clip(pulses,0,1)
            return base * (0.95 + 0.05*np.cos(8*np.pi*t))
        if archetype=='TIME_PRESSURE':
            accel = t**0.35
            trem = 0.5+0.5*np.sin(2*np.pi*(4+20*accel)*t)
            return (0.2+0.8*accel)*trem
        if archetype=='INACCURACY':
            a = np.zeros(n)
            peak = int(n*0.25)
            a[:peak] = np.linspace(0,0.6,peak)
            mid = int(n*0.15)
            a[peak:peak+mid] = np.linspace(0.6,0.3,mid)
            a[peak+mid:] = np.linspace(0.3,0.0,n-(peak+mid))
            return a
        return self._ease(n)

    def _filter_shape(self, n, archetype):
        t = np.linspace(0,1,n)
        if archetype=='BLUNDER':
            return np.linspace(4000,400, n)
        if archetype=='BRILLIANT':
            return 1200 + 6000*(t**0.8)
        if archetype=='TACTICAL_SEQUENCE':
            return 1800 + 800*np.sin(2*np.pi*8*t)
        if archetype=='TIME_PRESSURE':
            return 900 + 3000*(t**1.4)
        if archetype=='INACCURACY':
            return 1500 + 500*np.sin(2*np.pi*3*t)*np.exp(-3*t)
        return np.linspace(800,1400,n)

    def _width_shape(self, n, archetype):
        t = np.linspace(0,1,n)
        if archetype in ['BRILLIANT']:
            return t**0.6
        if archetype=='BLUNDER':
            return np.concatenate([np.linspace(0.6,0.1,int(n*0.7)),
                                   np.linspace(0.1,0.0,n-int(n*0.7))])
        if archetype=='TIME_PRESSURE':
            return 0.2 + 0.5*(t**1.2)
        if archetype=='TACTICAL_SEQUENCE':
            return 0.3 + 0.2*np.sin(2*np.pi*6*t)
        if archetype=='INACCURACY':
            return np.linspace(0.15,0.0,n)
        return np.linspace(0.2,0.4,n)

    def _noise_curve(self, n, archetype):
        t = np.linspace(0,1,n)
        if archetype=='BLUNDER':
            return np.concatenate([np.linspace(0,0.5,int(n*0.2)),
                                   np.linspace(0.5,0.05,n-int(n*0.2))])
        if archetype=='BRILLIANT':
            return (t**2)*0.4
        if archetype=='TACTICAL_SEQUENCE':
            return 0.15 + 0.1*np.sin(2*np.pi*10*t)
        if archetype=='TIME_PRESSURE':
            return 0.2 + 0.5*(t**1.3)
        if archetype=='INACCURACY':
            return np.linspace(0.25,0.0,n)
        return np.linspace(0.1,0.2,n)

    def build_spec(self, archetype, base_freq, duration, entropy, tension):
        n = int(duration * self.synth.sample_rate)
        if n < 32:
            n = 32
        # pitch motion
        if archetype=='BLUNDER':
            pitch_curve = self._gliss_curve(n, base_freq*2, base_freq*0.5, 'exp')
        elif archetype=='BRILLIANT':
            pitch_curve = self._gliss_curve(n, base_freq*0.5, base_freq*4, 'exp')
        elif archetype=='TIME_PRESSURE':
            pitch_curve = base_freq * (1 + 0.015*np.sin(2*np.pi*np.linspace(0,1,n)*(4+entropy*10)))
        elif archetype=='TACTICAL_SEQUENCE':
            cells = [1, 4/3, 3/2, 2]
            seq = np.array([base_freq * cells[i%4] for i in range(n)])
            pitch_curve = seq
        elif archetype=='INACCURACY':
            pitch_curve = base_freq * (1 + 0.03*np.sin(2*np.pi*np.linspace(0,1,n)*3) - 0.05*np.linspace(0,1,n))
        else:
            pitch_curve = np.full(n, base_freq)

        amp_curve = self._amp_shape(n, archetype)
        filter_curve = self._filter_shape(n, archetype)
        width_curve = self._width_shape(n, archetype)
        noise_curve = self._noise_curve(n, archetype)

        # entropy/tension scaling tweaks
        amp_curve *= (0.7 + 0.6*entropy)
        filter_curve *= (0.8 + 0.4*tension)

        return GestureSpec(
            duration=duration,
            sample_rate=self.synth.sample_rate,
            pitch_curve=pitch_curve,
            amp_curve=amp_curve,
            filter_curve=filter_curve,
            width_curve=width_curve,
            noise_curve=noise_curve,
            base_freq=base_freq,
            waveform='saw' if archetype in ['BRILLIANT','TACTICAL_SEQUENCE','TIME_PRESSURE'] else 'triangle',
            blend=0.8
        )

    def render(self, spec: GestureSpec):
        n = len(spec.amp_curve)
        audio = np.zeros(n)
        # tonal core (frame-by-frame simple oscillator + filter chunked)
        phase = 0.0
        sr = spec.sample_rate
        block = 128
        for i in range(0,n,block):
            end = min(i+block, n)
            freq_block = spec.pitch_curve[i:end]
            t = (np.arange(end-i)/sr)
            # naive phase accumulation
            local = np.sin(2*np.pi*np.cumsum(freq_block)/sr + phase)
            phase = (phase + 2*np.pi*np.sum(freq_block)/sr) % (2*np.pi)
            # apply simple 1-pole low-pass approximation per sample via cumulative smoothing
            cutoff_block = spec.filter_curve[i:end]
            # quick filter (exponential smoothing)
            y = 0
            alpha = np.clip(cutoff_block/(sr*0.5),0.001,0.99)
            filt_out = np.zeros_like(local)
            for k in range(len(local)):
                y = y + alpha[k]*(local[k]-y)
                filt_out[k] = y
            audio[i:end] = filt_out

        # add noise component
        noise = (self.rng.standard_normal(n) * spec.noise_curve)
        audio += noise * 0.7

        # amplitude
        audio *= spec.amp_curve

        # soft clip
        audio = np.tanh(audio * 1.8) * 0.55

        return audio
````

---

## synth_composer.py Patch (Replace current moment step-sequencer block)

````python
// ...existing code...
from moment_generators import MomentGestureGenerator
// ...existing code inside ChessSynthComposer.__init__ after synth_layer3 init...
        self.moment_generator = MomentGestureGenerator(self.synth_layer3, self.rng, self.config)
// ...existing code...

// Inside compose_section(), in Layer 3 where current sequencer logic starts, REPLACE the entire
// moment sequence building section (from "if self.config.LAYER_ENABLE['sequencer']:" after
// heartbeat generation for sub-layer 3a, up to before global filter sweep) with:

            # === NEW GESTURE-BASED MOMENT LAYER (3b) ===
            gesture_buffer = np.zeros_like(sequencer_layer)

            if self.config.LAYER_ENABLE.get('moments', True):
                for event in moment_events:
                    start = event['start_sample']
                    end = min(event['end_sample'], len(gesture_buffer))
                    if end <= start:
                        continue

                    # Context entropy snapshot
                    event_time = event['start_time']
                    ply_offset = int(event_time)
                    if entropy_curve is not None and 0 <= ply_offset < len(entropy_curve):
                        e_val = float(entropy_curve[ply_offset])
                    else:
                        e_val = 0.5

                    tension_val = tension

                    # Derive archetype
                    archetype = event['type']
                    if archetype not in ['BLUNDER','TACTICAL_SEQUENCE','TIME_PRESSURE','BRILLIANT','INACCURACY']:
                        archetype = 'INACCURACY'

                    # Base freq: anchor around scale[0] with slight offset per type
                    base_freq = scale[0]
                    if archetype=='BRILLIANT':
                        base_freq *= 2
                    elif archetype=='BLUNDER':
                        base_freq *= 1.2
                    elif archetype=='TIME_PRESSURE':
                        base_freq *= 1.4

                    dur = max(0.25, min(event['end_time']-event['start_time'], 3.5))

                    spec = self.moment_generator.build_spec(
                        archetype=archetype,
                        base_freq=base_freq,
                        duration=dur,
                        entropy=e_val,
                        tension=tension_val
                    )

                    gesture_audio = self.moment_generator.render(spec)

                    # Fit into buffer
                    g_len = len(gesture_audio)
                    slice_len = min(g_len, len(gesture_buffer)-start)
                    if slice_len > 0:
                        # Scale so it never dominates
                        max_target = self.config.LAYER_MIXING['sequencer_note_level'] * 0.9
                        peak = np.max(np.abs(gesture_audio[:slice_len])) or 1.0
                        gesture_buffer[start:start+slice_len] += (gesture_audio[:slice_len]/peak) * max_target

                sequencer_layer = gesture_buffer
// ...existing code continues (skip old note-by-note sequence logic)...
````

Remove or comment out the old full_sequence / step logic to avoid duplication.

---

## Config Additions (Optional)
Add a dict for gesture caps:

````python
// ...existing code...
    GESTURE_LIMITS = {
        'max_peak': 0.38,
        'max_rms': -20.0,
        'post_sum_normalize': True
    }
// ...existing code...
````

(Use later if you add post-layer normalization.)

---

## Result
- No grid; every moment is a sculpted audio gesture.
- Psycho-acoustic shapes per archetype.
- Entropy + tension modulate intensity, not structure dominance.
- Heartbeat preserved as anchor (moments are transient overlays).

---

## Next Increment Ideas
- Add stereo widening per gesture (width_curve → mid/side).
- Add shimmer (short diffused delays) for BRILLIANT only.
- Add low transient thump (DC-blocked exponential) for BLUNDER impact.
- Add micro-loop granules for TIME_PRESSURE acceleration.

---
