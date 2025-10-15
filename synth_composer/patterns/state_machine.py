"""
State machine patterns - KING_HUNT, DESPERATE_DEFENSE, TACTICAL_CHAOS, CRUSHING_ATTACK.

These patterns use explicit state machines to create evolving musical behaviors.
"""

import numpy as np
from typing import List, Dict, Any
from .base import PatternGenerator
from ..core.note_event import NoteEvent
from ..core.timing_engine import TimingEngine

try:
    from synth_config import get_envelope, get_filter_envelope
except ImportError:
    def get_envelope(name, config):
        presets = {
            'soft': (0.05, 0.2, 0.8, 0.3),
            'stab': (0.01, 0.1, 0.3, 0.1),
            'pluck': (0.001, 0.05, 0.0, 0.05),
        }
        return presets.get(name, (0.01, 0.1, 0.7, 0.2))

    def get_filter_envelope(name, config):
        presets = {
            'gentle': (0.1, 0.3, 0.4, 0.3),
            'sweep': (0.01, 0.15, 0.3, 0.2),
            'closing': (0.3, 0.2, 0.3, 0.5),
        }
        return presets.get(name, (0.01, 0.15, 0.3, 0.2))


class KingHuntPattern(PatternGenerator):
    """
    KING_HUNT: Generative aggressive pursuit algorithm.

    Musical characteristics:
    - State machine: ATTACK (ascending), RETREAT (descending), PAUSE (regrouping)
    - Probabilistic upward bias with evolving aggression
    - Random octave jumps and velocity variations
    - Building intensity and speed over time
    """

    # State machine constants
    STATE_ATTACK = 0
    STATE_RETREAT = 1
    STATE_PAUSE = 2

    def __init__(self, rng: np.random.Generator):
        super().__init__(rng)

    def generate_events(self,
                       duration: float,
                       scale: List[float],
                       params: Dict[str, Any]) -> List[NoteEvent]:
        """Generate note events using aggressive pursuit state machine."""
        events = []

        timing = TimingEngine(
            sample_rate=params['sample_rate'],
            section_start_time=params.get('section_start_time', 0.0)
        )

        total_samples = int(duration * params['sample_rate'])
        base_note_dur = params['note_duration'] * 0.5  # Fast, aggressive

        # State machine initialization
        current_state = self.STATE_ATTACK
        current_note_idx = 0  # Position in scale [0-7]
        current_octave = 0  # Octave offset (0, 1, 2)
        attack_run_length = 0  # How long we've been attacking

        while not timing.is_finished(total_samples):
            # CRITICAL: progress calculated at loop start (matches original line 358)
            progress = timing.current_sample / total_samples

            # State transitions (probabilistic)
            if current_state == self.STATE_ATTACK:
                # Stay in attack mode with increasing probability
                if self.rng.random() < 0.15 - progress * 0.1:  # Less retreat as hunt intensifies
                    current_state = self.STATE_RETREAT
                    attack_run_length = 0
                elif self.rng.random() < 0.05:
                    current_state = self.STATE_PAUSE
                    attack_run_length = 0
                else:
                    attack_run_length += 1

            elif current_state == self.STATE_RETREAT:
                # Quick retreat, return to attack
                if self.rng.random() < 0.6 + progress * 0.3:  # Return faster as hunt intensifies
                    current_state = self.STATE_ATTACK

            elif current_state == self.STATE_PAUSE:
                # Brief pause, then attack
                if self.rng.random() < 0.8:
                    current_state = self.STATE_ATTACK

            # Note selection based on state
            if current_state == self.STATE_ATTACK:
                # Upward bias (70% up, 20% repeat, 10% down)
                rand = self.rng.random()
                if rand < 0.7:
                    # Move up
                    current_note_idx = min(7, current_note_idx + self.rng.integers(1, 3))

                    # Octave jumps (increasing probability with progress and attack length)
                    if self.rng.random() < 0.1 + progress * 0.2 + attack_run_length * 0.05:
                        current_octave = min(2, current_octave + 1)
                        current_note_idx = self.rng.integers(0, 4)  # Reset to lower note after jump

                elif rand < 0.9:
                    # Repeat (stabbing same note)
                    pass
                else:
                    # Small step down
                    current_note_idx = max(0, current_note_idx - 1)

            elif current_state == self.STATE_RETREAT:
                # Downward motion
                current_note_idx = max(0, current_note_idx - self.rng.integers(1, 3))
                if current_note_idx == 0 and current_octave > 0:
                    current_octave -= 1

            elif current_state == self.STATE_PAUSE:
                # Hold on dominant or tonic
                if self.rng.random() < 0.5:
                    current_note_idx = 0  # Tonic
                else:
                    current_note_idx = 4  # Dominant

            # Get frequency
            note_freq = scale[current_note_idx] * (2 ** current_octave)

            # Duration varies by state and progress
            if current_state == self.STATE_ATTACK:
                # Faster as hunt intensifies
                note_dur = base_note_dur * self.rng.uniform(0.4, 0.8) * (1.0 - progress * 0.3)
            elif current_state == self.STATE_RETREAT:
                note_dur = base_note_dur * self.rng.uniform(0.5, 1.0)
            else:  # PAUSE
                note_dur = base_note_dur * self.rng.uniform(1.5, 2.5)

            # Velocity varies by state
            if current_state == self.STATE_ATTACK:
                velocity = self.rng.uniform(0.8, 1.0) * (1.0 + progress * 0.3)  # Louder as hunt intensifies
            elif current_state == self.STATE_RETREAT:
                velocity = self.rng.uniform(0.6, 0.8)
            else:  # PAUSE
                velocity = self.rng.uniform(0.5, 0.7)

            # Quantize to samples (CRITICAL: match original behavior exactly)
            note_samples = int(note_dur * params['sample_rate'])
            note_samples = min(note_samples, timing.remaining_samples(total_samples))

            if note_samples > 0:
                # Convert back to seconds (quantized duration)
                note_dur_quantized = note_samples / params['sample_rate']

                # Filter and resonance evolve with progress and state
                filter_mult = 1.0 + progress * 2.0  # Much brighter as hunt intensifies
                if current_state == self.STATE_ATTACK:
                    filter_mult *= self.rng.uniform(1.2, 1.5)  # Extra brightness on attack

                resonance_mult = 1.0 + progress * 0.8

                events.append(NoteEvent(
                    freq=note_freq,
                    duration=note_dur_quantized,
                    timestamp=timing.get_timestamp(),
                    velocity=velocity,
                    waveform='saw',  # Aggressive, bright
                    filter_base=params['filter'] * filter_mult,
                    filter_env_amount=params['filter_env'] * self.rng.uniform(0.8, 1.2),
                    resonance=params['resonance'] * resonance_mult,
                    amp_env=get_envelope('stab', params['config']),
                    filter_env=get_filter_envelope('sweep', params['config']),
                    amp_env_name='stab',
                    filter_env_name='sweep',
                    extra_context={
                        'state': self._state_name(current_state),
                        'scale_degree': current_note_idx,
                        'octave': current_octave,
                        'attack_run_length': attack_run_length,
                        'filter_mult': filter_mult,
                        'resonance_mult': resonance_mult,
                        'velocity': velocity,
                        'level': params['config'].LAYER_MIXING['pattern_note_level'] * velocity
                    }
                ))

            # Advance timeline (CRITICAL: outside if block, match original line 473)
            timing.advance(note_samples)

            # Pause between notes (minimal in attack, longer in pause)
            # Uses progress from START of this iteration
            if current_state == self.STATE_ATTACK:
                pause_dur = base_note_dur * 0.05 * (1.0 - progress * 0.5)
            elif current_state == self.STATE_RETREAT:
                pause_dur = base_note_dur * 0.15
            else:  # PAUSE
                pause_dur = base_note_dur * 0.5

            timing.add_pause(pause_dur)

        # Debug output
        if events:
            state_counts = {}
            for e in events:
                state = e.extra_context.get('state', 'unknown')
                state_counts[state] = state_counts.get(state, 0) + 1
            self.print_debug_summary(events, extra_stats={'states': state_counts})

        return events

    def _state_name(self, state: int) -> str:
        """Convert state constant to readable name."""
        if state == self.STATE_ATTACK:
            return 'attack'
        elif state == self.STATE_RETREAT:
            return 'retreat'
        else:
            return 'pause'

    def __repr__(self) -> str:
        return "KingHuntPattern(KING_HUNT)"


# Placeholder stubs for other state machine patterns
# TODO: Implement full versions
class DesperateDefensePattern(PatternGenerator):
    """
    DESPERATE_DEFENSE: Reactive, hesitant defensive patterns.

    Musical characteristics:
    - Lower register, darker tone
    - Syncopated rhythms with pauses (hesitation)
    - Responds to pressure with tentative moves
    - Inspired by defender struggling under attack
    """

    # Defensive states
    STATE_RETREAT = 0      # Move pieces back
    STATE_BLOCKADE = 1     # Hold defensive structure
    STATE_COUNTER = 2      # Brief counter-attack attempt

    def __init__(self, rng: np.random.Generator):
        super().__init__(rng)

    def generate_events(self,
                       duration: float,
                       scale: List[float],
                       params: Dict[str, Any]) -> List[NoteEvent]:
        """
        Generate note events using defensive hesitant state machine.

        Now entropy-driven (Laurie Spiegel-inspired):
        - Low entropy: Simple, resigned, slow
        - High entropy: Complex, desperate complications, faster
        """
        events = []

        timing = TimingEngine(
            sample_rate=params['sample_rate'],
            section_start_time=params.get('section_start_time', 0.0)
        )

        total_samples = int(duration * params['sample_rate'])
        base_note_dur = params['note_duration'] * 1.4  # Slower, more deliberate

        # Extract entropy curve (game-specific complexity)
        entropy_curve = params.get('entropy_curve', None)
        start_ply = params.get('section_start_ply', 1)
        end_ply = params.get('section_end_ply', start_ply + 20)

        current_state = self.STATE_RETREAT
        tension_accumulator = 0

        while not timing.is_finished(total_samples):
            progress = timing.current_sample / total_samples

            # Get current entropy value for this position
            if entropy_curve is not None and len(entropy_curve) > 0:
                # Map progress to ply index
                ply_index = int(progress * len(entropy_curve))
                ply_index = min(ply_index, len(entropy_curve) - 1)
                current_entropy = entropy_curve[ply_index]
            else:
                current_entropy = 0.5  # Default medium entropy

            # ENTROPY-DRIVEN STATE TRANSITIONS
            # Higher entropy = more tactical complexity = faster state changes
            entropy_rate = 1.0 + current_entropy * 2.0  # 1.0 to 3.0 multiplier
            tension_accumulator += self.rng.uniform(0, params['tension'] * 2 * entropy_rate)

            if current_state == self.STATE_RETREAT:
                if tension_accumulator > 3:
                    current_state = self.STATE_BLOCKADE
                    tension_accumulator = 0
            elif current_state == self.STATE_BLOCKADE:
                # Higher entropy = more likely to try counter-attacks (desperate complications)
                counter_prob = 0.2 + current_entropy * 0.3  # 0.2 to 0.5
                if tension_accumulator > 5 and self.rng.random() < counter_prob:
                    current_state = self.STATE_COUNTER
                    tension_accumulator = 0
                elif tension_accumulator > 4:
                    current_state = self.STATE_RETREAT
                    tension_accumulator = 0
            elif current_state == self.STATE_COUNTER:
                # Counter attempts are brief
                if self.rng.random() < 0.6:
                    current_state = self.STATE_RETREAT
                    tension_accumulator = 0

            # ENTROPY-DRIVEN NOTE SELECTION
            if current_state == self.STATE_RETREAT:
                # Descending patterns, lower register
                note_idx = self.rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                octave_mult = 0.5  # Drop octave
            elif current_state == self.STATE_BLOCKADE:
                # Hold root and fifth (stable but tense)
                note_idx = self.rng.choice([0, 4], p=[0.6, 0.4])
                octave_mult = 1.0
            elif current_state == self.STATE_COUNTER:
                # Brief ascending attempt
                # Higher entropy = more chromatic (desperate, reaching for anything)
                if current_entropy > 0.6:
                    note_idx = self.rng.choice([3, 4, 5, 6, 7])  # Wider range
                else:
                    note_idx = self.rng.choice([4, 5, 6, 7], p=[0.25, 0.25, 0.25, 0.25])
                octave_mult = 1.0

            note_freq = scale[note_idx] * octave_mult

            # ENTROPY-DRIVEN HESITATION
            # Low entropy: predictable pauses (resignation)
            # High entropy: chaotic timing (frantic, tactical complexity)
            hesitation_prob = 0.4 - current_entropy * 0.2  # 0.4 to 0.2 (less hesitation in tactics)
            if self.rng.random() < hesitation_prob:
                hesitation_samples = int(base_note_dur * 0.3 * params['sample_rate'] * self.rng.random())
                timing.advance(hesitation_samples)

            if timing.is_finished(total_samples):
                break

            # ENTROPY-DRIVEN NOTE DURATION
            # High entropy = faster notes (tactical sequences, more moves)
            duration_mult = 1.0 - current_entropy * 0.4  # 1.0 to 0.6
            note_dur = base_note_dur * duration_mult * self.rng.uniform(0.7, 1.1)

            # ENTROPY-DRIVEN VELOCITY (replaces linear fade)
            # Low entropy: quiet, resigned
            # High entropy: louder, fighting back
            base_velocity = 0.45 + current_entropy * 0.2  # 0.45 to 0.65
            velocity = base_velocity - progress * 0.1  # Still slight fade overall

            note_samples = int(note_dur * params['sample_rate'])
            note_samples = min(note_samples, timing.remaining_samples(total_samples))

            if note_samples > 0:
                note_dur_quantized = note_samples / params['sample_rate']

                # Waveform selection by state
                waveform = 'saw' if current_state == self.STATE_RETREAT else 'triangle'

                # ENTROPY-DRIVEN FILTER
                # High entropy = brighter (more activity, tactical tension)
                filter_mult = 0.5 + progress * 0.3 + current_entropy * 0.4

                events.append(NoteEvent(
                    freq=note_freq,
                    duration=note_dur_quantized,
                    timestamp=timing.get_timestamp(),
                    velocity=velocity,
                    waveform=waveform,
                    filter_base=params['filter'] * filter_mult,
                    filter_env_amount=params['filter_env'] * 0.5,
                    resonance=params['resonance'] * (0.8 + params['tension'] * 0.4 + current_entropy * 0.3),
                    amp_env=get_envelope('sustained', params['config']),
                    filter_env=get_filter_envelope('closing', params['config']),
                    amp_env_name='sustained',
                    filter_env_name='closing',
                    extra_context={
                        'state': self._state_name(current_state),
                        'note_idx': note_idx,
                        'octave_mult': octave_mult,
                        'tension_accumulator': tension_accumulator,
                        'velocity': velocity,
                        'entropy': current_entropy,  # Track entropy for debugging
                        'level': params['config'].LAYER_MIXING['pattern_note_level'] * velocity
                    }
                ))

            # Advance timeline
            timing.advance(note_samples)

            # ENTROPY-DRIVEN PAUSES
            # High entropy = shorter pauses (rapid tactical exchanges)
            # Low entropy = longer pauses (slow, thoughtful defense)
            pause_mult = 1.5 - current_entropy * 0.8  # 1.5 to 0.7
            pause_dur = base_note_dur * pause_mult * self.rng.uniform(0.15, 0.35)
            timing.add_pause(pause_dur)

        # Debug output
        if events:
            # Count states
            state_counts = {}
            entropy_values = []
            for e in events:
                state = e.extra_context.get('state', 'unknown')
                state_counts[state] = state_counts.get(state, 0) + 1
                if 'entropy' in e.extra_context:
                    entropy_values.append(e.extra_context['entropy'])

            # Build stats dict
            extra_stats = {'states': state_counts}
            if entropy_values:
                extra_stats['entropy'] = f"mean={np.mean(entropy_values):.3f}, range=[{np.min(entropy_values):.3f}, {np.max(entropy_values):.3f}]"

            self.print_debug_summary(events, extra_stats=extra_stats)

        return events

    def _state_name(self, state: int) -> str:
        """Convert state constant to readable name."""
        if state == self.STATE_RETREAT:
            return 'retreat'
        elif state == self.STATE_BLOCKADE:
            return 'blockade'
        else:
            return 'counter'

    def __repr__(self) -> str:
        return "DesperateDefensePattern(DESPERATE_DEFENSE)"


class TacticalChaosPattern(PatternGenerator):
    """
    TACTICAL_CHAOS: Rapid, unpredictable tactical exchanges.

    Musical characteristics:
    - Wide register jumps
    - Dense overlapping bursts
    - Alternating attack/defense fragments
    - High entropy, nervous energy
    """

    def __init__(self, rng: np.random.Generator):
        super().__init__(rng)

    def generate_events(self,
                       duration: float,
                       scale: List[float],
                       params: Dict[str, Any]) -> List[NoteEvent]:
        """Generate note events using chaotic burst patterns."""
        events = []

        timing = TimingEngine(
            sample_rate=params['sample_rate'],
            section_start_time=params.get('section_start_time', 0.0)
        )

        total_samples = int(duration * params['sample_rate'])
        base_note_dur = params['note_duration'] * 0.6  # Fast exchanges

        # Chaos parameters
        burst_mode = False
        burst_countdown = 0
        attack_side = 0  # 0=white, 1=black (alternating tactical blows)

        while not timing.is_finished(total_samples):
            progress = timing.current_sample / total_samples

            # Random burst mode activation
            if not burst_mode and self.rng.random() < 0.15:
                burst_mode = True
                burst_countdown = self.rng.integers(3, 8)  # 3-8 rapid notes

            if burst_mode:
                burst_countdown -= 1
                if burst_countdown <= 0:
                    burst_mode = False
                    attack_side = 1 - attack_side  # Switch attacker

            # Note selection - chaotic jumps
            if burst_mode:
                # Tactical burst - wide jumps
                note_idx = self.rng.choice(range(len(scale)))
                # Random octave shifts
                octave_choices = [0.5, 1.0, 2.0]
                octave_mult = self.rng.choice(octave_choices, p=[0.2, 0.5, 0.3])
            else:
                # Cautious repositioning between bursts
                note_idx = self.rng.choice([0, 2, 4, 5], p=[0.3, 0.25, 0.25, 0.2])
                octave_mult = 1.0

            note_freq = scale[note_idx] * octave_mult

            # Timing - erratic in bursts, steadier between
            if burst_mode:
                note_dur = base_note_dur * self.rng.uniform(0.4, 0.7)
                velocity = 0.7 + self.rng.random() * 0.2
            else:
                note_dur = base_note_dur * self.rng.uniform(1.0, 1.5)
                velocity = 0.5 + self.rng.random() * 0.15

            note_samples = int(note_dur * params['sample_rate'])
            note_samples = min(note_samples, timing.remaining_samples(total_samples))

            if note_samples > 0:
                note_dur_quantized = note_samples / params['sample_rate']

                # Aggressive waveform, wide filter sweeps
                waveform = 'square' if burst_mode else 'pulse'
                amp_env_name = 'percussive' if burst_mode else 'stab'
                filter_env_name = 'sharp' if burst_mode else 'smooth'

                events.append(NoteEvent(
                    freq=note_freq,
                    duration=note_dur_quantized,
                    timestamp=timing.get_timestamp(),
                    velocity=velocity,
                    waveform=waveform,
                    filter_base=params['filter'] * (0.6 + progress * 0.8),
                    filter_env_amount=params['filter_env'] * (1.5 if burst_mode else 0.8),
                    resonance=params['resonance'] * (1.3 if burst_mode else 0.9),
                    amp_env=get_envelope(amp_env_name, params['config']),
                    filter_env=get_filter_envelope(filter_env_name, params['config']),
                    amp_env_name=amp_env_name,
                    filter_env_name=filter_env_name,
                    extra_context={
                        'burst_mode': burst_mode,
                        'attack_side': attack_side,
                        'note_idx': note_idx,
                        'octave_mult': octave_mult,
                        'velocity': velocity
                    }
                ))

            # Advance timeline
            timing.advance(note_samples)

            # Minimal pause in bursts, longer between
            if burst_mode:
                pause_dur = base_note_dur * 0.05  # Almost no pause
            else:
                pause_dur = base_note_dur * self.rng.uniform(0.2, 0.5)

            timing.add_pause(pause_dur)

        # Debug output
        self.print_debug_summary(events, extra_stats={'burst_mode': f"{sum(1 for e in events if 'square' == e.waveform)}/{len(events)}"})

        return events

    def __repr__(self) -> str:
        return "TacticalChaosPattern(TACTICAL_CHAOS)"

class CrushingAttackPattern(PatternGenerator):
    """
    CRUSHING_ATTACK: Generative relentless assault algorithm.

    Musical characteristics:
    - State machine: ADVANCE (building pressure), STRIKE (hammer blows), OVERWHELM (climax)
    - Downward bias (crushing down on opponent)
    - Random chord stabs (multiple simultaneous notes)
    - Accelerating rhythm and increasing volume
    """

    # State machine constants
    STATE_ADVANCE = 0
    STATE_STRIKE = 1
    STATE_OVERWHELM = 2

    def __init__(self, rng: np.random.Generator):
        super().__init__(rng)

    def generate_events(self,
                       duration: float,
                       scale: List[float],
                       params: Dict[str, Any]) -> List[NoteEvent]:
        """Generate note events using relentless assault state machine with chord stabs."""
        events = []

        timing = TimingEngine(
            sample_rate=params['sample_rate'],
            section_start_time=params.get('section_start_time', 0.0)
        )

        total_samples = int(duration * params['sample_rate'])
        base_note_dur = params['note_duration'] * 0.4  # Fast, aggressive

        # State machine initialization
        current_state = self.STATE_ADVANCE
        current_note_idx = 7  # Start high (crushing down from above)
        strike_count = 0  # Consecutive strikes

        while not timing.is_finished(total_samples):
            progress = timing.current_sample / total_samples

            # State transitions
            if current_state == self.STATE_ADVANCE:
                # Build pressure, then strike
                if self.rng.random() < 0.3 + progress * 0.2:  # More strikes as attack intensifies
                    current_state = self.STATE_STRIKE
                    strike_count = self.rng.integers(2, 5)  # Multiple hammer blows

            elif current_state == self.STATE_STRIKE:
                strike_count -= 1
                if strike_count <= 0:
                    if progress > 0.6 and self.rng.random() < 0.3:
                        current_state = self.STATE_OVERWHELM  # Climax
                    else:
                        current_state = self.STATE_ADVANCE

            elif current_state == self.STATE_OVERWHELM:
                # Stay in overwhelm mode once reached
                pass

            # Note selection based on state
            if current_state == self.STATE_ADVANCE:
                # Downward movement (pressure building)
                if self.rng.random() < 0.7:
                    current_note_idx = max(0, current_note_idx - self.rng.integers(1, 3))
                else:
                    current_note_idx = min(7, current_note_idx + 1)  # Occasional upward jab

            elif current_state == self.STATE_STRIKE:
                # Hammer on low notes (powerful blows)
                current_note_idx = self.rng.choice([0, 1, 2])  # Low register only

            elif current_state == self.STATE_OVERWHELM:
                # Chaotic attacks across entire range
                current_note_idx = self.rng.integers(0, 8)

            # Get base frequency
            note_freq = scale[current_note_idx]

            # Octave variations (more powerful with wider range)
            octave_shift = 0.0  # Default no shift
            if current_state == self.STATE_STRIKE:
                # Strike uses lower octave (bass power)
                if self.rng.random() < 0.5:
                    note_freq *= 0.5
                    octave_shift = -1.0
            elif current_state == self.STATE_OVERWHELM:
                # Overwhelm uses wide octave range
                octave_shift = float(self.rng.choice([-1, 0, 0, 1]))  # Bias toward higher
                note_freq *= (2.0 ** octave_shift)

            # Duration varies by state and progress
            if current_state == self.STATE_ADVANCE:
                note_dur = base_note_dur * self.rng.uniform(0.8, 1.2) * (1.0 - progress * 0.3)
            elif current_state == self.STATE_STRIKE:
                # Short, sharp attacks
                note_dur = base_note_dur * self.rng.uniform(0.3, 0.5)
            else:  # OVERWHELM
                # Very fast, relentless
                note_dur = base_note_dur * self.rng.uniform(0.2, 0.4) * (1.0 - progress * 0.2)

            # Velocity increases with progress and state
            # Capped to prevent clipping when multiple notes overlap
            if current_state == self.STATE_ADVANCE:
                velocity = self.rng.uniform(0.7, 0.9) * (1.0 + progress * 0.3)
            elif current_state == self.STATE_STRIKE:
                velocity = 1.0
            else:  # OVERWHELM
                velocity = self.rng.uniform(0.9, 1.0)  # No progress multiplier - prevents clipping

            # Quantize to samples
            note_samples = int(note_dur * params['sample_rate'])
            note_samples = min(note_samples, timing.remaining_samples(total_samples))

            if note_samples > 0:
                # Convert back to seconds (quantized duration)
                note_dur_quantized = note_samples / params['sample_rate']

                # Filter opens more as attack intensifies
                filter_mult = 1.0 + progress * 2.5
                if current_state == self.STATE_STRIKE or current_state == self.STATE_OVERWHELM:
                    filter_mult *= 1.5  # Extra brightness on strikes

                resonance_mult = 1.0 + progress * 0.5  # Reduced to prevent filter self-oscillation

                # CHORD STABS: Add harmonic notes on strikes and overwhelm
                has_chord = (current_state == self.STATE_STRIKE or current_state == self.STATE_OVERWHELM) and self.rng.random() < 0.6

                # Voice normalization: scale down when multiple voices present
                num_voices = 2 if has_chord else 1
                voice_scale = 1.0 / np.sqrt(num_voices)

                # Reduce level for this pattern to prevent clipping when overlapping
                # CRUSHING_ATTACK can have 3+ simultaneous note+chord pairs
                base_level = params['config'].LAYER_MIXING['pattern_note_level'] * 0.3
                level = base_level * velocity

                # Create main note event
                events.append(NoteEvent(
                    freq=note_freq,
                    duration=note_dur_quantized,
                    timestamp=timing.get_timestamp(),
                    velocity=velocity,
                    waveform='saw',
                    filter_base=params['filter'] * filter_mult,
                    filter_env_amount=params['filter_env'] * self.rng.uniform(1.0, 1.5),
                    resonance=params['resonance'] * resonance_mult,
                    amp_env=get_envelope('stab', params['config']),
                    filter_env=get_filter_envelope('sweep', params['config']),
                    amp_env_name='stab',
                    filter_env_name='sweep',
                    extra_context={
                        'state': self._state_name(current_state),
                        'scale_degree': current_note_idx,
                        'octave_shift': octave_shift,
                        'has_chord': has_chord,
                        'num_voices': num_voices,
                        'voice_scale': voice_scale,
                        'filter_mult': filter_mult,
                        'resonance_mult': resonance_mult,
                        'velocity': velocity,
                        'level': level * voice_scale
                    }
                ))

                # Add chord note if applicable
                if has_chord:
                    # Add fifth or octave
                    chord_interval = self.rng.choice([4, 7])  # Perfect fourth or fifth
                    chord_idx = min(7, current_note_idx + chord_interval)
                    chord_freq = scale[chord_idx] * (2 ** octave_shift if current_state == self.STATE_OVERWHELM else 1.0)

                    events.append(NoteEvent(
                        freq=chord_freq,
                        duration=note_dur_quantized,
                        timestamp=timing.get_timestamp(),
                        velocity=velocity,
                        waveform='saw',
                        filter_base=params['filter'] * filter_mult,
                        filter_env_amount=params['filter_env'] * self.rng.uniform(1.0, 1.5),
                        resonance=params['resonance'] * resonance_mult,
                        amp_env=get_envelope('stab', params['config']),
                        filter_env=get_filter_envelope('sweep', params['config']),
                        amp_env_name='stab',
                        filter_env_name='sweep',
                        extra_context={
                            'state': self._state_name(current_state),
                            'is_chord_note': True,
                            'chord_interval': chord_interval,
                            'chord_idx': chord_idx,
                            'scale_degree': current_note_idx,
                            'octave_shift': octave_shift,
                            'num_voices': num_voices,
                            'voice_scale': voice_scale,
                            'filter_mult': filter_mult,
                            'resonance_mult': resonance_mult,
                            'velocity': velocity,
                            'level': level * voice_scale
                        }
                    ))

            # Advance timeline
            timing.advance(note_samples)

            # Pause between notes (decreases as attack intensifies)
            if current_state == self.STATE_ADVANCE:
                pause_dur = base_note_dur * 0.2 * (1.0 - progress * 0.5)
            elif current_state == self.STATE_STRIKE:
                pause_dur = base_note_dur * 0.05  # Minimal pause
            else:  # OVERWHELM
                pause_dur = base_note_dur * 0.02  # Almost no pause

            timing.add_pause(pause_dur)

        return events

    def _state_name(self, state: int) -> str:
        """Convert state constant to readable name."""
        if state == self.STATE_ADVANCE:
            return 'advance'
        elif state == self.STATE_STRIKE:
            return 'strike'
        else:
            return 'overwhelm'

    def __repr__(self) -> str:
        return "CrushingAttackPattern(CRUSHING_ATTACK)"
