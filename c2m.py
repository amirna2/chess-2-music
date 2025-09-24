#!/usr/bin/env python3
import sys
import chess.pgn
import mido
import re
import csv
import pathlib
import yaml
import math
import argparse
import subprocess
from collections import defaultdict

# MIDI channel assignments (0-15)
CHANNELS = {
    'white': 0,
    'black': 1,
    'drone': 2,
    'arpeggio': 3,  # thinking time arpeggios
    'effects': 4,  # optional separate channel for NAG/capture/check accents
    'white_drone': 5,
    'black_drone': 6,
}

_LAST_PROGRAM = {}


def calculate_move_distance(move, board):
    """Calculate the actual distance a piece travels."""
    from_square = move.from_square
    to_square = move.to_square

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    # For most pieces, use Chebyshev distance (max of file/rank difference)
    # This matches how pieces move in chess
    file_dist = abs(to_file - from_file)
    rank_dist = abs(to_rank - from_rank)

    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.KNIGHT:
        # Knight always moves "3" in the L-shape sense
        # (2+1 squares in L pattern)
        return 3
    else:
        # King, Queen, Rook, Bishop: maximum of file/rank distance
        # Pawn: will be 1 or 2 based on actual move
        return max(file_dist, rank_dist)

def move_distance_to_duration(distance, piece_type, config):
    """Convert move distance to musical duration."""

    # Base duration per square of movement
    ticks_per_square = config["dynamic_durations"].get('ticks_per_square', 120)

    # Minimum duration so even 1-square moves are audible
    min_duration = config["dynamic_durations"].get('min_duration', 240)

    # Calculate base duration
    duration = max(min_duration, distance * ticks_per_square)

    # Optional: piece-specific adjustments
    piece_modifiers = config["dynamic_durations"].get('piece_modifiers', {})
    if piece_type in piece_modifiers:
        duration = int(duration * piece_modifiers[piece_type])

    # Cap maximum duration
    max_duration = config["dynamic_durations"].get('max_duration', 960)

    return min(duration, max_duration)


# Logging / tracking structures
MOVE_EVENTS = []  # Stores dicts: start_tick, note, duration, piece, program, channel, ply, side

def format_note(note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[note % 12]}{note // 12 - 1}"

def print_move_line(ply, side, san, piece, note, program, velocity, duration, start_tick, thinking_time=None, compressed_window=None, arpeggio_pattern=None):
    """Print a single move line in tabular form.

    Prints a header once on first invocation.
    Columns:
      PLY | Side | SAN | EMT(s) | CWin(s) | Expr | Arp | Note | Pitch | Prog | Vel | Dur(t) | StartTick
    """
    # One-time header
    if not getattr(print_move_line, "_header_printed", False):
        # Define columns with widths matching the data row formatting
        col_defs = [
            ("PLY", 3), ("Side", 5), ("SAN", 8), ("EMT(s)", 6), ("CWin(s)", 6), ("Expr", 6), ("Arp", 4),
            ("Note", 4), ("Pitch", 5), ("Prog", 4), ("Vel", 3), ("Dur(t)", 6), ("StartTick", 9)
        ]
        header = " | ".join(f"{name:<{w}}" for name, w in col_defs)
        ruler = "-+-".join('-'*w for _, w in col_defs)
        print(header)
        print(ruler)
        print_move_line._header_printed = True

    note_str = format_note(note)
    tt = f"{thinking_time:.1f}" if thinking_time is not None else "--"
    cw = f"{compressed_window:.2f}" if (compressed_window is not None and compressed_window > 0) else "--"
    # Ensure SAN fits column (truncate with ellipsis if very long)
    san_disp = san if len(san) <= 8 else san[:7] + "…"
    expr_full = getattr(print_move_line, "_last_expr", "--") or "--"
    # Abbreviate to keep column alignment (max 6 chars)
    abbrev_map = {
        'default': 'def', 'great': 'great', 'brilliant': 'bril', 'blunder': 'blun',
        'mistake': 'mist', 'interesting': 'intr', 'inaccuracy': 'inac', 'forced': 'forc',
        'checkmate': 'mate', 'check': 'check', 'capture': 'capt', 'epic': 'epic', 'long': 'long', 'none': '--'
    }
    expr = abbrev_map.get(expr_full, expr_full[:6])
    # Format arpeggio pattern for display
    arp_display = "--"
    if arpeggio_pattern:
        # Abbreviate pattern names to fit in 4 characters
        arp_abbrev = {
            'micro': 'mcro', 'short': 'shrt', 'medium': 'med',
            'long': 'long', 'epic': 'epic'
        }
        arp_display = arp_abbrev.get(arpeggio_pattern, arpeggio_pattern[:4])

    line = (
        f"{ply:03d} | {side:<5} | {san_disp:<8} | {tt:>6} | {cw:>6} | {expr:<6} | {arp_display:<4} | "
        f"{note:>4} | {note_str:<5} | {program:>4} | {velocity:>3} | {duration:>6} | {start_tick:>9}"
    )
    print(line)

def build_ascii_sheet(events, ppq, max_measures=24):
    if not events:
        return ["(no move events)"]
    measure_len = 4 * ppq  # assume 4/4
    measures = defaultdict(list)
    for ev in events:
        m = ev['start_tick'] // measure_len
        measures[m].append(ev)
    lines = []
    for m in range(0, min(max(measures.keys()) + 1, max_measures)):
        evs = sorted(measures.get(m, []), key=lambda e: e['start_tick'])
        label = f"M{m+1:02d}"
        if not evs:
            lines.append(f"{label} | (rest)")
            continue
        cells = []
        for e in evs:
            cells.append(f"{e['side'][0]}:{format_note(e['note'])}")
        lines.append(f"{label} | " + ' '.join(cells))
    return lines

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PGN with EMT annotations into an expressive multi-track MIDI: "
            "includes drone thinking windows, expression-based scaling, and timing summary."
        )
    )
    parser.add_argument('pgn_file', help='Input PGN with EMT/time comments (mainline only)')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration YAML (default: config.yaml)')
    parser.add_argument('--output', '-o', help='Explicit output MIDI path (default: <pgn_basename>_music.mid)')
    parser.add_argument('--play', action='store_true', help='After render, attempt to play the MIDI file (macOS: open / Linux: xdg-open)')
    parser.add_argument('--sheet', action='store_true', help='Print first measures as ASCII sheet at end')
    parser.add_argument('--no-move-log', action='store_true', help='Suppress per-move tabular move/event log')
    parser.add_argument('--track-stats', action='store_true', help='Print per-track basic statistics summary')
    return parser.parse_args()

def get_optimal_octave_shift(program, base_midpoint):
    """Expert-level orchestral register placement based on Spiegel-level knowledge."""

    # Standard orchestral registers (MIDI note targets for optimal timbre)
    REGISTER_TARGETS = {
        # STRINGS - treble to bass progression
        (40, 44, 45): 79,    # Violin family - treble register (G5)
        (41, 48, 49): 65,    # Viola/ensemble - alto register (F4)
        (42,): 50,           # Cello - tenor register (D3)
        (43,): 40,           # Contrabass - bass register (E2)

        # WOODWINDS - soprano to bass
        (72, 73): 84,        # Piccolo/Flute - soprano register (C6)
        (68, 69): 74,        # Oboe/English Horn - mezzo register (D5)
        (71,): 72,           # Clarinet - alto register (C5)
        (70,): 55,           # Bassoon - bass register (G3)

        # BRASS - standard orchestral placement
        (56, 59): 69,        # Trumpet - tenor register (A4)
        (60,): 55,           # French Horn - alto register (G3)
        (57,): 48,           # Trombone - bass register (C3)
        (58,): 43,           # Tuba - contrabass register (G2)

        # KEYBOARDS/PIANO - preserve board mapping
        (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23): base_midpoint,

        # PERCUSSION - preserve board mapping
        (112, 113, 114, 115, 116, 117, 118, 119, 47): base_midpoint,

        # PLUCKED STRINGS
        (24, 25, 26, 27, 28, 29, 30, 31): 64,  # Guitars - standard guitar range
        (32, 33, 34, 35, 36, 37, 38, 39): 41,  # Bass - bass register
        (46,): 60,           # Harp - wide range centered on C4

        # SAXOPHONE FAMILY
        (64,): 72,           # Soprano Sax - alto register
        (65,): 65,           # Alto Sax - tenor register
        (66,): 60,           # Tenor Sax - bass register
        (67,): 53,           # Baritone Sax - low bass register
    }

    # Find matching register target
    for programs, target_midpoint in REGISTER_TARGETS.items():
        if program in programs:
            return round(target_midpoint - base_midpoint)

    # Default: no shift for synths, FX, and unknown instruments
    return 0


def load_config(config_file="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Calculate current base range midpoint
    pitch_values = list(config['pitch_mapping'].values())
    base_low = min(pitch_values)
    base_high = max(pitch_values) + 21  # +21 for rank 8 calculation
    base_midpoint = (base_low + base_high) / 2

    # Apply expert orchestral register placement
    for piece, instrument_config in config['instruments'].items():
        if isinstance(instrument_config, dict) and 'program' in instrument_config:
            program = instrument_config['program']
            optimal_shift = get_optimal_octave_shift(program, base_midpoint)
            instrument_config['octave_shift'] = optimal_shift

    return config


def extract_nag_code(comment):
    # Look for NAG codes like $1, $2, $4, $6, etc.
    nag_match = re.search(r'\$(\d+)', comment)
    return int(nag_match.group(1)) if nag_match else None

def extract_clock_time(comment):
    """Extract remaining clock time in seconds."""
    m = re.search(r"\[%clk (\d+):(\d+):(\d+)\]", comment)
    if m:
        hours, minutes, seconds = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return hours * 3600 + minutes * 60 + seconds
    return None

def extract_timestamp(comment):
    # Try timestamp format first (direct thinking time)
    m = re.search(r"\[%timestamp (\d+)\]", comment)
    if m:
        return int(m.group(1))
    return None

def safe_get_emt(node):
    """Safely obtain elapsed move time (EMT) from a python-chess node.

    Uses the official Node.emt() API when available. Falls back to a local
    regex parse of the node comment if running on an older python-chess
    version or if multiple comment fragments are merged differently.
    Returns float seconds or None.
    """
    # Preferred: python-chess provided API (added upstream)
    try:
        emt = node.emt()  # type: ignore[attr-defined]
        if emt is not None:
            return float(emt)
    except AttributeError:
        pass

    # Fallback: manual regex on node.comment
    comment_text = getattr(node, "comment", "") or ""
    m = re.search(r"\[%emt\s+(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)\]", comment_text)
    if m:
        return (
            int(m.group("hours")) * 3600
            + int(m.group("minutes")) * 60
            + float(m.group("seconds"))
        )
    return None

def preprocess_timing_data(game):
    """Collect elapsed move times (EMT) per ply using python-chess.

    Returns dict: ply_number (1-based) -> elapsed move time in seconds (float).
    """
    timing_data = {}
    node = game
    ply = 0
    # Traverse the mainline by walking variations instead of re-parsing moves
    while node.variations:
        node = node.variation(0)
        ply += 1
        emt = safe_get_emt(node)
        if emt is not None and emt > 0:
            timing_data[ply] = emt
    return timing_data

def detect_time_control(game):
    """Detect game type from TimeControl header."""
    time_control = game.headers.get("TimeControl", "")

    if not time_control or time_control == "-":
        return "unknown"

    try:
        # Handle different time control formats
        if ":" in time_control:
            # Format: "40/7200:1800" (moves/seconds:seconds)
            first_part = time_control.split(":")[0]
            if "/" in first_part:
                base_time = int(first_part.split("/")[1])  # Get the seconds part
            else:
                base_time = int(first_part)
        elif "/" in time_control:
            # Format: "40/5400" (moves/seconds)
            base_time = int(time_control.split("/")[1])
        elif "+" in time_control:
            # Format: "180+2" (seconds+increment)
            base_time = int(time_control.split("+")[0])
        else:
            # Simple format: "180"
            base_time = int(time_control)

        if base_time <= 300:  # 5 minutes or less
            return "blitz"
        elif base_time <= 1800:  # 30 minutes or less
            return "rapid"
        else:
            return "classical"

    except (ValueError, IndexError):
        # If parsing fails, default to unknown
        return "unknown"

def compress_thinking_time(seconds, game_type):
    """Compress extreme time ranges to musical durations with logarithmic scaling."""
    if seconds <= 0:
        return 200  # Minimum duration

    # Different compression based on game type
    if game_type == "blitz":
        # Blitz: 0.1-120 seconds → 80-1200 ticks (more aggressive)
        if seconds <= 2:
            return int(seconds * 300)
        else:
            # Logarithmic compression for long blitz thinks
            compressed = 600 + math.log10(seconds) * 400
            return min(int(compressed), 1200)

    elif game_type == "rapid":
        # Rapid: 0.1-30 seconds → 100-2800 ticks
        if seconds <= 2:
            return int(seconds * 300)
        else:
            compressed = 600 + math.log10(seconds) * 600
            return min(int(compressed), 2800)

    else:  # classical
        # Classical: Much more aggressive compression - thinking windows are background texture only
        if seconds <= 2:
            return int(seconds * 200)  # 0.4-0.8 seconds
        elif seconds <= 60:
            return int(400 + (seconds - 2) * 50)  # 0.8-3.3 seconds max
        else:
            # Very short cap - just hint at the thinking, don't dominate
            return min(1800, int(400 + 58 * 50 + (seconds - 60) * 5))  # ~4 seconds max

def map_think_time_to_playback_seconds(think_seconds: float, cfg: dict) -> float:
    """Return musically useful compressed window seconds for a thinking duration.

    New model (bigger, more expressive):
        window = reference_window_seconds * curve_ratio( min(think, reference_seconds) / reference_seconds )
    curve options:
        - log: ratio = log1p(think)/log1p(reference_seconds) (smooth, slower start)
        - pow: ratio = (think/reference_seconds) ** pow_exponent (faster early growth if exponent < 1)

    Config (optional, under drone_modulation.windowed):
        reference_seconds (default 600)
        reference_window_seconds (default 15)
        max_window_seconds (default reference_window_seconds or larger)
        curve: 'log' or 'pow' (default 'log')
        pow_exponent: (default 0.6)
        min_trigger_seconds (default 1.0)
    Special rule:
        If think_seconds <= 1.0 we return the raw EMT (a 1:1 micro-window) instead of suppressing.
    """
    if think_seconds is None or think_seconds <= 1.0:
        return 0.0
    dm_cfg = cfg.get('drone_modulation', {})
    win_cfg = dm_cfg.get('windowed', {}) if isinstance(dm_cfg.get('windowed'), dict) else {}
    # If disabled, suppress ALL windowing including micro-windows
    if win_cfg and not win_cfg.get('enable', True):
        return 0.0
    # Micro window rule only applies when enabled
    if think_seconds <= 1.0:
        return think_seconds

    ref_seconds = float(win_cfg.get('reference_seconds', 600.0))
    ref_window = float(win_cfg.get('reference_window_seconds', 15.0))
    max_window = float(win_cfg.get('max_window_seconds', max(ref_window, 15.0)))
    curve = win_cfg.get('curve', 'log')
    pow_exp = float(win_cfg.get('pow_exponent', 0.6))
    min_trigger = float(win_cfg.get('min_trigger_seconds', 1.0))

    if think_seconds < min_trigger:
        return 0.0

    x = min(think_seconds, ref_seconds)
    if curve == 'pow':
        base_ratio = (x / ref_seconds) ** pow_exp if ref_seconds > 0 else 0.0
    else:  # log (default)
        denom = math.log1p(ref_seconds)
        base_ratio = (math.log1p(x) / denom) if denom > 0 else 0.0

    window = ref_window * base_ratio
    # Clamp so compressed window never exceeds raw EMT
    if window > think_seconds:
        window = think_seconds
    return min(max_window, window)

def classify_move_expression(thinking_time: float | None, san: str, nag: int | None, cfg: dict) -> tuple[str, str | None]:
    """Classify emotional/expression tag for upcoming move.

    Returns (tag, forced_phase) where tag is a short label and forced_phase is one of
    None | 'swell' | 'tension' | 'pulses'.
    Heuristics hierarchy:
      1. Direct NAG mapping (brilliant, blunder, etc.)
      2. SAN features (#, +, 'x')
      3. Long think thresholds
    """
    expr_cfg = cfg.get('expression_modulation', {}) if cfg else {}
    if not expr_cfg.get('enable', False):
        return 'none', None

    tag = 'default'
    forced_phase = None

    # NAG-based
    nag_map = {
        1: 'great',
        2: 'mistake',
        3: 'brilliant',
        4: 'blunder',
        5: 'interesting',
        6: 'inaccuracy',
        7: 'forced'
    }
    if nag in nag_map:
        tag = nag_map[nag]

    # SAN indicators (override or refine)
    if san:
        if '#' in san:
            tag = 'checkmate'
        elif '+' in san and tag in ('default', 'none'):
            tag = 'check'
        # A capture might add mild emphasis if still default
        if 'x' in san and tag in ('default', 'none'):
            tag = 'capture'

    # Long think emphasis if still mundane
    if thinking_time is not None and tag in ('default', 'none', 'capture'):
        long_think = expr_cfg.get('long_think_seconds', 120)
        epic_think = expr_cfg.get('epic_think_seconds', 600)
        if thinking_time >= epic_think:
            tag = 'epic'
        elif thinking_time >= long_think:
            tag = 'long'

    # Forced phases from config
    forced_phases = expr_cfg.get('forced_phases', {})
    if tag in forced_phases:
        forced_phase = forced_phases[tag]

    return tag, forced_phase

def classify_arpeggio_pattern(thinking_time: float | None, cfg: dict) -> dict | None:
    """Classify which arpeggio pattern to use based on thinking time.

    Returns the pattern configuration dict, or None if arpeggios disabled
    or thinking time doesn't warrant arpeggio activity.
    """
    if thinking_time is None or thinking_time <= 0:
        return None

    arp_cfg = cfg.get('thinking_arpeggios', {})
    if not arp_cfg.get('enable', False):
        return None

    patterns = arp_cfg.get('patterns', {})

    # Find the appropriate pattern based on time thresholds
    # Check in order: epic, long, medium, short, micro
    pattern_names = ['epic', 'long', 'medium', 'short', 'micro']

    for pattern_name in pattern_names:
        pattern_cfg = patterns.get(pattern_name, {})
        min_secs = pattern_cfg.get('min_seconds', 0)
        max_secs = pattern_cfg.get('max_seconds', float('inf'))

        if min_secs <= thinking_time < max_secs:
            # Return the pattern config with additional metadata
            return {
                'name': pattern_name,
                'pattern': pattern_cfg.get('pattern', [0, 4, 7]),
                'rate_ticks': pattern_cfg.get('rate_ticks', 120),
                'velocity_modifier': pattern_cfg.get('velocity_modifier', 0),
                'base_velocity': arp_cfg.get('velocity_base', 45)
            }

    # Fallback to micro pattern if no match (shouldn't happen with proper config)
    micro_cfg = patterns.get('micro', {})
    return {
        'name': 'micro',
        'pattern': micro_cfg.get('pattern', [0, 4, 7]),
        'rate_ticks': micro_cfg.get('rate_ticks', 120),
        'velocity_modifier': micro_cfg.get('velocity_modifier', 0),
        'base_velocity': arp_cfg.get('velocity_base', 45)
    }

def generate_thinking_arpeggios(arpeggio_track, thinking_time: float | None, compressed_ticks: int,
                               pattern_config: dict | None, eco_note: int, cfg: dict,
                               next_move_note: int | None = None, drone_phase: str = 'idle') -> int:
    """Generate arpeggio patterns during thinking time window.

    Returns the number of ticks consumed (should equal compressed_ticks).
    """
    if pattern_config is None or compressed_ticks <= 0:
        return 0

    arp_cfg = cfg.get('thinking_arpeggios', {})
    if not arp_cfg.get('enable', False):
        return 0

    channel = CHANNELS['arpeggio']
    pattern_degrees = pattern_config['pattern']
    rate_ticks = pattern_config['rate_ticks']
    base_velocity = pattern_config['base_velocity']
    velocity_modifier = pattern_config['velocity_modifier']

    # Apply drone phase modulations
    phase_mods = arp_cfg.get('phase_modifiers', {}).get(drone_phase, {})
    rate_multiplier = phase_mods.get('rate_multiplier', 1.0)
    velocity_boost = phase_mods.get('velocity_boost', 0)

    # Adjust rate based on phase
    effective_rate = max(10, int(rate_ticks * rate_multiplier))

    # Calculate how many notes we can fit in the compressed window
    note_count = max(1, compressed_ticks // effective_rate)

    # If harmonize_with_next_move is enabled, bias toward next move's harmony
    root_note = eco_note
    if arp_cfg.get('harmonize_with_next_move', False) and next_move_note is not None:
        # Subtle bias - blend 70% eco note, 30% next move note
        # This creates anticipatory harmony
        root_note = int(eco_note * 0.7 + next_move_note * 0.3)

    # Generate note pattern, snapping to scale
    scale = cfg.get('scale', [0, 2, 4, 5, 7, 9, 11])

    ticks_used = 0
    crescendo = arp_cfg.get('crescendo', False)

    # Set instrument program for arpeggio channel
    arp_instrument = arp_cfg.get('instrument', 46)
    if _LAST_PROGRAM.get(channel) != arp_instrument:
        arpeggio_track.append(mido.Message("program_change", program=arp_instrument, channel=channel, time=0))
        _LAST_PROGRAM[channel] = arp_instrument

    # Add pan setting from config
    arp_pan = cfg.get('panning', {}).get('arpeggio', 64)
    arpeggio_track.append(mido.Message("control_change", control=10, value=arp_pan, channel=channel, time=0))

    for i in range(note_count):
        # Cycle through pattern degrees
        degree_index = i % len(pattern_degrees)
        scale_degree = pattern_degrees[degree_index]

        # Calculate actual note
        candidate_note = root_note + scale_degree
        arp_note = snap_to_scale(candidate_note, scale)

        # Calculate velocity with crescendo
        velocity = base_velocity + velocity_modifier + velocity_boost
        if crescendo and note_count > 1:
            # Gradual crescendo from 80% to 120% of base velocity
            crescendo_factor = 0.8 + (0.4 * i / (note_count - 1))
            velocity = int(velocity * crescendo_factor)

        velocity = max(1, min(127, velocity))

        # Note duration - slightly shorter than interval for articulation
        note_duration = max(10, int(effective_rate * 0.8))

        # Add time delay for first note, subsequent notes are spaced by effective_rate
        time_delay = effective_rate if i > 0 else 0

        # Add note messages
        arpeggio_track.append(mido.Message("note_on", note=arp_note, velocity=velocity, channel=channel, time=time_delay))
        arpeggio_track.append(mido.Message("note_off", note=arp_note, velocity=0, channel=channel, time=note_duration))

        ticks_used += effective_rate

        # If we're close to filling the window, stop
        if ticks_used + effective_rate > compressed_ticks:
            break

    # Fill any remaining time with silence
    remaining_ticks = compressed_ticks - ticks_used
    if remaining_ticks > 0:
        # Add a dummy control message to advance time
        arpeggio_track.append(mido.Message("control_change", control=7, value=base_velocity, channel=channel, time=remaining_ticks))

    return compressed_ticks

def snap_to_scale(note, scale):
    """Snap note to the nearest pitch in the provided musical scale."""
    scale_notes = sorted(list(set(scale)))
    note_pitch_class = note % 12

    # Find the closest scale pitch class
    closest_pitch = min(scale_notes, key=lambda x: min(abs(note_pitch_class - x), 12 - abs(note_pitch_class - x)))

    # Calculate the difference and adjust the original note
    diff = closest_pitch - note_pitch_class
    # Handle wraparound case (e.g., B to C)
    if diff > 6:
        diff -= 12
    if diff < -6:
        diff += 12

    new_note = note + diff
    return min(new_note, 127)

def move_to_note(move, board, config):
    """
    Map a chess move to a MIDI note number based on the destination square.
    Supports multiple pitch mapping modes for different musical styles.
    """
    dest = move.uci()[2:]
    file, rank = dest[0], int(dest[1])

    # Flexible pitch mapping modes for better musicality
    mode = config.get('pitch_mode', 'legacy')  # 'legacy' keeps original spacing
    root = config.get('pitch_root', 60)        # MIDI note for file 'a' base (C4 default)

    if mode == 'legacy':
        base_pitch = config['pitch_mapping'][file]
        return base_pitch + (rank - 1) * 3

    elif mode == 'diatonic_2rank_octaves':
        # Files map to scale degrees, every 2 ranks = +1 octave (12 semitones) for clarity
        # a b c d e f g h  -> 0 2 4 5 7 9 11 (13)
        scale_degrees = [0, 2, 4, 5, 7, 9, 11, 12+1]  # Lift file h slightly into next octave
        file_index = ord(file) - ord('a')
        file_offset = scale_degrees[file_index]
        octave_offset = ((rank - 1) // 2) * 12  # Every 2 ranks a full octave
        intra_octave_rank = (rank - 1) % 2
        vertical_step = intra_octave_rank * 5    # Add perfect 4th within the pair for color
        note = root + file_offset + octave_offset + vertical_step
        return note

    elif mode == 'chromatic_linear':
        # Simple uniform grid: file steps by 2 semitones, rank steps by 3
        file_offset = (ord(file) - ord('a')) * 2
        rank_offset = (rank - 1) * 3
        return root + file_offset + rank_offset

    elif mode == 'major_files_octave_ranks':
        # Pure major scale across files, each rank is a full octave jump -> very consonant grid
        # Files: a b c d e f g h -> 1 2 3 4 5 6 7 8(root up)
        major_degrees = [0, 2, 4, 5, 7, 9, 11, 12]  # h-file lands on next octave tonic
        file_index = ord(file) - ord('a')
        file_offset = major_degrees[file_index]
        octave_offset = (rank - 1) * 12  # Each rank pushes an octave for vertical clarity
        return root + file_offset + octave_offset

    # Fallback
    base_pitch = config['pitch_mapping'][file]
    return base_pitch + (rank - 1) * 3

def get_phase_for_ply(ply: int) -> str:
    """Classify game phase based on ply number."""

    # Naive thresholds; could be refined by position analysis
    if ply < 14:
        return "opening"
    elif ply < 31:
        return "middlegame"
    return "endgame"

def modulate_drone_for_thinking(drone_track, thinking_time: float | None, ply: int, eco_note: int, config, base_velocity: int, channel=None, duck_other=None):
    """Inject modulation events into the sustained drone based on thinking time.

    Returns list of post-move (cleanup) messages (note_off / restore CC) to append after move note.
    Uses CC7 (volume) for broad changes; could be adapted to CC11 for finer control.
    """
    dm_cfg = config.get("drone_modulation", {})
    if not dm_cfg.get("enable", False):
        return []
    if thinking_time is None:
        return []

    swell_min = dm_cfg.get("swell_min_seconds", 30)
    tension_min = dm_cfg.get("tension_min_seconds", 120)
    pulse_min = dm_cfg.get("pulse_min_seconds", 600)

    if thinking_time < swell_min:
        return []

    post_move = []

    # Phase 1: Swell
    if swell_min <= thinking_time < tension_min:
        swell_amount = dm_cfg.get("swell_amount", 20)
        target = max(0, min(127, base_velocity + swell_amount))
        drone_track.append(mido.Message("control_change", control=7, value=target, channel=(channel if channel is not None else CHANNELS['drone']), time=0))
        post_move.append(mido.Message("control_change", control=7, value=base_velocity, channel=(channel if channel is not None else CHANNELS['drone']), time=0))
        if duck_other:
            # (pre duck already applied outside) schedule restore handled in outer logic
            pass
        return post_move

    # Phase 2: Tension harmony
    if tension_min <= thinking_time < pulse_min:
        tension_interval = dm_cfg.get("tension_interval", 6)
        tension_velocity = dm_cfg.get("tension_velocity", 60)
        tension_note = eco_note + tension_interval
        drone_track.append(mido.Message("note_on", note=tension_note, velocity=tension_velocity, channel=(channel if channel is not None else CHANNELS['drone']), time=0))
        post_move.append(mido.Message("note_off", note=tension_note, velocity=0, channel=(channel if channel is not None else CHANNELS['drone']), time=0))
        return post_move

    # Phase 3: Pulses
    if thinking_time >= pulse_min:
        pulse_count = dm_cfg.get("pulse_count", 4)
        pulse_rate = dm_cfg.get("pulse_rate_ticks", 240)
        pulse_drop = dm_cfg.get("pulse_drop", -25)
        pulse_rise = dm_cfg.get("pulse_rise", 10)
        low_val = max(0, min(127, base_velocity + pulse_drop))
        high_val = max(0, min(127, base_velocity + pulse_rise))
        for i in range(pulse_count):
            drone_track.append(mido.Message("control_change", control=7, value=low_val, channel=(channel if channel is not None else CHANNELS['drone']), time=(0 if i == 0 else pulse_rate)))
            drone_track.append(mido.Message("control_change", control=7, value=high_val, channel=(channel if channel is not None else CHANNELS['drone']), time=pulse_rate))
        post_move.append(mido.Message("control_change", control=7, value=base_velocity, channel=(channel if channel is not None else CHANNELS['drone']), time=0))
        return post_move

    return []

def schedule_drone_think_window(drone_track, compressed_ticks: int, think_seconds: float, cfg: dict,
                                eco_note: int, base_velocity: int, channel=None, override_phase: str | None = None):
    """Insert a pre-move drone-only window with modulation.

    override_phase allows expression layer to escalate to 'swell','tension','pulses'.
    Phase precedence still holds: pulses > tension > swell > idle. If only_escalate is set,
    we won't downgrade a naturally higher phase.
    """
    if compressed_ticks <= 0 or think_seconds <= 0:
        return 0
    dm_cfg = cfg.get('drone_modulation', {})
    ch = channel if channel is not None else CHANNELS['drone']
    swell_min = dm_cfg.get('swell_min_seconds', 30)
    tension_min = dm_cfg.get('tension_min_seconds', 120)
    pulse_min = dm_cfg.get('pulse_min_seconds', 600)
    swell_amount = dm_cfg.get('swell_amount', 20)
    tension_interval = dm_cfg.get('tension_interval', 6)
    tension_velocity = dm_cfg.get('tension_velocity', 60)
    pulse_drop = dm_cfg.get('pulse_drop', -25)
    pulse_rise = dm_cfg.get('pulse_rise', 10)
    pulse_count = max(1, dm_cfg.get('pulse_count', 4))
    expr_cfg = cfg.get('expression_modulation', {})
    only_escalate = expr_cfg.get('only_escalate', True)

    def clamp_cc(v):
        return max(0, min(127, v))

    # Determine natural phase
    natural_phase = 'idle'
    if think_seconds >= pulse_min:
        natural_phase = 'pulses'
    elif think_seconds >= tension_min:
        natural_phase = 'tension'
    elif think_seconds >= swell_min:
        natural_phase = 'swell'

    target_phase = natural_phase
    if override_phase:
        hierarchy = ['idle', 'swell', 'tension', 'pulses']
        if override_phase not in hierarchy:
            override_phase = None
        else:
            if only_escalate:
                if hierarchy.index(override_phase) > hierarchy.index(natural_phase):
                    target_phase = override_phase
            else:
                target_phase = override_phase

    # Pulses phase
    if target_phase == 'pulses':
        pair_span = compressed_ticks // pulse_count if pulse_count > 0 else compressed_ticks
        low_val = clamp_cc(base_velocity + pulse_drop)
        high_val = clamp_cc(base_velocity + pulse_rise)
        used = 0
        for _ in range(pulse_count):
            drone_track.append(mido.Message('control_change', control=7, value=low_val, channel=ch, time=(0 if used == 0 else 0)))
            half = max(1, pair_span // 2)
            drone_track.append(mido.Message('control_change', control=7, value=high_val, channel=ch, time=half))
            end_delta = pair_span - half
            drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=end_delta))
            used += pair_span
        tail = compressed_ticks - used
        if tail > 0:
            drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=tail))
        return compressed_ticks

    # Tension phase
    if target_phase == 'tension':
        swell_part = compressed_ticks * 20 // 100
        sustain_part = compressed_ticks * 60 // 100
        release_part = compressed_ticks - swell_part - sustain_part
        if swell_part > 0:
            mid_val = clamp_cc(base_velocity + swell_amount // 2)
            peak_val = clamp_cc(base_velocity + swell_amount)
            half = max(1, swell_part // 2)
            drone_track.append(mido.Message('control_change', control=7, value=mid_val, channel=ch, time=0))
            drone_track.append(mido.Message('control_change', control=7, value=peak_val, channel=ch, time=half))
            drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=swell_part - half))
        tension_note = eco_note + tension_interval
        drone_track.append(mido.Message('note_on', note=tension_note, velocity=tension_velocity, channel=ch, time=0))
        drone_track.append(mido.Message('note_off', note=tension_note, velocity=0, channel=ch, time=sustain_part))
        if release_part > 0:
            drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=release_part))
        return compressed_ticks

    # Swell phase
    if target_phase == 'swell':
        mid_val = clamp_cc(base_velocity + swell_amount // 2)
        peak_val = clamp_cc(base_velocity + swell_amount)
        third = max(1, compressed_ticks // 3)
        last = compressed_ticks - 2 * third
        drone_track.append(mido.Message('control_change', control=7, value=mid_val, channel=ch, time=0))
        drone_track.append(mido.Message('control_change', control=7, value=peak_val, channel=ch, time=third))
        drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=third))
        drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=last))
        return compressed_ticks

    # Idle
    drone_track.append(mido.Message('control_change', control=7, value=base_velocity, channel=ch, time=compressed_ticks))
    return compressed_ticks

def add_note(track, program, note, velocity, duration, pan=64, delay=0, channel=0, init_program=True):
    """Add a note with the delay applied to the NOTE ON (not swallowed by program/pan)."""
    original_delay = delay
    # Program change only if needed; never consume the musical delay here
    if init_program and (_LAST_PROGRAM.get(channel) != program):
        track.append(mido.Message("program_change", program=program, channel=channel, time=0))
        _LAST_PROGRAM[channel] = program
    # Reassert pan (0 delta)
    track.append(mido.Message("control_change", control=10, value=pan, channel=channel, time=0))
    # Apply the scheduled delay to the sounding event
    track.append(mido.Message("note_on", note=note, velocity=velocity, channel=channel, time=original_delay))
    track.append(mido.Message("note_off", note=note, velocity=velocity, channel=channel, time=duration))


def set_tempo(track, bpm):
    mpqn = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=mpqn, time=0))


def load_eco_map(tsv_dir="openings"):
    """Load ECO→Opening name mapping once from the Lichess TSV dataset."""
    eco_map = {}
    for part in "abcde":
        path = pathlib.Path(tsv_dir) / f"{part}.tsv"
        with open(path, newline="") as f:
            for eco, name, *_ in csv.reader(f, delimiter="\t"):
                eco_map[eco] = name
    return eco_map

def get_opening_name(game, eco_map):
    """Return the opening name from a python-chess Game object."""
    eco_code = game.headers.get("ECO")
    return eco_map.get(eco_code, "Unknown Opening")

def main():
    args = parse_args()
    pgn_file = args.pgn_file
    debug_pan = args.track_stats  # repurpose old flag
    config = load_config(args.config)

    with open(pgn_file) as pgn:
        game = chess.pgn.read_game(pgn)

    eco_map = load_eco_map("./openings")  # folder with a.tsv … e.tsv
    opening_name = get_opening_name(game, eco_map)
    game_type = detect_time_control(game)
    print(game.headers.get("ECO"), opening_name)
    print(f"Game type: {game_type} (TimeControl: {game.headers.get('TimeControl', 'unknown')})")

    # Preprocess all timing data
    timing_data = preprocess_timing_data(game)

    mid = mido.MidiFile()
    white_track = mido.MidiTrack()
    black_track = mido.MidiTrack()
    drone_track = mido.MidiTrack()
    arpeggio_track = mido.MidiTrack()
    mid.tracks.extend([white_track, black_track, drone_track, arpeggio_track])

    # Initialize track volumes to ensure they're not stuck at low levels
    # Channel-specific volumes
    white_track.append(mido.Message("control_change", channel=CHANNELS['white'], control=7, value=100, time=0))
    black_track.append(mido.Message("control_change", channel=CHANNELS['black'], control=7, value=100, time=0))

    # Initialize arpeggio track volume
    arp_base_velocity = config.get('thinking_arpeggios', {}).get('velocity_base', 45)
    arpeggio_track.append(mido.Message("control_change", channel=CHANNELS['arpeggio'], control=7, value=arp_base_velocity, time=0))

    # Initialize drone if enabled
    drone_base_note = None
    base_drone_velocity = None
    if config['drones']['enabled']:
        eco_code = game.headers.get("ECO", "A")
        eco_letter = eco_code[0] if eco_code else "A"
        drone_base_note = config['drones']['eco_mapping'].get(eco_letter, 36)
        drone_program = config['drones']['instrument']
        opening_phase_cfg = config['drones']['phases']['opening']
        base_drone_velocity = opening_phase_cfg['velocity']
        drone_track.append(mido.Message("program_change", program=drone_program, channel=CHANNELS['drone'], time=0))
        drone_track.append(mido.Message("control_change", control=10, value=64, channel=CHANNELS['drone'], time=0))
        # Set initial volume
        drone_track.append(mido.Message("control_change", control=7, value=base_drone_velocity, channel=CHANNELS['drone'], time=0))
        # Base drone note
        drone_track.append(mido.Message("note_on", note=drone_base_note, velocity=base_drone_velocity, channel=CHANNELS['drone'], time=0))
        for h in opening_phase_cfg.get("harmonies", []):
            drone_track.append(mido.Message("note_on", note=drone_base_note + h, velocity=max(0, base_drone_velocity - 10), channel=CHANNELS['drone'], time=0))

    board = game.board()
    node = game  # needed to access comments in sync with moves

    # Collect moves into list for lookahead scheduling
    moves_list = list(game.mainline_moves())
    total_plies = len(moves_list)

    # Timeline tracking (ticks). We treat each move's note as a discrete musical event spaced by a base_gap.
    PPQ = 480
    base_gap = 10  # gap after a move note before any thinking window starts
    global_time_ticks = 0  # conceptual main timeline (not strictly needed per track but for debug)
    # Maintain per-track elapsed time (ticks) for white/black move tracks
    track_times = {CHANNELS['white']: 0, CHANNELS['black']: 0}
    # Track current tempo (BPM) for tempo-aware scaling
    current_bpm = config['tempo']['opening']

    # Initialize channel program/pan once. Use config panning.
    pan_white = config.get('panning', {}).get('white', 64)
    pan_black = config.get('panning', {}).get('black', 64)
    # Aggregate timing stats
    total_emt_seconds = 0.0
    total_condensed_seconds = 0.0
    # Small think mode state
    small_think_active = False
    last_phase_velocity = None

    for idx, move in enumerate(moves_list, start=1):
        ply = idx
        node = node.variation(0)       # advance node alongside move
        comment = node.comment         # extract PGN comment (timestamps)

        # Reset arpeggio pattern for this move
        print_move_line._last_arpeggio_pattern = None

        base_note = move_to_note(move, board, config)
        raw_piece = board.piece_at(move.from_square)
        if raw_piece is None:
            # Fallback: skip this move gracefully
            print(f"[WARN] No piece at from-square for move {move.uci()} (ply {ply}) - skipping.")
            board.push(move)
            continue
        piece = raw_piece.symbol().upper()
        velocity = config['velocity'].get(piece, 80)

        instrument_config = config['instruments'].get(piece)
        if isinstance(instrument_config, dict):
            program = instrument_config.get('program', 0)
            octave_shift = instrument_config.get('octave_shift', 0)
            note = base_note + octave_shift  # Apply octave shift AFTER snap_to_scale
        else:
            program = instrument_config or 0
            note = base_note

        # Get preprocessed thinking time (EMT refers to time spent BEFORE this move)
        thinking_time = timing_data.get(ply)

        # Set duration for the actual move note
        move_distance = calculate_move_distance(move, board)
        duration = move_distance_to_duration(move_distance, piece.lower(), config)
        duration = int(duration * config["dynamic_durations"]["piece_modifiers"][piece.lower()])
        nag = extract_nag_code(comment)

        track = white_track if board.turn else black_track
        pan = pan_white if board.turn else pan_black

        if ply == 1:
            set_tempo(track, config['tempo']['opening'])
            current_bpm = config['tempo']['opening']
        elif ply == 11:
            set_tempo(track, config['tempo']['middlegame'])
            current_bpm = config['tempo']['middlegame']
        elif ply == 31:
            set_tempo(track, config['tempo']['endgame'])
            current_bpm = config['tempo']['endgame']

        # Debug output (unused variables suppressed purposely)
        # Insert compressed thinking window BEFORE any move notes so timeline reflects cognition
        think_window_ticks = 0
        if thinking_time and config.get('drone_modulation', {}).get('enable') and config['drones']['enabled']:
            current_phase = get_phase_for_ply(ply)
            dm_cfg_local = config.get('drone_modulation', {})
            small_cfg = dm_cfg_local.get('small_think', {})
            small_threshold = small_cfg.get('max_seconds', 2.0)
            flat_offset = small_cfg.get('flat_velocity_offset', -15)
            # Determine base velocity for current phase
            phase_cfg_base = config['drones']['phases'].get(current_phase, {})
            phase_velocity_base = phase_cfg_base.get('velocity', base_drone_velocity or 90)
            # Special case: very small think (<=1s) -> create an idle window equal to EMT (no compression)
            if thinking_time <= 1.0:
                compressed_secs = thinking_time  # one-to-one mapping
                expr_tag, forced_phase = classify_move_expression(thinking_time, board.san(move) if board.is_legal(move) else move.uci(), nag, config)
                print_move_line._last_expr = expr_tag
                # Totals
                total_emt_seconds += thinking_time
                total_condensed_seconds += compressed_secs
                # Schedule window (idle baseline – do not escalate even if forced_phase suggests; keep it subtle)
                ticks_per_second = PPQ * current_bpm / 60.0
                think_window_ticks = int(compressed_secs * ticks_per_second)
                # Use idle by passing think_seconds=0.5 (below swell_min) but we still want duration, so call with original
                schedule_drone_think_window(
                    drone_track=drone_track,
                    compressed_ticks=think_window_ticks,
                    think_seconds=thinking_time,  # natural phase logic will resolve to 'idle'
                    cfg=config,
                    eco_note=drone_base_note or 36,
                    base_velocity=phase_velocity_base,
                    channel=CHANNELS['drone'],
                    override_phase=None  # force idle even if expression forces escalation for dramatic moves
                )

                # Generate arpeggios during micro think window
                pattern_config = classify_arpeggio_pattern(thinking_time, config)
                if pattern_config:
                    print_move_line._last_arpeggio_pattern = pattern_config.get('name', 'none')
                    next_move_note = move_to_note(move, board, config) if len(moves_list) > idx else None
                    generate_thinking_arpeggios(
                        arpeggio_track=arpeggio_track,
                        thinking_time=thinking_time,
                        compressed_ticks=think_window_ticks,
                        pattern_config=pattern_config,
                        eco_note=drone_base_note or 36,
                        cfg=config,
                        next_move_note=next_move_note,
                        drone_phase='idle'  # micro thinks always use idle phase
                    )

                global_time_ticks += think_window_ticks
            # Small think gating ( >1s and <= threshold )
            elif thinking_time <= small_threshold:
                # Enter small think mode: ensure drone CC7 set to flat value once
                flat_vel = max(0, min(127, phase_velocity_base + flat_offset))
                if not small_think_active:
                    drone_track.append(mido.Message('control_change', control=7, value=flat_vel, channel=CHANNELS['drone'], time=0))
                    small_think_active = True
                    last_phase_velocity = phase_velocity_base
                # Do NOT allocate window or accumulate condensed seconds
                total_emt_seconds += thinking_time or 0.0
                # classify expression for logging but skip window scheduling
                expr_tag, forced_phase = classify_move_expression(thinking_time, board.san(move) if board.is_legal(move) else move.uci(), nag, config)
                print_move_line._last_expr = expr_tag

                # For small thinks, check if arpeggios are enabled independently
                pattern_config = classify_arpeggio_pattern(thinking_time, config)
                if pattern_config:
                    print_move_line._last_arpeggio_pattern = pattern_config.get('name', 'none')
                    # Create a small window for arpeggios even if drone doesn't get one
                    compressed_secs = thinking_time * 0.5  # Compress to 50% for small thinks
                    ticks_per_second = PPQ * current_bpm / 60.0
                    small_arp_ticks = max(480, int(compressed_secs * ticks_per_second))  # Minimum 1 second

                    next_move_note = move_to_note(move, board, config) if len(moves_list) > idx else None
                    generate_thinking_arpeggios(
                        arpeggio_track=arpeggio_track,
                        thinking_time=thinking_time,
                        compressed_ticks=small_arp_ticks,
                        pattern_config=pattern_config,
                        eco_note=drone_base_note or 36,
                        cfg=config,
                        next_move_note=next_move_note,
                        drone_phase='idle'  # small thinks use idle
                    )
                    # Add small window to global timeline
                    global_time_ticks += small_arp_ticks
                    think_window_ticks = small_arp_ticks  # For logging consistency
            else:
                # If we were in small think mode and now leaving, restore phase velocity
                if small_think_active:
                    restore_vel = phase_velocity_base
                    drone_track.append(mido.Message('control_change', control=7, value=restore_vel, channel=CHANNELS['drone'], time=0))
                    small_think_active = False
                compressed_secs = map_think_time_to_playback_seconds(thinking_time, config)
                # Expression-aware scaling
                expr_tag, forced_phase = classify_move_expression(thinking_time, board.san(move) if board.is_legal(move) else move.uci(), nag, config)
                print_move_line._last_expr = expr_tag  # stash for logging function
                expr_cfg = config.get('expression_modulation', {})
                if expr_cfg.get('enable', False):
                    mults = expr_cfg.get('window_multipliers', {})
                    factor = mults.get(expr_tag, mults.get('default', 1.0))
                    compressed_secs *= factor
                # Accumulate totals
                total_emt_seconds += thinking_time or 0.0
                total_condensed_seconds += compressed_secs
                if compressed_secs > 0:
                    ticks_per_second = PPQ * current_bpm / 60.0
                    think_window_ticks = int(compressed_secs * ticks_per_second)
                    phase_cfg = config['drones']['phases'].get(current_phase, {})
                    phase_velocity = phase_cfg.get('velocity', base_drone_velocity or 90)
                    schedule_drone_think_window(
                        drone_track=drone_track,
                        compressed_ticks=think_window_ticks,
                        think_seconds=thinking_time,
                        cfg=config,
                        eco_note=drone_base_note or 36,
                        base_velocity=phase_velocity,
                        channel=CHANNELS['drone'],
                        override_phase=forced_phase
                    )

                    # Generate arpeggios during thinking window - determine drone phase for interaction
                    pattern_config = classify_arpeggio_pattern(thinking_time, config)
                    if pattern_config:
                        print_move_line._last_arpeggio_pattern = pattern_config.get('name', 'none')
                        # Determine the actual drone phase being used
                        dm_cfg = config.get("drone_modulation", {})
                        swell_min = dm_cfg.get("swell_min_seconds", 30)
                        tension_min = dm_cfg.get("tension_min_seconds", 120)
                        pulse_min = dm_cfg.get("pulse_min_seconds", 600)

                        active_drone_phase = 'idle'
                        if thinking_time >= pulse_min:
                            active_drone_phase = 'pulses'
                        elif thinking_time >= tension_min:
                            active_drone_phase = 'tension'
                        elif thinking_time >= swell_min:
                            active_drone_phase = 'swell'

                        # Use forced_phase if it escalates beyond natural phase
                        if forced_phase:
                            hierarchy = ['idle', 'swell', 'tension', 'pulses']
                            if (forced_phase in hierarchy and active_drone_phase in hierarchy and
                                hierarchy.index(forced_phase) > hierarchy.index(active_drone_phase)):
                                active_drone_phase = forced_phase

                        next_move_note = move_to_note(move, board, config) if len(moves_list) > idx else None
                        generate_thinking_arpeggios(
                            arpeggio_track=arpeggio_track,
                            thinking_time=thinking_time,
                            compressed_ticks=think_window_ticks,
                            pattern_config=pattern_config,
                            eco_note=drone_base_note or 36,
                            cfg=config,
                            next_move_note=next_move_note,
                            drone_phase=active_drone_phase
                        )

                    global_time_ticks += think_window_ticks
        else:
            # still classify so column isn't blank
            expr_tag, _ = classify_move_expression(thinking_time, board.san(move) if thinking_time else '', nag, config)
            print_move_line._last_expr = expr_tag
            if thinking_time is not None:
                total_emt_seconds += thinking_time

        # No legacy immediate modulation / cleanup anymore
        post_drone_msgs = []

        # Schedule move immediately after any drone window
        move_start_tick = global_time_ticks
        current_channel = CHANNELS['white'] if board.turn else CHANNELS['black']
        # Use per-track accumulated time to compute the delta
        last_channel_time = track_times[current_channel]
        delay_for_move = max(0, move_start_tick - last_channel_time)
        add_note(track, program, note, velocity, duration, pan, delay_for_move, channel=current_channel)
        track_times[current_channel] = last_channel_time + delay_for_move + duration

        # (No deferred cleanup needed)
        if not args.no_move_log:
            compressed_window = None
            if thinking_time is not None:
                try:
                    compressed_window = map_think_time_to_playback_seconds(thinking_time, config)
                except Exception:
                    compressed_window = None
            # Get the arpeggio pattern that was used (if any)
            arpeggio_pattern = getattr(print_move_line, "_last_arpeggio_pattern", None)
            print_move_line(
                ply=ply,
                side='White' if board.turn else 'Black',
                san=board.san(move) if board.is_legal(move) else move.uci(),
                piece=piece,
                note=note,
                program=program,
                velocity=velocity,
                duration=duration,
                start_tick=move_start_tick,
                thinking_time=thinking_time,
                compressed_window=compressed_window,
                arpeggio_pattern=arpeggio_pattern
            )
        MOVE_EVENTS.append({
            'ply': ply,
            'side': 'White' if board.turn else 'Black',
            'note': note,
            'start_tick': move_start_tick,
            'duration': duration,
            'program': program,
            'piece': piece,
            'channel': current_channel
        })

        if nag in config['effects']['nag']:
            effect = config['effects']['nag'][nag]
            effect_program = effect.get('instrument', program)
            raw_effect_note = note + effect.get('octave_shift', 0) + effect.get('pitch_shift', 0)
            effect_note = snap_to_scale(raw_effect_note, config['scale'])
            effect_velocity = effect.get('velocity', velocity + effect.get('velocity_boost', 0) + effect.get('velocity_change', 0))
            effect_duration = int(duration * effect.get('duration_multiplier', effect.get('duration_ratio', 1)))
            effect_delay = effect.get('delay', 0)
            add_note(track, effect_program, effect_note, effect_velocity, effect_duration, pan, effect_delay, channel=CHANNELS['effects'])

        if board.is_capture(move):
            capture_effect = config['effects']['capture']
            raw_capture_note = note + capture_effect['pitch_shift']
            capture_note = snap_to_scale(raw_capture_note, config['scale'])
            capture_velocity = velocity + capture_effect['velocity_change']
            add_note(track, program, capture_note, capture_velocity, duration, pan, capture_effect['delay'], channel=current_channel, init_program=False)

        board.push(move)

        if board.is_checkmate():
            checkmate_effect = config['effects']['checkmate']
            for n in checkmate_effect['chord']:
                checkmate_note = snap_to_scale(n, config['scale'])
                add_note(track, program, checkmate_note, checkmate_effect['velocity'], checkmate_effect['duration'], pan, channel=current_channel, init_program=False)
        elif board.is_check():
            check_effect = config['effects']['check']
            raw_check_note = note + check_effect['pitch_shift']
            check_note = snap_to_scale(raw_check_note, config['scale'])
            add_note(track, program, check_note, check_effect['velocity'], duration, pan, channel=current_channel, init_program=False)

        move_end_tick = move_start_tick + duration
        # Advance global timeline (shared) only if this move pushes it forward beyond existing previous move timeline
        candidate_global = move_end_tick + base_gap
        if candidate_global > global_time_ticks:
            global_time_ticks = candidate_global


    out_name = args.output if args.output else (pgn_file.rsplit(".", 1)[0] + "_music.mid")
    mid.save(out_name)
    print(f"Saved MIDI to {out_name}")
    if args.play:
        try:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', out_name])
            elif sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', out_name])
            elif sys.platform.startswith('win'):
                os.startfile(out_name)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[WARN] Could not auto-play MIDI: {e}")

    if args.sheet:
        print("\n=== ASCII MOVE SHEET (first measures) ===")
        for line in build_ascii_sheet(MOVE_EVENTS, PPQ):
            print(line)
        print("========================================")

    if args.track_stats:
        print("\n=== TRACK STATISTICS ===")
        counts = defaultdict(int)
        for ev in MOVE_EVENTS:
            counts[ev['side']] += 1
        for side, cnt in counts.items():
            print(f" {side}: {cnt} move notes")
        print(f" Total move notes: {len(MOVE_EVENTS)}")
        print("========================")

    if debug_pan:
        print("=== MIDI CHANNEL / PAN ANALYSIS ===")
        try:
            mf = mido.MidiFile(out_name)
            note_counts = {}
            pans = {}
            for tr in mf.tracks:
                for msg in tr:
                    if msg.type == 'control_change' and msg.control == 10:
                        pans[msg.channel] = msg.value
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_counts[msg.channel] = note_counts.get(msg.channel, 0) + 1
            for ch in sorted(note_counts.keys()):
                print(f"Channel {ch}: notes={note_counts[ch]} pan={pans.get(ch,'?')}")
        except Exception as e:
            print("MIDI analysis failed:", e)

    # Final timing compression summary
    if total_emt_seconds > 0:
        ratio = (total_condensed_seconds / total_emt_seconds) if total_emt_seconds else 0
        def _fmt_hms(sec: float) -> str:
            h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        print("\n=== TIME COMPRESSION SUMMARY ===")
        print(f" Total EMT:        {total_emt_seconds:10.2f} s  ({_fmt_hms(total_emt_seconds)})")
        print(f" Condensed Window: {total_condensed_seconds:10.2f} s  ({_fmt_hms(total_condensed_seconds)})")
        print(f" Compression Ratio: {ratio:.3f} (condensed / real)")
        if total_condensed_seconds > 0:
            print(f" Average Scaling:   {total_emt_seconds/total_condensed_seconds:.3f}x real->musical")
        print("================================")

if __name__ == "__main__":
    main()

