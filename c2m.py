#!/usr/bin/env python3
import sys
import chess.pgn
import mido
import re
import csv
import pathlib
import yaml
import math

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

def preprocess_timing_data(game):
    """Preprocess all timing data before main loop."""
    timing_data = {}  # ply_number -> thinking_time_seconds

    white_prev_clock = None
    black_prev_clock = None

    node = game
    for ply, move in enumerate(game.mainline_moves(), start=1):
        node = node.variation(0)
        comment = node.comment

        # Try direct timestamp first
        thinking_time = extract_timestamp(comment)

        if thinking_time is None:
            # Try clock calculation
            current_clock = extract_clock_time(comment)
            if current_clock is not None:
                is_white_move = (ply % 2 == 1)

                if is_white_move:
                    if white_prev_clock is not None:
                        thinking_time = white_prev_clock - current_clock
                    white_prev_clock = current_clock
                else:
                    if black_prev_clock is not None:
                        thinking_time = black_prev_clock - current_clock
                    black_prev_clock = current_clock

        if thinking_time is not None and thinking_time > 0:
            timing_data[ply] = thinking_time

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
        # Classical: preserve timing ratios but cap for musical coherence
        if seconds <= 2:
            return int(seconds * 400)  # 0.8-1.6 seconds
        elif seconds <= 30:
            return int(800 + (seconds - 2) * 200)  # 1.6-7.2 seconds
        else:
            # Cap at ~8 seconds, drone handles the drama
            return min(3600, int(800 + 28 * 200 + (seconds - 30) * 20))

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
    dest = move.uci()[2:]
    file, rank = dest[0], int(dest[1])
    base_pitch = config['pitch_mapping'][file]
    note = base_pitch + (rank - 1) * 3
    return snap_to_scale(note, config['scale'])

def add_note(track, program, note, velocity, duration, pan=64, delay=0):
    track.append(mido.Message("program_change", program=program, time=delay))
    track.append(mido.Message("control_change", control=10, value=pan, time=0))
    track.append(mido.Message("note_on", note=note, velocity=velocity, time=0))
    track.append(mido.Message("note_off", note=note, velocity=velocity, time=duration))


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
    if len(sys.argv) < 2:
        print("Usage: python3 c2m.py <game.pgn>")
        sys.exit(1)

    pgn_file = sys.argv[1]
    config = load_config()

    with open(pgn_file) as pgn:
        game = chess.pgn.read_game(pgn)

    eco_map = load_eco_map("./openings")  # folder with a.tsv … e.tsv
    opening_name = get_opening_name(game, eco_map)
    game_type = detect_time_control(game)
    print(game.headers.get("ECO"), opening_name)
    print(f"Game type: {game_type} (TimeControl: {game.headers.get('TimeControl', 'unknown')})")

    # Preprocess all timing data
    timing_data = preprocess_timing_data(game)

    # Debug timing data
    print("=== TIMING DEBUG ===")
    for ply in sorted(timing_data.keys())[:10]:  # Show first 10 moves
        print(f"Ply {ply}: {timing_data[ply]} seconds thinking time")
    print("====================")

    mid = mido.MidiFile()
    white_track = mido.MidiTrack()
    black_track = mido.MidiTrack()
    drone_track = mido.MidiTrack()
    mid.tracks.extend([white_track, black_track, drone_track])

    # Initialize drone if enabled
    if config['drones']['enabled']:
        eco_code = game.headers.get("ECO", "A")
        eco_letter = eco_code[0] if eco_code else "A"
        drone_note = config['drones']['eco_mapping'].get(eco_letter, 36)
        drone_program = config['drones']['instrument']

        # Start opening drone
        drone_track.append(mido.Message("program_change", program=drone_program, time=0))
        drone_track.append(mido.Message("note_on", note=drone_note, velocity=config['drones']['phases']['opening']['velocity'], time=0))

    board = game.board()
    node = game  # needed to access comments in sync with moves

    # Track timing for staggered notes

    for ply, move in enumerate(game.mainline_moves(), start=1):
        node = node.variation(0)       # advance node alongside move
        comment = node.comment         # extract PGN comment (timestamps)

        base_note = move_to_note(move, board, config)
        piece = board.piece_at(move.from_square).symbol().upper()
        velocity = config['velocity'].get(piece, 80)

        instrument_config = config['instruments'].get(piece)
        if isinstance(instrument_config, dict):
            program = instrument_config.get('program', 0)
            octave_shift = instrument_config.get('octave_shift', 0)
            note = base_note + octave_shift  # Apply octave shift AFTER snap_to_scale
        else:
            program = instrument_config or 0
            note = base_note

        # Get preprocessed thinking time
        thinking_time = timing_data.get(ply)

        # Set duration based on thinking time
        if thinking_time is not None:
            duration = compress_thinking_time(thinking_time, game_type)
        else:
            duration = config['durations']['pawn_default'] if piece == "P" else config['durations']['piece_default']

        nag = extract_nag_code(comment)

        track = white_track if board.turn else black_track
        pan = config['panning']['white'] if board.turn else config['panning']['black']

        if ply == 1:
            set_tempo(track, config['tempo']['opening'])
        elif ply == 11:
            set_tempo(track, config['tempo']['middlegame'])
        elif ply == 31:
            set_tempo(track, config['tempo']['endgame'])

        # Debug output
        note_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][note % 12]
        octave = note // 12 - 1
        print(f"Ply {ply}: {piece} {move.uci()} -> Note {note} ({note_name}{octave}) Program {program} Vel {velocity} Dur {duration} ticks")

        # Use small stagger delay for main note (not absolute time)
        stagger_delay = 60 if ply > 1 else 0  # First note has no delay
        add_note(track, program, note, velocity, duration, pan, stagger_delay)

        if nag in config['effects']['nag']:
            effect = config['effects']['nag'][nag]
            effect_program = effect.get('instrument', program)
            raw_effect_note = note + effect.get('octave_shift', 0) + effect.get('pitch_shift', 0)
            effect_note = snap_to_scale(raw_effect_note, config['scale'])
            effect_velocity = effect.get('velocity', velocity + effect.get('velocity_boost', 0) + effect.get('velocity_change', 0))
            effect_duration = int(duration * effect.get('duration_multiplier', effect.get('duration_ratio', 1)))
            effect_delay = effect.get('delay', 0)
            add_note(track, effect_program, effect_note, effect_velocity, effect_duration, pan, effect_delay)


        if board.is_capture(move):
            capture_effect = config['effects']['capture']
            raw_capture_note = note + capture_effect['pitch_shift']
            capture_note = snap_to_scale(raw_capture_note, config['scale'])
            capture_velocity = velocity + capture_effect['velocity_change']
            add_note(track, program, capture_note, capture_velocity, duration, pan, capture_effect['delay'])

        board.push(move)
        if board.is_checkmate():
            checkmate_effect = config['effects']['checkmate']
            for n in checkmate_effect['chord']:
                checkmate_note = snap_to_scale(n, config['scale'])
                add_note(track, program, checkmate_note, checkmate_effect['velocity'], checkmate_effect['duration'], pan)
        elif board.is_check():
            check_effect = config['effects']['check']
            raw_check_note = note + check_effect['pitch_shift']
            check_note = snap_to_scale(raw_check_note, config['scale'])
            add_note(track, program, check_note, check_effect['velocity'], duration, pan)

    out_name = pgn_file.rsplit(".", 1)[0] + "_music.mid"
    mid.save(out_name)
    print(f"Saved MIDI to {out_name}")

if __name__ == "__main__":
    main()

