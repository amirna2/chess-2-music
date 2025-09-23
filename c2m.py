#!/usr/bin/env python3
import sys
import chess.pgn
import mido
import re
import csv
import pathlib
import yaml
import math

# MIDI channel assignments (0-15)
CHANNELS = {
    'white': 0,
    'black': 1,
    'drone': 2,
    'arpeggio': 3,
    'effects': 4,  # optional separate channel for NAG/capture/check accents
}

# Track which channels have had their program/pan initialized so we don't spam events
_CHANNEL_INITIALIZED = set()

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

def add_note(track, program, note, velocity, duration, pan=64, delay=0, channel=0, init_program=True):
    """Add a note with optional initial program/pan setup per channel.

    program changes are only inserted the first time we see a channel (or if init_program=True explicitly).
    Pan is also only set once per channel to avoid overwriting spatial field repeatedly.
    """
    if channel not in _CHANNEL_INITIALIZED:
        # First event on this channel: apply delay here so timing stays correct
        if init_program:
            track.append(mido.Message("program_change", program=program, channel=channel, time=delay))
            delay = 0  # consumed
        track.append(mido.Message("control_change", control=10, value=pan, channel=channel, time=0))
        _CHANNEL_INITIALIZED.add(channel)
    else:
        # Subsequent note: if caller supplied delay, use on first note_on
        if init_program:
            track.append(mido.Message("program_change", program=program, channel=channel, time=delay))
            delay = 0
    # Note on/off
    track.append(mido.Message("note_on", note=note, velocity=velocity, channel=channel, time=delay))
    track.append(mido.Message("note_off", note=note, velocity=velocity, channel=channel, time=duration))

def add_arpeggio_note(track, note, velocity, duration, pan=64, delay=0, channel=CHANNELS['arpeggio']):
    """Add arpeggio note (channel-specific)."""
    track.append(mido.Message("note_on", note=note, velocity=velocity, channel=channel, time=delay))
    track.append(mido.Message("note_off", note=note, velocity=0, channel=channel, time=duration))

def add_arpeggio_fadeout(track, program, pan=64):
    """Add graceful fadeout when arpeggio is interrupted."""
    # Let arpeggio naturally complete - don't use volume control changes
    # as they persist and affect all subsequent notes on the track
    pass

def calculate_arpeggio_duration(thinking_time, game_type):
    """Calculate arpeggio duration based on pure cognitive complexity, not game type."""
    MAX_ARPEGGIO_DURATION = 30  # Much shorter cap - arpeggios are musical ornaments

    # Pure cognitive time mapping - regardless of game format
    if thinking_time <= 90:
        duration = thinking_time * 0.3  # 30% of thinking time
    elif thinking_time <= 300:  # 1.5-5 minutes
        duration = 27 + (thinking_time - 90) * 0.1  # Gentle scaling
    else:  # 5+ minutes (epic thinks)
        duration = 48 + (thinking_time - 300) * 0.05  # Very gentle for longest thinks

    return min(duration, MAX_ARPEGGIO_DURATION)

def generate_thinking_arpeggio(
    track,
    thinking_time_seconds_raw,
    piece_about_to_move,
    source_square_note,
    target_square_note,
    config,
    program,
    game_type,
    pan=64
):
    """Generate musical phrase during thinking time that builds toward the move."""
    # Only consider sufficiently long thinks
    if thinking_time_seconds_raw < 10:
        return

    # Compress raw cognitive time to playable musical duration (seconds)
    compressed_duration_seconds = calculate_arpeggio_duration(thinking_time_seconds_raw, game_type)
    print(f"    Arpeggio: raw {thinking_time_seconds_raw:.1f}s -> compressed {compressed_duration_seconds:.2f}s")

    # Convert to MIDI ticks (assumes 480 ppq)
    total_ticks = int(compressed_duration_seconds * 480)

    # Determine phrase structure based on RAW thinking duration (musical narrative tied to real time spent)
    if thinking_time_seconds_raw < 60:       # 10-60 seconds: Simple arpeggio
        phase_structure = ["contemplative"]
    elif thinking_time_seconds_raw < 300:   # 1-5 minutes: Two-phase buildup
        phase_structure = ["contemplative", "building"]
    elif thinking_time_seconds_raw < 900:   # 5-15 minutes: Three-phase development
        phase_structure = ["contemplative", "building", "urgent"]
    else:                                    # 15+ minutes: Full orchestral buildup
        phase_structure = ["contemplative", "building", "urgent", "climactic"]

    ticks_per_phase = total_ticks // len(phase_structure)

    # Build harmonic progression from source to target
    harmonic_steps = create_harmonic_progression(source_square_note, target_square_note, len(phase_structure) * 4)

    current_time = 0
    step_index = 0

    for phase_num, phase in enumerate(phase_structure):
        phase_ticks = ticks_per_phase

        # Phase characteristics
        if phase == "contemplative":
            note_duration = 400  # Long notes
            velocity_base = 30
            notes_per_phase = 4
        elif phase == "building":
            note_duration = 240  # Medium notes
            velocity_base = 45
            notes_per_phase = 6
        elif phase == "urgent":
            note_duration = 120  # Short notes
            velocity_base = 60
            notes_per_phase = 8
        else:  # climactic
            note_duration = 60   # Very short notes
            velocity_base = 75
            notes_per_phase = 12

        # Generate notes for this phase
        ticks_between_notes = phase_ticks // notes_per_phase

        for note_in_phase in range(notes_per_phase):
            if step_index < len(harmonic_steps):
                note = harmonic_steps[step_index]
                note = snap_to_scale(note, config['scale'])

                # Gradual velocity increase within phase
                velocity_progression = note_in_phase / notes_per_phase
                velocity = int(velocity_base + velocity_progression * 15)
                velocity = min(velocity, 80)  # Cap to avoid overwhelming

                # Add the note - proper MIDI timing
                if note_in_phase == 0 and phase_num == 0:
                    # First note of arpeggio - small initial delay
                    delay = 30
                else:
                    # Subsequent notes - spacing between notes
                    delay = ticks_between_notes
                actual_duration = min(note_duration, ticks_between_notes - 20) if ticks_between_notes > 20 else 20
                add_arpeggio_note(track, note, velocity, actual_duration, pan, delay)

                current_time += ticks_between_notes
                step_index += 1

    return total_ticks  # Inform caller how many ticks were consumed by arpeggio

def create_harmonic_progression(source_note, target_note, num_steps):
    """Create harmonic progression from source to target note."""
    progression = []

    # Start with source note and its harmonics
    progression.append(source_note)
    progression.append(source_note + 4)  # Major third
    progression.append(source_note + 7)  # Perfect fifth

    # Calculate interval to target
    interval = target_note - source_note

    # Create smooth progression toward target
    for i in range(4, num_steps - 3):
        # Interpolate between source harmony and target
        progress = i / (num_steps - 1)

        # Mix of harmonic and melodic movement
        if i % 3 == 0:
            note = source_note + int(interval * progress)  # Direct melodic line
        elif i % 3 == 1:
            note = source_note + 4 + int(interval * progress * 0.7)  # Third harmony
        else:
            note = source_note + 7 + int(interval * progress * 0.5)  # Fifth harmony

        progression.append(note)

    # End with target note approach
    progression.append(target_note + 7)  # Dominant approach
    progression.append(target_note + 2)  # Leading tone
    progression.append(target_note)      # Target resolution

    return progression


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
    arpeggio_track = mido.MidiTrack()
    mid.tracks.extend([white_track, black_track, drone_track, arpeggio_track])

    # Initialize track volumes to ensure they're not stuck at low levels
    white_track.append(mido.Message("control_change", control=7, value=100, time=0))  # Set main volume
    black_track.append(mido.Message("control_change", control=7, value=100, time=0))  # Set main volume

    # Initialize drone if enabled
    if config['drones']['enabled']:
        eco_code = game.headers.get("ECO", "A")
        eco_letter = eco_code[0] if eco_code else "A"
        drone_note = config['drones']['eco_mapping'].get(eco_letter, 36)
        drone_program = config['drones']['instrument']

        # Start opening drone
    drone_track.append(mido.Message("program_change", program=drone_program, channel=CHANNELS['drone'], time=0))
    drone_track.append(mido.Message("control_change", control=10, value=64, channel=CHANNELS['drone'], time=0))
    drone_track.append(mido.Message("note_on", note=drone_note, velocity=config['drones']['phases']['opening']['velocity'], channel=CHANNELS['drone'], time=0))

    board = game.board()
    node = game  # needed to access comments in sync with moves

    # (Arpeggio timing now managed locally per move; no cross-player overlap control yet)

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

        # Set duration for the actual move note
        if thinking_time is not None:
            # For moves with arpeggios, use shorter final note
            duration = 240 if thinking_time >= 10 else compress_thinking_time(thinking_time, game_type)
        else:
            # Ensure first move has substantial duration to start the music
            if ply == 1:
                duration = 960  # Longer first move
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

        # Generate thinking arpeggio BEFORE this move if there was deep thinking (threshold 25s raw EMT)
        if thinking_time is not None and thinking_time >= 25:
            print(f"*** GENERATING ARPEGGIO BEFORE Ply {ply}: raw EMT {thinking_time:.1f} seconds ***")

            # Derive source square note (piece starting square) distinct from destination mapping
            from_square = move.from_square
            from_file = chess.FILE_NAMES[chess.square_file(from_square)]
            from_rank = chess.square_rank(from_square) + 1
            source_base_pitch = config['pitch_mapping'][from_file] + (from_rank - 1) * 3
            source_square_note = snap_to_scale(source_base_pitch, config['scale'])

            arpeggio_ticks = generate_thinking_arpeggio(
                track=arpeggio_track,
                thinking_time_seconds_raw=thinking_time,
                piece_about_to_move=piece,
                source_square_note=source_square_note,
                target_square_note=base_note,
                config=config,
                program=program,
                game_type=game_type,
                pan=pan
            ) or 0
            # Move note will follow shortly regardless of arpeggio length (separate track already carries buildup)
            stagger_delay = 60  # constant small gap
        else:
            # No arpeggio - normal small delay
            stagger_delay = 60 if ply > 1 else 0

        current_channel = CHANNELS['white'] if board.turn else CHANNELS['black']
        add_note(track, program, note, velocity, duration, pan, stagger_delay, channel=current_channel)

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


    out_name = pgn_file.rsplit(".", 1)[0] + "_music.mid"
    mid.save(out_name)
    print(f"Saved MIDI to {out_name}")

if __name__ == "__main__":
    main()

