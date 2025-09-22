#!/usr/bin/env python3
import sys
import chess.pgn
import mido
import re
import csv
import pathlib
import yaml
import argparse

def load_config(config_file="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def extract_nag_code(comment):
    # Look for NAG codes like $1, $2, $4, $6, etc.
    nag_match = re.search(r'\$(\d+)', comment)
    return int(nag_match.group(1)) if nag_match else None

def extract_timestamp(comment):
    m = re.search(r"\[%timestamp (\d+)\]", comment)
    if m:
        return int(m.group(1))
    return None

def snap_to_scale(note, scale):
    while note % 12 not in set(scale):
        note += 1
    return min(note, 127)

def move_to_note(move, board, config):
    dest = move.uci()[2:]
    file, rank = dest[0], int(dest[1])
    base_pitch = config['pitch_mapping'][file]
    note = base_pitch + (rank - 1) * 3 - 12 # lower octave
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
    parser = argparse.ArgumentParser(
        description="Convert chess games (PGN) to MIDI music files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 c2m.py data/game.pgn
  python3 c2m.py data/game.pgn --config custom_config.yaml
  python3 c2m.py data/game.pgn -c my_settings.yaml
        """
    )

    parser.add_argument('pgn_file',
                       help='Path to the PGN file to convert')
    parser.add_argument('-c', '--config',
                       default='config.yaml',
                       help='Path to configuration YAML file (default: config.yaml)')
    parser.add_argument('-o', '--output',
                       help='Output MIDI file path (default: <pgn_file>.mid)')

    args = parser.parse_args()

    # Validate PGN file exists
    if not pathlib.Path(args.pgn_file).exists():
        print(f"Error: PGN file '{args.pgn_file}' not found")
        sys.exit(1)

    # Validate config file exists
    if not pathlib.Path(args.config).exists():
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)

    config = load_config(args.config)

    with open(args.pgn_file) as pgn:
        game = chess.pgn.read_game(pgn)

    eco_map = load_eco_map("./openings")  # folder with a.tsv … e.tsv
    opening_name = get_opening_name(game, eco_map)
    print(game.headers.get("ECO"), opening_name)

    mid = mido.MidiFile()
    white_track = mido.MidiTrack()
    black_track = mido.MidiTrack()
    mid.tracks.extend([white_track, black_track])

    board = game.board()
    node = game  # needed to access comments in sync with moves

    for ply, move in enumerate(game.mainline_moves(), start=1):
        node = node.variation(0)       # advance node alongside move
        comment = node.comment         # extract PGN comment (timestamps)

        note = move_to_note(move, board, config)
        piece = board.piece_at(move.from_square).symbol().upper()
        velocity = config['velocity'].get(piece, 80)
        program = config['instruments'].get(piece, 0)

        ts = extract_timestamp(comment)
        if ts is not None:
            duration = max(config['durations']['min_duration'],
                         min(ts * config['durations']['timestamp_multiplier'],
                             config['durations']['max_duration']))
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

        add_note(track, program, note, velocity, duration, pan)

        if nag in config['effects']['nag']:
            effect = config['effects']['nag'][nag]
            effect_program = effect.get('instrument', program)
            effect_note = note + effect.get('octave_shift', 0) + effect.get('pitch_shift', 0)
            effect_velocity = effect.get('velocity', velocity + effect.get('velocity_boost', 0) + effect.get('velocity_change', 0))
            effect_duration = int(duration * effect.get('duration_multiplier', effect.get('duration_ratio', 1)))
            effect_delay = effect.get('delay', 0)
            add_note(track, effect_program, effect_note, effect_velocity, effect_duration, pan, delay=effect_delay)


        if board.is_capture(move):
            capture_effect = config['effects']['capture']
            capture_note = min(note + capture_effect['pitch_shift'], 127)
            capture_velocity = velocity + capture_effect['velocity_change']
            add_note(track, program, capture_note, capture_velocity, duration, pan, delay=capture_effect['delay'])

        board.push(move)
        if board.is_checkmate():
            checkmate_effect = config['effects']['checkmate']
            for n in checkmate_effect['chord']:
                add_note(track, program, n, checkmate_effect['velocity'], checkmate_effect['duration'], pan)
        elif board.is_check():
            check_effect = config['effects']['check']
            check_note = min(note + check_effect['pitch_shift'], 127)
            add_note(track, program, check_note, check_effect['velocity'], duration, pan)

    # Determine output filename
    if args.output:
        out_name = args.output
    else:
        out_name = args.pgn_file.rsplit(".", 1)[0] + ".mid"

    mid.save(out_name)
    print(f"Saved MIDI to {out_name}")

if __name__ == "__main__":
    main()

