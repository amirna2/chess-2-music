#!/usr/bin/env python3
"""Feature extraction module for chess game narrative.

This module parses a PGN into structured dataclasses capturing metadata,
per-move features, and derived aggregate metrics. It is deliberately
pure (no musical mapping) so downstream composer layers can consume
semantic chess game information to construct musical narratives.

Design Principles:
- Separation of concerns (parse -> features -> later narrative planning)
- Deterministic, testable extraction
- JSON-serializable dataclasses (via to_dict)
- Extensible placeholders for future engine eval / semantic tags

Dependencies: python-chess
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import chess
import chess.pgn
import csv

# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------

class MoveKind(Enum):
    """Categorical classification of a move's primary tactical nature."""
    NORMAL = auto()
    CAPTURE = auto()
    CHECK = auto()
    CHECKMATE = auto()
    PROMOTION = auto()
    CASTLE_KING = auto()
    CASTLE_QUEEN = auto()
    UNKNOWN = auto()

@dataclass
class MoveFeature:
    """Feature bundle for a single half-move (ply)."""
    ply: int
    fullmove: int
    side: str                 # "white" or "black"
    san: str
    uci: str
    piece: Optional[str]      # 'P','N','B','R','Q','K'
    from_sq: str
    to_sq: str
    kind: MoveKind
    nag_codes: List[int] = field(default_factory=list)
    is_capture: bool = False
    is_check: bool = False
    is_mate: bool = False
    is_promotion: bool = False
    promotion_piece: Optional[str] = None
    emt_seconds: Optional[float] = None          # Elapsed move time (if present)
    clock_after_seconds: Optional[float] = None  # Remaining clock after move (if present)
    comment_raw: Optional[str] = None
    eval_cp: Optional[int] = None                # Future engine centipawn value
    eval_mate: Optional[int] = None              # Future engine mate distance
    tags: List[str] = field(default_factory=list)  # Higher-level semantic tags (future)
    move_distance: Optional[int] = None           # Chebyshev-style move distance (as in c2m)
    # Event state after this move
    both_castled_after: Optional[bool] = None
    both_queens_off_after: Optional[bool] = None
    first_both_castled: Optional[bool] = None      # True only on ply where event first becomes true
    first_both_queens_off: Optional[bool] = None   # True only on ply where event first becomes true

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.name
        return d

@dataclass
class GameMetadata:
    event: Optional[str]
    site: Optional[str]
    date: Optional[str]
    round: Optional[str]
    white: Optional[str]
    black: Optional[str]
    result: Optional[str]
    eco: Optional[str]
    opening: Optional[str]
    variation: Optional[str]
    time_control: Optional[str]
    speed_category: Optional[str]

@dataclass
class DerivedMetrics:
    total_moves: int
    total_plies: int
    total_think_time_white: float
    total_think_time_black: float
    avg_think_white: Optional[float]
    avg_think_black: Optional[float]
    longest_think_white: Optional[float]
    longest_think_black: Optional[float]
    num_captures: int
    num_checks: int
    num_promotions: int
    num_mates: int
    both_castled_ply: Optional[int] = None
    both_queens_off_ply: Optional[int] = None

@dataclass
class GameFeatures:
    metadata: GameMetadata
    moves: List[MoveFeature]
    metrics: DerivedMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "metrics": asdict(self.metrics),
            "moves": [m.to_dict() for m in self.moves],
        }

# ---------------------------------------------------------------------------
# Regex helpers for EMT / clock parsing in comments
# ---------------------------------------------------------------------------

_EMT_PATTERN = re.compile(r"\[%emt\s+([\d:.]+)\]", re.IGNORECASE)
_CLK_PATTERN = re.compile(r"\[%clk\s+([\d:.]+)\]", re.IGNORECASE)
_EVAL_PATTERN = re.compile(r"\[%eval\s+([^\]]+)\]", re.IGNORECASE)


def _parse_hms_to_seconds(token: str) -> Optional[float]:
    """Convert H:MM:SS(.f) | M:SS(.f) | SS(.f) string to seconds.

    Chess.com clock / emt annotations often include fractional seconds
    in the final field (e.g. 0:02:59.3 or 0:00:00.7). This parser
    tolerates that while keeping earlier components integral.
    """
    if not token:
        return None
    parts = token.split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        # All but last must be ints; last may be float.
        *head, last = parts
        head_ints = [int(p) for p in head] if head else []
        sec = float(last)
    except ValueError:
        return None
    # Normalize to h, m depending on length
    if len(head_ints) == 2:  # h, m provided
        h, m = head_ints
    elif len(head_ints) == 1:  # only m provided
        h = 0
        m = head_ints[0]
    else:  # no head -> only seconds
        h = 0
        m = 0
    return h * 3600 + m * 60 + sec

# ---------------------------------------------------------------------------
# NAG mapping (basic common codes 1-6)
# ---------------------------------------------------------------------------
NAG_MAP: Dict[int, Dict[str, str]] = {
    1: {"symbol": "!",  "label": "good"},
    2: {"symbol": "?",  "label": "mistake"},
    3: {"symbol": "!!", "label": "brilliant"},
    4: {"symbol": "??", "label": "blunder"},
    5: {"symbol": "!?", "label": "interesting"},
    6: {"symbol": "?!", "label": "dubious"},
}

def decode_nags(nags: List[int]) -> List[str]:
    return [NAG_MAP[n]["symbol"] for n in nags if n in NAG_MAP]


def extract_emt_seconds(comment: str) -> Optional[float]:
    """Extract elapsed move time from a comment block."""
    if not comment:
        return None
    m = _EMT_PATTERN.search(comment)
    return _parse_hms_to_seconds(m.group(1)) if m else None


def extract_clock_seconds(comment: str) -> Optional[float]:
    """Extract post-move remaining clock time from a comment block."""
    if not comment:
        return None
    m = _CLK_PATTERN.search(comment)
    return _parse_hms_to_seconds(m.group(1)) if m else None


def extract_eval(comment: str) -> tuple[Optional[int], Optional[int]]:
    """Extract computer evaluation from comment.

    Returns (cp, mate) where only one is set:
      cp   : centipawn integer (e.g. 0.37 -> 37, -1.25 -> -125)
      mate : moves to mate (positive if side to move mates, negative if mated)
    Supports formats like '0.18', '-1.23', '#5'.
    If multiple eval tags, first match taken.
    """
    if not comment:
        return (None, None)
    m = _EVAL_PATTERN.search(comment)
    if not m:
        return (None, None)
    token = m.group(1).strip()
    # Mate notation might appear as #5 or #-3
    if token.startswith('#'):
        try:
            mate_val = int(token[1:])
            return (None, mate_val)
        except ValueError:
            return (None, None)
    # Otherwise float centipawn in pawns
    try:
        val = float(token)
        return (int(round(val * 100)), None)
    except ValueError:
        return (None, None)


def identify_move_kind(board: chess.Board, move: chess.Move, san: str) -> MoveKind:
    """Infer the primary move kind before move is executed."""
    if board.is_castling(move):
        # King side castling ends on file g (file index 6)
        return MoveKind.CASTLE_KING if chess.square_file(move.to_square) == 6 else MoveKind.CASTLE_QUEEN
    if san.endswith("#"):
        return MoveKind.CHECKMATE
    if san.endswith("+"):
        return MoveKind.CHECK
    if move.promotion:
        return MoveKind.PROMOTION
    if board.is_capture(move):
        return MoveKind.CAPTURE
    return MoveKind.NORMAL


def compute_move_distance(board: chess.Board, move: chess.Move) -> int:
    """Replicate c2m move distance (chebyshev except knight fixed 3)."""
    from_sq = move.from_square
    to_sq = move.to_square
    file_dist = abs(chess.square_file(to_sq) - chess.square_file(from_sq))
    rank_dist = abs(chess.square_rank(to_sq) - chess.square_rank(from_sq))
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.KNIGHT:
        return 3
    return max(file_dist, rank_dist)




def classify_time_control(tc: Optional[str]) -> Optional[str]:
    """Classify time control into a coarse speed category.
    Rules (increment ignored):
        bullet   : base < 180 seconds
        blitz    : 180 <= base <= 600
        rapid    : 600 < base <= 3600
        classical: base > 3600

    Accepted PGN TimeControl forms:
        "<base>+<inc>" (e.g., 180+2)  -> uses <base>
        "<base>"      (e.g., 600)
        multi-stage ("40/7200:1800") -> classical
        missing / '?' -> None
    """
    if not tc or tc == "?":
        return None
    # Multi-stage -> classical (treat as long form)
    if "/" in tc:
        return "classical"
    if "+" in tc:
        try:
            base, _inc = tc.split("+", 1)
            base = int(base)
        except ValueError:
            return None
    else:
        try:
            base = int(tc)
        except ValueError:
            return None
    if base < 180:
        return "bullet"
    if base <= 600:
        return "blitz"
    if base <= 3600:
        return "rapid"
    return "classical"


def load_eco_openings(openings_dir: str | Path) -> Dict[str, List[Dict[str, str]]]:
    """Load ECO openings from TSV files (columns: eco, name, pgn).

    Returns mapping eco_code -> list of entries {'name': str, 'pgn': str}.
    Multiple lines per ECO are preserved (variations). We treat the first as
    the primary if no variation header given.
    """
    openings: Dict[str, List[Dict[str, str]]] = {}
    openings_path = Path(openings_dir)
    for tsv in openings_path.glob('*.tsv'):
        with tsv.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eco = row.get('eco') or row.get('ECO')
                name = row.get('name') or row.get('Name')
                pgn_line = row.get('pgn') or row.get('PGN')
                if not eco or not name:
                    continue
                openings.setdefault(eco, []).append({'name': name, 'pgn': pgn_line})
    return openings

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_game_features(pgn_path: str | Path,
                          eco_opening_lookup: Dict[str, List[Dict[str, str]]] | None = None,
                          openings_dir: str | Path | None = None) -> GameFeatures:
    """Parse a single PGN file and return structured feature data.

    Parameters
    ----------
    pgn_path : str | Path
        Path to a PGN file containing exactly one game (first game is read).
    eco_opening_lookup : Optional mapping
        Optional pre-loaded mapping ECO -> {"name": str, "variation": str}
        to fill missing opening / variation headers.
    """
    pgn_path = Path(pgn_path)
    with pgn_path.open("r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        raise ValueError("No game found in PGN file.")

    headers = game.headers
    eco = headers.get("ECO")
    opening = headers.get("Opening")
    variation = headers.get("Variation")

    # Load openings dynamically if directory provided and no mapping passed
    if openings_dir and eco_opening_lookup is None:
        try:
            eco_opening_lookup = load_eco_openings(openings_dir)
        except Exception:
            eco_opening_lookup = None

    # Apply ECO lookup fallback if opening missing
    if eco and opening is None and eco_opening_lookup and eco in eco_opening_lookup:
        # Pick first entry as primary name
        first = eco_opening_lookup[eco][0]
        opening = first['name']

    tc_header = headers.get("TimeControl")
    speed_category = classify_time_control(tc_header)

    metadata = GameMetadata(
        event=headers.get("Event"),
        site=headers.get("Site"),
        date=headers.get("Date"),
        round=headers.get("Round"),
        white=headers.get("White"),
        black=headers.get("Black"),
        result=headers.get("Result"),
        eco=eco,
        opening=opening,
        variation=variation,
        time_control=tc_header,
        speed_category=speed_category,
    )

    board = game.board()
    moves_features: List[MoveFeature] = []

    ply = 1
    white_castled = False
    black_castled = False
    both_castled_ply: Optional[int] = None
    both_queens_off_ply: Optional[int] = None
    for node in game.mainline():
        move = node.move
        san = board.san(move)
        nag_codes = [int(nag) for nag in node.nags] if node.nags else []
        comment = node.comment or ""
        emt = extract_emt_seconds(comment)
        clk = extract_clock_seconds(comment)
        eval_cp, eval_mate = extract_eval(comment)
        kind = identify_move_kind(board, move, san)
        piece_obj = board.piece_at(move.from_square)
        piece_symbol = piece_obj.symbol().upper() if piece_obj else None

        # Compute distance BEFORE pushing (need original piece)
        distance = compute_move_distance(board, move)

        # Capture fullmove number BEFORE pushing move
        current_fullmove = board.fullmove_number

        board.push(move)

        is_check = board.is_check()
        is_mate = board.is_checkmate()
        is_capture = "x" in san or kind == MoveKind.CAPTURE
        is_promo = move.promotion is not None
        promo_piece = None
        if is_promo:
            promo_piece = chess.Piece(move.promotion, chess.WHITE).symbol().upper()

        # Update castling state
        if kind in (MoveKind.CASTLE_KING, MoveKind.CASTLE_QUEEN):
            if ply % 2 == 1:  # white move
                white_castled = True
            else:
                black_castled = True
        current_both_castled = white_castled and black_castled
        if current_both_castled and both_castled_ply is None:
            both_castled_ply = ply

        # Update queens-off state
        white_queen_count = len(board.pieces(chess.QUEEN, chess.WHITE))
        black_queen_count = len(board.pieces(chess.QUEEN, chess.BLACK))
        current_both_queens_off = (white_queen_count == 0 and black_queen_count == 0)
        if current_both_queens_off and both_queens_off_ply is None:
            both_queens_off_ply = ply

        mf = MoveFeature(
            ply=ply,
            fullmove=current_fullmove,
            side="white" if ply % 2 == 1 else "black",
            san=san,
            uci=move.uci(),
            piece=piece_symbol,
            from_sq=chess.square_name(move.from_square),
            to_sq=chess.square_name(move.to_square),
            kind=MoveKind.CHECKMATE if is_mate else kind,
            nag_codes=sorted(nag_codes),
            is_capture=is_capture,
            is_check=is_check or kind == MoveKind.CHECK,
            is_mate=is_mate,
            is_promotion=is_promo,
            promotion_piece=promo_piece,
            emt_seconds=emt,
            clock_after_seconds=clk,
            comment_raw=comment if comment else None,
            move_distance=distance,
            eval_cp=eval_cp,
            eval_mate=eval_mate,
            both_castled_after=current_both_castled,
            both_queens_off_after=current_both_queens_off,
            first_both_castled=True if (current_both_castled and ply == both_castled_ply) else None,
            first_both_queens_off=True if (current_both_queens_off and ply == both_queens_off_ply) else None,
        )
        moves_features.append(mf)
        ply += 1

    # Derived aggregate metrics
    white_thinks = [m.emt_seconds for m in moves_features if m.side == "white" and m.emt_seconds is not None]
    black_thinks = [m.emt_seconds for m in moves_features if m.side == "black" and m.emt_seconds is not None]

    metrics = DerivedMetrics(
        total_moves=moves_features[-1].fullmove if moves_features else 0,
        total_plies=moves_features[-1].ply if moves_features else 0,
        total_think_time_white=sum(white_thinks) if white_thinks else 0.0,
        total_think_time_black=sum(black_thinks) if black_thinks else 0.0,
        avg_think_white=(sum(white_thinks) / len(white_thinks)) if white_thinks else None,
        avg_think_black=(sum(black_thinks) / len(black_thinks)) if black_thinks else None,
        longest_think_white=max(white_thinks) if white_thinks else None,
        longest_think_black=max(black_thinks) if black_thinks else None,
        num_captures=sum(1 for m in moves_features if m.is_capture),
        num_checks=sum(1 for m in moves_features if m.is_check),
        num_promotions=sum(1 for m in moves_features if m.is_promotion),
        num_mates=sum(1 for m in moves_features if m.is_mate),
    both_castled_ply=both_castled_ply,
    both_queens_off_ply=both_queens_off_ply,
    )

    return GameFeatures(metadata=metadata, moves=moves_features, metrics=metrics)

# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo_print(game_features: GameFeatures, full: bool = False) -> None:
    md = game_features.metadata
    mt = game_features.metrics
    print("=== Game Metadata ===")
    print(f"Players      : {md.white} vs {md.black}")
    print(f"Result       : {md.result}")
    print(f"ECO / Opening: {md.eco} | {md.opening} ({md.variation})")
    print(f"Time Control : {md.time_control}  Speed={md.speed_category}")
    print(f"Event/Site   : {md.event} @ {md.site}")
    print()
    print("=== Aggregate Metrics ===")
    print(f"Total moves     : {mt.total_moves}")
    print(f"Total plies     : {mt.total_plies}")
    print(f"Captures/Checks : {mt.num_captures} / {mt.num_checks}")
    print(f"Promotions/Mates: {mt.num_promotions} / {mt.num_mates}")
    print(f"Think White (tot/avg/max): {mt.total_think_time_white:.2f} / {mt.avg_think_white} / {mt.longest_think_white}")
    print(f"Think Black (tot/avg/max): {mt.total_think_time_black:.2f} / {mt.avg_think_black} / {mt.longest_think_black}")
    print(f"Both castled ply : {mt.both_castled_ply}")
    print(f"Both queens off ply: {mt.both_queens_off_ply}")
    print()
    subset = game_features.moves if full else game_features.moves[:10]
    print(f"=== Move Features ({'all' if full else 'first 10'}) ===")
    # Adopt a fixed-width table with separators similar to c2m's print_move_line style.
    col_defs = [
        ("PLY", 3), ("SD", 2), ("FM", 3), ("SAN", 10), ("KIND", 11),
        ("CAP", 3), ("CHK", 3), ("EMT", 7), ("CLOCK", 9), ("EVAL", 7), ("DIST", 4), ("PROM", 4), ("NAGS", 8), ("NSYM", 6)
    ]
    header = " | ".join(f"{name:<{w}}" for name, w in col_defs)
    ruler = "-+-".join('-'*w for _, w in col_defs)
    print(header)
    print(ruler)
    for m in subset:
        nags_num = ','.join(str(n) for n in m.nag_codes) if m.nag_codes else ''
        nags_sym = ''.join(decode_nags(m.nag_codes))
        emt_str = f"{m.emt_seconds:.2f}" if m.emt_seconds is not None else '--'
        clk_str = f"{m.clock_after_seconds:.1f}" if m.clock_after_seconds is not None else '--'
        san_disp = m.san if len(m.san) <= 10 else m.san[:9] + '…'
        # Format eval: mate overrides cp; cp in pawns with sign
        if m.eval_mate is not None:
            eval_disp = f"#{m.eval_mate}"
        elif m.eval_cp is not None:
            eval_disp = f"{m.eval_cp/100:+.2f}"  # show +0.37 style
        else:
            eval_disp = '--'
        row_vals = [
            f"{m.ply:03d}", m.side[0], f"{m.fullmove:02d}", san_disp, m.kind.name,
            str(int(m.is_capture)), str(int(m.is_check)), emt_str, clk_str,
            eval_disp,
            (str(m.move_distance) if m.move_distance is not None else '-'),
            (m.promotion_piece or '-'), nags_num, nags_sym
        ]
        line = " | ".join(f"{val:<{w}}" for val, (_, w) in zip(row_vals, col_defs))
        print(line)
    if not full and len(game_features.moves) > 10:
        print(f"... ({len(game_features.moves)-10} more plies) use --full to list all")


def main():  # pragma: no cover - simple convenience CLI
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <game.pgn> [--json] [--full]")
        raise SystemExit(1)
    pgn_file = sys.argv[1]
    full = "--full" in sys.argv
    feats = extract_game_features(pgn_file)
    if "--json" in sys.argv:
        print(json.dumps(feats.to_dict(), indent=2))
    else:
        _demo_print(feats, full=full)


if __name__ == "__main__":  # pragma: no cover
    main()
