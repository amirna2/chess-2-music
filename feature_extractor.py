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
    material_balance: Optional[int] = None       # Simple material (white minus black)
    tags: List[str] = field(default_factory=list)  # Higher-level semantic tags (future)

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


def compute_material_balance(board: chess.Board) -> int:
    """Return (white material - black material) using simple piece values."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    for pt, val in piece_values.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val
    return score


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
    for node in game.mainline():
        move = node.move
        san = board.san(move)
        nag_codes = [nag for nag in node.nags] if node.nags else []
        comment = node.comment or ""
        emt = extract_emt_seconds(comment)
        clk = extract_clock_seconds(comment)
        kind = identify_move_kind(board, move, san)
        piece_obj = board.piece_at(move.from_square)
        piece_symbol = piece_obj.symbol().upper() if piece_obj else None

        board.push(move)

        is_check = board.is_check()
        is_mate = board.is_checkmate()
        is_capture = "x" in san or kind == MoveKind.CAPTURE
        is_promo = move.promotion is not None
        promo_piece = None
        if is_promo:
            promo_piece = chess.Piece(move.promotion, chess.WHITE).symbol().upper()

        mat_bal = compute_material_balance(board)

        mf = MoveFeature(
            ply=ply,
            fullmove=board.fullmove_number,
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
            material_balance=mat_bal,
        )
        moves_features.append(mf)
        ply += 1

    # Derived aggregate metrics
    white_thinks = [m.emt_seconds for m in moves_features if m.side == "white" and m.emt_seconds is not None]
    black_thinks = [m.emt_seconds for m in moves_features if m.side == "black" and m.emt_seconds is not None]

    metrics = DerivedMetrics(
        total_moves=len(moves_features),
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
    print(f"Total plies     : {mt.total_moves}")
    print(f"Captures/Checks : {mt.num_captures} / {mt.num_checks}")
    print(f"Promotions/Mates: {mt.num_promotions} / {mt.num_mates}")
    print(f"Think White (tot/avg/max): {mt.total_think_time_white:.2f} / {mt.avg_think_white} / {mt.longest_think_white}")
    print(f"Think Black (tot/avg/max): {mt.total_think_time_black:.2f} / {mt.avg_think_black} / {mt.longest_think_black}")
    print()
    subset = game_features.moves if full else game_features.moves[:10]
    print(f"=== Move Features ({'all' if full else 'first 10'}) ===")
    header = (
        f"{'PLY':>3} {'SD':<1} {'FM':>2} {'SAN':<10} {'KIND':<11} {'CAP':<3} "
        f"{'CHK':<3} {'MAT':>4} {'EMT':>6} {'CLOCK':>8} {'PROM':<4} {'NAGS':<8} {'NSYM':<6}"
    )
    print(header)
    print('-' * len(header))
    for m in subset:
        nags_num = ','.join(str(n) for n in m.nag_codes) if m.nag_codes else ''
        nags_sym = ''.join(decode_nags(m.nag_codes))
        print(
            f"{m.ply:03d} {m.side[0]:<1} {m.fullmove:02d} {m.san:<10} {m.kind.name:<11} "
            f"{int(m.is_capture):<3} {int(m.is_check):<3} {m.material_balance:>4} "
            f"{(f'{m.emt_seconds:.2f}' if m.emt_seconds is not None else '-'):>6} "
            f"{(f'{m.clock_after_seconds:.1f}' if m.clock_after_seconds is not None else '-'):>8} "
            f"{(m.promotion_piece or '-'):>4} {nags_num:<8} {nags_sym:<6}"
        )
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
