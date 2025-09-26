"""Phrase planner layer.

Takes enriched game JSON (or PGN path) and segments moves into narrative phrases.
Phrases are contiguous ply ranges bounded by trigger events (eval swings,
structural transitions, tactical bursts, long thinks) and a max length cap.

Goals:
- Deterministic segmentation
- Lightweight, no musical terms
- Attach per-phrase summary stats (avg tension, net eval change, tag counts)

Input expectation (moves contain original extractor fields + enrichment fields):
  moves[i] must have: ply, eval_cp, eval_bucket, eval_delta_bucket, enriched_tags, tension,
                      is_capture, is_check, promotion_piece,
                      first_both_castled, first_both_queens_off

Output:
  {
    "phrases": [
       {"index":0, "start_ply":1, "end_ply":12, "length":12,
        "triggers":["both_castled"],
        "stats": {"avg_tension":0.23, "peak_tension":0.41, "eval_net": +45,
                   "captures":2, "checks":1, "promotions":0,
                   "swing_tags":5}
       }, ...
    ],
    "config_phrase": {...}
  }

"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from feature_extractor import extract_game_features
    from narrative_tags import enrich_game, _load_raw_from_pgn as load_raw
except ImportError:  # pragma: no cover
    extract_game_features = None  # type: ignore
    enrich_game = None  # type: ignore
    load_raw = None  # type: ignore

DEFAULT_PHRASE_CONFIG = {
    "max_phrase_length": 14,          # Hard cap to keep phrases musically manageable
    "min_phrase_length": 4,           # Avoid tiny phrases unless triggered
    "big_swing_delta": 120,           # Centipawn swing that forces a boundary
    "huge_swing_delta": 250,          # Larger swing stronger justification
    "tension_spike_threshold": 0.25,  # Absolute increase over previous tension local mean
    "capture_burst_window": 4,        # Window length for capture burst detection
    "capture_burst_min": 3,           # >= this many captures in window triggers boundary after window
    "long_think_boundary_seconds": 45.0,  # A very long think can start a new phrase
    "allow_mid_short_if_trigger": True,
}

@dataclass
class Phrase:
    index: int
    start_ply: int
    end_ply: int
    triggers: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_ply": self.start_ply,
            "end_ply": self.end_ply,
            "length": self.end_ply - self.start_ply + 1,
            "triggers": self.triggers,
            "stats": self.stats,
        }

# ------------------------------ Core Logic -------------------------------- #

def detect_boundary(prev_move: Dict[str, Any], curr_move: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Return list of boundary trigger labels between prev and curr."""
    triggers: List[str] = []
    if not prev_move:
        return triggers

    # Eval swing
    pe = prev_move.get("eval_cp")
    ce = curr_move.get("eval_cp")
    if pe is not None and ce is not None:
        swing = abs(ce - pe)
        if swing >= cfg["huge_swing_delta"]:
            triggers.append("huge_swing")
        elif swing >= cfg["big_swing_delta"]:
            triggers.append("big_swing")

    # Structural first occurrence flags on current move
    if curr_move.get("first_both_castled"):
        triggers.append("both_castled")
    if curr_move.get("first_both_queens_off"):
        triggers.append("both_queens_off")

    # Long think starts new phrase at current if threshold passed
    emt = curr_move.get("emt_seconds")
    if isinstance(emt, (int, float)) and emt >= cfg["long_think_boundary_seconds"]:
        triggers.append("long_think_start")

    return triggers

def capture_burst_trigger(moves: List[Dict[str, Any]], idx: int, cfg: Dict[str, Any]) -> bool:
    win = cfg["capture_burst_window"]
    if idx + 1 < win:
        return False
    window = moves[idx+1-win: idx+1]
    captures = sum(1 for m in window if m.get("is_capture"))
    return captures >= cfg["capture_burst_min"]


def plan_phrases(raw_enriched: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = {**DEFAULT_PHRASE_CONFIG, **(cfg or {})}
    moves: List[Dict[str, Any]] = raw_enriched.get("moves", [])
    phrases: List[Phrase] = []

    if not moves:
        return {"phrases": [], "config_phrase": cfg}

    start_idx = 0  # index into moves list
    phrase_index = 0

    i = 1
    while i < len(moves):
        prev_m = moves[i-1]
        curr_m = moves[i]
        triggers = detect_boundary(prev_m, curr_m, cfg)

        # Capture burst triggers boundary AFTER the burst window ends
        if capture_burst_trigger(moves, i, cfg):
            triggers.append("capture_burst")

        length = i - start_idx  # prospective length if boundary before curr
        force_boundary = False

        if triggers:
            if length >= cfg["min_phrase_length"] or cfg["allow_mid_short_if_trigger"]:
                force_boundary = True

        # Hard length cap
        if not force_boundary and length >= cfg["max_phrase_length"]:
            force_boundary = True
            triggers.append("length_cap")

        if force_boundary:
            seg_moves = moves[start_idx:i]
            phrases.append(build_phrase(seg_moves, phrase_index, triggers))
            phrase_index += 1
            start_idx = i
        i += 1

    # Final phrase
    if start_idx < len(moves):
        seg_moves = moves[start_idx:]
        phrases.append(build_phrase(seg_moves, phrase_index, ["final"]))

    return {"phrases": [p.to_dict() for p in phrases], "config_phrase": cfg}


def build_phrase(seg_moves: List[Dict[str, Any]], index: int, triggers: List[str]) -> Phrase:
    tensions = [m.get("tension", 0.0) for m in seg_moves]
    evals = [m.get("eval_cp") for m in seg_moves if m.get("eval_cp") is not None]
    start_eval = evals[0] if evals else None
    end_eval = evals[-1] if evals else None
    net = (end_eval - start_eval) if (start_eval is not None and end_eval is not None) else None
    stats = {
        "avg_tension": round(sum(tensions)/len(tensions), 4) if tensions else 0.0,
        "peak_tension": round(max(tensions), 4) if tensions else 0.0,
        "eval_net": net,
        "captures": sum(1 for m in seg_moves if m.get("is_capture")),
        "checks": sum(1 for m in seg_moves if m.get("is_check")),
        "promotions": sum(1 for m in seg_moves if m.get("promotion_piece")),
        "swing_tags": sum(1 for m in seg_moves if any(t.startswith("eval_") for t in m.get("enriched_tags", []))),
    }
    return Phrase(index=index,
                  start_ply=seg_moves[0]["ply"],
                  end_ply=seg_moves[-1]["ply"],
                  triggers=triggers,
                  stats=stats)

# ------------------------------ CLI -------------------------------------- #

def _load_full_chain(source: str) -> Dict[str, Any]:
    """If source is PGN run extractor+enrich; if JSON load directly."""
    if source.lower().endswith('.pgn'):
        if load_raw is None or enrich_game is None:
            raise RuntimeError("Required modules not available")
        raw = load_raw(source)
        enriched = enrich_game(raw)
        out = raw.copy()
        # merge enrichment fields into moves already done by enrich_game when run standalone, so replicate
        for base_mv, enr_mv in zip(out.get('moves', []), enriched.moves):
            base_mv.setdefault('eval_bucket', enr_mv.eval_bucket)
            base_mv.setdefault('eval_delta_bucket', enr_mv.eval_delta_bucket)
            base_mv.setdefault('enriched_tags', enr_mv.tags)
            base_mv.setdefault('tension', enr_mv.tension)
        out['sections'] = enriched.sections
        out['config_enrichment'] = enriched.config
        return out
    else:
        with open(source, 'r', encoding='utf-8') as f:
            return json.load(f)


def main(argv: Sequence[str]) -> int:
    import argparse, sys
    ap = argparse.ArgumentParser(description="Phrase planner")
    ap.add_argument('source', help='PGN or enriched JSON')
    ap.add_argument('--config', help='Phrase config JSON file')
    ap.add_argument('--pretty', action='store_true')
    args = ap.parse_args(argv)

    cfg_override = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg_override = json.load(f)

    chain = _load_full_chain(args.source)
    phrases_doc = plan_phrases(chain, cfg_override)

    # Attach to chain (non-destructive)
    output = chain.copy()
    output['phrases'] = phrases_doc['phrases']
    output['config_phrase'] = phrases_doc['config_phrase']

    if args.pretty:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, separators=(',',':')))
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
