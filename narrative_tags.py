"""Neutral narrative tag enrichment layer.

Takes raw extractor JSON (or a PGN path which it will extract) and produces
light-weight semantic tags that are:
- Chess narrative only (no musical domain terms)
- Deterministic & reproducible
- Configurable via simple threshold dictionary

Outputs (added fields):
- per-move tags list (strings)
- eval_bucket (centipawn advantage coarse bucket)
- eval_delta_bucket (change since previous eval bucket)
- tension (0..1 float) derived from eval swings & critical events
- sections: coarse segmentation boundaries (opening, midgame, endgame-ish) using
  structural events (castling, queens off) + move count heuristics

This intentionally excludes any musical mapping (no tempo, harmony, timbre, etc.).
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Lazy import to avoid hard dependency here; feature_extractor already exists.
try:
    from feature_extractor import extract_game_features
except ImportError:  # pragma: no cover
    extract_game_features = None  # type: ignore

# ------------------------------- Config ------------------------------------ #

DEFAULT_CONFIG = {
    "eval_cp_buckets": [-300, -120, -60, -20, 20, 60, 120, 300],  # centipawn thresholds dividing advantage bands
    "eval_delta_cp_big_swing": 80,   # change in centipawns considered a notable swing
    "eval_delta_cp_huge_swing": 180,
    "long_think_seconds": 30.0,      # EMT threshold for a long think
    "snap_think_seconds": 2.0,       # EMT threshold for a snap move (fast reply)
    "tension_decay": 0.92,           # exponential decay factor per ply
    "tension_swing_weight": 0.015,   # contribution per centipawn swing (scaled abs delta)
    "tension_event_bonus": 0.18,     # added for structural or tactical events (capture/check/promotion)
    "tension_cap": 1.0,
    "min_section_ply_gap": 10,       # avoid overly dense sections
}

# ------------------------------- Data ------------------------------------- #

@dataclass
class EnrichedMove:
    ply: int
    san: str
    eval_cp: Optional[int]
    eval_bucket: str
    eval_delta_bucket: Optional[str]
    tags: List[str]
    tension: float

@dataclass
class EnrichedGame:
    source: str
    moves: List[EnrichedMove]
    sections: List[Dict[str, Any]]  # each: {"name": str, "start_ply": int}
    config: Dict[str, Any] = field(default_factory=dict)

# --------------------------- Helper Functions ------------------------------ #

def bucket_eval(eval_cp: Optional[int], thresholds: Sequence[int]) -> str:
    if eval_cp is None:
        return "unknown"
    # thresholds define boundaries for intervals; produce label like "<=-300" or "(-60,-20]" etc.
    if not thresholds:
        return "0" if abs(eval_cp) < 10 else ("+" if eval_cp > 0 else "-")
    # Build extended list with infinities
    extended = [-math.inf, *thresholds, math.inf]
    for lo, hi in zip(extended, extended[1:]):
        if lo < eval_cp <= hi:
            # produce canonical text
            if lo == -math.inf:
                return f"<={hi}"
            if hi == math.inf:
                return f">{lo}"
            return f"({lo},{hi}]"
    return "unknown"

def bucket_delta(prev_eval: Optional[int], curr_eval: Optional[int], big: int, huge: int) -> Optional[str]:
    if prev_eval is None or curr_eval is None:
        return None
    delta = curr_eval - prev_eval
    ad = abs(delta)
    if ad >= huge:
        return "+huge" if delta > 0 else "-huge"
    if ad >= big:
        return "+big" if delta > 0 else "-big"
    if ad >= big / 2:
        return "+med" if delta > 0 else "-med"
    if ad >= 10:
        return "+small" if delta > 0 else "-small"
    return "flat"

# --------------------------- Enrichment Core ------------------------------- #

def compute_sections(raw: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    min_gap = cfg["min_section_ply_gap"]

    def add_section(name: str, ply: int):
        if sections and ply - sections[-1]["start_ply"] < min_gap:
            return
        sections.append({"name": name, "start_ply": ply})

    add_section("opening", 1)

    # metrics structure mirrors feature_extractor.GameFeatures.metrics
    derived = raw.get("metrics", {})
    bc_ply = derived.get("both_castled_ply")
    if isinstance(bc_ply, int):
        add_section("post_castling", bc_ply)

    bq_ply = derived.get("both_queens_off_ply")
    if isinstance(bq_ply, int):
        add_section("queenless_midgame", bq_ply)

    # crude endgame guess: last 15 plies or when <= 10 pieces each side (not currently tracked -> fallback to depth)
    total_plies = len(raw.get("moves", []))
    if total_plies > 0:
        add_section("late", max(total_plies - 15, 1))

    return sections

def enrich_game(raw: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> EnrichedGame:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    thresholds = cfg["eval_cp_buckets"]

    moves_out: List[EnrichedMove] = []
    prev_eval: Optional[int] = None
    tension = 0.0

    original_moves = raw.get("moves", [])
    for mv in original_moves:
        ply = mv.get("ply")
        san = mv.get("san", "")
        eval_cp = mv.get("eval_cp")
        tags: List[str] = []

        # Basic flags
        if mv.get("is_capture"): tags.append("capture")
        if mv.get("is_check"): tags.append("check")
        if mv.get("promotion_piece"): tags.append("promotion")

        # Think time categories
        emt = mv.get("emt_seconds")
        if isinstance(emt, (int, float)):
            if emt >= cfg["long_think_seconds"]: tags.append("long_think")
            elif emt <= cfg["snap_think_seconds"]: tags.append("snap_move")

        # Structural first-occurrence flags
        if mv.get("first_both_castled"): tags.append("both_castled_now")
        if mv.get("first_both_queens_off"): tags.append("both_queens_off_now")

        # Evaluation buckets
        eval_bucket = bucket_eval(eval_cp, thresholds)
        delta_bucket = bucket_delta(prev_eval, eval_cp, cfg["eval_delta_cp_big_swing"], cfg["eval_delta_cp_huge_swing"])
        if delta_bucket and delta_bucket != "flat":
            tags.append(f"eval_{delta_bucket}")

        # Tension accumulation
        # base decay
        tension *= cfg["tension_decay"]
        # add swing
        if prev_eval is not None and eval_cp is not None:
            swing = abs(eval_cp - prev_eval)
            tension += swing * cfg["tension_swing_weight"] / 100.0  # scale down centipawns
        # event bonuses
        if any(k in tags for k in ("capture", "check", "promotion")):
            tension += cfg["tension_event_bonus"]
        tension = min(tension, cfg["tension_cap"])

        moves_out.append(EnrichedMove(
            ply=ply,
            san=san,
            eval_cp=eval_cp,
            eval_bucket=eval_bucket,
            eval_delta_bucket=delta_bucket,
            tags=tags,
            tension=round(tension, 4),
        ))

        prev_eval = eval_cp if eval_cp is not None else prev_eval

    sections = compute_sections(raw, cfg)

    return EnrichedGame(
        source=raw.get("metadata", {}).get("source", "unknown"),
        moves=moves_out,
        sections=sections,
        config=cfg,
    )

# ------------------------------- CLI --------------------------------------- #

def _load_raw_from_pgn(pgn_path: str) -> Dict[str, Any]:
    if extract_game_features is None:
        raise RuntimeError("feature_extractor.extract_game_features not available")
    features = extract_game_features(pgn_path)
    # Convert dataclasses to dict via asdict-like manual extraction (avoid importing dataclasses.asdict for control)
    # Use full extractor dict to avoid data loss
    feats_dict = features.to_dict()
    feats_dict["metadata"]["source"] = pgn_path
    raw = feats_dict
    return raw

def _load_raw_from_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Neutral narrative tag enrichment")
    ap.add_argument("source", help="Path to PGN file OR raw extractor JSON (extension .json) or '-' for stdin JSON")
    ap.add_argument("--config", help="Optional JSON config override file")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    ap.add_argument("--raw-json", action="store_true", help="Force treat source as raw JSON even if extension not .json")
    args = ap.parse_args(argv)

    # Determine input mode
    if args.source == '-':
        raw = json.load(sys.stdin)
    elif args.raw_json or args.source.lower().endswith('.json'):
        raw = _load_raw_from_json(args.source)
    else:
        raw = _load_raw_from_pgn(args.source)
    cfg_override = {}
    if args.config:
        cfg_override = json.loads(Path(args.config).read_text())
    enriched = enrich_game(raw, cfg_override)

    # Merge enriched fields into original move objects for output
    merged_moves = []
    for base_mv, enr in zip(raw.get("moves", []), enriched.moves):
        merged = dict(base_mv)  # copy
        merged.update({
            "eval_bucket": enr.eval_bucket,
            "eval_delta_bucket": enr.eval_delta_bucket,
            "enriched_tags": enr.tags,  # keep separate from original extractor 'tags'
            "tension": enr.tension,
        })
        merged_moves.append(merged)
    out = raw.copy()
    out["sections"] = enriched.sections
    out["config_enrichment"] = enriched.config
    out["moves"] = merged_moves
    if args.pretty:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, separators=(",", ":")))
    return 0

if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
