"""Neutral narrative tag enrichment layer.

Transforms raw extractor output into chess-narrative semantic signals.

Principles:
- Chess narrative only (no musical domain decisions embedded)
- Deterministic & reproducible
- Threshold-driven (config dictionary)

Per-move added fields / semantics:
    tags: List[str]
        Core tactical/structural: capture, check, promotion, both_castled_now, both_queens_off_now
        Evaluation swing tags: eval_+huge, eval_-huge, eval_+big, eval_-big, eval_+med, eval_-med, eval_+small, eval_-small
        Time usage: long_think, snap_move
        Mate context: mate_threat (<= mate_threat_distance), mate_imminent (<= mate_imminent_distance)
        Theory exit: leaving_theory (first significant NAG 1..6 within early plies)
        Time pressure: white_time_adv_small|big|huge, black_time_adv_*, time_scramble
        Pressure (eval + tension conjunction): white_pressure, black_pressure
    eval_bucket: coarse centipawn interval string (e.g. (20,60], <=-300, >300, unknown)
    eval_delta_bucket: swing classification vs previous (flat / +small / -huge etc.)
    tension: 0..1 float combining decayed prior tension + swing contribution + event bonuses

Mate handling:
    If only eval_mate (ply distance to mate) is present, we synthesize a centipawn equivalent:
        cp = mate_cp_base - (distance-1)*mate_cp_decay_per_ply (floored at mate_cp_min_effective) with sign.
    This keeps bucket logic & tension uniform near forced mates.

Sections:
    Derived structural chapter markers: opening (ply 1), post_castling, queenless_midgame, late
    Each section augmented with metrics:
        avg_eval_cp, eval_trend_cp, phase_tags[] which may include:
            white_advantage_building / black_advantage_building (moderate sustained edge)
            white_conversion_phase / black_conversion_phase (large edge averaging >= threshold)
            tactical_phase (avg tension >= 0.55)
            white_desperate_defense / black_desperate_defense (large negative trend)

Configuration extensions also cover: mate mapping, clock gap thresholds, scramble conditions.

No musical mapping decisions (tempo, harmony, orchestration) live here; consumers layer those downstream.
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
    # --- Mate mapping (when only eval_mate is available) ---
    # We synthesize a large centipawn value so buckets and swings still function.
    # Effective cp = mate_cp_base - (distance-1)*mate_cp_decay_per_ply (clamped >= mate_cp_min_effective)
    "mate_cp_base": 2200,
    "mate_cp_decay_per_ply": 110,
    "mate_cp_min_effective": 600,
    # distance thresholds for tagging imminence
    "mate_imminent_distance": 3,   # <= N plies
    "mate_threat_distance": 6,     # <= N plies (broader)
    # --- Time pressure / clock gap thresholds (seconds) ---
    "clock_gap_small": 15,          # side ahead on clock by > small
    "clock_gap_big": 60,
    "clock_gap_huge": 120,
    "time_scramble_each_under": 30, # both players under this -> scramble
    "time_scramble_total_under": 90 # sum of both clocks under -> late scramble accent
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
        eval_mate = mv.get("eval_mate")  # positive means side to move mates? depends on extractor convention
        # Synthetic mapping for mate scores if numeric cp absent
        if eval_cp is None and isinstance(eval_mate, int):
            # distance in plies to mate; sign indicates side winning (positive = white winning normally in engines)
            dist = abs(eval_mate)
            base = cfg["mate_cp_base"]
            decay = cfg["mate_cp_decay_per_ply"]
            min_eff = cfg["mate_cp_min_effective"]
            synth = base - (dist - 1) * decay
            if synth < min_eff:
                synth = min_eff
            eval_cp = synth if eval_mate > 0 else -synth
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

        # Mate imminence / threat tagging (independent of bucket labels)
        if isinstance(eval_mate, int):
            dist = abs(eval_mate)
            if dist <= cfg["mate_imminent_distance"]:
                tags.append("mate_imminent")
            elif dist <= cfg["mate_threat_distance"]:
                tags.append("mate_threat")

        # Leaving theory detection: first meaningful NAG (assuming extractor provides nags list/int codes)
        # Common significant NAGs: 1 (!), 2 (?), 3 (!!), 4 (??), 5 (!?), 6 (?!). We treat any of 1..6 as a sign of novelty evaluation.
        if '___left_theory' not in locals():
            ___left_theory = False  # type: ignore
        if not ___left_theory and ply <= 40:  # only consider relatively early plies
            nags = mv.get("nags")
            if isinstance(nags, list) and any(isinstance(n, int) and 1 <= n <= 6 for n in nags):
                tags.append("leaving_theory")
                ___left_theory = True

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

        # Side pressure tags: large eval + elevated tension
        if eval_cp is not None and tension >= 0.4:
            if eval_cp >= 180:
                tags.append("white_pressure")
            elif eval_cp <= -180:
                tags.append("black_pressure")

        # --- Time pressure & clock gap tagging ---
        # Expect extractor to provide remaining clock after move for side that just moved:
        # mv['clock_after_seconds'] and side_to_move BEFORE move can be inferred from ply parity.
        clock_after = mv.get("clock_after_seconds")
        # We'll keep a rolling dict of last known clocks for each side so we can compute gap each ply.
        if '___clock_state' not in locals():
            ___clock_state = {"white": None, "black": None}  # type: ignore
        side_just_moved = 'white' if ply % 2 == 1 else 'black'
        if isinstance(clock_after, (int, float)):
            ___clock_state[side_just_moved] = clock_after
        w_clk = ___clock_state.get('white')
        b_clk = ___clock_state.get('black')
        if isinstance(w_clk, (int, float)) and isinstance(b_clk, (int, float)):
            gap = w_clk - b_clk  # positive means white ahead
            abs_gap = abs(gap)
            small = cfg["clock_gap_small"]; big = cfg["clock_gap_big"]; huge = cfg["clock_gap_huge"]
            if abs_gap >= small:
                if gap > 0:
                    if abs_gap >= huge:
                        tags.append("white_time_adv_huge")
                    elif abs_gap >= big:
                        tags.append("white_time_adv_big")
                    else:
                        tags.append("white_time_adv_small")
                else:
                    if abs_gap >= huge:
                        tags.append("black_time_adv_huge")
                    elif abs_gap >= big:
                        tags.append("black_time_adv_big")
                    else:
                        tags.append("black_time_adv_small")
            scramble_each = cfg["time_scramble_each_under"]
            scramble_total = cfg["time_scramble_total_under"]
            if w_clk <= scramble_each and b_clk <= scramble_each and (w_clk + b_clk) <= scramble_total:
                tags.append("time_scramble")

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

    # ---------------- Section metrics aggregation & phase tags -----------------
    # Build index boundaries
    if sections:
        # Append synthetic end sentinel
        indexed_sections = []
        for idx, sec in enumerate(sections):
            start_ply = sec["start_ply"]
            end_ply = sections[idx + 1]["start_ply"] - 1 if idx + 1 < len(sections) else moves_out[-1].ply if moves_out else start_ply
            indexed_sections.append({**sec, "end_ply": end_ply})
        # Compute metrics per section
        for sec in indexed_sections:
            seg_moves = [m for m in moves_out if sec["start_ply"] <= m.ply <= sec["end_ply"]]
            evals = [m.eval_cp for m in seg_moves if m.eval_cp is not None]
            if evals:
                avg_eval = statistics.mean(evals)
                first_eval = evals[0]; last_eval = evals[-1]
                trend = last_eval - first_eval
            else:
                avg_eval = None; trend = 0
            sec["avg_eval_cp"] = round(avg_eval, 1) if avg_eval is not None else None
            sec["eval_trend_cp"] = trend
            # Advantage phase tags
            adv_tag: Optional[str] = None
            if avg_eval is not None:
                if abs(avg_eval) >= 250:
                    adv_tag = ("white" if avg_eval > 0 else "black") + "_conversion_phase"
                elif abs(avg_eval) >= 120:
                    adv_tag = ("white" if avg_eval > 0 else "black") + "_advantage_building"
            if adv_tag:
                sec.setdefault("phase_tags", []).append(adv_tag)
            # Tactical phase detection via tension average
            if seg_moves:
                avg_tension = statistics.mean(m.tension for m in seg_moves)
                if avg_tension >= 0.55:
                    sec.setdefault("phase_tags", []).append("tactical_phase")
            # Desperate defense: strong negative trend crossing buckets
            if trend <= -200:
                sec.setdefault("phase_tags", []).append("black_desperate_defense")
            elif trend >= 200:
                sec.setdefault("phase_tags", []).append("white_desperate_defense")
        # Replace sections with enriched version (dropping end_ply to keep interface stable)
        sections = [{k: v for k, v in s.items() if k != 'end_ply'} for s in indexed_sections]

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
