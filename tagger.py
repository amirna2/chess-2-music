#!/usr/bin/env python3
"""
narrative_tagger.py

Analyzes chess game features and generates narrative structure tags for algorithmic music composition.
Takes output from feature_extractor.py and produces temporal narrative structure for a musical piece.
"""

import json
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
import numpy as np

@dataclass
class KeyMoment:
    ply: int
    second: int  # 1:1 mapping with ply
    score: int   # importance score
    type: str    # BLUNDER, BRILLIANT, TIME_PRESSURE, etc.
    side: str    # "white" or "black" who made the move

@dataclass
class NarrativeSection:
    name: str  # OPENING, MIDDLEGAME_1, etc.
    start_ply: int
    end_ply: int
    duration: str  # "0:00-0:18" format
    narrative: str  # POSITIONAL_SQUEEZE, TACTICAL_CHAOS, etc.
    tension: float = 0.0  # 0.0 to 1.0 tension level
    key_moments: List[KeyMoment] = field(default_factory=list)

@dataclass
class NarrativeStructure:
    total_plies: int
    duration_seconds: int
    game_result: str
    overall_narrative: str
    eco: str
    sections: List[NarrativeSection]

    def to_dict(self) -> Dict:
        return asdict(self)

class ChessNarrativeTagger:
    def __init__(self, game_data: Union[Dict, Any], debug: bool = False):
        """Initialize with game data (either GameFeatures object or dict from JSON)"""
        # Handle both dictionary and object inputs
        if isinstance(game_data, dict):
            self.moves = game_data['moves']
            self.metadata = game_data['metadata']
            self.metrics = game_data['metrics']
        else:
            # Assuming it's a GameFeatures object
            self.moves = [m.to_dict() for m in game_data.moves]
            self.metadata = asdict(game_data.metadata)
            self.metrics = asdict(game_data.metrics)

        self.total_plies = self.metrics['total_plies']
        self.debug = debug

        # Validate that we have moves to process
        if self.total_plies == 0:
            raise ValueError("Error: No moves found in game data (total_plies is 0). Cannot generate narrative structure for an empty game.")

    def detect_sections(self) -> List[Tuple[str, int, int]]:
        """Detect major game sections (opening, middlegame, endgame)"""
        sections = []

        # Use the derived metrics directly
        both_castled = self.metrics.get('both_castled_ply')
        queens_off = self.metrics.get('both_queens_off_ply')

        # Opening ends when both castle or at ply 28 (move 14) - Lichess standard
        opening_end = min(both_castled + 2, 28) if both_castled else 28
        opening_end = min(opening_end, self.total_plies)

        # Endgame starts when queens are off or last 25% of game
        if queens_off:
            endgame_start = queens_off
        else:
            endgame_start = max(opening_end + 1, int(self.total_plies * 0.75))

        # Handle short games (less than typical opening length)
        if self.total_plies <= 28:
            sections.append(("OPENING", 1, min(20, self.total_plies)))
            if self.total_plies > 20:
                sections.append(("MIDDLEGAME", 21, self.total_plies))
        else:
            sections.append(("OPENING", 1, opening_end))

            if endgame_start > opening_end + 1:
                middlegame_length = endgame_start - opening_end - 1
                # Split middlegame if it's long enough (using move 25/ply 50 as guide)
                if middlegame_length > 40:  # Increased threshold for splitting
                    mid_point = opening_end + middlegame_length // 2
                    sections.append(("MIDDLEGAME_1", opening_end + 1, mid_point))
                    sections.append(("MIDDLEGAME_2", mid_point + 1, endgame_start - 1))
                else:
                    sections.append(("MIDDLEGAME", opening_end + 1, endgame_start - 1))

            if endgame_start <= self.total_plies:
                sections.append(("ENDGAME", endgame_start, self.total_plies))

        return sections


    def calculate_tension(self, section_moves: List[Dict]) -> float:
        """Calculate tension level (0.0 to 1.0) for a section with key moment density"""
        tension = 0.0

        # 1. Evaluation volatility (swings = tension)
        evals = [m.get('eval_cp', 0) for m in section_moves if m.get('eval_cp')]
        if len(evals) > 1:
            volatility = np.std(evals) / 100.0  # in pawns
            tension += min(volatility / 2.0, 0.3)  # Cap at 0.3

        # 2. Balance (close games = tension)
        if evals:
            avg_eval = abs(np.mean(evals)) / 100.0
            if avg_eval < 0.5:  # Very balanced
                tension += 0.3
            elif avg_eval < 1.0:  # Slightly favored
                tension += 0.2

        # 3. Time pressure (only if clock data available)
        clock_values = [m.get('clock_after_seconds') for m in section_moves if m.get('clock_after_seconds') is not None]
        if clock_values:
            avg_clock = np.mean(clock_values)
            if avg_clock < 300:  # Under 5 minutes
                tension += 0.3
            elif avg_clock < 600:  # Under 10 minutes
                tension += 0.2

        # 4. Tactical density
        captures = sum(1 for m in section_moves if m.get('is_capture'))
        checks = sum(1 for m in section_moves if m.get('is_check'))
        tactical_density = (captures + checks * 2) / len(section_moves)
        tension += min(tactical_density, 0.2)

        # 5. Critical decisions (long thinks, if EMT data available)
        emt_values = [m.get('emt_seconds') for m in section_moves if m.get('emt_seconds') is not None]
        if emt_values:
            deep_thinks = sum(1 for emt in emt_values if emt > 600)
            if deep_thinks > 0:
                tension += 0.1

        # 6. Key moment density - simple relationship
        key_moments = sum(1 for m in section_moves
                         if self.calculate_moment_score(m, None)[0] >= 3)
        if key_moments > 0:
            moment_density = key_moments / len(section_moves)
            tension += min(moment_density * 0.2, 0.15)  # Small boost from key moments

        # 7. Overall tension level rounded and capped
        tension = min(round(tension, 2), 1.0)
        return tension


    def calculate_moment_score(self, move: Dict, prev_move: Optional[Dict]) -> Tuple[int, str]:
        """Enhanced scoring to capture more subtle events in quiet positions"""
        score = 0
        moment_type = "MOVE"

        # EXISTING EVALUATION SWING DETECTION
        if prev_move and move.get('eval_cp') is not None and prev_move.get('eval_cp') is not None:
            eval_change = abs(move['eval_cp'] - prev_move['eval_cp']) / 100.0  # Convert to pawns

            if eval_change > 2.0:
                score += 4
                moment_type = "GAME_CHANGING"
            elif eval_change > 1.0:
                score += 3
                moment_type = "CRITICAL_SWING"
            elif eval_change > 0.5:
                score += 2
                moment_type = "SIGNIFICANT_SHIFT"
            elif eval_change > 0.3:
                score += 1

        # Handle mate evaluations
        if move.get('eval_mate') is not None:
            score += 5
            moment_type = "MATE_SEQUENCE"

        # Time usage patterns
        emt = move.get('emt_seconds')
        if emt:
            if emt > 900:  # 15+ minutes
                score += 3
                if moment_type == "MOVE":
                    moment_type = "DEEP_THINK"
            elif emt > 600:  # 10+ minutes
                score += 2
            elif emt > 300:  # 5+ minutes
                score += 1

        clock = move.get('clock_after_seconds')
        if clock:
            if clock < 180:  # Under 3 minutes
                score += 2
                if moment_type == "MOVE":
                    moment_type = "TIME_PRESSURE"
            elif clock < 300:  # Under 5 minutes
                score += 1

        # NAG codes (annotations)
        nag_codes = move.get('nag_codes', [])
        if nag_codes:
            if 4 in nag_codes:  # ?? blunder
                score += 4
                moment_type = "BLUNDER"
            elif 3 in nag_codes:  # !! brilliant
                score += 4
                moment_type = "BRILLIANT"
            elif 2 in nag_codes:  # ? mistake
                score += 3
                moment_type = "MISTAKE"
            elif 6 in nag_codes:  # ?! dubious/inaccuracy
                score += 3
                moment_type = "INACCURACY"
            elif 5 in nag_codes:  # !? interesting
                score += 2
                if moment_type == "MOVE":
                    moment_type = "INTERESTING"
            elif 1 in nag_codes:  # ! good
                score += 3
                moment_type = "STRONG"

        # Tactical elements
        if move.get('is_check'):
            score += 1
            if score >= 3:
                moment_type = "KING_ATTACK"

        if move.get('is_capture'):
            score += 1
            if prev_move and prev_move.get('is_capture'):
                moment_type = "TACTICAL_SEQUENCE"

        if move.get('is_mate'):
            score += 5
            moment_type = "CHECKMATE"

        # Special moves
        kind = move.get('kind')
        if kind in ['CASTLE_KING', 'CASTLE_QUEEN']:
            score += 1
            if moment_type == "MOVE":
                moment_type = "CASTLING"

        if move.get('is_promotion'):
            score += 3
            moment_type = "PROMOTION"

        # NEW SCORING FOR QUIET POSITIONS
        # Only apply these if we haven't found a more significant event
        if score < 2:
            ply = move.get('ply', 0)

            # Pawn breaks and advances
            if move.get('piece') == 'P':
                to_rank = int(move['to_sq'][1])
                if ply > 10:  # After opening
                    if to_rank >= 5 and move['side'] == 'white':
                        score += 1
                        if moment_type == "MOVE":
                            moment_type = "PAWN_ADVANCE"
                    elif to_rank <= 4 and move['side'] == 'black':
                        score += 1
                        if moment_type == "MOVE":
                            moment_type = "PAWN_ADVANCE"

                # Central pawn moves in opening
                if ply <= 20 and move['to_sq'] in ['e4', 'e5', 'd4', 'd5']:
                    score += 1
                    if moment_type == "MOVE":
                        moment_type = "CENTER_CONTROL"

            # Piece development and maneuvers
            if move.get('piece') in ['N', 'B']:
                # Long-distance maneuvers
                if move.get('move_distance', 0) >= 4:
                    score += 1
                    if moment_type == "MOVE":
                        moment_type = "PIECE_MANEUVER"

                # Developing to active squares in opening
                if ply <= 15:
                    active_squares = ['c3', 'f3', 'c6', 'f6', 'e5', 'd5', 'b5', 'g5']
                    if move['to_sq'] in active_squares:
                        score += 1
                        if moment_type == "MOVE":
                            moment_type = "DEVELOPMENT"

            # Rook activity
            if move.get('piece') == 'R':
                # Rook to open file or 7th rank
                if move['to_sq'][1] in '78' or move['to_sq'][0] in 'de':
                    score += 1
                    if moment_type == "MOVE":
                        moment_type = "ROOK_ACTIVATION"

                # Doubling rooks
                if prev_move and prev_move.get('piece') == 'R':
                    if move['to_sq'][0] == prev_move['to_sq'][0]:  # Same file
                        score += 1
                        moment_type = "ROOKS_DOUBLED"

            # Queen centralization
            if move.get('piece') == 'Q' and ply > 20:
                if move['to_sq'] in ['d4', 'd5', 'e4', 'e5', 'c4', 'c5', 'f4', 'f5']:
                    score += 1
                    if moment_type == "MOVE":
                        moment_type = "QUEEN_CENTRALIZED"

            # First exchange of the game
            if move.get('is_capture') and not any(m.get('is_capture') for m in self.moves[:ply-1]):
                score += 2
                moment_type = "FIRST_EXCHANGE"

            # Breaking symmetry
            if ply <= 10 and move.get('piece') and move['to_sq'][0] not in 'de':
                score += 1
                if moment_type == "MOVE":
                    moment_type = "ASYMMETRY"

        # Clock milestones (enhance existing time pressure detection)
        if clock and prev_move:
            prev_clock = prev_move.get('clock_after_seconds', 7200)

            # First time crossing important thresholds
            if prev_clock >= 600 and clock < 600:  # Under 10 minutes
                score += 1
                if moment_type == "MOVE":
                    moment_type = "TIME_MILESTONE"
            elif prev_clock >= 60 and clock < 60:  # Under 1 minute
                score += 2
                if moment_type == "MOVE":
                    moment_type = "TIME_SCRAMBLE"

        # Endgame transition markers
        if move.get('both_queens_off_after') and (not prev_move or not prev_move.get('both_queens_off_after')):
            score += 2
            moment_type = "QUEENS_TRADED"

        return score, moment_type

    def detect_key_moments(self) -> List[KeyMoment]:
        """Identify key moments throughout the game"""
        key_moments = []

        for i, move in enumerate(self.moves):
            prev_move = self.moves[i-1] if i > 0 else None
            score, moment_type = self.calculate_moment_score(move, prev_move)

            if score >= 3:  # Threshold for significance
                key_moments.append(KeyMoment(
                    ply=move['ply'],
                    second=move['ply'],  # 1:1 mapping
                    side=move['side'],
                    score=score,
                    type=moment_type
                ))

        return key_moments

    def classify_section_narrative(self, section_moves: List[Dict]) -> str:
        """Determine the narrative character of a game section"""
        if not section_moves:
            return "UNKNOWN"

        # Calculate metrics
        evals_cp = [m.get('eval_cp') for m in section_moves if m.get('eval_cp') is not None]

        if not evals_cp:
            return "COMPLEX_POSITION"

        # Evaluation trend (in pawns)
        eval_trend = (evals_cp[-1] - evals_cp[0]) / 100.0 if len(evals_cp) > 1 else 0
        eval_volatility = (max(evals_cp) - min(evals_cp)) / 100.0 if evals_cp else 0

        # Other metrics
        captures = sum(1 for m in section_moves if m.get('is_capture'))
        capture_density = captures / len(section_moves)

        # Calculate average thinking time (only if EMT data available)
        emt_values = [m.get('emt_seconds') for m in section_moves if m.get('emt_seconds') is not None]
        avg_time = sum(emt_values) / len(emt_values) if emt_values else 0

        checks = sum(1 for m in section_moves if m.get('is_check'))

        # Enhanced king hunt detection: check if kings are in vulnerable positions during checks
        vulnerable_checks = 0
        for m in section_moves:
            if m.get('is_check'):
                # The checking piece's position can indicate where the checked king is
                # If checking piece is on rank 1-3, white king is likely trapped low
                # If checking piece is on rank 6-8, black king is likely trapped high
                to_rank = int(m['to_sq'][1])
                checking_side = m['side']

                if checking_side == 'black' and to_rank <= 3:
                    # Black checking on low ranks = white king trapped
                    vulnerable_checks += 1
                elif checking_side == 'white' and to_rank >= 6:
                    # White checking on high ranks = black king trapped
                    vulnerable_checks += 1

        # Count move types (tactical elements)
        tactical_moves = sum(1 for m in section_moves if m.get('kind') in [
            'CHECK', 'CAPTURE', 'PROMOTION', 'CHECKMATE'
        ])

        # Classification rules
        if eval_volatility > 2.0 and capture_density > 0.3:
            return "TACTICAL_CHAOS"
        elif checks >= 3 and vulnerable_checks >= 2:
            # Enhanced: Need both 3+ checks AND at least 2 on vulnerable ranks
            return "KING_HUNT"
        elif checks >= 3:
            # Fallback: Many checks but not necessarily a hunt
            return "KING_ATTACK"
        elif any(m.get('is_mate') for m in section_moves):
            return "MATING_ATTACK"
        elif eval_trend > 1.5:
            return "CRUSHING_ATTACK"
        elif eval_trend < -1.5:
            return "DESPERATE_DEFENSE"
        elif eval_trend > 0.5 and eval_volatility < 0.5:
            return "POSITIONAL_SQUEEZE"
        elif abs(eval_trend) < 0.3 and avg_time > 300:
            return "TENSE_EQUILIBRIUM"
        elif capture_density > 0.4:
            # High capture density - distinguish between tactical battle and liquidation
            # Get final evaluation to see if position is lopsided after exchanges
            final_eval = evals_cp[-1] / 100.0 if evals_cp else 0

            # Check if queens were traded (indicates simplification)
            queen_trades = sum(1 for m in section_moves if m.get('is_capture') and m.get('piece') == 'Q')

            # Liquidation: balanced eval after exchanges, especially if queens traded
            if abs(final_eval) < 0.5 and (queen_trades > 0 or abs(eval_trend) < 0.3):
                return "LIQUIDATION"
            # True tactical battle: someone gained significant advantage through tactics
            else:
                return "TACTICAL_BATTLE"
        elif capture_density < 0.05 and avg_time < 60:
            return "QUIET_MANEUVERING"
        elif avg_time > 400:
            return "CRITICAL_DECISIONS"
        else:
            return "COMPLEX_STRUGGLE"

    def generate_structure(self) -> NarrativeStructure:
        """Generate narrative structure with guaranteed events per section"""
        sections_raw = self.detect_sections()

        # Collect ALL moments with scores (not just high-scoring ones)
        all_moments_data = []
        for i, move in enumerate(self.moves):
            prev_move = self.moves[i-1] if i > 0 else None
            score, moment_type = self.calculate_moment_score(move, prev_move)

            # Assign to section
            section_idx = None
            for idx, (_, start, end) in enumerate(sections_raw):
                if start <= move['ply'] <= end:
                    section_idx = idx
                    break

            all_moments_data.append({
                'ply': move['ply'],
                'side': move['side'],
                'score': score,
                'type': moment_type,
                'section_idx': section_idx
            })

        sections = []
        for idx, (section_name, start, end) in enumerate(sections_raw):
            section_moves = [m for m in self.moves if start <= m['ply'] <= end]
            section_moments_data = [m for m in all_moments_data if m['section_idx'] == idx]

            # Dynamic thresholds and limits based on section
            if 'OPENING' in section_name:
                score_threshold = 2.0  # Lower for opening
                min_moments = 2
                max_moments = 3
            elif 'MIDDLEGAME' in section_name:
                score_threshold = 2.5  # Medium for middlegame
                min_moments = 3
                max_moments = 4
            elif 'ENDGAME' in section_name:
                score_threshold = 2.5  # Keep reasonable for endgame
                min_moments = 2
                max_moments = 4
            else:
                score_threshold = 2.5
                min_moments = 2
                max_moments = 3

            # Filter by threshold
            significant_moments = [m for m in section_moments_data if m['score'] >= score_threshold]

            # If not enough moments, add the best available
            if len(significant_moments) < min_moments and section_moments_data:
                # Sort all moments by score
                sorted_moments = sorted(section_moments_data, key=lambda x: x['score'], reverse=True)
                # Take top moments up to min_moments
                significant_moments = sorted_moments[:min_moments]

            # Convert to KeyMoment objects (limit to max_moments)
            key_moments = []
            for m in significant_moments[:max_moments]:
                key_moments.append(KeyMoment(
                    ply=m['ply'],
                    second=m['ply'],
                    side=m['side'],
                    score=m['score'],
                    type=m['type']
                ))

            # Sort by ply for chronological order
            key_moments = sorted(key_moments, key=lambda x: x.ply)

            # Calculate tension with adjustment for sparse moments
            base_tension = self.calculate_tension(section_moves)

            # If we had to add artificial moments, slightly reduce tension
            natural_moments = len([m for m in section_moments_data if m['score'] >= score_threshold])
            if natural_moments < min_moments and len(section_moments_data) > 0:
                # Reduce tension to reflect quieter play
                base_tension = max(0.3, base_tension * 0.85)

            sections.append(NarrativeSection(
                name=section_name,
                start_ply=start,
                end_ply=end,
                duration=f"{start-1:02d}:{end:02d}",
                narrative=self.classify_section_narrative(section_moves),
                tension=base_tension,
                key_moments=key_moments
            ))

        # Determine overall narrative
        overall = self.classify_game_narrative(sections)

        return NarrativeStructure(
            total_plies=self.total_plies,
            duration_seconds=self.total_plies,
            game_result=self.metadata.get('result', ''),
            overall_narrative=overall,
            eco=self.metadata.get('eco', 'A00'),
            sections=sections
        )

    def classify_game_narrative(self, sections: List[NarrativeSection]) -> str:
        """Determine overall game narrative using scoring system"""
        narratives = [s.narrative for s in sections]
        result = self.metadata.get('result')

        # Calculate scores for each narrative type
        scores = {
            'DEATH_SPIRAL': self.score_death_spiral(narratives, result),
            'ATTACKING_MASTERPIECE': self.score_attacking_masterpiece(narratives, result),
            'TACTICAL_MASTERPIECE': self.score_tactical_masterpiece(narratives, result),
            'DOMINATION': self.score_domination(narratives, result),
            'TACTICAL_THRILLER': self.score_tactical_thriller(narratives, result),
            'QUIET_PRECISION': self.score_quiet_precision(narratives, result),
            'PEACEFUL_DRAW': self.score_peaceful_draw(narratives, result),
            'ATTACKING_GAME': self.score_attacking_game(narratives, result),
            'HEROIC_DEFENSE': self.score_heroic_defense(narratives, result),
            'FIGHTING_DEFEAT': self.score_fighting_defeat(narratives, result),
            'TUMBLING_DEFEAT': self.score_tumbling_defeat(narratives, result),
            'STRATEGIC_BATTLE': self.score_strategic_battle(narratives, result),
            'COMPLEX_GAME': self.score_complex_game(narratives, result),
        }

        # Debug output: show all scores sorted by value
        if hasattr(self, 'debug') and self.debug:
            print("\n=== NARRATIVE CLASSIFICATION DEBUG ===")
            print(f"Game result: {result}")
            print(f"Section narratives: {narratives}")
            print("\nScoring breakdown:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for narrative, score in sorted_scores:
                print(f"  {narrative:20s}: {score:2d}")
            print(f"\nSelected: {max(scores, key=scores.get)}")
            print("=" * 40)

        # Return the narrative type with the highest score
        return max(scores, key=scores.get)

    def detect_death_spiral(self) -> bool:
        """Detect death spiral pattern: gradual decline + time pressure + multiple mistakes"""

        # Determine losing side from result
        result = self.metadata.get('result', '')
        if result == '1-0':
            losing_side = 'black'
        elif result == '0-1':
            losing_side = 'white'
        else:
            return False

        # Get moves for losing side only
        losing_moves = [m for m in self.moves if m['side'] == losing_side]

        # Need substantial game length
        if len(losing_moves) < 20:
            return False

        # 1. Check for gradual evaluation decline (no big single blunder)
        evals = [m.get('eval_cp') for m in losing_moves if m.get('eval_cp') is not None]
        if len(evals) < 15:
            return False

        # Adjust eval signs for losing side perspective
        if losing_side == 'black':
            evals = [-e for e in evals]

        # Find the point where position became clearly worse
        decline_start = None
        for i, eval_cp in enumerate(evals):
            if eval_cp < -50:  # 0.5 pawns worse
                decline_start = i
                break

        if decline_start is None or decline_start > len(evals) - 10:
            return False

        # Check that decline was gradual (no huge single drop)
        max_single_drop = 0
        for i in range(decline_start, len(evals) - 1):
            drop = evals[i] - evals[i + 1]  # How much worse it got
            max_single_drop = max(max_single_drop, drop)

        # If there was a huge single blunder (>3 pawns), not a death spiral
        if max_single_drop > 300:
            return False

        # 2. Check for multiple mistakes/inaccuracies
        mistake_count = 0
        for m in losing_moves:
            nags = m.get('nag_codes', [])
            if any(nag in [2, 4, 6] for nag in nags):  # ?, ??, ?!
                mistake_count += 1

        if mistake_count < 3:
            return False

        # 3. Check for time pressure in final third of game
        final_third_start = len(losing_moves) * 2 // 3
        final_moves = losing_moves[final_third_start:]

        time_pressure_moves = 0
        low_clock_moves = 0

        for m in final_moves:
            emt = m.get('emt_seconds')
            clock = m.get('clock_after_seconds')

            # Fast moves under pressure
            if emt and emt < 30:  # Very quick moves
                time_pressure_moves += 1

            # Low clock readings
            if clock and clock < 300:  # Under 5 minutes
                low_clock_moves += 1

        # Time pressure: either many quick moves OR multiple low-clock situations
        has_time_pressure = (time_pressure_moves >= 3) or (low_clock_moves >= 2)

        return has_time_pressure

    def score_death_spiral(self, narratives: List[str], result: str) -> int:
        """Score for death spiral pattern"""
        if result not in ['1-0', '0-1']:
            return 0
        return 10 if self.detect_death_spiral() else 0

    def score_attacking_masterpiece(self, narratives: List[str], result: str) -> int:
        """Score for attacking masterpiece - requires forced mate sequences"""
        score = 0

        # check for blunder nags
        blunder_moves = [m for m in self.moves if 4 in m.get('nag_codes', [])]
        if len(blunder_moves) > 0:
            return 0  # No blunders allowed for masterpiece

        # Check for forced mate evaluations (eval_mate field)
        mate_evals = [m.get('eval_mate') for m in self.moves if m.get('eval_mate') is not None]
        if mate_evals:
            score += 10  # Has forced mate sequences

        if 'MATING_ATTACK' in narratives:
            score += 8   # Section classified as mating attack

        if 'KING_HUNT' in narratives and score > 0:  # Only if already has mate elements
            score += 2   # Bonus for king hunt leading to mate

        if result in ['1-0', '0-1'] and score > 0:
            score += 1   # Decisive result

        return score

    def score_tactical_masterpiece(self, narratives: List[str], result: str) -> int:
        """Score for tactical masterpiece - brilliant moves and sacrifices"""
        score = 0

        # Count brilliant moves (NAG 3 = !!)
        brilliant_moves = [m for m in self.moves if 3 in m.get('nag_codes', [])]

        if len(brilliant_moves) >= 3:
            score += 10  # Multiple brilliant moves = masterpiece
        elif len(brilliant_moves) == 2:
            score += 8   # Two brilliant moves = very strong
        elif len(brilliant_moves) == 1:
            score += 5   # Single brilliant move = notable

        # Bonus for decisive result (brilliance should lead to win)
        if result in ['1-0', '0-1'] and score > 0:
            score += 2

        # Bonus for tactical elements in sections
        if ('TACTICAL_CHAOS' in narratives or 'TACTICAL_BATTLE' in narratives) and score > 0:
            score += 1

        return score

    def score_domination(self, narratives: List[str], result: str) -> int:
        """Score for positional domination"""
        score = 0
        if 'CRUSHING_ATTACK' in narratives:
            score += 6
        if 'POSITIONAL_SQUEEZE' in narratives:
            score += 6
        if result in ['1-0', '0-1']:
            score += 3
        else:
            return 0  # Must be decisive
        return score

    def score_tactical_thriller(self, narratives: List[str], result: str) -> int:
        """Score for tactical thriller"""
        score = 0
        if 'TACTICAL_CHAOS' in narratives:
            score += 6
        if 'TACTICAL_BATTLE' in narratives:
            score += 5
        if 'KING_HUNT' in narratives:
            score += 2
        return score

    def score_quiet_precision(self, narratives: List[str], result: str) -> int:
        """Score for quiet precision - technical draws with minimal eval variance"""
        if result != '1/2-1/2':
            return 0  # Must be a draw

        score = 0

        # Check evaluation stability (flat line around 0.00)
        evals = [m.get('eval_cp', 0) for m in self.moves if m.get('eval_cp') is not None]
        if evals:
            # Calculate mean and standard deviation in centipawns
            import statistics
            mean_eval = statistics.mean(evals)
            std_eval = statistics.stdev(evals) if len(evals) > 1 else 0

            # High score for stable, balanced evaluations
            if abs(mean_eval) < 25:  # Mean within 0.25 pawns of equality
                score += 6
                if std_eval < 75:  # Low volatility (less than 0.75 pawn swings)
                    score += 6  # Very stable = high precision
                elif std_eval < 150:  # Moderate volatility
                    score += 3

        # Bonus for quiet section narratives
        quiet_sections = sum(1 for n in narratives if n in [
            'COMPLEX_STRUGGLE', 'QUIET_MANEUVERING', 'TENSE_EQUILIBRIUM'
        ])
        total_sections = len(narratives)

        if quiet_sections == total_sections and total_sections > 0:
            score += 4  # All sections were quiet/precise
        elif quiet_sections >= total_sections * 0.75:
            score += 2  # Mostly quiet sections

        # Penalty for dramatic elements
        if any(n in ['TACTICAL_CHAOS', 'KING_HUNT', 'MATING_ATTACK'] for n in narratives):
            score = max(0, score - 4)

        return score

    def score_peaceful_draw(self, narratives: List[str], result: str) -> int:
        """Score for peaceful draw"""
        score = 0
        if 'LIQUIDATION' in narratives and result == '1/2-1/2':
            score += 7
        if 'QUIET_MANEUVERING' in narratives and result == '1/2-1/2':
            score += 4
        return score

    def score_attacking_game(self, narratives: List[str], result: str) -> int:
        """Score for attacking game"""
        score = 0
        if 'KING_HUNT' in narratives:
            score += 5
        if 'MATING_ATTACK' in narratives:
            score += 3
        return score

    def score_heroic_defense(self, narratives: List[str], result: str) -> int:
        """Score for heroic defense"""
        score = 0
        if 'DESPERATE_DEFENSE' in narratives and result == '1/2-1/2':
            score += 7
        return score

    def score_fighting_defeat(self, narratives: List[str], result: str) -> int:
        """Score for fighting defeat"""
        score = 0
        if 'DESPERATE_DEFENSE' in narratives and result in ['1-0', '0-1']:
            score += 6
        return score

    def score_tumbling_defeat(self, narratives: List[str], result: str) -> int:
        """Score for tumbling defeat - gradual deterioration through mistakes"""
        if result not in ['1-0', '0-1']:
            return 0  # Must be decisive

        score = 0

        # Must have desperate defense (position deteriorating)
        if 'DESPERATE_DEFENSE' not in narratives:
            return 0

        # Check for multiple mistakes/inaccuracies by losing side
        losing_side = 'white' if result == '0-1' else 'black'
        losing_mistakes = 0

        for move in self.moves:
            if move['side'] == losing_side:
                nag_codes = move.get('nag_codes', [])
                # Count mistakes (?), blunders (??), and inaccuracies (?!)
                if any(nag in [2, 4, 6] for nag in nag_codes):
                    losing_mistakes += 1

        # Score based on mistake count
        if losing_mistakes >= 4:
            score += 8  # Many mistakes = clear tumbling
        elif losing_mistakes >= 3:
            score += 6  # Multiple mistakes
        elif losing_mistakes >= 2:
            score += 4  # Some mistakes
        else:
            return 0  # Not enough mistakes for tumbling pattern

        # Bonus: Check if winner didn't need brilliance (no brilliant moves)
        winner_side = 'white' if result == '1-0' else 'black'
        brilliant_moves = sum(1 for m in self.moves
                            if m['side'] == winner_side and 3 in m.get('nag_codes', []))

        if brilliant_moves == 0:
            score += 2  # Winner didn't need brilliance, just steady pressure

        # Penalty: Reduce score if this looks more like death spiral
        if self.detect_death_spiral():
            score = max(0, score - 3)  # Death spiral takes precedence

        return score

    def score_strategic_battle(self, narratives: List[str], result: str) -> int:
        """Score for strategic battle"""
        score = 0
        quiet_count = sum(1 for n in narratives if n in ['QUIET_MANEUVERING', 'TENSE_EQUILIBRIUM'])
        if quiet_count == len(narratives) and quiet_count > 0:
            score += 5
        return score

    def score_complex_game(self, narratives: List[str], result: str) -> int:
        """Default fallback score"""
        return 1  # Always has minimal score as fallback


def main():
    if len(sys.argv) < 2:
        print("Usage: python narrative_tagger.py <game_features.json> [--output narrative.json] [--debug]")
        sys.exit(1)

    # Load game features from JSON
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        game_features = json.load(f)

    # Check for debug flag
    debug_mode = '--debug' in sys.argv

    # Generate narrative structure
    tagger = ChessNarrativeTagger(game_features, debug=debug_mode)
    structure = tagger.generate_structure()

    # Output
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_file = sys.argv[idx + 1]
        with open(output_file, 'w') as f:
            json.dump(structure.to_dict(), f, indent=2)
        print(f"Narrative structure written to {output_file}")
    else:
        # Pretty print to console
        print(json.dumps(structure.to_dict(), indent=2))


if __name__ == "__main__":
    main()
