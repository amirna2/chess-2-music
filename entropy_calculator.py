#!/usr/bin/env python3
"""
Entropy Calculator for Chess Music Composition
Inspired by Laurie Spiegel's information theory approach to musical structure

Calculates informational entropy from chess game features to control
musical predictability, tension, and emotional arc over time.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


class ChessEntropyCalculator:
    """
    Calculate informational entropy from chess positions using:
    - Evaluation volatility (position uncertainty)
    - Tactical density (complexity of possibilities)
    - Decision time (computational difficulty)
    - Move pattern diversity (Shannon entropy)
    """

    def __init__(self, moves: List[Dict], debug: bool = False):
        self.moves = moves
        self.debug = debug

    def calculate_eval_entropy(self, start_ply: int, end_ply: int, window_size: int = 5) -> np.ndarray:
        """
        Calculate entropy from evaluation volatility using sliding window.

        High volatility = high entropy (uncertain, many possibilities)
        Low volatility = low entropy (clear, forced)

        Args:
            start_ply: Starting ply of section
            end_ply: Ending ply of section
            window_size: Number of moves to look at for local volatility

        Returns:
            Array of normalized entropy values (0-1) for each ply in range
        """
        section_moves = [m for m in self.moves if start_ply <= m['ply'] <= end_ply]

        if not section_moves:
            return np.array([0.5])  # Default medium entropy

        # Extract eval sequence
        evals = []
        for m in section_moves:
            if m.get('eval_mate') is not None:
                # Mate in N: very low entropy (forced, inevitable)
                # Map mate distance to eval: M1 = ±20 pawns, M5 = ±10 pawns, etc.
                mate_dist = abs(m['eval_mate'])
                mate_eval = (10000 / max(1, mate_dist)) * (1 if m['eval_mate'] > 0 else -1)
                evals.append(mate_eval)
            elif m.get('eval_cp') is not None:
                evals.append(m['eval_cp'])
            else:
                # No eval: use previous if available, else 0
                evals.append(evals[-1] if evals else 0)

        if len(evals) < 2:
            return np.full(len(section_moves), 0.5)

        # Calculate rolling standard deviation (local volatility)
        entropy_values = np.zeros(len(evals))

        for i in range(len(evals)):
            # Get window centered on current position
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(evals), i + window_size // 2 + 1)
            window = evals[start_idx:end_idx]

            if len(window) > 1:
                # Standard deviation in centipawns
                volatility = np.std(window)

                # Normalize: 0-200cp volatility -> 0-1 entropy
                # 0cp std = 0 entropy (perfectly stable)
                # 200cp std = 1 entropy (very volatile)
                # 400cp+ std = capped at 1 (extreme chaos)
                normalized = min(1.0, volatility / 200.0)
                entropy_values[i] = normalized
            else:
                entropy_values[i] = 0.5

        return entropy_values

    def calculate_tactical_entropy(self, start_ply: int, end_ply: int, window_size: int = 5) -> np.ndarray:
        """
        Calculate entropy from tactical density and move diversity.

        More captures/checks/threats = more tactical complexity = higher entropy
        Quiet positions = lower entropy

        Args:
            start_ply: Starting ply of section
            end_ply: Ending ply of section
            window_size: Number of moves for local density calculation

        Returns:
            Array of normalized entropy values (0-1) for each ply in range
        """
        section_moves = [m for m in self.moves if start_ply <= m['ply'] <= end_ply]

        if not section_moves:
            return np.array([0.5])

        entropy_values = np.zeros(len(section_moves))

        for i, move in enumerate(section_moves):
            # Get local window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(section_moves), i + window_size // 2 + 1)
            window = section_moves[start_idx:end_idx]

            # Count tactical elements
            captures = sum(1 for m in window if m.get('is_capture'))
            checks = sum(1 for m in window if m.get('is_check'))
            promotions = sum(1 for m in window if m.get('is_promotion'))

            # Tactical density: 0 (all quiet) to 1 (all tactical)
            tactical_count = captures + checks * 1.5 + promotions * 2  # Weight checks/promos higher
            max_possible = len(window) * 2.5  # Max if every move is capture + check + promo
            tactical_density = min(1.0, tactical_count / max_possible)

            # Calculate move type diversity (Shannon entropy)
            move_kinds = [m.get('kind', 'NORMAL') for m in window]
            move_entropy = self._shannon_entropy(move_kinds)

            # Combine: tactical density + pattern diversity
            combined = tactical_density * 0.6 + move_entropy * 0.4
            entropy_values[i] = combined

        return entropy_values

    def calculate_time_entropy(self, start_ply: int, end_ply: int) -> np.ndarray:
        """
        Calculate entropy from thinking time patterns.

        Long thinks = difficult positions = high computational entropy
        Quick moves = obvious/forced moves = low entropy

        Args:
            start_ply: Starting ply of section
            end_ply: Ending ply of section

        Returns:
            Array of normalized entropy values (0-1) for each ply in range
        """
        section_moves = [m for m in self.moves if start_ply <= m['ply'] <= end_ply]

        if not section_moves:
            return np.array([0.5])

        entropy_values = np.zeros(len(section_moves))

        # Check if we have EMT data
        has_emt = any(m.get('emt_seconds') is not None for m in section_moves)

        if not has_emt:
            # No time data: return medium entropy
            return np.full(len(section_moves), 0.5)

        for i, move in enumerate(section_moves):
            emt = move.get('emt_seconds')

            if emt is None:
                # No time data for this move
                entropy_values[i] = 0.5
            else:
                # Normalize thinking time to entropy
                # 0-10 seconds = low entropy (quick, obvious)
                # 60+ seconds = high entropy (difficult decision)
                if emt < 10:
                    normalized = 0.1 + (emt / 10) * 0.3  # 0.1 to 0.4
                elif emt < 60:
                    normalized = 0.4 + ((emt - 10) / 50) * 0.4  # 0.4 to 0.8
                else:
                    normalized = min(1.0, 0.8 + (emt - 60) / 300)  # 0.8 to 1.0

                entropy_values[i] = normalized

        return entropy_values

    def calculate_combined_entropy(self, start_ply: int, end_ply: int,
                                   weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Calculate combined entropy curve from multiple sources.

        This is the main entropy curve that drives musical parameters.

        Args:
            start_ply: Starting ply of section
            end_ply: Ending ply of section
            weights: Optional dict with keys 'eval', 'tactical', 'time' (default: equal weight)

        Returns:
            Array of normalized entropy values (0-1) for each ply in range
        """
        if weights is None:
            weights = {'eval': 0.5, 'tactical': 0.4, 'time': 0.1}

        # Calculate component entropies
        eval_entropy = self.calculate_eval_entropy(start_ply, end_ply)
        tactical_entropy = self.calculate_tactical_entropy(start_ply, end_ply)
        time_entropy = self.calculate_time_entropy(start_ply, end_ply)

        # Ensure all arrays same length (should be, but defensive)
        min_len = min(len(eval_entropy), len(tactical_entropy), len(time_entropy))
        eval_entropy = eval_entropy[:min_len]
        tactical_entropy = tactical_entropy[:min_len]
        time_entropy = time_entropy[:min_len]

        # Weighted combination
        combined = (eval_entropy * weights['eval'] +
                   tactical_entropy * weights['tactical'] +
                   time_entropy * weights['time'])

        # Normalize to ensure 0-1 range
        combined = np.clip(combined, 0.0, 1.0)

        if self.debug:
            print(f"\n=== ENTROPY CALCULATION ===")
            print(f"Section: ply {start_ply} to {end_ply}")
            print(f"Eval entropy:    mean={np.mean(eval_entropy):.3f}, std={np.std(eval_entropy):.3f}")
            print(f"Tactical entropy: mean={np.mean(tactical_entropy):.3f}, std={np.std(tactical_entropy):.3f}")
            print(f"Time entropy:    mean={np.mean(time_entropy):.3f}, std={np.std(time_entropy):.3f}")
            print(f"Combined:        mean={np.mean(combined):.3f}, std={np.std(combined):.3f}")

        return combined

    def get_entropy_at_ply(self, ply: int, start_ply: int, entropy_curve: np.ndarray) -> float:
        """
        Get entropy value for a specific ply from precomputed curve.

        Args:
            ply: Ply number to query
            start_ply: Starting ply of the entropy curve
            entropy_curve: Precomputed entropy array

        Returns:
            Entropy value (0-1) at that ply, or 0.5 if out of range
        """
        idx = ply - start_ply
        if 0 <= idx < len(entropy_curve):
            return float(entropy_curve[idx])
        return 0.5

    def _shannon_entropy(self, sequence: List) -> float:
        """
        Calculate Shannon entropy of a sequence.

        H = -Σ p(x) * log2(p(x))

        Returns value normalized to 0-1 range based on max possible entropy
        for the sequence length.
        """
        if not sequence:
            return 0.0

        # Count frequencies
        counts = Counter(sequence)
        total = len(sequence)

        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy
        # Max entropy = log2(n) where n is number of unique items
        n_unique = len(counts)
        max_entropy = math.log2(n_unique) if n_unique > 1 else 1.0

        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, normalized)


def demonstrate_entropy(moves: List[Dict], section_name: str, start_ply: int, end_ply: int):
    """Demo function showing entropy calculation and visualization"""
    calc = ChessEntropyCalculator(moves, debug=True)

    entropy_curve = calc.calculate_combined_entropy(start_ply, end_ply)

    print(f"\n{section_name} entropy profile:")
    print("Ply    Entropy  Bar")
    print("-" * 40)

    section_moves = [m for m in moves if start_ply <= m['ply'] <= end_ply]

    for i, (move, entropy_val) in enumerate(zip(section_moves, entropy_curve)):
        bar_len = int(entropy_val * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"{move['ply']:3d}    {entropy_val:.3f}   {bar}  {move['san']}")


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python entropy_calculator.py <game_features.json>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        game_features = json.load(f)

    moves = game_features['moves']

    # Demo on first 20 moves
    demonstrate_entropy(moves, "Opening", 1, min(20, len(moves)))
