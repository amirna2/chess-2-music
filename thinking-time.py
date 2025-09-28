#!/usr/bin/env python3
# annotate_emt.py
# pip install python-chess
import re, csv, argparse
from datetime import timedelta
import chess.pgn

# Match  M:SS(.fff)  or  H:MM:SS(.fff)
CLK_RE = re.compile(r"\[%clk\s+(\d+:\d{2}(?::\d{2})?(?:\.\d+)?)\]")

TC_STAGE_RE     = re.compile(r"(\d+)\/(\d+):(\d+)(?:\+(\d+))?$")  # 40/7200:1800(+30)
TC_BASE_INC_RE  = re.compile(r"(\d+)\+(\d+)$")                    # 600+5
TC_BASE_ONLY_RE = re.compile(r"^\d+$")                            # 180

def clock_to_seconds(s: str) -> float:
    parts = s.split(":")
    if len(parts) == 2:       # M:SS(.fff)
        m = int(parts[0]); sec = float(parts[1])
        return m*60 + sec
    elif len(parts) == 3:     # H:MM:SS(.fff)
        h = int(parts[0]); m = int(parts[1]); sec = float(parts[2])
        return h*3600 + m*60 + sec
    raise ValueError(f"Bad clock '{s}'")

def fmt_hms(sec: float) -> str:
    return str(timedelta(seconds=max(0.0, sec)))

def parse_time_control(tc: str):
    tc = (tc or "").strip()
    m = TC_STAGE_RE.match(tc)
    if m:
        moves, base, bonus, inc = m.groups()
        return [
            {'moves': int(moves), 'init': int(base),  'inc': 0},
            {'moves': None,       'init': int(bonus), 'inc': int(inc) if inc else 0},
        ]
    m = TC_BASE_INC_RE.match(tc)
    if m:
        base, inc = map(int, m.groups())
        return [{'moves': None, 'init': base, 'inc': inc}]
    m = TC_BASE_ONLY_RE.match(tc)
    if m:
        base = int(m.group(0))
        return [{'moves': None, 'init': base, 'inc': 0}]
    # Fallback
    return [{'moves': 40, 'init': 7200, 'inc': 0},
            {'moves': None, 'init': 1800, 'inc': 30}]

def annotate_game_with_emt(game, clip_eps=0.6):
    # Check if EMT values are already present using python-chess's emt() method
    for node in game.mainline():
        if node.parent is None:
            continue
        if node.emt() is not None:
            return None  # Signal that EMT already exists

    # Check if clock times exist before doing anything else
    has_clocks = False
    for node in game.mainline():
        if node.parent is None:
            continue
        if node.clock() is not None:
            has_clocks = True
            break

    if not has_clocks:
        return []  # Signal that no clocks exist

    periods = parse_time_control(game.headers.get("TimeControl", ""))

    prev_clock = {'W': float(periods[0]['init']), 'B': float(periods[0]['init'])}
    pidx       = {'W': 0, 'B': 0}
    moved_in   = {'W': 0, 'B': 0}

    print(f"{'Move':<6} {'S':<1}  {'SAN':<22} {'EMT(s)':>9}  {'EMT':>10}  {'Clock after':>12}")
    print("-"*70)

    rows = []
    for node in game.mainline():
        if node.parent is None:
            continue

        board = node.parent.board()
        side = 'W' if board.turn else 'B'
        fullmove = board.fullmove_number
        label = f"{fullmove}{'.' if side=='W' else '...'}"
        san = board.san(node.move)

        clock_time = node.clock()
        if clock_time is None:
            continue
        curr = clock_time

        # current period
        period = periods[pidx[side]]
        inc = float(period['inc'])

        # count this move in the current period
        moved_in[side] += 1

        # Check if this completes a period (e.g., move 40)
        # The clock already includes the bonus time AND increment for new period
        if period['moves'] is not None and moved_in[side] == period['moves']:
            if pidx[side] + 1 < len(periods):
                # Both bonus time and new period's increment are in the clock reading
                next_period = periods[pidx[side] + 1]
                bonus_time = float(next_period['init'])
                next_inc = float(next_period['inc'])
                total_time_added = bonus_time + next_inc
                # EMT = prev + current_period_inc + total_time_added - curr
                emt = prev_clock[side] + inc + total_time_added - curr
            else:
                # No more periods
                emt = prev_clock[side] + inc - curr
        else:
            # Normal case: EMT = prev + inc - curr
            emt = prev_clock[side] + inc - curr
        if emt < 0 and abs(emt) <= clip_eps:
            emt = 0.0

        node.set_emt(emt)

        # Update prev_clock for next calculation
        # After a period boundary, the clock includes bonus so store as-is
        prev_clock[side] = curr

        print(f"{label:<6} {side:<1}  {san:<22} {emt:>9.3f}  {fmt_hms(emt):>10}  {fmt_hms(curr):>12}")

        rows.append({"Move": fullmove, "Side": side, "SAN": san,
                     "EMT(s)": f"{emt:.3f}", "EMT": fmt_hms(emt), "Clock after": fmt_hms(curr)})

        # advance period after boundary
        if period['moves'] is not None and moved_in[side] == period['moves']:
            pidx[side] += 1
            moved_in[side] = 0

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--csv", dest="csv_out")
    ap.add_argument("--clip-epsilon", type=float, default=0.6)  # fractional clocks → smaller epsilon
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as fin:
        while True:
            game = chess.pgn.read_game(fin)
            if game is None:
                break

            rows = annotate_game_with_emt(game, clip_eps=args.clip_epsilon)

            # Handle different cases
            if rows is None:
                print("EMT values already present in PGN. No annotation needed.")
                exit(0)
            elif not rows:
                print("No clock times ([%clk]) found in PGN. Cannot calculate EMT.")
                exit(0)

            # Only open output file if we have data to write
            with open(args.outp, "w", encoding="utf-8") as fout:
                print(game, file=fout)

            if args.csv_out:
                with open(args.csv_out, "a", newline="", encoding="utf-8") as cf:
                    w = csv.DictWriter(cf, fieldnames=["Move","Side","SAN","EMT(s)","EMT","Clock after"])
                    if cf.tell() == 0:
                        w.writeheader()
                    w.writerows(rows)

if __name__ == "__main__":
    main()
