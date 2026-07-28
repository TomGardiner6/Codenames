"""codenames spymaster. run it, look at the board, hit enter for clues."""

import argparse
import random
import shutil

from board import Board, grid, keycard, side_by_side, INK, OURS, RESET
from clue import rank
from vectors import Vectors, card_words

HELP = "  enter  next clue    n  new board    b  redraw    q  quit"


def draw(board):
    left, right = grid(board), keycard(board)
    print()
    if shutil.get_terminal_size().columns >= 100:
        print(side_by_side(left, right))
    else:
        print(left)
        print(right)
        print()


def draw_clue(c, i):
    head = f"{INK[OURS]} {c.word} {c.number} {RESET}"
    print(f"  {i:>2}. {head}  {' · '.join(t.lower() for t in c.targets)}")
    print(f"      ev {c.ev:.2f}   assassin {c.risk:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vec = Vectors()
    words = card_words()

    board, clues, at = None, [], 0
    while True:
        if board is None:
            board = Board.deal(words, rng)
            clues = rank(board, vec, top=args.top)
            at = 0
            draw(board)
            print(HELP)

        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if cmd in ("q", "quit", "exit"):
            return
        elif cmd in ("n", "new"):
            board = None
        elif cmd in ("b", "board"):
            draw(board)
        elif cmd == "":
            if at >= len(clues):
                print("  that's the lot, n for a new board")
                continue
            draw_clue(clues[at], at + 1)
            at += 1
        elif cmd.isdigit():
            for i, c in enumerate(clues[: int(cmd)]):
                draw_clue(c, i + 1)
            at = max(at, int(cmd))
        else:
            print(HELP)


if __name__ == "__main__":
    main()
