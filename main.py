"""codenames. it spymasters for both sides, you do the guessing for both."""

import argparse
import random

from board import (Board, FIRST, INK, NAMES, RED, BLUE, ASSASSIN, RESET,
                   cell_at, grid, keycard, other, side_by_side, winner)
from clue import rank
from term import CLEAR, HIDE_CURSOR, Terminal
from vectors import Vectors, card_words

HELP = "click a card, or type it. enter to stop guessing, q to give up."


def header(board, team):
    score = "  ".join(f"{INK[t]} {NAMES[t]} {board.left(t)} left {RESET}" for t in (RED, BLUE))
    return f"{score}   {NAMES[team]} to guess"


def draw(t, board, team, clue, left, note):
    body = [header(board, team), "", grid(board)]
    if clue:
        # A clue of n gets n+1 guesses, the last one being the bonus. That's the
        # actual rule and not an off by one, but it reads like one, so say it.
        total = clue.number + 1
        this = total - left + 1
        bonus = "  (the bonus)" if this == total else ""
        body.append(f"  {INK[team]} {clue.word} {clue.number} {RESET}"
                    f"   +1 bonus, so guess {this} of {total}{bonus}")
    else:
        body.append("  no clue passes the safety limits, turn passes")
    body += ["", f"  {note}" if note else "", f"  {HELP}"]
    t.write(CLEAR + HIDE_CURSOR + "\n".join(body) + "\n")


def recap(board, history):
    """What it was actually going for, next to what you took."""
    out = ["  what it meant by each clue", ""]
    for team, clue, picked in history:
        who = f"{INK[team]} {NAMES[team]} {RESET}"
        if clue is None:
            out.append(f"  {who}  passed, nothing safe to say")
            continue
        meant = " · ".join(w.lower() for w in clue.targets)
        took = " ".join(f"{INK[board.key[i]]} {board.words[i].lower()} {RESET}"
                        for i in picked) or " nothing "
        out.append(f"  {who}  {clue.word} {clue.number}")
        out.append(f"          meant  {meant}")
        out.append(f"          took   {took}")
        out.append("")
    return "\n".join(out)


def finish(t, board, won, why, history):
    board.revealed = set(range(25))
    title = f"{INK[won]} {NAMES[won]} wins {RESET}  {why}" if won is not None else f"  {why}"
    body = [title, "",
            side_by_side(grid(board, reveal_all=True), keycard(board)),
            "", "  any key to see what it was going for"]
    t.write(CLEAR + "\n".join(body) + "\n")
    t.read()
    t.write(CLEAR + recap(board, history) + "\n  any key\n")
    t.read()


def play(t, board, vec):
    team, note, history = FIRST, "", []
    while True:
        clues = rank(board, vec, team, top=1)
        clue = clues[0] if clues else None
        left = clue.number + 1 if clue else 0
        picked = []

        while left > 0:
            draw(t, board, team, clue, left, note)
            note = ""
            kind, val = t.read("  > ")

            if kind == "click":
                i = cell_at(*val)
                if i is None or i in board.revealed:
                    continue
            else:
                said = val.strip().lower()
                if said in ("q", "quit"):
                    history.append((team, clue, picked))
                    return finish(t, board, None, "gave up", history)
                if said == "":
                    break
                i = board.find(said)
                if i is None:
                    note = f"there's no {said} on the table"
                    continue
                if i in board.revealed:
                    note = "that one's already turned over"
                    continue

            board.revealed.add(i)
            picked.append(i)
            role, word = board.key[i], board.words[i]
            if role == ASSASSIN:
                history.append((team, clue, picked))
                return finish(t, board, other(team),
                              f"{NAMES[team]} found the assassin", history)
            done = winner(board)
            if done is not None:
                history.append((team, clue, picked))
                return finish(t, board, done, "all their cards found", history)
            if role != team:
                note = f"{word} was {NAMES[role]}, turn over"
                break
            left -= 1
            note = f"{word}, yes"

        history.append((team, clue, picked))
        team = other(team)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    vec = Vectors()
    board = Board.deal(card_words(), random.Random(args.seed))
    try:
        with Terminal() as t:
            play(t, board, vec)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
