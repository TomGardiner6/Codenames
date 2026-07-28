"""The 25 words, the key, and drawing them."""

import random
from dataclasses import dataclass, field

RED, BLUE, NEUTRAL, ASSASSIN = range(4)

COUNTS = {RED: 9, BLUE: 8, NEUTRAL: 7, ASSASSIN: 1}
NAMES = {RED: "red", BLUE: "blue", NEUTRAL: "neutral", ASSASSIN: "the assassin"}
TEAMS = (RED, BLUE)
FIRST = RED  # whoever has the nine goes first

RESET = "\x1b[0m"
INK = {
    RED: "\x1b[48;5;131m\x1b[38;5;231m",
    BLUE: "\x1b[48;5;67m\x1b[38;5;231m",
    NEUTRAL: "\x1b[48;5;180m\x1b[38;5;235m",
    ASSASSIN: "\x1b[48;5;236m\x1b[38;5;231m",
}
FACEDOWN = "\x1b[48;5;253m\x1b[38;5;235m"

# geometry, so a click can be turned back into a card
CELL = 13
GAP = 1
ROW = 4         # three lines of card then a blank one
GRID_TOP = 3    # first screen row the grid sits on, 1 based
GRID_LEFT = 1


@dataclass
class Board:
    words: list
    key: list
    revealed: set = field(default_factory=set)

    @classmethod
    def deal(cls, wordlist, rng=random):
        words = rng.sample(list(wordlist), 25)
        key = [role for role, n in COUNTS.items() for _ in range(n)]
        rng.shuffle(key)
        return cls(words, key)

    def live(self):
        return [i for i in range(25) if i not in self.revealed]

    def live_of(self, role):
        return [i for i in self.live() if self.key[i] == role]

    def left(self, role):
        return len(self.live_of(role))

    def find(self, word):
        want = word.strip().lower()
        for i, w in enumerate(self.words):
            if w.lower() == want:
                return i
        return None


def other(team):
    return BLUE if team == RED else RED


def winner(board):
    """Whoever has run out of cards to find has found them all."""
    for t in TEAMS:
        if board.left(t) == 0:
            return t
    return None


def cell_at(col, row):
    """Screen position back to a card index, or None if it wasn't on one."""
    dc, dr = col - GRID_LEFT, row - GRID_TOP
    if dc < 0 or dr < 0:
        return None
    if dc % (CELL + GAP) >= CELL:
        return None                     # the gutter between two cards
    if dr % ROW == ROW - 1:
        return None                     # the blank line under a row
    c, r = dc // (CELL + GAP), dr // ROW
    return r * 5 + c if c < 5 and r < 5 else None


def _cell(text, role, face_up):
    return f"{INK[role] if face_up else FACEDOWN}{text.center(CELL)}{RESET}"


def grid(board, reveal_all=False):
    """The 5x5. Words always show, it's the colours that are secret."""
    out = []
    for r in range(5):
        for line in range(3):
            cells = []
            for c in range(5):
                i = r * 5 + c
                up = reveal_all or i in board.revealed
                cells.append(_cell(board.words[i] if line == 1 else "", board.key[i], up))
            out.append((" " * GAP).join(cells))
        out.append("")
    return "\n".join(out)


def keycard(board):
    """The little card only the spymasters get to look at."""
    edge = INK[FIRST] + "  " + RESET
    inner = []
    for row in range(5):
        blocks = [INK[board.key[row * 5 + c]] + "    " + RESET for c in range(5)]
        line = edge + " " + " ".join(blocks) + " " + edge
        inner += [line, line]
    rule = INK[FIRST] + " " * (5 * 4 + 4 + 2 + 4) + RESET
    return "\n".join([rule, rule] + inner + [rule, rule])


def side_by_side(left, right, pad=4):
    ls, rs = left.split("\n"), right.split("\n")
    ls += [""] * (len(rs) - len(ls))
    rs += [""] * (len(ls) - len(rs))
    # can't use len() to line things up, the colour codes aren't printable
    width = 5 * CELL + 4 * GAP
    return "\n".join(l + " " * (width - _visible(l) + pad) + r for l, r in zip(ls, rs))


def _visible(s):
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\x1b":
            i = s.index("m", i) + 1
        else:
            n, i = n + 1, i + 1
    return n
