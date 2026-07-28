"""The 25 words, the key, and drawing both."""

import random
from dataclasses import dataclass

OURS, THEIRS, NEUTRAL, ASSASSIN = range(4)

COUNTS = {OURS: 9, THEIRS: 8, NEUTRAL: 7, ASSASSIN: 1}
NAMES = {OURS: "us", THEIRS: "them", NEUTRAL: "neutral", ASSASSIN: "assassin"}

RESET = "\x1b[0m"
INK = {
    OURS: "\x1b[48;5;131m\x1b[38;5;231m",
    THEIRS: "\x1b[48;5;67m\x1b[38;5;231m",
    NEUTRAL: "\x1b[48;5;180m\x1b[38;5;235m",
    ASSASSIN: "\x1b[48;5;236m\x1b[38;5;231m",
}

CELL = 13
GAP = " "


@dataclass
class Board:
    words: list
    key: list

    @classmethod
    def deal(cls, wordlist, rng=random):
        words = rng.sample(list(wordlist), 25)
        key = [role for role, n in COUNTS.items() for _ in range(n)]
        rng.shuffle(key)
        return cls(words, key)

    def of(self, role):
        return [w for w, r in zip(self.words, self.key) if r == role]

    def indices(self, role):
        return [i for i, r in enumerate(self.key) if r == role]


def _cell(text, role, hidden):
    ink = "\x1b[48;5;253m\x1b[38;5;235m" if hidden else INK[role]
    return f"{ink}{text.center(CELL)}{RESET}"


def grid(board, hidden=False):
    """The 5x5 as it sits on the table, coloured in as the spymaster sees it."""
    out = []
    for row in range(5):
        cells = [(board.words[row * 5 + c], board.key[row * 5 + c]) for c in range(5)]
        out.append(GAP.join(_cell("", r, hidden) for _, r in cells))
        out.append(GAP.join(_cell(w, r, hidden) for w, r in cells))
        out.append(GAP.join(_cell("", r, hidden) for _, r in cells))
        out.append("")
    return "\n".join(out)


def keycard(board):
    """The little card only the spymasters get to look at."""
    edge = INK[OURS] + "  " + RESET
    inner = []
    for row in range(5):
        blocks = [INK[board.key[row * 5 + c]] + "    " + RESET for c in range(5)]
        line = edge + " " + " ".join(blocks) + " " + edge
        inner += [line, line]
    width = 5 * 4 + 4 * 1 + 2
    rule = INK[OURS] + " " * (width + 4) + RESET
    return "\n".join([rule, rule] + inner + [rule, rule])


def side_by_side(left, right, pad=4):
    ls, rs = left.split("\n"), right.split("\n")
    ls += [""] * (len(rs) - len(ls))
    rs += [""] * (len(ls) - len(rs))
    # can't use len() to line things up, the colour codes aren't printable
    width = 5 * CELL + 4 * len(GAP)
    return "\n".join(l + " " * (width - _visible(l) + pad) + r for l, r in zip(ls, rs))


def _visible(s):
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\x1b":
            i = s.index("m", i) + 1
        else:
            n, i = n + 1, i + 1
    return n
