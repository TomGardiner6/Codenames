"""Picking the clue. See the README for where the sums come from."""

from dataclasses import dataclass

import numpy as np

from board import OURS, THEIRS, NEUTRAL, ASSASSIN
from vectors import term

# Temperature, in raw cosine. Do NOT scale this by the spread of the clue over
# the board - that was the first thing I tried and it's actively wrong, because
# it normalises away the difference between a strong clue and a weak one. PET is
# 0.75 to DOG and RUFUS is 0.24 to CAT, and after dividing by each one's own
# spread they look about the same. The absolute number is the whole signal.
TAU = 0.06

VALUE = {OURS: 1.0, THEIRS: -1.4, NEUTRAL: -0.4, ASSASSIN: -10.0}

# EV is an average, and I don't want it trading a small chance of losing outright
# against a nice clue, so the assassin gets a hard limit as well as a bad value.
MAX_ASSASSIN_RISK = 0.05

SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ies", "ied",
            "es", "ed", "ly", "ness", "ment", "re", "s")

# numberbatch is american, the board sometimes isn't, and CENTRE for CENTER is
# cheating. can't catch these on similarity - centre/center is 0.85 and so is
# kitten/cat, which is a clue I want to keep.
SPELLINGS = (("our", "or"), ("tre", "ter"), ("ise", "ize"), ("isation", "ization"),
             ("ogue", "og"), ("yse", "yze"), ("ae", "e"), ("oe", "e"))


@dataclass
class Clue:
    word: str
    number: int
    targets: list
    ev: float
    risk: float


def respell(w):
    for a, b in SPELLINGS:
        if w.endswith(a):
            return w[: -len(a)] + b
    return w


def roots(w):
    """Every stem this word might reduce to. Two rounds so CENTRES gets to CENT."""
    seen = {w, respell(w)}
    for _ in range(2):
        for x in list(seen):
            for s in SUFFIXES:
                if x.endswith(s) and len(x) - len(s) >= 3:
                    r = x[: -len(s)]
                    seen.update({r, respell(r)})
                    if not r.endswith("e"):
                        seen.add(r + "e")  # BRIDGING -> BRIDG -> BRIDGE
    return seen


def illegal(clue, board_terms):
    """No saying a word that's on the table, or anything that's obviously it again.

    The prefix test is deliberately blunt: it throws away CARBON for CAR, which
    was fine, to also throw away LABORATORY for LAB, which wasn't. Cheap trade,
    there are 30k candidates and only ever one of them gets said out loud.
    """
    cr = roots(clue)
    for b in board_terms:
        for part in b.split("_"):
            if cr & roots(part):
                return True
            if len(part) >= 3 and (clue.startswith(part) or part.startswith(clue)):
                return True
    return False


def rank(board, vec, top=10, tau=TAU, value=VALUE, max_risk=MAX_ASSASSIN_RISK):
    sims = vec.matrix @ vec.card_vectors(board.words).T
    sims -= vec.baseline[:, None]  # the "close to everything" correction

    z = sims / tau
    z -= z.max(axis=1, keepdims=True)  # p is a ratio, so this is free
    w = np.exp(z)

    ours = board.indices(OURS)
    bad = [i for i in range(25) if board.key[i] != OURS]
    bad_value = np.array([value[board.key[i]] for i in bad])

    bw = w[:, bad]
    w_bad = bw.sum(axis=1, keepdims=True)
    miss = (bw @ bad_value)[:, None] / w_bad  # what a wrong guess costs on average

    order = np.argsort(-w[:, ours], axis=1)
    ow = np.take_along_axis(w[:, ours], order, axis=1)

    # chance each of ours comes up before any bad word at all
    p = ow / (ow + w_bad)
    ones = np.ones((len(p), 1))
    before = np.concatenate([ones, np.cumprod(p, axis=1)[:, :-1]], axis=1)
    ev = np.cumsum(before * (p * value[OURS] + (1 - p) * miss), axis=1)

    n = ev.argmax(axis=1)
    best = ev[np.arange(len(ev)), n]

    assassin = board.key.index(ASSASSIN)
    risk = w[:, assassin] / (w[:, assassin] + ow[:, 0])
    best = np.where(risk > max_risk, -np.inf, best)

    # legality is a slow per word check, so only do it on the way down the list
    board_terms = [term(x) for x in board.words]
    out = []
    for i in np.argsort(-best):
        if not np.isfinite(best[i]):
            break
        word = str(vec.clues[i])
        if illegal(word, board_terms):
            continue
        k = int(n[i]) + 1
        out.append(Clue(
            word=word.upper(),
            number=k,
            targets=[board.words[ours[j]] for j in order[i, :k]],
            ev=float(best[i]),
            risk=float(risk[i]),
        ))
        if len(out) >= top:
            break
    return out
