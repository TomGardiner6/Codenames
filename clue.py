"""Picking the clue. See the README for where the sums come from."""

from dataclasses import dataclass

import numpy as np

from board import NEUTRAL, ASSASSIN
from vectors import SOURCES, term

# Temperature, in raw cosine, per model. Do NOT scale this by the spread of the
# clue over the board - that was the first thing I tried and it's actively wrong,
# because it normalises away the difference between a strong clue and a weak one.
# PET is 0.75 to DOG and RUFUS is 0.24 to CAT, and after dividing by each one's
# own spread they look about the same. The absolute number is the whole signal.
# Separate values per model because glove's cosines run on a different scale -
# glove's board spread is about 1.29x numberbatch's, so its t is scaled to match.
#
# Both are lower than the 0.06 that was right for numberbatch on its own.
# Averaging two models is a stricter test than passing one, so at the old value
# it turned timid and the mean number fell to 1.9. Recalibrated the same way as
# before, on the distribution of numbers over 60 boards, back to about 2.2.
TAU = {"numberbatch": 0.042, "glove": 0.055}

VALUE = {"own": 1.0, "opponent": -1.4, "neutral": -0.4, "assassin": -10.0}

# EV is an average, and I don't want it trading a small chance of losing outright
# against a nice clue, so the assassin gets a hard limit as well as a bad value.
# Same argument for the opponent, only the loss is a card rather than the game,
# so the limit is looser. Both are worst case over the models, not averaged - the
# point of a safety bound is that it holds whoever is reading the clue.
MAX_ASSASSIN_RISK = 0.05
MAX_ENEMY_MASS = 0.15

# Two words spelled alike and meaning the same thing are the same word. Neither
# half works alone: CENTRE/CENTER sit at 0.85 and so do KITTEN/CAT, and
# BOTTOM/BOTTLE look as alike as PIRACY/PIRATE. Together they separate cleanly,
# same word lands 0.69 to 0.84 and lookalikes top out at 0.44.
SAME_WORD_SIM = 0.55

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
    enemy: float


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


def _shared(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


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
            # BONFIRE for FIRE, RASPBERRY for BERRY. The prefix test only looked
            # at the front, so anything with the board word buried in it walked
            # straight through. Costs the odd ORCHESTRA when CHEST is out.
            if len(part) >= 4 and part in clue:
                return True
            # CANADIANS for CANADA. Shares CANAD then goes its own way, so it's
            # neither a stem match nor a prefix, and it sailed through both of
            # the tests above in an actual game. Catch the case where one word
            # is basically the stem and the other carries on past it.
            n = _shared(clue, part)
            if (n >= 4 and min(len(clue), len(part)) - n <= 1
                    and max(len(clue), len(part)) - n >= 2):
                return True
    return False


def outcome(role, team):
    if role == team:
        return "own"
    if role == ASSASSIN:
        return "assassin"
    if role == NEUTRAL:
        return "neutral"
    return "opponent"


def _edit(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def same_word(clue, board_terms, sims):
    """Spelled alike and meaning the same. Catches what spelling alone can't:
    PIRACY/PIRATE and BOTTOM/BOTTLE are equally alike on paper, and SWUNG/SWING
    and PIVOT/PILOT are both one letter apart. The similarity tells them apart."""
    for k, b in enumerate(board_terms):
        if sims[k] < SAME_WORD_SIM:
            continue
        for part in b.split("_"):
            if _shared(clue, part) >= 4 or _edit(clue, part) <= 2:
                return True
    return False


def _model_ev(board, model, vec, tau, value, team, live):
    """What saying (clue, n) is worth if your partner reads it with this model.

    Only over the cards still face down. A word that's been turned over is off
    the table and can't be guessed again, so it shouldn't be scaring the search
    off a clue or soaking up probability mass.
    """
    sims = model.matrix @ vec.card_vectors(model, [board.words[i] for i in live]).T
    sims -= model.baseline[:, None]  # the "close to everything" correction

    z = sims / tau
    z -= z.max(axis=1, keepdims=True)  # p is a ratio, so this is free
    w = np.exp(z)

    kinds = [outcome(board.key[i], team) for i in live]
    ours = [k for k, o in enumerate(kinds) if o == "own"]
    bad = [k for k, o in enumerate(kinds) if o != "own"]
    bad_value = np.array([value[kinds[k]] for k in bad])

    bw = w[:, bad]
    w_bad = bw.sum(axis=1, keepdims=True)
    miss = (bw @ bad_value)[:, None] / w_bad  # what a wrong guess costs on average

    order = np.argsort(-w[:, ours], axis=1)
    ow = np.take_along_axis(w[:, ours], order, axis=1)

    # chance each of ours comes up before any bad word at all
    p = ow / (ow + w_bad)
    ones = np.ones((len(p), 1))
    before = np.concatenate([ones, np.cumprod(p, axis=1)[:, :-1]], axis=1)
    ev = np.cumsum(before * (p * value["own"] + (1 - p) * miss), axis=1)

    kill = [k for k, o in enumerate(kinds) if o == "assassin"]
    risk = (w[:, kill[0]] / (w[:, kill[0]] + ow[:, 0]) if kill
            else np.zeros(len(w)))
    theirs = [k for k, o in enumerate(kinds) if o == "opponent"]
    enemy = w[:, theirs].sum(axis=1) / w.sum(axis=1)
    return ev, order, risk, enemy


def rank(board, vec, team, top=10, tau=TAU, value=VALUE,
         max_risk=MAX_ASSASSIN_RISK, max_enemy=MAX_ENEMY_MASS):
    """Average the turn value over the models, on the grounds that we don't know
    which one the person across the table is running."""
    live = board.live()
    if not [i for i in live if board.key[i] == team]:
        return []

    evs, risks, enemies, orders = [], [], [], {}
    for m in vec.models:
        ev, order, risk, enemy = _model_ev(board, m, vec, tau[m.name], value, team, live)
        evs.append(ev)
        risks.append(risk)
        enemies.append(enemy)
        orders[m.name] = order

    ev = np.mean(evs, axis=0)
    risk = np.max(risks, axis=0)      # a safety bound has to hold for either
    enemy = np.max(enemies, axis=0)

    n = ev.argmax(axis=1)
    best = ev[np.arange(len(ev)), n]
    best = np.where((risk > max_risk) | (enemy > max_enemy), -np.inf, best)

    ours = [i for i in live if board.key[i] == team]
    order = orders[SOURCES[0]]  # only for showing which words were meant

    # A word that's been turned over is covered up, so it's fair game as a clue.
    # Legality is a slow per word check, so only do it on the way down the list.
    board_terms = [term(board.words[i]) for i in live]
    nb = vec.models[0]
    # raw cosine, not the baseline corrected version, for the same word test
    raw = nb.matrix @ vec.card_vectors(nb, [board.words[i] for i in live]).T

    out = []
    for i in np.argsort(-best):
        if not np.isfinite(best[i]):
            break
        word = str(vec.clues[i])
        if illegal(word, board_terms) or same_word(word, board_terms, raw[i]):
            continue
        k = int(n[i]) + 1
        out.append(Clue(
            word=word.upper(),
            number=k,
            targets=[board.words[ours[j]] for j in order[i, :k]],  # nb's ordering
            ev=float(best[i]),
            risk=float(risk[i]),
            enemy=float(enemy[i]),
        ))
        if len(out) >= top:
            break
    return out
