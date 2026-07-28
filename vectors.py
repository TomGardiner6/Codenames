"""Word vectors, cut down to the bit we actually need and cached.

Two sources, not one. The clue has to survive being read by someone whose head
isn't wired the same way as numberbatch, and the cheapest stand in for that is a
second model built on completely different principles. Numberbatch comes off a
knowledge graph, glove off raw wikipedia co-occurrence, and a clue that only
works in one of them is a clue that only works if your partner thinks like that
one. See the ensemble bit in clue.py.
"""

import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
CACHE = DATA / "vectors.npz"

SOURCES = ("numberbatch", "glove")
RAW = {
    "numberbatch": DATA / "numberbatch-en.txt.gz",
    "glove": DATA / "glove.6B.zip",
}
GET = {
    "numberbatch": "curl -L -o data/numberbatch-en.txt.gz https://conceptnet.s3."
                   "amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz",
    "glove": "curl -L -o data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip",
}

CLUE_VOCAB = 30000  # after intersecting both models, so ask for more than this
REFERENCE = 2000    # sample used for the "close to everything" correction

# Place names are the one thing that really poisons this. Numberbatch links
# ENGLAND, PHOENIX and ORANGE because all three are also American towns, so it
# comes back with WESTCHESTER 3 and means it. Can't filter on frequency - the
# towns are commoner than KITTEN - and can't filter on a dictionary. But they
# sit in a tight cluster, so measure the direction and drop anything down it.
# Only places. The same trick on personal names takes PRINCESS with it.
#
# Has to run on both models. Glove is co-occurrence, so it's the worse of the
# two for this, and filtering only numberbatch let ARIZONA and CAIRO straight
# back in through the average. Cut by share of vocabulary rather than by a fixed
# cosine, because the two models put their similarities on different scales.
PLACE_SEEDS = ("county", "township", "municipality", "borough", "suburb", "village",
               "hometown", "mayor", "riverside", "springfield", "hampshire", "birthplace")
PLACE_CUT = 0.12  # drop the most place-like eighth of the vocabulary, per model

# An english frequency list is not a list of english words. Enough german and
# french leaks into english text that DER, EINE, IST and QUE are all in the top
# 30k, and it offered DER for GERMANY and VAN in a real game. If a word is much
# commoner somewhere else than it is here, it isn't ours. 2.5 catches those and
# stops short of TEMPO, RAPPORT and PERSONA, which are.
FOREIGN_LANGS = ("de", "fr", "es", "it", "nl", "pt")
FOREIGN_GAP = 2.5


@dataclass
class Model:
    name: str
    matrix: np.ndarray   # (vocab, dims), unit rows
    cardm: np.ndarray    # (400, dims), unit rows
    baseline: np.ndarray  # (vocab,) mean similarity to a reference sample


def card_words():
    lines = DATA.joinpath("wordlist.txt").read_text().splitlines()
    return [w.strip().upper() for w in lines if w.strip()]


def term(word):
    """Board and clue words to the keys the vector files use."""
    return word.strip().lower().replace(" ", "_")


def _open(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        name = next(n for n in sorted(z.namelist()) if n.endswith("300d.txt"))
        return io.TextIOWrapper(z.open(name), encoding="utf-8")
    return open(path, encoding="utf-8")


def _read(path, wanted):
    """One pass, only parsing floats for lines we actually want."""
    found = {}
    with _open(path) as f:
        first = f.readline()
        head = first.split(" ", 2)
        if not (len(head) == 2 and head[0].strip().isdigit()):
            f = _chain(first, f)  # glove has no header, don't eat a real row
        for line in f:
            key, _, rest = line.partition(" ")
            key = key.rsplit("/", 1)[-1]  # tolerate numberbatch's /c/en/word form
            if key in wanted and key not in found:
                found[key] = np.fromstring(rest, dtype=np.float32, sep=" ")
    return found


def _chain(first, rest):
    yield first
    yield from rest


def _unit(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _compose(t, found):
    """ICE CREAM isn't a glove token, so average ICE and CREAM instead."""
    if t in found:
        return found[t]
    parts = [found[p] for p in t.split("_") if p in found]
    return np.mean(parts, axis=0) if len(parts) == len(t.split("_")) else None


def build():
    from wordfreq import top_n_list, zipf_frequency

    def english(w):
        return max(zipf_frequency(w, l) for l in FOREIGN_LANGS) \
            - zipf_frequency(w, "en") < FOREIGN_GAP

    for s in SOURCES:
        if not RAW[s].exists():
            raise SystemExit(f"missing {RAW[s]}. get it with:\n  {GET[s]}")

    cards = [term(w) for w in card_words()]

    # Common english only. Without this the argmax is always some obscure exact
    # synonym, which scores beautifully and tells your team nothing.
    common = [w for w in top_n_list("en", CLUE_VOCAB * 3)
              if len(w) >= 3 and w.isalpha() and w.isascii() and english(w)]

    wanted = set(common) | set(cards) | {p for c in cards for p in c.split("_")}
    found = {s: _read(RAW[s], wanted) for s in SOURCES}

    # a clue is only usable if we can score it under every model
    usable = [w for w in common if all(w in found[s] for s in SOURCES)]

    seeds = [usable.index(s) for s in PLACE_SEEDS if s in usable]
    keep = np.ones(len(usable), dtype=bool)
    for s in SOURCES:
        m = _unit(np.array([found[s][w] for w in usable]))
        place = m[seeds].mean(axis=0)
        place /= np.linalg.norm(place)
        along = m @ place
        keep &= along < np.quantile(along, 1 - PLACE_CUT)

    clues = [w for w, k in zip(usable, keep) if k][:CLUE_VOCAB]

    out = {"clues": np.array(clues), "cards": np.array(cards)}
    for s in SOURCES:
        missing = [c for c in cards if _compose(c, found[s]) is None]
        if missing:
            raise SystemExit(f"{s} has no vectors for card words: {missing}")
        matrix = _unit(np.array([found[s][w] for w in clues]))
        # Some words sit close to everything, mostly the common abstract ones.
        # Their mean similarity to a reference sample is what to subtract off.
        ref = matrix[:: max(1, len(clues) // REFERENCE)].mean(axis=0)
        out[f"{s}_matrix"] = matrix
        out[f"{s}_cardm"] = _unit(np.array([_compose(c, found[s]) for c in cards]))
        out[f"{s}_baseline"] = matrix @ ref

    np.savez_compressed(CACHE, **out)
    return len(clues)


class Vectors:
    def __init__(self):
        if not CACHE.exists():
            print("building vector cache, one off, takes a minute...")
            build()

        d = np.load(CACHE, allow_pickle=False)
        self.clues = d["clues"]
        self.cards = list(d["cards"])
        self._index = {w: i for i, w in enumerate(self.cards)}
        self.models = [
            Model(s, d[f"{s}_matrix"], d[f"{s}_cardm"], d[f"{s}_baseline"])
            for s in SOURCES
        ]

    def card_vectors(self, model, words):
        return model.cardm[[self._index[term(w)] for w in words]]
