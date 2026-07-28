"""Numberbatch vectors, cut down to the bit we actually need and cached."""

import gzip
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
RAW = DATA / "numberbatch-en.txt.gz"
CACHE = DATA / "vectors.npz"

CLUE_VOCAB = 30000  # after intersecting with numberbatch, so ask for more than this
REFERENCE = 2000    # sample used for the "close to everything" correction

# Place names are the one thing that really poisons this. Numberbatch links
# ENGLAND, PHOENIX and ORANGE because all three are also American towns, so it
# comes back with WESTCHESTER 3 and means it. Can't filter on frequency - the
# towns are commoner than KITTEN - and can't filter on a dictionary. But they
# sit in a tight cluster, so measure the direction and drop anything down it.
# Only places. The same trick on personal names takes PRINCESS with it.
PLACE_SEEDS = ("county", "township", "municipality", "borough", "suburb", "village",
               "hometown", "mayor", "riverside", "springfield", "hampshire", "birthplace")
PLACE_MAX = 0.23


def card_words():
    lines = DATA.joinpath("wordlist.txt").read_text().splitlines()
    return [w.strip().upper() for w in lines if w.strip()]


def term(word):
    """Board and clue words to the keys numberbatch uses."""
    return word.strip().lower().replace(" ", "_")


def show(t):
    return t.replace("_", " ").upper()


def _read(wanted):
    """One pass over the gz, only parsing floats for lines we want."""
    found = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        next(f)  # header is "<count> <dims>"
        for line in f:
            key, _, rest = line.partition(" ")
            key = key.rsplit("/", 1)[-1]  # tolerate the /c/en/word form too
            if key in wanted and key not in found:
                found[key] = np.fromstring(rest, dtype=np.float32, sep=" ")
    return found


def build():
    from wordfreq import top_n_list

    cards = [term(w) for w in card_words()]

    # Common english only. Without this the argmax is always some obscure exact
    # synonym, which scores beautifully and tells your team nothing.
    common = [w for w in top_n_list("en", CLUE_VOCAB * 3)
              if len(w) >= 3 and w.isalpha() and w.isascii()]

    found = _read(set(common) | set(cards))

    missing = [w for w in cards if w not in found]
    if missing:
        raise SystemExit(f"no vectors for card words: {missing}")

    usable = [w for w in common if w in found]
    matrix = np.array([found[w] for w in usable])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    seeds = [usable.index(s) for s in PLACE_SEEDS if s in usable]
    place = matrix[seeds].mean(axis=0)
    place /= np.linalg.norm(place)
    keep = (matrix @ place) < PLACE_MAX

    clues = [w for w, k in zip(usable, keep) if k][:CLUE_VOCAB]
    matrix = matrix[keep][:CLUE_VOCAB]

    cardm = np.array([found[w] for w in cards])
    cardm /= np.linalg.norm(cardm, axis=1, keepdims=True)

    # Some words sit close to everything, mostly the common abstract ones. Their
    # mean similarity to a reference sample is the bit to subtract off later.
    ref = matrix[:: max(1, len(clues) // REFERENCE)].mean(axis=0)

    np.savez_compressed(
        CACHE,
        clues=np.array(clues),
        matrix=matrix,
        baseline=matrix @ ref,
        cards=np.array(cards),
        cardm=cardm,
    )
    return len(clues)


class Vectors:
    def __init__(self):
        if not CACHE.exists():
            if not RAW.exists():
                raise SystemExit(
                    f"missing {RAW}. get it with:\n"
                    "  curl -L -o data/numberbatch-en.txt.gz "
                    "https://conceptnet.s3.amazonaws.com/downloads/2019/"
                    "numberbatch/numberbatch-en-19.08.txt.gz"
                )
            print("building vector cache, one off, takes a minute...")
            build()

        d = np.load(CACHE, allow_pickle=False)
        self.clues = d["clues"]
        self.matrix = d["matrix"]
        self.baseline = d["baseline"]
        self.cards = list(d["cards"])
        self.cardm = d["cardm"]
        self._card_index = {w: i for i, w in enumerate(self.cards)}

    def card_vectors(self, words):
        return self.cardm[[self._card_index[term(w)] for w in words]]
