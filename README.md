# Codenames

A spymaster bot. Give it the 25 words and the key, it gives you a clue.

Rules:

25 words in a 5x5 grid. 9 belong to the team going first, 8 to the other team, 7 are
neutral, 1 is the assassin. Only the spymasters see the key.

The spymaster says one word and a number, where the number is how many of their own
words the clue is meant to cover. Their team then guesses one at a time, and can keep
going while they keep hitting their own words, up to the number plus one. A neutral ends
the turn. An opponent word ends the turn and counts for the opponent. The assassin loses
the game on the spot.

The clue can't be a word on the board. Zero and unlimited are legal numbers in the real
game but I'm ignoring them, they're a different kind of move and cosine similarity has
nothing useful to say about them.

Data:

Card words are the 400 from the base game, at
https://github.com/Gullesnuffs/Codenames/blob/master/wordlist-eng.txt.

Vectors from ConceptNet Numberbatch, english only, 300d, about 516k terms, 325MB
gzipped. Picked it over GloVe because GloVe is trained on co-occurrence, so it decides
two words are related if they turn up in the same Wikipedia article, which gives clues
that score well and mean nothing to a person. Numberbatch is built off a knowledge graph
so it's much closer to actual association. Load it behind a thin interface so GloVe can
be dropped in later and the two compared.

Candidates for the clue itself are the vector vocab intersected with an english
frequency list, top 30k or so. Without a frequency cut the argmax is always something
like "cetacean" - genuinely the nearest word, useless as a clue. Also drop anything
sharing a stem with a board word, or it just answers OCEANS for OCEAN.

Size is a non issue. The whole vocab is 620MB at float32 and the 30k we keep is 36MB, so
it all sits in memory and the search is one matmul.

Boards are dealt at random from the 400 for now.

Scoring:

Since the other player may make mistakes, we should simply optimise for each turn in
isolation.

Raw cosine is the wrong scale to score on. The gap between 0.85 and 0.80 is worth
almost nothing because they'll get both, and the gap between 0.30 and 0.25 is worth
nothing either because they'll get neither. Only the gaps in the middle, where it flips
from no to yes, decide anything. So similarity has to go through something saturating
before it becomes a score, or the search spends all its effort making obvious things
more obvious and irrelevant things more irrelevant.

Rather than pick a squashing function out of the air, model what the operative actually
does. They rank the board by how related each word looks and work down the list. Take
that as a soft ranking, each word getting weight

    w = exp(s / t)

for similarity s and a temperature t setting how sharply people discriminate. Then for
one of our words, the chance it comes up before any bad word at all is

    p = w / (w + sum of w over all the bad words)

which is exact if the ranking really is drawn that way, and is a sigmoid in the gap
between that word and the bad ones. So it saturates at both ends by construction, which
is the property we were after. It's relative rather than absolute, so a clue only has to
beat this board rather than clear some fixed bar. And the assassin sitting close raises
the bar for every target at once through the denominator, with no separate term for it.

The turn is then just: keep going while we keep hitting our own.

    EV = sum over k of (product of p_i for i < k) * (p_k * 1 + (1 - p_k) * cost of a miss)

Cost of a miss is the average over the bad words weighted by how likely each is to be
the one hit, which is exp(s/t) again. Neutral costs about -0.4, a wasted turn. Opponent
about -1.4, a wasted turn plus a card for them. Assassin about -10, and on top of that a
hard reject above a few percent, because EV is an average and I don't want it trading a
small chance of losing outright against a nice clue.

Two corrections before this works on real data. Some clue words are close to everything,
usually the common abstract ones, so subtract the clue's mean similarity to a fixed
reference sample first or those win every board. And t should scale with the spread of
similarities on the board rather than being fixed in cosine units.

t is the only parameter that really matters and it has a meaning - the similarity gap at
which a person is about 73% likely to prefer one word over the other. Guess it, then
tune by hand on real boards.

How do we decide how many words to go for?

Falls out of the same sum. Work out EV for n = 1, 2, 3 and so on, take the best. No need
for a bonus for going wide: a good extra target adds a nearly free +1, and a bad one
contributes p * 1 + (1 - p) * something negative, which goes negative on its own once p
drops. So it stops where it should without being told to.

That replaces the earlier idea of walking down the ranking and counting how many of ours
you clear before hitting a bad one. Same function with t = 0, and this is the softened
version of it.

Search:

For a fixed clue the best set of k targets is always the k nearest of our own words, so
there's no reason to look at subsets. That makes it 30k clues by up to 9 prefix lengths
rather than 30k by 2^9. All of it is dot products of a 30k x 300 against a 300 x 25 plus
some sorting, so it should be fast enough not to think about.

Layout:

    data/       wordlist, cached vectors
    vectors.py  loading and similarity
    board.py    the 25 words and the key
    clue.py     the search
    main.py     cli

Later:

Point the phone at the table and read the board off a photo instead of typing it in. The
25 words are the easy half, and knowing all 400 possible words makes it easier still -
OCR whatever it gives and fuzzy match each result against the list, so RUER comes back as
RULER on its own. Don't need good OCR, just OCR plus the constraint.

The key card is a different problem, no text on it at all, just a 5x5 of red, blue, beige
and black. Find the card, warp it square, sample the middle of each cell. Don't threshold
the colours independently though, lighting will ruin that - beige against white, red
against orange under a warm bulb. The card always has exactly 9, 8, 7 and 1, so cluster
the 25 cells and solve for the assignment matching those counts. Never has to know what
red is in the absolute, only which cells look like each other, so it should survive more
or less any white balance.

Real risk isn't reading it, it's getting the key card rotated 90 degrees against the
board and then confidently playing a mirrored game. Wants an orientation check off the
border colour before it's allowed to say anything.

Still to sort out:

Proper nouns and multi-word terms are allowed for now, which means Numberbatch entries
like new_york are in the running. Worth seeing what it actually produces before deciding
whether to switch them off.

No good way to tell if it's any good yet. Self-play is circular - a guesser built on the
same vectors agrees with the spymaster by construction, so it would score near perfect
and prove nothing. Start by eyeballing 20 boards, and if that isn't enough, build the
guesser on a different embedding to the spymaster.

t, the card values and the assassin floor all need tuning by hand. Don't see a way around
that, though t is the one to spend the time on, the rest are roughly right by argument.
