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
gzipped. Better than GloVe on its own, because GloVe is trained on co-occurrence and so
decides two words are related if they turn up in the same Wikipedia article, which gives
clues that score well and mean nothing to a person. Numberbatch is built off a knowledge
graph so it's much closer to actual association.

Load GloVe as well though, and keep both. Not to pick a winner - to average over them,
see below. Shared vocabulary, so a word is only a candidate if both models have it, and
the four two word cards like ICE CREAM get composed out of their parts for GloVe, which
has no token for them.

Candidates for the clue itself are the vector vocab intersected with an english
frequency list, top 30k or so. Without a frequency cut the argmax is always something
like "cetacean" - genuinely the nearest word, useless as a clue. Also drop anything
sharing a stem with a board word, or it just answers OCEANS for OCEAN.

Place names are the other thing that poisons it, and worse, because they don't look
wrong to the maths. ENGLAND, PHOENIX and ORANGE are all American towns as well, so it
came back with WESTCHESTER 3 and meant it. Can't cut those on frequency, ROCHESTER is a
commoner word than KITTEN. Can't cut them on a dictionary either, /usr/share/dict is
Webster's and it keeps BURLINGTON while throwing out PAWS. What does work is that they
sit in a tight cluster, so take a dozen obvious ones like COUNTY and TOWNSHIP, average
them, and drop anything too far down that direction. Costs a few real words - HARBOUR
scores the same as PASADENA - but it takes the towns out.

Only do that for places. The same trick on people's names looks fine until you notice
it's also thrown out PRINCESS.

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

Some clue words are close to everything, usually the common abstract ones, so subtract
the clue's mean similarity to a fixed reference sample first or those win every board.

I also thought t should scale with the spread of similarities on the board rather than
being fixed in cosine units. That's wrong, and wrong in a way that quietly ruins it. PET
is 0.75 to DOG, RUFUS is 0.24 to CAT, and dividing each by its own spread says they're
about equally good. They are not. The absolute number is the entire signal for whether a
person will actually make the connection, and normalising it away leaves you ranking on
nothing but the ordering, so any obscure word that happens to sort cleanly wins. Fixed t
in raw cosine. It gave RUFUS 3 for cat, dog, berry until I took it out.

t is the only parameter that really matters and it has a meaning - the similarity gap at
which a person is about 73% likely to prefer one word over the other. Tuning it by eye
board by board doesn't converge, so instead deal 60 boards and look at the distribution
of the number it comes back with, on the grounds that a decent spymaster averages a bit
over 2. That puts it at 0.06, where it says 2 about eighty percent of the time and 3 the
rest. 0.04 has it saying 3 constantly and 0.10 reduces it to nervous 1s.

Two models, not one:

The sum above assumes the person guessing reads the board the way numberbatch does. They
don't, and I've no way of knowing how they do read it. So don't commit to one model -
work out the whole turn value separately under numberbatch and under glove and average
the two. It's the same expected value argument one level up: uncertain which model your
partner is running, so marginalise over both instead of betting on one.

Each model ranks the board its own way and that's fine, it should. The spymaster only
ever says a word and a number, never which words were meant, so each model predicting
its own guess order is exactly the thing being averaged. Only ever show the numberbatch
ordering, and only so there's something to read on screen.

Worth it because a clue that only scores well under one model is usually scoring on that
model's private quirks rather than on anything a person would recognise. Averaging drags
those down and leaves the clues that work for both. Also gives a way out of the circular
evaluation problem further down - a guesser built on glove isn't agreeing with a
numberbatch spymaster by construction.

It changes the answer on eleven boards out of twelve, so it isn't cosmetic. Mostly for
the better - NHL becomes HOCKEY, PUSS becomes PET, PRESSURIZED 3 becomes PUMP 2, all
cases of numberbatch liking a word that glove had no particular time for. Not uniformly
though. MERMAID 2 for princess and dragon went to DUCHESS 1, which is worse and more
timid, and that's the trade: insisting a clue work in two heads at once costs you the
clever ones that only work in one.

Two things had to move to make it work. The place filter now runs on both models rather
than just numberbatch, because glove is co-occurrence and therefore far worse for this,
and filtering only numberbatch let ARIZONA and CAIRO walk back in through the average.
And t came down by a third, see above.

The safety limits go the other way, worst case across the models rather than averaged. A
bound that only holds if your partner happens to think like numberbatch isn't a bound.

Cap the opponent too, not just the assassin:

Total probability mass sitting on their words, capped at 15%, which is the chance they
open the turn by handing over a card. The EV already prices opponent words in through
the cost of a miss, but that's a mean, and a mean will happily accept a fair chance of
giving one away in exchange for a good enough upside. Same argument as the assassin,
looser limit because the loss is a card rather than the game.

Being honest about it though, it doesn't currently do anything. Over 60 boards the
winning clue never got near either limit - worst was 8.9% on the opponent against a cap
of 15%, and 2.1% on the assassin against 5%. The caps throw out about 95% of the
candidate list but every one of those was already scoring badly on EV, so the clue that
comes out the top is the same with them and without them. They're insurance against a
board where the best scoring clue is also a dangerous one, which hasn't come up yet.
Leaving them in on that basis, not because they're earning their keep.

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

Running it:

    uv sync
    curl -L -o data/numberbatch-en.txt.gz \
      https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
    curl -L -o data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
    uv run main.py

Both downloads are big, 325MB and 862MB, and both are read straight out of the archive
rather than unpacked.

Deals a board, draws it with the key colours on, then enter for the next clue, n for a
new board, q to quit. First run builds a cache off the gz and takes a few seconds. The
search itself is about 10ms.

Layout:

    data/       wordlist, cached vectors
    vectors.py  loading and similarity
    board.py    the 25 words and the key, and drawing them
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

Personal names still get through. RUFUS, CHEVALIER, and DENZEL for hollywood and
washington, which is at least funny. Can't reuse the place trick, the name direction
takes PRINCESS and KNIGHT down with it, so it needs something else.

The legality check is crude and knows it. Stems, british spellings and prefixes, so it
gets CENTRE for CENTER and BRIDGING for BRIDGE, but LAB and LABORATORY only because the
prefix rule is blunt enough to take CARBON for CAR down with it.

No good way to tell if it's any good yet. Self-play is circular - a guesser built on the
same vectors agrees with the spymaster by construction, so it would score near perfect
and prove nothing. Start by eyeballing 20 boards, and if that isn't enough, build the
guesser on a different embedding to the spymaster.

t, the card values and the assassin floor all need tuning by hand. Don't see a way around
that, though t is the one to spend the time on, the rest are roughly right by argument.
