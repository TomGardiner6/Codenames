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

Scoring:

Since the other player may make mistakes, we should simply optimise for each turn in
isolation.

The thing to score against is that the operative works down the board in order of how
related each word looks, so what matters is the ranking, not the total. A clue works
exactly as long as every word we're going for sits above every word we're not. So score
the margin - worst target minus best bad word - rather than summing similarities. A sum
will happily hide a lethal assassin behind three excellent targets.

Bad words aren't equal, so they get different weights: neutral is a wasted turn,
opponent is a wasted turn plus a free card for them, assassin is the game. Assassin
wants a hard floor rather than a weight, reject the clue outright if it ranks anywhere
near the targets.

How do we decide how many words to go for?

Falls out of the ranking. For each candidate clue, sort all 25 board words by similarity,
walk down from the top and count how many of ours you get through before you hit
something that isn't. That count is the number. No need for a separate bonus for going
wide, the count already rewards it, and it can't talk itself into a 5 that dies on the
second guess.

That's a hard cutoff and real people don't rank things that precisely, so later on it's
worth softening into a probability of the operative agreeing with the order, and taking
an expected number of correct guesses instead. Not for the first version.

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

Still to sort out:

Proper nouns and multi-word terms are allowed for now, which means Numberbatch entries
like new_york are in the running. Worth seeing what it actually produces before deciding
whether to switch them off.

No good way to tell if it's any good yet. Self-play is circular - a guesser built on the
same vectors agrees with the spymaster by construction, so it would score near perfect
and prove nothing. Start by eyeballing 20 boards, and if that isn't enough, build the
guesser on a different embedding to the spymaster.

The bad-word weights and the assassin floor need tuning by hand. Don't see a way around
that.
