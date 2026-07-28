Plan:

Find out wordlist for cards and condensed game rules. Also get vector dict of linked words

Since the other player may make mistakes, we should simply optimise for each turn in isolation. Evaluation function should be cosine similarity in vector db of matches minus some multiplier of other player / neutral words minus a larger multiplier (or just infinity past some cutoff) of the assassin.

How do we decide how many words to go for?

Find optimal score for each choice (up to say 5 in one go) and then decide which is best (with more words being rewarded more).
