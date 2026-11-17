"""Remove short flickers after decoding ([post] in params.toml).

spellF1 counts whole possessions, so a false 0.5 s switch H -> A -> H costs a lot: the one real
H possession gets cut in two (neither half matches it well) and the A blip is a false positive.
Same for a short false D inside one team's possession.

Rule, kept simple on purpose: a segment shorter than its minimum length whose neighbours on BOTH
sides carry the same label gets that label (X - y - X -> X - X - X). Boundaries between two
different labels are never moved, so this can't make us later at a real change. Shortest
segments go first, so merging one can't hide a shorter one next to it.

Minimum lengths: [post] min_poss_s for H/A blips, min_dead_s for D blips. Checked on Metrica +
IDSSE + the exam (compare.py), never on our match.
"""

import numpy as np

FPS = 25


def _runs(codes):
    # (start, end, label) for each run of equal labels, end exclusive
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(codes)]))
    return starts, ends, codes[starts]


def merge_short(codes, p, dead=2):
    post = p.get("post", {})
    if not post.get("enabled") or len(codes) == 0:
        return codes
    min_poss = int(round(post["min_poss_s"] * FPS))
    min_dead = int(round(post["min_dead_s"] * FPS))
    codes = codes.copy()
    while True:
        starts, ends, labels = _runs(codes)
        if len(starts) < 3:
            return codes
        length = ends - starts
        limit = np.where(labels == dead, min_dead, min_poss)
        inner = np.arange(1, len(starts) - 1)
        ok = (length[inner] < limit[inner]) & (labels[inner - 1] == labels[inner + 1])
        cand = inner[ok]
        if len(cand) == 0:
            return codes
        i = cand[np.argmin(length[cand])]          # shortest first (ties: earliest)
        codes[starts[i]:ends[i]] = labels[i - 1]
