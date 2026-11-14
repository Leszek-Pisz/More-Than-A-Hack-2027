"""Module 7: decoder that knows how long possessions and stoppages usually last.

The plain HMM in baseline.py assumes a possession can end at any frame with the same chance, so a
1 s possession looks as normal as a 10 s one. Here every segment (a possession or a stoppage) gets
a length prior instead: log-normal, fitted on the Metrica games (values in params.toml).

To keep it fast it works on blocks of 5 frames (0.2 s), then moves each boundary to the best
frame within +-5 frames using the per-frame evidence.
"""

import numpy as np

FPS = 25
H, A, D = 0, 1, 2


def _log_lognormal(d_s, mu, sigma):
    return -np.log(d_s * sigma * np.sqrt(2 * np.pi)) - (np.log(d_s) - mu) ** 2 / (2 * sigma ** 2)


def hsmm(loglik, p):
    c = p["decoder"]
    h = p["hmm"]
    b = int(c["block"])
    n = len(loglik)
    nb = -(-n // b)

    # block evidence (pad the end by repeating the last frame)
    pad = np.vstack([loglik, np.repeat(loglik[-1:], nb * b - n, axis=0)])
    blocks = pad.reshape(nb, b, 3).sum(axis=1) * c["evidence_scale"]
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(blocks, axis=0)])

    dmax = int(c["max_s"] * FPS / b)
    d_s = np.arange(1, dmax + 1) * b / FPS
    logdur = np.stack([_log_lognormal(d_s, c["poss_mu"], c["poss_sigma"]),
                       _log_lognormal(d_s, c["poss_mu"], c["poss_sigma"]),
                       _log_lognormal(d_s, c["dead_mu"], c["dead_sigma"])], axis=1)  # (dmax, 3)

    # segment -> next segment. Same state again is allowed only as a costly "split" so that
    # segments longer than max_s still work (the labels just continue).
    turn = h["p_turnover"]
    trans = np.array([[np.nan, turn, 1 - turn],
                      [turn, np.nan, 1 - turn],
                      [0.5, 0.5, np.nan]])
    logtrans = np.log(np.where(np.isnan(trans), 1.0, trans))
    np.fill_diagonal(logtrans, c["split_penalty"])

    best = np.full((nb + 1, 3), -np.inf)
    enter = np.full((nb + 1, 3), -np.inf)   # best score of a segment of state k starting at s
    enter_from = np.zeros((nb + 1, 3), dtype=np.int8)
    back_d = np.zeros((nb + 1, 3), dtype=np.int32)
    enter[0] = 0.0
    enter_from[0] = -1

    for t in range(1, nb + 1):
        dm = min(dmax, t)
        s = t - np.arange(1, dm + 1)                       # segment start for each length
        score = (cum[t] - cum[s]) + logdur[:dm] + enter[s]  # (dm, 3)
        k_best = np.argmax(score, axis=0)
        best[t] = score[k_best, np.arange(3)]
        back_d[t] = k_best + 1
        # a new segment can start at t: coming from state j ending at t
        cand = best[t][:, None] + logtrans                  # (from j, to k)
        enter_from[t] = np.argmax(cand, axis=0)
        enter[t] = cand[enter_from[t], np.arange(3)]

    # backtrack
    labels_b = np.empty(nb, dtype=np.int8)
    t, k = nb, int(np.argmax(best[nb]))
    while t > 0:
        d = back_d[t, k]
        labels_b[t - d:t] = k
        t -= d
        k = int(enter_from[t, k]) if t > 0 else k
    frames = np.repeat(labels_b, b)[:n]
    return refine(frames, loglik, b)


def refine(labels, loglik, w):
    # move each boundary to the best frame within +-w
    out = labels.copy()
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    for f in changes:
        lo, hi = max(f - w, 1), min(f + w, len(labels) - 1)
        a, b = labels[f - 1], labels[f]
        if lo >= hi:
            continue
        gain = np.cumsum(loglik[lo - 1:hi, a] - loglik[lo - 1:hi, b])  # keep "a" up to x
        x = lo + int(np.argmax(gain))
        out[lo:x] = a
        out[x:hi] = b
    return out
