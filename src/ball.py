"""Patch short holes in the ball track.

A ball that vanishes for a few frames mid-pass is still in play and still roughly on the line
between where we lost it and where we found it again. So holes up to [ball] fill_max_s get a
straight line between the two ends, but only if the ball would need a sane speed to make that
trip. Longer holes stay empty: after a second or more anything could have happened.
"""

import numpy as np

FPS = 25


def fill_short_gaps(x, y, max_s, max_speed):
    x, y = x.copy(), y.copy()
    seen = np.flatnonzero(~np.isnan(x))
    if len(seen) < 2:
        return x, y
    jumps = np.diff(seen)
    for i in np.flatnonzero((jumps > 1) & (jumps <= max_s * FPS + 1)):
        a, b = seen[i], seen[i + 1]
        dist = np.hypot(x[b] - x[a], y[b] - y[a])
        if dist / ((b - a) / FPS) > max_speed:
            continue
        t = np.arange(1, b - a) / (b - a)
        x[a + 1:b] = x[a] + t * (x[b] - x[a])
        y[a + 1:b] = y[a] + t * (y[b] - y[a])
    return x, y


def kicks(x, y, min_dv):
    # True where the ball's velocity jumps by more than min_dv (m/s) within ~0.25 s:
    # somebody kicked, headed or stopped it
    k = 3
    vx = np.gradient(x) * FPS
    vy = np.gradient(y) * FPS
    dvx = np.r_[vx[k:], [np.nan] * k] - np.r_[[np.nan] * k, vx[:-k]]
    dvy = np.r_[vy[k:], [np.nan] * k] - np.r_[[np.nan] * k, vy[:-k]]
    with np.errstate(invalid="ignore"):
        return np.hypot(dvx, dvy) > min_dv


def smooth_between_kicks(x, y, window, min_dv):
    # Kalman-ish without the Kalman: average out the jitter, but never across a gap or a kick,
    # so the sharp turns at touches stay sharp
    x, y = x.copy(), y.copy()
    cut = np.isnan(x) | kicks(x, y, min_dv)
    edges = np.r_[0, np.flatnonzero(cut), len(x)]
    for a, b in zip(edges[:-1], edges[1:]):
        a = a + 1 if cut[a] else a           # the cut frame itself stays as it is
        if b - a < 3:
            continue
        kern = np.ones(min(window, b - a))
        norm = np.convolve(np.ones(b - a), kern, mode="same")
        x[a:b] = np.convolve(x[a:b], kern, mode="same") / norm
        y[a:b] = np.convolve(y[a:b], kern, mode="same") / norm
    return x, y


def crowd_point(px, py, radius=10.0):
    # where the play probably is when we can't see the ball: the player with the most other
    # players around them. (n, 2), NaN when nobody's tracked
    n = len(px)
    best = np.full(n, -1)
    pos = np.full((n, 2), np.nan)
    for j in range(px.shape[1]):
        d = np.hypot(px - px[:, [j]], py - py[:, [j]])
        count = np.sum(d <= radius, axis=1)
        count[np.isnan(px[:, j])] = -1
        better = count > best
        best[better] = count[better]
        pos[better] = np.column_stack([px[better, j], py[better, j]])
    return pos
