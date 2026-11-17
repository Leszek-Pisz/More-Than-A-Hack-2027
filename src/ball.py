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
