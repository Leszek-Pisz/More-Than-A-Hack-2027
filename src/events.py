"""Module 4: extra clues about stoppages and who really controls the ball.

Called from baseline.evidence(), adds to the H/A/D log-likelihoods:
- ball standing still for a while -> someone's placing it, play is stopped
- ...and standing on a set-piece spot (corner, touchline, goal area, centre) -> even more so
- ball disappeared right after being out / on a line -> probably out, stays dead during the gap
- control: a team's player stays by the ball for a bit with no opponent close. Just running
  past the ball doesn't count (that's what made m0 flip possession too often)
- restart: when a ball that stood still starts moving, the team whose player was standing by it
  (with the other team kept back, like a wall at a free kick) probably takes the restart, so it
  gets a nudge for the first seconds after. Meant to make edges after stoppages sharper.
Each piece can be switched off in params.toml [events].
"""

import numpy as np
import pandas as pd

FPS = 25
H, A, D = 0, 1, 2


def _runs_at_least(mask, n):
    # True where mask is part of a run of >= n consecutive True
    if n <= 1:
        return mask.copy()
    m = mask.astype(np.int8)
    edges = np.diff(np.r_[0, m, 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros_like(mask)
    for a, b in zip(starts, stops):
        if b - a >= n:
            out[a:b] = True
    return out


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _restart_team(still, dist_h, dist_a, e):
    # (n,) +1 = home takes the restart, -1 = away, 0 = no idea
    n = len(still)
    out = np.zeros(n)
    look = max(1, int(0.5 * FPS))          # who's at the ball in the last 0.5 s before it moves
    after = int(e["restart_s"] * FPS)
    r, gap = e["restart_radius"], e["restart_opp_min"]
    for a, b in _runs(still):
        dh, da = dist_h[max(a, b - look):b], dist_a[max(a, b - look):b]
        home = np.sum((dh <= r) & (da >= gap))
        away = np.sum((da <= r) & (dh >= gap))
        if home == away:
            continue
        out[b:min(n, b + after)] = 1 if home > away else -1
    return out


def _add(loglik, mask, p_dead, p_alive):
    # evidence that only says "dead" vs "alive"
    loglik[mask, D] += np.log(p_dead)
    loglik[mask, H] += np.log(p_alive)
    loglik[mask, A] += np.log(p_alive)


def extra_evidence(bx, by, dist_h, dist_a, p, pitch):
    e = p["events"]
    L, W = pitch
    n = len(bx)
    loglik = np.zeros((n, 3))
    seen = ~np.isnan(bx)

    # ball speed, a bit smoothed
    vx = pd.Series(np.gradient(np.nan_to_num(bx)) * FPS).rolling(5, center=True, min_periods=1)
    vy = pd.Series(np.gradient(np.nan_to_num(by)) * FPS).rolling(5, center=True, min_periods=1)
    speed = np.hypot(vx.mean().to_numpy(), vy.mean().to_numpy())
    # around a gap the gradient is garbage
    near_gap = pd.Series(~seen).rolling(5, center=True, min_periods=1).max().to_numpy() > 0
    speed[near_gap] = np.nan

    still = _runs_at_least(seen & (speed < e["still_speed"]), int(e["still_s"] * FPS))
    if e["use_still"]:
        _add(loglik, still, e["p_still_dead"], e["p_still_alive"])

        if e["use_spots"]:
            corner = (np.minimum(bx, L - bx) < 3) & (np.minimum(by, W - by) < 3)
            touchline = np.minimum(by, W - by) < 1.5
            goal_area = (np.minimum(bx, L - bx) < 6) & (np.abs(by - W / 2) < 9.2)
            centre = np.hypot(bx - L / 2, by - W / 2) < 2
            spot = still & (corner | touchline | goal_area | centre)
            _add(loglik, spot, e["p_spot_dead"], e["p_spot_alive"])

    if e["use_gap_after_line"]:
        # where was the ball last seen, for every frame of a gap
        last_x = pd.Series(bx).ffill().to_numpy()
        last_y = pd.Series(by).ffill().to_numpy()
        m = e["line_margin"]
        by_line = ((last_x < m) | (last_x > L - m) | (last_y < m) | (last_y > W - m))
        gap_line = ~seen & by_line
        _add(loglik, gap_line, e["p_gapline_dead"], e["p_gapline_alive"])

    if e["use_control"]:
        r = e["control_radius"]
        k = int(e["control_s"] * FPS)
        ctrl_h = _runs_at_least(seen & (dist_h <= r) & (dist_a > r), k)
        ctrl_a = _runs_at_least(seen & (dist_a <= r) & (dist_h > r), k)
        q = e["p_control_is_owner"]
        for mask, team, other in ((ctrl_h, H, A), (ctrl_a, A, H)):
            loglik[mask, team] += np.log(q)
            loglik[mask, other] += np.log(1 - q)
            loglik[mask, D] += np.log(0.5)
    if e.get("use_restart"):
        team = _restart_team(still, dist_h, dist_a, e)
        q = e["p_restart_team"]
        for mask, t, other in ((team > 0, H, A), (team < 0, A, H)):
            loglik[mask, t] += np.log(q)
            loglik[mask, other] += np.log(1 - q)
            loglik[mask, D] += np.log(0.5)
    return loglik
