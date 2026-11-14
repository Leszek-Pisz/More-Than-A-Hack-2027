"""Smart Tagging as a weak hint about who has the ball (the team's idea, Tomek's notebook).

A tag like "BUILD UP" by a team says that team probably had the ball somewhere in that window.
"DEFENDING IN ..." by a team means the other team had it. We just nudge H/A a little inside the
window. Boundaries still come from the tracking, tags have fake fixed lengths (30-35 s).

Can't be tuned: the external games have no Smart Tagging, so the weight is a fixed guess
(params.toml [tags]). Off by default.
"""

import numpy as np
import pandas as pd

H, A = 0, 1

ATTACKING = {"BUILD UP", "CREATING CHANCES", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL",
             "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"}
DEFENDING = {"DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION", "DEFENDING IN ATTACKING THIRD",
             "DEFENDING IN DEFENSIVE THIRD"}


def tags_loglik(tags, n_frames, weight):
    """(n_frames, 3) extra log-likelihood: +weight for the team that probably has the ball."""
    votes = np.zeros((n_frames, 2))
    for row in tags.itertuples():
        if not isinstance(row.team, str):
            continue
        if row.code in ATTACKING:
            team = H if row.team == "H" else A
        elif row.code in DEFENDING:
            team = A if row.team == "H" else H
        else:
            continue
        lo, hi = max(int(row.start_frame), 0), min(int(row.end_frame), n_frames)
        votes[lo:hi, team] += 1

    out = np.zeros((n_frames, 3))
    # only the difference matters, and two tags saying the same thing shouldn't count double
    diff = np.clip(votes[:, H] - votes[:, A], -1, 1)
    out[:, H] += weight * diff
    out[:, A] -= weight * diff
    return out


def load(path, n_frames, weight):
    return tags_loglik(pd.read_parquet(path), n_frames, weight)
