"""Module 0, the simple version everything else gets compared to.

    uv run python src/baseline.py

Needs data/processed/ (run convert_data.py first). Writes evaluation/results/<name>.csv.

Per frame we look at: how fast players move (walking = play stopped), whether the ball is
missing / out / on the pitch, and who's within 2 m of it. Then an HMM with H/A/D states +
Viterbi, per half. Transitions are sticky so one frame with an opponent closer doesn't flip
possession, and a missing ball keeps the last team. Outside the halves we just write D.
"""

import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "evaluation" / "results"
FPS = 25
PITCH_LENGTH, PITCH_WIDTH = 105.3, 68.0
N_FRAMES = 199319
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def load_params(path=ROOT / "configs" / "params.toml"):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def clean_ball(x, y, max_speed):
    # throw away ball positions that would need a crazy speed to get to
    x, y = x.copy(), y.copy()
    seen = np.flatnonzero(~np.isnan(x))
    if len(seen) < 2:
        return x, y
    dt = np.diff(seen) / FPS
    speed = np.hypot(np.diff(x[seen]), np.diff(y[seen])) / dt
    jump = seen[1:][speed > max_speed]
    x[jump] = np.nan
    y[jump] = np.nan
    return x, y


def rolling_mean(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def gauss_logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd)


def evidence(table, slots_h, slots_a, p):
    # log-likelihood of what we see in each frame, for H / A / D -> (n, 3)
    n = len(table)
    loglik = np.zeros((n, 3))

    # tempo
    speeds = table[[f"p{s}_v" for s in slots_h + slots_a]].to_numpy(np.float64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with nobody tracked
        tempo = np.nanmean(speeds, axis=1)
    tempo = pd.Series(tempo).ffill().bfill().fillna(p["tempo"]["alive_mean"]).to_numpy()
    tempo = rolling_mean(tempo, int(p["tempo"]["window_s"] * FPS))
    alive = gauss_logpdf(tempo, p["tempo"]["alive_mean"], p["tempo"]["alive_sd"])
    dead = gauss_logpdf(tempo, p["tempo"]["dead_mean"], p["tempo"]["dead_sd"])
    loglik[:, H] += alive
    loglik[:, A] += alive
    loglik[:, D] += dead

    # ball missing / out / on the pitch
    e = p["evidence"]
    bx, by = clean_ball(table["ball_x"].to_numpy(np.float64), table["ball_y"].to_numpy(np.float64),
                        p["ball"]["max_speed"])
    missing = np.isnan(bx)
    m = p["ball"]["out_margin"]
    out = ~missing & ((bx < -m) | (bx > PITCH_LENGTH + m) | (by < -m) | (by > PITCH_WIDTH + m))
    on_pitch = ~missing & ~out
    p_in_alive = 1 - e["p_missing_alive"] - e["p_out_alive"]
    p_in_dead = 1 - e["p_missing_dead"] - e["p_out_dead"]
    for mask, pa, pd_ in ((missing, e["p_missing_alive"], e["p_missing_dead"]),
                          (out, e["p_out_alive"], e["p_out_dead"]),
                          (on_pitch, p_in_alive, p_in_dead)):
        loglik[mask, H] += np.log(pa)
        loglik[mask, A] += np.log(pa)
        loglik[mask, D] += np.log(pd_)

    # who's close to the ball
    def nearest(slots):
        px = table[[f"p{s}_x" for s in slots]].to_numpy(np.float64)
        py = table[[f"p{s}_y" for s in slots]].to_numpy(np.float64)
        dist = np.hypot(px - bx[:, None], py - by[:, None])
        return np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)

    dist_h, dist_a = nearest(slots_h), nearest(slots_a)
    close = on_pitch & (np.minimum(dist_h, dist_a) <= e["control_radius"])
    home_closer = dist_h < dist_a
    q = e["p_nearest_is_owner"]
    for mask, team in ((close & home_closer, H), (close & ~home_closer, A)):
        other = A if team == H else H
        loglik[mask, team] += np.log(q)
        loglik[mask, other] += np.log(1 - q)
        loglik[mask, D] += np.log(0.5)
    return loglik


def transition_matrix(p):
    h = p["hmm"]
    leave_poss = 1.0 / (h["possession_mean_s"] * FPS)
    leave_dead = 1.0 / (h["dead_mean_s"] * FPS)
    turn = leave_poss * h["p_turnover"]
    stop = leave_poss * (1 - h["p_turnover"])
    trans = np.array([[1 - leave_poss, turn, stop],
                      [turn, 1 - leave_poss, stop],
                      [leave_dead / 2, leave_dead / 2, 1 - leave_dead]])
    return np.log(trans)


def viterbi(loglik, log_trans, start=D):
    n, k = loglik.shape
    back = np.empty((n, k), dtype=np.int8)
    score = np.full(k, -np.inf)
    score[start] = 0.0
    score = score + loglik[0]
    for t in range(1, n):
        cand = score[:, None] + log_trans          # rows = from, cols = to
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(k)] + loglik[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(p):
    table = pd.read_parquet(PROCESSED / "tactical_tracking.parquet")
    tracks = pd.read_parquet(PROCESSED / "tactical_tracks.parquet")
    slots_h = tracks.loc[tracks["team"] == "H", "slot"].tolist()
    slots_a = tracks.loc[tracks["team"] == "A", "slot"].tolist()
    log_trans = transition_matrix(p)

    codes = np.full(N_FRAMES, D, dtype=np.int8)
    for half in (1, 2):
        rows = table.index[table["half"].eq(half).fillna(False).to_numpy(bool)]
        part = table.loc[rows].reset_index(drop=True)
        codes[part["frame"].to_numpy()] = viterbi(evidence(part, slots_h, slots_a, p), log_trans)
    return LABELS[codes]


def main():
    p = load_params()
    labels = predict(p)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{p['submission']['name']}.csv"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out, index=False)
    print(f"zapisano {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
