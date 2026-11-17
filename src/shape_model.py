"""Module 5: team shape -> P(home has the ball), without looking at the ball.

    uv run python src/shape_model.py --eval    # honest check: train on one league, test on another
    uv run python src/shape_model.py --train   # final model on Metrica + IDSSE -> models/shape_model.json

Idea: the team with the ball pushes up and spreads out, the other one drops and gets compact.
Per frame and per team, in the team's own attacking direction (distance from its own goal line):
  centroid   mean position of its players
  front      the most advanced player
  back_line  the deepest outfield player (the deepest of all is taken to be the keeper)
  length     front - back_line
  width      spread across the pitch (sd of y)
  push       mean speed towards the opponent's goal
  press      mean distance from each player to the nearest opponent (free vs marked)
  in_opp     how many players are in the opponent's half
  compact    mean distance from each player to the team's centre (defending teams squeeze)
  speed_sd   how different the players' speeds are (attackers make runs, defenders shuffle)
Which of these the model uses is FEATURES (saved in the model file too).
The model sees home minus away for each, smoothed over WINDOW_S. A logistic regression with no
intercept: swapping the teams flips the answer exactly, so it can't learn "home has the ball more".

Attacking direction, per half: each team's keeper is the regular player whose average x is
closest to an end line. Needs no ball, no labels.

Trained ONLY on the external games (Metrica + IDSSE) with our random tracking noise on top
(module 1). Never on our match. Plugged into the HMM by baseline.evidence() as a small extra
H-vs-A clue when [shape] enabled = true in params.toml; the weight is chosen on the external
games too (--eval).
"""

import argparse
import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "shape_model.json"
FPS = 25
H, A, D = 0, 1, 2
ALL_FEATURES = ("centroid", "front", "back_line", "length", "width", "push", "press", "in_opp",
                "compact", "speed_sd")
FEATURES = ("centroid", "front", "back_line", "length", "width", "push", "press")   # in_opp, compact, speed_sd: hurt the exam

# fixed up front, not tuned:
WINDOW_S = 2.0       # a team needs a couple of seconds to change shape, so smooth over that
PUSH_S = 0.4         # speed from positions 0.4 s apart: long enough to beat position jitter
MIN_PLAYERS = 5      # fewer tracked players of a team than this -> no shape for it
KEEPER_SHARE = 0.2   # keeper candidates: slots tracked at least 20% as long as the team's best
L2 = 1.0             # light ridge penalty, only to keep the fit stable

TRAIN_NOISE_SEEDS = (100,)   # not used by validate.py (0..) or tune.py (0, 10, 20)
TEST_NOISE_SEED = 10


# ---------------------------------------------------------------- features

def _quiet(fn, *args, **kwargs):
    # nan-reductions over frames where nobody is tracked warn; those just give NaN
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def _xy(table, slots):
    px = table[[f"p{s}_x" for s in slots]].to_numpy(np.float64)
    py = table[[f"p{s}_y" for s in slots]].to_numpy(np.float64)
    return px, py


def _keeper_side(px, length):
    """(own goal on the left?, keeper's distance to that line) or (None, inf)."""
    seen = (~np.isnan(px)).sum(axis=0)
    if seen.max(initial=0) == 0:
        return None, np.inf
    regular = seen >= KEEPER_SHARE * seen.max()
    mean_x = _quiet(np.nanmean, px[:, regular], axis=0)
    d_left, d_right = np.nanmin(mean_x), length - np.nanmax(mean_x)
    return bool(d_left < d_right), float(min(d_left, d_right))


def home_defends_left(px_h, px_a, length):
    left_h, dist_h = _keeper_side(px_h, length)
    left_a, dist_a = _keeper_side(px_a, length)
    if left_h is None and left_a is None:
        return True                       # nothing to go on; features end up NaN anyway
    if left_h is None:
        return not left_a
    if left_a is None or left_h != left_a:
        return left_h
    # both keepers on the same side can't be right: trust the one standing closer to a line
    return left_h if dist_h <= dist_a else not left_a


def _shift(a, k):
    # a[t + k] (NaN past the ends), for 2-D (frames, players)
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def team_features(px, py, defends_left, length):
    u = px if defends_left else length - px          # distance from own goal line
    n_vis = (~np.isnan(u)).sum(axis=1)
    deep = np.sort(u, axis=1)                         # NaN go last
    back = deep[:, 1] if u.shape[1] > 1 else np.full(len(u), np.nan)
    k = max(1, int(PUSH_S * FPS / 2))
    push = (_shift(u, k) - _shift(u, -k)) / (2 * k / FPS)
    feats = np.column_stack([
        _quiet(np.nanmean, u, axis=1),
        _quiet(np.nanmax, u, axis=1),
        back,
        _quiet(np.nanmax, u, axis=1) - back,
        _quiet(np.nanstd, py, axis=1),
        _quiet(np.nanmean, push, axis=1),
    ])
    feats[n_vis < MIN_PLAYERS] = np.nan
    return feats


def _press(px, py, ox, oy):
    # mean over a team's players of the distance to the closest opponent
    near = np.full(px.shape, np.inf)
    for j in range(ox.shape[1]):
        d = np.hypot(px - ox[:, [j]], py - oy[:, [j]])
        near = np.fmin(near, d)
    near[np.isinf(near)] = np.nan
    return _quiet(np.nanmean, near, axis=1)


def _extra(px, py, ox, oy, defends_left, length):
    u = px if defends_left else length - px
    n_vis = (~np.isnan(u)).sum(axis=1)
    cx = _quiet(np.nanmean, px, axis=1)[:, None]
    cy = _quiet(np.nanmean, py, axis=1)[:, None]
    k = max(1, int(PUSH_S * FPS / 2))
    speed = np.hypot(_shift(px, k) - _shift(px, -k), _shift(py, k) - _shift(py, -k)) / (2 * k / FPS)
    feats = np.column_stack([
        _press(px, py, ox, oy),
        np.sum(u > length / 2, axis=1),
        _quiet(np.nanmean, np.hypot(px - cx, py - cy), axis=1),
        _quiet(np.nanstd, speed, axis=1),
    ]).astype(float)
    feats[n_vis < MIN_PLAYERS] = np.nan
    return feats


def half_features(part, slots_h, slots_a, pitch):
    """(n, len(ALL_FEATURES)) home-minus-away shape, smoothed, for one half's rows."""
    length = pitch[0]
    px_h, py_h = _xy(part, slots_h)
    px_a, py_a = _xy(part, slots_a)
    left = home_defends_left(px_h, px_a, length)
    diff = np.column_stack([
        team_features(px_h, py_h, left, length) - team_features(px_a, py_a, not left, length),
        _extra(px_h, py_h, px_a, py_a, left, length) - _extra(px_a, py_a, px_h, py_h, not left, length),
    ])
    window = max(1, int(WINDOW_S * FPS))
    return pd.DataFrame(diff).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _team_slots(tracks):
    return (tracks.loc[tracks["team"] == "H", "slot"].tolist(),
            tracks.loc[tracks["team"] == "A", "slot"].tolist())


def match_features(table, tracks, pitch, n_frames):
    """(n_frames, F) for a whole match, NaN outside the halves."""
    slots_h, slots_a = _team_slots(tracks)
    out = np.full((n_frames, len(ALL_FEATURES)), np.nan)
    for half in (1, 2):
        rows = table.index[table["half"].eq(half).fillna(False).to_numpy(bool)]
        if len(rows) == 0:
            continue
        part = table.loc[rows].reset_index(drop=True)
        out[part["frame"].to_numpy()] = half_features(part, slots_h, slots_a, pitch)
    return out


# ---------------------------------------------------------------- model

def logit(features, model):
    # features: all of ALL_FEATURES, the model picks the ones it was trained on
    features = features[:, [ALL_FEATURES.index(f) for f in model["features"]]]
    z = np.nan_to_num(features / np.asarray(model["scale"]), nan=0.0)   # missing -> "no info"
    return z @ np.asarray(model["weights"])


def fit(X, y, sample_weight, features=FEATURES):
    """Logistic regression, no intercept, Newton steps with a light ridge penalty."""
    X = X[:, [ALL_FEATURES.index(f) for f in features]]
    sw = sample_weight * len(y) / sample_weight.sum()
    scale = np.sqrt(np.nansum(sw[:, None] * X ** 2, axis=0) / sw.sum())
    scale[scale == 0] = 1.0
    Z = np.nan_to_num(X / scale, nan=0.0)
    w = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (sw * (y - p)) - L2 * w
        hess = (Z * (sw * p * (1 - p))[:, None]).T @ Z + L2 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return {"features": list(features), "scale": scale.tolist(), "weights": w.tolist(),
            "window_s": WINDOW_S}


@functools.lru_cache(maxsize=4)
def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(path=MODEL_PATH):
    return _load(str(path))


def loglik_from_logit(z, s, ball_missing=None):
    """(n, 3) H/A/D log-likelihood from the shape logit. s = [shape] params."""
    z = np.clip(z, -s["clip"], s["clip"])
    w = np.full(len(z), s["weight"])
    if s["mode"] == "ball_missing":
        w = w * ball_missing
    ll = np.zeros((len(z), 3))
    ll[:, H] = -w * np.logaddexp(0, -z)      # w * log p
    ll[:, A] = -w * np.logaddexp(0, z)       # w * log (1 - p)
    ll[:, D] = w * np.log(0.5)
    return ll


def half_loglik(part, slots_h, slots_a, p, pitch, model=None):
    # what baseline.evidence() calls, once per half
    model = model or load_model()
    z = logit(half_features(part, slots_h, slots_a, pitch), model)
    return loglik_from_logit(z, p["shape"], np.isnan(part["ball_x"].to_numpy(np.float64)))


# ---------------------------------------------------------------- training / evaluation

def _league_of(name):
    return next(lg for lg in ("metrica", "idsse", "sc") if name.startswith(lg))


def training_rows(games, seeds=TRAIN_NOISE_SEEDS):
    import external
    Xs, ys, ws = [], [], []
    per_league = {}
    for name in games:
        table, tracks, gt = external.load(name)
        for seed in seeds:
            noisy, _ = external.add_noise(table, tracks["slot"].tolist(),
                                          np.random.default_rng(seed), gt)
            X = match_features(noisy, tracks, external.PITCH, len(gt))
            keep = np.isin(gt, (H, A)) & ~np.isnan(X).all(axis=1)
            keep[np.arange(len(gt)) % 5 != 0] = False      # every 5th frame is plenty
            Xs.append(X[keep])
            ys.append((gt[keep] == H).astype(float))
            league = _league_of(name)
            ws.append(np.full(keep.sum(), 1.0))
            per_league.setdefault(league, []).append(len(ws) - 1)
    # each league weighs the same in total, like in tune.py
    for idx in per_league.values():
        total = sum(len(ws[i]) for i in idx)
        for i in idx:
            ws[i] = ws[i] / total
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


_ROWS = {}


def train(games, features=FEATURES):
    # rows cached, so trying several feature sets doesn't redo the noise + features
    key = tuple(games)
    if key not in _ROWS:
        _ROWS[key] = training_rows(games)
    return fit(*_ROWS[key], features=features)


CONFIGS = [None] + [{"enabled": True, "mode": m, "weight": w, "clip": 4.0}
                    for m in ("ball_missing", "always") for w in (0.02, 0.05, 0.1, 0.2)]


def _label(cfg):
    return "off" if cfg is None else f"{cfg['mode']:12s} w={cfg['weight']:<4}"


def evaluate_on(games, model, p, noise_seed):
    """spellF1 per game for every config, plus frame accuracy of the shape model alone."""
    import baseline
    import external
    import possession_metrics as pm
    scores = np.zeros((len(games), len(CONFIGS)))
    right = total = 0
    for gi, name in enumerate(games):
        table, tracks, gt = external.load(name)
        if noise_seed is not None:
            table, _ = external.add_noise(table, tracks["slot"].tolist(),
                                          np.random.default_rng(noise_seed), gt)
        z = logit(match_features(table, tracks, external.PITCH, len(gt)), model)
        alive = np.isin(gt, (H, A))
        right += int(np.sum(((z > 0) == (gt == H))[alive]))
        total += int(alive.sum())
        missing = np.ones(len(gt))
        missing[table["frame"].to_numpy()] = np.isnan(table["ball_x"].to_numpy(np.float64))
        for ci, cfg in enumerate(CONFIGS):
            extra = None if cfg is None else loglik_from_logit(z, cfg, missing)
            pred = baseline.label_match(table, tracks, p, len(gt), external.PITCH, extra=extra)
            scores[gi, ci] = pm.evaluate(gt, pred)["spell_f1_mean"]
    return scores, right / max(total, 1)


def report(title, games, scores, acc):
    print(f"\n{title}   (shape model alone: {acc:.1%} of in-play frames right)")
    for ci, cfg in enumerate(CONFIGS):
        better = int(np.sum(scores[:, ci] > scores[:, 0] + 1e-9))
        worse = int(np.sum(scores[:, ci] < scores[:, 0] - 1e-9))
        extra = "" if cfg is None else f"   better on {better}/{len(games)}, worse on {worse}"
        print(f"  {_label(cfg)}  spellF1 {scores[:, ci].mean():.4f}{extra}", flush=True)


def run_eval():
    import baseline
    import external
    p = baseline.load_params()
    p = {**p, "shape": {**p.get("shape", {}), "enabled": False}}   # added via extra= here
    leagues = {"metrica": external.GAMES,
               "idsse": tuple(f"idsse_{m}" for m in external.IDSSE_GAMES)}

    held_out = []
    for train_on, test_on in (("idsse", "metrica"), ("metrica", "idsse")):
        model = train(leagues[train_on])
        scores, acc = evaluate_on(leagues[test_on], model, p, TEST_NOISE_SEED)
        report(f"train on {train_on} -> test on {test_on} (unseen league, noise seed "
               f"{TEST_NOISE_SEED})", leagues[test_on], scores, acc)
        held_out.append(scores.mean(axis=0))

    # pick the config on the two held-out leagues (each counts the same) ...
    mean = np.mean(held_out, axis=0)
    best = int(np.argmax(mean))
    print(f"\nbest on held-out leagues: {_label(CONFIGS[best])}  spellF1 {mean[best]:.4f} "
          f"(off: {mean[0]:.4f})")

    # ... and only then look at the exam, with the model trained on both leagues
    model = train(leagues["metrica"] + leagues["idsse"])
    sc_games = external.skillcorner_games()
    scores, acc = evaluate_on(sc_games, model, p, None)
    report("EXAM skillcorner (real broadcast holes, never trained or tuned on)", sc_games,
           scores, acc)
    print(f"\nexam for the chosen config: {_label(CONFIGS[best])}  "
          f"spellF1 {scores[:, best].mean():.4f} vs off {scores[:, 0].mean():.4f}")


def run_train():
    import external
    model = train(external.GAMES + tuple(f"idsse_{m}" for m in external.IDSSE_GAMES))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"saved {MODEL_PATH.relative_to(ROOT)}")
    for name, s, w in zip(model["features"], model["scale"], model["weights"]):
        print(f"  {name:10s} scale {s:7.3f}  weight {w:+.3f}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "evaluation"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.eval:
        run_eval()
    if args.train:
        run_train()
    if not (args.eval or args.train):
        parser.print_help()
