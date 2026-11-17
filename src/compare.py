"""Switch things on/off and see what happens, on games we never tuned on in this run.

    uv run python src/compare.py

Metrica + IDSSE on fresh noise (seed 10, tuning uses 0) and the SkillCorner exam. Shape models
are held out like in tune.py: each game gets a model trained on the other league (SkillCorner:
on both). A change stays only if it doesn't make the exam or either league worse.
"""

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import baseline  # noqa: E402
import external  # noqa: E402
import possession_metrics as pm  # noqa: E402
import shape_model  # noqa: E402
import tune  # noqa: E402

BASE_SHAPE = shape_model.FEATURES

# name -> (params changes, shape features)
VARIANTS = {
    "base": ({}, BASE_SHAPE),
    "fill ball gaps": ({("ball", "fill_gaps"): True}, BASE_SHAPE),
    "restart team": ({("events", "use_restart"): True}, BASE_SHAPE),
    "shape + press": ({}, BASE_SHAPE + ("press",)),
    "shape + in_opp": ({}, BASE_SHAPE + ("in_opp",)),
    "shape + both": ({}, BASE_SHAPE + ("press", "in_opp")),
}


def league(name):
    return "metrica" if name in tune.LEAGUES["metrica"] else \
        "idsse" if name in tune.LEAGUES["idsse"] else "exam"


def prepare(noise_seed=10):
    games = []
    for name in tune.LEAGUES["metrica"] + tune.LEAGUES["idsse"] + external.skillcorner_games():
        table, tracks, gt = external.load(name)
        if league(name) != "exam":
            table, _ = external.add_noise(table, tracks["slot"].tolist(),
                                          np.random.default_rng(noise_seed), gt)
        feats = shape_model.match_features(table, tracks, external.PITCH, len(gt))
        missing = np.ones(len(gt))
        missing[table["frame"].to_numpy()] = np.isnan(table["ball_x"].to_numpy(np.float64))
        games.append((name, table, tracks, gt, feats, missing))
    return games


def held_out_models(features):
    m, i = tune.LEAGUES["metrica"], tune.LEAGUES["idsse"]
    return {"metrica": shape_model.train(i, features), "idsse": shape_model.train(m, features),
            "exam": shape_model.train(m + i, features)}


def run(p, features, games):
    models = held_out_models(features) if p["shape"]["enabled"] else None
    out = {}
    for name, table, tracks, gt, feats, missing in games:
        shape = None
        if models:
            shape = (shape_model.logit(feats, models[league(name)]), missing)
        pred = tune.label(p, table, tracks, len(gt), shape)
        out[name] = pm.evaluate(gt, pred)["spell_f1_mean"]
    return out


def with_changes(p, changes):
    p = copy.deepcopy(p)
    for (sec, key), val in changes.items():
        p[sec][key] = val
    return p


def main(variants=VARIANTS):
    base_p = baseline.load_params()
    games = prepare()
    results = {}
    for vname, (changes, features) in variants.items():
        results[vname] = run(with_changes(base_p, changes), features, games)
        ref = next(iter(results.values()))      # the first variant is the reference
        line = []
        for lg in ("metrica", "idsse", "exam"):
            names = [n for n in ref if league(n) == lg]
            new = np.mean([results[vname][n] for n in names])
            better = sum(results[vname][n] > ref[n] + 1e-9 for n in names)
            worse = sum(results[vname][n] < ref[n] - 1e-9 for n in names)
            line.append(f"{lg} {new:.4f} (+{better}/-{worse})")
        print(f"{vname:22s} " + "   ".join(line), flush=True)


if __name__ == "__main__":
    main()
