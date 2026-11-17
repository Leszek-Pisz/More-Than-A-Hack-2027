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
# 17.11 round 1 (ball gaps, restart team, shape + press / in_opp): results in STRATEGY.md.
# Now: the short-blip filter (post.py), a few minimum lengths.
def _post(poss, dead):
    return {("post", "enabled"): True, ("post", "min_poss_s"): poss, ("post", "min_dead_s"): dead}


POST_OFF = {("post", "enabled"): False}
EDGES_OFF = {("decoder", "refine_edges"): False}

# The first variant is the reference. params.toml now has both the blip filter (post.py, 1 s / 1 s)
# and moving switches to the ball (refine_edges) on; this checks they work together.
VARIANTS = {
    "edges, no post (ref)": (POST_OFF, BASE_SHAPE),
    "edges + post 1/1": ({}, BASE_SHAPE),
    "post 1/1, no edges": (EDGES_OFF, BASE_SHAPE),
    "no post, no edges": ({**POST_OFF, **EDGES_OFF}, BASE_SHAPE),
}


def league(name):
    return "metrica" if name in tune.LEAGUES["metrica"] else \
        "idsse" if name in tune.LEAGUES["idsse"] else "exam"


# Without IDSSE downloaded: skip those games and use the saved shape model (models/shape_model.json)
# everywhere. It saw Metrica in training, so Metrica scores are a bit flattering, but on/off of the
# same switch is still a fair comparison. The exam stays honest (the model never saw SkillCorner).
HAVE_IDSSE = all((external.OUT / f"{n}_labels.npy").exists() for n in tune.LEAGUES["idsse"])


def prepare(noise_seed=10, metrica_only=False):
    games = []
    idsse = tune.LEAGUES["idsse"] if HAVE_IDSSE and not metrica_only else ()
    exam = () if metrica_only else external.skillcorner_games()
    if not HAVE_IDSSE:
        print("IDSSE not found: Metrica + exam only, saved shape model", flush=True)
    for name in tune.LEAGUES["metrica"] + idsse + exam:
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
    if not HAVE_IDSSE:
        saved = shape_model.load_model()
        return {"metrica": saved, "idsse": saved, "exam": saved}
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
    # `compare.py metrica`: Metrica only, a quick look while the rest downloads (not for deciding)
    base_p = baseline.load_params()
    games = prepare(metrica_only="metrica" in sys.argv[1:])
    results = {}
    for vname, (changes, features) in variants.items():
        results[vname] = run(with_changes(base_p, changes), features, games)
        ref = next(iter(results.values()))      # the first variant is the reference
        line = []
        for lg in ("metrica", "idsse", "exam"):
            names = [n for n in ref if league(n) == lg]
            if not names:
                continue
            new = np.mean([results[vname][n] for n in names])
            better = sum(results[vname][n] > ref[n] + 1e-9 for n in names)
            worse = sum(results[vname][n] < ref[n] - 1e-9 for n in names)
            line.append(f"{lg} {new:.4f} (+{better}/-{worse})")
        print(f"{vname:22s} " + "   ".join(line), flush=True)


if __name__ == "__main__":
    main()
