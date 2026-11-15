"""Tune the numbers on the external games (never on our match).

    uv run python src/tune.py --from-prior --trials 150   # tune on Metrica + IDSSE
    uv run python src/tune.py --cv --trials 100           # tune on one league, test on the other
    uv run python src/tune.py --exam                      # score params.toml on SkillCorner

Simple search: start from a params file, each trial changes ~30% of the numbers of the best set
so far and keeps it if it scores better. Score = mean spellF1 per league (Metrica, IDSSE), games
with our random tracking noise. Re-checked on fresh noise, then the SkillCorner exam (real
broadcast holes, never tuned on). Fixed seeds, so it's reproducible.

Module 5 (shape model, when [shape] enabled): every game is scored with a shape model trained on
the OTHER league (SkillCorner: on both, it never saw any of those). Otherwise the shape clue would
look better on the tuning games than on a new match and the search would trust it too much.
"""

import argparse
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

# (section, key, low, high, log scale?)
SPACE = [
    ("tempo", "alive_mean", 1.6, 2.6, False),
    ("tempo", "alive_sd", 0.3, 1.2, False),
    ("tempo", "dead_mean", 0.6, 1.5, False),
    ("tempo", "dead_sd", 0.2, 0.9, False),
    ("tempo", "window_s", 1.0, 5.0, False),
    ("evidence", "control_radius", 1.0, 4.0, False),
    ("evidence", "p_nearest_is_owner", 0.55, 0.9, False),
    ("evidence", "p_missing_alive", 0.1, 0.5, False),
    ("evidence", "p_missing_dead", 0.3, 0.85, False),
    ("evidence", "p_out_dead", 0.05, 0.5, False),
    ("hmm", "possession_mean_s", 5.0, 60.0, True),
    ("hmm", "dead_mean_s", 5.0, 60.0, True),
    ("hmm", "p_turnover", 0.2, 0.8, False),
    ("events", "still_speed", 0.3, 2.0, False),
    ("events", "still_s", 0.4, 2.0, False),
    ("events", "p_still_alive", 0.01, 0.3, True),
    ("events", "p_spot_alive", 0.02, 0.5, True),
    ("events", "line_margin", 0.3, 3.0, False),
    ("events", "p_gapline_alive", 0.05, 0.6, True),
    ("events", "control_radius", 0.8, 2.5, False),
    ("events", "control_s", 0.1, 1.0, False),
    ("events", "p_control_is_owner", 0.55, 0.97, False),
]
# only used when [shape] enabled = true
SPACE_SHAPE = [
    ("shape", "weight", 0.02, 0.5, True),
]
# only used when [decoder] type = "hsmm"
SPACE_HSMM = [
    ("decoder", "evidence_scale", 0.1, 1.5, True),
    ("decoder", "split_penalty", -20.0, -1.0, False),
]


LEAGUES = {"metrica": external.GAMES,
           "idsse": tuple(f"idsse_{m}" for m in external.IDSSE_GAMES)}


_SHAPE_MODELS = {}


def shape_model_for(name):
    # held-out shape model: trained on the other league, or on both for SkillCorner
    key = ("idsse" if name in LEAGUES["metrica"] else
           "metrica" if name in LEAGUES["idsse"] else "both")
    if key not in _SHAPE_MODELS:
        games = LEAGUES["metrica"] + LEAGUES["idsse"] if key == "both" else LEAGUES[key]
        _SHAPE_MODELS[key] = shape_model.train(games)
    return _SHAPE_MODELS[key]


def shape_inputs(name, table, tracks, n):
    # the shape logit doesn't depend on any tuned number, so it's computed once per game
    feats = shape_model.match_features(table, tracks, external.PITCH, n)
    z = shape_model.logit(feats, shape_model_for(name))
    missing = np.ones(n)
    missing[table["frame"].to_numpy()] = np.isnan(table["ball_x"].to_numpy(np.float64))
    return z, missing


def label(p, table, tracks, n, shape):
    s = p.get("shape", {})
    if not s.get("enabled"):
        return baseline.label_match(table, tracks, p, n, external.PITCH)
    z, missing = shape
    plain = {**p, "shape": {**s, "enabled": False}}     # added via extra= instead
    return baseline.label_match(table, tracks, plain, n, external.PITCH,
                                extra=shape_model.loglik_from_logit(z, s, missing))


def make_sets(seeds, games):
    sets = []
    for name in games:
        table, tracks, gt = external.load(name)
        for seed in seeds:
            noisy, _ = external.add_noise(table, tracks["slot"].tolist(),
                                          np.random.default_rng(seed), gt)
            sets.append((name, noisy, tracks, gt, shape_inputs(name, noisy, tracks, len(gt))))
    return sets


def score(p, sets):
    out = []
    for _, noisy, tracks, gt, shape in sets:
        pred = label(p, noisy, tracks, len(gt), shape)
        r = pm.evaluate(gt, pred)
        out.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    return np.array(out, dtype=float)


def objective(p, sets):
    # each league counts the same, so 7 Bundesliga games don't drown the 3 Metrica ones
    res = score(p, sets)
    names = [n for n, *_ in sets]
    per = [res[[i for i, n in enumerate(names) if n in games], 0].mean()
           for games in LEAGUES.values() if any(n in games for n in names)]
    return float(np.mean(per))


def space_of(p):
    return (SPACE + (SPACE_HSMM if p.get("decoder", {}).get("type") == "hsmm" else [])
            + (SPACE_SHAPE if p.get("shape", {}).get("enabled") else []))


def sample(best, rng, share=0.3):
    # change ~30% of the numbers of the best set so far, keep the rest
    p = copy.deepcopy(best)
    for sec, key, lo, hi, log in space_of(p):
        if rng.random() > share:
            continue
        p[sec][key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if log
                            else rng.uniform(lo, hi))
    # the probabilities of missing/out/in have to fit in 1
    e = p["evidence"]
    e["p_out_dead"] = min(e["p_out_dead"], 0.95 - e["p_missing_dead"])
    return p


def search(base, sets, trials, rng, log=True):
    best_p, best = base, objective(base, sets)
    if log:
        print(f"start: spellF1 {best:.4f}", flush=True)
    for i in range(trials):
        p = sample(best_p, rng)
        sc = objective(p, sets)
        if sc > best:
            best, best_p = sc, p
            if log:
                print(f"trial {i}: spellF1 {sc:.4f}  new best", flush=True)
    return best_p


def report(title, before, after, sets):
    print(title)
    names = [n for n, *_ in sets]
    for league, games in LEAGUES.items():
        idx = [i for i, n in enumerate(names) if n in games]
        if not idx:
            continue
        o, n = before[idx].mean(0), after[idx].mean(0)
        print(f"  {league:8s} spellF1 {o[0]:.3f} -> {n[0]:.3f}   macroF1 {o[1]:.3f} -> "
              f"{n[1]:.3f}   poss/game {o[2]:.0f} -> {n[2]:.0f} (true {o[3]:.0f})", flush=True)


def cross_validate(args, base):
    # tune on one league, test on the other one it never saw
    for train, test in (("idsse", "metrica"), ("metrica", "idsse")):
        print(f"\ntune on {train} -> test on {test} (unseen league)", flush=True)
        best_p = search(base, make_sets([20], LEAGUES[train]), args.trials,
                        np.random.default_rng(7), log=False)
        sets = make_sets([10], LEAGUES[test])
        report(f"  result on {test}:", score(base, sets), score(best_p, sets), sets)


_EXAM_SHAPE = {}


def exam(p):
    # SkillCorner: real broadcast tracking holes, no added noise, never tuned on
    rows = []
    for name in external.skillcorner_games():
        table, tracks, gt = external.load(name)
        if name not in _EXAM_SHAPE:
            _EXAM_SHAPE[name] = shape_inputs(name, table, tracks, len(gt))
        pred = label(p, table, tracks, len(gt), _EXAM_SHAPE[name])
        r = pm.evaluate(gt, pred)
        rows.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    rows = np.array(rows)
    m = rows.mean(0)
    print(f"EXAM skillcorner ({len(rows)} games): spellF1 {m[0]:.3f}  macroF1 {m[1]:.3f}  "
          f"poss/game {m[2]:.0f} (true {m[3]:.0f})", flush=True)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--decoder", choices=["hmm", "hsmm"], help="override [decoder] type")
    parser.add_argument("--cv", action="store_true", help="tune on one league, test on the other")
    parser.add_argument("--exam", action="store_true", help="only score params.toml on SkillCorner")
    parser.add_argument("--from-prior", action="store_true",
                        help="start from params_prior.toml (hand-picked) instead of params.toml")
    args = parser.parse_args()

    path = ROOT / "configs" / ("params_prior.toml" if args.from_prior or args.cv else "params.toml")
    base = baseline.load_params(path)
    if args.decoder:
        base["decoder"]["type"] = args.decoder
    if args.exam:
        return exam(base)
    if args.cv:
        return cross_validate(args, base)

    games = LEAGUES["metrica"] + LEAGUES["idsse"]
    best_p = search(base, make_sets([0], games), args.trials, np.random.default_rng(2027))
    check = make_sets([10], games)
    report("\nre-check on fresh noise (seed 10):", score(base, check), score(best_p, check), check)
    print("\nexam before / after:")
    exam(base)
    exam(best_p)

    print("\nbest params:")
    for sec, key, *_ in space_of(best_p):
        print(f"  [{sec}] {key} = {best_p[sec][key]:.3f}")


if __name__ == "__main__":
    main()
