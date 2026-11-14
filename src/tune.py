"""Tune the m0 numbers on the external games (never on our match).

    uv run python src/tune.py --trials 120

Simple search: start from params.toml, each trial changes ~30% of the numbers of the best set
so far and keeps it if it scores better. Each trial is scored on 3 Metrica games x 2
noise seeds (mean spellF1). The best one is then re-checked on different noise seeds (10-12) so
we don't fool ourselves with lucky noise. Fixed seeds, so it's reproducible.
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


def make_sets(seeds):
    sets = []
    for name in external.GAMES:
        table, tracks, gt = external.load(name)
        for seed in seeds:
            noisy, _ = external.add_noise(table, tracks["slot"].tolist(),
                                          np.random.default_rng(seed), gt)
            sets.append((name, noisy, tracks, gt))
    return sets


def score(p, sets):
    out = []
    for _, noisy, tracks, gt in sets:
        pred = baseline.label_match(noisy, tracks, p, len(gt), external.PITCH)
        r = pm.evaluate(gt, pred)
        out.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    return np.array(out, dtype=float)


def sample(best, rng, share=0.3):
    # change ~30% of the numbers of the best set so far, keep the rest
    p = copy.deepcopy(best)
    for sec, key, lo, hi, log in SPACE:
        if rng.random() > share:
            continue
        p[sec][key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if log
                            else rng.uniform(lo, hi))
    # the probabilities of missing/out/in have to fit in 1
    e = p["evidence"]
    e["p_out_dead"] = min(e["p_out_dead"], 0.95 - e["p_missing_dead"])
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=120)
    args = parser.parse_args()

    base = baseline.load_params()
    tune_sets = make_sets([0, 1])
    rng = np.random.default_rng(2027)

    best_p, best = base, score(base, tune_sets)[:, 0].mean()
    print(f"start: spellF1 {best:.4f}", flush=True)
    for i in range(args.trials):
        p = sample(best_p, rng)
        s = score(p, tune_sets)[:, 0].mean()
        if s > best:
            best, best_p = s, p
            print(f"trial {i}: spellF1 {s:.4f}  new best", flush=True)

    check = make_sets([10, 11, 12])
    old, new = score(base, check), score(best_p, check)
    print("\nre-check on fresh noise (seeds 10-12), per game:")
    for k, name in enumerate(external.GAMES):
        o, n = old[k * 3:(k + 1) * 3].mean(0), new[k * 3:(k + 1) * 3].mean(0)
        print(f"  {name}: spellF1 {o[0]:.3f} -> {n[0]:.3f}   macroF1 {o[1]:.3f} -> {n[1]:.3f}   "
              f"poss {o[2]:.0f} -> {n[2]:.0f} (true {o[3]:.0f})")
    print(f"  MEAN: spellF1 {old[:, 0].mean():.3f} -> {new[:, 0].mean():.3f}   "
          f"macroF1 {old[:, 1].mean():.3f} -> {new[:, 1].mean():.3f}")

    print("\nbest params:")
    for sec, key, *_ in SPACE:
        print(f"  [{sec}] {key} = {best_p[sec][key]:.3f}")


if __name__ == "__main__":
    main()
