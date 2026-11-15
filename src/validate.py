"""Score our labeller on the external games (module 1). This is our real validation.

    uv run python src/validate.py            # 3 games x 3 noise seeds
    uv run python src/validate.py --seeds 5

Each run: take a clean game, add random tracking noise, label it, score it with the organisers'
own metric code against the labels from events. We also score the organisers' smooth50
baseline on the same noisy data, as a reference point (on the real match it gets 0.2433).

Careful: with [shape] enabled this uses models/shape_model.json, which was trained on these same
Metrica games, so the number comes out too good. Honest numbers with module 5: shape_model.py --eval
and tune.py (held-out shape models).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evaluation"))

import baseline  # noqa: E402
import external  # noqa: E402
import possession_metrics as pm  # noqa: E402


def smooth50(table, tracks, n_frames, window=50):
    # organisers' baseline: nearest player's team, majority vote over +-window, D if nobody
    slots_h = tracks.loc[tracks["team"] == "H", "slot"].tolist()
    slots_a = tracks.loc[tracks["team"] == "A", "slot"].tolist()
    bx, by = table["ball_x"].to_numpy(), table["ball_y"].to_numpy()

    def dist(slots):
        px = table[[f"p{s}_x" for s in slots]].to_numpy()
        py = table[[f"p{s}_y" for s in slots]].to_numpy()
        d = np.hypot(px - bx[:, None], py - by[:, None])
        return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)

    dh, da = dist(slots_h), dist(slots_a)
    seen = np.isfinite(np.minimum(dh, da))
    home = np.zeros(n_frames)
    away = np.zeros(n_frames)
    f = table["frame"].to_numpy()
    home[f[seen & (dh <= da)]] = 1
    away[f[seen & (da < dh)]] = 1
    kernel = np.ones(2 * window + 1)
    h = np.convolve(home, kernel, "same")
    a = np.convolve(away, kernel, "same")
    return np.where(h + a == 0, baseline.D, np.where(h >= a, baseline.H, baseline.A)).astype(np.int8)


def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def run(p, seeds=3, verbose=True):
    rows = []
    for name in external.GAMES:
        table, tracks, gt = external.load(name)
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            noisy, _ = external.add_noise(table, tracks["slot"].tolist(), rng, gt)
            ours = score(gt, baseline.label_match(noisy, tracks, p, len(gt), external.PITCH))
            ref = score(gt, smooth50(noisy, tracks, len(gt)))
            rows.append((name, seed, *ours, *ref[:2]))
            if verbose:
                print(f"{name} seed {seed}:  ours spellF1 {ours[0]:.3f} macroF1 {ours[1]:.3f} "
                      f"poss {ours[2]} (true {ours[3]})   |  smooth50 spellF1 {ref[0]:.3f} "
                      f"macroF1 {ref[1]:.3f}")
    rows = np.array([r[2:] for r in rows], dtype=float)
    mean = rows.mean(axis=0)
    if verbose:
        print(f"\nMEAN  ours spellF1 {mean[0]:.3f} macroF1 {mean[1]:.3f}   |  "
              f"smooth50 spellF1 {mean[4]:.3f} macroF1 {mean[5]:.3f}")
    return mean[0], mean[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    run(baseline.load_params(), args.seeds)
