# More-Than-A-Hack-2027

Team **Leszek Pisz**, possession labelling for BAR-ATH (More Than A Hack 2027).

Tracking in, one label per frame out: `H` (Barcelona has the ball), `A` (Athletic), `D` (dead
ball). An HMM over per-frame clues (player tempo, ball state, who's near the ball, team shape),
decoded with Viterbi. All numbers were tuned on other matches only, never on this one.
Details: [docs/technical.md](docs/technical.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # .venv with the exact versions from uv.lock
uv run python src/predict.py     # raw data -> evaluation/results/Leszek Pisz.csv
```

`predict.py` converts the raw files to Parquet (`src/convert_data.py`), labels the match
(`src/baseline.py`) and prints the sha256 of the file. No randomness, no GPU, so every run gives
the same file byte for byte. The team name lives in `configs/params.toml` (`[submission] name`).

Check the format:

```bash
uv run python evaluation/validate_submission.py "evaluation/results/Leszek Pisz.csv"
```

## Match data

Not in this repo: it's licensed for the competition only. Download it from the link sent after
enrolment and put the files here, keeping their names:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

Videos aren't needed. Everything in `data/` is gitignored. Only the tactical files are used for
the labels; broadcast is converted but not used.

The public ground truth (5 min) is only a check that the pipeline runs, never used to tune or
pick a version (organisers' rules):

```bash
uv run python evaluation/run_evaluation.py
```

## External data (only for tuning and training)

You don't need any of this to reproduce the submission: the tuned numbers are in
`configs/params.toml` and the trained shape model in `models/shape_model.json`. It's only needed
to redo the tuning.

| Set | Use | Licence | Put it in |
|---|---|---|---|
| [Metrica sample data](https://github.com/metrica-sports/sample-data), 3 games | tuning + training | free to use, credit Metrica Sports | `data/external/metrica-sample-data/` (`git clone`) |
| [IDSSE](https://doi.org/10.6084/m9.figshare.28196177), 7 Bundesliga games (Bassek et al. 2025, Scientific Data) | tuning + training | CC-BY 4.0 | `data/external/idsse/` (the `DFL_02_01_matchinformation_*` and `DFL_04_03_positions_raw_observed_*` XMLs) |
| [SkillCorner open data](https://github.com/SkillCorner/opendata), A-League | exam: no numbers fitted on it, used to keep/drop changes | MIT | `data/external/skillcorner-opendata/` (`git clone`) |

### Redo the tuning

```bash
uv run python src/external.py                             # parse the games -> data/external/processed/
uv run python src/shape_model.py --eval                   # shape model, train on one league / test on the other
uv run python src/shape_model.py --train                  # final shape model -> models/shape_model.json
uv run python src/tune.py --from-prior --trials 150       # tune from the hand-picked numbers (params_prior.toml)
uv run python src/tune.py --trials 150                    # or: continue from params.toml (what we did last)
uv run python src/tune.py --exam                          # score params.toml on SkillCorner
```

`tune.py` prints the best numbers; we copied them into `configs/params.toml` by hand. Seeds are
fixed, so it's reproducible (but slow).

## Repository layout

```
configs/params.toml        all numbers, with a note on where each one comes from
configs/params_prior.toml  the hand-picked starting numbers, before any tuning
docs/notes.md              observations about the data
docs/technical.md          technical document
evaluation/                organisers' scorer, validator and public ground truth (unmodified)
games.py                   organisers' match constants, used by evaluation/
models/shape_model.json    team-shape model (trained on Metrica + IDSSE)
src/predict.py             one command: raw data -> submission
src/convert_data.py        raw files -> Parquet
src/baseline.py            per-frame clues + HMM + Viterbi
src/ball.py                patch short holes in the ball track
src/events.py              stoppage, ball-control and restart clues
src/shape_model.py         team shape -> who has the ball
src/decode.py              semi-Markov decoder (tried, off)
src/tags_prior.py          Smart Tagging hint (tried, off)
src/external.py            external games, labels, tracking noise
src/tune.py, validate.py   tuning and scoring on the external games
src/compare.py             switch things on/off, score on fresh noise + the exam
```

`evaluation/` and `games.py` are copied unmodified from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(commit `f8f2629`), under the MIT licence in `evaluation/LICENSE`.
