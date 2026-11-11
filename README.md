# More-Than-A-Hack-2027
This repository serves as a project for More Than A Hack 2027

## Data

The match data is not in this repository: it is licensed for the competition only and must not
be redistributed. Each team member downloads it from the link sent after enrolment and copies the
files here, keeping their names:

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

The videos (`.mp4`) are not needed. Everything in `data/` is gitignored.

## Pipeline

Requires [uv](https://docs.astral.sh/uv/).

```bash
# 1. environment: creates .venv with the exact versions in uv.lock
uv sync

# 2. data: copy the files into data/tactical/ and data/broadcast/ (see above)

# 3. convert the raw files to Parquet in data/processed/ (once, ~3 s)
uv run python src/convert_data.py

# 4. score every CSV in evaluation/results/ against the public 5-minute ground truth
uv run python evaluation/run_evaluation.py

# 5. check a submission's format before sending it
uv run python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Not built yet: the step that generates our submission (`src/predict.py`, writing
`evaluation/results/<team_name>.csv`). It will slot in between steps 3 and 4.

The public ground truth is 5 minutes and 17 possessions: use it to check the pipeline is correct,
never to tune or choose between versions (organisers' rules).

## Repository layout

```
configs/          parameters
data/             match data (gitignored); processed/ holds the Parquet files
docs/notes.md     observations about the data
evaluation/       organisers' scorer, validator and public ground truth (unmodified)
games.py          organisers' match constants, used by evaluation/
models/           saved models
src/              our code
```

`evaluation/` and `games.py` are copied unmodified from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(commit `f8f2629`), under the MIT licence in `evaluation/LICENSE`.
