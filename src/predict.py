"""Raw match files -> submission, in one go.

    uv run python src/predict.py

Runs convert_data.py (data/ -> data/processed/) and then baseline.py, which writes
evaluation/results/<team>.csv. The team name is [submission] name in configs/params.toml.
Everything is deterministic (no randomness, no GPU), so two runs give the same file.
"""

import hashlib

import baseline
import convert_data


def main():
    convert_data.main()
    baseline.main()
    out = baseline.RESULTS / f"{baseline.load_params()['submission']['name']}.csv"
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
