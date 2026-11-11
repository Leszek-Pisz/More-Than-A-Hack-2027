"""Convert the raw match data in data/ to Parquet files in data/processed/.

    uv run python src/convert_data.py

Raw files are read, never modified. Output, per source (tactical, broadcast):

    <source>_tracking.parquet     one row per video frame: frame, [half], ball_*, p<slot>_*
    <source>_tracks.parquet       one row per track slot: slot, team (H/A), team_name
    <source>_homography.parquet   the homography CSV as is
    tactical_tags.parquet         Smart Tagging intervals, in video frames (tactical only)

Conversions done here, once, so nothing downstream has to remember them:
  - frame counts from 0, like a submission row (the raw tracking counts from 1)
  - positions in metres (raw is normalised 0-1); speeds stay in m/s
  - teams as H (home, Barcelona) / A (away, Athletic)
  - tactical only: `half` = 1 or 2 on graded frames, missing elsewhere
  - Smart Tagging seconds -> video frames (round(seconds * 25))
"""

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"

FPS = 25
# Real pitch size in metres, as measured by the organisers (games.py). The ATD metadata says
# 105 x 68; the 30 cm only matters for distances in metres.
PITCH_LENGTH, PITCH_WIDTH = 105.3, 68.0
# Graded video frames of each half, inclusive (organisers' DATA.md). Tactical timeline only; the
# broadcast recording is on its own, unaligned timeline.
HALVES = {1: (11185, 90651), 2: (110808, 190197)}


def read_tracks(meta_path):
    """Slot -> team, from the ATD metadata. Home is the Score's idLocalTeam."""
    root = ET.parse(meta_path).getroot()
    home_id = root.find(".//Score").get("idLocalTeam")
    names = {team.get("id"): team.findtext("Name") for team in root.iter("Team")}
    rows = [{"slot": int(player.get("id")),
             "team": "H" if player.get("teamId") == home_id else "A",
             "team_name": names[player.get("teamId")]}
            for player in root.iter("Player")]
    return pd.DataFrame(rows)


def read_tracking(raw_path, slots, n_frames=None):
    """Raw "frame:x,y,v;x,y,v;...:x,y,v" lines -> one wide DataFrame, positions in metres.

    The three separators are all turned into commas, so pandas' fast C parser reads the file as
    a plain CSV with 1 + 3 * (n_slots + 1) columns.
    """
    text = raw_path.read_text().replace(":", ",").replace(";", ",")
    columns = (["frame"]
               + [f"p{slot}_{field}" for slot in slots for field in ("x", "y", "v")]
               + ["ball_x", "ball_y", "ball_v"])
    table = pd.read_csv(io.StringIO(text), header=None, names=columns,
                        dtype={name: np.float32 for name in columns[1:]})

    frame = table["frame"].to_numpy() - 1
    if n_frames is not None and not np.array_equal(frame, np.arange(n_frames)):
        raise ValueError(f"{raw_path.name}: frames are not 1..{n_frames} in order")

    scale = np.array([PITCH_LENGTH if name.endswith("_x") else
                      PITCH_WIDTH if name.endswith("_y") else 1.0
                      for name in columns[1:]], dtype=np.float32)
    values = table[columns[1:]].to_numpy(np.float32) * scale
    return pd.concat([pd.DataFrame({"frame": frame}),
                      pd.DataFrame(values, columns=columns[1:])], axis=1)


def add_half(table):
    half = pd.Series(pd.NA, index=table.index, dtype="Int8")
    for number, (first, last) in HALVES.items():
        half[table["frame"].between(first, last)] = number
    table.insert(1, "half", half)
    return table


def read_tags(xml_path):
    """Smart Tagging instances -> one row each, tag groups as snake_case columns."""
    rows = []
    for instance in ET.parse(xml_path).iter("instance"):
        row = {
            "id": int(instance.findtext("ID")),
            "code": instance.findtext("code").strip().upper(),
            "start_frame": round(float(instance.findtext("start")) * FPS),
            "end_frame": round(float(instance.findtext("end")) * FPS),
        }
        for label in instance.findall("label"):
            column = label.findtext("group").strip().lower().replace(" ", "_")
            row[column] = label.findtext("text")
        rows.append(row)
    tags = pd.DataFrame(rows)
    # "Barcelona" / "Atletic Club" (sic) / "N/A" -> H / A / missing
    tags["team"] = tags["team"].map({"Barcelona": "H", "Atletic Club": "A"})
    return tags


def convert(source, n_frames=None):
    folder = DATA_DIR / source
    stem = f"BAR-ATH_{source}"
    print(f"{source}:")

    tracks = read_tracks(folder / f"{stem}_atd.xml")
    tracking = read_tracking(folder / f"{stem}_atd.txt", tracks["slot"].tolist(), n_frames)
    if source == "tactical":
        tracking = add_half(tracking)
    homography = pd.read_csv(folder / f"{stem}_homography.csv")

    outputs = {"tracking": tracking, "tracks": tracks, "homography": homography}
    tags_path = folder / f"{stem}_smart_tagging.xml"
    if tags_path.is_file():
        outputs["tags"] = read_tags(tags_path)

    for name, table in outputs.items():
        path = OUT_DIR / f"{source}_{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"  {path.relative_to(ROOT)}  {len(table):>7} rows  "
              f"{path.stat().st_size / 1e6:6.1f} MB")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    convert("tactical", n_frames=199319)
    if (DATA_DIR / "broadcast").is_dir():
        convert("broadcast", n_frames=164488)


if __name__ == "__main__":
    main()
