"""Module 1: other matches (Metrica sample data) so we can actually check / tune stuff.

    uv run python src/external.py      # parses the 3 games once -> data/external/processed/

For each game we get the same kind of table as our match (frame, half, ball_x/y, p<slot>_x/y/v
in metres, plus a tracks table slot -> H/A) and true labels H/A/D/X built from the events.
add_noise() then messes up the clean tracking the way automatic tracking gets messed up.

Labels from events:
- possession = team of the last on-ball event (pass, carry, set piece, recovery, shot, ball
  lost). Challenges don't change it (contested ball stays with the last team).
- dead ball starts at: ball out, shot that ends out/goal, foul, offside/handball,
  and ends at the next set piece (the restart). If there's no set piece we use the next event.
- X outside the periods (before kick-off etc.), same as the organisers' GT.

Metrica data: github.com/metrica-sports/sample-data (use it, credit the source).
"""

import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICA = ROOT / "data" / "external" / "metrica-sample-data" / "data"
OUT = ROOT / "data" / "external" / "processed"
FPS = 25
PITCH = (105.0, 68.0)
GAMES = ("metrica1", "metrica2", "metrica3")
H, A, D, X = 0, 1, 2, 3

OWNER_EVENTS = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}


# ---------------------------------------------------------------- loading

def _speeds(table, slots):
    # tracking systems give a smoothed speed, so do the same: diff + 0.4 s rolling mean
    for s in slots:
        x, y = table[f"p{s}_x"].to_numpy(), table[f"p{s}_y"].to_numpy()
        v = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * FPS
        table[f"p{s}_v"] = pd.Series(v).rolling(10, center=True, min_periods=1).mean().to_numpy()
    return table


def _csv_game(n):
    folder = METRICA / f"Sample_Game_{n}"
    parts = {}
    for side, team in (("Home", "H"), ("Away", "A")):
        path = folder / f"Sample_Game_{n}_RawTrackingData_{side}_Team.csv"
        names = pd.read_csv(path, skiprows=2, nrows=0).columns
        players = [c for c in names if c.startswith("Player")]
        cols = ["period", "frame", "time"]
        for p in players:
            cols += [f"{p}_x", f"{p}_y"]
        cols += ["ball_x", "ball_y"]
        parts[team] = (pd.read_csv(path, skiprows=3, header=None, names=cols), players)

    home, home_players = parts["H"]
    away, away_players = parts["A"]
    table = pd.DataFrame({"frame": home["frame"].to_numpy() - 1,
                          "half": home["period"].to_numpy(),
                          "ball_x": home["ball_x"].to_numpy() * PITCH[0],
                          "ball_y": home["ball_y"].to_numpy() * PITCH[1]})
    tracks, slot = [], 1
    for team, df, players in (("H", home, home_players), ("A", away, away_players)):
        for p in players:
            table[f"p{slot}_x"] = df[f"{p}_x"].to_numpy() * PITCH[0]
            table[f"p{slot}_y"] = df[f"{p}_y"].to_numpy() * PITCH[1]
            tracks.append({"slot": slot, "team": team})
            slot += 1
    tracks = pd.DataFrame(tracks)

    ev = pd.read_csv(folder / f"Sample_Game_{n}_RawEventsData.csv")
    events = pd.DataFrame({
        "team": ev["Team"].map({"Home": "H", "Away": "A"}),
        "type": ev["Type"].str.upper(),
        "subtype": ev["Subtype"].fillna("").str.upper(),
        "start": ev["Start Frame"] - 1,
        "end": ev["End Frame"] - 1,
    })
    return _speeds(table, tracks["slot"]), tracks, events


def _game3():
    folder = METRICA / "Sample_Game_3"
    root = ET.parse(folder / "Sample_Game_3_metadata.xml").getroot()
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    teams = [t.get("id") for t in root.iter("Team")]
    home_id = teams[0]
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}

    # every column position has to stay with the same team across the spec blocks (subs)
    col_teams = None
    for spec in root.iter("DataFormatSpecification"):
        ids = [channel_player[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
               if r.get("playerChannelId").endswith("_x")]
        teams_here = ["H" if team_of_player[i] == home_id else "A" for i in ids]
        if col_teams is None:
            col_teams = teams_here
        elif teams_here != col_teams:
            raise ValueError("game 3: a column switches team between blocks")

    n_players = len(col_teams)
    text = (folder / "Sample_Game_3_tracking.txt").read_text()
    text = text.replace(":", ",").replace(";", ",")
    cols = ["frame"] + [f"c{i}_{a}" for i in range(n_players) for a in "xy"] + ["ball_x", "ball_y"]
    raw = pd.read_csv(io.StringIO(text), header=None, names=cols)

    first_end = int(params["first_half_end"])
    frame = raw["frame"].to_numpy()
    table = pd.DataFrame({"frame": frame - 1,
                          "half": np.where(frame <= first_end, 1, 2),
                          "ball_x": raw["ball_x"].to_numpy() * PITCH[0],
                          "ball_y": raw["ball_y"].to_numpy() * PITCH[1]})
    for i in range(n_players):
        table[f"p{i + 1}_x"] = raw[f"c{i}_x"].to_numpy() * PITCH[0]
        table[f"p{i + 1}_y"] = raw[f"c{i}_y"].to_numpy() * PITCH[1]
    tracks = pd.DataFrame({"slot": np.arange(1, n_players + 1), "team": col_teams})

    data = json.loads((folder / "Sample_Game_3_events.json").read_text())["data"]

    def sub(e):
        s = e.get("subtypes")
        if not s:
            return ""
        s = s if isinstance(s, list) else [s]
        return "-".join(x["name"] for x in s).upper()

    events = pd.DataFrame({
        "team": ["H" if e["team"]["id"] == home_id else "A" for e in data],
        "type": [e["type"]["name"].upper() for e in data],
        "subtype": [sub(e) for e in data],
        "start": [e["start"]["frame"] - 1 for e in data],
        "end": [(e["end"]["frame"] or e["start"]["frame"]) - 1 for e in data],
    })
    return _speeds(table, tracks["slot"]), tracks, events


# ---------------------------------------------------------------- labels

def _dead_start(row):
    t, s = row.type, row.subtype
    end = row.end if row.end >= row.start else row.start
    if t == "BALL OUT":
        return end
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return end
    if t == "FAULT RECEIVED":
        return row.start
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return row.start
    if t in ("BALL LOST", "PASS") and ("OFFSIDE" in s or "HAND BALL" in s or "END HALF" in s):
        return end
    return None


def labels_from_events(events, table, n_frames):
    codes = np.full(n_frames, X, dtype=np.int8)
    ev = events.sort_values("start", kind="stable").reset_index(drop=True)

    for half in (1, 2):
        frames = table.loc[table["half"] == half, "frame"]
        lo, hi = frames.min(), frames.max()
        e = ev[(ev["start"] >= lo) & (ev["start"] <= hi)]
        if e.empty:
            continue
        first, last = int(e["start"].min()), int(max(e["start"].max(), e["end"].max()))
        last = min(last, hi)

        # who owns the ball
        own = e[e["type"].isin(OWNER_EVENTS)]
        owner = np.full(last - first + 1, -1, dtype=np.int8)
        for row in own.itertuples():
            owner[max(row.start, first) - first:] = H if row.team == "H" else A
        owner[owner < 0] = D   # before the first on-ball event (shouldn't really happen)

        # dead ball stretches: trigger -> first time someone plays the ball again.
        # Metrica logs goal kicks etc. as plain passes, not set pieces, so any pass/carry/shot
        # counts as the restart. Challenges, cards, "fault received" don't.
        restarts = e.loc[e["type"].isin({"SET PIECE", "PASS", "CARRY", "SHOT"}), "start"]
        restarts = restarts.to_numpy()
        dead = np.zeros_like(owner, dtype=bool)
        for row in e.itertuples():
            t0 = _dead_start(row)
            if t0 is None or t0 > last:
                continue
            nxt = restarts[restarts > t0]
            t1 = nxt[0] if len(nxt) else last + 1
            dead[max(t0, first) - first:min(t1, last + 1) - first] = True
        owner[dead] = D
        codes[first:last + 1] = owner
    return codes


# ---------------------------------------------------------------- noise

def _wobble(rng, n, sd, seconds=1.0):
    # smooth position error (real tracking drifts, it doesn't shake every frame)
    k = int(seconds * FPS)
    w = np.convolve(rng.normal(0, 1, n + k), np.ones(k) / np.sqrt(k), "valid")[:n]
    return w * sd


def add_noise(table, slots, rng, labels=None):
    """Mess up clean tracking like automatic tracking does. Every knob is drawn from a wide
    range picked up front, NOT measured on our match (rules)."""
    t = table.copy()
    n = len(t)
    knobs = {
        "ball_missing": rng.uniform(0.15, 0.5),     # share of frames without the ball
        "gap_median_s": rng.uniform(3, 20),         # long gaps, not flicker
        "dead_bias": rng.uniform(0.0, 0.7),         # gaps like stoppages more
        "box_bias": rng.uniform(1.0, 3.0),          # ...and the area near goals
        "jumps_per_min": rng.uniform(0.5, 3.0),     # one-frame fake ball positions
        "ball_sd": rng.uniform(0.05, 0.5),          # m, smooth drift (see _wobble)
        "player_sd": rng.uniform(0.05, 0.4),        # m
        "player_missing": rng.uniform(0.02, 0.15),
        "player_gap_s": rng.uniform(1, 10),
        "speed_err": rng.uniform(0.05, 0.25),       # relative error on player speed
    }

    bx, by = t["ball_x"].to_numpy().copy(), t["ball_y"].to_numpy().copy()
    bx += _wobble(rng, n, knobs["ball_sd"], 2.0)
    by += _wobble(rng, n, knobs["ball_sd"], 2.0)

    # where a gap likes to start
    weight = np.ones(n)
    if labels is not None:
        weight[labels[t["frame"].to_numpy()] == D] += knobs["dead_bias"] * 5
    in_box = ((bx < 16.5) | (bx > PITCH[0] - 16.5)) & (np.abs(by - PITCH[1] / 2) < 20.2)
    weight[in_box] *= knobs["box_bias"]
    weight /= weight.sum()

    missing = np.isnan(bx)
    target = knobs["ball_missing"] * n
    sigma = 0.8
    while missing.sum() < target:
        start = rng.choice(n, p=weight)
        length = int(rng.lognormal(np.log(knobs["gap_median_s"] * FPS), sigma))
        missing[start:start + max(length, 1)] = True
    bx[missing] = np.nan
    by[missing] = np.nan

    n_jumps = int(knobs["jumps_per_min"] * n / (60 * FPS))
    seen = np.flatnonzero(~missing)
    if len(seen):
        idx = rng.choice(seen, size=min(n_jumps, len(seen)), replace=False)
        bx[idx] = rng.uniform(0, PITCH[0], len(idx))
        by[idx] = rng.uniform(0, PITCH[1], len(idx))
    t["ball_x"], t["ball_y"] = bx, by

    for s in slots:
        x = t[f"p{s}_x"].to_numpy().copy() + _wobble(rng, n, knobs["player_sd"])
        y = t[f"p{s}_y"].to_numpy().copy() + _wobble(rng, n, knobs["player_sd"])
        gone = np.zeros(n, dtype=bool)
        mean_len = knobs["player_gap_s"] * FPS
        n_gaps = int(knobs["player_missing"] * n / mean_len)
        for start in rng.integers(0, n, n_gaps):
            gone[start:start + int(rng.exponential(mean_len)) + 1] = True
        x[gone] = np.nan
        y[gone] = np.nan
        t[f"p{s}_x"], t[f"p{s}_y"] = x, y
        # trackers give a filtered speed, so we don't recompute it from the shaky positions,
        # just add a bit of error to the clean one
        v = t[f"p{s}_v"].to_numpy() * (1 + _wobble(rng, n, knobs["speed_err"], 2.0))
        v[gone] = np.nan
        t[f"p{s}_v"] = np.clip(v, 0, None)
    return t, knobs


# ---------------------------------------------------------------- cache

def load(name):
    """(table, tracks, labels) for one game, from the cache."""
    table = pd.read_parquet(OUT / f"{name}_tracking.parquet")
    tracks = pd.read_parquet(OUT / f"{name}_tracks.parquet")
    labels = np.load(OUT / f"{name}_labels.npy")
    return table, tracks, labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaders = {"metrica1": lambda: _csv_game(1), "metrica2": lambda: _csv_game(2),
               "metrica3": _game3}
    for name in GAMES:
        table, tracks, events = loaders[name]()
        labels = labels_from_events(events, table, int(table["frame"].max()) + 1)
        table.to_parquet(OUT / f"{name}_tracking.parquet", index=False)
        tracks.to_parquet(OUT / f"{name}_tracks.parquet", index=False)
        np.save(OUT / f"{name}_labels.npy", labels)
        counts = {k: int((labels == i).sum()) for i, k in enumerate("HADX")}
        spells = int(np.sum(np.diff(labels) != 0))
        print(f"{name}: {len(table)} frames, {len(tracks)} slots, labels {counts}, "
              f"~{spells} label changes")


if __name__ == "__main__":
    main()
