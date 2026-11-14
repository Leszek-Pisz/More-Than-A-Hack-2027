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


# ---------------------------------------------------------------- IDSSE (Bundesliga, DFL)
# 7 games, CC-BY 4.0. Bassek et al. 2025, Scientific Data, figshare 10.6084/m9.figshare.28196177
# The ball frames carry BallPossession (1 home / 2 guest) and BallStatus (1 alive / 0 dead) for
# every frame, i.e. the same kind of truth the organisers built our GT from.

IDSSE = ROOT / "data" / "external" / "idsse"
IDSSE_GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


def _attr(line, key):
    i = line.find(f' {key}="')
    if i < 0:
        return None
    i += len(key) + 3
    return line[i:line.index('"', i)]


def _idsse_game(match):
    info = next(IDSSE.glob(f"DFL_02_01_matchinformation_*{match}.xml"))
    root = ET.parse(info).getroot()
    team_role = {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}

    # stream the big positions file line by line, way faster than a full XML parse
    pos = next(IDSSE.glob(f"DFL_04_03_positions_raw_observed_*{match}.xml"))
    series = {}       # (section, team, person) -> (frames, x, y, s, poss, status)
    cur = None
    with open(pos, encoding="utf-8") as f:
        for line in f:
            if line.startswith("<FrameSet"):
                key = (_attr(line, "GameSection"), _attr(line, "TeamId"), _attr(line, "PersonId"))
                cur = series.setdefault(key, ([], [], [], [], [], []))
            elif line.startswith("<Frame ") and cur is not None:
                cur[0].append(int(_attr(line, "N")))
                cur[1].append(float(_attr(line, "X")))
                cur[2].append(float(_attr(line, "Y")))
                cur[3].append(float(_attr(line, "S") or "nan"))
                cur[4].append(int(_attr(line, "BallPossession") or 0))
                cur[5].append(int(_attr(line, "BallStatus") or 0))

    halves = [s for s in ("firstHalf", "secondHalf")]
    tables, labels, offset = [], [], 0
    slot_of, tracks = {}, []
    for h, section in enumerate(halves, start=1):
        ball = next(v for k, v in series.items() if k[0] == section and k[1] == "BALL")
        n0 = ball[0][0]
        n = ball[0][-1] - n0 + 1
        t = {"frame": np.arange(n) + offset, "half": np.full(n, h),
             "ball_x": np.full(n, np.nan), "ball_y": np.full(n, np.nan)}
        idx = np.array(ball[0]) - n0
        t["ball_x"][idx] = np.array(ball[1]) + 52.5
        t["ball_y"][idx] = np.array(ball[2]) + 34.0
        lab = np.full(n, D, dtype=np.int8)
        poss, status = np.array(ball[4]), np.array(ball[5])
        lab[idx] = np.where(status == 0, D, np.where(poss == 1, H, A))
        for (sec, team, person), v in series.items():
            if sec != section or team not in team_role:
                continue
            if person not in slot_of:
                slot_of[person] = len(slot_of) + 1
                tracks.append({"slot": slot_of[person],
                               "team": "H" if team_role[team] == "home" else "A"})
            s = slot_of[person]
            for col in ("x", "y", "v"):
                t.setdefault(f"p{s}_{col}", np.full(n, np.nan))
            i = np.array(v[0]) - n0
            ok = (i >= 0) & (i < n)
            t[f"p{s}_x"][i[ok]] = np.array(v[1])[ok] + 52.5
            t[f"p{s}_y"][i[ok]] = np.array(v[2])[ok] + 34.0
            t[f"p{s}_v"][i[ok]] = np.array(v[3])[ok] / 3.6     # DFL gives km/h
        tables.append(pd.DataFrame(t))
        labels.append(lab)
        labels.append(np.array([X], dtype=np.int8))   # one ungraded frame between halves
        offset += n + 1

    table = pd.concat(tables, ignore_index=True)
    # players missing in one half get all-NaN columns
    cols = [f"p{s}_{c}" for s in range(1, len(slot_of) + 1) for c in ("x", "y", "v")]
    table = table.reindex(columns=["frame", "half", "ball_x", "ball_y"] + cols)
    return table, pd.DataFrame(tracks), np.concatenate(labels)[:-1]


# ---------------------------------------------------------------- SkillCorner (A-League)
# 10 games, MIT licence, github.com/SkillCorner/opendata. Broadcast tracking at 10 fps, so it's
# naturally full of holes: we keep only what was really detected (is_detected), resample to 25 fps
# and DON'T add our own noise. Truth: phases of play (only defined while the ball is in play,
# with the team in possession) -> H/A inside a phase, D between phases. Used as the exam only.

SKILLCORNER = ROOT / "data" / "external" / "skillcorner-opendata" / "data" / "matches"


def skillcorner_games():
    return tuple(f"sc{p.name}" for p in sorted(SKILLCORNER.iterdir()) if p.is_dir())


def _skillcorner_game(match):
    folder = SKILLCORNER / match
    meta = json.loads((folder / f"{match}_match.json").read_text(encoding="utf-8"))
    L, W = meta["pitch_length"], meta["pitch_width"]
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: ("H" if p["team_id"] == home_id else "A") for p in meta["players"]}
    slot_of = {pid: i + 1 for i, pid in enumerate(team_of)}

    rows = []
    with open(folder / f"{match}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            if fr["period"] is None:
                continue
            b = fr["ball_data"]
            r = {"f10": fr["frame"], "half": fr["period"],
                 "ball_x": b["x"] if b["is_detected"] else None,
                 "ball_y": b["y"] if b["is_detected"] else None}
            for p in fr["player_data"]:
                s = slot_of.get(p["player_id"])
                if s is None:
                    continue
                r[f"p{s}_ex"], r[f"p{s}_ey"] = p["x"], p["y"]          # extrapolated
                if p["is_detected"]:
                    r[f"p{s}_x"], r[f"p{s}_y"] = p["x"], p["y"]
            rows.append(r)
    raw = pd.DataFrame(rows).set_index("f10").sort_index()

    # 10 fps -> 25 fps, linear between neighbours, never across a hole
    f10 = raw.index.to_numpy()
    t25 = np.arange(f10[0] * 2.5, f10[-1] * 2.5 + 1)
    pos = t25 / 2.5
    lo = np.searchsorted(f10, np.floor(pos), side="right") - 1
    lo = np.clip(lo, 0, len(f10) - 1)
    hi = np.clip(lo + 1, 0, len(f10) - 1)
    w = np.clip(pos - f10[lo], 0, 1)
    out = {"frame": np.arange(len(t25)), "half": raw["half"].to_numpy()[lo]}

    def interp(col):
        v = raw[col].to_numpy(dtype=float) if col in raw else np.full(len(raw), np.nan)
        return v[lo] * (1 - w) + v[hi] * w

    # SkillCorner coords are centred
    out["ball_x"] = interp("ball_x") + L / 2
    out["ball_y"] = interp("ball_y") + W / 2
    tracks = []
    for pid, s in slot_of.items():
        out[f"p{s}_x"] = interp(f"p{s}_x") + L / 2
        out[f"p{s}_y"] = interp(f"p{s}_y") + W / 2
        ex, ey = interp(f"p{s}_ex"), interp(f"p{s}_ey")
        # speed from the extrapolated (smooth) track, a tracker would also give a filtered one
        v = np.hypot(np.diff(ex, prepend=np.nan), np.diff(ey, prepend=np.nan)) * FPS
        out[f"p{s}_v"] = pd.Series(v).rolling(25, center=True, min_periods=5).mean().to_numpy()
        tracks.append({"slot": s, "team": team_of[pid]})
    table = pd.DataFrame(out)

    # truth from phases of play
    phases = pd.read_csv(folder / f"{match}_phases_of_play.csv")
    labels = np.full(len(table), X, dtype=np.int8)
    for h in (1, 2):
        rows_h = np.flatnonzero(table["half"].to_numpy() == h)
        if len(rows_h):
            labels[rows_h] = D
    for ph in phases.itertuples():
        a = int(round((ph.frame_start - f10[0]) * 2.5))
        b = int(round((ph.frame_end - f10[0]) * 2.5))
        labels[max(a, 0):max(b, 0)] = H if ph.team_in_possession_id == home_id else A
    return table, pd.DataFrame(tracks), labels, (float(L), float(W))


# ---------------------------------------------------------------- cache

TUNE_GAMES = GAMES + tuple(f"idsse_{m}" for m in IDSSE_GAMES)


def load(name):
    """(table, tracks, labels) for one game, from the cache."""
    table = pd.read_parquet(OUT / f"{name}_tracking.parquet")
    tracks = pd.read_parquet(OUT / f"{name}_tracks.parquet")
    labels = np.load(OUT / f"{name}_labels.npy")
    return table, tracks, labels


def _save(name, table, tracks, labels):
    table.to_parquet(OUT / f"{name}_tracking.parquet", index=False)
    tracks.to_parquet(OUT / f"{name}_tracks.parquet", index=False)
    np.save(OUT / f"{name}_labels.npy", labels)
    graded = labels[labels != X]
    dead = (graded == D).mean()
    seg = np.flatnonzero(np.diff(labels) != 0)
    n_poss = sum(1 for i in seg if labels[i + 1] in (H, A))
    poss_s = (graded != D).sum() / FPS / max(n_poss, 1)
    print(f"{name}: {len(graded) / FPS / 60:.0f} min graded, dead {dead:.0%}, "
          f"{n_poss} possessions of mean {poss_s:.1f} s, {len(tracks)} slots", flush=True)


def main(which=("metrica", "idsse", "skillcorner")):
    OUT.mkdir(parents=True, exist_ok=True)
    if "metrica" in which:
        loaders = {"metrica1": lambda: _csv_game(1), "metrica2": lambda: _csv_game(2),
                   "metrica3": _game3}
        for name in GAMES:
            table, tracks, events = loaders[name]()
            _save(name, table, tracks,
                  labels_from_events(events, table, int(table["frame"].max()) + 1))
    if "idsse" in which:
        for m in IDSSE_GAMES:
            _save(f"idsse_{m}", *_idsse_game(m))
    if "skillcorner" in which:
        for name in skillcorner_games():
            table, tracks, labels, pitch = _skillcorner_game(name[2:])
            _save(name, table, tracks, labels)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("metrica", "idsse", "skillcorner"))
