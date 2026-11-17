# Leszek Pisz: technical document

More Than A Hack 2027, possession for BAR-ATH.
Code: https://github.com/Leszek-Pisz/More-Than-A-Hack-2027 (all paths below are relative to it). Submission file: `Leszek Pisz.csv`.

## 1. What we did

Every frame gets H, A or D from a 3-state HMM (one state per label), decoded with Viterbi, each
half on its own. For each frame we score how well H, A and D fit what we see in the tactical
tracking. We use a handful of simple clues:

- Tempo. Average player speed over about 4.4 s. When play stops, players walk.
- Ball state. Is the ball tracked on the pitch, tracked out of play, or not tracked at all?
  Before that we drop ball positions that would need more than 40 m/s to get to (glitches),
  and patch holes of up to 1 s with a straight line (`src/ball.py`, more in section 5).
- Nearest player. If someone is within about 1.6 m of the ball, that's a soft vote for their team.
- Stoppages and control (`src/events.py`). The ball standing still, or standing still on a corner
  / touchline / goal area / centre spot, the ball disappearing right at a line, and "control":
  a player stays by the ball for 0.4 s or more with no opponent close.
- Who restarts (`src/events.py`). When a ball that stood still starts moving, the team whose
  player was right next to it, with no opponent within 3 m, gets a small nudge for 3 s. That's
  usually the team taking the throw-in, corner or free kick.
- Team shape (`src/shape_model.py`). A small logistic model that guesses who has the ball only
  from how the two teams are standing: the team with the ball pushes up and spreads out, the
  other one drops and gets compact. We also look at pressing: the team without the ball marks,
  so its players stand closer to their nearest opponent. It doesn't look at the ball, so it's
  most useful when the ball isn't tracked.

The transitions are sticky: per frame it's very unlikely to leave a possession or a stoppage, so a
single frame where an opponent happens to be closer doesn't flip the label. That's our
"hysteresis".

Two clean-ups after decoding, in this order:

1. Drop fake blips (`src/post.py`): a stretch shorter than 1 s with the same label on both sides
   takes that label (H-A-H becomes H, H-D-H becomes H). It never moves a boundary between two
   different labels, so it can't make a real change late, it only removes fake switches, each of
   which would cost two possessions. The 1 s comes from football (control needs a couple of
   touches, a dribble or a pass; a real stoppage lasts seconds), fixed before the exam.
2. Move switches to the ball: a sticky HMM tends to switch H to A a little late, once enough
   evidence has piled up. So each H/A switch is moved, at most 1 s either way, to the start of
   the closest stretch where a player of the new team is at the ball (within 1.6 m) and closer
   than anyone from the other team. Switches into or out of D aren't touched.

All numbers are in `configs/params.toml`, and all of them were tuned on other matches.

Why this and not something fancier:

The metric really hates flicker. A possession is a maximal run of H or A and any D cuts it, so one
wrong switch or a fake D breaks two possessions. Even a perfect prediction shifted by 1 s only
gets about 0.65 spellF1. We wanted long, clean segments with sharp edges, which is what Viterbi on
a sticky HMM gives you. Smoothing filters like the organisers' smooth50/125 move the edges.

We also can't train anything on this match, no labels, no tuning on its GT, no self-supervision.
With ~25 numbers that each mean something, a few external games are enough to tune honestly. A
deep sequence model would need way more labelled tracking than our 10 games, and a lot more time
to build and train than we had left after the main pipeline.

And it's deterministic: no GPU, nothing random at prediction time, so the repo rebuilds the same
file byte for byte.

## 2. What we used and what we dropped

From the match itself we only use the tactical tracking. The broadcast tracking gets converted
but isn't used, and neither are the videos. We thought about filling ball gaps from the broadcast
tracking and from a pretrained YOLO ball detector on the video, but none of our external games
has a second camera or video, so there's no honest way to check whether it helps, and we only
kept things we could measure. On top of that it's a lot of work: lining up the broadcast clock
with the tactical one (a different offset per camera shot), hours of GPU for the detector, and
GPU inference isn't deterministic, so the detections would have to be saved once as an
intermediate file. We put that time into the parts we could validate instead.
Smart Tagging was tried and is switched off (more below).

For tuning and training we used Metrica sample data and IDSSE (Bundesliga), and SkillCorner open
data only as a final exam. We also looked at PFF FC World Cup 2022, but didn't get to it: it's big, and the three sets
above already cover event-based truth, per-frame tracking truth and real broadcast holes.

Things we built, checked on the external games, and dropped:

- Voting from Smart Tagging (an idea from our team): take the team in each tag as the team in
  possession. The tag windows have fixed, made-up lengths (30-35 s), and the only way to tune
  this would be on our match's GT, which isn't allowed. What's left is `src/tags_prior.py`, a
  very weak nudge (attacking tag of a team pushes a bit towards that team, defending tag towards
  the other one, weight 0.05 per frame picked up front). It's off in the submission because none
  of the external games have Smart Tagging, so we had no way to check it.
- A semi-Markov decoder (`src/decode.py`) with log-normal length priors for possessions and
  stoppages fitted on Metrica, working on 0.2 s blocks and then refining the edges. After tuning
  it lost to the plain HMM, 0.528 vs 0.546 on Metrica. Our guess is that possession lengths are
  all over the place (0.8 to 25 s) so a length prior doesn't tell you much, and the blocks blur
  the edges. It's still there as a switch (`[decoder] type`).
- Using team shape only where the ball is missing. Clearly worse than using it on every frame.
- "Players in the opponent's half" as a shape feature. A bit better on Metrica, but worse on
  IDSSE and clearly worse on the exam (17 of 20 games), so it's out. We also thought about the
  distance between the two teams' centres, but it's the same number for both teams, so home
  minus away is always zero and it can't tell them apart.
- Two more shape features, compactness (mean distance to the team's centre) and how much the
  players' speeds differ. Both helped a bit on Metrica but made the exam worse.
- Snapping switches to real touches (the ball's velocity changing by more than 4 m/s within
  ~0.25 s) instead of just "new team at the ball". Better on both leagues, a bit worse on the
  exam (0.516 to 0.515), so it's out.
- Guessing where the ball is during long gaps (where the players crowd) and giving a weak vote
  to the team closest to that spot. Worse everywhere.
- Smoothing the ball track between kicks, as a light stand-in for a Kalman filter. Better on the
  exam, worse on Metrica, so it's out.
- Tuning only the new numbers (gap filling, restart clue), keeping the rest. Same story as below:
  better on its own noise, very slightly worse on fresh noise and the exam.
- Retuning everything after adding the gap filling, the restart clue and pressing. It got a
  tiny bit better on its own noise, but worse on fresh noise for IDSSE (0.400 to 0.365) and on
  the exam (0.515 to 0.506). Looks like overfitting, so we kept the old numbers and left the
  new ones as fixed guesses (1 s holes, 30 m/s, 3 m, 3 s, 0.7).
- In the Metrica labels, ending a stoppage only at a "SET PIECE" event. Metrica logs goal kicks
  as normal passes, so we got fake 80 s stoppages. Now a stoppage ends at the next on-ball event.
- Tuning on Metrica only. Fine on Metrica, but on IDSSE it ended up worse than our hand-picked
  starting numbers (0.269 vs 0.333). So the final tuning uses both leagues, weighted equally.
- Not built: broadcast fusion and YOLO (can't be measured, see above), stitching player tracks
  into 22 identities (everything we do is per team, so it wouldn't change much), a learned
  sequence model (not enough labelled games, and not enough time to do it properly).

## 3. Training and tuning data

Nothing from BAR-ATH was used to train, tune or calibrate anything. Not the GT, not the video,
not even stats about how good its tracking is. We ran the public 5 minutes twice just to see the
pipeline works (0.40 with the hand-picked numbers, 0.378 after tuning) and didn't change anything
because of it.

Metrica sample data (github.com/metrica-sports/sample-data, free to use if you credit Metrica
Sports). 3 anonymised games with tracking and events. The events come from Metrica, we turn
them into H/A/D in code: possession is the team of the last on-ball event (pass, carry, set
piece, recovery, shot, ball lost), challenges don't change it. The ball is dead from an out, a
shot that goes out or in, a foul or an offside, until the next on-ball event. Used for tuning
and the shape model.

IDSSE (Bassek et al. 2025, Scientific Data, figshare 10.6084/m9.figshare.28196177, CC-BY 4.0).
7 Bundesliga games. The DFL tracking (Sportec Solutions) has `BallPossession` and `BallStatus`
on every frame, which is basically the same kind of truth the organisers built their GT from,
so we just read it. Used for tuning and the shape model.

SkillCorner open data (github.com/SkillCorner/opendata, MIT). 20 A-League games, broadcast
tracking at 10 fps. Labels come from SkillCorner's automatic "phases of play": H/A inside a
phase, D between phases. Used only as an exam: no number was ever fitted on it, but we did use it (together with the
two leagues) to decide whether to keep a change, so it's not a perfectly clean test any more. We keep only positions that
were really detected, resample to 25 fps and add no noise, since the holes there are real.

Metrica and IDSSE tracking is clean, so we mess it up ourselves (`external.add_noise`). The ball
goes missing in 15-50% of frames, in long gaps (median 3-20 s), more often during stoppages and
near the goals. There are 0.5-3 fake one-frame ball jumps per minute, smooth drift on positions
(ball 0.05-0.5 m, players 0.05-0.4 m), player tracks missing 2-15% of the time in 1-10 s gaps,
and 5-25% error on player speed. Each run draws these from the ranges above, which we fixed up
front. None of it is fitted to our match.

Tuning (`src/tune.py`) is a simple local random search, 150 tries, each one changing ~30% of the
numbers from the best set so far. The score is mean spellF1 per league (Metrica, IDSSE), both
leagues count the same, on noise seed 0. Then we check on fresh noise (seed 10) and on the
SkillCorner exam. For team shape, each game is scored with a shape model trained on the other
league, so the search can't trust it more than it would on a new match.

The shape model is a logistic regression without an intercept, so swapping the teams flips the
answer exactly and it can't learn "home has the ball more". Per team, in its own attacking
direction: centroid, most advanced player, back line, length, width, speed towards goal, and pressing
(mean distance from each player to the closest opponent). It gets home minus away, smoothed over 2 s. Attacking direction per half comes from the keeper
(the player whose average x is closest to an end line), no ball or labels needed. The final
model is trained on all 10 Metrica + IDSSE games with noise (`models/shape_model.json`).

Results (spellF1; Metrica and IDSSE on fresh noise):

| Version | Metrica | IDSSE | SkillCorner exam |
|---|---|---|---|
| organisers' smooth50 | 0.330 | | |
| hand-picked numbers | 0.420* | 0.333* | 0.387 |
| tuned on both leagues | 0.447 | 0.336 | 0.453 |
| plus team shape | 0.487 | 0.378 | 0.485 |
| everything retuned | 0.496 | 0.396 | 0.511 |
| plus gap filling, restart clue, pressing | 0.501 | 0.400 | 0.515 |
| plus moving switches to the ball | 0.503 | 0.402 | 0.516 |
| plus dropping blips under 1 s (submitted) | 0.506† | not run | 0.537 |

\* these two come from the league-to-league runs.

Each of the three last additions on its own was no worse on any league, except the restart
clue, which lost a hair on Metrica (0.4965 to 0.4960) but gained on the exam (0.511 to 0.514).
Together they're better on all three. Moving the switches was checked with windows of 0.5, 1
and 2 s; we picked 1 s because it was best on Metrica + IDSSE, and the exam agreed. All of this
was checked with `src/compare.py`.

† The blip filter was checked without IDSSE (not downloaded on the machine that ran it), so the
shape model there was the saved one, which saw Metrica in training: Metrica numbers in that run
are a bit higher across the board and only comparable within the run. Same run, Metrica / exam:
switches moved, no filter 0.502 / 0.516; plus filter 0.506 (2/3 games better) / 0.537 (20/20);
filter without moving switches 0.507 / 0.537; neither 0.499 / 0.515. With the filter on, moving
the switches barely matters (Metrica -0.001, exam +0.0002); we kept it because it was picked on
IDSSE, which this run couldn't check.

Team shape made every single game better: 3/3 Metrica, 7/7 IDSSE, 20/20 SkillCorner. smooth50
gets 0.2433 on the real match and 0.33 on our noisy Metrica, so our noise is probably nicer than
real life.

## 4. Possession rules

The organisers' GT is mechanical: D where the tracking says the ball is dead, otherwise the team
the tracking says owns the ball, no smoothing. We try to follow the same logic.

Loose or contested ball: it stays with the last team. A missing ball, or a ball with nobody near
it, doesn't vote for H or A, so the sticky transitions keep whoever had it. Possession changes
only when someone gets control: a player near the ball, and even more so a player who stays with
it for 0.4 s with no opponent close. Just running past the ball doesn't count.

Duels: when players from both teams are near the ball, the nearest-player vote is weak (tuned to
about 0.69) and the control clue doesn't fire because an opponent is close. So the last team keeps
it until someone clearly comes out with the ball.

Restarts: D starts when the ball goes out (0.5 m past a line, or disappearing right at a line) or
when play stops and players slow down to walking. D ends when play picks up again and the ball
leaves its spot. The team taking the restart gets possession from then on, through the
nearest-player and control clues, and the restart clue (whoever stood by the still ball while
the opponents kept their distance). Throw-ins, corners, goal kicks and free kicks all work like
this, and a ball standing still on a set-piece spot counts as extra evidence for D.

Frames outside the two halves are D (they aren't graded anyway).

## 5. Missing and noisy tracking

Ball positions that would need more than 40 m/s are thrown out, the hardest shots are around 35.

Short holes in the ball, up to 1 s, get a straight line between the last and next position, but
only if the ball wouldn't need more than 30 m/s for that. Mid-pass the ball is still in play and
roughly on that line. Longer holes stay empty, after a second anything could have happened.

When the ball is missing, that frame gives no "who's near" vote and only a weak hint about alive
vs dead (tuned: the ball is missing in 50% of alive frames and 42% of dead ones). The HMM just
carries the last state through the gap. Team shape doesn't need the ball, so long gaps still get
some H vs A signal from it.

Tempo is the mean over whoever is tracked, over 4.4 s, so missing or doubled player tracks barely
move it. Frames with nobody tracked take the value from their neighbours.

Everything was tuned with heavy random noise on top of the external games and checked on real
broadcast holes (SkillCorner), never on clean data.

## 6. Limitations

The datasets don't agree on what a possession is. IDSSE has ~440 short possessions per game
(per frame, from tracking), Metrica ~370 (from events). Tuning mostly learns "how many
possessions", and tuning on one league made the other one worse. We took the version that
doesn't fall apart on either, but we don't know which definition is closer to the organisers'.

We predict fewer possessions than the truth on Metrica and IDSSE, and more on SkillCorner.

The 1 s blip filter was checked on Metrica and SkillCorner, not on IDSSE (no time to download it).
IDSSE's truth is built per frame from tracking, like the organisers' GT, and has more short
possessions, so if the organisers' GT has many real possessions under 1 s the filter removes them.

Our noise is probably too gentle (see smooth50 above).

The tuning ended up with the ball missing in about half of alive frames. If our match loses the
ball in play a lot more than that, D detection will suffer.

We only use the tactical tracking, broadcast and video are ignored. H/A switches are moved to
when the new team reaches the ball, but switches into and out of D come straight from the HMM,
so those can be a few frames off.

Next steps: fill ball gaps with the broadcast tracking and a pretrained YOLO (no
fine-tuning on our video), stitch
players into 22 identities to follow passes, and add more labelled games (PFF World Cup 2022) to
train a proper sequence model. For broadcast and YOLO the first job would be finding external
games with both a second camera and possession labels, so they can be checked the same way.

## 7. How to reproduce

```bash
uv sync                          # exact versions from uv.lock
# match files go in data/tactical/ and data/broadcast/ (see README.md in the repo)
uv run python src/predict.py     # writes evaluation/results/Leszek Pisz.csv, prints its sha256
uv run python evaluation/validate_submission.py "evaluation/results/Leszek Pisz.csv"
```

We ran it twice from the raw files and got the same file both times, sha256
`51b7fd02bc60a0d71b9c176a644b0ba0130367e647777766e0c4a797f899a9d0`.

You don't need any external data for this. The tuned numbers are in `configs/params.toml` and the
shape model in `models/shape_model.json`. To redo the tuning (external data in
`data/external/`, see README.md in the repo):

```bash
uv run python src/external.py
uv run python src/shape_model.py --eval
uv run python src/shape_model.py --train
uv run python src/tune.py --trials 150
uv run python src/tune.py --exam
```

Seeds are fixed everywhere (noise 0 / 10 / 20, search 2027).
