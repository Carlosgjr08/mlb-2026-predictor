# ⚾ MLB 2026 Run/Score Predictor

Machine-learning run and score predictions for the 2026 MLB season.
Two XGBoost Poisson models predict each team's runs; those turn into win
probabilities, likely scorelines, playoff-odds simulations, and a
tracking scoreboard that grades every pick against reality.

## 🚪 Getting back in (copy-paste this every time)

Open Terminal, then paste these two lines:

```bash
cd ~/Downloads/mlb-2026-predictor-main
source .venv/bin/activate
```

You're in when `(.venv)` appears at the start of your prompt. Then the
everyday commands:

```bash
python -m mlb_predictor predict --upcoming        # predict coming games
python -m mlb_predictor predict --home "Philadelphia Phillies" --away "New York Mets"
python -m mlb_predictor track record              # lock in picks
python -m mlb_predictor track result --home "Philadelphia Phillies" --away "New York Mets" --score 1-4
python -m mlb_predictor track board               # refresh scoreboard
python -m mlb_predictor fetch                     # update data (daily/weekly)
python -m mlb_predictor fetch --include-live      # ...including games already started
python -m mlb_predictor train                     # retrain after fetch
```

Done for the day? Type `deactivate` (or just close the Terminal window).

> If the folder path above doesn't match where the project lives on your
> machine, type `cd ` (with a space) and drag the project folder from
> Finder into the Terminal window, then press Enter.

## How it works

- **Runs, not goals** — two XGBoost regressors (`objective="count:poisson"`)
  predict expected home runs and away runs (MLB averages ~4.5/side).
- **No ties** — win/loss only; equal-run games are resolved as extra
  innings, with the usual small home-field edge.
- **25 features** — rolling team form (last 20 games, no leakage) **plus
  the starting-pitcher matchup and ballpark run factor**, the two biggest
  baseball-specific drivers:

```python
FEATURE_COLS = [
    "home_runs_for_avg", "home_runs_against_avg", "home_batting_avg",
    "home_on_base_pct", "home_slugging_pct", "home_era", "home_whip",
    "home_bullpen_era", "home_win_pct", "home_league",
    "away_runs_for_avg", "away_runs_against_avg", "away_batting_avg",
    "away_on_base_pct", "away_slugging_pct", "away_era", "away_whip",
    "away_bullpen_era", "away_win_pct", "away_league",
    # Starting-pitcher matchup (xFIP, K-BB%) + ballpark run factor
    "home_starter_xfip", "home_starter_k_bb_pct",
    "away_starter_xfip", "away_starter_k_bb_pct", "park_factor",
]
```

On the sample data these three additions are among the model's most
important features and lift winner accuracy by ~1–2 points.

## Quick start

```bash
pip install -r requirements.txt

# Offline sample data, or real data from the free MLB Stats API:
python -m mlb_predictor sample
python -m mlb_predictor fetch --seasons 2023 2024 2025 2026

python -m mlb_predictor train
python -m mlb_predictor predict --home "Los Angeles Dodgers" --away "Colorado Rockies"
python -m mlb_predictor predict --upcoming
python -m mlb_predictor simulate --runs 5000
```

Example:

```
Colorado Rockies @ Los Angeles Dodgers
  expected runs: Los Angeles Dodgers 6.34 - 3.73 Colorado Rockies
  win prob: Los Angeles Dodgers 80% | Colorado Rockies 20%  ->  pick Los Angeles Dodgers
  most likely scores: 6-3 (3%), 5-3 (3%), 6-4 (3%)
```

## Tracking predictions vs. reality

Locks in each game's pick and most likely score before first pitch,
grades it once you enter the final, and regenerates
[`MLB_RESULTS.md`](MLB_RESULTS.md) — a scoreboard that renders on GitHub.

```bash
python -m mlb_predictor track record          # log upcoming games
python -m mlb_predictor track result --home "Los Angeles Dodgers" \
                                     --away "Colorado Rockies" --score 7-2
python -m mlb_predictor track board            # refresh the scoreboard
```

## Project layout

```
mlb_predictor/
├── config.py       # teams, leagues/divisions, FEATURE_COLS, constants
├── fetch.py        # real data from the MLB Stats API -> data/mlb_games.csv
├── sample_data.py  # offline realistic sample dataset (same schema)
├── features.py     # rolling form + feature assembly
├── train.py        # XGBoost Poisson training + time-split evaluation
├── predict.py      # run means -> win probs, scorelines, matchup CLI
├── simulate.py     # Monte Carlo rest-of-season wins & playoff odds
├── track.py        # prediction tracking + MLB_RESULTS.md scoreboard
└── cli.py          # argparse entry point
```

## Data sources & the roadmap

- **Schedules + scores + probable pitchers:** the free public **MLB Stats
  API** (`statsapi.mlb.com`, no key), including `hydrate=probablePitcher`.
- **Starting-pitcher quality (xFIP, K-BB%):** FanGraphs via the optional
  **`pybaseball`** library (`pip install pybaseball`). Without it, starters
  fall back to league average and everything else still runs.
- **Ballpark run factors:** a static `PARK_FACTORS` table (Coors inflates,
  Petco/Oracle suppress, ...).
- **Still imputed** (next upgrades): per-game box-score rates
  (AVG/OBP/SLG/ERA/WHIP), real bullpen quality, rest/travel, and betting
  lines for calibration checks. A known simplification: pitcher stats are
  season-level, not as-of-game-date.

## Honest notes

- The sample dataset is **synthetic** (team strengths loosely calibrated
  to recent MLB form) so the pipeline runs without a network. Run `fetch`
  for real data.
- Runs are mildly over-dispersed, so Poisson is a good first
  approximation, not a perfect fit. Judge the model on **winner accuracy
  and log loss vs. the always-home baseline**, not exact scores.
