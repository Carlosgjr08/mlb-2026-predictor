# ⚾ MLB 2026 Run/Score Predictor

Machine-learning run and score predictions for the 2026 MLB season.
Two XGBoost Poisson models predict each team's runs; those turn into win
probabilities, likely scorelines, playoff-odds simulations, and a
tracking scoreboard that grades every pick against reality.

## How it works

- **Runs, not goals** — two XGBoost regressors (`objective="count:poisson"`)
  predict expected home runs and away runs (MLB averages ~4.5/side).
- **No ties** — win/loss only; equal-run games are resolved as extra
  innings, with the usual small home-field edge.
- **20 features** — rolling form over each team's last 20 games, computed
  only from games *before* the one being predicted (no leakage):

```python
FEATURE_COLS = [
    "home_runs_for_avg", "home_runs_against_avg", "home_batting_avg",
    "home_on_base_pct", "home_slugging_pct", "home_era", "home_whip",
    "home_bullpen_era", "home_win_pct", "home_league",
    "away_runs_for_avg", "away_runs_against_avg", "away_batting_avg",
    "away_on_base_pct", "away_slugging_pct", "away_era", "away_whip",
    "away_bullpen_era", "away_win_pct", "away_league",
]
```

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

- **Now:** the free public **MLB Stats API** (`statsapi.mlb.com`, no key)
  for schedules and final scores. Box-score rates (AVG/OBP/SLG/ERA/WHIP)
  are imputed with league averages — rolling runs for/against, win %, and
  home field carry the signal.
- **Biggest planned upgrade — starting pitchers.** The starter is the
  most predictive single input in baseball. Plan: probable pitchers from
  the MLB Stats API + pitcher quality (FIP / xFIP / SIERA / K-BB%) from
  Baseball Savant / FanGraphs (via the `pybaseball` library).
- **Then:** ballpark run factors (Coors!), real bullpen quality, rest/
  travel, and betting lines for calibration checks.

## Honest notes

- The sample dataset is **synthetic** (team strengths loosely calibrated
  to recent MLB form) so the pipeline runs without a network. Run `fetch`
  for real data.
- Runs are mildly over-dispersed, so Poisson is a good first
  approximation, not a perfect fit. Judge the model on **winner accuracy
  and log loss vs. the always-home baseline**, not exact scores.
