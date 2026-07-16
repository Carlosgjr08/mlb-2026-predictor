"""Turns raw game rows into the 20-feature layout the model trains on.

Every rolling feature is a mean over a team's last FORM_WINDOW played
games *before* the game being featurized (no leakage).
"""

import numpy as np
import pandas as pd

from .config import (FEATURE_COLS, FORM_WINDOW, LEAGUE, LEAGUE_AVG_K_BB_PCT,
                     LEAGUE_AVG_XFIP, LEAGUE_CODE, MATCHES_CSV, MIN_GAMES,
                     PARK_FACTORS, normalize_team)

# Per-team-per-game stats that feed the rolling averages.
_FORM_STATS = ["runs_for", "runs_against", "batting_avg", "on_base_pct",
               "slugging_pct", "era", "whip", "bullpen_era", "win"]

# League-average fallbacks so a stat a source leaves blank degrades to a
# neutral value instead of dropping the whole game.
_STAT_DEFAULTS = {
    "runs_for": 4.45, "runs_against": 4.45, "batting_avg": 0.245,
    "on_base_pct": 0.315, "slugging_pct": 0.410, "era": 4.20,
    "whip": 1.30, "bullpen_era": 4.10, "win": 0.5,
}


def load_matches() -> pd.DataFrame:
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(
            f"{MATCHES_CSV} not found — run `python -m mlb_predictor sample` "
            "(offline) or `python -m mlb_predictor fetch` (live API) first.")
    df = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(normalize_team)
    df = df.dropna(subset=["home_team", "away_team"])
    return df.sort_values(["date", "home_team"]).reset_index(drop=True)


def _team_appearances(played: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played game, with that team's stats."""
    def side(df, us, them):
        return pd.DataFrame({
            "date": df["date"],
            "team": df[f"{us}_team"],
            "runs_for": df[f"{us}_runs"],
            "runs_against": df[f"{them}_runs"],
            "batting_avg": df[f"{us}_batting_avg"],
            "on_base_pct": df[f"{us}_on_base_pct"],
            "slugging_pct": df[f"{us}_slugging_pct"],
            "era": df[f"{us}_era"],
            "whip": df[f"{us}_whip"],
            "bullpen_era": df[f"{us}_bullpen_era"],
            "win": (df[f"{us}_runs"] > df[f"{them}_runs"]).astype(float),
        })

    long = pd.concat([side(played, "home", "away"), side(played, "away", "home")])
    for stat, default in _STAT_DEFAULTS.items():
        long[stat] = long[stat].fillna(long[stat].mean()).fillna(default)
    return long.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def _team_states(played: pd.DataFrame) -> pd.DataFrame:
    """Rolling form snapshot per team after each played game."""
    long = _team_appearances(played)
    grouped = long.groupby("team", sort=False)
    states = long[["team", "date"]].copy()
    for stat in _FORM_STATS:
        states[stat] = grouped[stat].transform(
            lambda s: s.rolling(FORM_WINDOW, min_periods=MIN_GAMES).mean())
    states["games_played"] = grouped.cumcount() + 1
    return states.sort_values("date", kind="stable")


def _attach_side(games: pd.DataFrame, states: pd.DataFrame, side: str) -> pd.DataFrame:
    rename = {
        "runs_for": f"{side}_runs_for_avg", "runs_against": f"{side}_runs_against_avg",
        "batting_avg": f"{side}_batting_avg", "on_base_pct": f"{side}_on_base_pct",
        "slugging_pct": f"{side}_slugging_pct", "era": f"{side}_era",
        "whip": f"{side}_whip", "bullpen_era": f"{side}_bullpen_era",
        "win": f"{side}_win_pct", "games_played": f"{side}_games_played",
    }
    return pd.merge_asof(
        games.sort_values("date", kind="stable"),
        states.rename(columns={"team": f"{side}_team", **rename}),
        on="date", by=f"{side}_team", allow_exact_matches=False)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    played = df[df["status"] == "played"]
    states = _team_states(played)

    # Keep game identity + runs + the pre-game "point in time" features
    # (starter matchup, park). The rolling box-score columns are dropped
    # here — they're raw material for the averages and would collide with
    # the attached *_avg feature names.
    keep = ["date", "season", "home_team", "away_team", "status",
            "home_runs", "away_runs",
            "home_starter_xfip", "home_starter_k_bb_pct",
            "away_starter_xfip", "away_starter_k_bb_pct", "park_factor"]
    base = df[[c for c in keep if c in df.columns]].copy()

    # Starter / park are known before first pitch and attach directly.
    # Fill any missing starter line with league average; missing park with
    # the home team's known factor (falling back to neutral 1.0).
    starter_defaults = {
        "home_starter_xfip": LEAGUE_AVG_XFIP, "away_starter_xfip": LEAGUE_AVG_XFIP,
        "home_starter_k_bb_pct": LEAGUE_AVG_K_BB_PCT,
        "away_starter_k_bb_pct": LEAGUE_AVG_K_BB_PCT,
    }
    for col, default in starter_defaults.items():
        if col not in base.columns:
            base[col] = np.nan
        base[col] = base[col].fillna(default)
    if "park_factor" not in base.columns:
        base["park_factor"] = np.nan
    base["park_factor"] = base["park_factor"].fillna(base["home_team"].map(PARK_FACTORS)).fillna(1.0)

    out = _attach_side(base, states, "home")
    out = _attach_side(out, states, "away")
    out["home_league"] = out["home_team"].map(LEAGUE).map(LEAGUE_CODE)
    out["away_league"] = out["away_team"].map(LEAGUE).map(LEAGUE_CODE)

    ready = (out["home_games_played"].fillna(0) >= MIN_GAMES) & \
            (out["away_games_played"].fillna(0) >= MIN_GAMES) & \
            out[FEATURE_COLS].notna().all(axis=1)
    return out[ready].reset_index(drop=True)


def training_frame() -> pd.DataFrame:
    df = build_features(load_matches())
    return df[df["status"] == "played"].reset_index(drop=True)


def upcoming_frame() -> pd.DataFrame:
    df = build_features(load_matches())
    return df[df["status"] == "scheduled"].reset_index(drop=True)


def outcome_labels(df: pd.DataFrame) -> np.ndarray:
    """1 = home win, 0 = away win (no ties in baseball)."""
    return (df["home_runs"] > df["away_runs"]).astype(int).to_numpy()
