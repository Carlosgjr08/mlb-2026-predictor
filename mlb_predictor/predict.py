"""Scoreline + win-probability predictions from the trained run means."""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import EXTRA_INNINGS_HOME, FEATURE_COLS, MAX_RUNS
from .features import upcoming_frame


def score_grid(mu_home: float, mu_away: float) -> np.ndarray:
    home = poisson.pmf(np.arange(MAX_RUNS + 1), mu_home)
    away = poisson.pmf(np.arange(MAX_RUNS + 1), mu_away)
    return np.outer(home, away)


def win_probabilities(mu_home: float, mu_away: float) -> np.ndarray:
    """[P(home win), P(away win)]. Tie (equal-run) mass is split as extra
    innings, with a slight home edge."""
    grid = score_grid(mu_home, mu_away)
    p_home = np.tril(grid, -1).sum()
    p_away = np.triu(grid, 1).sum()
    p_tie = np.trace(grid)
    p_home += p_tie * EXTRA_INNINGS_HOME
    p_away += p_tie * (1 - EXTRA_INNINGS_HOME)
    total = p_home + p_away
    return np.array([p_home, p_away]) / total


def most_likely_scores(mu_home: float, mu_away: float, top: int = 3) -> list[tuple[str, float]]:
    grid = score_grid(mu_home, mu_away)
    flat = [(f"{i}-{j}", grid[i, j])
            for i in range(MAX_RUNS + 1) for j in range(MAX_RUNS + 1) if i != j]
    return sorted(flat, key=lambda x: -x[1])[:top]


def best_scoreline_for_winner(mu_home: float, mu_away: float, home_wins: bool) -> str:
    grid = score_grid(mu_home, mu_away)
    best, best_p = (0, 0), -1.0
    for i in range(MAX_RUNS + 1):
        for j in range(MAX_RUNS + 1):
            if i == j:
                continue
            if (i > j) == home_wins and grid[i, j] > best_p:
                best, best_p = (i, j), grid[i, j]
    return f"{best[0]}-{best[1]}"


def predict_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    df = df.copy()
    df["pred_home_runs"] = bundle["home_model"].predict(X)
    df["pred_away_runs"] = bundle["away_model"].predict(X)
    probs = np.array([win_probabilities(h, a)
                      for h, a in zip(df["pred_home_runs"], df["pred_away_runs"])])
    df[["p_home_win", "p_away_win"]] = probs
    return df


def _format_game(row) -> str:
    scores = most_likely_scores(row.pred_home_runs, row.pred_away_runs)
    score_str = ", ".join(f"{s} ({p:.0%})" for s, p in scores)
    winner = row.home_team if row.p_home_win >= row.p_away_win else row.away_team
    return (f"{row.date.date()}  {row.away_team} @ {row.home_team}\n"
            f"  expected runs: {row.home_team} {row.pred_home_runs:.2f} - "
            f"{row.pred_away_runs:.2f} {row.away_team}\n"
            f"  win prob: {row.home_team} {row.p_home_win:.0%} | "
            f"{row.away_team} {row.p_away_win:.0%}  ->  pick {winner}\n"
            f"  most likely scores: {score_str}")


def predict_matchup(bundle: dict, home: str, away: str) -> None:
    upcoming = upcoming_frame()
    row = upcoming[(upcoming["home_team"] == home) & (upcoming["away_team"] == away)]
    if row.empty:
        h = upcoming[upcoming["home_team"] == home].head(1)
        a = upcoming[upcoming["away_team"] == away].head(1)
        if h.empty or a.empty:
            known = sorted(set(upcoming["home_team"]) | set(upcoming["away_team"]))
            raise SystemExit(f"No feature data for '{home}' or '{away}'. "
                             f"Known teams: {', '.join(known)}")
        row = h.copy()
        for col in [c for c in FEATURE_COLS if c.startswith("away_")]:
            row[col] = a.iloc[0][col]
        row["away_team"] = away
    print(_format_game(predict_frame(bundle, row).iloc[0]))


def predict_upcoming(bundle: dict, limit: int | None = None,
                     date: str | None = None) -> pd.DataFrame:
    df = predict_frame(bundle, upcoming_frame())
    if date:
        df = df[df["date"].dt.strftime("%Y-%m-%d") == date]
        if df.empty:
            print(f"No scheduled games on {date}.")
    for row in (df.head(limit) if limit else df).itertuples():
        print(_format_game(row) + "\n")
    return df
