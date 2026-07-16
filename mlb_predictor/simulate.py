"""Monte Carlo the rest of the 2026 MLB season.

Current wins from played 2026 games + model win probabilities for every
remaining game, sampled `runs` times, to project wins and playoff odds
(top 6 per league: 3 division winners + 3 wild cards, approximated here
as the top 6 finishers by wins in each league)."""

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import DIVISION, LEAGUE
from .features import load_matches, upcoming_frame
from .predict import predict_frame

PLAYOFF_SPOTS = 6  # per league


def current_wins(season: int = 2026) -> dict[str, int]:
    df = load_matches()
    played = df[(df["season"] == season) & (df["status"] == "played")]
    wins: dict[str, int] = defaultdict(int)
    for row in played.itertuples():
        winner = row.home_team if row.home_runs > row.away_runs else row.away_team
        wins[winner] += 1
    return dict(wins)


def simulate_season(bundle: dict, runs: int = 5000, seed: int = 26) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_wins = current_wins()
    remaining = predict_frame(bundle, upcoming_frame())
    remaining = remaining[remaining["season"] == 2026]

    teams = sorted(set(base_wins) |
                   set(remaining["home_team"]) | set(remaining["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    al_mask = np.array([LEAGUE[t] == "AL" for t in teams])

    p_home = remaining["p_home_win"].to_numpy()
    hi = remaining["home_team"].map(idx).to_numpy()
    ai = remaining["away_team"].map(idx).to_numpy()
    start = np.zeros(len(teams))
    for t, w in base_wins.items():
        start[idx[t]] = w

    total_wins = np.zeros((runs, len(teams)))
    playoffs = np.zeros(len(teams))
    home_won = rng.random((runs, len(p_home))) < p_home
    for r in range(runs):
        w = start.copy()
        np.add.at(w, hi[home_won[r]], 1)
        np.add.at(w, ai[~home_won[r]], 1)
        total_wins[r] = w
        for mask in (al_mask, ~al_mask):
            lg_idx = np.flatnonzero(mask)
            playoffs[lg_idx[np.argsort(-w[lg_idx])[:PLAYOFF_SPOTS]]] += 1

    out = pd.DataFrame({
        "team": teams,
        "division": [DIVISION[t] for t in teams],
        "current_wins": start.astype(int),
        "proj_wins": total_wins.mean(axis=0).round(1),
        "playoff_odds": (playoffs / runs).round(3),
    })
    return out.sort_values(["division", "proj_wins"],
                           ascending=[True, False]).reset_index(drop=True)


def main(bundle: dict, runs: int = 5000) -> None:
    table = simulate_season(bundle, runs)
    for div in sorted(table["division"].unique()):
        sub = table[table["division"] == div]
        print(f"\n=== {div} (projected, {runs} sims) ===")
        print(f"{'Team':<24}{'W now':>7}{'Proj W':>9}{'Playoffs':>10}")
        for row in sub.itertuples():
            print(f"{row.team:<24}{row.current_wins:>7}{row.proj_wins:>9}"
                  f"{row.playoff_odds:>10.1%}")
