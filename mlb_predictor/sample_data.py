"""Offline sample dataset generator (2023–2026).

Realistic synthetic MLB games with the same schema fetch.py writes, so
the whole pipeline runs without network access. Team run-scoring and
run-prevention strengths are loosely calibrated to recent form; runs are
Poisson-sampled and the box-score rates (AVG/OBP/SLG/ERA/WHIP) are
derived from the same strengths so features carry real signal.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import MATCHES_CSV, PARK_FACTORS, TEAMS

# (offense, run-prevention) multipliers around 1.0. Lower prevention =
# better pitching/defense. Loose nods to recent MLB reality.
TEAM_RATINGS = {
    "Los Angeles Dodgers": (1.20, 0.85), "New York Yankees": (1.18, 0.90),
    "Atlanta Braves": (1.17, 0.92), "Philadelphia Phillies": (1.12, 0.90),
    "Houston Astros": (1.10, 0.90), "Baltimore Orioles": (1.10, 0.95),
    "San Diego Padres": (1.08, 0.92), "Cleveland Guardians": (1.00, 0.88),
    "Seattle Mariners": (0.96, 0.86), "New York Mets": (1.06, 0.96),
    "Milwaukee Brewers": (1.02, 0.92), "Texas Rangers": (1.05, 0.98),
    "Boston Red Sox": (1.05, 1.00), "Arizona Diamondbacks": (1.06, 0.99),
    "Tampa Bay Rays": (0.98, 0.94), "Minnesota Twins": (1.00, 0.96),
    "Detroit Tigers": (0.97, 0.93), "Kansas City Royals": (0.98, 0.97),
    "San Francisco Giants": (0.97, 0.97), "Chicago Cubs": (1.02, 0.98),
    "St. Louis Cardinals": (0.99, 1.00), "Cincinnati Reds": (1.00, 1.02),
    "Toronto Blue Jays": (1.00, 0.99), "Pittsburgh Pirates": (0.93, 1.00),
    "Los Angeles Angels": (0.95, 1.06), "Washington Nationals": (0.94, 1.05),
    "Miami Marlins": (0.90, 1.04), "Athletics": (0.92, 1.08),
    "Chicago White Sox": (0.85, 1.12), "Colorado Rockies": (0.98, 1.20),
}

BASE_RUNS = 4.45       # league-average runs per team per game
HOME_ADVANTAGE = 1.04  # small, as in real MLB (~54% home win)
SEASONS = (2023, 2024, 2025, 2026)
CUTOFF_2026 = date(2026, 7, 15)   # games on/before this are "played"


def _season_schedule(season: int, rng: np.random.Generator) -> list[tuple[date, str, str]]:
    """~120 games per team spread across a late-March -> late-Sept calendar
    (a scaled-down 162-game season, enough for solid rolling form)."""
    teams = list(TEAMS)
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            # four meetings, alternating home team
            pairs += [(a, b), (b, a), (a, b), (b, a)]

    start = date(season, 3, 27)
    end = date(season, 9, 28)
    span = (end - start).days
    offsets = rng.integers(0, span + 1, size=len(pairs))
    sched = [(start + timedelta(days=int(o)), h, a) for o, (h, a) in zip(offsets, pairs)]
    sched.sort(key=lambda x: x[0])
    return sched


def _build_rotations(rng: np.random.Generator) -> dict[str, list[tuple[float, float]]]:
    """Five starters per team, each with (xFIP, K-BB%). A team's rotation
    is centered on its run-prevention rating, with ace-to-backend spread."""
    spread = np.array([-0.65, -0.25, 0.10, 0.45, 0.85])
    rotations = {}
    for team, (_atk, dfn) in TEAM_RATINGS.items():
        team_xfip = 4.20 * dfn
        starters = []
        for off in spread:
            xfip = float(np.clip(team_xfip + off + rng.normal(0, 0.15), 2.6, 5.8))
            kbb = float(np.clip(0.13 + (4.20 - xfip) * 0.06 + rng.normal(0, 0.02), 0.02, 0.32))
            starters.append((round(xfip, 2), round(kbb, 3)))
        rotations[team] = starters
    return rotations


def _simulate_game(home: str, away: str, park: float,
                   h_starter: tuple[float, float], a_starter: tuple[float, float],
                   rng: np.random.Generator) -> dict:
    atk_h, def_h = TEAM_RATINGS[home]
    atk_a, def_a = TEAM_RATINGS[away]
    h_xfip, _ = h_starter
    a_xfip, _ = a_starter

    # Run prevention each side faces this game = 60% starter, 40% team
    # defense/bullpen; park scales the whole run environment.
    mu_home = BASE_RUNS * atk_h * (0.6 * a_xfip / 4.20 + 0.4 * def_a) * HOME_ADVANTAGE * park
    mu_away = BASE_RUNS * atk_a * (0.6 * h_xfip / 4.20 + 0.4 * def_h) * park
    runs_home = int(rng.poisson(mu_home))
    runs_away = int(rng.poisson(mu_away))
    # No ties in baseball: extra innings, slight home edge.
    if runs_home == runs_away:
        if rng.random() < 0.54:
            runs_home += 1
        else:
            runs_away += 1

    def batting(mu):  # offense strength -> rate stats
        avg = float(np.clip(0.245 + 0.05 * (mu / BASE_RUNS - 1) + rng.normal(0, 0.02), 0.19, 0.31))
        obp = float(np.clip(avg + 0.068 + rng.normal(0, 0.015), 0.26, 0.40))
        slg = float(np.clip(0.400 + 0.18 * (mu / BASE_RUNS - 1) + rng.normal(0, 0.03), 0.30, 0.58))
        return round(avg, 3), round(obp, 3), round(slg, 3)

    def pitching(runs_allowed):  # runs allowed -> ERA/WHIP proxies
        era = float(np.clip(runs_allowed * 0.92 + rng.normal(0, 0.8), 0.5, 12.0))
        whip = float(np.clip(1.05 + 0.09 * runs_allowed + rng.normal(0, 0.12), 0.6, 2.4))
        bullpen = float(np.clip(era + rng.normal(0, 1.0), 0.5, 12.0))
        return round(era, 2), round(whip, 2), round(bullpen, 2)

    h_avg, h_obp, h_slg = batting(mu_home)
    a_avg, a_obp, a_slg = batting(mu_away)
    h_era, h_whip, h_pen = pitching(runs_away)
    a_era, a_whip, a_pen = pitching(runs_home)

    return {
        "home_runs": runs_home, "away_runs": runs_away,
        "home_batting_avg": h_avg, "away_batting_avg": a_avg,
        "home_on_base_pct": h_obp, "away_on_base_pct": a_obp,
        "home_slugging_pct": h_slg, "away_slugging_pct": a_slg,
        "home_era": h_era, "away_era": a_era,
        "home_whip": h_whip, "away_whip": a_whip,
        "home_bullpen_era": h_pen, "away_bullpen_era": a_pen,
    }


def generate(seed: int = 26) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in SEASONS:
        rotations = _build_rotations(rng)  # rosters turn over year to year
        for game_date, home, away in _season_schedule(season, rng):
            played = season < 2026 or game_date <= CUTOFF_2026
            park = PARK_FACTORS[home]
            h_starter = rotations[home][rng.integers(5)]
            a_starter = rotations[away][rng.integers(5)]
            # Starter matchup + park are known before first pitch, so they
            # ride on every game (played or scheduled).
            row = {"date": game_date.isoformat(), "season": season,
                   "home_team": home, "away_team": away,
                   "status": "played" if played else "scheduled",
                   "home_starter_xfip": h_starter[0], "home_starter_k_bb_pct": h_starter[1],
                   "away_starter_xfip": a_starter[0], "away_starter_k_bb_pct": a_starter[1],
                   "park_factor": park}
            if played:
                row.update(_simulate_game(home, away, park, h_starter, a_starter, rng))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "home_team"]).reset_index(drop=True)


def main(seed: int = 26) -> None:
    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = generate(seed)
    df.to_csv(MATCHES_CSV, index=False)
    played = (df["status"] == "played").sum()
    print(f"Wrote {len(df)} games ({played} played, {len(df) - played} scheduled) "
          f"-> {MATCHES_CSV}")
