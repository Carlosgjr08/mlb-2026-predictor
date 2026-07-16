"""Real data fetcher: the free public MLB Stats API (statsapi.mlb.com).

The schedule endpoint gives final scores (runs) and upcoming fixtures
with no API key. Per-game box-score rates (AVG/OBP/SLG/ERA/WHIP) would
need an extra call per game, so they're imputed with league averages
here — the rolling runs for/against, win %, and home field carry the
signal, exactly as xG did in the MLS project. Add per-game box scores
later for a stronger model.

NOTE: written offline (the API was blocked in the build sandbox). Run
locally and sanity-check the first response.
"""

import sys
from datetime import date

import pandas as pd
import requests

from .config import MATCHES_CSV, normalize_team

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"

# League-average box-score rates used where the schedule endpoint gives
# no per-game batting/pitching detail.
IMPUTE = {
    "batting_avg": 0.245, "on_base_pct": 0.315, "slugging_pct": 0.410,
    "era": 4.20, "whip": 1.30, "bullpen_era": 4.10,
}


def _get(params: dict) -> dict:
    resp = requests.get(SCHEDULE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_season(season: int) -> pd.DataFrame:
    """All regular-season games for a season: finals with runs, plus any
    upcoming fixtures as 'scheduled'."""
    data = _get({
        "sportId": 1, "season": season, "gameType": "R",
        "startDate": f"{season}-03-01", "endDate": f"{season}-11-15",
    })
    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            state = g.get("status", {}).get("abstractGameState", "")
            final = state == "Final"
            if not final and state not in ("Preview", "Scheduled"):
                continue  # skip in-progress / suspended
            row = {
                "date": g["gameDate"][:10], "season": season,
                "home_team": home["team"]["name"], "away_team": away["team"]["name"],
                "status": "played" if final else "scheduled",
            }
            if final:
                row["home_runs"] = home.get("score")
                row["away_runs"] = away.get("score")
                if row["home_runs"] is None or row["away_runs"] is None:
                    continue
            rows.append(row)
    return pd.DataFrame(rows)


def main(seasons: list[int]) -> None:
    frames = []
    for s in seasons:
        print(f"Fetching {s} schedule from MLB Stats API...")
        try:
            frames.append(fetch_season(s))
        except requests.RequestException as exc:
            print(f"  warning: request failed for {s}: {exc}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True)

    for col in ("home_team", "away_team"):
        df[col] = df[col].map(normalize_team)
    df = df.dropna(subset=["home_team", "away_team"])

    # Box-score rates aren't in the schedule feed: impute league averages
    # for played games so features are always computable.
    played = df["status"] == "played"
    for side in ("home", "away"):
        for stat, value in IMPUTE.items():
            col = f"{side}_{stat}"
            if col not in df.columns:
                df[col] = None
            df.loc[played, col] = df.loc[played, col].fillna(value)

    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MATCHES_CSV, index=False)
    n_played = int(played.sum())
    print(f"Wrote {len(df)} games ({n_played} played, {len(df) - n_played} scheduled) "
          f"-> {MATCHES_CSV}")
