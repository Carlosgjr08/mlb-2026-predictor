"""Real data fetcher: the free public MLB Stats API (statsapi.mlb.com)
plus starting-pitcher stats from FanGraphs (via pybaseball).

- Schedule endpoint (no key): final scores, upcoming fixtures, and — with
  hydrate=probablePitcher — each game's probable starters.
- pybaseball (optional): season xFIP and K-BB% per pitcher, joined to the
  probable starters. Install with `pip install pybaseball`; without it,
  starters fall back to league average and the rest still works.
- Ballpark run factors come from the static PARK_FACTORS table.

Per-game box-score rates (AVG/OBP/SLG/ERA/WHIP) still aren't pulled and
are imputed with league averages — rolling runs for/against, win %, the
starter matchup, and park carry the signal.

NOTE: written offline (both APIs were blocked in the build sandbox). Run
locally and sanity-check the first response, especially the pybaseball
column names ('xFIP', 'K-BB%').
"""

import sys

import pandas as pd
import requests

from .config import (LEAGUE_AVG_K_BB_PCT, LEAGUE_AVG_XFIP, MATCHES_CSV,
                     PARK_FACTORS, normalize_team)

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


def _starter_stats(seasons: list[int]) -> dict[str, tuple[float, float]]:
    """Map pitcher full name -> (xFIP, K-BB% as a fraction) using FanGraphs
    season lines via pybaseball. Returns {} if pybaseball isn't installed
    or the pull fails, in which case starters fall back to league average.
    Season-level (not as-of-date) stats are a deliberate simplification."""
    try:
        from pybaseball import pitching_stats
    except ImportError:
        print("  note: pybaseball not installed — starters imputed to league "
              "average. `pip install pybaseball` for real pitcher data.",
              file=sys.stderr)
        return {}
    stats: dict[str, tuple[float, float]] = {}
    for s in seasons:
        try:
            df = pitching_stats(s, qual=1)
        except Exception as exc:  # network/parse issues shouldn't be fatal
            print(f"  warning: pybaseball pitching_stats({s}) failed: {exc}",
                  file=sys.stderr)
            continue
        for row in df.itertuples():
            name = getattr(row, "Name", None)
            xfip = getattr(row, "xFIP", None)
            kbb = getattr(row, "_K_BB_pct", None)  # 'K-BB%' -> sanitized attr
            if kbb is None:
                kbb = getattr(row, "K_BB_pct", None)
            if name and xfip is not None:
                frac = float(kbb) / 100 if kbb is not None else LEAGUE_AVG_K_BB_PCT
                stats[str(name)] = (float(xfip), frac)  # latest season wins
    return stats


def fetch_season(season: int, starters: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """All regular-season games for a season: finals with runs, plus any
    upcoming fixtures as 'scheduled', each with its probable-starter line
    and ballpark factor."""
    data = _get({
        "sportId": 1, "season": season, "gameType": "R",
        "startDate": f"{season}-03-01", "endDate": f"{season}-11-15",
        "hydrate": "probablePitcher",
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

            def starter(side):
                name = side.get("probablePitcher", {}).get("fullName")
                xfip, kbb = starters.get(name, (LEAGUE_AVG_XFIP, LEAGUE_AVG_K_BB_PCT))
                return xfip, kbb

            h_xfip, h_kbb = starter(home)
            a_xfip, a_kbb = starter(away)
            home_name = home["team"]["name"]
            row = {
                "date": g["gameDate"][:10], "season": season,
                "home_team": home_name, "away_team": away["team"]["name"],
                "status": "played" if final else "scheduled",
                "home_starter_xfip": h_xfip, "home_starter_k_bb_pct": h_kbb,
                "away_starter_xfip": a_xfip, "away_starter_k_bb_pct": a_kbb,
                "park_factor": PARK_FACTORS.get(normalize_team(home_name), 1.0),
            }
            if final:
                row["home_runs"] = home.get("score")
                row["away_runs"] = away.get("score")
                if row["home_runs"] is None or row["away_runs"] is None:
                    continue
            rows.append(row)
    return pd.DataFrame(rows)


def main(seasons: list[int]) -> None:
    print("Loading starting-pitcher stats (xFIP, K-BB%)...")
    starters = _starter_stats(seasons)
    if starters:
        print(f"  got season lines for {len(starters)} pitchers")

    frames = []
    for s in seasons:
        print(f"Fetching {s} schedule + probable pitchers from MLB Stats API...")
        try:
            frames.append(fetch_season(s, starters))
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
