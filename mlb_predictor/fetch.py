"""Real data fetcher: everything from the free public MLB Stats API
(statsapi.mlb.com) — no key, no scraping.

- Schedule endpoint: final scores, upcoming fixtures, and — with
  hydrate=probablePitcher — each game's probable starters.
- Season pitching stats endpoint: per-pitcher K, BB, HBP, HR, IP, and
  batters faced, from which FIP and K-BB% are computed and joined to the
  probable starters. (FanGraphs/pybaseball was the original source, but
  it now blocks automated requests with 403s.)
- Ballpark run factors come from the static PARK_FACTORS table.

Per-game box-score rates (AVG/OBP/SLG/ERA/WHIP) still aren't pulled and
are imputed with league averages — rolling runs for/against, win %, the
starter matchup, and park carry the signal.

NOTE: written offline (the API is blocked in the build sandbox). Run
locally and sanity-check the first response.
"""

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import (LEAGUE_AVG_K_BB_PCT, LEAGUE_AVG_XFIP, MATCHES_CSV,
                     PARK_FACTORS, normalize_team)

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
EASTERN = ZoneInfo("America/New_York")


def _local_date(iso_utc: str) -> str:
    """Game date in US/Eastern. The API's gameDate is UTC, which pushes
    every evening game onto the next calendar day."""
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        return dt.astimezone(EASTERN).date().isoformat()
    except ValueError:
        return str(iso_utc)[:10]

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


MLB_STATS = "https://statsapi.mlb.com/api/v1/stats"


def _innings(ip_str) -> float:
    """MLB innings-pitched strings use .1/.2 for thirds of an inning."""
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole or 0) + int(frac or 0) / 3
    except ValueError:
        return 0.0


def _starter_stats(seasons: list[int]) -> dict[str, tuple[float, float]]:
    """Map pitcher full name -> (FIP, K-BB% as a fraction), computed from
    the MLB Stats API's season pitching lines (same free API as the
    schedule — no scraping, no key).

    FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.10 — the defense-independent
    'what ERA should have been'. Stored under the *_xfip feature names.
    Season-level (not as-of-date) stats are a deliberate simplification."""
    stats: dict[str, tuple[float, float]] = {}
    for s in seasons:
        try:
            resp = requests.get(MLB_STATS, params={
                "stats": "season", "group": "pitching", "season": s,
                "sportId": 1, "playerPool": "all", "limit": 3000}, timeout=60)
            resp.raise_for_status()
            groups = resp.json().get("stats", [])
        except (requests.RequestException, ValueError) as exc:
            print(f"  warning: pitching stats failed for {s}: {exc}",
                  file=sys.stderr)
            continue
        for split in (groups[0].get("splits", []) if groups else []):
            name = split.get("player", {}).get("fullName")
            st = split.get("stat", {})
            ip = _innings(st.get("inningsPitched", 0))
            bf = float(st.get("battersFaced", 0) or 0)
            if not name or ip < 10 or bf < 40:
                continue  # too small a sample to say anything
            k = float(st.get("strikeOuts", 0) or 0)
            bb = float(st.get("baseOnBalls", 0) or 0)
            hbp = float(st.get("hitByPitch", 0) or 0)
            hr = float(st.get("homeRuns", 0) or 0)
            fip = (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + 3.10
            fip = min(max(fip, 1.0), 8.0)
            kbb = (k - bb) / bf
            stats[str(name)] = (round(fip, 2), round(kbb, 3))  # latest season wins
    return stats


def fetch_season(season: int, starters: dict[str, tuple[float, float]],
                 include_live: bool = False) -> pd.DataFrame:
    """All regular-season games for a season: finals with runs, plus any
    upcoming fixtures as 'scheduled', each with its probable-starter line
    and ballpark factor.

    In-progress games are normally skipped (no final score to train on,
    not upcoming either). With include_live=True they're kept as
    'scheduled' so today's already-started games can still be predicted —
    the model only uses pre-game info, so the prediction is the same one
    it would have made before first pitch."""
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
            keep_live = include_live and state == "Live"
            if not final and not keep_live and state not in ("Preview", "Scheduled"):
                continue  # skip in-progress (unless asked) / suspended

            def starter(side):
                name = side.get("probablePitcher", {}).get("fullName")
                xfip, kbb = starters.get(name, (LEAGUE_AVG_XFIP, LEAGUE_AVG_K_BB_PCT))
                return xfip, kbb

            h_xfip, h_kbb = starter(home)
            a_xfip, a_kbb = starter(away)
            home_name = home["team"]["name"]
            row = {
                "date": _local_date(g["gameDate"]), "season": season,
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


def main(seasons: list[int], include_live: bool = False) -> None:
    print("Loading starting-pitcher stats (xFIP, K-BB%)...")
    starters = _starter_stats(seasons)
    if starters:
        print(f"  got season lines for {len(starters)} pitchers")

    frames = []
    for s in seasons:
        print(f"Fetching {s} schedule + probable pitchers from MLB Stats API...")
        try:
            frames.append(fetch_season(s, starters, include_live))
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
