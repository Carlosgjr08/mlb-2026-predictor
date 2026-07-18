"""Injured-list report for a day's MLB slate, from the free MLB Stats
API (no key needed). Scouting info for the human alongside the model's
picks — the model doesn't see injuries directly, so a favorite with two
starters on the IL deserves a mental downgrade.
"""

from datetime import date

import requests

from .config import normalize_team

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
ROSTER = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"


def _get(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _injured(team_id: int) -> list[str]:
    """Players on any injured list from the 40-man roster."""
    data = _get(ROSTER.format(team_id=team_id), {"rosterType": "40Man"})
    out = []
    for slot in data.get("roster", []):
        status = slot.get("status", {})
        code = str(status.get("code", ""))
        if code.startswith("D") or "IL" in code:   # D7/D10/D15/D60 etc.
            name = slot.get("person", {}).get("fullName", "?")
            pos = slot.get("position", {}).get("abbreviation", "")
            desc = status.get("description", code)
            out.append(f"{name} ({pos}, {desc})")
    return out


def main(day: str | None = None) -> None:
    day = day or date.today().isoformat()
    sched = _get(SCHEDULE, {"sportId": 1, "date": day})
    games = [g for d in sched.get("dates", []) for g in d.get("games", [])]
    if not games:
        print(f"No MLB games found on {day}.")
        return

    cache: dict[int, list[str]] = {}
    for g in games:
        home = g["teams"]["home"]["team"]
        away = g["teams"]["away"]["team"]
        h_name = normalize_team(home["name"]) or home["name"]
        a_name = normalize_team(away["name"]) or away["name"]
        print(f"\n{a_name} @ {h_name}")
        for side in (h_name, a_name):
            team_id = home["id"] if side == h_name else away["id"]
            if team_id not in cache:
                cache[team_id] = _injured(team_id)
            hurt = cache[team_id]
            if hurt:
                print(f"  🚑 {side}: " + ", ".join(hurt))
            else:
                print(f"  🚑 {side}: nobody on the injured list")
