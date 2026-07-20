"""Compare the model's probabilities with real sportsbook odds.

Uses The Odds API (the-odds-api.com, free tier). The key is read from
the ODDS_API_KEY environment variable — set it once in ~/.zprofile:

    export ODDS_API_KEY="your-key"

Implied probabilities are averaged across US books and de-vigged
(normalized so they sum to 1), giving the market's honest opinion to
hold the model against.
"""

import os
import sys

import requests

from .config import normalize_team
from .features import upcoming_frame
from .predict import predict_frame

SPORT_KEY = "baseball_mlb"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def _implied(american) -> float:
    """American odds -> implied probability (vig included)."""
    a = float(american)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def fetch_market() -> tuple[dict, str | None]:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit(
            "No ODDS_API_KEY set. Run:\n"
            '  echo \'export ODDS_API_KEY="your-key"\' >> ~/.zprofile\n'
            "  source ~/.zprofile")
    resp = requests.get(ODDS_URL.format(sport=SPORT_KEY), params={
        "apiKey": key, "regions": "us", "markets": "h2h,totals",
        "oddsFormat": "american"}, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")

    market: dict = {}
    for ev in resp.json():
        home = normalize_team(ev.get("home_team"))
        away = normalize_team(ev.get("away_team"))
        if not home or not away:
            continue
        home_probs, totals = [], []
        for book in ev.get("bookmakers", []):
            for mk in book.get("markets", []):
                if mk["key"] == "h2h":
                    ph = pa = None
                    for o in mk.get("outcomes", []):
                        team = normalize_team(o.get("name"))
                        if team == home:
                            ph = _implied(o["price"])
                        elif team == away:
                            pa = _implied(o["price"])
                    if ph and pa:
                        home_probs.append(ph / (ph + pa))  # de-vig
                elif mk["key"] == "totals":
                    points = [o.get("point") for o in mk.get("outcomes", [])
                              if o.get("point") is not None]
                    if points:
                        totals.append(sum(points) / len(points))
        if home_probs:
            market[(home, away)] = {
                "p_home": sum(home_probs) / len(home_probs),
                "total": sum(totals) / len(totals) if totals else None,
                "books": len(home_probs),
            }
    return market, remaining


def main(bundle: dict) -> None:
    market, remaining = fetch_market()
    if not market:
        raise SystemExit("The Odds API returned no MLB games with odds right now.")
    preds = predict_frame(bundle, upcoming_frame())

    print(f"{'Game':<34}{'Model':>8}{'Market':>8}{'Edge':>7}   totals model/line")
    seen, rows, big = set(), [], 0
    for row in preds.itertuples():
        key = (row.home_team, row.away_team)
        m = market.get(key)
        # Teams meet several times upcoming; the market has one line per
        # matchup, so keep only the nearest game (preds is date-sorted).
        if not m or key in seen:
            continue
        seen.add(key)
        edge = (row.p_home_win - m["p_home"]) * 100
        model_total = row.pred_home_runs + row.pred_away_runs
        line = f"{m['total']:.1f}" if m["total"] else "  — "
        game = f"{row.away_team.split()[-1]} @ {row.home_team.split()[-1]}"
        rows.append((game, row.home_team, row.away_team, row.p_home_win,
                     m["p_home"], edge, model_total, m["total"], m["books"]))
        flag = "  <- BIG" if abs(edge) >= 8 else ""
        print(f"{game:<34}{row.p_home_win:>7.0%} {m['p_home']:>7.0%} {edge:>+6.1f}   "
              f"{model_total:.1f} / {line}   ({m['books']} books){flag}")
        big += abs(edge) >= 8
    if not rows:
        print("No overlap between the model's upcoming games and the odds feed "
              "— refetch data (`python -m mlb_predictor fetch`) and retry.")
        return
    print("\nModel/Market = home team's win probability. Positive edge: the "
          "model likes the home side more than Vegas does.")
    print(f"{big} BIG edge(s) (>=8 pts) flagged — those usually mean the market "
          "knows something the model can't (injury, bullpen use, a scratch). "
          "Check `team-news` before trusting them.")
    _write_markdown(rows)
    if remaining:
        print(f"The Odds API requests remaining this month: {remaining}",
              file=sys.stderr)


def _write_markdown(rows) -> None:
    """Write ODDS.md — a model-vs-market table that renders on GitHub."""
    from .config import PROJECT_ROOT
    from datetime import datetime
    lines = ["# ⚾ Model vs. Market — MLB", "",
             f"_Updated {datetime.now():%Y-%m-%d %H:%M}. Home-team win "
             "probability; positive edge = model higher than the books._", "",
             "| Game (away @ home) | Model | Market | Edge | Proj. total | Line |",
             "|---|:---:|:---:|:---:|:---:|:---:|"]
    for game, _h, _a, model_p, mkt_p, edge, total, line, books in rows:
        flag = " ⚠️" if abs(edge) >= 8 else ""
        line_s = f"{line:.1f}" if line else "—"
        lines.append(f"| {game} | {model_p:.0%} | {mkt_p:.0%} | "
                     f"{edge:+.1f}{flag} | {total:.1f} | {line_s} |")
    lines += ["", "⚠️ = model and market disagree by 8+ points — the book "
              "usually knows something the model can't. Verify before trusting."]
    (PROJECT_ROOT / "ODDS.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote model-vs-market table -> {PROJECT_ROOT / 'ODDS.md'}")
