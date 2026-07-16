"""Track MLB model predictions vs. actual results over time.

Locks in each game's pick (winning team) and most likely exact score
before first pitch, scores it once you enter the final, and regenerates
MLB_RESULTS.md — a scoreboard that renders as a table on GitHub.
"""

from datetime import datetime

import pandas as pd

from .config import DATA_DIR, PROJECT_ROOT
from .predict import best_scoreline_for_winner, predict_frame
from .features import upcoming_frame

LOG_CSV = DATA_DIR / "mlb_predictions_log.csv"
RESULTS_MD = PROJECT_ROOT / "MLB_RESULTS.md"

LOG_COLUMNS = [
    "date", "home_team", "away_team",
    "pred_home_runs", "pred_away_runs", "p_home_win", "p_away_win",
    "predicted_pick", "predicted_score",
    "actual_home_runs", "actual_away_runs",
    "actual_winner", "outcome_hit", "score_hit",
]


def _is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() in ("", "nan", "<NA>", "None")


def _pick(row) -> tuple[str, bool, float]:
    """Favored team, whether that's the home team, and its win prob."""
    home_wins = row["p_home_win"] >= row["p_away_win"]
    name = row["home_team"] if home_wins else row["away_team"]
    prob = max(row["p_home_win"], row["p_away_win"])
    return name, home_wins, prob


def _confidence(prob: float) -> str:
    if prob >= 0.60:
        return "🟢 Strong"
    if prob >= 0.53:
        return "🟡 Lean"
    return "🔴 Toss-up"


def load_log() -> pd.DataFrame:
    if LOG_CSV.exists():
        return pd.read_csv(LOG_CSV)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(df: pd.DataFrame) -> None:
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_CSV, index=False)


def record(bundle: dict) -> None:
    preds = predict_frame(bundle, upcoming_frame())
    log = load_log()
    existing = set(zip(log["date"].astype(str), log["home_team"], log["away_team"]))

    new_rows = []
    for row in preds.itertuples():
        date = str(row.date.date())
        if (date, row.home_team, row.away_team) in existing:
            continue
        name, home_wins, _ = _pick({
            "p_home_win": row.p_home_win, "p_away_win": row.p_away_win,
            "home_team": row.home_team, "away_team": row.away_team})
        new_rows.append({
            "date": date, "home_team": row.home_team, "away_team": row.away_team,
            "pred_home_runs": round(row.pred_home_runs, 2),
            "pred_away_runs": round(row.pred_away_runs, 2),
            "p_home_win": round(row.p_home_win, 3),
            "p_away_win": round(row.p_away_win, 3),
            "predicted_pick": name,
            "predicted_score": best_scoreline_for_winner(
                row.pred_home_runs, row.pred_away_runs, home_wins),
            "actual_home_runs": "", "actual_away_runs": "",
            "actual_winner": "", "outcome_hit": "", "score_hit": "",
        })

    if not new_rows:
        print("No new games to log — everything scheduled is already tracked.")
        return
    log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    log = log.sort_values(["date", "home_team"]).reset_index(drop=True)
    save_log(log)
    build_results_md(log)
    print(f"Logged {len(new_rows)} new game(s) -> {LOG_CSV}")


def enter_result(home: str, away: str, score: str) -> None:
    """`score` is home-away runs, e.g. 5-3."""
    try:
        hr, ar = (int(x) for x in score.replace(":", "-").split("-"))
    except ValueError:
        raise SystemExit(f"Couldn't read score '{score}'. Use the form 5-3.")
    if hr == ar:
        raise SystemExit("Baseball games don't end tied — check the score.")

    log = load_log()
    mask = (log["home_team"] == home) & (log["away_team"] == away) & \
           log["actual_home_runs"].map(_is_blank)
    if not mask.any():
        raise SystemExit(f"No un-scored logged game found for {away} @ {home}. "
                         "Run `track record` first, or check the team names.")
    i = log[mask].index[0]

    for col in ("actual_home_runs", "actual_away_runs", "actual_winner",
                "outcome_hit", "score_hit"):
        log[col] = log[col].astype(object)

    winner = home if hr > ar else away
    log.loc[i, "actual_home_runs"] = hr
    log.loc[i, "actual_away_runs"] = ar
    log.loc[i, "actual_winner"] = winner
    log.loc[i, "outcome_hit"] = int(log.loc[i, "predicted_pick"] == winner)
    log.loc[i, "score_hit"] = int(log.loc[i, "predicted_score"] == f"{hr}-{ar}")
    save_log(log)
    build_results_md(log)

    hit = "✅ correct" if log.loc[i, "predicted_pick"] == winner else "❌ missed"
    print(f"Recorded {away} {ar}-{hr} @ {home}. Model picked "
          f"'{log.loc[i, 'predicted_pick']}' — {hit}.")


def _summary(log: pd.DataFrame) -> dict:
    done = log[~log["outcome_hit"].map(_is_blank)].copy()
    if done.empty:
        return {"played": 0}
    done["outcome_hit"] = done["outcome_hit"].astype(float).astype(int)
    done["score_hit"] = done["score_hit"].astype(float).astype(int)
    return {"played": len(done), "outcome_hits": int(done["outcome_hit"].sum()),
            "outcome_acc": done["outcome_hit"].mean(),
            "score_hits": int(done["score_hit"].sum())}


def build_results_md(log: pd.DataFrame | None = None) -> None:
    if log is None:
        log = load_log()
    s = _summary(log)

    lines = ["# ⚾ MLB 2026 — Model vs. Reality", ""]
    if s["played"]:
        lines += [
            f"**Winner record:** {s['outcome_hits']} / {s['played']} correct "
            f"({s['outcome_acc']:.0%})  ·  "
            f"**Exact scores nailed:** {s['score_hits']} / {s['played']}",
            "",
            "> Picking the winning team is where the model has an edge. Exact "
            "run totals are a long shot for any model — tracked here for fun.",
            "",
        ]
    else:
        lines += ["_No games scored yet — predictions are locked in and waiting._", ""]

    lines += [
        "| Date | Game (away @ home) | Model pick | Conf. | Pred. score | Actual | Pick | Score |",
        "|------|--------------------|-----------|:-----:|:-----------:|:------:|:----:|:-----:|",
    ]
    for row in log.itertuples():
        _, _, prob = _pick(row._asdict())
        played = not _is_blank(row.actual_home_runs)
        if played:
            actual = f"{int(float(row.actual_home_runs))}-{int(float(row.actual_away_runs))}"
            pick_mark = "✅" if int(float(row.outcome_hit)) else "❌"
            score_mark = "✅" if int(float(row.score_hit)) else "❌"
        else:
            actual, pick_mark, score_mark = "⏳", "⏳", "⏳"
        lines.append(
            f"| {row.date} | {row.away_team} @ {row.home_team} "
            f"| **{row.predicted_pick}** | {_confidence(prob)} "
            f"| {row.predicted_score} | {actual} | {pick_mark} | {score_mark} |")

    lines += ["", f"_Last updated {datetime.now():%Y-%m-%d}. "
              "Generated by `python -m mlb_predictor track board`._"]
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def board() -> None:
    log = load_log()
    build_results_md(log)
    s = _summary(log)
    if s["played"]:
        print(f"Record: {s['outcome_hits']}/{s['played']} winners "
              f"({s['outcome_acc']:.0%}), {s['score_hits']} exact scores. "
              f"Scoreboard -> {RESULTS_MD}")
    else:
        print(f"{len(log)} games logged, none scored yet. Scoreboard -> {RESULTS_MD}")
