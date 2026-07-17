"""Browser dashboard for the MLB predictor.

    pip install streamlit
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from mlb_predictor.features import upcoming_frame
from mlb_predictor.predict import predict_frame
from mlb_predictor.track import _is_blank, load_log
from mlb_predictor.train import load_bundle

st.set_page_config(page_title="MLB 2026 Predictor", page_icon="⚾", layout="wide")
st.title("⚾ MLB 2026 Predictor")


@st.cache_resource
def _bundle():
    return load_bundle()


@st.cache_data(ttl=1800)
def _predictions() -> pd.DataFrame:
    return predict_frame(_bundle(), upcoming_frame())


tab_picks, tab_board = st.tabs(["🔮 Picks", "📊 Scoreboard"])

with tab_picks:
    df = _predictions()
    dates = sorted(df["date"].dt.date.unique())
    if not dates:
        st.info("No scheduled games — run `python -m mlb_predictor fetch` first.")
    else:
        day = st.selectbox("Slate", dates, format_func=lambda d: d.strftime("%A %B %d, %Y"))
        games = df[df["date"].dt.date == day].copy()
        home_wins = games["p_home_win"] >= games["p_away_win"]
        view = pd.DataFrame({
            "Game": games["away_team"] + " @ " + games["home_team"],
            "Pick": np.where(home_wins, games["home_team"], games["away_team"]),
            "Win prob": games[["p_home_win", "p_away_win"]].max(axis=1),
            "Expected score": (games["pred_home_runs"].round(1).astype(str) + " – "
                               + games["pred_away_runs"].round(1).astype(str)),
            "Proj. total": (games["pred_home_runs"] + games["pred_away_runs"]).round(1),
        })
        view = view.sort_values("Win prob", ascending=False)
        st.dataframe(
            view, hide_index=True, width='stretch',
            column_config={
                "Win prob": st.column_config.ProgressColumn(
                    "Win prob", format="percent", min_value=0.0, max_value=1.0),
                "Expected score": st.column_config.TextColumn(
                    "Exp. runs (home – away)"),
            })
        st.caption("🟢 ≥60%: strong · 🟡 53–60%: lean · 🔴 <53%: toss-up. "
                   "Even strong picks lose ~40% of the time — that's baseball.")

with tab_board:
    log = load_log()
    if log.empty:
        st.info("No tracked picks yet — `python -m mlb_predictor track record`.")
    else:
        done = log[~log["outcome_hit"].map(_is_blank)]
        if len(done):
            hits = done["outcome_hit"].astype(float).astype(int)
            scores = done["score_hit"].astype(float).astype(int)
            c1, c2, c3 = st.columns(3)
            c1.metric("Winner record", f"{hits.sum()} / {len(done)}")
            c2.metric("Accuracy", f"{hits.mean():.0%}")
            c3.metric("Exact scores", f"{scores.sum()} / {len(done)}")

        def status(row):
            if _is_blank(row["actual_home_runs"]):
                return "⏳"
            return "✅" if int(float(row["outcome_hit"])) else "❌"

        board = pd.DataFrame({
            "Date": log["date"],
            "Game": log["away_team"] + " @ " + log["home_team"],
            "Pick": log["predicted_pick"],
            "Conf.": log[["p_home_win", "p_away_win"]].max(axis=1),
            "Pred. score": log["predicted_score"],
            "Actual": [
                "⏳" if _is_blank(h) else f"{int(float(h))}-{int(float(a))}"
                for h, a in zip(log["actual_home_runs"], log["actual_away_runs"])],
            "Hit": log.apply(status, axis=1),
        })
        st.dataframe(
            board.sort_values("Date", ascending=False), hide_index=True,
            width='stretch',
            column_config={"Conf.": st.column_config.ProgressColumn(
                "Conf.", format="percent", min_value=0.0, max_value=1.0)})
