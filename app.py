"""Streamlit website for the Lesson 10 EPL attack-vs-defence predictor.

Start locally: python -m streamlit run app.py
Publish: Streamlit Community Cloud, app.py entry point.
"""
from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_model import (
    BLEND_ATTACK_WEIGHT,
    attack_defence_rating,
    clean_season,
    club_ratings,
    combine_seasons,
    download_matches,
    fit_shots_model,
    historical_brier,
    latest_clubs,
    model_training_data,
    most_likely_score,
    outcome_probs,
    prediction_for_match,
    prematch_records,
    results_for_prediction,
    season_label,
)

st.set_page_config(page_title="EPL Match Lab | Attack × Defence", page_icon="⚽", layout="wide")

CSS = """
<style>
  .block-container {padding-top: 2.1rem; max-width: 1160px;}
  [data-testid="stSidebar"] {border-right: 1px solid #23323B;}
  .hero-label {font-size: .72rem; font-weight: 750; letter-spacing: .17em; color: #6ee2bd; text-transform: uppercase;}
  .hero-title {font-size: clamp(2rem, 4.5vw, 3.55rem); font-weight: 800; line-height: 1.08; letter-spacing: -.045em; margin: .65rem 0 .7rem;}
  .hero-subtitle {color: #A9B8C0; font-size: 1.02rem; line-height: 1.6; max-width: 740px;}
  .score-panel {background: linear-gradient(140deg,#172A34,#132027); border: 1px solid #31464E; border-radius: 18px; padding: 20px; text-align: center;}
  .score-label {color: #9AB1BB; font-size: .78rem; letter-spacing:.1em; text-transform: uppercase;}
  .score-main {font-size: 3.2rem; color: #F4FAF8; font-weight: 850; letter-spacing:-.05em; padding: .35rem 0;}
  .score-goals {font-size: .87rem; color: #A9C5C3;}
  .soft-note {border-left: 3px solid #4DD9B0; padding: .65rem 1rem; background:#17322F; border-radius:4px; color:#D5EDE4;}
  div.stButton > button[kind="primary"] {font-weight:750;}
  h3 {padding-top: .4rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_data(ttl=6 * 3600, show_spinner="Downloading Premier League results…")
def cached_download():
    return download_matches()

@st.cache_data(show_spinner="Preparing pre-match features…")
def cached_features(matches: pd.DataFrame):
    return prematch_records(matches)

@st.cache_resource(show_spinner="Fitting the goals + shots comparison model…")
def cached_shot_model(prepared: pd.DataFrame):
    return fit_shots_model(model_training_data(prepared))

@st.cache_data(show_spinner="Recomputing historical tests…")
def cached_brier(matches: pd.DataFrame, prepared: pd.DataFrame):
    return historical_brier(matches, prepared)


def make_score_heatmap(grid: np.ndarray, home: str, away: str):
    n = 7  # 0–6 goals shown; the full model also includes 7–12
    values = grid[:n, :n] * 100
    figure = go.Figure(data=go.Heatmap(
        z=values,
        x=list(range(n)), y=list(range(n)),
        colorscale=[[0,"#16242B"],[0.25,"#215E62"],[0.65,"#289D84"],[1,"#5BE8B8"]],
        colorbar={"title": "%", "ticksuffix":"%", "thickness": 12},
        hovertemplate=f"{home}: %{{y}} goals<br>{away}: %{{x}} goals<br>Probability: %{{z:.2f}}%<extra></extra>",
        text=np.round(values,1),
        texttemplate="%{text:.1f}",
        textfont={"size": 11, "color":"#E9FAF2"},
    ))
    figure.update_layout(
        xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
        plot_bgcolor="#101820", paper_bgcolor="rgba(0,0,0,0)",
        font={"color":"#CBDBDE", "size": 12},
        xaxis={"side":"bottom", "dtick":1}, yaxis={"autorange":"reversed", "dtick":1},
        margin={"t": 20, "b": 55, "l": 50, "r": 10},
        height=450,
    )
    return figure


def goal_probability_bars(probs: np.ndarray, home: str, away: str):
    labels = [f"{home} win", "Draw", f"{away} win"]
    colors = ["#3DD5A9", "#93A8AF", "#E3A96D"]
    figure = go.Figure(go.Bar(
        x=probs * 100, y=labels,
        orientation="h", marker_color=colors,
        text=[f"{100*x:.1f}%" for x in probs], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    figure.update_layout(
        xaxis={"range":[0,105],"ticksuffix":"%", "visible":False},
        yaxis={"autorange":"reversed", "tickfont":{"size":13}},
        font={"color":"#CFDDE0"}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t":0,"b":0,"l":0,"r":55}, height=175,
    )
    return figure


st.markdown('<div class="hero-label">Independent football analytics project · Lesson 11</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">EPL Match <span style="color:#6EE2BD">Lab.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">How does one team’s attack match up against the opponent’s defence? Explore expected goals and score probabilities using real Premier League results.</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("### ⚙️ Match Lab")
    uploaded = st.file_uploader(
        "Optional: upload Football-Data match CSV files instead of downloading",
        type="csv", accept_multiple_files=True,
        help="CSV files must contain Date, HomeTeam, AwayTeam, FTHG and FTAG. Upload multiple seasons for comparisons.",
    )
    if st.button("↻ Refresh downloaded match data", use_container_width=True):
        cached_download.clear()
        st.rerun()

try:
    if uploaded:
        parts = [clean_season(pd.read_csv(io.BytesIO(item.getvalue()), on_bad_lines="skip", low_memory=False))
                 for item in uploaded]
        matches = combine_seasons(parts)
        fetch_warnings = []
        source_name = "Your uploaded Football-Data CSV files"
    else:
        matches, fetch_warnings = cached_download()
        source_name = "Football-Data.co.uk"
except Exception as exc:
    st.error(f"Cannot load Premier League match data: {exc}")
    st.info("Try 'Refresh downloaded match data' in the sidebar, or upload season CSV files from football-data.co.uk.")
    st.stop()

if len(matches) < 30:
    st.error("There are too few completed matches to provide useful ratings. Upload additional season CSV files.")
    st.stop()

clubs, latest_season = latest_clubs(matches)
latest_date = matches.Date.max().date()

with st.sidebar:
    st.markdown("### 📊 Data status")
    st.write(f"**Source:** {source_name}")
    st.write(f"**Completed results:** {len(matches):,}")
    st.write(f"**Latest match:** {latest_date.isoformat()}")
    st.write(f"**Latest season in data:** {season_label(latest_season)}")
    if fetch_warnings:
        with st.expander(f"{len(fetch_warnings)} season download warnings"):
            for warning in fetch_warnings:
                st.caption(warning)
    st.divider()
    st.caption("Model scope: full-time Premier League scores only. Not live, and no player, injury, formation or lineup inputs.")
    st.caption("Historical results contain outcomes, not pre-match betting information.")

if len(clubs) < 2:
    st.error("Need at least two different clubs in the latest uploaded season.")
    st.stop()

st.caption(f"Match data through **{latest_date.isoformat()}** · {len(clubs)} clubs shown from **{season_label(latest_season)}** · Predictions are hypothetical scenarios, not live fixture forecasts.")

tab1, tab2, tab3 = st.tabs(["⚽ Match predictor", "📈 Attack & defence ratings", "🧪 Model evidence"])

with tab1:
    st.subheader("Choose your matchup")
    col_h, col_a = st.columns(2)
    h_index = clubs.index("Arsenal") if "Arsenal" in clubs else 0
    a_index = clubs.index("Chelsea") if "Chelsea" in clubs else (1 if h_index == 0 else 0)
    with col_h:
        home = st.selectbox("Home team", clubs, index=h_index, key="home_team")
    with col_a:
        away = st.selectbox("Away team", clubs, index=a_index, key="away_team")
    if home == away:
        st.warning("Choose different home and away teams.")
    else:
        prepared, history = cached_features(matches)
        shot_model = None
        if len(model_training_data(prepared)) >= 100:
            try:
                shot_model = cached_shot_model(prepared)
            except Exception as exc:
                st.warning(f"Shots comparison currently unavailable: {exc}")
        else:
            st.info("Upload more historical seasons to enable the recent-goals-and-shots comparison model.")
        pred = prediction_for_match(matches, history, shot_model, home, away)
        rows = results_for_prediction(pred)
        grid = pred.distributions["Attack × Defence"]
        probs = outcome_probs(grid)
        score, score_prob = most_likely_score(grid)
        goals = pred.goals["Attack × Defence"]
        st.markdown("#### Preferred model · Attack × Defence")
        safe_home = html.escape(home)
        safe_away = html.escape(away)
        st.markdown(f"""
        <div class="score-panel">
          <div class="score-label">Most likely exact score · {score_prob*100:.1f}% probability</div>
          <div class="score-main">{safe_home} &nbsp; {score} &nbsp; {safe_away}</div>
          <div class="score-goals">Estimated goals &nbsp; {goals[0]:.2f} — {goals[1]:.2f}</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(goal_probability_bars(probs, home, away), use_container_width=True, config={"displayModeBar":False})
        st.caption("The most likely exact score may have a low probability even when one team is favoured to win overall.")

        col_l, col_r = st.columns([1, 1.04], gap="large")
        with col_l:
            st.markdown("#### Why the model favours this result")
            ratings = pred.ratings
            why = pd.DataFrame([
                {"Component":"Home attack", "Club":home, "Rating":ratings["Home attack"]},
                {"Component":"Away defensive vulnerability", "Club":away, "Rating":ratings["Away defence vulnerability"]},
                {"Component":"Away attack", "Club":away, "Rating":ratings["Away attack"]},
                {"Component":"Home defensive vulnerability", "Club":home, "Rating":ratings["Home defence vulnerability"]},
            ])
            st.dataframe(why.style.format({"Rating":"{:.2f}"}), use_container_width=True, hide_index=True)
            st.caption("1.00 is the league average for the relevant home or away venue. Higher attack = stronger; lower defensive vulnerability = harder to score against.")
            st.markdown(f'<div class="soft-note">{safe_home} attack × {safe_away} defensive vulnerability determines the home expected goals; the reverse matchup determines the away expected goals.</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown("#### Exact-score probability heatmap")
            st.plotly_chart(make_score_heatmap(grid, home, away), use_container_width=True, config={"displayModeBar":False})
            st.caption("Shows 0–6 goals per side; the model sums probabilities over 0–12 goals to calculate match outcomes.")

        st.markdown("#### Compare model approaches")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.download_button("Download these predictions (CSV)", rows.to_csv(index=False).encode("utf-8"),
                           file_name="epl_matchlab_prediction.csv", mime="text/csv")
        st.caption("The 25/75 blend was the best *blend weight* tested in Lesson 10 validation, not the best overall validation model. Its results are experimental.")

with tab2:
    st.subheader("Compare every club’s attack and defence")
    st.write("Ratings use the same rolling 730-day history and venue-specific smoothing as the score predictor.")
    ranking = club_ratings(matches, clubs)
    chosen_sort = st.selectbox("Sort ratings by", [
        "Home attack", "Away attack", "Home defence vulnerability", "Away defence vulnerability"
    ])
    # Bigger is desirable for attack, but a lower vulnerability is desirable for defence.
    ascending = "vulnerability" in chosen_sort
    ranking = ranking.sort_values(chosen_sort, ascending=ascending).reset_index(drop=True)
    st.dataframe(ranking.style.format({
        "Home attack":"{:.2f}", "Away attack":"{:.2f}",
        "Home defence vulnerability":"{:.2f}", "Away defence vulnerability":"{:.2f}",
    }), hide_index=True, use_container_width=True)
    st.download_button("Download club ratings (CSV)", ranking.to_csv(index=False).encode("utf-8"),
                       file_name="epl_matchlab_ratings.csv", mime="text/csv")
    st.info("These are relative goal-scoring / conceding ratings, not separate measures of true underlying player quality or tactical ability. Recently promoted clubs may have little recent EPL history and are smoothed toward league averages.")

with tab3:
    st.subheader("How reliable are the predictions?")
    st.write("We previously evaluated these model specifications on 2024–25 and 2025–26. Lower **Brier scores** mean more accurate win/draw/loss probabilities. These seasons have already influenced our project decisions and are **not fresh independent tests**.")
    if st.button("Calculate historical Brier scores", type="primary"):
        st.session_state["show_brier"] = True
    if st.session_state.get("show_brier", False):
        try:
            prepared_all, _ = cached_features(matches)
            historical = cached_brier(matches, prepared_all)
            st.dataframe(historical.pivot(index="Model",columns="Season",values="Brier"), use_container_width=True)
            st.caption("Identical fixtures and pre-match data cutoff for each model. A historical advantage doesn't guarantee future accuracy.")
        except Exception as exc:
            st.warning(f"Backtest unavailable: {exc}. Upload all historical seasons from 2018–19 to 2025–26.")
    st.markdown("""
    **How Attack × Defence works**

    - A club's home attack rating = its smoothed average home goals divided by the league's average home goals.
    - Its opponent's away defensive vulnerability = its smoothed average goals conceded away divided by the league average of home goals.
    - Multiply the league goal average by these two ratings to estimate goals, and reverse the comparison for the away team.
    - A Poisson distribution converts these two goal estimates into exact-score and outcome probabilities.

    **What this version cannot do:** reliably assess player swaps, coaching tactics, injuries, or real-time starting XIs. No betting claims or guarantees. We should save predictions before kickoff and assess genuinely future matches next.
    """)
    st.caption("Source: Football-Data.co.uk historical Premier League CSVs. Collected match results may have reporting delays. Refresh the dataset when returning to the site.")
