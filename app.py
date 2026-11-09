"""Streamlit website for the Lesson 10 EPL attack-vs-defence predictor.

Start locally: python -m streamlit run app.py
Publish: Streamlit Community Cloud, app.py entry point.
"""
from __future__ import annotations

import html
import io
import hmac
from pathlib import Path
from PIL import Image
from datetime import datetime, date, time, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prediction_tracker import (
    list_predictions, make_prediction_record, save_prediction, table_for_predictions,
    same_fixture, save_predictions_batch, UK_ZONE,
)
from fixtures import download_fixtures
from gameweek import group_fixture_windows, forecast_window, forecast_table
from live_scores import download_scoreboard, completed_score_dataframe, recent_scoreboard

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

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Match Lab | Premier League Analytics",
    page_icon=Image.open(APP_DIR / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_REPO = "dantakeuchinihs-alt/epl-match-lab"

def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default

REPO = secret("GITHUB_REPO", DEFAULT_REPO)
GITHUB_TOKEN = secret("GITHUB_TOKEN")
SAVE_PIN = secret("SAVE_PIN")
SCORES_API_TOKEN = secret("FOOTBALL_DATA_API_TOKEN")

# A single maintainable stylesheet; football model and data sources are unchanged.
CSS = (APP_DIR / "assets" / "site.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

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


@st.cache_data(ttl=90, show_spinner=False)
def cached_records(repo: str, token: str):
    return list_predictions(repo, token)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def cached_upcoming(clubs_tuple: tuple[str, ...]):
    return download_fixtures(list(clubs_tuple))


@st.cache_data(ttl=5 * 60, show_spinner=False)
def cached_scores(clubs_tuple: tuple[str, ...], token: str):
    # External score feed: separate from the historical training dataset.
    return download_scoreboard(token, list(clubs_tuple))


def make_score_heatmap(grid: np.ndarray, home: str, away: str):
    n = 7  # 0–6 goals shown; the full model also includes 7–12
    values = grid[:n, :n] * 100
    figure = go.Figure(data=go.Heatmap(
        z=values,
        x=list(range(n)), y=list(range(n)),
        colorscale=[[0,"#222722"],[0.25,"#49554A"],[0.65,"#8B9788"],[1,"#C9B38E"]],
        colorbar={"title": "%", "ticksuffix":"%", "thickness": 12},
        hovertemplate=f"{home}: %{{y}} goals<br>{away}: %{{x}} goals<br>Probability: %{{z:.2f}}%<extra></extra>",
        text=np.round(values,1),
        texttemplate="%{text:.1f}",
        textfont={"size": 11, "color":"#F1EEE6"},
    ))
    figure.update_layout(
        xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
        plot_bgcolor="#111412", paper_bgcolor="rgba(0,0,0,0)",
        font={"color":"#C3C9C0", "size": 12, "family":"DM Sans, Arial, sans-serif"},
        xaxis={"side":"bottom", "dtick":1}, yaxis={"autorange":"reversed", "dtick":1},
        margin={"t": 20, "b": 55, "l": 50, "r": 10},
        height=450,
    )
    return figure


def goal_probability_bars(probs: np.ndarray, home: str, away: str):
    labels = [f"{home} win", "Draw", f"{away} win"]
    colors = ["#BBA681", "#7D8A80", "#9FA9A1"]
    figure = go.Figure(go.Bar(
        x=probs * 100, y=labels,
        orientation="h", marker_color=colors,
        text=[f"{100*x:.1f}%" for x in probs], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    figure.update_layout(
        xaxis={"range":[0,105],"ticksuffix":"%", "visible":False},
        yaxis={"autorange":"reversed", "tickfont":{"size":13}},
        font={"color":"#C3C9C0", "family":"DM Sans, Arial, sans-serif"}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t":0,"b":0,"l":0,"r":55}, height=175,
    )
    return figure


st.markdown("""
<section class="masthead" aria-label="Match Lab introduction">
  <div class="brand-row">
    <span class="brand-kicker">Premier League / Independent analysis</span>
    <span class="brand-edition">Match Lab — 2026/27</span>
  </div>
  <div class="hero-title">Match <em>Lab.</em></div>
  <div class="hero-subrow">
    <p class="hero-subtitle">Premier League match forecasts, attack and defence analysis, and predictions checked against the final score.</p>
    <span class="hero-metadata">Pre-match analysis / Not live odds</span>
  </div>
</section>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('''<div class="sidebar-logo"><div class="sidebar-wordmark">Match <i>Lab.</i></div><div class="sidebar-subline">Premier League analysis</div></div>''', unsafe_allow_html=True)
    st.markdown("### Data controls")
    uploaded = st.file_uploader(
        "Optional: upload Football-Data match CSV files instead of downloading",
        type="csv", accept_multiple_files=True,
        help="CSV files must contain Date, HomeTeam, AwayTeam, FTHG and FTAG. Upload multiple seasons for comparisons.",
    )
    if st.button("Refresh historical data", use_container_width=True):
        cached_download.clear()
        st.rerun()
    if st.button("Refresh match scores", use_container_width=True):
        cached_scores.clear()
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

# The scores feed may be fresher than the model's historical training feed.
# Never silently add those goals to the model: that would change its validated specification.
provider_scores = None
provider_finals = None
provider_problem = None
if SCORES_API_TOKEN:
    try:
        provider_scores = cached_scores(tuple(clubs), SCORES_API_TOKEN)
        provider_finals = completed_score_dataframe(provider_scores)
    except Exception as exc:
        provider_problem = str(exc)

with st.sidebar:
    st.markdown("### Historical data")
    st.write(f"**Source:** {source_name}")
    st.write(f"**Completed results:** {len(matches):,}")
    st.write(f"**Latest match:** {latest_date.isoformat()}")
    st.write(f"**Latest season in data:** {season_label(latest_season)}")
    st.caption("These historical statistics power the prediction model. They may update slower than match results.")
    st.markdown("### Match results feed")
    if provider_scores is None:
        st.caption("Not connected" if not SCORES_API_TOKEN else f"Unavailable: {provider_problem}")
    else:
        completed = provider_scores.loc[provider_scores.is_finished]
        st.write(f"**Confirmed score records:** {len(completed)}")
        if len(completed):
            st.write(f"**Latest final:** {completed.kickoff_utc.max().astimezone(UK_ZONE).strftime('%d %b %Y')} (UK)")
        st.caption("Separate football-data.org API; free-plan scores can be delayed.")
    if fetch_warnings:
        with st.expander(f"{len(fetch_warnings)} season download warnings"):
            for warning in fetch_warnings:
                st.caption(warning)
    st.divider()
    st.caption("Prediction model uses historical full-time results, not in-play data, player availability or formations.")
    st.caption("Historical results contain outcomes, not pre-match betting information.")

if len(clubs) < 2:
    st.error("Need at least two different clubs in the latest uploaded season.")
    st.stop()

st.caption(f"HISTORICAL DATA TO {latest_date.isoformat()}   /   {len(clubs)} CLUBS   /   {season_label(latest_season)}   /   HYPOTHETICAL PREDICTIONS")

tab1, tab_scores, tab_gw, tab4, tab2, tab3 = st.tabs(["Predict", "Scores", "Fixtures", "Tracker", "Team ratings", "Method"])

with tab1:
    st.subheader("Match forecast")
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
        st.markdown("#### Preferred model — Attack × Defence")
        safe_home = html.escape(home)
        safe_away = html.escape(away)
        # The large editorial scoreline stays purely presentational. The
        # probabilities and the underlying predictive model are unchanged.
        score_parts = score.replace("—", "–").replace("-", "–").split("–")
        score_home = html.escape(score_parts[0].strip())
        score_away = html.escape(score_parts[1].strip())
        st.markdown(f"""
        <div class="score-panel">
          <div class="score-panel-top">
            <span class="score-label">Projected score / Attack × Defence</span>
            <span class="score-prob">{score_prob*100:.1f}% probability of this exact result</span>
          </div>
          <div class="score-line">
            <div class="score-club">{safe_home}<small>Home</small></div>
            <div class="score-numbers">{score_home}<span>:</span>{score_away}</div>
            <div class="score-club">{safe_away}<small>Away</small></div>
          </div>
          <div class="score-bottom">
            <span class="score-goals">Expected goals &nbsp; {goals[0]:.2f} – {goals[1]:.2f}</span>
            <span class="score-method">Based on historical match data</span>
          </div>
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



@st.fragment(run_every="5m")
def display_scores():
    st.subheader("Scores and results")
    st.write("Scores are retrieved from a separate football-data.org feed, so they can appear before the historical CSV updates. Final results can automatically settle your saved predictions when you open or refresh the tracker.")
    st.caption("The provider's free tier **delays scores**. This is not guaranteed real-time coverage. In-progress scores are never counted as final results.")
    st.caption("Football data provided by the Football-Data.org API.")
    if not SCORES_API_TOKEN:
        st.info("To enable the scores feed, create a football-data.org API key and add `FOOTBALL_DATA_API_TOKEN` to Streamlit Secrets. Your existing GitHub token and save PIN stay unchanged.")
        st.markdown("[Create a scores API account](https://www.football-data.org/client/register)")
        return
    if st.button("Check for newer scores", key="scores_tab_refresh"):
        cached_scores.clear()
    try:
        scoreboard = cached_scores(tuple(clubs), SCORES_API_TOKEN)
    except Exception as exc:
        st.error(f"Couldn't retrieve the scores feed: {exc}")
        st.caption("Historical predictions and your saved records are still available. The scores feed is optional.")
        return
    if scoreboard.empty:
        st.warning("The scores provider returned no Premier League matches.")
        return
    fetched = scoreboard.fetched_at_utc.iloc[0]
    confirmed = scoreboard.loc[scoreboard.is_finished]
    s1, s2, s3 = st.columns(3)
    s1.metric("Matches in provider feed", len(scoreboard))
    s2.metric("Confirmed finals", len(confirmed))
    s3.metric("Newest final (UK)", confirmed.kickoff_utc.max().astimezone(UK_ZONE).strftime("%d %b") if len(confirmed) else "—")
    st.caption(f"Retrieved from football-data.org at {fetched.strftime('%d %b %Y %H:%M')} UTC. "
               "Auto-checks about every five minutes while this app session is active, and when you press refresh. Actual update speed depends on the provider.")
    recent = recent_scoreboard(scoreboard)
    if recent.empty:
        st.info("No matches in the recent/coming 8-day / 7-day window.")
    else:
        st.dataframe(recent, hide_index=True, use_container_width=True)
    unmapped = scoreboard.loc[scoreboard.is_finished & (scoreboard.mapped_home.isna() | scoreboard.mapped_away.isna())]
    if len(unmapped):
        st.caption(f"{len(unmapped)} confirmed result(s) have names not yet matched to the model's club list; they are shown here but not used to settle saved predictions.")

with tab_scores:
    display_scores()



with tab_gw:
    st.subheader("Upcoming fixtures")
    st.write("Generate predictions for every scheduled Premier League match in an upcoming Friday–Thursday window, then save all unsaved forecasts to your existing prediction history with one confirmation.")
    st.caption("**Important:** The fixture source has dates and kickoff times, but no official Premier League gameweek numbers. These are **Friday–Thursday calendar windows**, not official Gameweek 1, 2, etc. The source can publish only a partial list of upcoming fixtures.")

    if st.button("Refresh upcoming fixtures", key="gameweek_refresh"):
        cached_upcoming.clear()
        st.rerun()

    try:
        gameweek_upcoming = cached_upcoming(tuple(clubs))
    except Exception as exc:
        gameweek_upcoming = []
        st.warning(f"Upcoming fixture feed is temporarily unavailable: {exc}")

    grouped_windows = group_fixture_windows(gameweek_upcoming)
    if not grouped_windows:
        st.info("No upcoming EPL fixtures were found in the provider's published window. Check again later or use **Prediction tracker** to enter an independently checked fixture manually.")
    else:
        chosen_window = st.selectbox(
            "Fixture window (UK dates)",
            options=list(range(len(grouped_windows))),
            format_func=lambda i: grouped_windows[i]["label"],
            key="selected_gameweek_window",
        )
        selected_window = grouped_windows[chosen_window]
        st.caption(f"Fixture feed currently lists {len(selected_window['fixtures'])} match(es) for this window. This is not necessarily the complete round; verify fixture times with the league or clubs.")

        try:
            saved_history = cached_records(REPO, GITHUB_TOKEN)
            history_available = True
        except Exception as exc:
            saved_history = []
            history_available = False
            st.warning(f"Couldn't check your saved history: {exc}. Saving is disabled until history can be checked for duplicates.")

        forecasts, skipped_fixtures = forecast_window(matches, selected_window["fixtures"], saved_history)
        if skipped_fixtures:
            with st.expander(f"{len(skipped_fixtures)} fixture(s) not forecast"):
                for warning in skipped_fixtures:
                    st.caption(warning)

        if not forecasts:
            st.info("No eligible future matches remain in this window.")
        else:
            prediction_table = forecast_table(forecasts)
            col_count, col_new, col_saved = st.columns(3)
            new_forecasts = [item["record"] for item in forecasts if not item["saved"]]
            col_count.metric("Fixture predictions", len(forecasts))
            col_new.metric("Not yet saved", len(new_forecasts))
            col_saved.metric("Previously saved", len(forecasts) - len(new_forecasts))
            st.dataframe(
                prediction_table, hide_index=True, use_container_width=True,
                column_config={"GitHub record": st.column_config.LinkColumn("Saved issue", display_text="Open")},
            )
            st.download_button(
                "Download this fixture window (CSV)",
                prediction_table.to_csv(index=False).encode("utf-8"),
                file_name=f"epl_predictions_{selected_window['key']}.csv", mime="text/csv",
                key="gameweek_csv",
            )
            st.caption("Predictions use only results dated **before each fixture's UK calendar day**, and use the preferred Attack × Defence model. A score is not a guarantee.")

            with st.expander("Inspect the attack vs defence reasoning for one match"):
                option = st.selectbox(
                    "Fixture", options=list(range(len(forecasts))),
                    format_func=lambda i: f"{forecasts[i]['record']['home']} vs {forecasts[i]['record']['away']}",
                    key="gameweek_match_detail",
                )
                one = forecasts[option]
                r = one["record"]
                st.write(f"**{r['home']} {r['score']} {r['away']}** · {100*r['p_exact']:.1f}% exact-score chance")
                st.write(f"Expected goals: **{r['expected_home']:.2f}–{r['expected_away']:.2f}**")
                st.write(f"Home win {100*r['p_home']:.1f}% · Draw {100*r['p_draw']:.1f}% · Away win {100*r['p_away']:.1f}%")
                st.dataframe(pd.DataFrame([
                    {"Metric": metric, "Relative rating": round(value, 2)}
                    for metric, value in one["ratings"].items()
                ]), hide_index=True, use_container_width=True)
                st.caption("Attack > 1 means above-average scoring; defensive vulnerability < 1 means fewer goals conceded than the relevant league average.")

            st.markdown("#### Lock all unsaved predictions before kickoff")
            if not GITHUB_TOKEN or not SAVE_PIN:
                st.info("To save forecasts, configure GITHUB_TOKEN and SAVE_PIN in your existing Streamlit Secrets. You do not need a new token.")
            elif not history_available:
                st.caption("Saving is unavailable until the saved-record check succeeds.")
            elif not new_forecasts:
                st.success("All eligible matches from this window already have saved predictions. View them in the Prediction tracker tab.")
            else:
                if len(new_forecasts) > 20:
                    st.warning("To protect against accidental bulk writes, this version saves at most 20 fixtures per operation.")
                st.write(f"This will create **{min(len(new_forecasts),20)} public GitHub Issue(s)**, one per fixture, with their current probabilities and model-data cutoff. Once saved, the app will not overwrite them.")
                with st.form("gameweek_save_all", clear_on_submit=True):
                    gameweek_pin = st.text_input("Your private save PIN", type="password", key="gameweek_pin")
                    agreed = st.checkbox("I checked the fixture dates and want to lock these predictions", key="gameweek_confirm")
                    do_save = st.form_submit_button("Save gameweek predictions", type="primary")
                    if do_save:
                        if not hmac.compare_digest(gameweek_pin, SAVE_PIN):
                            st.error("Incorrect PIN. Nothing was saved.")
                        elif not agreed:
                            st.warning("Confirm the fixture dates before saving.")
                        else:
                            try:
                                batch = save_predictions_batch(REPO, GITHUB_TOKEN, new_forecasts[:20])
                                cached_records.clear()
                                st.success(f"Saved {len(batch['saved'])} prediction(s); skipped {len(batch['skipped'])} already saved.")
                                for item in batch["saved"]:
                                    st.markdown(f"- [{item['fixture']}]({item['url']})")
                                if batch["failed"]:
                                    st.warning(f"{len(batch['failed'])} prediction(s) could not be saved; you can retry without duplicating successful records.")
                                    for item in batch["failed"]:
                                        st.caption(f"{item['fixture']}: {item['error']}")
                                st.caption("Use the Prediction tracker tab to view these forecasts and their eventual results. Refresh the page to update the 'Saved' column.")
                            except Exception as exc:
                                st.error(f"Unable to save the gameweek: {exc}")
    st.caption("Saving occurs only when you press the button — the app does not automatically predict or submit matches while nobody is using it. GitHub Issue records are public; your PIN and token remain private in Streamlit Secrets.")

with tab4:
    st.subheader("Save a pre-match prediction")
    st.write("Choose an upcoming Premier League fixture. Saving creates a timestamped public GitHub record. When a finished match is available from the scores feed, the tracker can show the result before the historical CSV updates.")

    try:
        upcoming = cached_upcoming(tuple(clubs))
    except Exception as exc:
        upcoming = []
        st.caption(f"Weekly fixture feed unavailable ({type(exc).__name__}). You can still enter the match date manually; manual dates are not independently verified.")

    choice_options = ["Upcoming fixtures", "Enter a matchup manually"] if upcoming else ["Enter a matchup manually"]
    choice = st.radio("How would you like to select a match?", choice_options, horizontal=True,
                      key="tracker_choice")

    if choice == "Upcoming fixtures":
        selected_index = st.selectbox("Upcoming Premier League fixtures", options=list(range(len(upcoming))),
                                      format_func=lambda i: upcoming[i]["label"])
        match = upcoming[selected_index]
        track_home, track_away = match["home"], match["away"]
        kickoff_utc = match["kickoff_utc"]
        fixture_source = match["source"]
    else:
        c1, c2 = st.columns(2)
        with c1:
            track_home = st.selectbox("Home club", clubs, index=clubs.index("Arsenal") if "Arsenal" in clubs else 0,
                                      key="tracker_home")
        with c2:
            track_away = st.selectbox("Away club", clubs, index=clubs.index("Chelsea") if "Chelsea" in clubs else min(1,len(clubs)-1),
                                      key="tracker_away")
        c3, c4 = st.columns(2)
        uk_now = datetime.now(UK_ZONE)
        with c3:
            kickoff_date = st.date_input("Kickoff date (UK)", value=uk_now.date() + timedelta(days=1),
                                         min_value=uk_now.date(), max_value=uk_now.date() + timedelta(days=365),
                                         key="tracker_date")
        with c4:
            kickoff_time = st.time_input("Kickoff time (UK)", value=time(15, 0), step=1800, key="tracker_time")
        kickoff_utc = datetime.combine(kickoff_date, kickoff_time, tzinfo=UK_ZONE).astimezone(timezone.utc)
        fixture_source = "Manual kickoff (unverified)"
        st.caption("Manual dates are your own entries, not confirmed fixtures. Double-check the official kickoff time. They are excluded from verified accuracy summaries.")

    valid_match = track_home != track_away and kickoff_utc > datetime.now(timezone.utc)
    if track_home == track_away:
        st.warning("Choose two different clubs.")
    elif kickoff_utc <= datetime.now(timezone.utc):
        st.warning("This kickoff has passed; only future matches can be saved.")
    else:
        local_kickoff = kickoff_utc.astimezone(UK_ZONE)
        st.caption(f"Kickoff: {local_kickoff.strftime('%A %d %B %Y, %H:%M')} UK · {kickoff_utc.strftime('%Y-%m-%d %H:%M')} UTC")
        # Conservative pre-match cutoff: exclude ALL results dated on or after
        # the upcoming fixture's UK calendar date, regardless of what the provider published.
        pre_matches = matches.loc[matches.Date.dt.date < local_kickoff.date()].copy()
        if len(pre_matches) < 30:
            st.warning("Not enough completed historical matches before this fixture.")
            valid_match = False
        elif len(matches.loc[(matches.HomeTeam == track_home) & (matches.AwayTeam == track_away) &
                               (matches.Date.dt.date == local_kickoff.date())]) > 0:
            st.warning("The provider already lists a completed match for these clubs on the selected day. Saving is blocked.")
            valid_match = False
        else:
            # For tracking we ONLY freeze the preferred Attack × Defence model.
            # The historical shots regression is deliberately not used here.
            prepared_before, history_before = cached_features(pre_matches)
            frozen = prediction_for_match(pre_matches, history_before, None, track_home, track_away)
            freeze_grid = frozen.distributions["Attack × Defence"]
            freeze_probs = outcome_probs(freeze_grid)
            freeze_score, freeze_exact = most_likely_score(freeze_grid)
            freeze_goals = frozen.goals["Attack × Defence"]
            a, b, c = st.columns(3)
            with a:
                st.metric("Most likely score", freeze_score, help=f"Exact-score probability {100*freeze_exact:.1f}%")
            with b:
                st.metric("Home win", f"{freeze_probs[0]*100:.1f}%")
            with c:
                st.metric("Draw / Away win", f"{freeze_probs[1]*100:.1f}% / {freeze_probs[2]*100:.1f}%")
            st.caption(f"Frozen estimates: {track_home} {freeze_goals[0]:.2f} – {freeze_goals[1]:.2f} {track_away} expected goals. "
                       f"Historical data ends on {pre_matches.Date.max().date()}; results from the fixture day are deliberately excluded.")

    st.markdown("#### Save this prediction")
    if not GITHUB_TOKEN or not SAVE_PIN:
        st.info("Saving isn't connected yet. Set GITHUB_TOKEN and SAVE_PIN in Streamlit Community Cloud → App settings → Secrets. You can still preview matches and view public history.")
    elif not valid_match:
        st.caption("Choose a valid future fixture to enable saving.")
    else:
        with st.form("save_forecast", clear_on_submit=True):
            entered_pin = st.text_input("Owner save PIN", type="password", help="Only you should know this PIN. Keep it out of GitHub and screenshots.")
            submitted = st.form_submit_button("Save prediction", type="primary")
            if submitted:
                if not hmac.compare_digest(entered_pin, SAVE_PIN):
                    st.error("Incorrect save PIN. Nothing was saved.")
                else:
                    try:
                        frozen_record = make_prediction_record(
                            track_home, track_away, kickoff_utc, fixture_source,
                            str(pre_matches.Date.max().date()), freeze_grid, freeze_goals,
                            freeze_score, freeze_exact, freeze_probs)
                        link = save_prediction(REPO, GITHUB_TOKEN, frozen_record)
                        cached_records.clear()
                        st.success("Prediction saved to GitHub! It will appear in the history below.")
                        st.markdown(f"[View your permanent GitHub prediction record]({link})")
                    except Exception as exc:
                        st.error(f"Couldn't save prediction: {exc}")
    st.caption("Your GitHub repository is public. Saved match predictions and their timestamps will be publicly visible. The secret token and PIN must only be entered in Streamlit Secrets.")

    st.divider()
    st.subheader("Saved predictions and actual results")
    if st.button("Refresh history and results"):
        cached_records.clear()
        cached_download.clear()
        cached_scores.clear()
        st.rerun()
    try:
        existing = cached_records(REPO, GITHUB_TOKEN)
    except Exception as exc:
        st.warning(f"Unable to load GitHub prediction history: {exc}")
        existing = []
    if not existing:
        st.info("No saved predictions found yet. After configuring the token and PIN, save your first upcoming match above.")
    else:
        records_table = table_for_predictions(existing, matches, provider_finals=provider_finals)
        if len(records_table) == 0:
            st.warning("Prediction records were found but couldn't be displayed.")
        else:
            status_filter = st.radio("Show", ["All", "Upcoming", "Final", "Awaiting result"], horizontal=True)
            visible = records_table if status_filter == "All" else records_table.loc[records_table.Status.eq(status_filter)]
            finals = records_table.loc[(records_table.Status == "Final") &
                                       (records_table.Source == "Football-Data upcoming fixtures CSV")]
            m1, m2, m3 = st.columns(3)
            m1.metric("Saved predictions", len(records_table))
            m2.metric("Verified finished", len(finals))
            m3.metric("Average Brier", f"{finals.Brier.mean():.3f}" if len(finals) else "—")
            if len(finals):
                st.caption(f"Among verified-feed forecasts: exact score hit rate {100*finals['Exact correct'].mean():.1f}% · "
                           f"Most probable outcome accuracy {100*finals['Outcome correct'].mean():.1f}% (n={len(finals)}). "
                           "These are descriptive results, not proof of future accuracy.")
            st.dataframe(visible, hide_index=True, use_container_width=True,
                         column_config={"GitHub record":st.column_config.LinkColumn("View record", display_text="Open")})
            st.download_button("Download prediction history (CSV)",
                               records_table.to_csv(index=False).encode("utf-8"),
                               file_name="epl_match_lab_tracked_predictions.csv", mime="text/csv")
            st.caption("Finished results are checked first against football-data.org (if configured), then against Football-Data.co.uk's historical results. A provider score is only accepted after it is marked FINISHED. Results may be delayed; matches rescheduled by more than three days require manual review. Predictions stay locked.")

with tab2:
    st.subheader("Team ratings")
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
    st.subheader("Model evaluation")
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

    **What this version cannot do:** reliably assess player swaps, coaching tactics, injuries, or real-time starting XIs. No betting claims or guarantees. The Prediction tracker tab can save timestamped pre-match forecasts and compare them with subsequently reported final scores.
    """)
    st.caption("Source: Football-Data.co.uk historical Premier League CSVs. Collected match results may have reporting delays. Refresh the dataset when returning to the site.")
