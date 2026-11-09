"""Gameweek calendar groups and frozen, pre-kickoff attack-vs-defence predictions.

A Fri–Thu window is an approximate scheduling group, NOT an official EPL gameweek.
The fixture CSV does not supply a Premier League gameweek number.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict

import pandas as pd

from football_model import (
    HISTORY_DAYS, attack_defence_rating, most_likely_score,
    outcome_probs, poisson_grid,
)
from prediction_tracker import make_prediction_record, same_fixture

UK = ZoneInfo("Europe/London")


def friday_of_week(day: date) -> date:
    """Group a scheduled date into a Friday–Thursday UK-time window."""
    return day - timedelta(days=(day.weekday() - 4) % 7)


def group_fixture_windows(fixtures: list[dict]) -> list[dict]:
    """Return windows ordered chronologically; does not invent official round numbers."""
    groups = defaultdict(list)
    for fixture in fixtures:
        kickoff = fixture["kickoff_utc"].astimezone(UK)
        groups[friday_of_week(kickoff.date())].append(fixture)
    out = []
    for start in sorted(groups):
        end = start + timedelta(days=6)
        games = sorted(groups[start], key=lambda f: (f["kickoff_utc"], f["home"]))
        out.append({"key": start.isoformat(), "start": start, "end": end,
                    "label": f"{start:%d %b}–{end:%d %b %Y} (UK) · {len(games)} fixtures",
                    "fixtures": games})
    return out


def forecast_fixture(matches: pd.DataFrame, fixture: dict,
                     now: datetime | None = None) -> dict:
    """Predict one fixture using only pre-match calendar days.

    Historical match records are not used if their dates fall on the fixture
    day, or later. The same venue model as the existing app is used.
    """
    now = now or datetime.now(timezone.utc)
    kickoff = fixture["kickoff_utc"].astimezone(timezone.utc)
    if kickoff <= now:
        raise ValueError("The match has already kicked off.")
    uk_day = kickoff.astimezone(UK).date()
    home, away = fixture["home"], fixture["away"]
    if home == away:
        raise ValueError("Choose two different teams.")
    if matches.empty:
        raise ValueError("No historical match results have been loaded.")

    dated = matches.loc[matches.Date.dt.date < uk_day]
    if len(dated) < 30:
        raise ValueError("Insufficient results before the kickoff date.")
    finished_same_day = matches.loc[
        matches.HomeTeam.eq(home) & matches.AwayTeam.eq(away) &
        matches.Date.dt.date.eq(uk_day)
    ]
    if not finished_same_day.empty:
        raise ValueError("The results dataset already contains this fixture on its scheduled date.")

    earliest = pd.Timestamp(uk_day - timedelta(days=HISTORY_DAYS))
    historical = dated.loc[dated.Date.ge(earliest)]
    goals, ratings = attack_defence_rating(historical, home, away)
    grid = poisson_grid(*goals)
    probs = outcome_probs(grid)
    score, exact = most_likely_score(grid)
    cutoff = dated.Date.max().date().isoformat()
    record = make_prediction_record(
        home, away, kickoff, fixture["source"], cutoff,
        grid, goals, score, exact, probs,
    )
    return {"record": record, "ratings": ratings}


def forecast_window(matches: pd.DataFrame, fixtures: list[dict],
                    existing: list[dict] | None = None,
                    now: datetime | None = None) -> tuple[list[dict], list[str]]:
    """Return display rows with frozen records, plus explanations of skipped fixtures."""
    records = []
    warnings = []
    existing = existing or []
    for fixture in fixtures:
        try:
            result = forecast_fixture(matches, fixture, now)
            record = result["record"]
            previous = next((r for r in existing if same_fixture(r, record)), None)
            records.append({"record": record, "ratings": result["ratings"],
                            "saved": previous is not None,
                            "issue_url": previous.get("issue_url", "") if previous else ""})
        except Exception as exc:
            warnings.append(f"{fixture['home']} vs {fixture['away']}: {exc}")
    return records, warnings


def forecast_table(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        record = item["record"]
        uk = datetime.fromisoformat(record["kickoff_utc"].replace("Z", "+00:00")).astimezone(UK)
        rows.append({
            "Kickoff (UK)": uk.strftime("%a %d %b %H:%M"),
            "Home": record["home"], "Away": record["away"],
            "Predicted score": record["score"],
            "Exact score chance": round(100 * record["p_exact"], 1),
            "Home win %": round(100 * record["p_home"], 1),
            "Draw %": round(100 * record["p_draw"], 1),
            "Away win %": round(100 * record["p_away"], 1),
            "Home expected goals": round(record["expected_home"], 2),
            "Away expected goals": round(record["expected_away"], 2),
            "Saved": "Saved" if item["saved"] else "Not saved",
            "Data through": record["data_cutoff_utc_date"],
            "GitHub record": item["issue_url"],
        })
    return pd.DataFrame(rows)
