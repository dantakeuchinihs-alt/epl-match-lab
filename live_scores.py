"""Optional up-to-date Premier League scores from the documented football-data.org API.

On free plans, the provider warns that scores can be delayed. This module never
uses unconfirmed in-play scores as final results and never changes the model's
historical Football-Data.co.uk training input.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import re
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
UK_ZONE = ZoneInfo("Europe/London")
FINAL_STATUSES = {"FINISHED"}
IN_PROGRESS_STATUSES = {"IN_PLAY", "PAUSED", "LIVE", "HALFTIME", "EXTRA_TIME", "PENALTY_SHOOTOUT"}

# Football-Data.co.uk names can differ from football-data.org names.
ALIASES = {
    "arsenal": "Arsenal", "aston villa": "Aston Villa", "bournemouth": "Bournemouth",
    "brighton and hove albion": "Brighton", "brighton": "Brighton",
    "brentford": "Brentford", "burnley": "Burnley", "cardiff city": "Cardiff",
    "chelsea": "Chelsea", "coventry city": "Coventry", "crystal palace": "Crystal Palace",
    "everton": "Everton", "fulham": "Fulham", "hull city": "Hull",
    "huddersfield town": "Huddersfield", "ipswich town": "Ipswich",
    "leeds united": "Leeds", "leicester city": "Leicester", "liverpool": "Liverpool",
    "luton town": "Luton", "manchester city": "Man City",
    "manchester united": "Man United", "newcastle united": "Newcastle",
    "norwich city": "Norwich", "nottingham forest": "Nott'm Forest",
    "nottingham": "Nott'm Forest", "sheffield united": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds", "southampton": "Southampton",
    "sunderland": "Sunderland", "swansea city": "Swansea",
    "tottenham hotspur": "Tottenham", "tottenham": "Tottenham",
    "watford": "Watford", "west bromwich albion": "West Brom",
    "west ham united": "West Ham", "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves", "wigan athletic": "Wigan",
}


def _clean_team(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = name.lower().replace("&", " and ").replace("’", "'")
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = " ".join(name.split())
    if name.startswith("afc "):
        name = name[4:]
    name = re.sub(r" (?:afc|fc)$", "", name)
    return name.strip()


def club_lookup(name: str, available_clubs: list[str]) -> str | None:
    """Conservative club-name reconciliation. Never silently guess ambiguous names."""
    norm = _clean_team(name)
    by_normal = {_clean_team(n): n for n in available_clubs}
    if norm in by_normal:
        return by_normal[norm]
    canonical = ALIASES.get(norm)
    if canonical and canonical in available_clubs:
        return canonical
    return None


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError, TypeError):
        return None


def parse_score_response(body: dict, clubs: list[str], fetched_at: datetime | None = None) -> pd.DataFrame:
    """Parse football-data.org match list, keeping raw names for visible unmatched teams."""
    if not isinstance(body, dict) or not isinstance(body.get("matches"), list):
        raise ValueError("Scores provider returned an unexpected response format.")
    when = fetched_at or datetime.now(timezone.utc)
    rows = []
    for m in body["matches"]:
        if not isinstance(m, dict):
            continue
        kickoff = _parse_utc(m.get("utcDate"))
        if kickoff is None:
            continue
        home_info, away_info = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        raw_home = home_info.get("name") or home_info.get("shortName") or ""
        raw_away = away_info.get("name") or away_info.get("shortName") or ""
        home = club_lookup(raw_home, clubs)
        away = club_lookup(raw_away, clubs)
        status = str(m.get("status") or "UNKNOWN").upper()
        full = (m.get("score") or {}).get("fullTime") or {}
        h, a = full.get("home"), full.get("away")
        valid_score = (isinstance(h, int) and not isinstance(h, bool) and h >= 0 and
                       isinstance(a, int) and not isinstance(a, bool) and a >= 0)
        finished = status in FINAL_STATUSES and valid_score
        rows.append({
            "match_id": str(m.get("id", "")),
            "kickoff_utc": kickoff, "match_date_uk": kickoff.astimezone(UK_ZONE).date(),
            "home": home or raw_home, "away": away or raw_away,
            "mapped_home": home, "mapped_away": away,
            "home_goals": h if valid_score else None,
            "away_goals": a if valid_score else None,
            "status": status, "is_finished": finished,
            "last_updated_utc": _parse_utc(m.get("lastUpdated")),
            "fetched_at_utc": when,
            "source": "football-data.org",
        })
    if not rows:
        return pd.DataFrame(columns=["match_id", "kickoff_utc", "match_date_uk", "home", "away",
                                     "mapped_home", "mapped_away", "home_goals", "away_goals", "status",
                                     "is_finished", "last_updated_utc", "fetched_at_utc", "source"])
    return pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)


def download_scoreboard(api_token: str, clubs: list[str], session=requests, now: datetime | None = None) -> pd.DataFrame:
    if not api_token or not api_token.strip():
        raise ValueError("Add FOOTBALL_DATA_API_TOKEN to Streamlit Secrets.")
    response = session.get(API_URL, headers={
        "X-Auth-Token": api_token.strip(), "Accept": "application/json",
        "User-Agent": "EPL-Match-Lab-Student-Project",
    }, timeout=20)
    if response.status_code in (401, 403):
        raise RuntimeError("Scores provider rejected this API key or plan. Check FOOTBALL_DATA_API_TOKEN and Premier League access.")
    if response.status_code == 429:
        raise RuntimeError("Scores provider rate limit reached. Try again later.")
    response.raise_for_status()
    return parse_score_response(response.json(), clubs, now)


def completed_score_dataframe(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """Convert only confirmed final scores to the tracker's historical column shape."""
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if scoreboard is None or scoreboard.empty:
        return pd.DataFrame(columns=cols)
    mask = scoreboard["is_finished"] & scoreboard["mapped_home"].notna() & scoreboard["mapped_away"].notna()
    finals = scoreboard.loc[mask].copy()
    if finals.empty:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame({
        "Date": pd.to_datetime(finals["match_date_uk"]),
        "HomeTeam": finals["mapped_home"].values,
        "AwayTeam": finals["mapped_away"].values,
        "FTHG": finals["home_goals"].astype(int).values,
        "FTAG": finals["away_goals"].astype(int).values,
    }).reset_index(drop=True)


def recent_scoreboard(scoreboard: pd.DataFrame, now: datetime | None = None, days_back: int = 8, days_forward: int = 7) -> pd.DataFrame:
    if scoreboard is None or scoreboard.empty:
        return pd.DataFrame()
    now = now or datetime.now(timezone.utc)
    low, high = now - timedelta(days=days_back), now + timedelta(days=days_forward)
    items = scoreboard.loc[scoreboard.kickoff_utc.between(low, high)].copy()
    if items.empty:
        return pd.DataFrame()
    def describe(row):
        if row.is_finished:
            return "Final"
        if row.status in IN_PROGRESS_STATUSES:
            return "In progress (score may change)"
        if row.status in {"POSTPONED", "SUSPENDED", "CANCELLED"}:
            return row.status.title()
        return "Scheduled / pending"
    items["Kickoff (UK)"] = items.kickoff_utc.map(lambda d: d.astimezone(UK_ZONE).strftime("%d %b %H:%M"))
    items["Match"] = items.home + " vs " + items.away
    items["Score"] = items.apply(lambda r: f"{int(r.home_goals)}–{int(r.away_goals)}"
                                  if (r.is_finished or r.status in IN_PROGRESS_STATUSES)
                                  and pd.notna(r.home_goals) and pd.notna(r.away_goals) else "—", axis=1)
    items["Status"] = items.apply(describe, axis=1)
    return items[["Kickoff (UK)", "Match", "Score", "Status"]].reset_index(drop=True)
