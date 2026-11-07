"""Get Premier League upcoming fixtures for the tracking selector, using provider weekly CSV."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import io
import pandas as pd
import requests

UK = ZoneInfo("Europe/London")
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def parse_fixtures(csv_bytes: bytes, clubs: list[str], now_utc: datetime | None = None) -> list[dict]:
    now_utc = now_utc or datetime.now(timezone.utc)
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False, on_bad_lines="skip")
    needed = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not needed.issubset(df.columns):
        raise ValueError("Upcoming fixture feed does not contain required fields.")
    if "Time" not in df.columns:
        return []  # No trustworthy time for a pre-kickoff lock
    fixtures = []
    known = set(clubs)
    for row in df.loc[df["Div"].eq("E0")].itertuples(index=False):
        home, away = str(row.HomeTeam).strip(), str(row.AwayTeam).strip()
        if home not in known or away not in known:
            continue
        date = pd.to_datetime(str(row.Date), dayfirst=True, errors="coerce")
        if pd.isna(date) or pd.isna(row.Time):
            continue
        try:
            hh, mm = [int(x) for x in str(row.Time).strip().split(":")[:2]]
            london = datetime(date.year, date.month, date.day, hh, mm, tzinfo=UK)
            kickoff = london.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
        if kickoff <= now_utc:
            continue
        fixtures.append({"home":home, "away":away, "kickoff_utc":kickoff,
                         "label":f"{london.strftime('%a %d %b, %H:%M')} UK — {home} vs {away}",
                         "source":"Football-Data upcoming fixtures CSV"})
    fixtures.sort(key=lambda x: x["kickoff_utc"])
    return fixtures


def download_fixtures(clubs: list[str], session=requests) -> list[dict]:
    response = session.get(FIXTURES_URL, timeout=18, headers={"User-Agent":"Mozilla/5.0"})
    response.raise_for_status()
    return parse_fixtures(response.content, clubs)
