"""EPL score-prediction calculations. Based on the Lesson 10 Colab notebook.

Data source: https://www.football-data.co.uk/englandm.php
No target match result or in-match stats are used to predict that match.
"""
from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
MAX_GOALS = 12
HISTORY_DAYS = 730
VENUE_GAMES = 20
VENUE_PRIOR_GAMES = 7
WINDOW = 5
FEATURE_PRIOR_GAMES = 3
FALLBACK = {"gf": 1.4, "ga": 1.4, "sot_for": 4.5, "sot_against": 4.5}
GOAL_COLS = ["Own_GF5", "Own_GA5", "Opp_GF5", "Opp_GA5"]
SHOT_COLS = ["Own_SOT5", "Own_SOTAgainst5", "Opp_SOT5", "Opp_SOTAgainst5"]
FEATURES = GOAL_COLS + SHOT_COLS
SHOT_INPUTS = ["Team", "Opponent", "Home"] + FEATURES
BLEND_ATTACK_WEIGHT = 0.25  # Chosen among blend candidates using 2023–24 in Lesson 10


def season_label(code: str) -> str:
    return f"20{code[:2]}–{code[2:]}"


def season_from_date(date: pd.Timestamp) -> str:
    """Infer a Premier League season for user-uploaded Football-Data CSVs."""
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def clean_season(raw: pd.DataFrame, season_code: str | None = None) -> pd.DataFrame:
    essential = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [col for col in essential if col not in raw.columns]
    if missing:
        raise ValueError(f"Not a Football-Data match CSV; missing: {', '.join(missing)}")
    data = raw.copy()
    for c in ("HST", "AST"):
        if c not in data.columns:
            data[c] = np.nan
    df = data[essential + ["HST", "AST"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    for c in ("FTHG", "FTAG", "HST", "AST"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=essential)
    df = df.loc[(df.FTHG >= 0) & (df.FTAG >= 0)].copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    if season_code is None:
        df["SeasonCode"] = df["Date"].map(season_from_date)
    else:
        df["SeasonCode"] = season_code
    return df


def combine_seasons(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        raise ValueError("No completed Premier League matches were loaded.")
    matches = pd.concat(parts, ignore_index=True)
    matches = matches.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def download_matches() -> tuple[pd.DataFrame, list[str]]:
    """Download real completed EPL fixtures. Network requests are cached by the UI."""
    parts: list[pd.DataFrame] = []
    warnings: list[str] = []
    for code in SEASONS:
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        try:
            response = requests.get(url, timeout=18, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            raw = pd.read_csv(io.BytesIO(response.content), low_memory=False, on_bad_lines="skip")
            data = clean_season(raw, code)
            if len(data) == 0:
                raise ValueError("no completed games")
            parts.append(data)
        except Exception as exc:
            warnings.append(f"{season_label(code)}: {type(exc).__name__}: {str(exc)[:110]}")
    return combine_seasons(parts), warnings


def smooth_mean(values: pd.Series, league_mean: float, k: int = VENUE_PRIOR_GAMES) -> float:
    a = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float((a.sum() + k * league_mean) / (len(a) + k))


def attack_defence_rating(past: pd.DataFrame, home: str, away: str):
    """Exactly the venue-specific 20-game, 7-game-prior model of Lesson 10."""
    if len(past):
        lg_home = max(float(past["FTHG"].mean()), 0.5)
        lg_away = max(float(past["FTAG"].mean()), 0.5)
    else:
        lg_home, lg_away = 1.5, 1.2
    home_games = past.loc[past.HomeTeam.eq(home)].tail(VENUE_GAMES)
    away_games = past.loc[past.AwayTeam.eq(away)].tail(VENUE_GAMES)
    h_attack = smooth_mean(home_games.FTHG, lg_home)
    h_concede = smooth_mean(home_games.FTAG, lg_away)
    a_attack = smooth_mean(away_games.FTAG, lg_away)
    a_concede = smooth_mean(away_games.FTHG, lg_home)
    ratings = {
        "Home attack": h_attack / lg_home,
        "Away defence vulnerability": a_concede / lg_home,
        "Away attack": a_attack / lg_away,
        "Home defence vulnerability": h_concede / lg_away,
    }
    goal_home = lg_home * ratings["Home attack"] * ratings["Away defence vulnerability"]
    goal_away = lg_away * ratings["Away attack"] * ratings["Home defence vulnerability"]
    return np.clip([goal_home, goal_away], 0.15, 4.5), ratings


def poisson_grid(home_goals: float, away_goals: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(g, float(home_goals)), poisson.pmf(g, float(away_goals)))
    return grid / grid.sum()  # Truncated tail, normalized


def outcome_probs(grid: np.ndarray) -> np.ndarray:
    # Row index = home goals, column index = away goals.
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def most_likely_score(grid: np.ndarray) -> tuple[str, float]:
    home, away = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return f"{home}–{away}", float(grid[home, away])


def snapshot(history: dict, team: str, asof: pd.Timestamp):
    relevant = [record for date, record in history.get(team, [])
                if date < asof and date >= asof - pd.Timedelta(days=365)]
    recent = relevant[-WINDOW:]
    result = {}
    for feature, prior in FALLBACK.items():
        values = [float(r[feature]) for r in recent if pd.notna(r[feature])]
        result[feature] = (sum(values) + FEATURE_PRIOR_GAMES * prior) / (len(values) + FEATURE_PRIOR_GAMES)
    return result


def side_features(own: dict, opp: dict) -> dict:
    return {
        "Own_GF5": own["gf"], "Own_GA5": own["ga"],
        "Opp_GF5": opp["gf"], "Opp_GA5": opp["ga"],
        "Own_SOT5": own["sot_for"], "Own_SOTAgainst5": own["sot_against"],
        "Opp_SOT5": opp["sot_for"], "Opp_SOTAgainst5": opp["sot_against"],
    }


def prematch_records(matches: pd.DataFrame):
    """Feature histories use only earlier calendar days, never match-day results."""
    history = defaultdict(list)
    rows = []
    for date, today in matches.groupby("Date", sort=True):
        for match in today.itertuples(index=False):
            h = snapshot(history, match.HomeTeam, date)
            a = snapshot(history, match.AwayTeam, date)
            record = match._asdict()
            record.update({f"H_{k}": v for k, v in side_features(h, a).items()})
            record.update({f"A_{k}": v for k, v in side_features(a, h).items()})
            rows.append(record)
        for match in today.itertuples(index=False):
            history[match.HomeTeam].append((date, {
                "gf": match.FTHG, "ga": match.FTAG,
                "sot_for": match.HST, "sot_against": match.AST,
            }))
            history[match.AwayTeam].append((date, {
                "gf": match.FTAG, "ga": match.FTHG,
                "sot_for": match.AST, "sot_against": match.HST,
            }))
    return pd.DataFrame(rows), history


def goal_rows(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in fixtures.itertuples(index=False):
        for side in ("H", "A"):
            home = side == "H"
            row = {
                "Team": m.HomeTeam if home else m.AwayTeam,
                "Opponent": m.AwayTeam if home else m.HomeTeam,
                "Home": int(home),
                "ActualGoals": int(m.FTHG if home else m.FTAG),
            }
            row.update({name: getattr(m, f"{side}_{name}") for name in FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def fit_shots_model(fixtures: pd.DataFrame):
    if len(fixtures) < 100:
        raise ValueError("Need at least 100 prior matches to train the shots model.")
    goals = goal_rows(fixtures)
    preprocess = ColumnTransformer([
        ("clubs", OneHotEncoder(handle_unknown="ignore"), ["Team", "Opponent"]),
        ("numeric", StandardScaler(), ["Home"] + FEATURES),
    ])
    model = Pipeline([
        ("features", preprocess),
        ("poisson", PoissonRegressor(alpha=0.05, max_iter=800)),
    ])
    model.fit(goals[SHOT_INPUTS], goals["ActualGoals"])
    return model


def predict_shots(model, fixtures: pd.DataFrame):
    goals = goal_rows(fixtures)
    predictions = model.predict(goals[SHOT_INPUTS]).reshape(-1, 2)
    return np.clip(predictions, 0.15, 4.5)


def latest_shots_features(history: dict, home: str, away: str, asof: pd.Timestamp) -> pd.DataFrame:
    h = snapshot(history, home, asof)
    a = snapshot(history, away, asof)
    row = {"Date": asof, "HomeTeam": home, "AwayTeam": away, "FTHG": 0, "FTAG": 0}
    row.update({f"H_{k}": v for k, v in side_features(h, a).items()})
    row.update({f"A_{k}": v for k, v in side_features(a, h).items()})
    return pd.DataFrame([row])


@dataclass
class Predictions:
    distributions: dict[str, np.ndarray]
    goals: dict[str, np.ndarray]
    ratings: dict[str, float]
    blend_attack_weight: float


def prediction_for_match(
    matches: pd.DataFrame,
    history: dict,
    shot_model,
    home: str,
    away: str,
    blend_weight: float = BLEND_ATTACK_WEIGHT,
) -> Predictions:
    if home == away:
        raise ValueError("Choose two different teams.")
    cutoff = matches.Date.max() + pd.Timedelta(days=1)
    past = matches.loc[matches.Date.ge(cutoff - pd.Timedelta(days=HISTORY_DAYS))]
    a_goals, ratings = attack_defence_rating(past, home, away)
    a_grid = poisson_grid(*a_goals)
    grids = {"Attack × Defence": a_grid}
    means = {"Attack × Defence": a_goals}
    if shot_model is not None:
        b_goals = predict_shots(shot_model, latest_shots_features(history, home, away, cutoff))[0]
        b_grid = poisson_grid(*b_goals)
        grids["Recent goals + shots"] = b_grid
        means["Recent goals + shots"] = b_goals
        grids["25% attack / 75% shots blend"] = blend_weight * a_grid + (1 - blend_weight) * b_grid
        means["25% attack / 75% shots blend"] = blend_weight * a_goals + (1 - blend_weight) * b_goals
    return Predictions(grids, means, ratings, blend_weight)


def results_for_prediction(prediction: Predictions) -> pd.DataFrame:
    data = []
    for name, grid in prediction.distributions.items():
        p = outcome_probs(grid)
        score, exact = most_likely_score(grid)
        means = prediction.goals[name]
        data.append({
            "Model": name,
            "Home expected goals": round(float(means[0]), 2),
            "Away expected goals": round(float(means[1]), 2),
            "Most likely score": score,
            "Exact score %": round(100 * exact, 1),
            "Home win %": round(100 * p[0], 1),
            "Draw %": round(100 * p[1], 1),
            "Away win %": round(100 * p[2], 1),
        })
    return pd.DataFrame(data)


def club_ratings(matches: pd.DataFrame, clubs: list[str]):
    cutoff = matches.Date.max() + pd.Timedelta(days=1)
    past = matches.loc[matches.Date.ge(cutoff - pd.Timedelta(days=HISTORY_DAYS))]
    rows = []
    for team in clubs:
        _, r = attack_defence_rating(past, team, team)
        rows.append({
            "Team": team,
            "Home attack": r["Home attack"],
            "Home defence vulnerability": r["Home defence vulnerability"],
            "Away attack": r["Away attack"],
            "Away defence vulnerability": r["Away defence vulnerability"],
        })
    return pd.DataFrame(rows)


def latest_clubs(matches: pd.DataFrame):
    # Current season is available when at least one match has been completed.
    last = matches.sort_values("Date").iloc[-1]["SeasonCode"]
    curr = matches.loc[matches.SeasonCode.eq(last)]
    return sorted(set(curr.HomeTeam) | set(curr.AwayTeam)), last


def model_training_data(prepared: pd.DataFrame):
    # Same as Lesson 10: fit the regression through completed 2025–26,
    # never using current partial 2026–27 results to fit coefficients.
    return prepared.loc[prepared.SeasonCode.isin(SEASONS[:-1])].copy()


def historical_brier(matches: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Recompute Lesson 10 benchmarks on identical same-season test fixtures.

    Does not use these familiar seasons as a new independent test.
    """
    if not {"2425", "2526"}.issubset(set(matches.SeasonCode)):
        raise ValueError("Need both 2024–25 and 2025–26 to run the historical benchmark.")
    by_date = []
    for date, day in matches.groupby("Date", sort=True):
        past = matches.loc[(matches.Date < date) & (matches.Date >= date - pd.Timedelta(days=HISTORY_DAYS))]
        for m in day.itertuples(index=False):
            by_date.append(attack_defence_rating(past, m.HomeTeam, m.AwayTeam)[0])
    attack = np.asarray(by_date)
    if len(attack) != len(prepared):
        raise ValueError("Fixture/feature alignment error.")
    rows = []
    for code in ("2425", "2526"):
        test = prepared.loc[prepared.SeasonCode.eq(code)]
        before = SEASONS[:SEASONS.index(code)]
        train = prepared.loc[prepared.SeasonCode.isin(before)]
        shot = fit_shots_model(train)
        a_goals = attack[test.index.to_numpy()]
        b_goals = predict_shots(shot, test)
        actual = np.where(test.FTHG > test.FTAG, 0, np.where(test.FTHG == test.FTAG, 1, 2))
        a = np.array([outcome_probs(poisson_grid(*g)) for g in a_goals])
        b = np.array([outcome_probs(poisson_grid(*g)) for g in b_goals])
        for name, probs in {
            "Attack × Defence": a,
            "Recent goals + shots": b,
            "25% attack / 75% shots blend": BLEND_ATTACK_WEIGHT * a + (1-BLEND_ATTACK_WEIGHT) * b,
        }.items():
            brier = float(np.mean(np.sum((probs - np.eye(3)[actual])**2, axis=1)))
            rows.append({"Season": season_label(code), "Model": name, "Brier": round(brier, 4), "Matches": len(test)})
    return pd.DataFrame(rows)
