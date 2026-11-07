# EPL Match Lab — Lesson 11

A Streamlit website that predicts Premier League scores using attack versus defence ratings. This is a beginner-friendly continuation of the Lesson 10 Google Colab notebook.

## What the app does

- Select any two clubs from the latest Premier League season with results in the downloaded dataset.
- View expected goals, win/draw/loss probabilities, the most likely exact score, and an interactive probability heatmap.
- Compare the preferred Attack × Defence approach to a recent-goals-and-shots Poisson regression and a 25/75 blend.
- Inspect home/away attack ratings and defensive vulnerability for all available clubs.
- Recompute familiar historical Brier tests for 2024–25 and 2025–26 (not fresh independent tests).
- Download predictions or team ratings as CSVs.

Data: [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php) downloadable Premier League result files. The app downloads seasons 2018–19 to 2026–27 (when available) and caches the download for six hours.

**Important:** Predictive, not causal; no live fixtures, confirmed lineups, injury updates, or guarantees. Historical out-of-sample scores may have been repeatedly inspected in this project and aren't fresh validation.

## Easiest way to publish (no coding needed)

1. Create or sign into a [GitHub](https://github.com/) account.
2. Create a **new repository**, for example `epl-match-lab`.
3. Click **Add file → Upload files** and upload the following **extracted** files from this folder:
   - `app.py`
   - `football_model.py`
   - `requirements.txt`
   - `README.md`
   You can optionally upload `.streamlit/config.toml` to apply the included dark theme (GitHub may hide dot folders in some file pickers; the app works without it).
4. Commit the files to the main branch.
5. Sign into [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub, select **Create app** (existing repository), choose your repo and main branch, and set the **Main file path** to `app.py`.
6. Deploy. Streamlit will install packages from `requirements.txt`. Open the resulting `*.streamlit.app` URL.

Community Cloud instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Run on your own computer (optional)

Install Python 3.11 or later, unzip this folder, then open Terminal in the extracted folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The site should open in your browser. Keep Terminal open while using it; press **Ctrl+C** to stop.

## Troubleshooting

- **Couldn't load data?** Refresh in the sidebar, or download multiple `E0.csv` season files from Football-Data.co.uk and upload them with the sidebar file uploader.
- **New teams not listed?** A club appears after it has a completed match in the latest downloaded season. Check the latest download date and refresh.
- **No recent-goals-and-shots model?** You'll need enough past completed matches to fit the comparison model; the Attack × Defence model can still be used.
- **Old data?** Use the sidebar Refresh button. Data updates depend on the provider.
- **App asleep?** The free hosting service may pause inactive apps; reopening may start the app again.

## Model details

Preferred model is unchanged from Lesson 10: 730-day rolling league context, up to 20 most recent home/away games for each team, with seven matches of shrinkage toward the league venue average. Team home attack is matched with away opponent defence; away attack is matched with home opponent defence. The estimated goals are converted to scoreline probabilities using independent Poisson scoring distributions, normalized over 0–12 goals per team.

The recent-goals-and-shots model trains a regularized (`alpha=0.05`) Poisson regression on completed seasons up through 2025–26, using team names, opponents, venue and last-five-match form features. Same-day match statistics and test-match results aren't used as features for their own fixture. A 25% attack / 75% recent-shots blend combines the full predicted score distributions; it was the best *blend candidate*, not the best overall model, on historical 2023–24 validation. Attack × Defence was best on average across the 2024–25 and 2025–26 historical tests examined so far.

## Project files

- `app.py`: the interactive Streamlit website.
- `football_model.py`: data processing, predictions, probability calculations, and backtesting.
- `requirements.txt`: Python packages Streamlit installs.
- `.streamlit/config.toml`: optional dark color scheme.
- `tests/test_model.py`: small unit tests using **synthetic** fixtures. No invented results are shown in the website.
