# EPL Match Lab — Gameweek Predictor (Website Update)

This package **updates your existing website** at https://epl-match-lab.streamlit.app/. It does not create a separate app and **does not replace your working prediction tracker**.

## What's new

Open the **🗓️ Gameweek predictions** tab to:

1. Load upcoming Premier League fixtures from [Football-Data's fixtures CSV](https://www.football-data.co.uk/fixtures.csv).
2. Pick a **Friday–Thursday UK-date window** and see projected scores, win/draw/loss probabilities, expected goals, and data cutoffs for the listed fixtures.
3. Inspect how the **Attack × Defence** ratings produced each forecast.
4. Download the selected window's predictions as CSV.
5. Enter your existing private `SAVE_PIN`, confirm the fixtures, and **save every unsaved prediction in one action**. Each forecast gets a separate timestamped GitHub Issue.
6. See saved predictions and compare their results later in the existing **📅 Prediction tracker** tab.

**Gameweek naming:** The free fixture CSV has kickoff dates/times, **not official Premier League gameweek numbers**. The app groups fixtures into Friday–Thursday periods and labels them *fixture windows*. Cup postponements, midweek games, and partial provider coverage mean these periods may differ from official gameweeks.

**Important:** Nothing is automatically saved in the background. Fixtures and predictions appear when you open the app; **you choose when to lock them before kickoff**. The upcoming fixture feed sometimes covers only the next few matches or weekends, rather than a complete round.

## Upload update to GitHub (beginner steps)

1. Download `EPL_Match_Lab_Gameweek_Predictor_Update.zip` and **unzip** it.
2. Go to **https://github.com/dantakeuchinihs-alt/epl-match-lab** and open the **Code** tab.
3. Choose **Add file → Upload files** (or use GitHub's file editor to overwrite existing files if your browser refuses duplicates).
4. Upload the following **extracted files at the top level** of the repository:
   - `app.py` — updated Streamlit app, **replaces** the old `app.py`
   - `gameweek.py` — **new** file (you must upload this)
   - `prediction_tracker.py` — updated storage with safe batch saving, replaces old file
   - `fixtures.py` — same upcoming-fixtures module
   - `football_model.py` — same prediction model
   - `requirements.txt` — dependencies, unchanged
   - `README.md` — new instructions, optional but helpful
5. Scroll down and click **Commit changes**, committing to branch `main`.
6. Refresh **https://epl-match-lab.streamlit.app/**. Streamlit Community Cloud should deploy the new code from GitHub automatically. You should see **🗓️ Gameweek predictions** among the tabs.

**Do not upload the ZIP itself.** Upload the individual extracted files. Keep `app.py` and `gameweek.py` directly at the repository's top level. If the website shows `ModuleNotFoundError: No module named 'gameweek'`, you forgot `gameweek.py` or placed it in a subfolder.

### Do I need to create another GitHub token?

**No.** You previously added the secrets in Streamlit; this update reuses the same values:

```toml
GITHUB_REPO = "dantakeuchinihs-alt/epl-match-lab"
GITHUB_TOKEN = "YOUR_EXISTING_GITHUB_TOKEN"
SAVE_PIN = "YOUR_EXISTING_PRIVATE_PASSWORD"
```

This block is **an example only**. Your real token and PIN must stay in **Streamlit Community Cloud → App settings → Secrets**. Never add a secrets file or token to GitHub. GitHub fine-grained token permission: **Issues: Read and write** for this repository. The repository's **Issues** feature must be enabled. Note that your GitHub Issue predictions are publicly visible because the repository is public.

## How to use it before a gameweek

1. Open your EPL Match Lab website.
2. In the sidebar click **↻ Refresh downloaded match data** if you want the latest finished results.
3. Open **🗓️ Gameweek predictions** and click **↻ Refresh upcoming fixtures** to request the latest schedule.
4. Pick one Friday–Thursday fixture window and inspect its predictions. Verify dates/kickoff times using an official team/league schedule if necessary.
5. Enter the private save PIN, check the confirmation box, and click **🔒 Save all unsaved gameweek predictions**.
6. Open **📅 Prediction tracker** to see locked predictions. After games finish, click **↻ Refresh prediction history and scores** to fetch new results and Brier scores.

If a previous operation saved 5 of 10 matches before a network failure, you can retry. The app will skip the 5 already recorded and attempt the other 5. Saving is limited to **20 fixtures per click** to prevent accidental bulk writes.

## Accuracy and data integrity

- Predictions use the **same preferred Attack × Defence model** as your website. No new or untested model has been inserted.
- Each fixture prediction excludes **all results dated on or after that fixture's UK calendar day**; only past dated results can influence it.
- GitHub's Issue creation timestamp is the audit record of when a prediction was saved. A forecast cannot count as verified pre-match if the Issue was created at or after kickoff.
- Predictions are **not retrospectively edited** by the website after they are saved. Result checking occurs in the original tracker tab, using Football-Data's completed-result feed.
- If a fixture is rescheduled by more than three days, result matching can stay pending. It is not silently paired to another game.
- The website is **not a real-time match-odds system** and does not account for live injuries, starting XIs or tactical adjustments.
- Current 2026–27 season is configured in `football_model.py`. For a future 2027–28 season, a further season-code update will be necessary.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Website interface with the new Gameweek predictions tab |
| `gameweek.py` | Calendar grouping and pre-match batch prediction helpers |
| `prediction_tracker.py` | GitHub Issue storage, history, result scoring, batch saving |
| `fixtures.py` | Football-Data upcoming fixtures and UK kickoff conversion |
| `football_model.py` | Existing Attack × Defence / shots comparison models |
| `requirements.txt` | Python dependencies |
| `tests/` | Offline synthetic tests (`pytest`) |

### Troubleshooting

**No fixtures available:** Upcoming feed may be empty, stale, or incomplete. Try **Refresh upcoming fixtures**. The existing tracker still supports manually entered matches.

**Token or permission error:** Confirm `GITHUB_REPO`, `GITHUB_TOKEN`, `SAVE_PIN` in Streamlit Secrets; token must be unexpired and include Issues: Read/write on the repository.

**Missing module:** Upload every `.py` file from this update to the repo root, particularly `gameweek.py` and `prediction_tracker.py`.

**Model data not current:** Click **Refresh downloaded match data**; football-data.co.uk may not have published the latest final scores yet.

**Why aren't results automatically showing?** Results are reconciled when you open or refresh the app; there is no background job running while the app is asleep.

Sources: [Football-Data fixture CSV](https://www.football-data.co.uk/fixtures.csv), [Streamlit deployment updates](https://docs.streamlit.io/deploy/streamlit-community-cloud), [GitHub Issues API](https://docs.github.com/en/rest/issues/issues).
