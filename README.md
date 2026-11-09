# Match Lab — Editorial Redesign

A visual refresh for the existing [EPL Match Lab](https://epl-match-lab.streamlit.app/) Streamlit application. This is **not** a new project: it keeps the original score predictor, matchweek forecasts, saved GitHub predictions, result checking, team ratings, and model evidence.

## What changed

- Editorial typography and quieter copy — no emojis in the navigation, buttons, or dashboard headings.
- Dark graphite background, warm ivory type, and a restrained champagne accent. Flat surfaces and thin rules replace neon green, bright red, gradients, and heavy rounded cards.
- A dedicated scoreline panel with team names, home/away labels, exact-score probability, and expected goals.
- Consistent colour palette on the prediction bars and score-probability heatmap.
- Shorter, cleaner navigation: **Predict / Scores / Fixtures / Tracker / Team ratings / Method**.
- A custom monogram favicon, new sidebar identity, and responsive layout.
- Separate theme and stylesheet files for easier future design adjustments.

## Install (beginner-friendly)

1. Download and **extract** the ZIP file.
2. Open your **existing** GitHub repository: https://github.com/dantakeuchinihs-alt/epl-match-lab
3. Go to **Add file → Upload files** and drag the extracted **files and folders** into the repository **root**, replacing the old versions. The folder structure must be:

   ```text
   epl-match-lab/
     app.py
     football_model.py
     fixtures.py
     gameweek.py
     live_scores.py
     prediction_tracker.py
     requirements.txt
     README.md
     assets/
       site.css
       favicon.png
     .streamlit/
       config.toml
   ```

   **Important:** GitHub's browser file uploader may not upload hidden `.streamlit` folders reliably on some setups. If you can't upload the folder, use **Add file → Create new file**, type `.streamlit/config.toml` in the file name box, and paste the contents of `config.toml` from this archive. Commit the file.

4. Click **Commit changes** on branch `main`.
5. Open https://epl-match-lab.streamlit.app/ and check the six navigation sections. Streamlit Community Cloud should redeploy automatically from the repository.

**You do not need a new GitHub or football-data API token.** All your Streamlit Secrets remain in the existing app settings. Do not upload `.streamlit/secrets.toml` or any tokens.

## If the new colours don't appear

- Confirm `.streamlit/config.toml` is present (it defines the base theme). The CSS in `assets/site.css` enhances that theme.
- Confirm both `assets/site.css` and `assets/favicon.png` are present in the root-level `assets` folder.
- Use your browser's hard refresh after Streamlit redeploys.
- If it reports a missing asset, the extracted folder structure was not preserved; re-upload the folder contents as shown above.

## Safety and accuracy

This redesign intentionally **does not change** any of the prediction formulas, selected model weights, caching intervals, fixture logic, GitHub Issue storage, PIN protection or match result matching. It does not add bets or in-play analysis. The football-data.org scores feed may be delayed on the free plan.

## Files

- `app.py` — all changes to layout, text labels, and charts
- `assets/site.css` — design tokens, typography, navigation, cards, controls, mobile layout
- `assets/favicon.png` — original Match Lab monogram
- `.streamlit/config.toml` — compatible Streamlit theme
- remaining `.py` files — original working logic, copied without functional changes
- `tests/` — original model, tracker, scores and fixture tests

Run locally (optional):

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
