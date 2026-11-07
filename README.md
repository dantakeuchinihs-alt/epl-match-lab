# EPL Match Lab — Lesson 12: Prediction Tracker

This upgrade **keeps the existing Attack × Defence model** and adds a **Prediction tracker** tab to [EPL Match Lab](https://epl-match-lab.streamlit.app/).

### New features

- Pick an upcoming EPL fixture from Football-Data.co.uk's weekly upcoming-fixtures CSV (when available), or manually enter the scheduled UK kickoff date and time.
- Preview the preferred model's pre-match expected goals, most likely exact score, and probabilities. On the tracker, **match-day results are excluded** to avoid information leakage.
- **Lock a prediction before kickoff**. The app saves the full probabilities and the data cutoff as a **timestamped GitHub Issue**, rather than relying on Streamlit's temporary storage.
- See all stored predictions, actual full-time scores when Football-Data has them, and a per-match Brier score. A summary evaluates provider-scheduled fixtures only; manually entered fixtures are marked unverified and excluded from the headline accuracy summary.
- Download all saved predictions as a CSV.

**Important:** Results do not update in the background. They are reconciled when you revisit or refresh the site. Football-Data's fixture CSV covers *a short upcoming window*, not necessarily the full season. A postponed match more than 3 days from its scheduled date remains pending for review.

## A. Update your existing GitHub repository

Your existing repo is: `dantakeuchinihs-alt/epl-match-lab`.

1. Download and unzip `EPL_Match_Lab_Prediction_Tracker_Update.zip`.
2. Open your GitHub repository's **Code** page and choose **Add file → Upload files**.
3. Upload these files from the extracted ZIP into the **top level** (not inside an extra folder):
   - `app.py` — updated app, replacing the existing version
   - `prediction_tracker.py` — new
   - `fixtures.py` — new
   - `README.md` — updated instructions, replaces old
   - `football_model.py` — unchanged model, safe to overwrite with same version
   - `requirements.txt` — same dependencies as before
4. Commit the changes on the `main` branch. If GitHub warns about a duplicate filename, use the repo's file editor to replace it, or GitHub Desktop to push the unzipped files.
5. Streamlit Community Cloud normally picks up code commits automatically. Reopen your existing site and look for a **Prediction tracker** tab.

Your existing URL is unchanged. You do **not** need to deploy a second app.

## B. Enable secure, permanent saving

A Streamlit app's local storage is not reliably persistent. The updated app uses **GitHub Issues** for each saved forecast. This is simpler and safer than giving it permission to edit the website code.

1. In GitHub, ensure the **Issues** feature is enabled for your repo. It is usually enabled automatically (look for the **Issues** tab).
2. Create a **fine-grained personal access token** at [GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens](https://github.com/settings/personal-access-tokens/new).
   - **Resource owner:** the owner of `epl-match-lab`.
   - **Repository access:** **Only select repositories** → `epl-match-lab`.
   - **Repository permissions:** **Issues → Read and write**. Leave Contents read-only; no code-writing permissions are needed.
   - Set a sensible expiration and regenerate the token when it expires.
   - Copy the token when GitHub generates it; GitHub may show it only once.
3. Visit [Streamlit Community Cloud](https://share.streamlit.io/). On your app's ⋯ menu choose **Settings → Secrets**. Alternatively, on the deployed app click **Manage app**, and from there open its app settings.
4. Paste the following, replacing the placeholders with your actual values:

```toml
GITHUB_REPO = "dantakeuchinihs-alt/epl-match-lab"
GITHUB_TOKEN = "PASTE_YOUR_FINE_GRAINED_GITHUB_TOKEN_HERE"
SAVE_PIN = "CHOOSE_A_PRIVATE_LONG_RANDOM_PIN_OR_PASSWORD"
```

5. Click **Save**. Return to the site and refresh. In the **Prediction tracker** tab, enter your private save PIN when locking a prediction.

**NEVER paste your token or PIN into a GitHub file, screenshot, or chat message.** Keep them only in Streamlit's **Secrets** settings. The site is public, so saved prediction Issues will be public too. Visitors can read the history but cannot save through the app without the PIN. The GitHub token is limited to Issues in this repository, rather than permission to change the app code.

Official references: [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [Streamlit app settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings), [GitHub Issues API](https://docs.github.com/en/rest/issues/issues).

## C. Save and review your first prediction

1. Open https://epl-match-lab.streamlit.app/ → **Prediction tracker**.
2. Select a fixture or enter its UK kickoff date/time manually.
3. Confirm the most likely score and 1X2 probabilities, then enter your private PIN.
4. Click **Save prediction permanently** and follow its GitHub record link.
5. Open the site later; your history will remain in GitHub, even if Streamlit restarts.
6. Once the final result is available in Football-Data's season results, click **Refresh prediction history and scores** to check the outcome and Brier score.

**Important statistical distinction:** the displayed before-kickoff model does not use that match's result. Saved probabilities are locked at issue creation. The score comparison happens only afterward. A Brier score is the sum of squared errors for the three win/draw/loss probabilities, with **lower better**.

### Limitations and integrity notes

- Upcoming fixtures from Football-Data's `fixtures.csv` can be unavailable or outdated. Confirm official kickoff times; the manually entered date is not independently verified.
- The tracker deliberately excludes same-day completed matches from the training history, which means a saved prediction can differ slightly from the ordinary Match Predictor tab.
- Only full-time Premier League results are automatically reconciled. Correct teams and roughly correct kickoff date are required; postponed games over 3 days stay pending. No live score tracking.
- For a proper out-of-sample evaluation, save future forecasts **before** kickoffs. Avoid changing the baseline model after reviewing those outcomes, unless you're starting a new versioned experiment.
- The site does not collect users' personal data beyond public GitHub issue metadata. There is no per-user login; only the owner with the PIN can save through the Streamlit form.

## Files

- `app.py`: updated Streamlit interface including the new tab.
- `prediction_tracker.py`: GitHub issue storage, locked predictions, result matching, scoring.
- `fixtures.py`: weekly EPL fixture CSV retrieval and UK timezone conversion.
- `football_model.py`: unchanged Lesson 10 forecasting method.
- `requirements.txt`: same existing dependencies.
- `tests/`: synthetic unit tests for the model and tracker.
