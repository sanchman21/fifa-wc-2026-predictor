# Deploying to Streamlit Community Cloud (free)

This app is ready to host for free on **Streamlit Community Cloud**. The match/goal data is
committed into `data/`, so there is **no network dependency at boot** — cold starts are fast
and reliable.

## One-time: push to GitHub

From this folder:

```bash
# (already done for you: git init + first commit)
git remote add origin https://github.com/<your-username>/fifa-wc-2026-predictor.git
git branch -M main
git push -u origin main
```

Create the empty `fifa-wc-2026-predictor` repo on GitHub first (public is simplest;
private also works on Streamlit Cloud). If you use the GitHub CLI:

```bash
gh repo create fifa-wc-2026-predictor --public --source . --push
```

## Deploy

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Fill in:
   - **Repository:** `<your-username>/fifa-wc-2026-predictor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Advanced settings → Python version: 3.12** (matches development; avoids wheel issues).
5. Click **Deploy**. First build installs `requirements.txt` (~1–2 min). You get a public
   `https://<something>.streamlit.app` URL.

That's it. To update the live app, just `git push` — Streamlit Cloud redeploys automatically.

## What to expect on the free tier

- **Sleeps when idle.** After a period of no traffic the app suspends; the next visitor wakes
  it in ~30 seconds. Normal for the free tier.
- **First load per wake:** ~5–10 s while it computes Elo and trains the weights (then cached).
  Changing the simulation count/seed in the sidebar only re-runs the fast Monte-Carlo.
- **Memory:** peaks comfortably under the ~1 GB free-tier limit.

## Refreshing the data later

The committed CSVs are a snapshot. To pull newer results/goals:

```bash
python -m src.data.fetch_data --force   # re-downloads into data/
git add data/ && git commit -m "Refresh match data" && git push
```

## Notes

- Entry point is `app.py` at the repo root; it imports the package in `src/`.
- `.streamlit/config.toml` sets the theme and is picked up automatically.
- Data is from [martj42/international_results](https://github.com/martj42/international_results)
  (CC BY-NC-SA 4.0) — keep the app free / non-commercial and the attribution intact.
