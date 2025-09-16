# SoundScope (Clean Restart)

End‑to‑end ML pipeline to predict Spotify song popularity ("hit" vs not) using audio features.

## What you get (in this zip)
- Ready-to-run Python project with **scripts**, **src** modules, **requirements**, and **VS Code settings**.
- Robust Spotify client that accepts **playlist URLs or IDs**, sets **market=US**, and paginates safely.
- Data collection → model training → metrics (R², MAE, accuracy) → **Top 10 feature importance** plot.
- Setup scripts for macOS/Linux and Windows.

---

## 0) Prereqs
- Python 3.10+
- Spotify Developer app (Client ID & Client Secret)

> You already made an app named **SoundScope** — great. You only need to paste the ID/Secret into `.env`.

---

## 1) One‑liner setup (macOS/Linux)
```bash
bash setup.sh
```
This will:
- create `.venv`
- install requirements
- create `.env` from `.env.example` (you then put your secrets in it)

**Windows (PowerShell):**
```powershell
./setup.ps1
```

If you prefer manual steps, see the bottom of this README.

---

## 2) Put secrets in .env
After setup, open `.env` and fill in:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

---

## 3) Quick sanity check (auth + API)
```bash
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

python sanity_check.py
```
If it prints a track name, your creds are working.

---

## 4) Collect data
Examples:

**Editorial playlists (IDs or full URLs both work):**
```bash
python -m scripts.collect_data --playlist 37i9dQZF1DXcBWIGoYBM5M 37i9dQZF1DWXRqgorJj26U --limit 1000 --out data/tracks.csv
# or:
python -m scripts.collect_data --playlist https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U --limit 1000 --out data/tracks.csv
```

**Search query (e.g., pop 2024–2025):**
```bash
python -m scripts.collect_data --query "year:2024-2025 genre:pop" --limit 800 --out data/tracks.csv
```

**New releases:**
```bash
python -m scripts.collect_data --new-releases --limit 500 --out data/tracks.csv
```

---

## 5) Train + evaluate
```bash
python -m scripts.train_models --csv data/tracks.csv --outdir outputs --hit-threshold 75
```
Outputs:
- `outputs/regression_model.joblib`
- `outputs/classifier_model.joblib`
- `outputs/feature_importance.png`
- `outputs/metrics.json`

---

## 6) Manual setup (if not using scripts)
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env   # (or create it manually on Windows)
# put your Client ID/Secret in .env
```

---

## Notes
- Uses Client Credentials flow via `spotipy` (sufficient for popularity + audio features).
- Classification label: **is_hit = 1 if popularity >= hit_threshold** (default 75).
- The Spotify client handles pagination, accepts URLs or IDs, and sets `market="US"` to reduce 404s.
