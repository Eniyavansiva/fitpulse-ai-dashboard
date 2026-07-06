# FitPulse AI - Fitness Intelligence Dashboard

FitPulse AI is a Streamlit portfolio project that combines Fitbit cohort analytics, individual fitness dashboards, AI-generated insights, and a LangGraph-powered chat assistant.

Live deployment: deployed on Streamlit Community Cloud from the GitHub `main` branch.

## Features

- Fixed historical Fitbit dataset period: March 12, 2016 to May 12, 2016
- Individual named-user analysis and an **All users** cohort analytics view
- Wellness Score hero metric using steps, sleep, active minutes, and resting heart rate
- Seven KPI cards including streak tracking
- Cohort attention flags and activity/recovery segmentation
- Top-five all-users leaderboard ranked by Wellness Score
- Plotly charts for activity, calories, sleep, heart rate, weight, and BMI
- Data-grounded AI insights with deterministic fallback behavior
- Right-side FitPulse AI chat panel with dynamic suggested questions
- Gemini 2.5 Flash support with Groq fallback

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Plotly
- LangGraph
- LangChain
- Gemini / Groq LLM providers
- Pytest
- GitHub Actions
- Docker-ready setup

## LangGraph Flow

```text
[START]
   |
[classify_intent]
   |
[fetch_context]
   |
[generate_response]
   |
[format_output]
   |
[END]
```

## Run Locally

From the project root:

```powershell
cd "C:\Eni data\PYTHON\GenAI_Projects\FitPulse AI Dashboard + LangGraph Health Agent"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\fitpulse_env\Scripts\Activate.ps1
python -m streamlit run app.py
```

If activation is blocked or Streamlit is not found, run directly with the virtual environment Python:

```powershell
.\fitpulse_env\Scripts\python.exe -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Environment Variables

For local development, keep API keys in `.env`:

```text
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_optional_key_here
```

The project also supports the earlier key name:

```text
Gemini_API_Key=your_key_here
```

Never commit `.env`.

For Streamlit Cloud, add the same key in app secrets:

```toml
GOOGLE_API_KEY = "your_key_here"
```

## Tests

Run tests locally:

```powershell
python -m pytest tests/ -v
```

The tests currently check:

- daily activity CSV loads with required columns
- sleep CSV loads with required columns
- every activity user ID has a friendly name in `USER_NAME_MAP`
- core loaded datasets expose `UserName`, not raw IDs
- heart-rate source data has expected columns
- `get_user_filtered()` handles missing and narrow-date user data correctly
- `compute_wellness_score()` stays within expected scoring ranges

These are data/helper tests. They do not perform screenshot-based UI testing.

## CI Pipeline

The GitHub Actions workflow is:

```text
.github/workflows/ci.yml
```

It runs automatically on:

- pushes to `main`
- pushes to `fix-*` branches
- pushes to `phase2-*` branches
- pull requests targeting `main`

The CI job creates a fresh Ubuntu runner, installs Python 3.11, installs `requirements.txt`, and runs:

```bash
pytest tests/ -v
```

LLM calls are disabled in CI with:

```text
FITPULSE_DISABLE_LLM=1
```

## Deployment

The current public deployment uses Streamlit Community Cloud.

Deployment flow:

```text
Merge PR into main
   |
Streamlit Cloud detects main changed
   |
Streamlit installs requirements.txt
   |
Streamlit runs app.py
   |
Public app URL updates
```

There is no separate `cd.yml` file right now because Streamlit Cloud handles deployment automatically from the `main` branch.

## Docker

Docker files are present:

```text
Dockerfile
.dockerignore
```

Docker packages the app, Python runtime, dependencies, and startup command into an image. A running instance of that image is a container.

Build image:

```powershell
docker build -t fitpulse-ai .
```

Run container:

```powershell
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key_here fitpulse-ai
```

Then open:

```text
http://localhost:8501
```

Docker has not been locally tested yet because Docker was not installed or available in the terminal when checked.

## Branch Workflow

Use one branch per feature or fix:

```powershell
git checkout main
git pull origin main
git checkout -b feature-my-change
```

Before pushing:

```powershell
python -m pytest tests/ -v
git add .
git commit -m "feat: describe change"
git push origin feature-my-change
```

Create a Pull Request into `main`. After CI passes, merge it. Streamlit Cloud redeploys from `main`.

## Monitoring Status

Monitoring is not fully implemented yet.

Currently available:

- Streamlit Cloud logs
- GitHub Actions CI results

Recommended next work:

- add Python structured logging for app startup, selected user, chatbot calls, and errors
- add chatbot response-time logging
- add UptimeRobot for public URL uptime checks
- add a CI badge to this README
- add dependency/security scanning later

## Data Source

Fitbit Fitness Tracker Data - Mobius, CC0 Public Domain.
