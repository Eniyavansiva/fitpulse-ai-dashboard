# FitPulse AI — Personal Fitness Intelligence Dashboard

FitPulse AI is a Streamlit portfolio project that combines Fitbit analytics with a LangGraph-powered health-data chat assistant.

## Features

- Fixed historical Fitbit dataset period: March 12, 2016 – May 12, 2016
- Individual named-user analysis or an **All users** cohort analytics view
- Wellness Score hero, attention flags, cohort averages, and segmentation counts
- Seven KPI cards including streak tracking; BMI stays in individual user view
- Top-five all-users leaderboard ranked by Wellness Score
- Five cached, data-grounded LLM insights with deterministic fallback coverage
- Right-side AI chat panel with dynamic question chips and structured, cited answers
- Gemini 2.5 Flash with Groq Llama 3.3 70B fallback
- LangGraph intent classification, data-context retrieval, structured output, and ten-message memory

## LangGraph architecture

```text
[START]
   ↓
[classify_intent]
   ↓ (metric_query | trend_query | comparison | advice | general)
[fetch_context]
   ↓
[generate_response]
   ↓
[format_output]
   ↓
[END]
```

## Setup

```bash
python -m venv fitpulse_env

# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\fitpulse_env\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Add `GOOGLE_API_KEY` (or the existing `Gemini_API_Key`) to `.env`. Optionally add `GROQ_API_KEY` to enable the fallback model.

## Data source

Fitbit Fitness Tracker Data — Mobius, CC0 Public Domain.
