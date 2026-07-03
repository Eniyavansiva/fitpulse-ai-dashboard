"""Cached LLM insight generation grounded in compact FitPulse dashboard facts."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


# ── Structured insight generation ──────────────────────────────────────────

class InsightResponse(BaseModel):
    """Define the five concise narrative insights requested from an AI provider."""

    insights: list[str] = Field(min_length=5, max_length=5)


def _get_insight_models() -> list[object]:
    """Return Gemini first and Groq second to provide a quota-aware provider fallback."""
    load_dotenv(override=False)
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("Gemini_API_Key")
    groq_api_key = os.getenv("GROQ_API_KEY")
    models: list[object] = []
    if google_api_key:
        models.append(
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_api_key,
                temperature=0.35,
            )
        )
    if groq_api_key:
        models.append(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_api_key,
                temperature=0.35,
            )
        )
    return models


def _normalize_insights(insights: list[str]) -> list[str]:
    """Trim provider output into five non-empty single-paragraph dashboard insights."""
    cleaned = [" ".join(str(insight).split()) for insight in insights if str(insight).strip()]
    return cleaned[:5]


def _fallback_insights(fallback_insights: tuple[str, ...]) -> list[str]:
    """Return exactly five deterministic insights when no model response is available."""
    fallback = _normalize_insights(list(fallback_insights))
    while len(fallback) < 5:
        fallback.append("Use the dashboard trends to set one realistic movement or recovery goal for the next week.")
    return fallback[:5]


@st.cache_data(show_spinner=False, ttl=3600)
def generate_llm_insights(
    scope: str,
    fact_context: str,
    fallback_insights: tuple[str, ...],
) -> list[str]:
    """Generate five grounded fitness insights, returning deterministic facts if providers are unavailable."""
    if os.getenv("FITPULSE_DISABLE_LLM") == "1":
        return _fallback_insights(fallback_insights)

    audience_instruction = (
        "This is a cohort analysis. Never use 'you' or 'your'; use 'the cohort', 'participants', or 'users'. "
        "Reference averages and segment counts exactly as supplied."
        if scope == "all_users"
        else "This is an individual analysis. Address the selected Fitbit profile directly and respectfully."
    )
    prompt = f"""
You are FitPulse AI, a thoughtful fitness-data analyst. Create exactly five distinct, polished dashboard insights.
{audience_instruction}
Each insight must be one or two sentences, cite a relevant supplied number when available, and balance strengths,
opportunities, practical next steps, and recovery considerations. Keep health comments general and non-diagnostic.
Do not invent values, diagnoses, or causal claims.

Verified dashboard facts:
{fact_context}
"""
    for model in _get_insight_models():
        try:
            response = model.with_structured_output(InsightResponse).invoke(prompt)
            parsed = response if isinstance(response, InsightResponse) else InsightResponse.model_validate(response)
            insights = _normalize_insights(parsed.insights)
            if len(insights) == 5:
                return insights
        except Exception:
            continue
    return _fallback_insights(fallback_insights)
