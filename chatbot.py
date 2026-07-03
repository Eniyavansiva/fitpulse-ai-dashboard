"""LangGraph-powered, data-aware chat assistant for the FitPulse AI dashboard."""

from __future__ import annotations

import os
from html import escape
from typing import Annotated, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils.data_loader import DataLoadError, load_all_data
from utils.metrics import calculate_cohort_analysis, calculate_kpis, generate_insights


# ── State and structured response ──────────────────────────────────────────

ALL_USERS_OPTION = "All users"
DATA_PERIOD_LABEL = "Mar 12, 2016 – May 12, 2016"
ADVICE_CONTEXT = (
    "Fitness guidance must remain general and educational. Do not diagnose conditions, prescribe treatment, "
    "or replace a qualified healthcare professional."
)


class ChatResponse(BaseModel):
    """Define the structured response returned by the FitPulse language-model node."""

    answer: str
    data_points_used: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    follow_up_suggestions: list[str] = Field(default_factory=list)
    disclaimer: str | None = None


class FitState(TypedDict):
    """Represent the state passed through the FitPulse LangGraph workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    context: str
    intent: str
    chat_response: dict[str, Any]
    formatted_response: str


# ── Intent and data context ────────────────────────────────────────────────

def _latest_user_question(messages: list[BaseMessage]) -> str:
    """Return the most recent human message content from the graph history."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def classify_intent(state: FitState) -> dict[str, str]:
    """Classify a dashboard question as metric, trend, comparison, advice, or general."""
    question = _latest_user_question(state["messages"]).lower()
    if any(keyword in question for keyword in ("compare", "versus", "vs", "benchmark", "other users")):
        intent = "comparison"
    elif any(keyword in question for keyword in ("should", "recommend", "advice", "improve", "help me", "what can i do")):
        intent = "advice"
    elif any(keyword in question for keyword in ("trend", "change", "changed", "increase", "decrease", "drop", "over time")):
        intent = "trend_query"
    elif any(keyword in question for keyword in ("average", "how many", "how much", "steps", "calories", "sleep", "bmi", "heart rate")):
        intent = "metric_query"
    else:
        intent = "general"
    return {"intent": intent}


def _route_by_intent(state: FitState) -> str:
    """Return the explicit route label for LangGraph's conditional intent edge."""
    return state["intent"]


def _filter_for_user(data: dict[str, pd.DataFrame], user_id: str) -> dict[str, pd.DataFrame]:
    """Return all source data for the cohort or a single selected Fitbit user."""
    if user_id == ALL_USERS_OPTION:
        return {name: dataframe.copy() for name, dataframe in data.items()}
    return {
        name: dataframe.loc[dataframe["UserName"].astype(str) == str(user_id)].copy()
        for name, dataframe in data.items()
    }


def _recent_average(dataframe: pd.DataFrame, date_column: str, value_column: str) -> str | None:
    """Summarize the most recent seven dates of one metric for chat context."""
    if dataframe.empty or date_column not in dataframe or value_column not in dataframe:
        return None
    daily = (
        dataframe.dropna(subset=[date_column, value_column])
        .groupby(date_column)[value_column]
        .mean()
        .sort_index()
    )
    if daily.empty:
        return None
    recent = daily.tail(7)
    return f"Recent seven-record average {value_column}: {recent.mean():,.1f}"


def _format_context_value(value: float, suffix: str = "", decimals: int = 0) -> str:
    """Format a numeric cohort fact safely before it is added to an LLM context string."""
    if pd.isna(value):
        return "not available"
    return f"{value:,.{decimals}f}{suffix}"


def _build_data_context(filtered_data: dict[str, pd.DataFrame], user_id: str, intent: str) -> tuple[str, list[str]]:
    """Build a concise, intent-aware factual context and citations from Fitbit data."""
    scope = "all tracked users combined" if user_id == ALL_USERS_OPTION else f"user {user_id}"
    kpis = calculate_kpis(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["heartrate"],
        filtered_data["weight"],
    )
    data_points = [f"{kpi['title']}: {kpi['value_label']}" for kpi in kpis]
    context_parts = [
        f"Scope: {scope}.",
        f"Source period: {DATA_PERIOD_LABEL}.",
        "Primary KPIs: " + "; ".join(data_points) + ".",
    ]
    if user_id == ALL_USERS_OPTION:
        cohort_analysis = calculate_cohort_analysis(
            filtered_data["activity"],
            filtered_data["sleep"],
            filtered_data["heartrate"],
        )
        activity_segments = "; ".join(
            f"{segment}: {count} users" for segment, count in cohort_analysis["activity_segments"].items()
        )
        sleep_segments = "; ".join(
            f"{segment}: {count} users" for segment, count in cohort_analysis["sleep_segments"].items()
        )
        cohort_points = [
            f"Activity profiles: {cohort_analysis['participant_count']}",
            f"Average heart rate: {_format_context_value(cohort_analysis['average_heart_rate'], ' bpm')}",
            f"Activity segments: {activity_segments}",
            f"Recovery segments: {sleep_segments}",
        ]
        context_parts.append("Cohort analysis: " + "; ".join(cohort_points) + ".")
        data_points.extend(cohort_points)

    if intent in {"trend_query", "comparison", "advice"}:
        trend_metrics = [
            _recent_average(filtered_data["activity"], "ActivityDate", "TotalSteps"),
            _recent_average(filtered_data["activity"], "ActivityDate", "Calories"),
            _recent_average(filtered_data["sleep"], "SleepDay", "SleepEfficiency"),
        ]
        trend_points = [point for point in trend_metrics if point]
        if trend_points:
            context_parts.append("Trend context: " + "; ".join(trend_points) + ".")
            data_points.extend(trend_points)

    insights = generate_insights(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["calories"],
        audience_label="Across all users, " if user_id == ALL_USERS_OPTION else "",
    )
    context_parts.append("Calculated insights: " + " ".join(insights))
    data_points.extend(insights)
    if intent == "advice":
        context_parts.append(ADVICE_CONTEXT)
    return "\n".join(context_parts), data_points[:10]


def fetch_context(state: FitState) -> dict[str, str]:
    """Fetch the selected dashboard data slice and append it to the visible dashboard summary."""
    try:
        data = load_all_data()
        filtered_data = _filter_for_user(data, state["user_id"])
        data_context, _ = _build_data_context(filtered_data, state["user_id"], state["intent"])
        return {"context": f"{state['context']}\n\nDetailed data context:\n{data_context}"}
    except DataLoadError as error:
        return {"context": f"{state['context']}\n\nData loading error: {error}"}


# ── Model providers ────────────────────────────────────────────────────────

def _get_chat_models() -> list[BaseChatModel]:
    """Initialize Gemini first and Groq second so Groq can handle Gemini quota failures."""
    load_dotenv(override=False)
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("Gemini_API_Key")
    groq_api_key = os.getenv("GROQ_API_KEY")
    models: list[BaseChatModel] = []

    if google_api_key:
        models.append(
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_api_key,
                temperature=0.2,
            )
        )
    if groq_api_key:
        models.append(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_api_key,
                temperature=0.2,
            )
        )
    return models


def _local_fallback_response(state: FitState) -> ChatResponse:
    """Return a transparent dashboard-only response when no configured model can answer."""
    question = _latest_user_question(state["messages"])
    response_points = [line.removeprefix("Primary KPIs: ") for line in state["context"].splitlines() if "KPIs:" in line]
    return ChatResponse(
        answer=(
            "I can surface the currently selected Fitbit data, but no available AI provider completed this response. "
            f"Your question was: “{question}”. Configure GOOGLE_API_KEY or GROQ_API_KEY to enable generated analysis."
        ),
        data_points_used=response_points or ["Dashboard context is available for the active profile."],
        confidence="low",
        follow_up_suggestions=[
            "What are the current dashboard KPIs?",
            "How do the recent step and sleep trends look?",
        ],
        disclaimer="Fitness data is informational and is not medical advice.",
    )


def _generate_structured_response(models: list[BaseChatModel], state: FitState) -> ChatResponse | None:
    """Ask each configured provider for a Pydantic response, falling back on provider failure."""
    audience_instruction = (
        "The scope is all users: describe the cohort in third person and never call it 'you' or 'your'."
        if state["user_id"] == ALL_USERS_OPTION
        else "The scope is one selected Fitbit profile: address the user directly and respectfully."
    )
    system_prompt = (
        "You are FitPulse AI, a concise fitness-data analyst. Answer only from the supplied dashboard context. "
        "Every answer must reference at least one exact number from the Dashboard Context. "
        "If the provided context lacks the needed number, say the dashboard does not include enough data. "
        "State uncertainty clearly, and never make medical diagnoses. "
        f"For advice, keep it general and include an informational-not-medical disclaimer. {audience_instruction}\n\n"
        f"Dashboard Context:\n{state['context']}"
    )
    history = state["messages"][-10:]
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]
    for model in models:
        try:
            response = model.with_structured_output(ChatResponse).invoke(messages)
            if isinstance(response, ChatResponse):
                return response
            return ChatResponse.model_validate(response)
        except Exception:
            continue
    return None


def _generate_response(models: list[BaseChatModel], state: FitState) -> dict[str, Any]:
    """Generate a structured chat response and preserve the assistant answer in graph memory."""
    response = _generate_structured_response(models, state) or _local_fallback_response(state)
    return {
        "chat_response": response.model_dump(),
        "messages": [AIMessage(content=response.answer)],
    }


def format_output(state: FitState) -> dict[str, str]:
    """Format the structured response into readable Streamlit markdown with data citations."""
    response = ChatResponse.model_validate(state["chat_response"])
    data_points = "\n".join(f"- {point}" for point in response.data_points_used) or "- No specific points available"
    suggestions = "\n".join(f"- {suggestion}" for suggestion in response.follow_up_suggestions)
    sections = [
        response.answer,
        f"**Data points used**\n{data_points}",
        f"**Confidence:** {response.confidence}",
    ]
    if suggestions:
        sections.append(f"**Try next**\n{suggestions}")
    if response.disclaimer:
        sections.append(f"_{response.disclaimer}_")
    return {"formatted_response": "\n\n".join(sections)}


# ── LangGraph construction ─────────────────────────────────────────────────

def _build_agent_graph(models: list[BaseChatModel]) -> Any:
    """Compile the four-node FitPulse LangGraph workflow with conditional intent routing."""
    def generate_response(state: FitState) -> dict[str, Any]:
        """Invoke the configured provider sequence for the graph's response-generation node."""
        return _generate_response(models, state)

    workflow = StateGraph(FitState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("fetch_context", fetch_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("format_output", format_output)
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "metric_query": "fetch_context",
            "trend_query": "fetch_context",
            "comparison": "fetch_context",
            "advice": "fetch_context",
            "general": "fetch_context",
        },
    )
    workflow.add_edge("fetch_context", "generate_response")
    workflow.add_edge("generate_response", "format_output")
    workflow.add_edge("format_output", END)
    return workflow.compile()


@st.cache_resource(show_spinner=False)
def init_agent() -> Any:
    """Create and cache the LangGraph agent and its configured AI-provider clients."""
    return _build_agent_graph(_get_chat_models())


# ── Streamlit chat interface ───────────────────────────────────────────────

def _inject_chat_styles() -> None:
    """Inject styling for a floating launcher and a YouTube-style right-side chat drawer."""
    st.markdown(
        """
        <style>
        .st-key-fitpulse_chat_launcher {
            bottom: 1.5rem;
            position: fixed;
            right: 1.5rem;
            width: 58px;
            z-index: 1000;
        }
        .st-key-fitpulse_chat_launcher button {
            background: #00F5A0 !important;
            border: 0 !important;
            border-radius: 50% !important;
            box-shadow: 0 10px 30px rgba(0, 245, 160, 0.28);
            color: #06140f !important;
            font-size: 1.4rem !important;
            height: 58px !important;
            min-height: 58px !important;
            padding: 0 !important;
            width: 58px !important;
        }
        [data-testid="stDialog"], div[data-baseweb="modal"] {
            align-items: stretch !important;
            background: rgba(0, 0, 0, 0.68) !important;
            justify-content: flex-end !important;
            padding: 0 !important;
        }
        [data-testid="stDialog"] div[role="dialog"], div[data-baseweb="modal"] div[role="dialog"] {
            border-left: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 18px 0 0 18px !important;
            bottom: 0 !important;
            box-shadow: -18px 0 45px rgba(0, 0, 0, 0.35) !important;
            height: 100vh !important;
            left: auto !important;
            margin: 0 !important;
            max-height: 100vh !important;
            max-width: 430px !important;
            right: 0 !important;
            top: 0 !important;
            transform: none !important;
            width: min(430px, 100vw) !important;
        }
        [data-testid="stDialog"] [data-testid="stDialogContent"] {
            padding: 1.2rem 1.1rem 1.4rem !important;
        }
        div[data-baseweb="modal"] > div {
            margin-left: auto !important;
        }
        .fitpulse-chat-welcome {
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            line-height: 1.4;
            margin: 0.6rem 0 1rem;
        }
        .fitpulse-chat-helper {
            color: #a0a0b0;
            font-size: 0.86rem;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }
        [class*="st-key-fitpulse_suggestion_"] button {
            background: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 14px !important;
            color: #ffffff !important;
            font-size: 0.9rem !important;
            justify-content: flex-start !important;
            margin-top: 0.25rem !important;
            padding: 0.7rem 0.8rem !important;
            text-align: left !important;
        }
        [class*="st-key-fitpulse_suggestion_"] button:hover {
            border-color: #00F5A0 !important;
            color: #00F5A0 !important;
        }
        [data-testid="stDialog"] [data-testid="stChatInput"] {
            bottom: 0.8rem;
            position: sticky;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _history_to_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert serialized Streamlit chat history into LangChain messages for graph memory."""
    messages: list[BaseMessage] = []
    for item in history[-10:]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def _suggested_questions(user_id: str) -> list[str]:
    """Return clickable prompts tailored to the cohort or the selected individual profile."""
    if user_id == ALL_USERS_OPTION:
        return [
            "Summarize the cohort dashboard insights.",
            "Which activity and recovery segments need the most attention?",
            "What are the strongest opportunities to improve cohort health outcomes?",
        ]
    return [
        "Summarize my dashboard insights.",
        "What is the clearest area for me to improve?",
        "How does my activity compare with the overall cohort?",
    ]


def _submit_chat_prompt(agent: Any, dashboard_context: dict[str, str], prompt: str) -> None:
    """Run one prompt through the graph and append its formatted answer to session history."""
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Analyzing Fitbit data..."):
            result = agent.invoke(
                {
                    "messages": _history_to_messages(st.session_state.chat_history),
                    "user_id": dashboard_context["user_id"],
                    "context": dashboard_context["summary"],
                    "intent": "general",
                    "chat_response": {},
                    "formatted_response": "",
                }
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["formatted_response"]}
        )
    except Exception as error:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"I could not complete that analysis: {error}",
            }
        )


def _render_chat_dialog(agent: Any, dashboard_context: dict[str, str]) -> None:
    """Render the interactive right-side FitPulse drawer and invoke the graph for submitted questions."""
    def chat_dialog() -> None:
        """Display suggested prompts, recent messages, and one text prompt inside the chat drawer."""
        scope = "the all-users cohort" if dashboard_context["user_id"] == ALL_USERS_OPTION else "this athlete profile"
        st.markdown(
            f"<div class='fitpulse-chat-welcome'>Ask about {scope}</div>"
            "<div class='fitpulse-chat-helper'>Get concise, data-backed answers from the dashboard currently on screen.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Not sure what to ask?**")
        for index, question in enumerate(_suggested_questions(dashboard_context["user_id"])):
            if st.button(question, key=f"fitpulse_suggestion_{index}", width="stretch"):
                _submit_chat_prompt(agent, dashboard_context, question)
                st.rerun()
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
        prompt = st.chat_input("Ask about steps, sleep, calories, or trends", key="fitpulse_chat_input")
        if prompt:
            _submit_chat_prompt(agent, dashboard_context, prompt)
            st.rerun()
        if st.button("Close chat", key="fitpulse_chat_close", width="stretch"):
            st.session_state.fitpulse_chat_open = False
            st.rerun()

    chat_dialog()


def render_chatbot(agent: Any, dashboard_context: dict[str, str]) -> None:
    """Render a floating launcher and a data-aware LangGraph chat panel for the active dashboard."""
    if "fitpulse_chat_open" not in st.session_state:
        st.session_state.fitpulse_chat_open = False
    _inject_chat_styles()
    if st.button("💬", key="fitpulse_chat_launcher", help="Ask FitPulse AI"):
        st.session_state.fitpulse_chat_open = True
    if st.session_state.fitpulse_chat_open:
        _render_chat_dialog(agent, dashboard_context)


# V2 sidebar chat implementation. These later definitions intentionally supersede
# the older dialog-based helpers above while preserving the existing graph code.

def _safe_number(value: Any, decimals: int = 0, suffix: str = "") -> str:
    """Format dashboard numbers for the mandatory live chat context."""
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):,.{decimals}f}{suffix}"


def _safe_signed(value: Any) -> str:
    """Format a signed percentage delta for the mandatory live chat context."""
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):+.1f}"


def build_context_string(user_name: str, kpis: dict[str, Any], is_all_users: bool) -> str:
    """Build the structured live dashboard context injected into every LLM call."""
    if is_all_users:
        return f"""
DASHBOARD CONTEXT (All Users Cohort):
- Total users: {kpis.get('total_users', 0)}
- Avg daily steps: {_safe_number(kpis.get('avg_steps'))}
- Avg sleep efficiency: {_safe_number(kpis.get('avg_sleep'), 0, '%')}
- Avg heart rate: {_safe_number(kpis.get('avg_hr'), 0, ' bpm')}
- Avg wellness score: {_safe_number(kpis.get('avg_wellness'), 1)}
- Top performer: {kpis.get('top_user', 'Not available')} (score: {_safe_number(kpis.get('top_score'), 1)})
- Users needing attention: {kpis.get('attention_count', 0)}
""".strip()

    return f"""
DASHBOARD CONTEXT (Individual: {user_name}):
- Wellness score: {_safe_number(kpis.get('wellness'), 1)} ({_safe_signed(kpis.get('wellness_delta'))} vs last week)
- Avg daily steps: {_safe_number(kpis.get('steps'))} (goal: 10,000)
- Sleep efficiency: {_safe_number(kpis.get('sleep'), 0, '%')}
- Active minutes/day: {_safe_number(kpis.get('active_mins'))}
- Resting HR: {_safe_number(kpis.get('resting_hr'), 0, ' bpm')}
- Current streak: {_safe_number(kpis.get('streak'))} days
- Steps trend: {_safe_signed(kpis.get('steps_delta'))}% vs last week
- Ranking: #{_safe_number(kpis.get('rank'))} of {kpis.get('total_users', 0)} users
""".strip()


def get_suggested_questions(kpis: dict[str, Any], is_all_users: bool) -> list[str]:
    """Return dynamic clickable prompt chips based on the active dashboard numbers."""
    if is_all_users:
        return [
            "Summarise the cohort health overview",
            "Which users need the most attention right now?",
            "What are the strongest opportunities to improve cohort outcomes?",
        ]

    questions: list[str] = []
    steps_delta = kpis.get("steps_delta")
    sleep = kpis.get("sleep")
    streak = kpis.get("streak")
    wellness = kpis.get("wellness")
    if steps_delta is not None and not pd.isna(steps_delta) and steps_delta < -10:
        questions.append(f"Why did my steps drop {abs(float(steps_delta)):.0f}% this week?")
    if sleep is not None and not pd.isna(sleep) and sleep < 80:
        questions.append("How can I improve my sleep efficiency?")
    if streak is not None and not pd.isna(streak) and int(streak) == 0:
        questions.append("How do I get back on track with my daily activity?")
    if wellness is not None and not pd.isna(wellness) and wellness < 60:
        questions.append("What is pulling my wellness score down?")
    questions.append("Give me a summary of my fitness this week")
    questions.append("How do I compare to other users?")
    return questions[:3]


def _inject_chat_styles() -> None:
    """Inject styling for a floating launcher and in-layout right-side chat panel."""
    st.markdown(
        """
        <style>
        .st-key-fitpulse_chat_launcher {
            bottom: 1.5rem;
            position: fixed;
            right: 1.5rem;
            width: 58px;
            z-index: 1000;
        }
        .st-key-fitpulse_chat_launcher button {
            background: #00F5A0 !important;
            border: 0 !important;
            border-radius: 50% !important;
            box-shadow: 0 10px 30px rgba(0, 245, 160, 0.28);
            color: #06140f !important;
            font-size: 1.35rem !important;
            height: 58px !important;
            min-height: 58px !important;
            padding: 0 !important;
            width: 58px !important;
        }
        .fitpulse-chat-panel {
            background: #11111d;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            box-shadow: -14px 0 36px rgba(0,0,0,0.18);
            min-height: calc(100vh - 2.2rem);
            padding: 1rem;
            position: sticky;
            top: 1rem;
        }
        .fitpulse-chat-title {
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .fitpulse-chat-helper {
            color: #a0a0b0;
            font-size: 0.84rem;
            line-height: 1.45;
            margin: 0.4rem 0 0.8rem;
        }
        .fitpulse-chat-message-list {
            max-height: 48vh;
            overflow-y: auto;
            padding-right: 0.15rem;
        }
        [class*="st-key-fitpulse_suggestion_"] button {
            background: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 14px !important;
            color: #ffffff !important;
            font-size: 0.86rem !important;
            justify-content: flex-start !important;
            margin-top: 0.25rem !important;
            padding: 0.68rem 0.78rem !important;
            text-align: left !important;
        }
        [class*="st-key-fitpulse_suggestion_"] button:hover,
        .st-key-fitpulse_chat_send button:hover,
        .st-key-fitpulse_chat_close button:hover {
            border-color: #00F5A0 !important;
            color: #00F5A0 !important;
        }
        @media (max-width: 900px) {
            .fitpulse-chat-panel {
                min-height: auto;
                position: relative;
                top: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _submit_chat_prompt(agent: Any, dashboard_context: dict[str, Any], prompt: str) -> None:
    """Run one prompt through the graph and append its formatted answer to session history."""
    if not prompt.strip():
        return
    live_context = build_context_string(
        dashboard_context["user_name"],
        dashboard_context["chat_kpis"],
        dashboard_context["is_all_users"],
    )
    combined_context = f"{live_context}\n\nVisible dashboard summary:\n{dashboard_context['summary']}"
    st.session_state.chat_history.append({"role": "user", "content": prompt.strip()})
    try:
        with st.spinner("Analyzing Fitbit data..."):
            result = agent.invoke(
                {
                    "messages": _history_to_messages(st.session_state.chat_history),
                    "user_id": dashboard_context["user_name"],
                    "context": combined_context,
                    "intent": "general",
                    "chat_response": {},
                    "formatted_response": "",
                }
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["formatted_response"]}
        )
    except Exception as error:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"I could not complete that analysis: {error}",
            }
        )


def render_chat_launcher() -> None:
    """Render the always-available fixed chat launcher."""
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    _inject_chat_styles()
    if st.button("💬", key="fitpulse_chat_launcher", help="Ask FitPulse AI"):
        st.session_state.chat_open = True
        st.rerun()


def render_chat_panel(agent: Any, dashboard_context: dict[str, Any]) -> None:
    """Render the right-side in-layout chat panel."""
    _inject_chat_styles()
    st.markdown("<div class='fitpulse-chat-panel'>", unsafe_allow_html=True)
    header_column, close_column = st.columns([5, 1], vertical_alignment="center")
    with header_column:
        st.markdown("<div class='fitpulse-chat-title'>Ask FitPulse AI</div>", unsafe_allow_html=True)
    with close_column:
        if st.button("✕", key="fitpulse_chat_close", help="Close chat"):
            st.session_state.chat_open = False
            st.rerun()

    scope = "the all-users cohort" if dashboard_context["is_all_users"] else dashboard_context["user_name"]
    st.markdown(
        f"<div class='fitpulse-chat-helper'>Ask about {escape(str(scope))}. Answers are grounded in the metrics currently visible on this dashboard.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Not sure what to ask?**")
    for index, question in enumerate(
        get_suggested_questions(dashboard_context["chat_kpis"], dashboard_context["is_all_users"])
    ):
        if st.button(question, key=f"fitpulse_suggestion_{index}", width="stretch"):
            _submit_chat_prompt(agent, dashboard_context, question)
            st.rerun()

    st.markdown("<div class='fitpulse-chat-message-list'>", unsafe_allow_html=True)
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    input_column, send_column = st.columns([4, 1], vertical_alignment="bottom")
    with input_column:
        prompt = st.text_input(
            "Ask a question",
            key="fitpulse_chat_text",
            label_visibility="collapsed",
            placeholder="Ask about steps, sleep, calories, or trends",
        )
    with send_column:
        send_clicked = st.button("➤", key="fitpulse_chat_send", help="Send")
    if send_clicked and prompt:
        _submit_chat_prompt(agent, dashboard_context, prompt)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_chatbot(agent: Any, dashboard_context: dict[str, Any]) -> None:
    """Backward-compatible wrapper for older app entry points."""
    render_chat_launcher()
    if st.session_state.get("chat_open", False):
        render_chat_panel(agent, dashboard_context)
