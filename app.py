"""Integrated FitPulse AI application with dashboard analytics and LangGraph chat."""

import streamlit as st

from dashboard import render_dashboard
from chatbot import init_agent, render_chat_launcher, render_chat_panel


def main() -> None:
    """Render the dashboard with an optional right-side FitPulse AI chat panel."""
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    agent = init_agent()
    if st.session_state.chat_open:
        dashboard_column, chat_column = st.columns([7, 3], gap="medium")
        with dashboard_column:
            dashboard_context = render_dashboard()
        with chat_column:
            render_chat_panel(agent, dashboard_context)
    else:
        render_dashboard()
    render_chat_launcher()


if __name__ == "__main__":
    main()
