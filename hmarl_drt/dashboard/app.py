"""
HMARL-DRT Dashboard — Streamlit Application
=============================================
Interactive dashboard for monitoring and analysing
Heterogeneous Multi-Agent RL training runs.

Run:  cd hmarl_drt && streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import os
import sys
import time
from pathlib import Path

# ── Ensure project root + dashboard dir are importable ──
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DASHBOARD_DIR = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

# ── Component imports ────────────────────────────────
from components.api_client import get_health
from components import home, settings
from components.pages import (
    page_training_manager,
    page_overview,
    page_agents,
    page_env_sim,
    page_rewards,
    page_architecture,
    page_raw,
)

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="HMARL-DRT Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme-aware CSS ──────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #5dade2;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        opacity: 0.65;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stSidebar"] { min-width: 260px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_training_log(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col != "phase":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def discover_runs(base_dir: str) -> dict:
    runs = {}
    base = Path(base_dir)
    for csv_file in sorted(base.rglob("training_log.csv")):
        runs[csv_file.parent.name] = str(csv_file)
    return runs


# ─────────────────────────────────────────────────────
# PAGE LIST
# ─────────────────────────────────────────────────────
PAGE_LIST = [
    "🏠 Home",
    "🚀 Training Manager",
    "📊 Training Overview",
    "🤖 Agent Analysis",
    "🌀 Environment Sim",
    "🎯 Reward Analysis",
    "🏗️ Architecture",
    "📋 Raw Data",
    "⚙️ Settings",
]


# ─────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">⚡ HMARL-DRT Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Heterogeneous Multi-Agent RL &mdash; '
        'Dynamic Reward Shaping &amp; Transfer Learning</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.title("Navigation")

        # Handle quick-action navigation from Home page
        default_idx = 0
        if "nav_page" in st.session_state:
            nav = st.session_state.pop("nav_page")
            for i, p in enumerate(PAGE_LIST):
                if nav in p:
                    default_idx = i
                    break

        page = st.radio("Page", PAGE_LIST, index=default_idx, key="sidebar_page")

        st.divider()

        # Run selector (for data-viewing pages)
        runs_dir = os.path.join(PROJECT_ROOT, "runs")
        available_runs = discover_runs(runs_dir)

        if available_runs:
            selected_run = st.selectbox("Select Run", list(available_runs.keys()))
        else:
            st.warning("No training runs found.")
            st.caption("Run training first or use the Training Manager.")
            selected_run = None

        st.divider()
        auto_refresh = st.checkbox("Auto-Refresh (5s)",
                                    value=st.session_state.get("auto_refresh", False))

        # Backend status indicator
        st.divider()
        health = get_health()
        if health:
            st.success(f"Backend: {health['device'].upper()} | "
                       f"{health['active_jobs']} active jobs")
        else:
            st.warning("Backend offline — start with:\n"
                       "`uvicorn backend.main:app`")

    # ── Load Data ──
    if selected_run and selected_run in available_runs:
        df = load_training_log(available_runs[selected_run])
        run_dir = str(Path(available_runs[selected_run]).parent)
    else:
        df, run_dir = pd.DataFrame(), ""

    # ── Route ──
    page_name = page.split(" ", 1)[1] if " " in page else page  # strip emoji

    if page_name == "Home":
        home.render()
    elif page_name == "Training Manager":
        page_training_manager()
    elif page_name == "Training Overview":
        page_overview(df, run_dir)
    elif page_name == "Agent Analysis":
        page_agents(df)
    elif page_name == "Environment Sim":
        page_env_sim()
    elif page_name == "Reward Analysis":
        page_rewards(df)
    elif page_name == "Architecture":
        page_architecture()
    elif page_name == "Raw Data":
        page_raw(df)
    elif page_name == "Settings":
        settings.render()

    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
