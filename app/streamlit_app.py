"""Financial peer-archetype demonstration application."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import (  # noqa: E402
    load_cluster_profiles,
    load_cluster_summary,
    load_company_assignments,
    load_method_catalog,
    load_method_results,
    load_robustness_summary,
    load_scenario_catalog,
    load_solution_summary,
)
from app.views import render_explorer, render_method_lab, render_robustness  # noqa: E402


st.set_page_config(
    page_title="Financial Peer Archetypes",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#10233F; --muted:#64748B; --line:#E2E8F0; --surface:#F7F9FC; }
      .stApp { background:#FBFCFE; }
      [data-testid="stSidebar"] { background:#10233F; }
      [data-testid="stSidebar"] * { color:#F8FAFC; }
      [data-testid="stSidebar"] label { font-weight:650; }
      [data-testid="stSidebar"] [data-baseweb="select"] * { color:#10233F; }
      [data-testid="stSidebar"] input {
        color:#10233F !important;
        -webkit-text-fill-color:#10233F !important;
      }
      .block-container { padding-top:2rem; padding-bottom:4rem; max-width:1500px; }
      .eyebrow { color:#2563EB; font-size:.75rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
      .hero-title { color:var(--ink); font-size:2.65rem; line-height:1.05; font-weight:760; margin:.35rem 0 .7rem; }
      .hero-copy { color:var(--muted); font-size:1.05rem; max-width:950px; line-height:1.65; margin-bottom:1.4rem; }
      .status-pill { display:inline-block; border:1px solid #86EFAC; color:#DCFCE7; background:#14532D;
                     padding:.3rem .65rem; border-radius:999px; font-size:.76rem; font-weight:750; }
      .callout { border-left:4px solid #2563EB; background:#EFF6FF; padding:1rem 1.1rem;
                 border-radius:0 .7rem .7rem 0; color:#1E3A5F; line-height:1.55; }
      .method-card { background:white; border:1px solid var(--line); border-radius:1rem; padding:1.25rem 1.35rem; margin:1rem 0; }
      .method-card h3 { margin:.35rem 0; }
      .method-card p { color:var(--muted); }
      .chip-row { display:flex; gap:.45rem; flex-wrap:wrap; }
      .chip { background:#EFF6FF; border:1px solid #BFDBFE; color:#1D4ED8; padding:.28rem .58rem; border-radius:999px; font-size:.76rem; font-weight:700; }
      .cluster-card { background:white; border:1px solid var(--line); border-top:4px solid #2563EB; border-radius:.7rem; padding:.85rem 1rem; margin-bottom:.65rem; }
      .cluster-card span { color:var(--muted); font-size:.84rem; line-height:1.45; }
      .decision-strip { padding:1rem 1.1rem; margin:1rem 0 1.35rem; border-radius:.75rem; color:#1E3A5F; line-height:1.55; }
      .evidence-card { background:white; border:1px solid var(--line); border-radius:.75rem; padding:.95rem 1rem; margin-bottom:.7rem; color:var(--muted); line-height:1.55; }
      .evidence-card strong { color:var(--ink); }
      div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:.8rem; padding:1rem; }
      div[data-testid="stMetric"] label { color:var(--muted); }
      div[data-testid="stMetricValue"] { color:var(--ink); }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.65rem; overflow:hidden; }
      h2, h3, h4 { color:var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_all_data():
    return (
        load_company_assignments(),
        load_cluster_profiles(),
        load_cluster_summary(),
        load_solution_summary(),
        load_scenario_catalog(),
        load_method_results(),
        load_robustness_summary(),
        load_method_catalog(),
    )


(
    assignments,
    profiles,
    cluster_summary,
    solution_summary,
    scenarios,
    method_results,
    robustness,
    method_catalog,
) = _load_all_data()

scenario_labels = {
    "technology_revenue_25m": "Technology · Revenue ≥ $25M",
    "healthcare_revenue_50m": "Healthcare · Revenue ≥ $50M",
    "industrials_revenue_20m": "Industrials · Revenue ≥ $20M",
    "consumer_cyclical_revenue_20m": "Consumer Cyclical · Revenue ≥ $20M",
    "non_financial_large_cap_revenue_1b": "Non-financial · Revenue ≥ $1B",
}
scenario_ids = scenarios["Scenario ID"].tolist()
method_names = [entry["method"] for entry in method_catalog]

with st.sidebar:
    st.markdown("## Financial Peer Archetypes")
    st.caption("Dissertation demonstration · precomputed v3 evidence")
    st.divider()
    workspace = st.radio(
        "Workspace",
        ["Cluster Explorer", "Method Laboratory", "Robustness & limitations"],
    )
    scenario_id = st.selectbox(
        "Canonical cohort",
        scenario_ids,
        format_func=lambda value: scenario_labels[value],
    )
    method = st.selectbox("Clustering method", method_names)
    st.divider()
    selected_scenario = scenarios[scenarios["Scenario ID"].eq(scenario_id)].iloc[0]
    st.markdown("**Data contract**")
    st.caption(
        f"{int(selected_scenario['Companies Retained']):,} companies · 8 KPIs · "
        "scenario-local preprocessing"
    )
    st.caption("Median imputation · 5–95% winsorisation · RobustScaler")
    st.markdown("<span class='status-pill'>Precomputed & verified</span>", unsafe_allow_html=True)
    st.caption("IDEC is retained as exploratory-only evidence.")

scenario_name = selected_scenario["Scenario Name"]

if workspace == "Cluster Explorer":
    render_explorer(
        scenario_id,
        scenario_name,
        method,
        assignments,
        profiles,
        cluster_summary,
        solution_summary,
        method_results,
        method_catalog,
    )
elif workspace == "Method Laboratory":
    render_method_lab(
        scenario_id,
        scenario_name,
        method,
        method_results,
        method_catalog,
    )
else:
    render_robustness(scenario_id, scenario_name, robustness, method_results)
