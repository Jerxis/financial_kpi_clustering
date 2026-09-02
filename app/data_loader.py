"""Cached readers for the Streamlit demo's versioned data contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "app"
SOLUTION_DIR = APP_DATA_DIR / "solutions"


@st.cache_data(show_spinner=False)
def load_company_assignments() -> pd.DataFrame:
    return pd.read_csv(SOLUTION_DIR / "traditional_company_assignments.csv.gz")


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> pd.DataFrame:
    return pd.read_csv(SOLUTION_DIR / "traditional_cluster_profiles.csv")


@st.cache_data(show_spinner=False)
def load_cluster_summary() -> pd.DataFrame:
    return pd.read_csv(SOLUTION_DIR / "traditional_cluster_summary.csv")


@st.cache_data(show_spinner=False)
def load_solution_summary() -> pd.DataFrame:
    return pd.read_csv(SOLUTION_DIR / "traditional_solution_summary.csv")


@st.cache_data(show_spinner=False)
def load_scenario_catalog() -> pd.DataFrame:
    return pd.read_csv(APP_DATA_DIR / "scenario_catalog.csv")


@st.cache_data(show_spinner=False)
def load_method_results() -> pd.DataFrame:
    return pd.read_csv(APP_DATA_DIR / "method_results.csv")


@st.cache_data(show_spinner=False)
def load_robustness_summary() -> pd.DataFrame:
    return pd.read_csv(APP_DATA_DIR / "robustness_summary.csv")


@st.cache_data(show_spinner=False)
def load_cluster_tendency() -> pd.DataFrame:
    return pd.read_csv(APP_DATA_DIR / "cluster_tendency.csv")


@st.cache_data(show_spinner=False)
def load_method_catalog() -> list[dict]:
    return json.loads(
        (APP_DATA_DIR / "method_catalog.json").read_text(encoding="utf-8")
    )


def report_figure_path(scenario_id: str, method: str, kind: str) -> Path:
    scenario_prefix = {
        "technology_revenue_25m": "01_technology_revenue_25m",
        "healthcare_revenue_50m": "02_healthcare_revenue_50m",
        "industrials_revenue_20m": "03_industrials_revenue_20m",
        "consumer_cyclical_revenue_20m": "04_consumer_cyclical_revenue_20m",
        "non_financial_large_cap_revenue_1b": (
            "05_non_financial_large_cap_revenue_1b"
        ),
    }[scenario_id]
    method_slug = {
        "K-Means": "01_k_means",
        "Agglomerative": "02_agglomerative",
        "Gaussian Mixture": "03_gaussian_mixture",
        "HDBSCAN": "04_hdbscan",
        "Autoencoder + K-Means": "05_autoencoder_k_means",
        "DEC": "06_dec",
    }[method]
    if kind == "cluster":
        directory, suffix = "clusters", "cluster_map"
    elif kind == "heatmap":
        directory, suffix = "heatmaps", "kpi_profile_heatmap"
    else:
        raise ValueError(f"Unknown report figure kind: {kind}")
    return (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / directory
        / f"{scenario_prefix}__{method_slug}__{suffix}.png"
    )
