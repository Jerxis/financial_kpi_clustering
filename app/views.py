"""Page renderers for the Phase 3 Streamlit demonstrator."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from app.charts import (
    PALETTE,
    cluster_scatter,
    hdbscan_frontier_chart,
    method_score_chart,
    observed_vs_null_chart,
    profile_heatmap,
)
from app.data_loader import report_figure_path


INTERACTIVE_METHODS = {
    "K-Means",
    "Agglomerative",
    "Gaussian Mixture",
    "HDBSCAN",
}
STATIC_METHODS = {"Autoencoder + K-Means", "DEC"}


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_value(value: object, format_spec: str = ".3f") -> str:
    return format(float(value), format_spec) if _finite(value) else "Not available"


def _balanced_hdbscan_row(rows: pd.DataFrame) -> pd.Series:
    frontier = rows[rows["Pareto Efficient"].fillna(False)].copy()
    frontier["Separation Rank"] = frontier["Common-Space Silhouette"].rank(
        ascending=False, method="min"
    )
    frontier["Coverage Rank"] = frontier["Evaluation Coverage"].rank(
        ascending=False, method="min"
    )
    frontier["Balanced Rank Sum"] = (
        frontier["Separation Rank"] + frontier["Coverage Rank"]
    )
    parts = frontier["Configuration ID"].str.extract(r"mcs=(\d+)\|ms=(\d+)")
    frontier["Min Cluster Size"] = pd.to_numeric(parts[0])
    frontier["Min Samples"] = pd.to_numeric(parts[1])
    return frontier.sort_values(
        [
            "Balanced Rank Sum",
            "Evaluation Coverage",
            "Common-Space Silhouette",
            "Min Cluster Size",
            "Min Samples",
        ],
        ascending=[True, False, False, True, True],
    ).iloc[0]


def comparison_frame(method_results: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    scenario = method_results[method_results["Scenario ID"].eq(scenario_id)]
    rows = []
    for method in [
        "K-Means",
        "Agglomerative",
        "Gaussian Mixture",
        "HDBSCAN",
        "Autoencoder + K-Means",
        "DEC",
        "IDEC",
    ]:
        candidates = scenario[scenario["Method"].eq(method)]
        if method == "HDBSCAN":
            selected = _balanced_hdbscan_row(candidates)
        elif len(candidates):
            selected = candidates.iloc[0]
        else:
            selected = pd.Series(dtype=object)
        rows.append(
            {
                "Method": method,
                "K": selected.get("K", np.nan),
                "Common-space Silhouette": selected.get(
                    "Common-Space Silhouette", np.nan
                ),
                "Davies–Bouldin": selected.get(
                    "Common-Space Davies-Bouldin", np.nan
                ),
                "Calinski–Harabasz": selected.get(
                    "Common-Space Calinski-Harabasz", np.nan
                ),
                "Evaluation Coverage": selected.get("Evaluation Coverage", np.nan),
                "Mean resampling ARI": selected.get("Mean Resampling ARI", np.nan),
                "Minimum resampling ARI": selected.get(
                    "Minimum Resampling ARI", np.nan
                ),
                "Configuration": selected.get("Configuration ID", "Not available"),
                "Canonical evidence": bool(
                    selected.get("Canonical v3 Evidence Available", False)
                ),
            }
        )
    return pd.DataFrame(rows)


def _method_card(method_entry: dict) -> None:
    status = (
        "Canonical v3 evidence"
        if method_entry["canonical_v3_evidence"]
        else "Exploratory only"
    )
    st.markdown(
        f"""
        <div class="method-card">
          <div class="eyebrow">{method_entry['family']}</div>
          <h3>{method_entry['method']}</h3>
          <p>{method_entry['demo_role']}</p>
          <div class="chip-row"><span class="chip">{method_entry['assignment_type']}</span>
          <span class="chip">{method_entry['cluster_count']}</span>
          <span class="chip">{status}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explorer(
    scenario_id: str,
    scenario_name: str,
    method: str,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    solution_summary: pd.DataFrame,
    method_results: pd.DataFrame,
    method_catalog: list[dict],
) -> None:
    st.markdown("<div class='eyebrow'>Workspace 01 · Cluster Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Financial Peer Archetypes</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='hero-copy'>Explore <strong>{scenario_name}</strong> with "
        f"<strong>{method}</strong>. Every displayed result was prepared before the live "
        "session; the app never trains a model while evaluators wait.</div>",
        unsafe_allow_html=True,
    )

    comparison = comparison_frame(method_results, scenario_id)
    evidence = comparison[comparison["Method"].eq(method)].iloc[0]
    solution_rows = solution_summary[
        solution_summary["Scenario ID"].eq(scenario_id)
        & solution_summary["Method"].eq(method)
    ]
    cohort_count = int(
        solution_summary[solution_summary["Scenario ID"].eq(scenario_id)][
            "Company Count"
        ].iloc[0]
    )

    if len(solution_rows):
        solution = solution_rows.iloc[0]
        k = solution["Cluster Count"]
        silhouette = solution["Frozen Evidence Silhouette"]
        coverage = solution["Evaluation Coverage"]
        configuration = solution["Configuration"]
    else:
        solution = None
        k = evidence["K"]
        silhouette = evidence["Common-space Silhouette"]
        coverage = evidence["Evaluation Coverage"]
        configuration = evidence["Configuration"]

    metrics = st.columns(5)
    metrics[0].metric("Eligible companies", f"{cohort_count:,}")
    metrics[1].metric("Clusters", _metric_value(k, ".0f"))
    metrics[2].metric("Common-space Silhouette", _metric_value(silhouette))
    metrics[3].metric("Evaluation coverage", _metric_value(coverage, ".1%"))
    metrics[4].metric(
        "Mean resampling ARI", _metric_value(evidence["Mean resampling ARI"])
    )
    st.caption(f"Selected configuration: {configuration}")

    if method in INTERACTIVE_METHODS:
        _render_interactive_solution(
            scenario_id,
            method,
            assignments,
            profiles,
            cluster_summary,
            solution,
        )
    elif method in STATIC_METHODS:
        _render_static_solution(scenario_id, method, solution)
    else:
        _render_idec_boundary(method_catalog)


def _render_interactive_solution(
    scenario_id: str,
    method: str,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    solution: pd.Series,
) -> None:
    selected_assignments = assignments[
        assignments["Scenario ID"].eq(scenario_id)
        & assignments["Method"].eq(method)
    ]
    selected_profiles = profiles[
        profiles["Scenario ID"].eq(scenario_id) & profiles["Method"].eq(method)
    ]
    selected_summary = cluster_summary[
        cluster_summary["Scenario ID"].eq(scenario_id)
        & cluster_summary["Method"].eq(method)
    ]
    labels = selected_summary["Cluster Label"].tolist()
    focus = st.segmented_control(
        "Cluster focus",
        ["All clusters", *labels],
        default="All clusters",
        key=f"focus_{scenario_id}_{method}",
    )
    visible = selected_assignments
    if focus and focus != "All clusters":
        visible = visible[visible["Cluster Label"].eq(focus)]

    if method == "HDBSCAN":
        st.info(
            "HDBSCAN reproduces the dissertation's balanced Pareto-frontier solution "
            "in canonical source-workbook order. Noise points remain visible but are "
            "excluded from separation metrics and KPI profiles."
        )

    chart_column, summary_column = st.columns([1.7, 1], gap="large")
    with chart_column:
        st.subheader("Common-space cluster map")
        st.plotly_chart(cluster_scatter(visible), width="stretch", config={"displayModeBar": False})
        st.caption(
            "PCA provides a two-dimensional display only. Clustering used all eight "
            "scenario-scaled KPIs."
        )
    with summary_column:
        st.subheader("Cluster composition")
        for index, (_, row) in enumerate(selected_summary.iterrows()):
            colour = "#94A3B8" if row["Cluster"] == -1 else PALETTE[index % len(PALETTE)]
            st.markdown(
                f"<div class='cluster-card' style='border-top-color:{colour}'>"
                f"<strong>{row['Cluster Label']}</strong> · {int(row['Company Count']):,} companies"
                f"<br><span>{row['Company Share']:.1%} of cohort · median revenue "
                f"${row['Median Revenue']/1e6:,.0f}M<br>Most common industry: "
                f"{row['Most Common Industry']}</span></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Financial signature")
    st.plotly_chart(profile_heatmap(selected_profiles), width="stretch", config={"displayModeBar": False})
    st.caption(
        "Blue means above the cohort median in robust-scaled model space; red means "
        "below. This is relative position—not business quality or investment advice."
    )

    cluster_options = [label for label in labels if label != "Noise / unassigned"]
    selected_label = st.selectbox(
        "KPI table cluster",
        cluster_options,
        key=f"profile_{scenario_id}_{method}",
    )
    selected_cluster = int(selected_label.split()[-1])
    profile_table = selected_profiles[selected_profiles["Cluster"].eq(selected_cluster)][[
        "KPI",
        "UoM",
        "What is better?",
        "Bottom Quartile",
        "Median",
        "Top Quartile",
        "Cohort Median",
        "Imputed Share",
    ]]
    st.dataframe(
        profile_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Bottom Quartile": st.column_config.NumberColumn(format="%.3f"),
            "Median": st.column_config.NumberColumn(format="%.3f"),
            "Top Quartile": st.column_config.NumberColumn(format="%.3f"),
            "Cohort Median": st.column_config.NumberColumn(format="%.3f"),
            "Imputed Share": st.column_config.ProgressColumn(
                format="%.1%%", min_value=0, max_value=1
            ),
        },
    )

    with st.expander("Inspect company assignments"):
        table = visible[[
            "Symbol",
            "Company Name",
            "Industry",
            "Cluster Label",
            "Total Revenue",
            "MarketCap",
            "Assignment Confidence",
            "Assignment Confidence Label",
            "Imputed Canonical KPI Count",
        ]]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Total Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                "MarketCap": st.column_config.NumberColumn(format="$%,.0f"),
                "Assignment Confidence": st.column_config.NumberColumn(format="%.3f"),
            },
        )


def _render_static_solution(
    scenario_id: str, method: str, solution: pd.Series | None
) -> None:
    st.info(
        "This is the report's median-seed neural illustration, not a best-seed result. "
        "The app does not retrain stochastic neural models during the demonstration."
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Representative cluster map")
        st.image(str(report_figure_path(scenario_id, method, "cluster")), width="stretch")
    with right:
        st.subheader("Representative KPI profile")
        st.image(str(report_figure_path(scenario_id, method, "heatmap")), width="stretch")
    if solution is not None:
        st.caption(f"Evidence policy: {solution['Selection Rule']}. {solution['Reconstruction Note']}")


def _render_idec_boundary(method_catalog: list[dict]) -> None:
    entry = next(item for item in method_catalog if item["method"] == "IDEC")
    st.error(
        "IDEC was researched as an exploratory method, but it was not carried into the "
        "final five-cohort v3 robustness analysis. Showing a canonical score or cluster "
        "map here would overstate the available evidence."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Why it remained relevant")
        for item in entry["strengths"]:
            st.markdown(f"- {item}")
    with right:
        st.markdown("#### Why it is separated")
        for item in entry["limitations"]:
            st.markdown(f"- {item}")


def render_method_lab(
    scenario_id: str,
    scenario_name: str,
    method: str,
    method_results: pd.DataFrame,
    method_catalog: list[dict],
) -> None:
    st.markdown("<div class='eyebrow'>Workspace 02 · Method Laboratory</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Compare the seven methods</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='hero-copy'>Use <strong>{scenario_name}</strong> to explain what each "
        "algorithm assumes, how it assigns companies and why a single validation score "
        "cannot choose the winner.</div>",
        unsafe_allow_html=True,
    )
    entry = next(item for item in method_catalog if item["method"] == method)
    _method_card(entry)
    strengths, limitations = st.columns(2, gap="large")
    with strengths:
        st.markdown("#### Advantages")
        for item in entry["strengths"]:
            st.markdown(f"- {item}")
    with limitations:
        st.markdown("#### Trade-offs")
        for item in entry["limitations"]:
            st.markdown(f"- {item}")

    comparison = comparison_frame(method_results, scenario_id)
    st.subheader("Separation, stability and coverage")
    chart, explanation = st.columns([1.45, 1], gap="large")
    with chart:
        st.plotly_chart(method_score_chart(comparison), width="stretch", config={"displayModeBar": False})
    with explanation:
        st.markdown(
            """
            <div class="callout"><strong>Read this as a three-way decision.</strong><br>
            Move right for stronger common-space separation; move up for more stable
            resampled assignments; larger points cover more of the cohort. A method can
            lead one objective without being the best overall choice.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Score vocabulary")
        st.markdown(
            "- **Silhouette:** cohesion versus separation; higher is better.\n"
            "- **Davies–Bouldin:** within-cluster scatter relative to separation; lower is better.\n"
            "- **Calinski–Harabasz:** between-cluster versus within-cluster dispersion; higher is better.\n"
            "- **ARI:** agreement between repeated partitions; 1 means identical assignments."
        )
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "Common-space Silhouette": st.column_config.NumberColumn(format="%.3f"),
            "Davies–Bouldin": st.column_config.NumberColumn(format="%.3f"),
            "Calinski–Harabasz": st.column_config.NumberColumn(format="%.1f"),
            "Evaluation Coverage": st.column_config.ProgressColumn(format="%.1%%", min_value=0, max_value=1),
            "Mean resampling ARI": st.column_config.NumberColumn(format="%.3f"),
            "Minimum resampling ARI": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "IDEC remains visible because it was one of the seven researched methods; its empty "
        "canonical cells are deliberate, not missing-data errors."
    )


def render_robustness(
    scenario_id: str,
    scenario_name: str,
    robustness: pd.DataFrame,
    method_results: pd.DataFrame,
) -> None:
    st.markdown("<div class='eyebrow'>Workspace 03 · Robustness & limitations</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Defend the result, not just the chart</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='hero-copy'>Evidence for <strong>{scenario_name}</strong>: cluster "
        "tendency, null comparison, confounding, imputation sensitivity, selection "
        "stability and HDBSCAN's coverage trade-off.</div>",
        unsafe_allow_html=True,
    )
    selected = robustness[robustness["Scenario ID"].eq(scenario_id)].iloc[0]
    metrics = st.columns(5)
    metrics[0].metric("Mean Hopkins", f"{selected['Mean Hopkins Statistic']:.3f}")
    metrics[1].metric(
        "Observed Silhouette", f"{selected['Observed Selection-Matched Silhouette']:.3f}"
    )
    metrics[2].metric(
        "Null 95th percentile",
        f"{selected['Null 95th Percentile Selection-Matched Silhouette']:.3f}",
    )
    metrics[3].metric(
        "Null tail probability", f"{selected['Null Empirical Tail Probability']:.3f}"
    )
    metrics[4].metric(
        "Complete-case fixed-K ARI",
        f"{selected['Canonical vs Complete-Case Fixed-K ARI']:.3f}",
    )

    leader_colour = "#DBEAFE" if "K-Means" in str(selected["Full-Assignment Research Conclusion"]) else "#FEF3C7"
    st.markdown(
        f"<div class='decision-strip' style='background:{leader_colour}'>"
        f"<strong>Research conclusion:</strong> {selected['Full-Assignment Research Conclusion']} · "
        f"separation leader: {selected['Separation-Focused Leader']} · balanced leader: "
        f"{selected['Balanced Leader']} · stability leader: {selected['Stability-Focused Leader']}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Observed structure versus matched null data")
        st.plotly_chart(observed_vs_null_chart(robustness), width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Sensitivity checks")
        st.markdown(
            f"""
            <div class="evidence-card"><strong>Confounding adjustment</strong><br>
            K-Means original vs adjusted ARI: <strong>{selected['K-Means Original vs Confounder-Adjusted ARI']:.3f}</strong><br>
            Mean KPI variance explained: {selected['Mean KPI Variance Explained by Confounders']:.1%}</div>
            <div class="evidence-card"><strong>Missing-data sensitivity</strong><br>
            Imputed feature cells: <strong>{selected['Imputed Feature Cell Share']:.1%}</strong><br>
            Complete-case retention: {selected['Complete-Case Retention vs Canonical']:.1%}</div>
            <div class="evidence-card"><strong>Configuration selection</strong><br>
            Modal choice: <strong>{selected['K-Means Modal Selected Configuration']}</strong><br>
            Modal share: {selected['K-Means Modal Selection Share']:.0%}</div>
            """,
            unsafe_allow_html=True,
        )

    hdbscan = method_results[
        method_results["Scenario ID"].eq(scenario_id)
        & method_results["Method"].eq("HDBSCAN")
    ]
    st.subheader("HDBSCAN coverage–separation frontier")
    hdb_chart, hdb_note = st.columns([1.3, 1], gap="large")
    with hdb_chart:
        st.plotly_chart(hdbscan_frontier_chart(hdbscan), width="stretch", config={"displayModeBar": False})
    with hdb_note:
        st.markdown(
            f"""
            <div class="callout"><strong>Why no universal winner?</strong><br>
            {selected['HDBSCAN Interpretation']}<br><br>
            Separation leader: <strong>{selected['HDBSCAN Separation-Leader Silhouette']:.3f}</strong>
            at {selected['HDBSCAN Separation-Leader Coverage']:.1%} coverage.<br>
            Coverage leader: <strong>{selected['HDBSCAN Coverage-Leader Silhouette']:.3f}</strong>
            at {selected['HDBSCAN Coverage-Leader Coverage']:.1%} coverage.</div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Q&A guardrails: what the analysis does not prove"):
        st.markdown(
            "- Clusters are exploratory peer archetypes, not natural laws or causal groups.\n"
            "- Internal validation does not establish future performance or investment value.\n"
            "- Cluster numbers are arbitrary identifiers with no ordinal meaning.\n"
            "- PCA is a display projection and cannot show every distinction used for fitting.\n"
            "- Stability and separation vary by cohort; conclusions should not be universalised."
        )
