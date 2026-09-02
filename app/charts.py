"""Plotly chart constructors for the Phase 3 demonstrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = ["#2563EB", "#F59E0B", "#14B8A6", "#8B5CF6", "#EC4899", "#22C55E"]


def cluster_colour_map(labels: list[str]) -> dict[str, str]:
    colours: dict[str, str] = {}
    cluster_labels = sorted(label for label in labels if label != "Noise / unassigned")
    for index, label in enumerate(cluster_labels):
        colours[label] = PALETTE[index % len(PALETTE)]
    colours["Noise / unassigned"] = "#94A3B8"
    return colours


def cluster_scatter(assignments: pd.DataFrame) -> go.Figure:
    colour_map = cluster_colour_map(assignments["Cluster Label"].unique().tolist())
    figure = px.scatter(
        assignments,
        x="PCA 1",
        y="PCA 2",
        color="Cluster Label",
        color_discrete_map=colour_map,
        hover_name="Company Name",
        hover_data={
            "Symbol": True,
            "Industry": True,
            "Total Revenue": ":$,.0f",
            "MarketCap": ":$,.0f",
            "Cluster Label": False,
            "PCA 1": ":.2f",
            "PCA 2": ":.2f",
        },
        opacity=0.72,
    )
    figure.update_traces(marker={"size": 8, "line": {"width": 0.45, "color": "white"}})
    figure.update_layout(
        height=525,
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        legend_title_text="Assignments",
        legend={"orientation": "h", "y": 1.04, "x": 0},
        xaxis_title="Common KPI-space PCA component 1",
        yaxis_title="Common KPI-space PCA component 2",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "white", "font_size": 13},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E8EDF4", zerolinecolor="#CBD5E1")
    figure.update_yaxes(showgrid=True, gridcolor="#E8EDF4", zerolinecolor="#CBD5E1")
    return figure


def profile_heatmap(profiles: pd.DataFrame) -> go.Figure:
    pivot = profiles.pivot(index="Cluster", columns="KPI", values="Model-Space Median")
    pivot = pivot.reindex(columns=profiles["KPI"].drop_duplicates().tolist())
    values = pivot.to_numpy()
    limit = max(float(np.nanmax(np.abs(values))), 0.5)
    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=pivot.columns,
            y=[f"Cluster {cluster}" for cluster in pivot.index],
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
            zmin=-limit,
            zmax=limit,
            text=[[f"{value:.2f}" for value in row] for row in values],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Median scaled value: %{z:.3f}<extra></extra>",
            colorbar={"title": "Median<br>scaled KPI", "thickness": 12},
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 20, "b": 80},
        xaxis={"tickangle": -28},
        yaxis={"title": ""},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def method_score_chart(comparison: pd.DataFrame) -> go.Figure:
    available = comparison.dropna(subset=["Common-space Silhouette"]).copy()
    figure = px.scatter(
        available,
        x="Common-space Silhouette",
        y="Mean resampling ARI",
        color="Method",
        size="Evaluation Coverage",
        hover_data={"K": True, "Evaluation Coverage": ":.1%"},
        size_max=28,
    )
    figure.update_traces(marker={"line": {"width": 1, "color": "white"}})
    figure.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E8EDF4")
    figure.update_yaxes(showgrid=True, gridcolor="#E8EDF4")
    return figure


def observed_vs_null_chart(robustness: pd.DataFrame) -> go.Figure:
    frame = robustness[[
        "Scenario Name",
        "Observed Selection-Matched Silhouette",
        "Null 95th Percentile Selection-Matched Silhouette",
    ]].melt(id_vars="Scenario Name", var_name="Evidence", value_name="Silhouette")
    frame["Evidence"] = frame["Evidence"].replace(
        {
            "Observed Selection-Matched Silhouette": "Observed",
            "Null 95th Percentile Selection-Matched Silhouette": "Null 95th percentile",
        }
    )
    figure = px.bar(
        frame,
        x="Scenario Name",
        y="Silhouette",
        color="Evidence",
        barmode="group",
        color_discrete_map={"Observed": "#2563EB", "Null 95th percentile": "#CBD5E1"},
    )
    figure.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 20, "b": 100},
        xaxis_title="",
        yaxis_title="Selection-matched Silhouette",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(tickangle=-22)
    figure.update_yaxes(showgrid=True, gridcolor="#E8EDF4")
    return figure


def hdbscan_frontier_chart(rows: pd.DataFrame) -> go.Figure:
    figure = px.scatter(
        rows,
        x="Evaluation Coverage",
        y="Common-Space Silhouette",
        color="Pareto Efficient",
        hover_name="Configuration ID",
        hover_data={
            "Mean Resampling ARI": ":.3f",
            "Minimum Resampling ARI": ":.3f",
        },
        color_discrete_map={True: "#2563EB", False: "#CBD5E1"},
    )
    figure.update_traces(marker={"size": 12, "line": {"width": 1, "color": "white"}})
    figure.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_tickformat=".0%",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E8EDF4")
    figure.update_yaxes(showgrid=True, gridcolor="#E8EDF4")
    return figure
