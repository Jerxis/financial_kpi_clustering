"""Consistent, publication-ready figures for canonical clustering results."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


METHOD_COLOURS = {
    "K-Means": "#4C78A8",
    "Agglomerative": "#F58518",
    "Gaussian Mixture": "#54A24B",
    "HDBSCAN": "#B279A2",
    "Autoencoder + K-Means": "#E45756",
    "DEC": "#72B7B2",
}


def slugify(value: str) -> str:
    """Create a stable filename component."""

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


def prepare_figure_directories(root: Path) -> dict[str, Path]:
    """Create the consolidated reporting-figure directory tree."""

    directories = {
        "root": Path(root),
        "clusters": Path(root) / "clusters",
        "clusters_3d": Path(root) / "clusters_3d",
        "heatmaps": Path(root) / "heatmaps",
        "cohort_panels": Path(root) / "cohort_panels",
        "diagnostics": Path(root) / "diagnostics",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def common_pca_projection(X_scaled: np.ndarray) -> tuple[np.ndarray, float]:
    """Project one cohort once so every method uses identical coordinates."""

    pca = PCA(n_components=2)
    projection = pca.fit_transform(X_scaled)
    explained_share = float(
        pca.explained_variance_ratio_.sum()
    )
    return projection, explained_share


def common_pca_projection_3d(
    X_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one cohort into three common components for visualisation."""

    pca = PCA(n_components=3)
    projection = pca.fit_transform(X_scaled)
    return projection, pca.explained_variance_ratio_.copy()


def _cluster_palette(labels: np.ndarray) -> dict:
    clusters = sorted(pd.unique(labels))
    non_noise = [cluster for cluster in clusters if cluster != -1]
    colours = sns.color_palette("tab10", n_colors=max(len(non_noise), 1))
    palette = {
        cluster: colours[index]
        for index, cluster in enumerate(non_noise)
    }
    if -1 in clusters:
        palette[-1] = (0.65, 0.65, 0.65)
    return palette


def save_cluster_map(
    projection: np.ndarray,
    labels: np.ndarray,
    scenario_name: str,
    method: str,
    configuration: str,
    explained_share: float,
    output_path: Path,
) -> None:
    """Save one consistently styled two-dimensional cluster map."""

    labels = np.asarray(labels)
    palette = _cluster_palette(labels)
    figure, axis = plt.subplots(figsize=(9, 7))

    for cluster in sorted(pd.unique(labels)):
        mask = labels == cluster
        cluster_name = "Noise (-1)" if cluster == -1 else f"Cluster {cluster}"
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=28,
            alpha=0.72,
            color=palette[cluster],
            edgecolor="white",
            linewidth=0.25,
            label=f"{cluster_name} (n={mask.sum():,})",
        )

    axis.axhline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
    axis.axvline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
    axis.set_title(f"{scenario_name}\n{method}: {configuration}", pad=14)
    axis.set_xlabel("Common KPI-space PCA component 1")
    axis.set_ylabel("Common KPI-space PCA component 2")
    axis.text(
        0.01,
        0.01,
        f"Two components explain {explained_share:.1%} of scaled-KPI variance",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    axis.legend(
        title="Assignments",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    sns.despine()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_cluster_map_3d(
    projection: np.ndarray,
    labels: np.ndarray,
    scenario_name: str,
    method: str,
    configuration: str,
    explained_ratios: np.ndarray,
    output_path: Path,
    elevation: float = 24,
    azimuth: float = 38,
) -> None:
    """Save a static three-dimensional common-PCA cluster view."""

    labels = np.asarray(labels)
    explained_ratios = np.asarray(explained_ratios, dtype=float)
    palette = _cluster_palette(labels)
    figure = plt.figure(figsize=(10.5, 8.5))
    axis = figure.add_subplot(111, projection="3d")

    for cluster in sorted(pd.unique(labels)):
        mask = labels == cluster
        cluster_name = "Noise (-1)" if cluster == -1 else f"Cluster {cluster}"
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            projection[mask, 2],
            s=25,
            alpha=0.68,
            color=palette[cluster],
            edgecolor="white",
            linewidth=0.2,
            depthshade=True,
            label=f"{cluster_name} (n={mask.sum():,})",
        )

    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title(
        f"{scenario_name}\n{method}: {configuration}",
        pad=18,
    )
    axis.set_xlabel(f"PC1 ({explained_ratios[0]:.1%})", labelpad=9)
    axis.set_ylabel(f"PC2 ({explained_ratios[1]:.1%})", labelpad=9)
    axis.set_zlabel(f"PC3 ({explained_ratios[2]:.1%})", labelpad=9)
    axis.text2D(
        0.01,
        0.01,
        (
            f"PC1–PC3 explain {explained_ratios.sum():.1%}; "
            f"PC3 adds {explained_ratios[2]:.1%}"
        ),
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    axis.legend(
        title="Assignments",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_profile_heatmap(
    model_features: pd.DataFrame,
    labels: np.ndarray,
    scenario_name: str,
    method: str,
    configuration: str,
    output_path: Path,
) -> None:
    """Save median robust-scaled KPI profiles for non-noise clusters."""

    profile_data = model_features.copy()
    profile_data["Cluster"] = np.asarray(labels)
    profile_data = profile_data[profile_data["Cluster"] != -1]
    profiles = profile_data.groupby("Cluster").median()

    figure_width = max(10, 1.15 * len(profiles.columns))
    figure_height = max(4, 0.75 * len(profiles) + 2.8)
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )
    limit = float(np.nanmax(np.abs(profiles.to_numpy())))
    limit = max(limit, 0.5)
    sns.heatmap(
        profiles,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Median robust-scaled KPI value"},
        ax=axis,
    )
    axis.set_title(
        f"{scenario_name}\n{method}: cluster KPI profiles ({configuration})",
        pad=14,
    )
    axis.set_xlabel("Canonical KPI")
    axis.set_ylabel("Aligned model cluster")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_cohort_method_panel(
    projection: np.ndarray,
    solutions: list[dict],
    scenario_name: str,
    explained_share: float,
    output_path: Path,
) -> None:
    """Save all method assignments for one cohort on common coordinates."""

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10.5),
        sharex=True,
        sharey=True,
    )
    for axis, solution in zip(axes.flat, solutions):
        labels = np.asarray(solution["labels"])
        palette = _cluster_palette(labels)
        for cluster in sorted(pd.unique(labels)):
            mask = labels == cluster
            axis.scatter(
                projection[mask, 0],
                projection[mask, 1],
                s=12,
                alpha=0.65,
                color=palette[cluster],
                linewidth=0,
            )
        n_clusters = len(set(labels) - {-1})
        noise_share = float(np.mean(labels == -1))
        subtitle = f"{n_clusters} clusters"
        if noise_share:
            subtitle += f"; {noise_share:.1%} noise"
        axis.set_title(
            f"{solution['method']}\n{solution['configuration']}\n{subtitle}",
            fontsize=10,
        )
        axis.axhline(0, color="#E5E5E5", linewidth=0.5, zorder=0)
        axis.axvline(0, color="#E5E5E5", linewidth=0.5, zorder=0)

    for axis in axes[-1, :]:
        axis.set_xlabel("Common PCA component 1")
    for axis in axes[:, 0]:
        axis.set_ylabel("Common PCA component 2")
    figure.suptitle(
        f"{scenario_name}: canonical method comparison\n"
        f"Identical common-space projection; two PCs explain "
        f"{explained_share:.1%}",
        fontsize=15,
        y=1.01,
    )
    sns.despine()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_metric_comparison(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save common-space separation and coverage across final solutions."""

    plot_data = metrics.copy()
    figure, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    sns.barplot(
        data=plot_data,
        x="Scenario Name",
        y="Common-Space Silhouette Score",
        hue="Method",
        palette=METHOD_COLOURS,
        ax=axes[0],
    )
    axes[0].set_title(
        "Canonical final solutions: common scaled-KPI-space separation"
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Silhouette score")
    axes[0].legend(
        title="Method",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    sns.barplot(
        data=plot_data,
        x="Scenario Name",
        y="Evaluation Coverage",
        hue="Method",
        palette=METHOD_COLOURS,
        ax=axes[1],
    )
    axes[1].set_title(
        "Evaluation coverage (HDBSCAN excludes noise)"
    )
    axes[1].set_xlabel("Cohort")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0, 1.05)
    legend = axes[1].get_legend()
    if legend is not None:
        legend.remove()
    axes[1].tick_params(axis="x", rotation=20)
    sns.despine()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_stability_comparison(
    stability: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save fixed-configuration assignment stability by cohort and method."""

    figure, axis = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=stability,
        x="Scenario Name",
        y="Mean ARI",
        hue="Method",
        palette=METHOD_COLOURS,
        ax=axis,
    )
    axis.set_title(
        "Repeated 80% subsampling: fixed-configuration assignment stability"
    )
    axis.set_xlabel("Cohort")
    axis.set_ylabel("Mean pairwise adjusted Rand index")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(
        title="Method",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    sns.despine()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
