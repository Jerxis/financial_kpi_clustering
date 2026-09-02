"""Direct SHAP explanations for K-Means distance-margin assignments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def kmeans_margin_scores(
    X: np.ndarray,
    cluster_centers: np.ndarray,
) -> np.ndarray:
    """Return each centroid's squared-distance margin over its nearest rival.

    For cluster ``c`` the output is:

    ``minimum squared distance to any other centroid - distance to c``.

    A positive value supports assignment to ``c``; the largest margin has the
    same label as nearest-centroid K-Means (apart from exact distance ties).
    """

    X = np.asarray(X, dtype=float)
    centers = np.asarray(cluster_centers, dtype=float)
    squared_distances = (
        (X[:, None, :] - centers[None, :, :]) ** 2
    ).sum(axis=2)
    margins = np.empty_like(squared_distances)

    for cluster_index in range(centers.shape[0]):
        competing_distances = np.delete(
            squared_distances,
            cluster_index,
            axis=1,
        )
        margins[:, cluster_index] = (
            competing_distances.min(axis=1)
            - squared_distances[:, cluster_index]
        )

    return margins


def make_kmeans_margin_function(
    cluster_centers: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create the model function passed to a model-agnostic SHAP explainer."""

    centers = np.asarray(cluster_centers, dtype=float).copy()

    def margin_function(X: np.ndarray) -> np.ndarray:
        return kmeans_margin_scores(X, centers)

    return margin_function


def stratified_sample_indices(
    labels: np.ndarray,
    maximum_rows: int,
    random_state: int,
) -> np.ndarray:
    """Sample approximately equally from every cluster."""

    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    unique_labels = np.unique(labels)
    base_per_cluster = max(maximum_rows // len(unique_labels), 1)
    selected: list[int] = []

    for cluster in unique_labels:
        cluster_indices = np.flatnonzero(labels == cluster)
        n_select = min(base_per_cluster, len(cluster_indices))
        selected.extend(
            rng.choice(
                cluster_indices,
                size=n_select,
                replace=False,
            ).tolist()
        )

    remaining_capacity = maximum_rows - len(selected)
    if remaining_capacity > 0:
        remaining = np.setdiff1d(
            np.arange(len(labels)),
            np.asarray(selected, dtype=int),
        )
        if len(remaining):
            selected.extend(
                rng.choice(
                    remaining,
                    size=min(remaining_capacity, len(remaining)),
                    replace=False,
                ).tolist()
            )

    return np.asarray(sorted(selected), dtype=int)


def assigned_cluster_shap_values(
    shap_values: np.ndarray,
    assigned_labels: np.ndarray,
) -> np.ndarray:
    """Extract each company's attributions for its assigned cluster output."""

    values = np.asarray(shap_values)
    assigned_labels = np.asarray(assigned_labels, dtype=int)
    if values.ndim != 3:
        raise ValueError(
            "Expected SHAP values with shape "
            "(companies, features, cluster outputs)."
        )
    return np.stack([
        values[row_index, :, cluster]
        for row_index, cluster in enumerate(assigned_labels)
    ])


def summarise_cluster_shap(
    assigned_values: np.ndarray,
    X_explained: pd.DataFrame,
    assigned_labels: np.ndarray,
) -> pd.DataFrame:
    """Create cluster-specific global attribution summaries."""

    records = []
    labels = np.asarray(assigned_labels)
    for cluster in sorted(np.unique(labels)):
        cluster_mask = labels == cluster
        cluster_values = assigned_values[cluster_mask]
        cluster_features = X_explained.loc[cluster_mask]
        for feature_index, feature in enumerate(X_explained.columns):
            feature_shap = cluster_values[:, feature_index]
            records.append({
                "Cluster": int(cluster),
                "KPI": feature,
                "Mean Absolute SHAP": float(
                    np.mean(np.abs(feature_shap))
                ),
                "Mean Signed SHAP": float(np.mean(feature_shap)),
                "Median Signed SHAP": float(np.median(feature_shap)),
                "Mean Scaled KPI Value": float(
                    cluster_features[feature].mean()
                ),
                "N Explained Companies": int(cluster_mask.sum()),
            })
    return pd.DataFrame(records)


def calculate_ranking_agreement(
    importance_a: pd.DataFrame,
    importance_b: pd.DataFrame,
) -> pd.DataFrame:
    """Compare KPI-importance rankings from two SHAP background samples."""

    records = []
    for cluster in sorted(
        set(importance_a["Cluster"]) & set(importance_b["Cluster"])
    ):
        first = (
            importance_a[importance_a["Cluster"].eq(cluster)]
            .set_index("KPI")["Mean Absolute SHAP"]
        )
        second = (
            importance_b[importance_b["Cluster"].eq(cluster)]
            .set_index("KPI")["Mean Absolute SHAP"]
        )
        shared = first.index.intersection(second.index)
        correlation = spearmanr(
            first.loc[shared].rank(ascending=False),
            second.loc[shared].rank(ascending=False),
        ).statistic
        records.append({
            "Cluster": int(cluster),
            "SHAP Background Ranking Spearman": float(correlation),
            "Top KPI Background A": first.idxmax(),
            "Top KPI Background B": second.idxmax(),
            "Top KPI Agreement": first.idxmax() == second.idxmax(),
        })
    return pd.DataFrame(records)
