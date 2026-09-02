"""Cluster alignment and empirical profile-uncertainty utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def align_labels_to_reference(
    reference_labels: pd.Series,
    resampled_labels: pd.Series,
    noise_label: int | None = None,
) -> tuple[dict, dict]:
    """Align resampled cluster IDs to reference IDs by maximum overlap.

    Only companies present in both samples are used. When ``noise_label`` is
    supplied, noise is excluded from the assignment problem and reported
    separately. The Hungarian algorithm finds the one-to-one mapping with the
    largest total company overlap; unmatched clusters remain explicit.
    """

    shared = reference_labels.index.intersection(resampled_labels.index)
    comparison = pd.DataFrame({
        "Reference": reference_labels.loc[shared],
        "Resampled": resampled_labels.loc[shared],
    })

    reference_noise_share = 0.0
    resampled_noise_share = 0.0
    if noise_label is not None and len(comparison):
        reference_noise_share = float(
            comparison["Reference"].eq(noise_label).mean()
        )
        resampled_noise_share = float(
            comparison["Resampled"].eq(noise_label).mean()
        )
        comparison = comparison[
            comparison["Reference"].ne(noise_label)
            & comparison["Resampled"].ne(noise_label)
        ]

    reference_clusters = sorted(comparison["Reference"].unique())
    resampled_clusters = sorted(comparison["Resampled"].unique())
    mapping: dict = {}
    matched_overlap = 0

    if reference_clusters and resampled_clusters:
        contingency = pd.crosstab(
            comparison["Reference"],
            comparison["Resampled"],
        ).reindex(
            index=reference_clusters,
            columns=resampled_clusters,
            fill_value=0,
        )
        row_indices, column_indices = linear_sum_assignment(
            -contingency.to_numpy()
        )
        for row_index, column_index in zip(row_indices, column_indices):
            reference_cluster = reference_clusters[row_index]
            resampled_cluster = resampled_clusters[column_index]
            mapping[resampled_cluster] = reference_cluster
            matched_overlap += int(
                contingency.iloc[row_index, column_index]
            )

    n_evaluated = len(comparison)
    diagnostics = {
        "N Shared Companies": len(shared),
        "N Evaluated Companies": n_evaluated,
        "Alignment Accuracy": (
            matched_overlap / n_evaluated if n_evaluated else np.nan
        ),
        "Matched Company Count": matched_overlap,
        "Reference Cluster Count": len(reference_clusters),
        "Resampled Cluster Count": len(resampled_clusters),
        "Matched Cluster Count": len(mapping),
        "Unmatched Reference Cluster Count": (
            len(reference_clusters) - len(set(mapping.values()))
        ),
        "Unmatched Resampled Cluster Count": (
            len(resampled_clusters) - len(mapping)
        ),
        "Reference Noise Share": reference_noise_share,
        "Resampled Noise Share": resampled_noise_share,
    }
    return mapping, diagnostics


def empirical_interval_summary(
    values: pd.Series,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> dict:
    """Summarise a finite resampling distribution without CI language."""

    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return {
            "Median": np.nan,
            "Empirical Lower": np.nan,
            "Empirical Upper": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "N Runs": 0,
        }

    return {
        "Median": float(numeric_values.median()),
        "Empirical Lower": float(numeric_values.quantile(lower_quantile)),
        "Empirical Upper": float(numeric_values.quantile(upper_quantile)),
        "Minimum": float(numeric_values.min()),
        "Maximum": float(numeric_values.max()),
        "N Runs": int(numeric_values.size),
    }


def direction_consistency(
    resampled_differences: pd.Series,
    reference_difference: float,
) -> float:
    """Share of runs whose profile direction matches the reference."""

    differences = pd.to_numeric(
        resampled_differences,
        errors="coerce",
    ).dropna()
    if differences.empty or pd.isna(reference_difference):
        return np.nan

    reference_direction = np.sign(reference_difference)
    return float(
        (np.sign(differences.to_numpy()) == reference_direction).mean()
    )
