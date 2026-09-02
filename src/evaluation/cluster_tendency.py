"""Cluster-tendency diagnostics for canonical clustering experiments."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def _validate_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Return a finite two-dimensional floating-point feature matrix."""

    validated_X = np.asarray(X, dtype=float)

    if validated_X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if len(validated_X) < 3:
        raise ValueError("At least three observations are required.")
    if validated_X.shape[1] < 1:
        raise ValueError("At least one feature is required.")
    if not np.isfinite(validated_X).all():
        raise ValueError("X must contain only finite values.")

    return validated_X


def calculate_hopkins_statistic(
    X: np.ndarray,
    sample_size: int,
    random_state: int,
) -> float:
    """Calculate one Hopkins statistic against a uniform reference.

    Values near 0.5 are consistent with spatial randomness under this
    reference, while larger values indicate greater clustering tendency.
    The statistic is diagnostic rather than a universal hypothesis test.
    """

    validated_X = _validate_feature_matrix(X)
    n_observations, n_features = validated_X.shape

    if not 2 <= sample_size < n_observations:
        raise ValueError(
            "sample_size must be at least 2 and smaller than the cohort."
        )

    rng = np.random.default_rng(random_state)
    sampled_indices = rng.choice(
        n_observations,
        size=sample_size,
        replace=False,
    )

    observed_neighbors = NearestNeighbors(n_neighbors=2).fit(validated_X)
    observed_distances = observed_neighbors.kneighbors(
        validated_X[sampled_indices],
        return_distance=True,
    )[0][:, 1]

    feature_minimums = validated_X.min(axis=0)
    feature_maximums = validated_X.max(axis=0)
    uniform_reference = rng.uniform(
        low=feature_minimums,
        high=feature_maximums,
        size=(sample_size, n_features),
    )

    reference_neighbors = NearestNeighbors(n_neighbors=1).fit(validated_X)
    reference_distances = reference_neighbors.kneighbors(
        uniform_reference,
        return_distance=True,
    )[0][:, 0]

    observed_power_sum = np.power(
        observed_distances,
        n_features,
    ).sum()
    reference_power_sum = np.power(
        reference_distances,
        n_features,
    ).sum()
    denominator = observed_power_sum + reference_power_sum

    if denominator == 0:
        return 0.5

    return float(reference_power_sum / denominator)


def run_repeated_hopkins(
    X: np.ndarray,
    n_repeats: int,
    sample_fraction: float,
    base_seed: int,
    maximum_sample_size: int = 100,
) -> pd.DataFrame:
    """Repeat Hopkins sampling to expose Monte Carlo variability."""

    validated_X = _validate_feature_matrix(X)

    if n_repeats < 1:
        raise ValueError("n_repeats must be positive.")
    if not 0 < sample_fraction < 1:
        raise ValueError("sample_fraction must lie between zero and one.")

    sample_size = min(
        maximum_sample_size,
        max(10, int(len(validated_X) * sample_fraction)),
    )
    sample_size = min(sample_size, len(validated_X) - 1)

    return pd.DataFrame([
        {
            "Replicate": replicate,
            "Seed": base_seed + replicate,
            "Sample Size": sample_size,
            "Hopkins Statistic": calculate_hopkins_statistic(
                X=validated_X,
                sample_size=sample_size,
                random_state=base_seed + replicate,
            ),
        }
        for replicate in range(n_repeats)
    ])


def select_best_kmeans_by_silhouette(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int,
    n_init: int,
) -> dict:
    """Select K-Means K using the highest Silhouette in a declared range."""

    validated_X = _validate_feature_matrix(X)
    candidate_records = []

    for k in k_values:
        if not 2 <= k < len(validated_X):
            continue

        labels = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        ).fit_predict(validated_X)

        candidate_records.append({
            "K": int(k),
            "Silhouette": float(
                silhouette_score(validated_X, labels)
            ),
        })

    if not candidate_records:
        raise ValueError("No valid K values were supplied.")

    return max(
        candidate_records,
        key=lambda record: record["Silhouette"],
    )


def run_selection_matched_permutation_null(
    X: np.ndarray,
    k_values: Iterable[int],
    n_replicates: int,
    n_init: int,
    base_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Compare observed structure with independently permuted KPI columns.

    Each null replicate preserves every feature's marginal distribution while
    destroying cross-feature company-level relationships. The same K search
    and K-Means fitting budget are applied to observed and null data, avoiding
    an unfair model-selection advantage for the observed cohort.
    """

    validated_X = _validate_feature_matrix(X)
    declared_k_values = list(k_values)

    if n_replicates < 1:
        raise ValueError("n_replicates must be positive.")
    if n_init < 1:
        raise ValueError("n_init must be positive.")

    observed_result = select_best_kmeans_by_silhouette(
        X=validated_X,
        k_values=declared_k_values,
        random_state=base_seed,
        n_init=n_init,
    )

    null_records = []
    for replicate in range(n_replicates):
        replicate_seed = base_seed + replicate + 1
        rng = np.random.default_rng(replicate_seed)
        null_X = np.column_stack([
            rng.permutation(validated_X[:, feature_index])
            for feature_index in range(validated_X.shape[1])
        ])

        null_result = select_best_kmeans_by_silhouette(
            X=null_X,
            k_values=declared_k_values,
            random_state=replicate_seed,
            n_init=n_init,
        )
        null_records.append({
            "Null Replicate": replicate,
            "Seed": replicate_seed,
            "Selected K": null_result["K"],
            "Best Silhouette": null_result["Silhouette"],
        })

    null_results_df = pd.DataFrame(null_records)
    null_silhouettes = null_results_df["Best Silhouette"]
    null_standard_deviation = null_silhouettes.std(ddof=1)

    summary = {
        "Observed Selected K": observed_result["K"],
        "Observed Best Silhouette": observed_result["Silhouette"],
        "Null Mean Best Silhouette": null_silhouettes.mean(),
        "Null Std Best Silhouette": null_standard_deviation,
        "Null 95th Percentile Best Silhouette": null_silhouettes.quantile(
            0.95
        ),
        "Observed Minus Null Mean": (
            observed_result["Silhouette"] - null_silhouettes.mean()
        ),
        "Standardised Observed-Null Gap": (
            (
                observed_result["Silhouette"]
                - null_silhouettes.mean()
            )
            / null_standard_deviation
            if null_standard_deviation > 0
            else np.nan
        ),
        "Empirical Tail Probability": (
            1
            + int(
                (
                    null_silhouettes
                    >= observed_result["Silhouette"]
                ).sum()
            )
        ) / (n_replicates + 1),
        "Null Replicates": n_replicates,
        "K-Means Initialisations per K": n_init,
    }

    return null_results_df, summary
