"""Build precomputed traditional solutions for every canonical demo cohort.

The app consumes these artifacts instead of fitting models during a live
demonstration.  Candidate selection and estimator parameters reproduce the
final v3 notebook: full-assignment methods use their selected canonical row,
while HDBSCAN uses the balanced member of the coverage--separation frontier.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.app_data import (
    APP_DATA_SCHEMA_VERSION,
    CANONICAL_KPIS,
    PREPROCESSING_VERSION,
    SCENARIOS,
    _write_csv,
    _write_json,
)
from src.demo_artifacts import prepare_canonical_features


SOLUTION_SCHEMA_VERSION = "2.1.0"
RANDOM_STATE = 42
TRADITIONAL_METHODS = (
    "K-Means",
    "Agglomerative",
    "Gaussian Mixture",
    "HDBSCAN",
)
INTERACTIVE_METHODS = (
    "K-Means",
    "Agglomerative",
    "Gaussian Mixture",
    "HDBSCAN",
)


def _select_balanced_hdbscan(frontier: pd.DataFrame, scenario_id: str) -> pd.Series:
    candidates = frontier[frontier["Scenario ID"].eq(scenario_id)].copy()
    if candidates.empty:
        raise ValueError(f"No HDBSCAN frontier rows for {scenario_id}.")
    candidates["Separation Rank"] = candidates[
        "Common-Space Silhouette Score"
    ].rank(ascending=False, method="min")
    candidates["Coverage Rank"] = candidates["Evaluation Coverage"].rank(
        ascending=False, method="min"
    )
    candidates["Balanced Rank Sum"] = (
        candidates["Separation Rank"] + candidates["Coverage Rank"]
    )
    return candidates.sort_values(
        [
            "Balanced Rank Sum",
            "Evaluation Coverage",
            "Common-Space Silhouette Score",
            "Min Cluster Size",
            "Min Samples",
        ],
        ascending=[True, False, False, True, True],
    ).iloc[0]


def _candidate_for(
    selected: pd.DataFrame,
    frontier: pd.DataFrame,
    scenario_id: str,
    method: str,
) -> pd.Series:
    if method == "HDBSCAN":
        return _select_balanced_hdbscan(frontier, scenario_id)
    rows = selected[
        selected["Scenario ID"].eq(scenario_id) & selected["Model"].eq(method)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected {scenario_id}/{method} row.")
    return rows.iloc[0]


def _configuration(candidate: pd.Series) -> str:
    if candidate["Model"] == "HDBSCAN":
        return (
            f"min_cluster_size={int(candidate['Min Cluster Size'])}, "
            f"min_samples={int(candidate['Min Samples'])}"
        )
    return f"K={int(candidate['K'])}"


def _fit_candidate(
    X: np.ndarray, candidate: pd.Series
) -> tuple[Any, np.ndarray, str]:
    method = candidate["Model"]
    k = int(candidate["K"])
    if method == "K-Means":
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    elif method == "Gaussian Mixture":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=10,
        )
    elif method == "HDBSCAN":
        model = HDBSCAN(
            min_cluster_size=int(candidate["Min Cluster Size"]),
            min_samples=int(candidate["Min Samples"]),
            copy=True,
        )
    else:
        raise ValueError(f"Unsupported traditional method: {method}")
    labels = model.fit_predict(X)
    return model, labels, "Canonical estimator parameters and source row order"


def _evaluate(X: np.ndarray, labels: np.ndarray, method: str) -> dict[str, float]:
    evaluated = labels != -1 if method == "HDBSCAN" else np.ones(len(labels), dtype=bool)
    evaluated_labels = labels[evaluated]
    evaluated_X = X[evaluated]
    return {
        "silhouette": float(silhouette_score(evaluated_X, evaluated_labels)),
        "davies_bouldin": float(davies_bouldin_score(evaluated_X, evaluated_labels)),
        "calinski_harabasz": float(
            calinski_harabasz_score(evaluated_X, evaluated_labels)
        ),
        "coverage": float(evaluated.mean()),
        "noise_share": float((labels == -1).mean()),
    }


def _assert_evidence(metrics: dict[str, float], candidate: pd.Series) -> dict[str, float]:
    expected = {
        "silhouette": float(candidate["Common-Space Silhouette Score"]),
        "davies_bouldin": float(candidate["Common-Space Davies-Bouldin Index"]),
        "calinski_harabasz": float(
            candidate["Common-Space Calinski-Harabasz Score"]
        ),
        "coverage": float(candidate["Evaluation Coverage"]),
    }
    deltas = {name: metrics[name] - value for name, value in expected.items()}
    for name, expected_value in expected.items():
        exact = np.isclose(metrics[name], expected_value, rtol=1e-10, atol=1e-12)
        if not exact:
            raise AssertionError(
                f"Rebuilt {name}={metrics[name]!r} does not match evidence "
                f"{expected_value!r} for {candidate['Scenario ID']}/{candidate['Model']}."
            )
    return deltas


def _assignments(
    prepared: dict[str, Any],
    labels: np.ndarray,
    model: Any,
    projection: np.ndarray,
    scenario_id: str,
    method: str,
    configuration: str,
) -> pd.DataFrame:
    metadata_columns = [
        "Symbol",
        "Company Name",
        "Exchange",
        "Sector",
        "Industry",
        "Employees",
        "MarketCap",
        "EnterpriseValue",
        "Total Revenue",
    ]
    frame = prepared["peer_universe"][metadata_columns].copy()
    frame.insert(0, "Scenario ID", scenario_id)
    frame.insert(1, "Method", method)
    frame.insert(2, "Configuration", configuration)
    frame["Cluster"] = labels.astype(int)
    frame["Cluster Label"] = frame["Cluster"].map(
        lambda value: "Noise / unassigned" if value == -1 else f"Cluster {value}"
    )
    frame["PCA 1"] = projection[:, 0]
    frame["PCA 2"] = projection[:, 1]
    frame["Assignment Confidence"] = np.nan
    frame["Assignment Confidence Label"] = "Not available for this method"

    X = prepared["X_scaled"]
    if method == "K-Means":
        distances = model.transform(X)
        squared = np.sort(distances**2, axis=1)
        frame["Assignment Confidence"] = squared[:, 1] - squared[:, 0]
        frame["Assignment Confidence Label"] = "Nearest-rival squared-distance margin"
    elif method == "Gaussian Mixture":
        frame["Assignment Confidence"] = model.predict_proba(X).max(axis=1)
        frame["Assignment Confidence Label"] = "Maximum posterior probability"
    elif method == "HDBSCAN":
        frame["Assignment Confidence"] = model.probabilities_
        frame["Assignment Confidence Label"] = "HDBSCAN membership strength"

    frame["Imputed Canonical KPI Count"] = prepared["missing_mask"].sum(axis=1)
    for kpi in CANONICAL_KPIS:
        frame[kpi] = prepared["raw_features"][kpi]
    stable = ["PCA 1", "PCA 2", "Assignment Confidence"]
    frame[stable] = frame[stable].round(12)
    return frame.sort_values(["Cluster", "Symbol"]).reset_index(drop=True)


def _profiles(
    prepared: dict[str, Any],
    labels: np.ndarray,
    kpi_catalog: pd.DataFrame,
    scenario_id: str,
    method: str,
    configuration: str,
) -> pd.DataFrame:
    raw = prepared["raw_features"]
    imputed = prepared["imputed_features"]
    model_space = prepared["model_features"]
    catalogue = kpi_catalog.set_index("KPI")
    rows: list[dict[str, Any]] = []
    for cluster in sorted(value for value in np.unique(labels) if value != -1):
        mask = labels == cluster
        for kpi in CANONICAL_KPIS:
            cluster_raw = raw.loc[mask, kpi]
            rows.append(
                {
                    "Scenario ID": scenario_id,
                    "Method": method,
                    "Configuration": configuration,
                    "Cluster": int(cluster),
                    "KPI": kpi,
                    "Category": catalogue.loc[kpi, "Category"],
                    "UoM": catalogue.loc[kpi, "UoM"],
                    "What is better?": catalogue.loc[kpi, "What is better?"],
                    "Company Count": int(mask.sum()),
                    "Observed Count": int(cluster_raw.notna().sum()),
                    "Imputed Count": int(cluster_raw.isna().sum()),
                    "Imputed Share": float(cluster_raw.isna().mean()),
                    "Bottom Quartile": float(cluster_raw.quantile(0.25)),
                    "Median": float(cluster_raw.median()),
                    "Top Quartile": float(cluster_raw.quantile(0.75)),
                    "Cohort Median": float(raw[kpi].median()),
                    "Difference from Cohort Median": float(
                        cluster_raw.median() - raw[kpi].median()
                    ),
                    "Imputed Median": float(imputed.loc[mask, kpi].median()),
                    "Model-Space Median": float(model_space.loc[mask, kpi].median()),
                }
            )
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=["float"]).columns
    result[numeric] = result[numeric].round(12)
    return result


def _cluster_summary(
    prepared: dict[str, Any],
    labels: np.ndarray,
    scenario_id: str,
    method: str,
    configuration: str,
) -> pd.DataFrame:
    metadata = prepared["peer_universe"].copy()
    metadata["Cluster"] = labels
    metadata["Imputed Canonical KPI Count"] = prepared["missing_mask"].sum(axis=1)
    rows: list[dict[str, Any]] = []
    for cluster, group in metadata.groupby("Cluster", sort=True):
        industries = group["Industry"].value_counts()
        rows.append(
            {
                "Scenario ID": scenario_id,
                "Method": method,
                "Configuration": configuration,
                "Cluster": int(cluster),
                "Cluster Label": (
                    "Noise / unassigned" if cluster == -1 else f"Cluster {int(cluster)}"
                ),
                "Company Count": len(group),
                "Company Share": len(group) / len(metadata),
                "Median Revenue": group["Total Revenue"].median(),
                "Median Market Cap": group["MarketCap"].median(),
                "Median Employees": group["Employees"].median(),
                "Industry Count": group["Industry"].nunique(),
                "Most Common Industry": (
                    industries.index[0] if len(industries) else "Unavailable"
                ),
                "Most Common Industry Companies": (
                    int(industries.iloc[0]) if len(industries) else 0
                ),
                "Companies with Any Imputation": int(
                    group["Imputed Canonical KPI Count"].gt(0).sum()
                ),
                "Company Share with Any Imputation": float(
                    group["Imputed Canonical KPI Count"].gt(0).mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=["float"]).columns
    result[numeric] = result[numeric].round(12)
    return result


def build_canonical_traditional_solutions(
    project_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate the twenty precomputed scenario/method solution slices."""

    project_root = project_root.resolve()
    app_data_dir = project_root / "data" / "processed" / "app"
    output_dir = (output_dir or app_data_dir / "solutions").resolve()
    evidence_dir = (
        project_root
        / "outputs"
        / "experiment_results"
        / "canonical_cross_cohort_robustness"
    )
    company_kpis = pd.read_csv(app_data_dir / "company_kpis.csv.gz")
    kpi_catalog = pd.read_csv(app_data_dir / "kpi_catalog.csv")
    selected = pd.read_csv(evidence_dir / "best_traditional_by_scenario_method.csv")
    frontier = pd.read_csv(evidence_dir / "hdbscan_pareto_frontier.csv")

    for evidence in (selected, frontier):
        versions = set(evidence["Preprocessing Version"].dropna())
        if versions != {PREPROCESSING_VERSION}:
            raise ValueError(f"Unexpected preprocessing evidence versions: {versions}")

    assignment_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    solution_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        prepared = prepare_canonical_features(company_kpis, scenario.scenario_id)
        pca = PCA(n_components=2)
        projection = pca.fit_transform(prepared["X_scaled"])
        for method in TRADITIONAL_METHODS:
            candidate = _candidate_for(
                selected, frontier, scenario.scenario_id, method
            )
            model, labels, fit_note = _fit_candidate(
                prepared["X_scaled"], candidate
            )
            metrics = _evaluate(prepared["X_scaled"], labels, method)
            evidence_deltas = _assert_evidence(metrics, candidate)
            configuration = _configuration(candidate)
            assignment_frames.append(
                _assignments(
                    prepared,
                    labels,
                    model,
                    projection,
                    scenario.scenario_id,
                    method,
                    configuration,
                )
            )
            profile_frames.append(
                _profiles(
                    prepared,
                    labels,
                    kpi_catalog,
                    scenario.scenario_id,
                    method,
                    configuration,
                )
            )
            cluster_frames.append(
                _cluster_summary(
                    prepared,
                    labels,
                    scenario.scenario_id,
                    method,
                    configuration,
                )
            )
            solution_rows.append(
                {
                    "Scenario ID": scenario.scenario_id,
                    "Scenario Name": scenario.scenario_name,
                    "Method": method,
                    "Configuration": configuration,
                    "K": int(candidate["K"]),
                    "Company Count": len(labels),
                    "Assigned Company Count": int((labels != -1).sum()),
                    "Cluster Count": int(len(set(labels) - {-1})),
                    "Silhouette": metrics["silhouette"],
                    "Davies-Bouldin": metrics["davies_bouldin"],
                    "Calinski-Harabasz": metrics["calinski_harabasz"],
                    "Frozen Evidence Silhouette": float(
                        candidate["Common-Space Silhouette Score"]
                    ),
                    "Frozen Evidence Davies-Bouldin": float(
                        candidate["Common-Space Davies-Bouldin Index"]
                    ),
                    "Frozen Evidence Calinski-Harabasz": float(
                        candidate["Common-Space Calinski-Harabasz Score"]
                    ),
                    "Maximum Relative Evidence Delta": max(
                        abs(evidence_deltas[name])
                        / max(1.0, abs(float(candidate[column])))
                        for name, column in {
                            "silhouette": "Common-Space Silhouette Score",
                            "davies_bouldin": "Common-Space Davies-Bouldin Index",
                            "calinski_harabasz": "Common-Space Calinski-Harabasz Score",
                            "coverage": "Evaluation Coverage",
                        }.items()
                    ),
                    "Evidence Reproduction": "Exact match to frozen v3 evidence",
                    "Evaluation Coverage": metrics["coverage"],
                    "Noise Share": metrics["noise_share"],
                    "PCA Explained Variance Share": float(
                        pca.explained_variance_ratio_.sum()
                    ),
                    "Selection Rule": (
                        "Balanced coverage-separation Pareto member"
                        if method == "HDBSCAN"
                        else "Canonical full-assignment candidate"
                    ),
                    "Reconstruction Note": fit_note,
                    "Interactive Assignment Available": True,
                }
            )

    assignments = pd.concat(assignment_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    clusters = pd.concat(cluster_frames, ignore_index=True)
    solutions = pd.DataFrame(solution_rows)
    numeric = solutions.select_dtypes(include=["float"]).columns
    solutions[numeric] = solutions[numeric].round(12)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="canonical_solutions_", dir=output_dir))
    try:
        _write_csv(
            assignments,
            staging / "traditional_company_assignments.csv.gz",
            compressed=True,
        )
        _write_csv(profiles, staging / "traditional_cluster_profiles.csv")
        _write_csv(clusters, staging / "traditional_cluster_summary.csv")
        _write_csv(solutions, staging / "traditional_solution_summary.csv")
        manifest = {
            "solution_schema_version": SOLUTION_SCHEMA_VERSION,
            "app_data_schema_version": APP_DATA_SCHEMA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_policy": {
                "full_assignment": "Canonical selected candidate from final v3 evidence",
                "hdbscan": "Balanced coverage-separation Pareto member",
                "row_order": "Original source-workbook order before fitting",
                "random_state": RANDOM_STATE,
            },
            "scenario_count": len(SCENARIOS),
            "method_count": len(TRADITIONAL_METHODS),
            "interactive_method_count": len(INTERACTIVE_METHODS),
            "solution_count": len(solutions),
            "outputs": {
                "traditional_company_assignments.csv.gz": {"rows": len(assignments)},
                "traditional_cluster_profiles.csv": {"rows": len(profiles)},
                "traditional_cluster_summary.csv": {"rows": len(clusters)},
                "traditional_solution_summary.csv": {"rows": len(solutions)},
            },
        }
        _write_json(staging / "traditional_solution_manifest.json", manifest)
        for path in staging.iterdir():
            path.replace(output_dir / path.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest
