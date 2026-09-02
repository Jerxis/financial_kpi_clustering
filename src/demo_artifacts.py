"""Generate the first canonical demo solution and its presentation artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

from src.app_data import (
    APP_DATA_SCHEMA_VERSION,
    BUSINESS_CAPS,
    CANONICAL_KPIS,
    MIN_COMPANY_KPI_COMPLETENESS,
    PREPROCESSING_VERSION,
    SCENARIOS,
    SOURCE_ROW_ORDER_COLUMN,
    _write_csv,
    _write_json,
)
from src.models.filtering import filter_peer_universe


SOLUTION_SCHEMA_VERSION = "1.0.0"
TECHNOLOGY_SCENARIO_ID = "technology_revenue_25m"
METHOD = "K-Means"
RANDOM_STATE = 42
N_INIT = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_by_id(scenario_id: str):
    try:
        return next(
            scenario for scenario in SCENARIOS if scenario.scenario_id == scenario_id
        )
    except StopIteration as error:
        raise ValueError(f"Unknown canonical scenario: {scenario_id}") from error


def _technology_scenario():
    return _scenario_by_id(TECHNOLOGY_SCENARIO_ID)


def prepare_canonical_features(
    company_kpis: pd.DataFrame,
    scenario_id: str = TECHNOLOGY_SCENARIO_ID,
) -> dict[str, Any]:
    """Reproduce the final v3 feature pipeline for one canonical cohort."""

    scenario = _scenario_by_id(scenario_id)
    peer_universe = filter_peer_universe(
        company_kpis,
        sectors=list(scenario.sectors),
        revenue_column="Total Revenue",
        revenue_range=(scenario.revenue_minimum, scenario.revenue_maximum),
        exclude_revenue_outliers=False,
    )
    completeness = peer_universe[CANONICAL_KPIS].notna().mean(axis=1)
    peer_universe = (
        peer_universe.loc[completeness >= MIN_COMPANY_KPI_COMPLETENESS]
        .copy()
    )
    if SOURCE_ROW_ORDER_COLUMN not in peer_universe.columns:
        raise ValueError(
            "Application data is missing the canonical source-row identifier. "
            "Rebuild Phase 1 artifacts before fitting demo solutions."
        )
    peer_universe = peer_universe.sort_values(
        SOURCE_ROW_ORDER_COLUMN, kind="stable"
    ).reset_index(drop=True)
    if len(peer_universe) != scenario.expected_companies:
        raise AssertionError(
            f"{scenario.scenario_name} contains {len(peer_universe)} companies; "
            f"expected {scenario.expected_companies}."
        )

    raw_features = peer_universe[CANONICAL_KPIS].copy()
    missing_mask = raw_features.isna()

    imputer = SimpleImputer(strategy="median")
    imputed_features = pd.DataFrame(
        imputer.fit_transform(raw_features),
        columns=CANONICAL_KPIS,
        index=peer_universe.index,
    )

    winsor_bounds: dict[str, dict[str, float]] = {}
    winsorized_features = imputed_features.copy()
    for kpi in CANONICAL_KPIS:
        lower = float(imputed_features[kpi].quantile(0.05))
        upper = float(imputed_features[kpi].quantile(0.95))
        winsor_bounds[kpi] = {"lower": lower, "upper": upper}
        winsorized_features[kpi] = imputed_features[kpi].clip(lower, upper)

    capped_features = winsorized_features.copy()
    for kpi, (lower, upper) in BUSINESS_CAPS.items():
        if kpi in capped_features.columns:
            capped_features[kpi] = capped_features[kpi].clip(lower, upper)

    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(capped_features)
    model_features = pd.DataFrame(
        scaled_array,
        columns=CANONICAL_KPIS,
        index=peer_universe.index,
    )
    return {
        "peer_universe": peer_universe,
        "raw_features": raw_features,
        "missing_mask": missing_mask,
        "imputed_features": imputed_features,
        "winsorized_features": winsorized_features,
        "capped_features": capped_features,
        "model_features": model_features,
        "X_scaled": scaled_array,
        "imputer": imputer,
        "scaler": scaler,
        "winsor_bounds": winsor_bounds,
    }


def _load_selected_candidate(evidence_path: Path) -> pd.Series:
    evidence = pd.read_csv(evidence_path)
    selected = evidence[
        (evidence["Scenario ID"] == TECHNOLOGY_SCENARIO_ID)
        & (evidence["Model"] == METHOD)
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one canonical Technology K-Means result.")
    candidate = selected.iloc[0]
    if candidate["Preprocessing Version"] != PREPROCESSING_VERSION:
        raise ValueError("Technology K-Means evidence is not from the final v3 pipeline.")
    return candidate


def _assert_metric_agreement(metrics: dict[str, float], evidence: pd.Series) -> None:
    comparisons = {
        "silhouette": float(evidence["Common-Space Silhouette Score"]),
        "davies_bouldin": float(evidence["Common-Space Davies-Bouldin Index"]),
        "calinski_harabasz": float(
            evidence["Common-Space Calinski-Harabasz Score"]
        ),
        "inertia": float(evidence["Inertia"]),
    }
    for metric, expected in comparisons.items():
        actual = metrics[metric]
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise AssertionError(
                f"Rebuilt {metric}={actual!r} does not match v3 evidence {expected!r}."
            )


def _build_assignments(
    prepared: dict[str, Any],
    labels: np.ndarray,
    model: KMeans,
    projection: np.ndarray,
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
    assignments = prepared["peer_universe"][metadata_columns].copy()
    assignments.insert(0, "Scenario ID", TECHNOLOGY_SCENARIO_ID)
    assignments.insert(1, "Method", METHOD)
    assignments.insert(2, "Configuration", "K=2")
    assignments["Cluster"] = labels.astype(int)
    assignments["Cluster Label"] = assignments["Cluster"].map(
        lambda cluster: f"Cluster {cluster}"
    )
    assignments["PCA 1"] = projection[:, 0]
    assignments["PCA 2"] = projection[:, 1]

    distances = model.transform(prepared["X_scaled"])
    assigned_distance = distances[np.arange(len(labels)), labels]
    sorted_squared_distances = np.sort(distances**2, axis=1)
    assignments["Distance to Assigned Centroid"] = assigned_distance
    assignments["Nearest-Rival Squared-Distance Margin"] = (
        sorted_squared_distances[:, 1] - sorted_squared_distances[:, 0]
    )
    # Parallel BLAS reductions can vary at the final machine-epsilon digits.
    # Twelve decimal places preserve far more precision than the UI displays
    # while keeping exported artifacts byte-for-byte reproducible.
    stable_numeric_columns = [
        "PCA 1",
        "PCA 2",
        "Distance to Assigned Centroid",
        "Nearest-Rival Squared-Distance Margin",
    ]
    assignments[stable_numeric_columns] = assignments[stable_numeric_columns].round(12)
    assignments["Imputed Canonical KPI Count"] = prepared["missing_mask"].sum(axis=1)
    for kpi in CANONICAL_KPIS:
        assignments[kpi] = prepared["raw_features"][kpi]
    return assignments.sort_values(["Cluster", "Symbol"]).reset_index(drop=True)


def _build_profiles(
    prepared: dict[str, Any],
    labels: np.ndarray,
    kpi_catalog: pd.DataFrame,
) -> pd.DataFrame:
    raw = prepared["raw_features"]
    imputed = prepared["imputed_features"]
    model_space = prepared["model_features"]
    catalogue = kpi_catalog.set_index("KPI")
    rows: list[dict[str, Any]] = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        for kpi in CANONICAL_KPIS:
            cluster_raw = raw.loc[mask, kpi]
            cohort_raw = raw[kpi]
            cluster_imputed = imputed.loc[mask, kpi]
            rows.append(
                {
                    "Scenario ID": TECHNOLOGY_SCENARIO_ID,
                    "Method": METHOD,
                    "Configuration": "K=2",
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
                    "Cohort Median": float(cohort_raw.median()),
                    "Difference from Cohort Median": float(
                        cluster_raw.median() - cohort_raw.median()
                    ),
                    "Imputed Median": float(cluster_imputed.median()),
                    "Model-Space Median": float(model_space.loc[mask, kpi].median()),
                }
            )
    return pd.DataFrame(rows)


def _build_cluster_summary(
    prepared: dict[str, Any], labels: np.ndarray
) -> pd.DataFrame:
    metadata = prepared["peer_universe"].copy()
    metadata["Cluster"] = labels
    metadata["Imputed Canonical KPI Count"] = prepared["missing_mask"].sum(axis=1)
    rows = []
    for cluster, cluster_data in metadata.groupby("Cluster", sort=True):
        industry_counts = cluster_data["Industry"].value_counts()
        rows.append(
            {
                "Scenario ID": TECHNOLOGY_SCENARIO_ID,
                "Method": METHOD,
                "Cluster": int(cluster),
                "Cluster Label": f"Cluster {int(cluster)}",
                "Company Count": len(cluster_data),
                "Company Share": len(cluster_data) / len(metadata),
                "Median Revenue": cluster_data["Total Revenue"].median(),
                "Median Market Cap": cluster_data["MarketCap"].median(),
                "Median Employees": cluster_data["Employees"].median(),
                "Industry Count": cluster_data["Industry"].nunique(),
                "Most Common Industry": (
                    industry_counts.index[0] if len(industry_counts) else "Unavailable"
                ),
                "Most Common Industry Companies": (
                    int(industry_counts.iloc[0]) if len(industry_counts) else 0
                ),
                "Companies with Any Imputation": int(
                    cluster_data["Imputed Canonical KPI Count"].gt(0).sum()
                ),
                "Company Share with Any Imputation": cluster_data[
                    "Imputed Canonical KPI Count"
                ].gt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_technology_kmeans_artifacts(
    project_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate and publish the first complete canonical solution slice."""

    project_root = project_root.resolve()
    app_data_dir = project_root / "data" / "processed" / "app"
    if output_dir is None:
        output_dir = app_data_dir / "solutions" / TECHNOLOGY_SCENARIO_ID / "kmeans"
    output_dir = output_dir.resolve()

    company_path = app_data_dir / "company_kpis.csv.gz"
    kpi_catalog_path = app_data_dir / "kpi_catalog.csv"
    evidence_path = (
        project_root
        / "outputs"
        / "experiment_results"
        / "canonical_cross_cohort_robustness"
        / "best_traditional_by_scenario_method.csv"
    )
    source_paths = [company_path, kpi_catalog_path, evidence_path]
    missing = [str(path) for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase 2 sources are missing: {missing}")

    company_kpis = pd.read_csv(company_path)
    kpi_catalog = pd.read_csv(kpi_catalog_path)
    candidate = _load_selected_candidate(evidence_path)
    selected_k = int(candidate["K"])
    if selected_k != 2:
        raise AssertionError(f"Expected canonical Technology K=2, found K={selected_k}.")

    prepared = prepare_canonical_features(company_kpis)
    model = KMeans(
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    labels = model.fit_predict(prepared["X_scaled"])
    metrics = {
        "silhouette": float(silhouette_score(prepared["X_scaled"], labels)),
        "davies_bouldin": float(
            davies_bouldin_score(prepared["X_scaled"], labels)
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(prepared["X_scaled"], labels)
        ),
        "inertia": float(model.inertia_),
    }
    _assert_metric_agreement(metrics, candidate)

    pca = PCA(n_components=2)
    projection = pca.fit_transform(prepared["X_scaled"])
    assignments = _build_assignments(prepared, labels, model, projection)
    profiles = _build_profiles(prepared, labels, kpi_catalog)
    cluster_summary = _build_cluster_summary(prepared, labels)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="technology_kmeans_", dir=output_dir.parent))
    try:
        _write_csv(assignments, staging / "company_assignments.csv.gz", compressed=True)
        _write_csv(profiles, staging / "cluster_profiles.csv")
        _write_csv(cluster_summary, staging / "cluster_summary.csv")

        audit = {
            "scenario_id": TECHNOLOGY_SCENARIO_ID,
            "method": METHOD,
            "configuration": {"k": selected_k, "random_state": RANDOM_STATE, "n_init": N_INIT},
            "company_count": len(assignments),
            "canonical_kpis": CANONICAL_KPIS,
            "imputation": {
                "strategy": "scenario-local median",
                "statistics": dict(
                    zip(CANONICAL_KPIS, prepared["imputer"].statistics_.tolist())
                ),
            },
            "winsorisation": prepared["winsor_bounds"],
            "business_caps": BUSINESS_CAPS,
            "scaling": {
                "method": "RobustScaler",
                "center": dict(zip(CANONICAL_KPIS, prepared["scaler"].center_.tolist())),
                "scale": dict(zip(CANONICAL_KPIS, prepared["scaler"].scale_.tolist())),
            },
            "pca": {
                "purpose": "visualisation only; clustering used all eight scaled KPIs",
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "explained_variance_share": float(pca.explained_variance_ratio_.sum()),
                "components": {
                    f"PC{index + 1}": dict(zip(CANONICAL_KPIS, component.tolist()))
                    for index, component in enumerate(pca.components_)
                },
            },
            "rebuilt_metrics": metrics,
            "evidence_metrics": {
                "silhouette": float(candidate["Common-Space Silhouette Score"]),
                "davies_bouldin": float(
                    candidate["Common-Space Davies-Bouldin Index"]
                ),
                "calinski_harabasz": float(
                    candidate["Common-Space Calinski-Harabasz Score"]
                ),
                "inertia": float(candidate["Inertia"]),
            },
        }
        _write_json(staging / "preprocessing_audit.json", audit)

        output_files = [
            "company_assignments.csv.gz",
            "cluster_profiles.csv",
            "cluster_summary.csv",
            "preprocessing_audit.json",
        ]
        manifest = {
            "solution_schema_version": SOLUTION_SCHEMA_VERSION,
            "app_data_schema_version": APP_DATA_SCHEMA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_id": TECHNOLOGY_SCENARIO_ID,
            "method": METHOD,
            "configuration": "K=2",
            "cluster_label_note": (
                "Numeric cluster identifiers are arbitrary labels and have no ordinal meaning."
            ),
            "source_files": [
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in source_paths
            ],
            "outputs": {
                "company_assignments.csv.gz": {
                    "rows": len(assignments),
                    "columns": len(assignments.columns),
                },
                "cluster_profiles.csv": {
                    "rows": len(profiles),
                    "columns": len(profiles.columns),
                },
                "cluster_summary.csv": {
                    "rows": len(cluster_summary),
                    "columns": len(cluster_summary.columns),
                },
                "preprocessing_audit.json": {"records": 1},
            },
        }
        _write_json(staging / "solution_manifest.json", manifest)
        output_files.append("solution_manifest.json")

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in output_files:
            (staging / name).replace(output_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return manifest
