"""Build the compact, versioned data layer consumed by the demo application.

The application data layer is derived from the cleaned financial statement
workbook and the final ``scenario_pipeline_v3_economic_validity`` experiment
artifacts.  It deliberately excludes model fitting and company assignments;
those are added as a separate, explicitly versioned phase so the demo never
silently recomputes or mixes results from different research pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.features.kpi_engine import (
    apply_kpi_economic_validity_mask,
    calculate_kpis,
    diagnose_kpi_economic_validity,
    load_kpi_definitions,
    validate_kpi_definitions,
)
from src.models.filtering import filter_peer_universe


APP_DATA_SCHEMA_VERSION = "1.1.0"
PREPROCESSING_VERSION = "scenario_pipeline_v3_economic_validity"
SOURCE_ROW_ORDER_COLUMN = "Source Row Order"
MINIMUM_POSITIVE_NET_INCOME_MARGIN = 0.01
MIN_COMPANY_KPI_COMPLETENESS = 0.70
MIN_COMPANIES_REQUIRED = 30

CANONICAL_KPIS = [
    "EBITDA Margin",
    "Return on Assets",
    "Asset Turnover",
    "Revenue per Employee",
    "Current Ratio",
    "Debt Ratio",
    "Free Cash Flow Margin",
    "Operating Cash Flow Margin",
]

BUSINESS_CAPS = {
    "EBITDA Margin": (-1.0, 1.0),
    "Free Cash Flow Margin": (-1.0, 1.0),
    "Operating Cash Flow Margin": (-1.0, 1.0),
    "Return on Assets": (-1.0, 1.0),
    "Cash Conversion Ratio": (-5.0, 5.0),
    "Current Ratio": (0.0, 10.0),
    "Debt Ratio": (0.0, 2.0),
}

NON_FINANCIAL_SECTORS = [
    "Technology",
    "Healthcare",
    "Industrials",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Utilities",
    "Basic Materials",
    "Communication Services",
    "Real Estate",
]


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    scenario_name: str
    sectors: tuple[str, ...]
    revenue_minimum: float
    revenue_maximum: float
    expected_companies: int


SCENARIOS = (
    ScenarioDefinition(
        scenario_id="technology_revenue_25m",
        scenario_name="Technology companies with revenue >= $25M",
        sectors=("Technology",),
        revenue_minimum=25_000_000,
        revenue_maximum=np.inf,
        expected_companies=587,
    ),
    ScenarioDefinition(
        scenario_id="healthcare_revenue_50m",
        scenario_name="Healthcare companies with revenue >= $50M",
        sectors=("Healthcare",),
        revenue_minimum=50_000_000,
        revenue_maximum=np.inf,
        expected_companies=444,
    ),
    ScenarioDefinition(
        scenario_id="industrials_revenue_20m",
        scenario_name="Industrials companies with revenue >= $20M",
        sectors=("Industrials",),
        revenue_minimum=20_000_000,
        revenue_maximum=np.inf,
        expected_companies=565,
    ),
    ScenarioDefinition(
        scenario_id="consumer_cyclical_revenue_20m",
        scenario_name="Consumer Cyclical companies with revenue >= $20M",
        sectors=("Consumer Cyclical",),
        revenue_minimum=20_000_000,
        revenue_maximum=np.inf,
        expected_companies=470,
    ),
    ScenarioDefinition(
        scenario_id="non_financial_large_cap_revenue_1b",
        scenario_name="Non-financial companies with revenue >= $1B",
        sectors=tuple(NON_FINANCIAL_SECTORS),
        revenue_minimum=1_000_000_000,
        revenue_maximum=np.inf,
        expected_companies=1_651,
    ),
)


METHOD_CATALOG = (
    {
        "method_id": "kmeans",
        "method": "K-Means",
        "family": "Traditional centroid-based",
        "assignment_type": "Full assignment",
        "cluster_count": "Required in advance",
        "strengths": [
            "Fast, reproducible and straightforward to interpret",
            "Strong baseline for compact groups in a scaled feature space",
        ],
        "limitations": [
            "Favours roughly convex clusters",
            "Sensitive to scaling, outliers and the selected number of clusters",
        ],
        "canonical_v3_evidence": True,
        "demo_role": "Primary production-oriented baseline",
    },
    {
        "method_id": "agglomerative",
        "method": "Agglomerative",
        "family": "Traditional hierarchical",
        "assignment_type": "Full assignment",
        "cluster_count": "Cut selected from a hierarchy",
        "strengths": [
            "Exposes nested structure without centroid assumptions",
            "Useful interpretable comparison with K-Means",
        ],
        "limitations": [
            "Early merges cannot be reversed",
            "Results depend on linkage and distance choices",
        ],
        "canonical_v3_evidence": True,
        "demo_role": "Hierarchical full-assignment alternative",
    },
    {
        "method_id": "gaussian_mixture",
        "method": "Gaussian Mixture",
        "family": "Probabilistic mixture",
        "assignment_type": "Full assignment with probabilities",
        "cluster_count": "Required in advance",
        "strengths": [
            "Provides soft membership probabilities",
            "Allows elliptical rather than only spherical groups",
        ],
        "limitations": [
            "Relies on distributional assumptions",
            "Can be unstable when components overlap or covariance estimates weaken",
        ],
        "canonical_v3_evidence": True,
        "demo_role": "Probabilistic comparison",
    },
    {
        "method_id": "hdbscan",
        "method": "HDBSCAN",
        "family": "Density-based",
        "assignment_type": "Partial assignment; noise permitted",
        "cluster_count": "Inferred from density",
        "strengths": [
            "Finds irregular dense groups and identifies noise",
            "Does not require a fixed cluster count",
        ],
        "limitations": [
            "May leave many companies unassigned",
            "Separation scores must be interpreted jointly with coverage",
        ],
        "canonical_v3_evidence": True,
        "demo_role": "Dense-archetype discovery diagnostic",
    },
    {
        "method_id": "autoencoder_kmeans",
        "method": "Autoencoder + K-Means",
        "family": "Neural representation plus centroid-based",
        "assignment_type": "Full assignment",
        "cluster_count": "Required after representation learning",
        "strengths": [
            "Can learn nonlinear low-dimensional representations",
            "Provides a bridge between classical and deep clustering",
        ],
        "limitations": [
            "Sensitive to architecture and random initialisation",
            "Native-space separation may not persist in the common KPI space",
        ],
        "canonical_v3_evidence": True,
        "demo_role": "Neural representation benchmark",
    },
    {
        "method_id": "dec",
        "method": "DEC",
        "family": "Deep embedded clustering",
        "assignment_type": "Full assignment",
        "cluster_count": "Required in advance",
        "strengths": [
            "Jointly refines representation and cluster assignments",
            "Can sharpen separation in its learned latent space",
        ],
        "limitations": [
            "Stochastic and computationally expensive",
            "Improved native separation does not guarantee stable assignments",
        ],
        "canonical_v3_evidence": True,
        "demo_role": "Deep-clustering research candidate",
    },
    {
        "method_id": "idec",
        "method": "IDEC",
        "family": "Deep embedded clustering with reconstruction",
        "assignment_type": "Full assignment",
        "cluster_count": "Required in advance",
        "strengths": [
            "Retains a reconstruction objective during clustering refinement",
            "Aims to preserve local data structure better than DEC",
        ],
        "limitations": [
            "Stochastic and computationally expensive",
            "Not included in the final five-cohort v3 robustness evidence",
        ],
        "canonical_v3_evidence": False,
        "demo_role": "Exploratory method; clearly separated from canonical v3 comparison",
    },
)


REQUIRED_FINANCIAL_COLUMNS = [
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

SUPPORT_COLUMNS = [
    "Employees",
    "MarketCap",
    "EnterpriseValue",
    "Total Revenue",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_normalise_json_value(payload), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    if compressed:
        # gzip otherwise embeds the current timestamp, causing identical data
        # builds to produce different file hashes.  A fixed mtime keeps the
        # application artifacts byte-for-byte reproducible.
        with path.open("wb") as raw_target:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_target,
                mtime=0,
            ) as compressed_target:
                with io.TextIOWrapper(
                    compressed_target,
                    encoding="utf-8",
                    newline="",
                ) as text_target:
                    frame.to_csv(text_target, index=False, lineterminator="\n")
    else:
        frame.to_csv(path, index=False, lineterminator="\n")


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_company_kpis(
    financials: pd.DataFrame,
    kpi_definitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate, economically validate and annotate the application KPI table."""

    _validate_columns(financials, REQUIRED_FINANCIAL_COLUMNS, "Clean financial data")
    if financials["Symbol"].isna().any():
        raise ValueError("Clean financial data contains missing symbols.")
    if financials["Symbol"].duplicated().any():
        duplicates = financials.loc[
            financials["Symbol"].duplicated(keep=False), "Symbol"
        ].tolist()
        raise ValueError(f"Clean financial data contains duplicate symbols: {duplicates[:10]}")

    validation = validate_kpi_definitions(financials, kpi_definitions)
    unavailable = validation.loc[~validation["Can Calculate"], "KPI"].tolist()
    if unavailable:
        raise ValueError(f"KPI definitions cannot be calculated: {unavailable}")

    calculated = calculate_kpis(financials, kpi_definitions)
    calculated = calculated.merge(
        financials[["Symbol", *SUPPORT_COLUMNS]],
        on="Symbol",
        how="left",
        validate="one_to_one",
    )

    validity_audit = diagnose_kpi_economic_validity(
        financials,
        minimum_positive_net_income_margin=MINIMUM_POSITIVE_NET_INCOME_MARGIN,
    )
    masked = apply_kpi_economic_validity_mask(calculated, validity_audit)
    source_order = pd.Series(
        np.arange(len(financials), dtype=int),
        index=financials["Symbol"],
    )
    masked[SOURCE_ROW_ORDER_COLUMN] = (
        masked["Symbol"].map(source_order).astype(int)
    )

    kpi_names = kpi_definitions["KPI"].tolist()
    missing_canonical = [kpi for kpi in CANONICAL_KPIS if kpi not in kpi_names]
    if missing_canonical:
        raise ValueError(f"Canonical KPI definitions are missing: {missing_canonical}")

    for kpi in kpi_names:
        masked[f"Available::{kpi}"] = masked[kpi].notna()

    masked["Canonical KPI Available Count"] = masked[CANONICAL_KPIS].notna().sum(axis=1)
    masked["Canonical KPI Completeness"] = (
        masked["Canonical KPI Available Count"] / len(CANONICAL_KPIS)
    )
    masked["Meets Canonical Completeness Rule"] = (
        masked["Canonical KPI Completeness"] >= MIN_COMPANY_KPI_COMPLETENESS
    )

    ordered_columns = [
        "Symbol",
        SOURCE_ROW_ORDER_COLUMN,
        "Company Name",
        "Exchange",
        "Sector",
        "Industry",
        *SUPPORT_COLUMNS,
        *kpi_names,
        *[f"Available::{kpi}" for kpi in kpi_names],
        "Canonical KPI Available Count",
        "Canonical KPI Completeness",
        "Meets Canonical Completeness Rule",
    ]
    return masked[ordered_columns].sort_values("Symbol").reset_index(drop=True), validity_audit


def build_scenario_tables(
    company_kpis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create canonical preset metadata and auditable company membership."""

    catalog_rows: list[dict[str, Any]] = []
    membership_frames: list[pd.DataFrame] = []

    for scenario in SCENARIOS:
        universe = filter_peer_universe(
            company_kpis,
            sectors=list(scenario.sectors),
            revenue_column="Total Revenue",
            revenue_range=(scenario.revenue_minimum, scenario.revenue_maximum),
            exclude_revenue_outliers=False,
        )
        universe["Canonical KPI Available Count"] = universe[CANONICAL_KPIS].notna().sum(axis=1)
        universe["Canonical KPI Completeness"] = (
            universe["Canonical KPI Available Count"] / len(CANONICAL_KPIS)
        )
        universe["Retained for Clustering"] = (
            universe["Canonical KPI Completeness"] >= MIN_COMPANY_KPI_COMPLETENESS
        )
        universe["Imputed Canonical KPI Count"] = np.where(
            universe["Retained for Clustering"],
            len(CANONICAL_KPIS) - universe["Canonical KPI Available Count"],
            np.nan,
        )
        universe["Missing Canonical KPIs"] = universe[CANONICAL_KPIS].apply(
            lambda row: " | ".join(row.index[row.isna()].tolist()),
            axis=1,
        )

        retained_count = int(universe["Retained for Clustering"].sum())
        if retained_count < MIN_COMPANIES_REQUIRED:
            raise ValueError(
                f"Scenario '{scenario.scenario_id}' retains only {retained_count} companies."
            )
        if retained_count != scenario.expected_companies:
            raise AssertionError(
                f"Scenario '{scenario.scenario_id}' retained {retained_count} companies; "
                f"expected {scenario.expected_companies} from the final v3 evidence."
            )

        actual_minimum = float(universe["Total Revenue"].min())
        if actual_minimum < scenario.revenue_minimum:
            raise AssertionError(
                f"Scenario '{scenario.scenario_id}' contains revenue below its declared minimum."
            )

        retained = universe[universe["Retained for Clustering"]]
        catalog_rows.append(
            {
                "Scenario ID": scenario.scenario_id,
                "Scenario Name": scenario.scenario_name,
                "Sectors": " | ".join(scenario.sectors),
                "Revenue Minimum": scenario.revenue_minimum,
                "Revenue Maximum": scenario.revenue_maximum,
                "Companies Before Completeness": len(universe),
                "Companies Retained": retained_count,
                "Companies Excluded for Completeness": len(universe) - retained_count,
                "Mean Canonical KPI Completeness": retained[
                    "Canonical KPI Completeness"
                ].mean(),
                "Companies with Any Imputation": int(
                    retained["Imputed Canonical KPI Count"].gt(0).sum()
                ),
                "Company Share with Any Imputation": retained[
                    "Imputed Canonical KPI Count"
                ].gt(0).mean(),
                "Minimum Company Completeness": MIN_COMPANY_KPI_COMPLETENESS,
                "Canonical KPI Count": len(CANONICAL_KPIS),
                "Canonical KPIs": " | ".join(CANONICAL_KPIS),
                "Preprocessing Version": PREPROCESSING_VERSION,
            }
        )

        membership = universe[[
            "Symbol",
            "Sector",
            "Industry",
            "Total Revenue",
            "MarketCap",
            "Canonical KPI Available Count",
            "Canonical KPI Completeness",
            "Retained for Clustering",
            "Imputed Canonical KPI Count",
            "Missing Canonical KPIs",
        ]].copy()
        membership.insert(0, "Scenario ID", scenario.scenario_id)
        membership.insert(1, "Scenario Name", scenario.scenario_name)
        membership_frames.append(membership)

    return pd.DataFrame(catalog_rows), pd.concat(membership_frames, ignore_index=True)


def _read_v3_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Preprocessing Version" in frame.columns:
        versions = set(frame["Preprocessing Version"].dropna().unique())
        if versions != {PREPROCESSING_VERSION}:
            raise ValueError(
                f"Evidence file '{path.name}' has unexpected preprocessing versions: "
                f"{sorted(versions)}"
            )
    return frame


def _build_full_assignment_results(evidence_dir: Path) -> pd.DataFrame:
    candidates = _read_v3_csv(evidence_dir / "full_assignment_candidate_assessment.csv")
    traditional = _read_v3_csv(evidence_dir / "best_traditional_by_scenario_method.csv")
    traditional = traditional[traditional["Model"] != "HDBSCAN"].copy()
    autoencoder = _read_v3_csv(evidence_dir / "autoencoder_robustness_summary.csv")
    dec = _read_v3_csv(evidence_dir / "dec_robustness_summary.csv")

    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        scenario_id = candidate["Scenario ID"]
        method = candidate["Method"]
        k = int(candidate["K"])
        record: dict[str, Any] = {
            "Experiment Phase": candidate["Experiment Phase"],
            "Preprocessing Version": candidate["Preprocessing Version"],
            "Scenario ID": scenario_id,
            "Scenario Name": candidate["Scenario Name"],
            "Method": method,
            "Method Type": candidate["Method Type"],
            "Configuration ID": f"{method}|K={k}",
            "K": k,
            "Native Evaluation Space": np.nan,
            "Common Evaluation Space": "Scenario-specific scaled KPI-8",
            "Native Silhouette": np.nan,
            "Common-Space Silhouette": candidate["Common-Space Silhouette"],
            "Common-Space Davies-Bouldin": np.nan,
            "Common-Space Calinski-Harabasz": np.nan,
            "Evaluation Coverage": 1.0,
            "Noise Share": 0.0,
            "Neural Seed Mean ARI": candidate["Neural Seed Mean ARI"],
            "Mean Resampling ARI": candidate["Mean Resampling ARI"],
            "Minimum Resampling ARI": candidate["Minimum Resampling ARI"],
            "Pareto Efficient": bool(candidate["Pareto Efficient"]),
            "Separation-Focused Weighted Rank": candidate[
                "Separation-focused Weighted Rank"
            ],
            "Balanced Weighted Rank": candidate["Balanced Weighted Rank"],
            "Stability-Focused Weighted Rank": candidate[
                "Stability-focused Weighted Rank"
            ],
            "Selection Role": "Full-assignment candidate",
            "Canonical v3 Evidence Available": True,
        }

        if method in {"K-Means", "Agglomerative", "Gaussian Mixture"}:
            match = traditional[
                (traditional["Scenario ID"] == scenario_id)
                & (traditional["Model"] == method)
                & (traditional["K"].astype(int) == k)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one traditional result for {scenario_id}, {method}, K={k}."
                )
            source = match.iloc[0]
            record.update(
                {
                    "Native Evaluation Space": source["Native Evaluation Space"],
                    "Common Evaluation Space": source["Common Evaluation Space"],
                    "Native Silhouette": source["Silhouette Score"],
                    "Common-Space Davies-Bouldin": source[
                        "Common-Space Davies-Bouldin Index"
                    ],
                    "Common-Space Calinski-Harabasz": source[
                        "Common-Space Calinski-Harabasz Score"
                    ],
                }
            )
        elif method == "Autoencoder + K-Means":
            match = autoencoder[autoencoder["Scenario ID"] == scenario_id]
            if len(match) != 1:
                raise ValueError(f"Expected one autoencoder summary for {scenario_id}.")
            source = match.iloc[0]
            record.update(
                {
                    "Native Evaluation Space": source["Native_Evaluation_Space"],
                    "Common Evaluation Space": source["Common_Evaluation_Space"],
                    "Native Silhouette": source["Mean_Silhouette"],
                    "Common-Space Davies-Bouldin": source["Mean_Common_Space_DBI"],
                    "Common-Space Calinski-Harabasz": source["Mean_Common_Space_CH"],
                }
            )
        elif method == "DEC":
            match = dec[dec["Scenario ID"] == scenario_id]
            if len(match) != 1:
                raise ValueError(f"Expected one DEC summary for {scenario_id}.")
            source = match.iloc[0]
            record.update(
                {
                    "Native Evaluation Space": source["Final_Native_Evaluation_Space"],
                    "Common Evaluation Space": source["Common_Evaluation_Space"],
                    "Native Silhouette": source["Mean_Final_Silhouette"],
                    "Common-Space Davies-Bouldin": source[
                        "Mean_Final_Common_Space_DBI"
                    ],
                    "Common-Space Calinski-Harabasz": source[
                        "Mean_Final_Common_Space_CH"
                    ],
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _build_hdbscan_results(evidence_dir: Path) -> pd.DataFrame:
    frontier = _read_v3_csv(evidence_dir / "hdbscan_pareto_frontier.csv")
    stability = _read_v3_csv(evidence_dir / "traditional_resampling_summary.csv")
    stability = stability[stability["Method"] == "HDBSCAN"].copy()

    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_frontier in frontier.groupby("Scenario ID", sort=False):
        scenario_frontier = scenario_frontier.copy()
        scenario_frontier["Separation Rank"] = scenario_frontier[
            "Common-Space Silhouette Score"
        ].rank(ascending=False, method="min")
        scenario_frontier["Coverage Rank"] = scenario_frontier[
            "Evaluation Coverage"
        ].rank(ascending=False, method="min")
        scenario_frontier["Balanced Rank Sum"] = (
            scenario_frontier["Separation Rank"] + scenario_frontier["Coverage Rank"]
        )
        separation_id = scenario_frontier.loc[
            scenario_frontier["Common-Space Silhouette Score"].idxmax(), "Candidate ID"
        ]
        coverage_id = scenario_frontier.loc[
            scenario_frontier["Evaluation Coverage"].idxmax(), "Candidate ID"
        ]
        balanced_id = scenario_frontier.loc[
            scenario_frontier["Balanced Rank Sum"].idxmin(), "Candidate ID"
        ]

        for _, source in scenario_frontier.iterrows():
            candidate_id = source["Candidate ID"]
            roles = []
            if candidate_id == separation_id:
                roles.append("Separation leader")
            if candidate_id == coverage_id:
                roles.append("Coverage leader")
            if candidate_id == balanced_id:
                roles.append("Balanced leader")

            stability_match = stability[
                (stability["Scenario ID"] == scenario_id)
                & (stability["Candidate ID"] == candidate_id)
            ]
            if len(stability_match) > 1:
                raise ValueError(
                    f"Multiple HDBSCAN stability summaries found for {scenario_id}, {candidate_id}."
                )
            stability_row = stability_match.iloc[0] if len(stability_match) == 1 else None
            rows.append(
                {
                    "Experiment Phase": source["Experiment Phase"],
                    "Preprocessing Version": source["Preprocessing Version"],
                    "Scenario ID": scenario_id,
                    "Scenario Name": source["Scenario Name"],
                    "Method": "HDBSCAN",
                    "Method Type": "Density-based partial-assignment",
                    "Configuration ID": candidate_id,
                    "K": int(source["K"]),
                    "Native Evaluation Space": source["Native Evaluation Space"],
                    "Common Evaluation Space": source["Common Evaluation Space"],
                    "Native Silhouette": source["Silhouette Score"],
                    "Common-Space Silhouette": source[
                        "Common-Space Silhouette Score"
                    ],
                    "Common-Space Davies-Bouldin": source[
                        "Common-Space Davies-Bouldin Index"
                    ],
                    "Common-Space Calinski-Harabasz": source[
                        "Common-Space Calinski-Harabasz Score"
                    ],
                    "Evaluation Coverage": source["Evaluation Coverage"],
                    "Noise Share": source["Noise Share"],
                    "Neural Seed Mean ARI": np.nan,
                    "Mean Resampling ARI": (
                        stability_row["Mean_ARI"] if stability_row is not None else np.nan
                    ),
                    "Minimum Resampling ARI": (
                        stability_row["Min_ARI"] if stability_row is not None else np.nan
                    ),
                    "Pareto Efficient": True,
                    "Separation-Focused Weighted Rank": np.nan,
                    "Balanced Weighted Rank": np.nan,
                    "Stability-Focused Weighted Rank": np.nan,
                    "Selection Role": " | ".join(roles) or "Pareto frontier",
                    "Canonical v3 Evidence Available": True,
                }
            )
    return pd.DataFrame(rows)


def build_method_results(evidence_dir: Path) -> pd.DataFrame:
    """Normalise final candidate evidence without importing stale v2 files."""

    full_assignment = _build_full_assignment_results(evidence_dir)
    hdbscan = _build_hdbscan_results(evidence_dir)
    results = pd.concat([full_assignment, hdbscan], ignore_index=True)

    scenario_names = {scenario.scenario_id: scenario.scenario_name for scenario in SCENARIOS}
    idec_placeholders = pd.DataFrame(
        [
            {
                "Experiment Phase": "Exploratory only",
                "Preprocessing Version": PREPROCESSING_VERSION,
                "Scenario ID": scenario_id,
                "Scenario Name": scenario_name,
                "Method": "IDEC",
                "Method Type": "Deep embedded clustering with reconstruction",
                "Configuration ID": "Not evaluated in canonical v3",
                "K": np.nan,
                "Native Evaluation Space": np.nan,
                "Common Evaluation Space": "Scenario-specific scaled KPI-8",
                "Native Silhouette": np.nan,
                "Common-Space Silhouette": np.nan,
                "Common-Space Davies-Bouldin": np.nan,
                "Common-Space Calinski-Harabasz": np.nan,
                "Evaluation Coverage": np.nan,
                "Noise Share": np.nan,
                "Neural Seed Mean ARI": np.nan,
                "Mean Resampling ARI": np.nan,
                "Minimum Resampling ARI": np.nan,
                "Pareto Efficient": False,
                "Separation-Focused Weighted Rank": np.nan,
                "Balanced Weighted Rank": np.nan,
                "Stability-Focused Weighted Rank": np.nan,
                "Selection Role": "No canonical v3 evidence; method catalogue only",
                "Canonical v3 Evidence Available": False,
            }
            for scenario_id, scenario_name in scenario_names.items()
        ]
    )
    results = pd.concat([results, idec_placeholders], ignore_index=True)
    return results.sort_values(
        ["Scenario ID", "Method", "Canonical v3 Evidence Available", "Selection Role"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def validate_app_tables(tables: dict[str, pd.DataFrame]) -> None:
    company_kpis = tables["company_kpis"]
    scenario_catalog = tables["scenario_catalog"]
    scenario_membership = tables["scenario_membership"]
    method_results = tables["method_results"]

    if len(company_kpis) != 4_576:
        raise AssertionError(f"Expected 4,576 companies, found {len(company_kpis)}.")
    if company_kpis["Symbol"].duplicated().any():
        raise AssertionError("Application company table contains duplicate symbols.")
    if set(scenario_catalog["Scenario ID"]) != {s.scenario_id for s in SCENARIOS}:
        raise AssertionError("Scenario catalogue does not contain exactly the canonical scenarios.")
    expected_counts = {s.scenario_id: s.expected_companies for s in SCENARIOS}
    actual_counts = dict(
        scenario_membership[scenario_membership["Retained for Clustering"]]
        .groupby("Scenario ID")["Symbol"]
        .nunique()
    )
    if actual_counts != expected_counts:
        raise AssertionError(
            f"Scenario membership counts differ from canonical v3: {actual_counts}."
        )
    methods = set(method_results["Method"])
    expected_methods = {entry["method"] for entry in METHOD_CATALOG}
    if methods != expected_methods:
        raise AssertionError(f"Method results contain {methods}; expected {expected_methods}.")
    idec = method_results[method_results["Method"] == "IDEC"]
    if idec["Canonical v3 Evidence Available"].any():
        raise AssertionError("IDEC must not be labelled as canonical v3 evidence.")
    evidence_versions = set(
        method_results.loc[
            method_results["Canonical v3 Evidence Available"], "Preprocessing Version"
        ].unique()
    )
    if evidence_versions != {PREPROCESSING_VERSION}:
        raise AssertionError(f"Unexpected method evidence versions: {evidence_versions}.")


def build_app_dataset(project_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Build and atomically publish all Phase 1 application data artifacts."""

    project_root = project_root.resolve()
    if output_dir is None:
        output_dir = project_root / "data" / "processed" / "app"
    output_dir = output_dir.resolve()

    source_workbook = project_root / "data" / "interim" / "clean_financials_with_metadata.xlsx"
    kpi_definition_path = project_root / "config" / "kpi_definitions.csv"
    evidence_dir = (
        project_root
        / "outputs"
        / "experiment_results"
        / "canonical_cross_cohort_robustness"
    )
    source_paths = [
        source_workbook,
        kpi_definition_path,
        evidence_dir / "canonical_cross_cohort_method_assessment_compact.csv",
        evidence_dir / "cluster_tendency_summary.csv",
        evidence_dir / "full_assignment_candidate_assessment.csv",
        evidence_dir / "best_traditional_by_scenario_method.csv",
        evidence_dir / "autoencoder_robustness_summary.csv",
        evidence_dir / "dec_robustness_summary.csv",
        evidence_dir / "hdbscan_pareto_frontier.csv",
        evidence_dir / "traditional_resampling_summary.csv",
    ]
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Application data sources are missing: {missing_sources}")

    financials = pd.read_excel(source_workbook)
    definitions = load_kpi_definitions(kpi_definition_path)
    company_kpis, validity_audit = build_company_kpis(financials, definitions)
    scenario_catalog, scenario_membership = build_scenario_tables(company_kpis)
    method_results = build_method_results(evidence_dir)
    robustness_summary = _read_v3_csv(
        evidence_dir / "canonical_cross_cohort_method_assessment_compact.csv"
    )
    cluster_tendency = _read_v3_csv(evidence_dir / "cluster_tendency_summary.csv")

    kpi_catalog = definitions.copy()
    kpi_catalog["Canonical Feature"] = kpi_catalog["KPI"].isin(CANONICAL_KPIS)
    kpi_catalog["Display Order"] = np.arange(1, len(kpi_catalog) + 1)

    tables = {
        "company_kpis": company_kpis,
        "scenario_catalog": scenario_catalog,
        "scenario_membership": scenario_membership,
        "method_results": method_results,
        "cluster_tendency": cluster_tendency,
        "robustness_summary": robustness_summary,
        "kpi_catalog": kpi_catalog,
        "economic_validity_audit": validity_audit,
    }
    validate_app_tables(tables)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="app_data_", dir=output_dir.parent))
    try:
        _write_csv(company_kpis, staging_dir / "company_kpis.csv.gz", compressed=True)
        _write_csv(
            scenario_membership,
            staging_dir / "scenario_membership.csv.gz",
            compressed=True,
        )
        _write_csv(
            validity_audit,
            staging_dir / "economic_validity_audit.csv.gz",
            compressed=True,
        )
        _write_csv(scenario_catalog, staging_dir / "scenario_catalog.csv")
        _write_csv(method_results, staging_dir / "method_results.csv")
        _write_csv(cluster_tendency, staging_dir / "cluster_tendency.csv")
        _write_csv(robustness_summary, staging_dir / "robustness_summary.csv")
        _write_csv(kpi_catalog, staging_dir / "kpi_catalog.csv")
        _write_json(staging_dir / "method_catalog.json", list(METHOD_CATALOG))

        output_metadata = {
            "company_kpis.csv.gz": {"rows": len(company_kpis), "columns": len(company_kpis.columns)},
            "scenario_membership.csv.gz": {
                "rows": len(scenario_membership),
                "columns": len(scenario_membership.columns),
            },
            "economic_validity_audit.csv.gz": {
                "rows": len(validity_audit),
                "columns": len(validity_audit.columns),
            },
            "scenario_catalog.csv": {"rows": len(scenario_catalog), "columns": len(scenario_catalog.columns)},
            "method_results.csv": {"rows": len(method_results), "columns": len(method_results.columns)},
            "cluster_tendency.csv": {"rows": len(cluster_tendency), "columns": len(cluster_tendency.columns)},
            "robustness_summary.csv": {"rows": len(robustness_summary), "columns": len(robustness_summary.columns)},
            "kpi_catalog.csv": {"rows": len(kpi_catalog), "columns": len(kpi_catalog.columns)},
            "method_catalog.json": {"records": len(METHOD_CATALOG)},
        }
        manifest = {
            "schema_version": APP_DATA_SCHEMA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": [
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "modified_at_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
                for path in source_paths
            ],
            "rules": {
                "canonical_kpis": CANONICAL_KPIS,
                "minimum_company_kpi_completeness": MIN_COMPANY_KPI_COMPLETENESS,
                "minimum_positive_net_income_margin": MINIMUM_POSITIVE_NET_INCOME_MARGIN,
                "winsorisation": {"lower_quantile": 0.05, "upper_quantile": 0.95},
                "business_caps": BUSINESS_CAPS,
                "scaling": "RobustScaler fitted independently per scenario",
                "imputation": "Median fitted independently per scenario",
            },
            "scenario_counts": {
                row["Scenario ID"]: int(row["Companies Retained"])
                for row in scenario_catalog.to_dict("records")
            },
            "method_evidence_note": (
                "IDEC is included in the method catalogue but has no final five-cohort "
                "scenario_pipeline_v3_economic_validity evidence."
            ),
            "outputs": output_metadata,
        }
        _write_json(staging_dir / "dataset_manifest.json", manifest)

        output_dir.mkdir(parents=True, exist_ok=True)
        generated_names = [*output_metadata.keys(), "dataset_manifest.json"]
        for name in generated_names:
            (staging_dir / name).replace(output_dir / name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return manifest
