"""Contract tests for the complete Phase 3 canonical solution layer."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.demo_artifacts import prepare_canonical_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = PROJECT_ROOT / "data" / "processed" / "app" / "solutions"


class CanonicalSolutionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assignments = pd.read_csv(
            SOLUTION_DIR / "traditional_company_assignments.csv.gz"
        )
        cls.profiles = pd.read_csv(
            SOLUTION_DIR / "traditional_cluster_profiles.csv"
        )
        cls.clusters = pd.read_csv(
            SOLUTION_DIR / "traditional_cluster_summary.csv"
        )
        cls.solutions = pd.read_csv(
            SOLUTION_DIR / "traditional_solution_summary.csv"
        )
        cls.company_kpis = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "app" / "company_kpis.csv.gz"
        )
        cls.frontier = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "experiment_results"
            / "canonical_cross_cohort_robustness"
            / "hdbscan_pareto_frontier.csv"
        )

    def test_catalogue_covers_five_cohorts_and_four_traditional_methods(self) -> None:
        self.assertEqual(len(self.solutions), 20)
        self.assertEqual(self.solutions["Scenario ID"].nunique(), 5)
        self.assertEqual(self.solutions["Method"].nunique(), 4)
        self.assertEqual(
            int(self.solutions["Interactive Assignment Available"].sum()), 20
        )

    def test_company_assignments_cover_all_interactive_solutions(self) -> None:
        self.assertEqual(len(self.assignments), 14_868)
        self.assertEqual(
            set(self.assignments["Method"]),
            {"K-Means", "Agglomerative", "Gaussian Mixture", "HDBSCAN"},
        )
        counts = self.assignments.groupby(["Scenario ID", "Method"]).size()
        self.assertEqual(len(counts), 20)
        self.assertTrue(counts.gt(0).all())

    def test_rebuilt_label_metrics_match_frozen_v3_evidence(self) -> None:
        interactive = self.solutions[
            self.solutions["Interactive Assignment Available"]
        ]
        self.assertTrue(
            interactive["Maximum Relative Evidence Delta"].lt(1e-10).all()
        )

    def test_every_interactive_cluster_has_eight_kpi_profiles(self) -> None:
        counts = self.profiles.groupby(
            ["Scenario ID", "Method", "Cluster"]
        )["KPI"].nunique()
        self.assertTrue(counts.eq(8).all())
        self.assertEqual(
            self.clusters[self.clusters["Cluster"].ge(0)].shape[0], len(counts)
        )

    def test_common_pca_coordinates_are_shared_between_methods(self) -> None:
        for scenario_id in self.assignments["Scenario ID"].unique():
            cohort = self.assignments[
                self.assignments["Scenario ID"].eq(scenario_id)
            ]
            baseline = (
                cohort[cohort["Method"].eq("K-Means")]
                .sort_values("Symbol")[["Symbol", "PCA 1", "PCA 2"]]
                .reset_index(drop=True)
            )
            for method in ["Agglomerative", "Gaussian Mixture", "HDBSCAN"]:
                comparison = (
                    cohort[cohort["Method"].eq(method)]
                    .sort_values("Symbol")[["Symbol", "PCA 1", "PCA 2"]]
                    .reset_index(drop=True)
                )
                self.assertTrue(baseline["Symbol"].equals(comparison["Symbol"]))
                self.assertTrue(
                    np.allclose(
                        baseline[["PCA 1", "PCA 2"]],
                        comparison[["PCA 1", "PCA 2"]],
                    )
                )

    def test_all_hdbscan_frontier_metrics_reproduce_in_source_order(self) -> None:
        metric_columns = [
            "Common-Space Silhouette Score",
            "Common-Space Davies-Bouldin Index",
            "Common-Space Calinski-Harabasz Score",
            "Evaluation Coverage",
        ]
        for scenario_id, candidates in self.frontier.groupby("Scenario ID"):
            prepared = prepare_canonical_features(self.company_kpis, scenario_id)
            X = prepared["X_scaled"]
            for _, candidate in candidates.iterrows():
                labels = HDBSCAN(
                    min_cluster_size=int(candidate["Min Cluster Size"]),
                    min_samples=int(candidate["Min Samples"]),
                    copy=True,
                ).fit_predict(X)
                assigned = labels != -1
                actual = np.array(
                    [
                        silhouette_score(X[assigned], labels[assigned]),
                        davies_bouldin_score(X[assigned], labels[assigned]),
                        calinski_harabasz_score(X[assigned], labels[assigned]),
                        assigned.mean(),
                    ]
                )
                expected = candidate[metric_columns].astype(float).to_numpy()
                self.assertTrue(
                    np.allclose(actual, expected, rtol=1e-10, atol=1e-12),
                    msg=f"HDBSCAN evidence mismatch for {scenario_id} / {candidate['Candidate ID']}",
                )


if __name__ == "__main__":
    unittest.main()
