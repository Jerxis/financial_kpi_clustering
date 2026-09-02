"""Contract tests for the canonical Technology/K-Means vertical slice."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "app"
    / "solutions"
    / "technology_revenue_25m"
    / "kmeans"
)


class DemoArtifactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assignments = pd.read_csv(SOLUTION_DIR / "company_assignments.csv.gz")
        cls.profiles = pd.read_csv(SOLUTION_DIR / "cluster_profiles.csv")
        cls.summary = pd.read_csv(SOLUTION_DIR / "cluster_summary.csv")
        cls.audit = json.loads(
            (SOLUTION_DIR / "preprocessing_audit.json").read_text(encoding="utf-8")
        )

    def test_assignments_cover_the_canonical_cohort(self) -> None:
        self.assertEqual(len(self.assignments), 587)
        self.assertFalse(self.assignments["Symbol"].duplicated().any())
        self.assertEqual(set(self.assignments["Cluster"]), {0, 1})
        self.assertEqual(self.summary["Company Count"].sum(), 587)

    def test_projection_and_assignment_fields_are_finite(self) -> None:
        fields = [
            "PCA 1",
            "PCA 2",
            "Distance to Assigned Centroid",
            "Nearest-Rival Squared-Distance Margin",
        ]
        self.assertTrue(np.isfinite(self.assignments[fields].to_numpy()).all())
        self.assertTrue(
            self.assignments["Nearest-Rival Squared-Distance Margin"].ge(0).all()
        )

    def test_profiles_have_two_clusters_and_eight_kpis(self) -> None:
        self.assertEqual(len(self.profiles), 16)
        self.assertEqual(self.profiles["Cluster"].nunique(), 2)
        self.assertEqual(self.profiles["KPI"].nunique(), 8)
        self.assertTrue(
            self.profiles.groupby("Cluster")["Company Count"].nunique().eq(1).all()
        )

    def test_metrics_match_the_saved_v3_evidence(self) -> None:
        rebuilt = self.audit["rebuilt_metrics"]
        evidence = self.audit["evidence_metrics"]
        for metric in evidence:
            self.assertAlmostEqual(rebuilt[metric], evidence[metric], places=10)

    def test_pca_is_explicitly_visual_only(self) -> None:
        self.assertIn("visualisation only", self.audit["pca"]["purpose"])
        self.assertGreater(self.audit["pca"]["explained_variance_share"], 0)
        self.assertLessEqual(self.audit["pca"]["explained_variance_share"], 1)


if __name__ == "__main__":
    unittest.main()

