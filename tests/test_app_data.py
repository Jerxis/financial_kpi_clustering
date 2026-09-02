"""Contract tests for the generated Phase 1 application data layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "app"


class AppDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (APP_DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        cls.companies = pd.read_csv(APP_DATA_DIR / "company_kpis.csv.gz")
        cls.scenarios = pd.read_csv(APP_DATA_DIR / "scenario_catalog.csv")
        cls.membership = pd.read_csv(APP_DATA_DIR / "scenario_membership.csv.gz")
        cls.methods = pd.read_csv(APP_DATA_DIR / "method_results.csv")

    def test_schema_and_pipeline_versions_are_explicit(self) -> None:
        self.assertEqual(self.manifest["schema_version"], "1.1.0")
        self.assertEqual(
            self.manifest["preprocessing_version"],
            "scenario_pipeline_v3_economic_validity",
        )

    def test_company_table_has_unique_symbols_and_expected_size(self) -> None:
        self.assertEqual(len(self.companies), 4_576)
        self.assertFalse(self.companies["Symbol"].isna().any())
        self.assertFalse(self.companies["Symbol"].duplicated().any())
        self.assertIn("Source Row Order", self.companies.columns)
        self.assertFalse(self.companies["Source Row Order"].duplicated().any())
        self.assertTrue(self.companies["Source Row Order"].ge(0).all())

    def test_scenario_counts_match_final_v3_evidence(self) -> None:
        expected = {
            "technology_revenue_25m": 587,
            "healthcare_revenue_50m": 444,
            "industrials_revenue_20m": 565,
            "consumer_cyclical_revenue_20m": 470,
            "non_financial_large_cap_revenue_1b": 1_651,
        }
        actual = dict(
            self.membership[self.membership["Retained for Clustering"]]
            .groupby("Scenario ID")["Symbol"]
            .nunique()
        )
        self.assertEqual(actual, expected)

    def test_all_seven_methods_are_declared_without_overstating_idec(self) -> None:
        self.assertEqual(self.methods["Method"].nunique(), 7)
        idec = self.methods[self.methods["Method"] == "IDEC"]
        self.assertEqual(len(idec), 5)
        self.assertFalse(idec["Canonical v3 Evidence Available"].any())
        canonical = self.methods[self.methods["Canonical v3 Evidence Available"]]
        self.assertEqual(
            set(canonical["Preprocessing Version"]),
            {"scenario_pipeline_v3_economic_validity"},
        )

    def test_manifest_row_counts_match_files(self) -> None:
        self.assertEqual(
            self.manifest["outputs"]["company_kpis.csv.gz"]["rows"],
            len(self.companies),
        )
        self.assertEqual(
            self.manifest["outputs"]["method_results.csv"]["rows"],
            len(self.methods),
        )

    def test_manifest_source_hashes_match_current_sources(self) -> None:
        for source in self.manifest["source_files"]:
            digest = hashlib.sha256()
            with (PROJECT_ROOT / source["path"]).open("rb") as source_file:
                for block in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(block)
            self.assertEqual(digest.hexdigest(), source["sha256"])

    def test_imputation_counts_reconcile_to_final_v3_evidence(self) -> None:
        evidence = pd.read_csv(
            PROJECT_ROOT
            / "outputs"
            / "experiment_results"
            / "canonical_cross_cohort_robustness"
            / "scenario_imputation_summary.csv"
        )
        rebuilt = self.scenarios.set_index("Scenario ID")
        expected = evidence.set_index("Scenario ID")
        for scenario_id in rebuilt.index:
            self.assertEqual(
                int(rebuilt.loc[scenario_id, "Companies Retained"]),
                int(expected.loc[scenario_id, "N Companies Retained"]),
            )
            self.assertEqual(
                int(rebuilt.loc[scenario_id, "Companies with Any Imputation"]),
                int(expected.loc[scenario_id, "Companies with Any Imputation"]),
            )


if __name__ == "__main__":
    unittest.main()
