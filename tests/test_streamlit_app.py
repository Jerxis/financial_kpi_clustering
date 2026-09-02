"""Smoke tests for the three Phase 3 Streamlit workspaces."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppSmokeTest(unittest.TestCase):
    def _app(self):
        return AppTest.from_file(
            str(PROJECT_ROOT / "app" / "streamlit_app.py"),
            default_timeout=45,
        ).run()

    def test_cluster_explorer_renders_without_exceptions(self) -> None:
        app = self._app()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 5)
        self.assertEqual(app.metric[0].value, "587")
        self.assertGreaterEqual(len(app.dataframe), 2)

    def test_method_laboratory_renders_without_exceptions(self) -> None:
        app = self._app()
        app.radio[0].set_value("Method Laboratory").run()
        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_robustness_workspace_renders_without_exceptions(self) -> None:
        app = self._app()
        app.radio[0].set_value("Robustness & limitations").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 5)

    def test_idec_boundary_renders_without_fabricated_metrics(self) -> None:
        app = self._app()
        method_selectbox = next(
            widget for widget in app.selectbox if widget.label == "Clustering method"
        )
        method_selectbox.set_value("IDEC").run()
        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.error), 1)

    def test_hdbscan_explorer_is_interactive(self) -> None:
        app = self._app()
        method_selectbox = next(
            widget for widget in app.selectbox if widget.label == "Clustering method"
        )
        method_selectbox.set_value("HDBSCAN").run()
        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.dataframe), 2)
        self.assertGreaterEqual(len(app.info), 1)


if __name__ == "__main__":
    unittest.main()
