import pathlib
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import run_db2_single_subject
from run_db2_single_subject import (
    discover_subject_recordings,
    parse_args,
    recording_sort_key,
    run_single_subject_experiment,
)
from run_semantic_dof_screen import apply_validation_first_decisions


ORIGINAL_BUILD_BEST_CFC_CONFIG = run_db2_single_subject.build_best_cfc_config


def parse_cli_args(*arguments: str) -> Namespace:
    with patch("sys.argv", ["run_db2_single_subject.py", *arguments]):
        return parse_args()


class FakeFigure:
    saved_path = None

    def savefig(self, path, **_kwargs) -> None:
        self.saved_path = pathlib.Path(path)


class SingleSubjectProtocolTests(unittest.TestCase):
    def test_parse_args_accepts_target_source_without_changing_target_column_default(self) -> None:
        args = parse_cli_args("--target-source", "inclin")

        self.assertEqual(args.target_source, "inclin")
        self.assertEqual(args.target_column, 10)

    def test_parse_args_preserves_explicit_target_column(self) -> None:
        args = parse_cli_args("--target-column", "2")

        self.assertEqual(args.target_source, "glove")
        self.assertEqual(args.target_column, 2)

    def test_recording_sort_key_orders_exercises_numerically(self) -> None:
        file_names = [
            "S1_E10_A1.mat",
            "S1_E2_A1.mat",
            "S1_E1_A1.mat",
        ]

        sorted_names = sorted(file_names, key=recording_sort_key)

        self.assertEqual(
            sorted_names,
            ["S1_E1_A1.mat", "S1_E2_A1.mat", "S1_E10_A1.mat"],
        )

    @patch("run_db2_single_subject.file_contains_target")
    @patch("run_db2_single_subject.list_db2_files")
    def test_discover_subject_recordings_filters_subject_and_missing_targets(
        self,
        mock_list_db2_files,
        mock_file_contains_target,
    ) -> None:
        mock_list_db2_files.return_value = [
            pathlib.Path("S1_E2_A1.mat"),
            pathlib.Path("S2_E1_A1.mat"),
            pathlib.Path("S1_E1_A1.mat"),
            pathlib.Path("S1_E3_A1.mat"),
        ]
        mock_file_contains_target.side_effect = lambda file_path, _: file_path.name != "S1_E3_A1.mat"

        subject_files = discover_subject_recordings(
            pathlib.Path("unused"),
            target_source="glove",
            subject_id="S1",
        )

        self.assertEqual(subject_files, ("S1_E1_A1.mat", "S1_E2_A1.mat"))

    @patch("run_db2_single_subject.plt.close")
    @patch("run_db2_single_subject.save_summary")
    @patch("run_db2_single_subject.train_cfc_regressor")
    @patch("run_db2_single_subject.discover_subject_recordings")
    @patch("run_db2_single_subject.build_best_cfc_config")
    def test_run_single_subject_uses_requested_target_source_and_preserves_target_column(
        self,
        mock_build_best_cfc_config,
        mock_discover_subject_recordings,
        mock_train_cfc_regressor,
        mock_save_summary,
        _mock_close,
    ) -> None:
        mock_build_best_cfc_config.side_effect = lambda **overrides: ORIGINAL_BUILD_BEST_CFC_CONFIG(
            **overrides
        )
        mock_discover_subject_recordings.return_value = ("S1_E1_A1.mat", "S1_E2_A1.mat")
        mock_train_cfc_regressor.return_value = {
            "prediction_figure": FakeFigure(),
            "history": [{"epoch": 1, "val_mae": 0.25}],
            "evaluations": {"test": {"metrics": {"mae": 0.5}}},
        }
        mock_save_summary.return_value = pathlib.Path("summary.json")

        args = Namespace(
            db2_dir=pathlib.Path("db2"),
            subject="S1",
            target_source="inclin",
            target_column=2,
            target_offset_samples=17,
            zc_threshold=None,
            ssc_threshold=None,
            feature_normalization=None,
            target_normalization=None,
            max_epochs=3,
            early_stopping_patience=1,
            max_windows_per_file=None,
            device="cpu",
            output_dir=REPO_ROOT / "tests",
        )

        summary = run_single_subject_experiment(args)

        mock_discover_subject_recordings.assert_called_once_with(
            pathlib.Path("db2"),
            target_source="inclin",
            subject_id="S1",
        )
        trained_config = mock_train_cfc_regressor.call_args.args[0]
        self.assertEqual(trained_config.target_source, "inclin")
        self.assertEqual(trained_config.target_columns, (2,))
        self.assertEqual(trained_config.target_offset_samples, 17)
        self.assertEqual(summary.get("target_source"), "inclin")
        self.assertEqual(summary.get("target_column"), 2)
        self.assertEqual(summary.get("target_offset_samples"), 17)
        self.assertEqual(summary["config"]["target_source"], "inclin")
        self.assertEqual(summary["config"]["target_columns"], [2])
        self.assertEqual(summary["config"]["target_offset_samples"], 17)

    def test_validation_first_selection_uses_fallback_after_primary_failure(self) -> None:
        rows = [
            {
                "semantic_dof": "wrist_flexion_extension",
                "role": "primary",
                "val_r2": 0.2,
                "test_r2": -0.1,
                "stable": True,
                "selected_by_validation": False,
                "decision": "evaluated",
                "failure_reason": "",
            },
            {
                "semantic_dof": "wrist_flexion_extension",
                "role": "fallback",
                "val_r2": 0.3,
                "test_r2": 0.55,
                "stable": True,
                "selected_by_validation": False,
                "decision": "evaluated",
                "failure_reason": "",
            },
            {
                "semantic_dof": "wrist_flexion_extension",
                "role": "fallback",
                "val_r2": 0.1,
                "test_r2": 0.9,
                "stable": True,
                "selected_by_validation": False,
                "decision": "evaluated",
                "failure_reason": "",
            },
        ]

        apply_validation_first_decisions(rows)

        self.assertFalse(rows[0]["selected_by_validation"])
        self.assertEqual(rows[0]["decision"], "fail")
        self.assertTrue(rows[1]["selected_by_validation"])
        self.assertEqual(rows[1]["decision"], "fallback_pass")
        self.assertFalse(rows[2]["selected_by_validation"])
        self.assertEqual(rows[2]["decision"], "rejected")


if __name__ == "__main__":
    unittest.main()
