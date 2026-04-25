import pathlib
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from run_db2_single_subject import discover_subject_recordings, recording_sort_key


class SingleSubjectProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
