from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import (
    DEFAULT_TARGET_COLUMN,
    build_best_cfc_config,
    file_contains_target,
    list_db2_files,
    train_cfc_regressor,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "log" / "db2_single_subject"


def parse_args() -> argparse.Namespace:
    """Expose the few knobs that matter for a single-subject feasibility run."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the fixed-feature CfC regressor on one DB2 subject only, "
            "using blocked time splits inside that subject's recordings."
        )
    )
    parser.add_argument(
        "--db2-dir",
        type=Path,
        default=REPO_ROOT / "src" / "data" / "DB2",
        help="Directory that stores the DB2 .mat recordings.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="S1",
        help="Subject id used for training, validation, and test splits.",
    )
    parser.add_argument(
        "--target-column",
        type=int,
        default=DEFAULT_TARGET_COLUMN,
        help="Zero-based glove column used as the continuous regression target.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs passed to the core trainer.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of non-improving validation epochs tolerated before early stopping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device string passed through to the core trainer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the final prediction plot and summary JSON.",
    )
    return parser.parse_args()


def recording_sort_key(file_name: str) -> tuple[int, str]:
    """
    Sort one subject's recordings by exercise number, then by the full file name.

    DB2 names follow `S1_E1_A1.mat`, `S1_E2_A1.mat`, etc. Sorting this way keeps
    the experiment report stable and easy to read.
    """

    parts = file_name.split("_")
    if len(parts) < 2:
        return 0, file_name

    exercise_tag = parts[1]
    digits = "".join(character for character in exercise_tag if character.isdigit())
    exercise_number = int(digits) if digits else 0
    return exercise_number, file_name


def discover_subject_recordings(
    db2_dir: Path,
    *,
    target_source: str,
    subject_id: str,
) -> tuple[str, ...]:
    """
    Return all usable recordings for one subject.

    Only files that contain both `emg` and the requested target source are kept.
    This intentionally drops DB2 recordings such as `E3` when they do not expose
    glove or inclin targets, so the caller cannot accidentally build an invalid
    regression split.
    """

    subject_prefix = f"{subject_id}_"
    recordings = [
        file_path.name
        for file_path in list_db2_files(db2_dir)
        if file_path.name.startswith(subject_prefix) and file_contains_target(file_path, target_source)
    ]
    if not recordings:
        raise FileNotFoundError(
            f"no recordings for subject '{subject_id}' under {db2_dir} contain target source '{target_source}'"
        )
    return tuple(sorted(recordings, key=recording_sort_key))


def summarize_best_epoch(history: list[dict[str, float]]) -> dict[str, float]:
    """Extract the checkpoint selected by validation MAE."""
    if not history:
        raise ValueError("history is empty")
    return min(history, key=lambda entry: entry["val_mae"])


def make_jsonable(value: Any) -> Any:
    """Convert numpy-like or path-like values into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def save_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Persist a compact experiment summary for later review."""
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(summary), handle, indent=2)
    return summary_path


def run_single_subject_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """
    Execute one single-subject feasibility experiment.

    Protocol:
    1. Discover every usable recording for the requested subject.
    2. Build blocked train/val/test time ranges inside those recordings.
    3. Train the fixed CfC regression path and save the final test visualization.
    """

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_best_cfc_config(
        db2_dir=args.db2_dir,
        device=args.device,
    )
    subject_files = discover_subject_recordings(
        base_config.db2_dir,
        target_source=base_config.target_source,
        subject_id=args.subject,
    )

    config = build_best_cfc_config(
        db2_dir=base_config.db2_dir,
        split_strategy="blocked_time",
        source_files=subject_files,
        target_columns=(args.target_column,),
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
    )

    print("Single-subject protocol")
    print(f"  subject      : {args.subject}")
    print(f"  target source: {config.target_source}")
    print(f"  target column: {args.target_column}")
    print(f"  source files : {list(subject_files)}")

    results = train_cfc_regressor(config)

    figure_path = output_dir / f"{args.subject}_target{args.target_column}_prediction.png"
    results["prediction_figure"].savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(results["prediction_figure"])

    summary = {
        "protocol": "single_subject_blocked_time",
        "subject": args.subject,
        "source_files": list(subject_files),
        "config": make_jsonable(asdict(config)),
        "best_epoch": make_jsonable(summarize_best_epoch(results["history"])),
        "history": make_jsonable(results["history"]),
        "metrics": {
            split_name: make_jsonable(evaluation["metrics"])
            for split_name, evaluation in results["evaluations"].items()
        },
        "artifacts": {
            "prediction_plot": str(figure_path),
        },
    }
    summary_path = save_summary(output_dir, summary)

    print(f"\nSaved prediction plot to: {figure_path}")
    print(f"Saved summary to: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_single_subject_experiment(args)


if __name__ == "__main__":
    main()
