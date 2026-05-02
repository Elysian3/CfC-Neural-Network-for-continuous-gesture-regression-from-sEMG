from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from train import (
    DEFAULT_TARGET_COLUMN,
    build_best_cfc_config,
    build_sequence_split,
    evaluate_split,
    file_contains_target,
    list_db2_files,
    load_recording_features,
    make_train_loader,
    normalize_sequence_inputs,
    resolve_device,
    train_cfc_regressor,
    train_one_epoch,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "log" / "db2_subject_adaptation"


def parse_args() -> argparse.Namespace:
    """Expose only the protocol choices that still matter after fixing the hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Pretrain a fixed-hyperparameter CfC regressor, then adapt it to one new DB2 subject for 1 epoch."
    )
    parser.add_argument(
        "--db2-dir",
        type=Path,
        default=REPO_ROOT / "src" / "data" / "DB2",
        help="Directory that stores the DB2 .mat recordings.",
    )
    parser.add_argument(
        "--adapt-subject",
        type=str,
        default=None,
        help="Held-out subject used for one-epoch calibration and final testing. Defaults to the highest-numbered subject.",
    )
    parser.add_argument(
        "--val-subject",
        type=str,
        default=None,
        help="Validation subject used for early stopping during pretraining. Defaults to the highest-numbered subject before the adaptation subject.",
    )
    parser.add_argument(
        "--support-exercise",
        type=str,
        default="E1",
        help="Exercise from the adaptation subject used for one-epoch calibration.",
    )
    parser.add_argument(
        "--query-exercise",
        type=str,
        default="E2",
        help="Exercise from the adaptation subject used for final testing.",
    )
    parser.add_argument(
        "--target-column",
        type=int,
        default=DEFAULT_TARGET_COLUMN,
        help="Zero-based glove column used as the continuous regression target.",
    )
    parser.add_argument(
        "--target-offset-samples",
        type=int,
        default=None,
        help="Target alignment shift in samples. Defaults to the core training config value.",
    )
    parser.add_argument(
        "--zc-threshold",
        type=float,
        default=None,
        help="Explicit zero-crossing threshold. Defaults to the core training config value.",
    )
    parser.add_argument(
        "--ssc-threshold",
        type=float,
        default=None,
        help="Explicit slope-sign-change threshold. Defaults to the core training config value.",
    )
    parser.add_argument(
        "--feature-normalization",
        choices=("zscore", "mu_law"),
        default=None,
        help="Feature normalization method. Defaults to the core training config value.",
    )
    parser.add_argument(
        "--target-normalization",
        choices=("zscore", "mu_law"),
        default=None,
        help="Target normalization method. Defaults to the core training config value.",
    )
    parser.add_argument(
        "--adapt-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the one-epoch subject adaptation step.",
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


def subject_sort_key(subject_id: str) -> tuple[int, str]:
    """Sort subject ids numerically instead of lexicographically."""
    digits = "".join(character for character in subject_id if character.isdigit())
    if not digits:
        raise ValueError(f"subject id '{subject_id}' does not contain a numeric suffix")
    return int(digits), subject_id


def discover_target_files(db2_dir: Path, target_source: str) -> dict[str, list[str]]:
    """Group usable DB2 recordings by subject id."""
    grouped: dict[str, list[str]] = {}
    for file_path in list_db2_files(db2_dir):
        if not file_contains_target(file_path, target_source):
            continue
        subject_id = file_path.stem.split("_")[0]
        grouped.setdefault(subject_id, []).append(file_path.name)

    sorted_subjects = sorted(grouped, key=subject_sort_key)
    return {
        subject_id: sorted(grouped[subject_id])
        for subject_id in sorted_subjects
    }


def select_adaptation_protocol(
    grouped_files: dict[str, list[str]],
    *,
    adapt_subject: str | None,
    val_subject: str | None,
    support_exercise: str,
    query_exercise: str,
) -> dict[str, Any]:
    """
    Split the available subjects into pretrain, validation, support, and query groups.

    Protocol:
    - pretrain on every subject except the held-out validation and adaptation subjects
    - use the validation subject for early stopping
    - use one exercise from the adaptation subject for 1-epoch calibration
    - test on a different exercise from that same subject
    """

    subject_ids = sorted(grouped_files, key=subject_sort_key)
    if len(subject_ids) < 3:
        raise ValueError("need at least three subjects for pretrain/validation/adaptation")

    selected_adapt_subject = adapt_subject or subject_ids[-1]
    if selected_adapt_subject not in grouped_files:
        raise KeyError(f"adaptation subject '{selected_adapt_subject}' not found")

    remaining_subjects = [subject_id for subject_id in subject_ids if subject_id != selected_adapt_subject]
    selected_val_subject = val_subject or remaining_subjects[-1]
    if selected_val_subject == selected_adapt_subject:
        raise ValueError("validation subject must differ from adaptation subject")
    if selected_val_subject not in grouped_files:
        raise KeyError(f"validation subject '{selected_val_subject}' not found")

    train_subjects = [
        subject_id
        for subject_id in subject_ids
        if subject_id not in {selected_adapt_subject, selected_val_subject}
    ]
    if not train_subjects:
        raise ValueError("training subject list is empty after selecting validation and adaptation subjects")

    adapt_recordings = grouped_files[selected_adapt_subject]
    support_files = tuple(
        file_name
        for file_name in adapt_recordings
        if f"_{support_exercise}_" in file_name
    )
    query_files = tuple(
        file_name
        for file_name in adapt_recordings
        if f"_{query_exercise}_" in file_name
    )
    if not support_files:
        raise ValueError(
            f"adaptation subject '{selected_adapt_subject}' does not contain support exercise '{support_exercise}'"
        )
    if not query_files:
        raise ValueError(
            f"adaptation subject '{selected_adapt_subject}' does not contain query exercise '{query_exercise}'"
        )

    return {
        "train_subjects": tuple(train_subjects),
        "val_subject": selected_val_subject,
        "adapt_subject": selected_adapt_subject,
        "train_files": tuple(
            file_name
            for subject_id in train_subjects
            for file_name in grouped_files[subject_id]
        ),
        "val_files": tuple(grouped_files[selected_val_subject]),
        "support_files": support_files,
        "query_files": query_files,
    }


def make_jsonable(value: Any) -> Any:
    """Convert common numeric and path types into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    return value


def summarize_best_epoch(history: list[dict[str, float]]) -> dict[str, float]:
    """Extract the validation-best checkpoint record from the pretraining history."""
    if not history:
        raise ValueError("history is empty")
    return min(history, key=lambda entry: entry["val_mae"])


def load_sequence_split_from_files(file_names: tuple[str, ...], config) -> Any:
    """Load multiple recordings and convert them into one many-to-one sequence split."""
    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    recordings = [
        load_recording_features(available_files[file_name], config)
        for file_name in file_names
    ]
    return build_sequence_split(recordings, seq_len=config.seq_len, seq_stride=config.seq_stride)


def plot_subject_adaptation(
    zero_shot_evaluation: dict,
    adapted_evaluation: dict,
    *,
    split_name: str,
    target_index: int,
    max_points: int,
) -> plt.Figure:
    """
    Compare zero-shot and adapted predictions against the same held-out target trace.

    The plot is restricted to the first query recording so the time axis remains
    physically meaningful instead of concatenating separate recordings.
    """

    target_names = zero_shot_evaluation["target_names"]
    if not target_names:
        raise ValueError("evaluation does not contain target names")
    if not 0 <= target_index < len(target_names):
        raise ValueError(f"target_index must be in [0, {len(target_names) - 1}]")

    recording_ids = np.asarray(zero_shot_evaluation["recording_ids"])
    first_recording = recording_ids[0]
    same_recording = recording_ids == first_recording

    time_s = np.asarray(zero_shot_evaluation["time_s"], dtype=np.float32)[same_recording][:max_points]
    y_true = np.asarray(zero_shot_evaluation["y_true"], dtype=np.float32)[same_recording, target_index][:max_points]
    y_pred_zero_shot = np.asarray(zero_shot_evaluation["y_pred"], dtype=np.float32)[same_recording, target_index][:max_points]
    y_pred_adapted = np.asarray(adapted_evaluation["y_pred"], dtype=np.float32)[same_recording, target_index][:max_points]

    zero_shot_error = y_pred_zero_shot - y_true
    adapted_error = y_pred_adapted - y_true
    target_name = target_names[target_index]

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(time_s, y_true, label="Actual angle", linewidth=1.6, color="#1f77b4")
    axes[0].plot(time_s, y_pred_zero_shot, label="Zero-shot prediction", linewidth=1.2, color="#ff7f0e")
    axes[0].plot(time_s, y_pred_adapted, label="1-epoch adapted prediction", linewidth=1.2, color="#2ca02c")
    axes[0].set_title(
        f"CfC Regression on {split_name}\n"
        f"{first_recording} | target={target_name}",
        fontsize=12,
    )
    axes[0].set_ylabel(target_name, fontsize=10)
    axes[0].grid(True, linewidth=0.3, alpha=0.5)
    axes[0].legend(fontsize=9)

    axes[1].plot(time_s, zero_shot_error, label="Zero-shot error", linewidth=1.0, color="#ff7f0e")
    axes[1].plot(time_s, adapted_error, label="Adapted error", linewidth=1.0, color="#2ca02c")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)", fontsize=10)
    axes[1].set_ylabel("Prediction Error", fontsize=10)
    axes[1].grid(True, linewidth=0.3, alpha=0.5)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    return figure


def save_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Persist the experiment summary as JSON."""
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(summary), handle, indent=2)
    return summary_path


def run_subject_adaptation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed-hyperparameter pretrain-and-adapt protocol."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_best_cfc_config(
        db2_dir=args.db2_dir,
        device=args.device,
    )
    grouped_files = discover_target_files(base_config.db2_dir, base_config.target_source)
    if not grouped_files:
        raise FileNotFoundError(
            f"no recordings under {base_config.db2_dir} contain target source '{base_config.target_source}'"
        )

    protocol = select_adaptation_protocol(
        grouped_files,
        adapt_subject=args.adapt_subject,
        val_subject=args.val_subject,
        support_exercise=args.support_exercise,
        query_exercise=args.query_exercise,
    )

    overrides = {
        "db2_dir": base_config.db2_dir,
        "split_strategy": "recording",
        "train_files": protocol["train_files"],
        "val_files": protocol["val_files"],
        "test_files": protocol["query_files"],
        "target_columns": (args.target_column,),
        "device": args.device,
    }
    if args.target_offset_samples is not None:
        overrides["target_offset_samples"] = args.target_offset_samples
    if args.zc_threshold is not None:
        overrides["zc_threshold"] = args.zc_threshold
    if args.ssc_threshold is not None:
        overrides["ssc_threshold"] = args.ssc_threshold
    if args.feature_normalization is not None:
        overrides["feature_normalization"] = args.feature_normalization
    if args.target_normalization is not None:
        overrides["target_normalization"] = args.target_normalization

    pretrain_config = build_best_cfc_config(**overrides)

    print("Subject adaptation protocol")
    print(f"  train subjects : {list(protocol['train_subjects'])}")
    print(f"  val subject    : {protocol['val_subject']}")
    print(f"  adapt subject  : {protocol['adapt_subject']}")
    print(f"  target column  : {args.target_column}")
    print(f"  target offset  : {pretrain_config.target_offset_samples} samples")
    print(f"  normalization  : x={pretrain_config.feature_normalization}, y={pretrain_config.target_normalization}")
    print(f"  ZC/SSC thresh  : {pretrain_config.zc_threshold} / {pretrain_config.ssc_threshold}")
    print(f"  support files  : {list(protocol['support_files'])}")
    print(f"  query files    : {list(protocol['query_files'])}")

    pretrain_results = train_cfc_regressor(pretrain_config)
    plt.close(pretrain_results["prediction_figure"])

    device = resolve_device(pretrain_config.device)
    support_split_raw = load_sequence_split_from_files(protocol["support_files"], pretrain_config)
    query_split_raw = load_sequence_split_from_files(protocol["query_files"], pretrain_config)

    support_split = normalize_sequence_inputs(
        support_split_raw,
        x_stats=pretrain_results["x_stats"],
        y_stats=pretrain_results["y_stats"],
    )
    query_split = normalize_sequence_inputs(
        query_split_raw,
        x_stats=pretrain_results["x_stats"],
        y_stats=pretrain_results["y_stats"],
    )

    adapted_model = copy.deepcopy(pretrain_results["model"]).to(device)
    support_loader = make_train_loader(support_split, pretrain_config)
    adapt_optimizer = torch.optim.AdamW(
        adapted_model.parameters(),
        lr=args.adapt_learning_rate,
        weight_decay=pretrain_config.weight_decay,
    )
    adapt_loss = train_one_epoch(
        adapted_model,
        support_loader,
        adapt_optimizer,
        nn.SmoothL1Loss(),
        device=device,
        gradient_clip_norm=pretrain_config.gradient_clip_norm,
    )
    adapted_evaluation = evaluate_split(
        adapted_model,
        query_split,
        target_stats=pretrain_results["y_stats"],
        batch_size=pretrain_config.batch_size,
        device=device,
    )
    zero_shot_evaluation = pretrain_results["evaluations"]["test"]

    comparison_figure = plot_subject_adaptation(
        zero_shot_evaluation,
        adapted_evaluation,
        split_name="adaptation query",
        target_index=0,
        max_points=pretrain_config.plot_max_points,
    )
    figure_path = output_dir / "adaptation_prediction.png"
    comparison_figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(comparison_figure)

    summary = {
        "protocol": "pretrain_then_one_epoch_subject_adaptation",
        "config": make_jsonable(asdict(pretrain_config)),
        "subjects": {
            "train_subjects": list(protocol["train_subjects"]),
            "val_subject": protocol["val_subject"],
            "adapt_subject": protocol["adapt_subject"],
        },
        "files": {
            "train_files": list(protocol["train_files"]),
            "val_files": list(protocol["val_files"]),
            "support_files": list(protocol["support_files"]),
            "query_files": list(protocol["query_files"]),
        },
        "pretrain_best_epoch": make_jsonable(summarize_best_epoch(pretrain_results["history"])),
        "pretrain_history": make_jsonable(pretrain_results["history"]),
        "support_one_epoch_loss": float(adapt_loss),
        "zero_shot_test_metrics": make_jsonable(zero_shot_evaluation["metrics"]),
        "adapted_test_metrics": make_jsonable(adapted_evaluation["metrics"]),
        "artifacts": {
            "prediction_plot": str(figure_path),
        },
    }
    summary_path = save_summary(output_dir, summary)

    print("\nFinal adaptation metrics")
    print(
        f"  zero-shot | MAE={zero_shot_evaluation['metrics']['mae_mean']:.5f} | "
        f"RMSE={zero_shot_evaluation['metrics']['rmse_mean']:.5f} | "
        f"R2={zero_shot_evaluation['metrics']['r2_mean']:.5f}"
    )
    print(
        f"  adapted   | MAE={adapted_evaluation['metrics']['mae_mean']:.5f} | "
        f"RMSE={adapted_evaluation['metrics']['rmse_mean']:.5f} | "
        f"R2={adapted_evaluation['metrics']['r2_mean']:.5f}"
    )
    print(f"Saved summary to: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_subject_adaptation(args)


if __name__ == "__main__":
    main()
