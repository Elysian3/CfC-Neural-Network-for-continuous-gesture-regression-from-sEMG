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

from run_db2_subject_adaptation import (
    DEFAULT_OUTPUT_DIR,
    discover_target_files,
    load_sequence_split_from_files,
    make_jsonable,
    save_summary,
    select_adaptation_protocol,
    summarize_best_epoch,
)
from train import (
    build_action_stratified_sequence_splits,
    build_blocked_sequence_splits,
    build_best_cfc_config,
    evaluate_split,
    list_db2_files,
    load_recording_features,
    make_train_loader,
    normalize_sequence_inputs,
    plot_angle_predictions,
    resolve_device,
    train_cfc_regressor,
    train_one_epoch,
    WeightedSmoothL1Loss,
)

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATAFLOW_DIR = SCRIPT_DIR.parent / "dataflow"
if str(DATAFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_DIR))

from doa_mapping import (  # noqa: E402
    DOA5_MAPPING_NAME,
    DOA5_MAPPING_VERSION,
    doa5_mapping_metadata,
)


PASS_R2_THRESHOLD = 0.5
MILD_FAILED_DOA_WEIGHTS = (1.0, 1.25, 1.0, 1.25, 1.0)
FAILED_DOA_INDICES = (1, 3)
FORBIDDEN_SELECTION_KEYS = {
    "adapted_test_metrics",
    "zero_shot_test_metrics",
    "pass_by_doa",
    "mvp_pass",
    "failing_doa",
    "prediction_plots",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-DoA multi-subject pretrain + S1 fine-tune MVP protocol."
    )
    parser.add_argument("--db2-dir", type=Path, default=REPO_ROOT / "src" / "data" / "DB2")
    parser.add_argument("--adapt-subject", type=str, default="S1")
    parser.add_argument("--val-subject", type=str, default=None)
    parser.add_argument("--support-exercise", type=str, default="E1")
    parser.add_argument("--query-exercise", type=str, default="E2")
    parser.add_argument(
        "--s1-split-mode",
        choices=("action_stratified", "blocked_time", "exercise_holdout"),
        default="action_stratified",
        help="Final target-subject split. Use action_stratified for same-action-distribution MVP evaluation.",
    )
    parser.add_argument("--target-offset-samples", type=int, default=None)
    parser.add_argument("--zc-threshold", type=float, default=None)
    parser.add_argument("--ssc-threshold", type=float, default=None)
    parser.add_argument("--feature-normalization", choices=("zscore", "mu_law"), default=None)
    parser.add_argument("--target-normalization", choices=("zscore", "mu_law"), default=None)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-patience", type=int, default=4)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument(
        "--loss-profile",
        choices=("baseline", "mild_failed_doa"),
        default="baseline",
    )
    parser.add_argument(
        "--candidate-selection-only",
        action="store_true",
        help="Run the validation-only candidate grid and do not evaluate S1 held-out test.",
    )
    parser.add_argument(
        "--candidate-hidden-units",
        type=str,
        default="64,128",
        help="Comma-separated hidden-unit candidates for validation-only selection.",
    )
    parser.add_argument(
        "--candidate-loss-profiles",
        type=str,
        default="baseline,mild_failed_doa",
        help="Comma-separated loss-profile candidates for validation-only selection.",
    )
    parser.add_argument(
        "--selection-artifact",
        type=Path,
        default=None,
        help="Frozen candidate-selection summary required before running S1 held-out final test.",
    )
    parser.add_argument(
        "--pretrain-test-subject",
        type=str,
        default=None,
        help=(
            "Optional non-adaptation subject reserved for pretrain-only test reporting. "
            "Leave unset when the final endpoint is S1 adaptation and the pretrain pool should not be reduced."
        ),
    )
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "doa5_subject_adaptation",
        help="Directory used to save DoA5 plots and summary JSON.",
    )
    return parser.parse_args()


def optional_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for name in (
        "target_offset_samples",
        "zc_threshold",
        "ssc_threshold",
        "feature_normalization",
        "target_normalization",
        "max_windows_per_file",
        "hidden_units",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return overrides


def per_doa_pass(metrics: dict[str, Any], target_names: list[str]) -> dict[str, bool]:
    r2_values = metrics["r2_by_target"]
    return {
        target_name: float(r2_values[index]) >= PASS_R2_THRESHOLD
        for index, target_name in enumerate(target_names)
    }


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("candidate hidden-unit list cannot be empty")
    return items


def parse_loss_profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"baseline", "mild_failed_doa"}
    invalid = sorted(set(profiles) - allowed)
    if invalid:
        raise ValueError(f"unsupported loss profiles: {invalid}")
    if not profiles:
        raise ValueError("candidate loss-profile list cannot be empty")
    return profiles


def make_fine_tune_loss(loss_profile: str, target_dim: int) -> nn.Module:
    if loss_profile == "baseline":
        return nn.SmoothL1Loss()
    if loss_profile == "mild_failed_doa":
        if target_dim != len(MILD_FAILED_DOA_WEIGHTS):
            raise ValueError(
                f"mild_failed_doa expects {len(MILD_FAILED_DOA_WEIGHTS)} targets, got {target_dim}"
            )
        return WeightedSmoothL1Loss(MILD_FAILED_DOA_WEIGHTS)
    raise ValueError(f"unsupported loss_profile: {loss_profile}")


def validation_selection_values(metrics: dict[str, Any]) -> dict[str, Any]:
    r2_values = np.asarray(metrics["r2_by_target"], dtype=np.float32)
    return {
        "worst_doa_r2": float(np.min(r2_values)),
        "pass_count": int(np.sum(r2_values >= PASS_R2_THRESHOLD)),
        "mean_r2": float(metrics["r2_mean"]),
        "mean_mae": float(metrics["mae_mean"]),
    }


def candidate_id(hidden_units: int, loss_profile: str) -> str:
    return f"h{hidden_units}_{loss_profile}"


def _candidate_by_config(candidates: list[dict[str, Any]], hidden_units: int, loss_profile: str) -> dict[str, Any] | None:
    for candidate in candidates:
        config = candidate["candidate_config"]
        if config["hidden_units"] == hidden_units and config["loss_profile"] == loss_profile:
            return candidate
    return None


def hidden_units_delta_passes(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    candidate_r2 = np.asarray(candidate["adapted_val_metrics"]["r2_by_target"], dtype=np.float32)
    baseline_r2 = np.asarray(baseline["adapted_val_metrics"]["r2_by_target"], dtype=np.float32)
    worst_delta = float(np.min(candidate_r2) - np.min(baseline_r2))
    candidate_passes = candidate_r2 >= PASS_R2_THRESHOLD
    baseline_passes = baseline_r2 >= PASS_R2_THRESHOLD
    adds_pass = int(np.sum(candidate_passes)) > int(np.sum(baseline_passes))
    preserves_passes = bool(np.all(candidate_passes[baseline_passes]))
    return worst_delta >= (0.02 - 1e-6) or (adds_pass and preserves_passes)


def weighted_loss_delta_passes(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    candidate_r2 = np.asarray(candidate["adapted_val_metrics"]["r2_by_target"], dtype=np.float32)
    baseline_r2 = np.asarray(baseline["adapted_val_metrics"]["r2_by_target"], dtype=np.float32)
    failed_delta = float(
        np.min(candidate_r2[list(FAILED_DOA_INDICES)])
        - np.min(baseline_r2[list(FAILED_DOA_INDICES)])
    )
    baseline_passes = baseline_r2 >= PASS_R2_THRESHOLD
    if np.any((baseline_r2 - candidate_r2)[baseline_passes] > (0.02 + 1e-6)):
        return False
    return failed_delta >= (0.015 - 1e-6)


def candidate_clears_delta_rules(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    config = candidate["candidate_config"]
    hidden_units = int(config["hidden_units"])
    loss_profile = str(config["loss_profile"])
    if hidden_units == 64 and loss_profile == "baseline":
        return True

    if hidden_units > 64:
        hidden_baseline = _candidate_by_config(candidates, 64, loss_profile)
        if hidden_baseline is None or not hidden_units_delta_passes(candidate, hidden_baseline):
            return False

    if loss_profile != "baseline":
        loss_baseline = _candidate_by_config(candidates, hidden_units, "baseline")
        if loss_baseline is None or not weighted_loss_delta_passes(candidate, loss_baseline):
            return False

    return True


def select_validation_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no candidates available for selection")

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["selection_values"]["worst_doa_r2"],
            candidate["selection_values"]["pass_count"],
            candidate["selection_values"]["mean_r2"],
            -candidate["selection_values"]["mean_mae"],
            -int(candidate["candidate_config"]["hidden_units"]),
            0 if candidate["candidate_config"]["loss_profile"] == "baseline" else -1,
        ),
        reverse=True,
    )
    for candidate in ranked:
        if candidate_clears_delta_rules(candidate, candidates):
            return candidate

    return _candidate_by_config(candidates, 64, "baseline") or ranked[-1]


def assert_selection_summary_is_uncontaminated(summary: dict[str, Any]) -> None:
    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in FORBIDDEN_SELECTION_KEYS:
                    raise ValueError(f"selection summary contains forbidden key: {'.'.join(path + (key,))}")
                visit(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (str(index),))

    visit(summary)


def save_per_doa_plots(evaluation: dict, output_dir: Path, *, prefix: str, max_points: int) -> dict[str, str]:
    plot_paths: dict[str, str] = {}
    target_names = list(evaluation["target_names"])
    for index, target_name in enumerate(target_names):
        figure = plot_angle_predictions(
            evaluation,
            split_name=prefix,
            target_index=index,
            max_points=max_points,
        )
        figure_path = output_dir / f"{prefix}_{target_name}_prediction.png"
        figure.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        plot_paths[target_name] = str(figure_path)
    return plot_paths


def load_blocked_subject_splits(subject_files: tuple[str, ...], config) -> dict[str, Any]:
    """Build train/val/test splits inside each target-subject recording."""
    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    recordings = [
        load_recording_features(available_files[file_name], config)
        for file_name in subject_files
    ]
    return build_blocked_sequence_splits(
        recordings,
        seq_len=config.seq_len,
        seq_stride=config.seq_stride,
        train_fraction=config.blocked_train_fraction,
        val_fraction=config.blocked_val_fraction,
        test_fraction=config.blocked_test_fraction,
        gap_windows=config.blocked_gap_windows,
    )


def load_action_stratified_subject_splits(subject_files: tuple[str, ...], config) -> dict[str, Any]:
    """Build train/val/test splits inside every action/repetition segment."""
    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    recordings = [
        load_recording_features(available_files[file_name], config)
        for file_name in subject_files
    ]
    return build_action_stratified_sequence_splits(
        recordings,
        seq_len=config.seq_len,
        seq_stride=config.seq_stride,
        train_fraction=config.blocked_train_fraction,
        val_fraction=config.blocked_val_fraction,
        test_fraction=config.blocked_test_fraction,
        gap_windows=config.blocked_gap_windows,
    )


def summarize_split_actions(splits: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Summarize action coverage so invalid held-out-action protocols are visible."""
    summary: dict[str, dict[str, Any]] = {}
    for split_name, split in splits.items():
        action_labels = split.action_labels
        if action_labels is None:
            summary[split_name] = {"actions": [], "counts": {}}
            continue
        unique_actions, counts = torch.unique(
            torch.as_tensor(action_labels.astype("int64")),
            return_counts=True,
        )
        summary[split_name] = {
            "actions": [int(value) for value in unique_actions.tolist()],
            "counts": {
                str(int(action)): int(count)
                for action, count in zip(unique_actions.tolist(), counts.tolist())
            },
        }
    return summary


def omit_split(summary: dict[str, Any], split_name: str) -> dict[str, Any]:
    """Return split metadata without a held-out split."""
    return {
        name: value
        for name, value in summary.items()
        if name != split_name
    }


def split_pretrain_subjects(
    protocol: dict[str, Any],
    grouped_files: dict[str, list[str]],
    *,
    pretrain_test_subject: str | None,
) -> tuple[tuple[str, ...], str | None, tuple[str, ...], tuple[str, ...]]:
    """Resolve pretraining files, optionally reserving a subject for pretrain-only test reporting."""
    train_subjects = tuple(protocol["train_subjects"])
    if pretrain_test_subject is None:
        pretrain_train_files = tuple(protocol["train_files"])
        return train_subjects, None, pretrain_train_files, tuple(protocol["val_files"])

    if pretrain_test_subject not in train_subjects:
        raise ValueError("pretrain test subject must be one of the non-validation pretrain subjects")
    if len(train_subjects) < 2:
        raise ValueError("need at least two non-adaptation training subjects for pretrain holdout")

    pretrain_train_subjects = tuple(
        subject_id for subject_id in train_subjects if subject_id != pretrain_test_subject
    )
    pretrain_train_files = tuple(
        file_name
        for subject_id in pretrain_train_subjects
        for file_name in grouped_files[subject_id]
    )
    pretrain_test_files = tuple(grouped_files[pretrain_test_subject])
    return pretrain_train_subjects, pretrain_test_subject, pretrain_train_files, pretrain_test_files


def build_protocol_context(args: argparse.Namespace, *, hidden_units: int) -> dict[str, Any]:
    base_config = build_best_cfc_config(
        db2_dir=args.db2_dir,
        target_source="glove",
        target_columns=tuple(range(22)),
        target_mapping=DOA5_MAPPING_NAME,
        target_mapping_source="glove",
        target_mapping_version=DOA5_MAPPING_VERSION,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        hidden_units=hidden_units,
        **{key: value for key, value in optional_overrides(args).items() if key != "hidden_units"},
    )
    grouped_files = discover_target_files(base_config.db2_dir, base_config.target_mapping_source)
    protocol = select_adaptation_protocol(
        grouped_files,
        adapt_subject=args.adapt_subject,
        val_subject=args.val_subject,
        support_exercise=args.support_exercise,
        query_exercise=args.query_exercise,
    )
    (
        pretrain_train_subjects,
        pretrain_test_subject,
        pretrain_train_files,
        pretrain_test_files,
    ) = split_pretrain_subjects(
        protocol,
        grouped_files,
        pretrain_test_subject=args.pretrain_test_subject,
    )
    pretrain_config = build_best_cfc_config(
        **{
            **asdict(base_config),
            "split_strategy": "recording",
            "train_files": pretrain_train_files,
            "val_files": protocol["val_files"],
            "test_files": pretrain_test_files,
        }
    )
    return {
        "grouped_files": grouped_files,
        "protocol": protocol,
        "pretrain_train_subjects": pretrain_train_subjects,
        "pretrain_test_subject": pretrain_test_subject,
        "pretrain_train_files": pretrain_train_files,
        "pretrain_test_files": pretrain_test_files,
        "pretrain_config": pretrain_config,
    }


def load_s1_raw_splits(args: argparse.Namespace, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = context["protocol"]
    grouped_files = context["grouped_files"]
    pretrain_config = context["pretrain_config"]
    if args.s1_split_mode == "action_stratified":
        s1_splits_raw = load_action_stratified_subject_splits(
            tuple(grouped_files[protocol["adapt_subject"]]),
            pretrain_config,
        )
    elif args.s1_split_mode == "blocked_time":
        s1_splits_raw = load_blocked_subject_splits(
            tuple(grouped_files[protocol["adapt_subject"]]),
            pretrain_config,
        )
    else:
        support_split_raw = load_sequence_split_from_files(protocol["support_files"], pretrain_config)
        query_split_raw = load_sequence_split_from_files(protocol["query_files"], pretrain_config)
        s1_splits_raw = {
            "train": support_split_raw,
            "val": support_split_raw,
            "test": query_split_raw,
        }
    return s1_splits_raw, summarize_split_actions(s1_splits_raw)


def normalize_s1_splits(s1_splits_raw: dict[str, Any], pretrain_results: dict[str, Any]) -> dict[str, Any]:
    return {
        split_name: normalize_sequence_inputs(
            split,
            x_stats=pretrain_results["x_stats"],
            y_stats=pretrain_results["y_stats"],
        )
        for split_name, split in s1_splits_raw.items()
    }


def fine_tune_with_validation(
    model: nn.Module,
    *,
    support_split,
    adapt_val_split,
    pretrain_config,
    target_stats: dict,
    args: argparse.Namespace,
    loss_profile: str,
    device: torch.device,
) -> dict[str, Any]:
    adapted_model = model.to(device)
    support_loader = make_train_loader(support_split, pretrain_config)
    optimizer = torch.optim.AdamW(
        adapted_model.parameters(),
        lr=args.fine_tune_learning_rate,
        weight_decay=pretrain_config.weight_decay,
    )
    loss_fn = make_fine_tune_loss(loss_profile, support_split.y.shape[1]).to(device)
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(adapted_model.state_dict())
    best_epoch = 0
    best_worst_r2 = float("-inf")
    patience_counter = 0

    for epoch in range(1, args.fine_tune_epochs + 1):
        loss = train_one_epoch(
            adapted_model,
            support_loader,
            optimizer,
            loss_fn,
            device=device,
            gradient_clip_norm=pretrain_config.gradient_clip_norm,
        )
        val_evaluation = evaluate_split(
            adapted_model,
            adapt_val_split,
            target_stats=target_stats,
            batch_size=pretrain_config.batch_size,
            device=device,
        )
        values = validation_selection_values(val_evaluation["metrics"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss),
                "val_worst_doa_r2": values["worst_doa_r2"],
                "val_pass_count": values["pass_count"],
                "val_mean_r2": values["mean_r2"],
                "val_mean_mae": values["mean_mae"],
            }
        )
        if values["worst_doa_r2"] > best_worst_r2:
            best_worst_r2 = values["worst_doa_r2"]
            best_state = copy.deepcopy(adapted_model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.fine_tune_patience:
                break

    adapted_model.load_state_dict(best_state)
    best_val_evaluation = evaluate_split(
        adapted_model,
        adapt_val_split,
        target_stats=target_stats,
        batch_size=pretrain_config.batch_size,
        device=device,
    )
    return {
        "model": adapted_model,
        "history": history,
        "best_epoch": best_epoch,
        "adapted_val_evaluation": best_val_evaluation,
    }


def compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": make_jsonable(evaluation["metrics"]),
        "per_action_metrics": make_jsonable(evaluation["per_action_metrics"]),
    }


def run_candidate_selection(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_unit_candidates = parse_int_list(args.candidate_hidden_units)
    loss_profiles = parse_loss_profiles(args.candidate_loss_profiles)
    candidates: list[dict[str, Any]] = []
    first_context: dict[str, Any] | None = None
    s1_action_coverage = None

    for hidden_units in hidden_unit_candidates:
        context = build_protocol_context(args, hidden_units=hidden_units)
        if first_context is None:
            first_context = context
        pretrain_config = context["pretrain_config"]
        print("Five-DoA candidate selection")
        print(f"  hidden_units : {hidden_units}")
        print(f"  loss profiles: {list(loss_profiles)}")
        pretrain_results = train_cfc_regressor(pretrain_config)
        plt.close(pretrain_results["prediction_figure"])
        device = resolve_device(pretrain_config.device)
        s1_splits_raw, s1_action_coverage = load_s1_raw_splits(args, context)
        normalized_splits = normalize_s1_splits(s1_splits_raw, pretrain_results)

        for loss_profile in loss_profiles:
            fine_tune_result = fine_tune_with_validation(
                copy.deepcopy(pretrain_results["model"]),
                support_split=normalized_splits["train"],
                adapt_val_split=normalized_splits["val"],
                pretrain_config=pretrain_config,
                target_stats=pretrain_results["y_stats"],
                args=args,
                loss_profile=loss_profile,
                device=device,
            )
            val_evaluation = fine_tune_result["adapted_val_evaluation"]
            candidate = {
                "candidate_id": candidate_id(hidden_units, loss_profile),
                "target_names": list(val_evaluation["target_names"]),
                "candidate_config": {
                    "hidden_units": hidden_units,
                    "loss_profile": loss_profile,
                    "loss_weights": list(MILD_FAILED_DOA_WEIGHTS) if loss_profile == "mild_failed_doa" else None,
                    "fine_tune_epochs": args.fine_tune_epochs,
                    "fine_tune_patience": args.fine_tune_patience,
                    "fine_tune_learning_rate": args.fine_tune_learning_rate,
                },
                "pretrain_best_epoch": make_jsonable(summarize_best_epoch(pretrain_results["history"])),
                "fine_tune_history": make_jsonable(fine_tune_result["history"]),
                "selected_checkpoint_epoch": fine_tune_result["best_epoch"],
                "adapted_val_metrics": make_jsonable(val_evaluation["metrics"]),
                "adapted_val_per_action_metrics": make_jsonable(val_evaluation["per_action_metrics"]),
                "selection_values": validation_selection_values(val_evaluation["metrics"]),
                "capacity_delta": {
                    "hidden_units": hidden_units,
                    "relative_to_64": float(hidden_units / 64.0),
                },
            }
            candidates.append(candidate)

    selected_candidate = select_validation_candidate(candidates)
    assert first_context is not None
    summary = {
        "protocol": "doa5_validation_only_candidate_selection",
        "contains_s1_test_metrics": False,
        "selection_rule": "S1 validation only; S1 held-out test is not evaluated in this phase",
        "s1_split_mode": args.s1_split_mode,
        "mapping": doa5_mapping_metadata(),
        "target_names": list(candidates[0]["target_names"]),
        "subjects": {
            "train_subjects": list(first_context["pretrain_train_subjects"]),
            "pretrain_test_subject": first_context["pretrain_test_subject"],
            "val_subject": first_context["protocol"]["val_subject"],
            "adapt_subject": first_context["protocol"]["adapt_subject"],
        },
        "files": {
            "train_files": list(first_context["pretrain_train_files"]),
            "val_files": list(first_context["protocol"]["val_files"]),
            "support_files": list(first_context["protocol"]["support_files"]),
            "adapt_subject_all_files": list(first_context["grouped_files"][first_context["protocol"]["adapt_subject"]]),
        },
        "base_config": make_jsonable(asdict(first_context["pretrain_config"])),
        "candidate_grid": {
            "hidden_units": list(hidden_unit_candidates),
            "loss_profiles": list(loss_profiles),
        },
        "delta_rules": {
            "hidden_units_128": "worst_doa_val_r2_delta >= 0.02 or one additional DoA val pass without passed-DoA regression",
            "weighted_loss": "failed_doa_min_val_r2_delta >= 0.015 and no passed DoA loses more than 0.02 R2",
        },
        "selection_priority": [
            "worst_doa_val_r2",
            "val_pass_count",
            "val_mean_r2",
            "val_mean_mae",
            "minimum_delta_rules",
            "lower_hidden_units",
            "baseline_loss",
        ],
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_candidate": selected_candidate,
        "candidates": candidates,
        "s1_action_coverage": omit_split(s1_action_coverage or {}, "test"),
    }
    assert_selection_summary_is_uncontaminated(summary)
    summary_path = save_summary(output_dir, summary)
    print(f"Saved DoA5 candidate-selection summary to: {summary_path}")
    return summary


def run_doa5_subject_adaptation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_selection_only:
        return run_candidate_selection(args)

    if args.selection_artifact is None:
        raise ValueError("final S1 held-out evaluation requires --selection-artifact")

    with args.selection_artifact.open("r", encoding="utf-8") as handle:
        selection_summary = json.load(handle)
    if selection_summary.get("contains_s1_test_metrics") is not False:
        raise ValueError("selection artifact must declare contains_s1_test_metrics=false")
    assert_selection_summary_is_uncontaminated(selection_summary)
    selected_config = selection_summary["selected_candidate"]["candidate_config"]
    args.hidden_units = int(selected_config["hidden_units"])
    args.loss_profile = str(selected_config["loss_profile"])

    context = build_protocol_context(args, hidden_units=args.hidden_units)
    grouped_files = context["grouped_files"]
    protocol = context["protocol"]
    pretrain_train_subjects = context["pretrain_train_subjects"]
    pretrain_test_subject = context["pretrain_test_subject"]
    pretrain_train_files = context["pretrain_train_files"]
    pretrain_test_files = context["pretrain_test_files"]
    pretrain_config = context["pretrain_config"]

    print("Five-DoA subject adaptation protocol")
    print(f"  mapping       : {DOA5_MAPPING_NAME} ({DOA5_MAPPING_VERSION})")
    print(f"  selected      : hidden_units={args.hidden_units}, loss_profile={args.loss_profile}")
    print(f"  train subjects: {list(pretrain_train_subjects)}")
    print(f"  pretrain test : {pretrain_test_subject or protocol['val_subject']}")
    print(f"  val subject   : {protocol['val_subject']}")
    print(f"  adapt subject : {protocol['adapt_subject']}")
    print(f"  S1 split mode : {args.s1_split_mode}")

    pretrain_results = train_cfc_regressor(pretrain_config)
    plt.close(pretrain_results["prediction_figure"])

    device = resolve_device(pretrain_config.device)
    s1_splits_raw, s1_action_coverage = load_s1_raw_splits(args, context)
    normalized_splits = normalize_s1_splits(s1_splits_raw, pretrain_results)
    zero_shot_model = copy.deepcopy(pretrain_results["model"]).to(device)
    fine_tune_result = fine_tune_with_validation(
        pretrain_results["model"],
        support_split=normalized_splits["train"],
        adapt_val_split=normalized_splits["val"],
        pretrain_config=pretrain_config,
        target_stats=pretrain_results["y_stats"],
        args=args,
        loss_profile=args.loss_profile,
        device=device,
    )
    adapted_model = fine_tune_result["model"]

    adapted_evaluation = evaluate_split(
        adapted_model,
        normalized_splits["test"],
        target_stats=pretrain_results["y_stats"],
        batch_size=pretrain_config.batch_size,
        device=device,
    )
    adapted_val_evaluation = fine_tune_result["adapted_val_evaluation"]
    zero_shot_evaluation = evaluate_split(
        zero_shot_model,
        normalized_splits["test"],
        target_stats=pretrain_results["y_stats"],
        batch_size=pretrain_config.batch_size,
        device=device,
    )
    target_names = list(adapted_evaluation["target_names"])
    pass_by_doa = per_doa_pass(adapted_evaluation["metrics"], target_names)
    mvp_pass = all(pass_by_doa.values())

    plot_paths = save_per_doa_plots(
        adapted_evaluation,
        output_dir,
        prefix="s1_heldout_adapted",
        max_points=pretrain_config.plot_max_points,
    )

    summary = {
        "protocol": "doa5_multi_subject_pretrain_s1_finetune_s1_heldout",
        "selection_rule": "mapping and hyperparameters must be selected without S1 held-out test metrics",
        "s1_split_mode": args.s1_split_mode,
        "mapping": doa5_mapping_metadata(),
        "target_names": target_names,
        "subjects": {
            "train_subjects": list(pretrain_train_subjects),
            "pretrain_test_subject": pretrain_test_subject,
            "val_subject": protocol["val_subject"],
            "adapt_subject": protocol["adapt_subject"],
        },
        "files": {
            "train_files": list(pretrain_train_files),
            "val_files": list(protocol["val_files"]),
            "pretrain_test_files": list(pretrain_test_files),
            "support_files": list(protocol["support_files"]),
            "query_files": list(protocol["query_files"]),
            "adapt_subject_all_files": list(grouped_files[protocol["adapt_subject"]]),
        },
        "config": make_jsonable(asdict(pretrain_config)),
        "pretrain_best_epoch": make_jsonable(summarize_best_epoch(pretrain_results["history"])),
        "pretrain_history": make_jsonable(pretrain_results["history"]),
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_patience": args.fine_tune_patience,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "fine_tune_loss_profile": args.loss_profile,
        "fine_tune_loss_weights": list(MILD_FAILED_DOA_WEIGHTS) if args.loss_profile == "mild_failed_doa" else None,
        "fine_tune_history": make_jsonable(fine_tune_result["history"]),
        "selected_checkpoint_epoch": fine_tune_result["best_epoch"],
        "selection_artifact": str(args.selection_artifact),
        "selection_artifact_candidate_id": selection_summary["selected_candidate_id"],
        "s1_action_coverage": s1_action_coverage,
        "zero_shot_test_metrics": compact_evaluation(zero_shot_evaluation),
        "adapted_val_metrics": compact_evaluation(adapted_val_evaluation),
        "adapted_test_metrics": compact_evaluation(adapted_evaluation),
        "pass_r2_threshold": PASS_R2_THRESHOLD,
        "pass_by_doa": pass_by_doa,
        "mvp_pass": mvp_pass,
        "conclusion": "solved" if mvp_pass else "not_solved",
        "failing_doa": [name for name, passed in pass_by_doa.items() if not passed],
        "artifacts": {
            "prediction_plots": plot_paths,
        },
    }
    summary_path = save_summary(output_dir, summary)
    print(f"Saved DoA5 summary to: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_doa5_subject_adaptation(args)


if __name__ == "__main__":
    main()
