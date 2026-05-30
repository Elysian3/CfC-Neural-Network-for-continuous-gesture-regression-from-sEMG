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
    build_cfc_regressor,
    evaluate_split,
    list_db2_files,
    load_recording_features,
    make_train_loader,
    normalize_sequence_inputs,
    plot_angle_predictions,
    resolve_device,
    SequenceSplit,
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
MODEL_FAMILIES = ("autoncp", "dense_cfc_linear")
ADAPTATION_MODES = (
    "autoncp_full",
    "autoncp_motor_only",
    "autoncp_motor_command",
    "dense_linear_head",
    "dense_linear_head_then_body",
    "dense_full",
)
DEFAULT_ADAPTATION_MODE = "autoncp_full"
PARAMETER_MASK_RULE_VERSION = "autoncp_layer_prefix_v1"
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
    parser.add_argument("--body-fine-tune-learning-rate", type=float, default=None)
    parser.add_argument("--linear-probe-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-patience", type=int, default=4)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, default="autoncp")
    parser.add_argument(
        "--candidate-model-families",
        type=str,
        default="autoncp",
        help="Comma-separated model families for validation-only selection.",
    )
    parser.add_argument("--cfc-dropout", type=float, default=0.0)
    parser.add_argument(
        "--adaptation-mode",
        choices=ADAPTATION_MODES,
        default=DEFAULT_ADAPTATION_MODE,
        help="AutoNCP adaptation surface used for final S1 adaptation.",
    )
    parser.add_argument(
        "--candidate-adaptation-modes",
        type=str,
        default=DEFAULT_ADAPTATION_MODE,
        help=(
            "Comma-separated adaptation modes for validation-only selection. "
            "Use AutoNCP modes for NCP comparisons and dense_linear_head_then_body for dense LP-FT."
        ),
    )
    parser.add_argument(
        "--support-sequences-per-action",
        type=int,
        default=None,
        help=(
            "Limit S1 adaptation support to at most this many sequences per action. "
            "Use this to simulate realistic short calibration budgets."
        ),
    )
    parser.add_argument(
        "--support-seconds-total",
        type=float,
        default=None,
        help=(
            "Limit S1 adaptation support by approximate raw signal seconds before windowing. "
            "The budget is converted to sequence count from stride_ms * seq_stride and balanced across actions."
        ),
    )
    parser.add_argument("--hidden-units", type=int, default=128)
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
        default="128",
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
        "--no-save-checkpoints",
        dest="save_checkpoints",
        action="store_false",
        help="Disable checkpoint persistence for pretrain and adaptation runs.",
    )
    parser.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Ignore reusable pretrain checkpoints in selection artifacts and train S2-S10 again.",
    )
    parser.set_defaults(save_checkpoints=True)
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
        "model_family",
        "cfc_dropout",
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


def parse_adaptation_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(modes) - set(ADAPTATION_MODES))
    if invalid:
        raise ValueError(f"unsupported adaptation modes: {invalid}")
    if not modes:
        raise ValueError("candidate adaptation-mode list cannot be empty")
    return modes


def parse_model_families(value: str) -> tuple[str, ...]:
    families = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(families) - set(MODEL_FAMILIES))
    if invalid:
        raise ValueError(f"unsupported model families: {invalid}")
    if not families:
        raise ValueError("candidate model-family list cannot be empty")
    return families


def validate_model_adaptation_pair(model_family: str, adaptation_mode: str) -> None:
    if model_family == "autoncp" and not adaptation_mode.startswith("autoncp_"):
        raise ValueError(f"{adaptation_mode} is incompatible with model_family={model_family}")
    if model_family == "dense_cfc_linear" and not adaptation_mode.startswith("dense_"):
        raise ValueError(f"{adaptation_mode} is incompatible with model_family={model_family}")


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


def candidate_id(
    hidden_units: int,
    loss_profile: str,
    adaptation_mode: str = DEFAULT_ADAPTATION_MODE,
    model_family: str = "autoncp",
) -> str:
    base = f"h{hidden_units}_{loss_profile}"
    if model_family == "autoncp" and adaptation_mode == DEFAULT_ADAPTATION_MODE:
        return base
    return f"{base}_{model_family}_{adaptation_mode}"


def _candidate_by_config(
    candidates: list[dict[str, Any]],
    hidden_units: int,
    loss_profile: str,
    adaptation_mode: str | None = None,
    model_family: str | None = None,
) -> dict[str, Any] | None:
    for candidate in candidates:
        config = candidate["candidate_config"]
        if config["hidden_units"] != hidden_units or config["loss_profile"] != loss_profile:
            continue
        if adaptation_mode is not None and config.get("adaptation_mode", DEFAULT_ADAPTATION_MODE) != adaptation_mode:
            continue
        if model_family is not None and config.get("model_family", "autoncp") != model_family:
            continue
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
    adaptation_mode = str(config.get("adaptation_mode", DEFAULT_ADAPTATION_MODE))
    if hidden_units == 64 and loss_profile == "baseline":
        return True

    if hidden_units > 64:
        hidden_baseline = _candidate_by_config(candidates, 64, loss_profile, adaptation_mode)
        if hidden_baseline is not None and not hidden_units_delta_passes(candidate, hidden_baseline):
            return False

    if loss_profile != "baseline":
        loss_baseline = _candidate_by_config(candidates, hidden_units, "baseline", adaptation_mode)
        if loss_baseline is not None and not weighted_loss_delta_passes(candidate, loss_baseline):
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
            -float(candidate.get("parameter_audit", {}).get("trainable_fraction", 1.0)),
            -int(candidate["candidate_config"]["hidden_units"]),
            0 if candidate["candidate_config"]["loss_profile"] == "baseline" else -1,
        ),
        reverse=True,
    )
    for candidate in ranked:
        if candidate_clears_delta_rules(candidate, candidates):
            return candidate

    return _candidate_by_config(candidates, 64, "baseline", DEFAULT_ADAPTATION_MODE) or ranked[-1]


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


def _autoncp_layer_count(model: nn.Module) -> int:
    try:
        return int(model.cfc.rnn_cell.num_layers)  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ValueError("adaptation-mode masks require an AutoNCP wired CfC model") from exc


def adaptation_trainable_prefixes(model: nn.Module, adaptation_mode: str) -> tuple[str, ...]:
    if adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(f"unsupported adaptation_mode: {adaptation_mode}")
    if adaptation_mode == "autoncp_full":
        return ("*",)

    layer_count = _autoncp_layer_count(model)
    if layer_count <= 0:
        raise ValueError("AutoNCP wired CfC reports no layers")
    motor_layer = layer_count - 1
    if adaptation_mode == "autoncp_motor_only":
        layer_indices = (motor_layer,)
    else:
        if layer_count < 2:
            raise ValueError("autoncp_motor_command requires at least two wired CfC layers")
        layer_indices = (motor_layer - 1, motor_layer)
    return tuple(f"cfc.rnn_cell.layer_{index}." for index in layer_indices)


def apply_adaptation_mode(model: nn.Module, adaptation_mode: str) -> dict[str, Any]:
    prefixes = adaptation_trainable_prefixes(model, adaptation_mode)
    full = prefixes == ("*",)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_shapes: dict[str, list[int]] = {}
    frozen_shapes: dict[str, list[int]] = {}
    trainable_count = 0
    total_count = 0

    for name, parameter in model.named_parameters():
        is_wiring_mask = name.endswith("sparsity_mask")
        should_train = (full or any(name.startswith(prefix) for prefix in prefixes)) and not is_wiring_mask
        parameter.requires_grad = should_train
        count = int(parameter.numel())
        total_count += count
        shape = [int(value) for value in parameter.shape]
        if should_train:
            trainable_names.append(name)
            trainable_shapes[name] = shape
            trainable_count += count
        else:
            frozen_names.append(name)
            frozen_shapes[name] = shape

    if trainable_count == 0:
        raise ValueError(f"adaptation_mode {adaptation_mode} matched zero trainable parameters")

    return {
        "adaptation_mode": adaptation_mode,
        "mask_rule_version": PARAMETER_MASK_RULE_VERSION,
        "matched_trainable_prefixes": list(prefixes),
        "matched_frozen_prefixes": [] if full else ["* except matched_trainable_prefixes"],
        "trainable_param_names": trainable_names,
        "frozen_param_names": frozen_names,
        "trainable_param_shapes": trainable_shapes,
        "frozen_param_shapes": frozen_shapes,
        "trainable_param_count": trainable_count,
        "total_param_count": total_count,
        "frozen_param_count": total_count - trainable_count,
        "trainable_fraction": float(trainable_count / total_count) if total_count else 0.0,
        "parameter_mask_valid": True,
    }


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    metadata: dict[str, Any],
    x_stats: dict[str, Any] | None = None,
    y_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": make_jsonable(metadata),
        "x_stats": x_stats,
        "y_stats": y_stats,
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "metadata": make_jsonable(metadata),
    }


def load_pretrain_checkpoint(
    checkpoint_info: dict[str, Any] | str | None,
    *,
    config,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_info is None:
        return None
    checkpoint_path = Path(checkpoint_info["path"] if isinstance(checkpoint_info, dict) else checkpoint_info)
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    input_dim = metadata.get("input_dim")
    output_dim = metadata.get("output_dim")
    if input_dim is None or output_dim is None:
        return None
    model = build_cfc_regressor(
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        hidden_units=int(config.hidden_units),
        model_family=str(getattr(config, "model_family", "autoncp")),
        cfc_dropout=float(getattr(config, "cfc_dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return {
        "config": config,
        "model": model,
        "history": [
            {
                **metadata.get("best_epoch", {"epoch": 0.0, "val_mae": float("inf")}),
                "loaded_from_checkpoint": 1.0,
                "checkpoint_path": str(checkpoint_path),
            }
        ],
        "evaluations": {},
        "prediction_figure": None,
        "x_stats": payload["x_stats"],
        "y_stats": payload["y_stats"],
        "sequence_splits": {},
        "loaded_checkpoint": str(checkpoint_path),
    }


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


def limit_split_sequences_per_action(
    split: SequenceSplit,
    *,
    max_sequences_per_action: int | None,
    random_seed: int,
) -> SequenceSplit:
    """Subsample a split to simulate a bounded target-subject calibration budget."""
    if max_sequences_per_action is None:
        return split
    if max_sequences_per_action <= 0:
        raise ValueError("--support-sequences-per-action must be positive")
    if split.action_labels is None:
        raise ValueError("--support-sequences-per-action requires action labels")

    labels = np.asarray(split.action_labels)
    rng = np.random.default_rng(random_seed)
    selected_indices: list[np.ndarray] = []
    for action in np.unique(labels):
        action_indices = np.flatnonzero(labels == action)
        if action_indices.size > max_sequences_per_action:
            action_indices = np.sort(
                rng.choice(action_indices, size=max_sequences_per_action, replace=False)
            )
        selected_indices.append(action_indices)

    if not selected_indices:
        raise ValueError("support split is empty after calibration-budget limiting")
    indices = np.sort(np.concatenate(selected_indices))
    repetition_labels = (
        None if split.repetition_labels is None else split.repetition_labels[indices]
    )
    return SequenceSplit(
        x=split.x[indices],
        y=split.y[indices],
        time_s=split.time_s[indices],
        alignment_indices=split.alignment_indices[indices],
        recording_ids=split.recording_ids[indices],
        feature_names=split.feature_names,
        target_names=split.target_names,
        action_labels=split.action_labels[indices],
        repetition_labels=repetition_labels,
    )


def _copy_split_by_indices(split: SequenceSplit, indices: np.ndarray) -> SequenceSplit:
    repetition_labels = None if split.repetition_labels is None else split.repetition_labels[indices]
    action_labels = None if split.action_labels is None else split.action_labels[indices]
    return SequenceSplit(
        x=split.x[indices],
        y=split.y[indices],
        time_s=split.time_s[indices],
        alignment_indices=split.alignment_indices[indices],
        recording_ids=split.recording_ids[indices],
        feature_names=split.feature_names,
        target_names=split.target_names,
        action_labels=action_labels,
        repetition_labels=repetition_labels,
    )


def limit_split_by_support_seconds(
    split: SequenceSplit,
    *,
    support_seconds_total: float | None,
    stride_ms: int,
    seq_stride: int,
) -> tuple[SequenceSplit, dict[str, Any]]:
    if support_seconds_total is None:
        return split, {
            "support_selection_mode": "all_available",
            "requested_total_seconds": None,
            "actual_total_seconds": None,
        }
    if support_seconds_total <= 0:
        raise ValueError("--support-seconds-total must be positive")
    if split.action_labels is None:
        raise ValueError("--support-seconds-total requires action labels")

    seconds_per_sequence = float(stride_ms * seq_stride / 1000.0)
    if seconds_per_sequence <= 0:
        raise ValueError("stride_ms * seq_stride must be positive for seconds-based support budgeting")
    max_sequences = int(np.floor(support_seconds_total / seconds_per_sequence))
    labels = np.asarray(split.action_labels)
    actions = [int(value) for value in np.unique(labels).tolist()]
    if max_sequences < len(actions):
        raise ValueError(
            "--support-seconds-total is too small to provide nonzero support for every action"
        )

    available_by_action = {
        action: int(np.sum(labels == action))
        for action in actions
    }
    allocation = {action: 0 for action in actions}
    remaining = max_sequences
    pending = set(actions)
    while pending and remaining > 0:
        share = max(1, remaining // len(pending))
        progressed = False
        for action in list(pending):
            capacity = available_by_action[action] - allocation[action]
            if capacity <= 0:
                pending.remove(action)
                continue
            take = min(share, capacity, remaining)
            allocation[action] += take
            remaining -= take
            progressed = progressed or take > 0
            if allocation[action] >= available_by_action[action]:
                pending.remove(action)
            if remaining <= 0:
                break
        if not progressed:
            break

    if any(count <= 0 for count in allocation.values()):
        raise ValueError("--support-seconds-total produced zero support for at least one action")

    selected_indices: list[np.ndarray] = []
    for action in actions:
        action_indices = np.flatnonzero(labels == action)
        selected_indices.append(action_indices[: allocation[action]])
    indices = np.sort(np.concatenate(selected_indices))
    limited = _copy_split_by_indices(split, indices)
    per_action_seconds = {
        str(action): float(allocation[action] * seconds_per_sequence)
        for action in actions
    }
    summary = {
        "support_selection_mode": "deterministic_earliest_per_action",
        "requested_total_seconds": float(support_seconds_total),
        "actual_total_seconds": float(indices.size * seconds_per_sequence),
        "seconds_per_sequence": seconds_per_sequence,
        "sequence_stride_ms": int(stride_ms),
        "seq_stride": int(seq_stride),
        "raw_time_definition": "selected_sequences * stride_ms * seq_stride",
        "per_action_seconds": per_action_seconds,
        "per_action_sequence_counts": {
            str(action): int(allocation[action])
            for action in actions
        },
        "available_per_action_sequence_counts": {
            str(action): int(available_by_action[action])
            for action in actions
        },
    }
    if summary["actual_total_seconds"] > support_seconds_total + 1e-9:
        raise ValueError("seconds-based support budget exceeded requested total")
    return limited, summary


def validate_support_action_coverage(splits: dict[str, SequenceSplit]) -> None:
    support = splits["train"]
    if support.action_labels is None:
        return
    support_actions = set(int(value) for value in np.unique(support.action_labels).tolist())
    for split_name in ("val", "test"):
        split = splits.get(split_name)
        if split is None or split.action_labels is None:
            continue
        required_actions = set(int(value) for value in np.unique(split.action_labels).tolist())
        missing = sorted(required_actions - support_actions)
        if missing:
            raise ValueError(f"S1 support split has zero coverage for {split_name} actions: {missing}")


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
    calibration_budget: dict[str, Any]
    if args.support_sequences_per_action is not None:
        train_split = limit_split_sequences_per_action(
            s1_splits_raw["train"],
            max_sequences_per_action=args.support_sequences_per_action,
            random_seed=pretrain_config.random_seed,
        )
        calibration_budget = {
            "support_selection_mode": "random_sequences_per_action",
            "support_sequences_per_action": args.support_sequences_per_action,
            "support_seconds_total": args.support_seconds_total,
            "seconds_budget_overridden": args.support_seconds_total is not None,
        }
    else:
        train_split, calibration_budget = limit_split_by_support_seconds(
            s1_splits_raw["train"],
            support_seconds_total=args.support_seconds_total,
            stride_ms=pretrain_config.stride_ms,
            seq_stride=pretrain_config.seq_stride,
        )
    s1_splits_raw = {**s1_splits_raw, "train": train_split}
    validate_support_action_coverage(s1_splits_raw)
    action_summary = summarize_split_actions(s1_splits_raw)
    action_summary.setdefault("train", {})["calibration_budget"] = calibration_budget
    return s1_splits_raw, action_summary


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
    parameter_audit = apply_adaptation_mode(adapted_model, args.adaptation_mode)
    support_loader = make_train_loader(support_split, pretrain_config)
    trainable_parameters = [
        parameter for parameter in adapted_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(f"adaptation_mode {args.adaptation_mode} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
        "parameter_audit": parameter_audit,
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
    adaptation_modes = parse_adaptation_modes(args.candidate_adaptation_modes)
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
        print(f"  adaptation   : {list(adaptation_modes)}")
        pretrain_results = train_cfc_regressor(pretrain_config)
        plt.close(pretrain_results["prediction_figure"])
        pretrain_checkpoint = None
        if args.save_checkpoints:
            pretrain_checkpoint = save_model_checkpoint(
                output_dir / "checkpoints" / f"pretrain_h{hidden_units}.pt",
                pretrain_results["model"],
                metadata={
                    "stage": "pretrain",
                    "hidden_units": hidden_units,
                    "input_dim": int(pretrain_results["sequence_splits"]["train"].x.shape[-1]),
                    "output_dim": int(pretrain_results["sequence_splits"]["train"].y.shape[-1]),
                    "normalization": {
                        "feature": pretrain_config.feature_normalization,
                        "target": pretrain_config.target_normalization,
                    },
                    "mapping": doa5_mapping_metadata(),
                    "best_epoch": make_jsonable(summarize_best_epoch(pretrain_results["history"])),
                },
                x_stats=pretrain_results["x_stats"],
                y_stats=pretrain_results["y_stats"],
            )
        device = resolve_device(pretrain_config.device)
        s1_splits_raw, s1_action_coverage = load_s1_raw_splits(args, context)
        normalized_splits = normalize_s1_splits(s1_splits_raw, pretrain_results)

        for loss_profile in loss_profiles:
            for adaptation_mode in adaptation_modes:
                args.adaptation_mode = adaptation_mode
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
                current_candidate_id = candidate_id(hidden_units, loss_profile, adaptation_mode)
                adaptation_checkpoint = None
                if args.save_checkpoints:
                    adaptation_checkpoint = save_model_checkpoint(
                        output_dir / "checkpoints" / f"{current_candidate_id}_adapted.pt",
                        fine_tune_result["model"],
                        metadata={
                            "stage": "s1_adaptation",
                            "candidate_id": current_candidate_id,
                            "hidden_units": hidden_units,
                            "loss_profile": loss_profile,
                            "adaptation_mode": adaptation_mode,
                            "best_epoch": fine_tune_result["best_epoch"],
                            "parameter_audit": fine_tune_result["parameter_audit"],
                            "metrics": compact_evaluation(val_evaluation),
                            "normalization": {
                                "feature": pretrain_config.feature_normalization,
                                "target": pretrain_config.target_normalization,
                            },
                            "mapping": doa5_mapping_metadata(),
                        },
                        x_stats=pretrain_results["x_stats"],
                        y_stats=pretrain_results["y_stats"],
                    )
                candidate = {
                    "candidate_id": current_candidate_id,
                    "target_names": list(val_evaluation["target_names"]),
                    "candidate_config": {
                        "hidden_units": hidden_units,
                        "loss_profile": loss_profile,
                        "adaptation_mode": adaptation_mode,
                        "loss_weights": list(MILD_FAILED_DOA_WEIGHTS) if loss_profile == "mild_failed_doa" else None,
                        "fine_tune_epochs": args.fine_tune_epochs,
                        "fine_tune_patience": args.fine_tune_patience,
                        "fine_tune_learning_rate": args.fine_tune_learning_rate,
                        "support_sequences_per_action": args.support_sequences_per_action,
                        "support_seconds_total": args.support_seconds_total,
                    },
                    "pretrain_best_epoch": make_jsonable(summarize_best_epoch(pretrain_results["history"])),
                    "fine_tune_history": make_jsonable(fine_tune_result["history"]),
                    "selected_checkpoint_epoch": fine_tune_result["best_epoch"],
                    "parameter_audit": make_jsonable(fine_tune_result["parameter_audit"]),
                    "pretrain_checkpoint": pretrain_checkpoint,
                    "adaptation_checkpoint": adaptation_checkpoint,
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
            "adaptation_modes": list(adaptation_modes),
        },
        "calibration_budget": {
            "support_sequences_per_action": args.support_sequences_per_action,
            "support_seconds_total": args.support_seconds_total,
            "sequence_stride_ms": first_context["pretrain_config"].stride_ms,
            "seq_stride": first_context["pretrain_config"].seq_stride,
        },
        "delta_rules": {
            "hidden_units_128": "worst_doa_val_r2_delta >= 0.02 or one additional DoA val pass without passed-DoA regression",
            "weighted_loss": "failed_doa_min_val_r2_delta >= 0.015 and no passed DoA loses more than 0.02 R2",
        },
        "selection_priority": [
            "validation_worst_doa_r2",
            "val_pass_count",
            "val_mean_r2",
            "val_mean_mae",
            "lower_trainable_fraction",
            "minimum_delta_rules",
            "lower_hidden_units",
            "baseline_loss",
        ],
        "selection_metric": "validation_worst_doa_r2",
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
    args.adaptation_mode = str(selected_config.get("adaptation_mode", DEFAULT_ADAPTATION_MODE))

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
    print(
        "  selected      : "
        f"hidden_units={args.hidden_units}, loss_profile={args.loss_profile}, "
        f"adaptation_mode={args.adaptation_mode}"
    )
    print(f"  train subjects: {list(pretrain_train_subjects)}")
    print(f"  pretrain test : {pretrain_test_subject or protocol['val_subject']}")
    print(f"  val subject   : {protocol['val_subject']}")
    print(f"  adapt subject : {protocol['adapt_subject']}")
    print(f"  S1 split mode : {args.s1_split_mode}")

    device = resolve_device(pretrain_config.device)
    pretrain_results = None
    selected_candidate = selection_summary.get("selected_candidate", {})
    if not args.force_pretrain:
        pretrain_results = load_pretrain_checkpoint(
            selected_candidate.get("pretrain_checkpoint"),
            config=pretrain_config,
            device=device,
        )
    if pretrain_results is None:
        pretrain_results = train_cfc_regressor(pretrain_config)
        if pretrain_results.get("prediction_figure") is not None:
            plt.close(pretrain_results["prediction_figure"])

    pretrain_checkpoint = selected_candidate.get("pretrain_checkpoint") if pretrain_results.get("loaded_checkpoint") else None
    if args.save_checkpoints and pretrain_checkpoint is None:
        pretrain_checkpoint = save_model_checkpoint(
            output_dir / "checkpoints" / f"pretrain_h{args.hidden_units}.pt",
            pretrain_results["model"],
            metadata={
                "stage": "pretrain",
                "hidden_units": args.hidden_units,
                "input_dim": int(pretrain_results["sequence_splits"]["train"].x.shape[-1]),
                "output_dim": int(pretrain_results["sequence_splits"]["train"].y.shape[-1]),
                "normalization": {
                    "feature": pretrain_config.feature_normalization,
                    "target": pretrain_config.target_normalization,
                },
                "mapping": doa5_mapping_metadata(),
                "best_epoch": make_jsonable(summarize_best_epoch(pretrain_results["history"])),
            },
            x_stats=pretrain_results["x_stats"],
            y_stats=pretrain_results["y_stats"],
        )

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
    adaptation_checkpoint = None
    if args.save_checkpoints:
        adaptation_checkpoint = save_model_checkpoint(
            output_dir
            / "checkpoints"
            / f"{selection_summary['selected_candidate_id']}_final_adapted.pt",
            adapted_model,
            metadata={
                "stage": "s1_final_adaptation",
                "candidate_id": selection_summary["selected_candidate_id"],
                "hidden_units": args.hidden_units,
                "loss_profile": args.loss_profile,
                "adaptation_mode": args.adaptation_mode,
                "best_epoch": fine_tune_result["best_epoch"],
                "parameter_audit": fine_tune_result["parameter_audit"],
                "normalization": {
                    "feature": pretrain_config.feature_normalization,
                    "target": pretrain_config.target_normalization,
                },
                "mapping": doa5_mapping_metadata(),
            },
            x_stats=pretrain_results["x_stats"],
            y_stats=pretrain_results["y_stats"],
        )

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
        "adaptation_mode": args.adaptation_mode,
        "parameter_audit": make_jsonable(fine_tune_result["parameter_audit"]),
        "calibration_budget": {
            "support_sequences_per_action": args.support_sequences_per_action,
            "support_seconds_total": args.support_seconds_total,
            "sequence_stride_ms": pretrain_config.stride_ms,
            "seq_stride": pretrain_config.seq_stride,
        },
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
            "pretrain_checkpoint": pretrain_checkpoint,
            "adaptation_checkpoint": adaptation_checkpoint,
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
