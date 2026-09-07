from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATAFLOW_DIR = SCRIPT_DIR.parent / "dataflow"
if str(DATAFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_DIR))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from datapreprocess import load_data, preprocess_emg
from doa_mapping import (
    DOA5_MAPPING_NAME,
    DOA5_NAMES,
    GLOVE_COLUMN_INDICES,
    GLOVE_COLUMN_NAMES,
    GLOVE_COLUMNS_MAPPING_NAME,
    JOINT_ANGLES10_CHANNELS_1BASED,
    JOINT_ANGLES10_INDICES,
    JOINT_ANGLES10_MAPPING_NAME,
    JOINT_ANGLES10_MAPPING_VERSION,
    JOINT_ANGLES10_NAMES,
    apply_linear_doa_mapping,
    glove_to_doa,
)
from feature_extraction import (
    DEFAULT_MU_LAW_MU,
    _compute_rest_thresholds,
    extract_emg_features,
)
from SwRectify import sliding_window
from torch import nn
from train import (
    DEFAULT_DB2_EMG_CHANNELS,
    DomainDiscriminator,
    GpuResidentBatches,
    GpuResidentStatefulBatches,
    RecordingFeatures,
    SegmentIndex,
    SequenceSplit,
    WeightedMSELoss,
    build_best_cfc_config,
    build_cfc_regressor,
    build_graphed_cfc,
    build_sequence_split,
    compute_grouped_regression_metrics,
    compute_regression_metrics,
    discover_target_files,
    evaluate_chain,
    evaluate_split,
    fit_feature_normalizer,
    fit_target_normalizer,
    list_db2_files,
    make_jsonable,
    make_train_loader,
    normalize_sequence_inputs,
    resolve_device,
    save_summary,
    set_random_seed,
    train_one_epoch,
    train_one_epoch_stateful,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "log" / "db2_joint_angles10_cfc_finetune"

PAPER_FEATURE_ORDER = ("rms",)
ALL_EXERCISES = ("E1", "E2")  # E3 has no glove data for any subject

_RECORDING_CACHE_VARIABLES = ("emg", "glove", "restimulus", "rerepetition")


def _recording_cache_path(file_path: Path, config, cache_dir: Path) -> Path:
    """Return a content-addressed cache path for pre-normalization recording features."""
    source_stat = file_path.stat()
    target_contract = resolve_target_contract(
        config.target_mapping,
        tuple(config.target_columns),
    )
    code_hash = hashlib.sha256()
    for path in (
        Path(__file__),
        DATAFLOW_DIR / "datapreprocess.py",
        DATAFLOW_DIR / "SwRectify.py",
        DATAFLOW_DIR / "feature_extraction.py",
        DATAFLOW_DIR / "doa_mapping.py",
    ):
        code_hash.update(path.read_bytes())
    key = {
        "source": str(file_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "emg_channels": list(config.emg_channels),
        "window_ms": config.window_ms,
        "stride_ms": config.stride_ms,
        "target_offset_samples": config.target_offset_samples,
        "feature_order": list(config.feature_order),
        "target_contract": target_contract,
        "code_hash": code_hash.hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{file_path.stem}-{digest}.npz"


def _load_recording_cache(cache_path: Path) -> tuple[RecordingFeatures, np.ndarray] | None:
    """Load one validated feature cache, returning None when it cannot be used."""
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            recording = RecordingFeatures(
                recording_id=str(cached["recording_id"].item()),
                x_windows=np.asarray(cached["x_windows"], dtype=np.float32),
                y_windows=np.asarray(cached["y_windows"], dtype=np.float32),
                target_alignment_indices=np.asarray(cached["alignment_indices"], dtype=np.int32),
                feature_names=json.loads(str(cached["feature_names"].item())),
                target_names=json.loads(str(cached["target_names"].item())),
                fs=float(cached["fs"].item()),
                action_labels=np.asarray(cached["action_labels"], dtype=np.int16),
                repetition_labels=np.asarray(cached["repetition_labels"], dtype=np.int16),
                source_recording_id=str(cached["source_recording_id"].item()),
            )
            eligible = np.asarray(cached["label_isolated"], dtype=bool)
        if recording.x_windows.ndim != 2 or recording.y_windows.ndim != 2:
            raise ValueError("cached features and targets must be rank-2 arrays")
        n_windows = recording.x_windows.shape[0]
        if recording.y_windows.shape[0] != n_windows or any(
            values.shape != (n_windows,)
            for values in (
                recording.target_alignment_indices, recording.action_labels,
                recording.repetition_labels, eligible,
            )
        ):
            raise ValueError("cached per-window arrays have inconsistent lengths")
        if (
            len(recording.feature_names) != recording.x_windows.shape[1]
            or len(recording.target_names) != recording.y_windows.shape[1]
        ):
            raise ValueError("cached feature or target names do not match array widths")
        return recording, eligible
    except Exception as exc:
        warnings.warn(f"Ignoring unusable recording feature cache {cache_path}: {exc}", UserWarning)
        return None


def _save_recording_cache(cache_path: Path, recording: RecordingFeatures, eligible: np.ndarray) -> None:
    """Atomically persist arrays needed to reconstruct the full and selected streams."""
    temporary_path: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_path.parent, prefix=f"{cache_path.stem}-", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                recording_id=np.asarray(recording.recording_id),
                source_recording_id=np.asarray(recording.source_recording_id),
                x_windows=recording.x_windows,
                y_windows=recording.y_windows,
                alignment_indices=recording.target_alignment_indices,
                feature_names=np.asarray(json.dumps(recording.feature_names)),
                target_names=np.asarray(json.dumps(recording.target_names)),
                fs=np.asarray(recording.fs),
                action_labels=recording.action_labels,
                repetition_labels=recording.repetition_labels,
                label_isolated=np.asarray(eligible, dtype=bool),
            )
        os.replace(temporary_path, cache_path)
    except OSError as exc:
        warnings.warn(f"Could not write recording feature cache {cache_path}: {exc}", UserWarning)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _discover_subjects(db2_dir: Path) -> list[str]:
    """Return sorted list of all subject IDs found under ``db2_dir``."""
    subjects: list[str] = []
    for entry in sorted(db2_dir.iterdir()):
        if entry.is_dir() and entry.name.upper().startswith("DB2_S"):
            sid = entry.name.replace("DB2_", "", 1).replace("db2_", "", 1).upper()
            subjects.append(sid)
    return sorted(subjects, key=lambda s: (int(s[1:]) if s[1:].isdigit() else 0, s))


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("integer list cannot be empty")
    return items


def resolve_target_contract(
    mapping: str | None,
    target_columns: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Resolve output width, names, and source-column semantics explicitly."""
    if mapping == JOINT_ANGLES10_MAPPING_NAME:
        return {
            "mapping_name": JOINT_ANGLES10_MAPPING_NAME,
            "mapping_version": JOINT_ANGLES10_MAPPING_VERSION,
            "output_dim": len(JOINT_ANGLES10_INDICES),
            "target_names": list(JOINT_ANGLES10_NAMES),
            "source_channels_one_based": list(JOINT_ANGLES10_CHANNELS_1BASED),
            "source_column_indices_zero_based": list(JOINT_ANGLES10_INDICES),
        }
    if mapping == GLOVE_COLUMNS_MAPPING_NAME:
        return {
            "mapping_name": GLOVE_COLUMNS_MAPPING_NAME,
            "mapping_version": "legacy_nonzero_doa_columns_v1",
            "output_dim": len(GLOVE_COLUMN_INDICES),
            "target_names": list(GLOVE_COLUMN_NAMES),
            "source_column_indices_zero_based": list(GLOVE_COLUMN_INDICES),
        }
    if mapping == DOA5_MAPPING_NAME:
        return {
            "mapping_name": DOA5_MAPPING_NAME,
            "mapping_version": "legacy_doa5",
            "output_dim": len(DOA5_NAMES),
            "target_names": list(DOA5_NAMES),
            "source_column_indices_zero_based": None,
        }
    if mapping is None:
        columns = tuple(int(column) for column in target_columns)
        if not columns:
            raise ValueError("target_columns cannot be empty when target_mapping is None")
        return {
            "mapping_name": None,
            "mapping_version": None,
            "output_dim": len(columns),
            "target_names": [f"glove_{column + 1}" for column in columns],
            "source_column_indices_zero_based": list(columns),
        }
    raise ValueError(f"unsupported target mapping: {mapping}")


def infer_checkpoint_output_dim(state_dict: dict[str, Any]) -> int:
    """Infer the readout width from checkpoint weights instead of mapping names."""
    head_weight = state_dict.get("head.weight")
    if head_weight is None or not hasattr(head_weight, "shape") or len(head_weight.shape) != 2:
        raise ValueError("checkpoint is missing a two-dimensional head.weight tensor")
    output_dim = int(head_weight.shape[0])
    if output_dim <= 0:
        raise ValueError(f"checkpoint head has invalid output dimension {output_dim}")
    return output_dim


def validate_checkpoint_target_contract(
    checkpoint_contract: dict[str, Any] | None,
    current_contract: dict[str, Any],
) -> None:
    """Reject semantic target mismatches while accepting legacy metadata gaps."""
    if checkpoint_contract is None:
        return
    contract_fields = (
        "mapping_name",
        "mapping_version",
        "output_dim",
        "target_names",
        "source_column_indices_zero_based",
    )
    mismatches = [
        field
        for field in contract_fields
        if checkpoint_contract.get(field) != current_contract.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint target contract disagrees with the current target contract "
            f"for fields: {mismatches}. Use a matching checkpoint or re-pretrain."
        )


def parse_emg_channels(value: str) -> tuple[int, ...]:
    """Parse one-based physical DB2 EMG channels from a strict CSV list."""
    raw_items = value.split(",")
    if not value.strip() or any(not item.strip() for item in raw_items):
        raise ValueError("--emg-channels must be a non-empty comma-separated list")
    try:
        channels = tuple(int(item.strip()) for item in raw_items)
    except ValueError as exc:
        raise ValueError("--emg-channels must contain integers only") from exc
    if any(channel <= 0 for channel in channels):
        raise ValueError("--emg-channels uses one-based positive channel numbers")
    if len(set(channels)) != len(channels):
        raise ValueError("--emg-channels cannot contain duplicate channels")
    maximum_channel = max(DEFAULT_DB2_EMG_CHANNELS)
    if any(channel > maximum_channel for channel in channels):
        raise ValueError(
            f"--emg-channels must be within 1..{maximum_channel} for Ninapro DB2"
        )
    return channels


def select_emg_channels(
    emg: np.ndarray,
    channels: tuple[int, ...],
    *,
    recording_id: str,
) -> np.ndarray:
    """Select one-based physical EMG channels before signal preprocessing."""
    if emg.ndim != 2:
        raise ValueError(f"{recording_id}: EMG must be a 2-D samples-by-channels array")
    if not channels:
        raise ValueError(f"{recording_id}: EMG channel selection cannot be empty")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{recording_id}: EMG channel selection contains duplicates")
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"{recording_id}: EMG channels must be one-based positive integers")
    unavailable = [channel for channel in channels if channel > emg.shape[1]]
    if unavailable:
        raise ValueError(
            f"{recording_id}: requested EMG channel {unavailable[0]}, "
            f"but recording contains only {emg.shape[1]} channels"
        )
    zero_based_indices = [channel - 1 for channel in channels]
    return emg[:, zero_based_indices]


def resolve_resume_input_dim(
    checkpoint_config: dict[str, Any],
    current_config,
    *,
    actual_input_dim: int,
) -> int:
    """Validate resume feature/channel semantics and derive model input width."""
    checkpoint_features = tuple(
        checkpoint_config.get("feature_order", current_config.feature_order)
    )
    current_features = tuple(current_config.feature_order)
    if checkpoint_features != current_features:
        raise ValueError(
            f"Checkpoint was trained with feature_order={checkpoint_features}, "
            f"but current config uses feature_order={current_features}. "
            "Use matching --feature-order or re-pretrain."
        )

    if "emg_channels" in checkpoint_config:
        checkpoint_channels = tuple(int(value) for value in checkpoint_config["emg_channels"])
    else:
        checkpoint_channels = DEFAULT_DB2_EMG_CHANNELS
        warnings.warn(
            "Legacy checkpoint has no emg_channels metadata; treating it as the historical "
            f"all-channel DB2 selection {DEFAULT_DB2_EMG_CHANNELS}.",
            UserWarning,
            stacklevel=2,
        )
    current_channels = tuple(current_config.emg_channels)
    if checkpoint_channels != current_channels:
        raise ValueError(
            f"Checkpoint was trained with EMG channels {checkpoint_channels}, "
            f"but current config uses EMG channels {current_channels}. "
            "Use matching --emg-channels or re-pretrain."
        )

    checkpoint_input_dim = len(checkpoint_channels) * len(checkpoint_features)
    if checkpoint_input_dim != actual_input_dim:
        raise ValueError(
            f"Checkpoint metadata implies input_dim={checkpoint_input_dim}, "
            f"but the current data pipeline produced input_dim={actual_input_dim}."
        )
    return checkpoint_input_dim


def validate_resume_architecture(
    checkpoint_config: dict[str, Any],
    current_config,
) -> None:
    """Reject resume when immutable model structure differs from the CLI config."""
    fields = ("hidden_units", "model_family", "cfc_dropout")
    missing = [field for field in fields if field not in checkpoint_config]
    if missing:
        raise ValueError(
            "Checkpoint lacks immutable architecture metadata "
            f"{missing}; cannot safely resume. Use a checkpoint with matching CLI metadata."
        )
    mismatches = {
        field: (checkpoint_config[field], getattr(current_config, field))
        for field in fields
        if checkpoint_config[field] != getattr(current_config, field)
    }
    if mismatches:
        detail = ", ".join(
            f"{field}: checkpoint={saved!r}, current={current!r}"
            for field, (saved, current) in mismatches.items()
        )
        raise ValueError(
            "Checkpoint architecture differs from the current CLI configuration "
            f"({detail}). Pass matching --hidden-units/--model-family/--cfc-dropout "
            "or re-pretrain."
        )


def parse_csv_subjects(value: str) -> tuple[str, ...]:
    subjects = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not subjects:
        raise ValueError("subject list cannot be empty")
    return subjects


def save_feature_normalization_stats(
    output_dir: Path,
    stats: dict[str, Any],
    *,
    emg_channels: tuple[int, ...],
    feature_order: tuple[str, ...],
    target_stats: dict[str, Any] | None = None,
    target_names: list[str] | tuple[str, ...] = (),
) -> Path:
    """Save training-fitted feature and optional target stats for C export."""
    if stats.get("method") != "mu_law":
        raise ValueError("hardware export requires mu-law feature normalization")
    center = np.asarray(stats["center"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    expected_input_dim = len(emg_channels) * len(feature_order)
    if center.size != expected_input_dim or scale.size != expected_input_dim:
        raise ValueError(
            f"normalization width must equal channels * features = {expected_input_dim}; "
            f"got center={center.size}, scale={scale.size}"
        )
    payload: dict[str, Any] = {
        "center": center,
        "scale": scale,
        "mu": np.float32(stats["mu"]),
        "emg_channels": np.asarray(emg_channels, dtype=np.int16),
        "feature_order": np.asarray(feature_order, dtype=np.str_),
    }
    if target_stats is not None:
        if target_stats.get("method") != "mu_law":
            raise ValueError("hardware export requires mu-law target normalization")
        target_center = np.asarray(target_stats["center"], dtype=np.float32)
        target_scale = np.asarray(target_stats["scale"], dtype=np.float32)
        if target_center.ndim != 1 or target_scale.shape != target_center.shape:
            raise ValueError("target normalization center/scale must be matching 1-D arrays")
        if target_names and len(target_names) != target_center.size:
            raise ValueError("target_names must match target normalization width")
        payload.update(
            {
                "target_center": target_center,
                "target_scale": target_scale,
                "target_mu": np.float32(target_stats["mu"]),
                "target_names": np.asarray(target_names, dtype=np.str_),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_normalization.npz"
    np.savez(
        path,
        **payload,
    )
    return path


RESUME_INTEGRITY_SCHEMA_VERSION = 2


def _sha256_array(value: np.ndarray) -> str:
    """Return a dtype- and shape-sensitive digest for an array."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    """Return a stable digest for JSON-compatible protocol metadata."""
    encoded = json.dumps(
        make_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_split_identity(split: SequenceSplit) -> str:
    """Fingerprint all rows and labels that define a source split."""
    fields: dict[str, np.ndarray | None] = {
        "x": split.x,
        "y": split.y,
        "time_s": split.time_s,
        "alignment_indices": split.alignment_indices,
        "recording_ids": split.recording_ids,
        "action_labels": split.action_labels,
        "repetition_labels": split.repetition_labels,
        "stream_start_flags": split.stream_start_flags,
        "source_recording_ids": split.source_recording_ids,
    }
    return _sha256_json(
        {
            "feature_names": list(split.feature_names),
            "target_names": list(split.target_names),
            "expected_alignment_step": split.expected_alignment_step,
            "arrays": {
                name: None if value is None else _sha256_array(value)
                for name, value in fields.items()
            },
        }
    )


def stream_input_provenance_identity(split: SequenceSplit) -> str:
    """Fingerprint full-stream inputs and provenance, excluding target values."""
    fields: dict[str, np.ndarray | None] = {
        "x": split.x,
        "time_s": split.time_s,
        "alignment_indices": split.alignment_indices,
        "recording_ids": split.recording_ids,
        "action_labels": split.action_labels,
        "repetition_labels": split.repetition_labels,
        "stream_start_flags": split.stream_start_flags,
        "source_recording_ids": split.source_recording_ids,
    }
    return _sha256_json(
        {
            "feature_names": list(split.feature_names),
            "expected_alignment_step": split.expected_alignment_step,
            "arrays": {
                name: None if value is None else _sha256_array(value)
                for name, value in fields.items()
            },
        }
    )


def normalization_stats_identity(stats: dict[str, Any]) -> str:
    """Fingerprint fitted normalization values, including method parameters."""
    normalized: dict[str, Any] = {}
    for name, value in stats.items():
        normalized[name] = _sha256_array(value) if isinstance(value, np.ndarray) else value
    return _sha256_json(normalized)


def build_resume_integrity_metadata(
    *,
    x_stats: dict[str, Any],
    y_stats: dict[str, Any],
    source_supervised: SequenceSplit,
    source_stream: SequenceSplit,
    source_stream_score_mask: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Bind checkpoint normalizers to the full supervised source protocol."""
    return {
        "schema_version": RESUME_INTEGRITY_SCHEMA_VERSION,
        "source_split_identities": {
            "supervised": source_split_identity(source_supervised),
            "stream": source_split_identity(source_stream),
            "stream_score_mask": _sha256_array(source_stream_score_mask),
        },
        "protocol": make_jsonable(protocol),
        "protocol_identity": _sha256_json(protocol),
        "normalization_stats_identities": {
            "feature": normalization_stats_identity(x_stats),
            "target": normalization_stats_identity(y_stats),
        },
    }


def project_pretrain_history(history: Any) -> list[dict[str, Any]]:
    """Drop legacy validation/selection fields from a resumed checkpoint history."""
    if not isinstance(history, (list, tuple)):
        return []
    return [
        {"epoch": entry["epoch"], "train_loss": entry["train_loss"]}
        for entry in history
        if isinstance(entry, dict) and "epoch" in entry and "train_loss" in entry
    ]


def project_fine_tune_history(history: Any) -> list[dict[str, Any]]:
    """Persist optimization losses only, excluding evaluation or selection fields."""
    if not isinstance(history, (list, tuple)):
        return []
    allowed = ("epoch", "train_loss", "L_DD", "L_mapping", "L_subject")
    return [
        {field: entry[field] for field in allowed if field in entry}
        for entry in history
        if isinstance(entry, dict) and "epoch" in entry
    ]


def build_adaptation_provenance(
    *,
    target_support_split: SequenceSplit,
    target_stream_split: SequenceSplit,
    support_score_mask: np.ndarray,
    mode: str,
    epochs: int,
    learning_rate: float,
    atl_subject_weight: float | None,
) -> dict[str, Any]:
    """Bind adaptation artifacts to supervised support inputs, never query labels."""
    return {
        "target_support_split_identity": source_split_identity(target_support_split),
        "target_stream_identity": stream_input_provenance_identity(target_stream_split),
        "support_score_mask": _sha256_array(support_score_mask),
        "mode": mode,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "atl_subject_weight": None if atl_subject_weight is None else float(atl_subject_weight),
    }


PRETRAIN_PROVENANCE_SCHEMA_VERSION = 1


def build_pretrain_provenance(config, args: argparse.Namespace) -> dict[str, Any]:
    """Record immutable conditions that produced a new pretraining checkpoint."""
    requested_augment = float(args.augment_prob)
    stateful = bool(config.stateful)
    return {
        "schema_version": PRETRAIN_PROVENANCE_SCHEMA_VERSION,
        "status": "originating_pretrain",
        "originating_config": make_jsonable(asdict(config)),
        "augment_prob_requested": requested_augment,
        "augment_prob_effective": 0.0 if stateful else requested_augment,
        "gpu_resident_requested": bool(args.gpu_resident),
        "gpu_resident_effective": bool(args.gpu_resident or stateful),
        "stateful_effective": stateful,
    }


def build_legacy_pretrain_provenance(ckpt_config: dict[str, Any]) -> dict[str, Any]:
    """Make a legacy checkpoint's unknown originating conditions explicit."""
    warnings.warn(
        "Checkpoint lacks pretrain_provenance; preserving its saved config but "
        "recording augmentation and execution policy as legacy_unknown.",
        UserWarning,
        stacklevel=2,
    )
    return {
        "schema_version": PRETRAIN_PROVENANCE_SCHEMA_VERSION,
        "status": "legacy_unknown",
        "originating_config": make_jsonable(ckpt_config),
        "augment_prob_requested": None,
        "augment_prob_effective": None,
        "gpu_resident_requested": None,
        "gpu_resident_effective": None,
        "stateful_effective": None,
    }


def build_resume_invocation_metadata(
    config,
    args: argparse.Namespace,
    *,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    """Record this invocation separately from the originating pretraining run."""
    return {
        "status": "resumed" if resume_checkpoint is not None else "not_resumed",
        "current_config": make_jsonable(asdict(config)),
        "augment_prob_requested": float(args.augment_prob),
        "resume_checkpoint_path": None if resume_checkpoint is None else str(resume_checkpoint),
    }


def validate_resume_integrity_metadata(
    checkpoint: dict[str, Any],
    expected: dict[str, Any],
    *,
    allow_legacy_resume_without_integrity_binding: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate checkpoint/source binding and return its authoritative stats."""
    metadata = checkpoint.get("resume_integrity")
    if metadata is None:
        if not allow_legacy_resume_without_integrity_binding:
            raise ValueError(
                "Checkpoint lacks resume integrity binding metadata. Refuse to resume "
                "by default; use --allow-legacy-resume-without-integrity-binding only "
                "after independently verifying its source data and normalizers."
            )
        warnings.warn(
            "Resuming legacy checkpoint without integrity binding because "
            "--allow-legacy-resume-without-integrity-binding was supplied.",
            UserWarning,
            stacklevel=2,
        )
    else:
        if metadata.get("schema_version") != RESUME_INTEGRITY_SCHEMA_VERSION:
            raise ValueError("Checkpoint has an unsupported resume integrity schema version.")
        for field in ("source_split_identities", "protocol_identity"):
            if metadata.get(field) != expected.get(field):
                raise ValueError(
                    f"Checkpoint resume integrity mismatch for {field}; source split "
                    "identity or protocol differs from the checkpoint."
                )

    x_stats = checkpoint.get("feature_normalization_stats")
    y_stats = checkpoint.get("target_normalization_stats")
    if not isinstance(x_stats, dict) or not isinstance(y_stats, dict):
        raise TypeError(
            "Checkpoint lacks feature or target normalization statistics required for resume."
        )

    if metadata is not None:
        actual_stats = {
            "feature": normalization_stats_identity(x_stats),
            "target": normalization_stats_identity(y_stats),
        }
        if metadata.get("normalization_stats_identities") != actual_stats:
            raise ValueError(
                "Checkpoint normalization statistics do not match its integrity binding."
            )
    return x_stats, y_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DB2 protocol following Lin & He 2024: RMS-only features, ten MCP/PIP glove targets, "
            "cross-subject dense CfC pretrain, head-only FT, or full target-network ATL."
        )
    )
    parser.add_argument("--db2-dir", type=Path, default=REPO_ROOT / "src" / "data" / "DB2")
    parser.add_argument("--target-subject", type=str, default="S1")
    parser.add_argument("--subjects", type=str, default="all",
                        help="'all' to auto-discover, or comma-separated subject IDs")
    parser.add_argument("--exercise", type=str, default="all",
                        help="'all' for E1+E2 (E3 has no glove data), or comma-separated (e.g. E1,E2)")
    parser.add_argument("--actions", type=str, default="all",
                        help="'all' to use every non-rest action, or comma-separated integers")
    parser.add_argument(
        "--emg-channels",
        type=str,
        default=",".join(str(channel) for channel in DEFAULT_DB2_EMG_CHANNELS),
        help="One-based physical DB2 EMG channels, e.g. 1,2,3,4,5,6,7,8",
    )
    parser.add_argument(
        "--glove-columns",
        type=str,
        default=",".join(str(value) for value in JOINT_ANGLES10_INDICES),
        help="Legacy custom zero-based CyberGlove columns; ignored when --target-mapping is set.",
    )
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--stride-ms", type=float, default=50.0)
    parser.add_argument("--mu-law-mu", type=float, default=DEFAULT_MU_LAW_MU)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Sequences per stateless batch, or maximum independent stream lanes per stateful batch.",
    )
    parser.add_argument("--seq-len", type=int, default=8,
                        help="Sequence length in feature windows (1 = single-frame "
                             "stateless baseline matching firmware input shape)")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument(
        "--additional-pretrain-epochs", type=int, default=0,
        help="Continue source pretraining from --resume-pretrain for this many extra epochs; "
             "starts a new AdamW optimizer. Default 0 keeps load-only resume behavior.",
    )
    parser.add_argument(
        "--pretrain-target-weights", type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(),
        help="Relative MSE weights in output order, normalized to mean 1; source pretraining only. "
             "J10 order: glove channels 2,3,5,6,8,9,12,13,16,17. Omit for equal weights.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--target-offset-samples",
        type=int,
        required=True,
        help=(
            "Glove target offset relative to the final EMG sample in each window. "
            "Use 0 for synchronized angle estimation or 200 for 100 ms-ahead "
            "prediction at the fixed 2 kHz DB2 sampling rate."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cfc-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for the regression head and dense CfC backbone; "
             "AutoNCP has no backbone dropout.",
    )
    parser.add_argument("--feature-order", type=str, default=",".join(PAPER_FEATURE_ORDER),
                        help="Comma-separated feature names (must be subset of supported features)")
    parser.add_argument(
        "--target-mapping",
        type=str,
        default=JOINT_ANGLES10_MAPPING_NAME,
        choices=[JOINT_ANGLES10_MAPPING_NAME, DOA5_MAPPING_NAME, GLOVE_COLUMNS_MAPPING_NAME],
        help=(
            "Target mapping: 'joint_angles10' keeps the ten MCP/PIP channels selected "
            "in Lin & He 2024; legacy modes are 'doa5' and 'glove_columns'."
        ),
    )
    parser.add_argument(
        "--train-repetitions-per-action",
        type=int,
        default=4,
        help="Target-subject support repetitions per action; source subjects supervise all selected repetitions.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=None,
        help="Directory for reusable unnormalized DB2 window features (default: <db2-dir>/.feature_cache).",
    )
    parser.add_argument(
        "--no-feature-cache", action="store_true", default=False,
        help="Recompute DB2 window features without reading or writing the feature cache.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="dense_cfc_linear",
        choices=["dense_cfc_linear", "autoncp_cfc_linear"],
        help="Dense CfC (default) or AutoNCP CfC with a motor-feature linear head.",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=1.0,
        help="Probability of augmentation per batch. 1.0 = always, 0.0 = never",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        default=False,
        help="Wrap the deterministic CfC body in a CUDA graph during pretraining and ATL target updates "
             "(faster for small models; falls back to eager on capture failure)",
    )
    parser.add_argument(
        "--gpu-resident",
        action="store_true",
        default=False,
        help="Keep the whole pre-computed dataset in VRAM and shuffle on GPU "
             "(no per-batch CPU assembly or host-to-device transfer)",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        default=False,
        help="Causal truncated BPTT: each overlapping sequence contributes only "
             "its new final frame, with hidden state carried through verified "
             "continuous streams and reset before explicit/data discontinuities. "
             "Action and repetition label changes alone do not reset state. "
             "Requires GPU-resident data; disables augmentation for this run.",
    )
    parser.add_argument("--enable-atl", action="store_true", default=False)
    parser.add_argument("--atl-subject-weight", type=float, default=1.0,
                        help="Target regression weight w in L_subject = w * MSE(pred_t, target_y) (Eq 1.11)")
    parser.add_argument("--skip-fine-tune", action="store_true", default=False,
                        help="Pretrain only, skip fine-tuning (saves pretrain checkpoint)")
    parser.add_argument("--resume-pretrain", type=Path, default=None,
                        help="Skip pretraining; load pretrained model from this checkpoint path")
    parser.add_argument(
        "--allow-legacy-resume-without-integrity-binding",
        action="store_true",
        default=False,
        help=(
            "Allow a checkpoint created before resume integrity bindings existed. "
            "This bypasses source-split/protocol binding validation and must only be "
            "used after independently verifying the checkpoint provenance."
        ),
    )
    return parser.parse_args()


def _load_subject_recordings(
    subject: str,
    exercises: tuple[str, ...],
    config,
    *,
    actions: tuple[int, ...] | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[RecordingFeatures], list[RecordingFeatures]]:
    """Extract full recordings plus optional label-isolated training runs."""
    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    target_columns = list(config.target_columns) if config.target_columns else []
    mapping = config.target_mapping

    recordings = []
    selected_recordings = []

    for exercise in exercises:
        file_name = f"{subject}_{exercise}_A1.mat"
        if file_name not in available_files:
            continue  # skip missing exercises quietly
        file_path = available_files[file_name]
        cache_path = _recording_cache_path(file_path, config, cache_dir) if cache_dir else None
        cached = _load_recording_cache(cache_path) if cache_path and cache_path.is_file() else None
        if cached is not None:
            recording, label_isolated = cached
        else:
            data = load_data(str(file_path), variable_names=_RECORDING_CACHE_VARIABLES)
            if "glove" not in data:
                continue

            emg = np.asarray(data["emg"], dtype=np.float32)
            glove = np.asarray(data["glove"], dtype=np.float32)
            restimulus = np.asarray(data["restimulus"]).reshape(-1)
            rerepetition = np.asarray(data["rerepetition"]).reshape(-1)
            n_samples = min(emg.shape[0], glove.shape[0], restimulus.shape[0], rerepetition.shape[0])
            emg = emg[:n_samples]
            glove = glove[:n_samples]
            restimulus = restimulus[:n_samples]
            rerepetition = rerepetition[:n_samples]

            emg = select_emg_channels(emg, tuple(config.emg_channels), recording_id=file_name)
            emg_filtered = preprocess_emg(emg, fs=2000.0)
            rest_threshold = _compute_rest_thresholds(
                emg_filtered, stimulus=restimulus, fs=2000.0,
                window_ms=config.window_ms, stride_ms=config.stride_ms,
            )
            if mapping is not None:
                recording_targets = apply_linear_doa_mapping(glove, mapping=mapping)
                target_names = resolve_target_contract(mapping)["target_names"]
            else:
                recording_targets = glove[:, target_columns]
                target_names = [f"glove_{column + 1}" for column in target_columns]
            windows = sliding_window(
                emg_filtered, recording_targets, fs=2000.0, window_ms=config.window_ms,
                stride_ms=config.stride_ms, target_offset_samples=config.target_offset_samples,
                target_names=target_names, target_prefix="glove",
            )
            feature_set = extract_emg_features(
                windows, feature_order=config.feature_order,
                zc_threshold=rest_threshold, ssc_threshold=rest_threshold,
            )
            alignment_indices = np.asarray(feature_set["target_alignment_indices"], dtype=np.int32)
            aligned_actions = restimulus[alignment_indices].astype(np.int16, copy=False)
            aligned_repetitions = rerepetition[alignment_indices].astype(np.int16, copy=False)
            if actions is not None or cache_path:
                transition_count = np.zeros(n_samples, dtype=np.int64)
                transition_count[1:] = np.cumsum(
                    (restimulus[1:] != restimulus[:-1]) | (rerepetition[1:] != rerepetition[:-1])
                )
                window_starts = np.asarray(feature_set["window_start_indices"], dtype=np.int64)
                window_ends = np.asarray(feature_set["window_end_indices"], dtype=np.int64)
                interval_starts = np.minimum(window_starts, alignment_indices)
                interval_ends = np.maximum(window_ends - 1, alignment_indices)
                label_isolated = (
                    (transition_count[interval_ends] == transition_count[interval_starts])
                    & (restimulus[window_starts] == aligned_actions)
                    & (rerepetition[window_starts] == aligned_repetitions)
                )
            else:
                label_isolated = np.ones(alignment_indices.shape, dtype=bool)
            recording = RecordingFeatures(
                recording_id=file_name,
                x_windows=np.asarray(feature_set["feature_matrix"], dtype=np.float32),
                y_windows=np.asarray(feature_set["target_values"], dtype=np.float32),
                target_alignment_indices=alignment_indices,
                feature_names=[
                    f"ch{channel}_{feature_name}"
                    for channel in config.emg_channels for feature_name in config.feature_order
                ],
                target_names=list(feature_set["target_names"] or []), fs=float(feature_set["fs"]),
                action_labels=aligned_actions, repetition_labels=aligned_repetitions,
                source_recording_id=file_name,
            )
            if cache_path:
                _save_recording_cache(cache_path, recording, label_isolated)
        recordings.append(recording)

        if actions is not None:
            aligned_actions = np.asarray(recording.action_labels)
            aligned_repetitions = np.asarray(recording.repetition_labels)
            selected_window_mask = (
                np.isin(aligned_actions, actions)
                & (aligned_repetitions > 0)
                & label_isolated
            )
            selected_indices = np.flatnonzero(selected_window_mask)
            if selected_indices.size:
                boundary_after = (
                    (np.diff(selected_indices) != 1)
                    | (aligned_actions[selected_indices[1:]] != aligned_actions[selected_indices[:-1]])
                    | (
                        aligned_repetitions[selected_indices[1:]]
                        != aligned_repetitions[selected_indices[:-1]]
                    )
                )
                group_boundaries = np.flatnonzero(boundary_after) + 1
                group_starts = np.concatenate(([0], group_boundaries))
                group_ends = np.concatenate((group_boundaries, [selected_indices.size]))
                for group_start, group_end in zip(group_starts.tolist(), group_ends.tolist()):
                    start = int(selected_indices[group_start])
                    end = int(selected_indices[group_end - 1]) + 1
                    if end - start < config.seq_len:
                        continue
                    action = int(aligned_actions[start])
                    repetition = int(aligned_repetitions[start])
                    selected_recordings.append(
                        RecordingFeatures(
                            recording_id=(
                                f"{file_name}:A{action}:R{repetition}:W{start}"
                            ),
                            x_windows=recording.x_windows[start:end],
                            y_windows=recording.y_windows[start:end],
                            target_alignment_indices=recording.target_alignment_indices[start:end],
                            feature_names=recording.feature_names,
                            target_names=recording.target_names,
                            fs=recording.fs,
                            action_labels=aligned_actions[start:end],
                            repetition_labels=aligned_repetitions[start:end],
                            source_recording_id=file_name,
                        )
                    )

    if not recordings:
        raise ValueError(f"subject {subject}: no usable recordings across exercises {exercises}")
    if actions is not None and not selected_recordings:
        raise ValueError(f"no action/repetition windows match requested actions {actions}")
    return recordings, selected_recordings


def load_subject_splits(
    subject: str,
    exercises: tuple[str, ...],
    config,
    *,
    actions: tuple[int, ...],
    cache_dir: Path | None = None,
) -> tuple[SequenceSplit, SequenceSplit]:
    """Return label-isolated selected rows and a full chronological feature stream."""
    recordings, selected_recordings = _load_subject_recordings(
        subject,
        exercises,
        config,
        actions=actions,
        cache_dir=cache_dir,
    )
    selected_split = build_sequence_split(
        selected_recordings,
        seq_len=config.seq_len,
        seq_stride=config.seq_stride,
    )
    stream_split = build_sequence_split(
        recordings,
        seq_len=config.seq_len,
        seq_stride=config.seq_stride,
    )
    return selected_split, stream_split


def load_subject_stream_split(
    subject: str,
    exercises: tuple[str, ...],
    config,
    *,
    cache_dir: Path | None = None,
) -> SequenceSplit:
    """Build chronological sequence streams for every available exercise file."""
    recordings, _ = _load_subject_recordings(
        subject,
        exercises,
        config,
        cache_dir=cache_dir,
    )
    return build_sequence_split(
        recordings,
        seq_len=config.seq_len,
        seq_stride=config.seq_stride,
    )


def load_subject_filtered_split(
    subject: str,
    exercises: tuple[str, ...],
    config,
    *,
    actions: tuple[int, ...],
) -> SequenceSplit:
    """Load one subject and retain selected action targets for training/scoring."""
    selected_split, _ = load_subject_splits(
        subject,
        exercises,
        config,
        actions=actions,
    )
    return selected_split


def select_repetition_split(
    split: SequenceSplit,
    *,
    actions: tuple[int, ...],
    train_repetitions_per_action: int,
    random_seed: int,
) -> tuple[SequenceSplit, SequenceSplit, dict[str, Any]]:
    if split.action_labels is None or split.repetition_labels is None:
        raise ValueError("paper repetition split requires action and repetition labels")

    labels = np.asarray(split.action_labels)
    repetitions = np.asarray(split.repetition_labels)
    rng = np.random.default_rng(random_seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    split_plan: dict[str, Any] = {}

    for action in actions:
        action_mask = labels == action
        available_repetitions = [
            int(value)
            for value in np.unique(repetitions[action_mask]).tolist()
            if int(value) > 0
        ]
        if len(available_repetitions) < train_repetitions_per_action + 1:
            raise ValueError(f"action {action} has too few repetitions: {available_repetitions}")
        train_reps = tuple(
            sorted(
                int(value)
                for value in rng.choice(
                    available_repetitions,
                    size=train_repetitions_per_action,
                    replace=False,
                ).tolist()
            )
        )
        test_reps = tuple(value for value in available_repetitions if value not in train_reps)
        train_indices.append(np.flatnonzero(action_mask & np.isin(repetitions, train_reps)))
        test_indices.append(np.flatnonzero(action_mask & np.isin(repetitions, test_reps)))
        split_plan[str(action)] = {
            "train_repetitions": list(train_reps),
            "test_repetitions": list(test_reps),
        }

    return (
        copy_split_by_indices(split, np.sort(np.concatenate(train_indices))),
        copy_split_by_indices(split, np.sort(np.concatenate(test_indices))),
        split_plan,
    )


def describe_all_selected_repetitions(
    split: SequenceSplit,
    *,
    actions: tuple[int, ...],
) -> dict[str, Any]:
    """Return reproducible source-supervision metadata without a train/test split."""
    if split.action_labels is None or split.repetition_labels is None:
        raise ValueError("source supervision metadata requires action and repetition labels")
    labels = np.asarray(split.action_labels)
    repetitions = np.asarray(split.repetition_labels)
    return {
        "supervision": "all_selected_repetitions",
        "selected_repetitions_by_action": {
            str(action): [
                int(value)
                for value in np.unique(repetitions[labels == action]).tolist()
                if int(value) > 0
            ]
            for action in actions
        },
        "n_selected_sequences": int(split.x.shape[0]),
    }


def copy_split_by_indices(split: SequenceSplit, indices: np.ndarray) -> SequenceSplit:
    indices = np.asarray(indices, dtype=np.int64)
    stream_start_flags = np.zeros(indices.size, dtype=bool)
    if split.stream_start_flags is not None:
        stream_start_flags |= np.asarray(split.stream_start_flags[indices], dtype=bool)
    if stream_start_flags.size:
        stream_start_flags[0] = True
        if stream_start_flags.size > 1:
            stream_start_flags[1:] |= np.diff(indices) != 1
    return SequenceSplit(
        x=split.x[indices],
        y=split.y[indices],
        time_s=split.time_s[indices],
        alignment_indices=split.alignment_indices[indices],
        recording_ids=split.recording_ids[indices],
        feature_names=split.feature_names,
        target_names=split.target_names,
        action_labels=None if split.action_labels is None else split.action_labels[indices],
        repetition_labels=None if split.repetition_labels is None else split.repetition_labels[indices],
        stream_start_flags=stream_start_flags,
        expected_alignment_step=split.expected_alignment_step,
        source_recording_ids=(
            None
            if split.source_recording_ids is None
            else split.source_recording_ids[indices]
        ),
    )


def concat_splits(splits: list[SequenceSplit]) -> SequenceSplit:
    if not splits:
        raise ValueError("cannot concatenate an empty split list")
    first = splits[0]
    action_labels = None
    if first.action_labels is not None:
        action_labels = np.concatenate([split.action_labels for split in splits if split.action_labels is not None])
    repetition_labels = None
    if first.repetition_labels is not None:
        repetition_labels = np.concatenate(
            [split.repetition_labels for split in splits if split.repetition_labels is not None]
        )
    stream_start_flags = None
    if all(split.stream_start_flags is not None for split in splits):
        flags = []
        for split in splits:
            split_flags = np.asarray(split.stream_start_flags, dtype=bool).copy()
            if split_flags.size:
                split_flags[0] = True
            flags.append(split_flags)
        stream_start_flags = np.concatenate(flags)
    expected_alignment_step = first.expected_alignment_step
    if any(split.expected_alignment_step != expected_alignment_step for split in splits[1:]):
        expected_alignment_step = None
    source_recording_ids = None
    if all(split.source_recording_ids is not None for split in splits):
        source_recording_ids = np.concatenate(
            [split.source_recording_ids for split in splits if split.source_recording_ids is not None]
        )
    return SequenceSplit(
        x=np.concatenate([split.x for split in splits], axis=0),
        y=np.concatenate([split.y for split in splits], axis=0),
        time_s=np.concatenate([split.time_s for split in splits], axis=0),
        alignment_indices=np.concatenate([split.alignment_indices for split in splits], axis=0),
        recording_ids=np.concatenate([split.recording_ids for split in splits], axis=0),
        feature_names=first.feature_names,
        target_names=first.target_names,
        action_labels=action_labels,
        repetition_labels=repetition_labels,
        stream_start_flags=stream_start_flags,
        expected_alignment_step=expected_alignment_step,
        source_recording_ids=source_recording_ids,
    )


def _augment_sequence_batch(
    x: torch.Tensor,
    *,
    amplitude_range: tuple[float, float] = (0.7, 1.3),
    noise_std: float = 0.05,
    time_shift_max: int = 2,
) -> torch.Tensor:
    """Apply EMG-specific augmentations to a batch of feature sequences.

    Simulates inter-subject variability during pretraining:
    - Amplitude scaling per channel (different muscle sizes / electrode impedance)
    - Additive Gaussian noise (electrode noise / skin conductivity differences)
    - Time shift per sequence (different electromechanical delays)
    """
    batch_size, _seq_len, n_features = x.shape
    x_aug = x.clone()

    # 1. Per-channel amplitude scaling
    scale = torch.empty(batch_size, 1, n_features, device=x.device).uniform_(
        amplitude_range[0], amplitude_range[1],
    )
    x_aug = x_aug * scale

    # 2. Additive Gaussian noise
    feat_std = x_aug.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    noise = torch.randn_like(x_aug) * noise_std * feat_std
    x_aug = x_aug + noise

    # 3. Random time shift
    shifts = torch.randint(
        -time_shift_max,
        time_shift_max + 1,
        (batch_size,),
        device=x.device,
    )
    return _roll_sequences_by_shift(x_aug, shifts)


def _roll_sequences_by_shift(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Cyclically shift each [time, feature] sequence by its own signed offset."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch, time, feature], got {tuple(x.shape)}")
    if shifts.shape != (x.shape[0],):
        raise ValueError(f"shifts must have shape ({x.shape[0]},), got {tuple(shifts.shape)}")

    # For a positive roll, output timestep t reads from input t - shift.
    time_indices = torch.arange(x.shape[1], device=x.device)
    source_indices = (time_indices.unsqueeze(0) - shifts.unsqueeze(1)) % x.shape[1]
    return x.gather(1, source_indices.unsqueeze(-1).expand(-1, -1, x.shape[2]))


def _chain_metrics_dict(chain_eval: dict[str, Any]) -> dict[str, Any]:
    """Extract jsonable fields from evaluate_chain for summary reporting."""
    return {
        "metrics": make_jsonable(chain_eval["metrics"]),
        "h_norm_stats": make_jsonable(chain_eval["h_norm_stats"]),
    }


def _sequence_provenance_indices(
    stream_split: SequenceSplit,
    selected_split: SequenceSplit,
) -> np.ndarray:
    """Map selected sequence provenance to full-stream row indices."""
    stream_sources = stream_split.source_recording_ids
    selected_sources = selected_split.source_recording_ids
    if stream_sources is None or selected_sources is None:
        raise ValueError("exact scoring requires source_recording_ids provenance")
    if stream_sources.shape != (stream_split.x.shape[0],):
        raise ValueError("stream source_recording_ids must have one entry per sequence")
    if selected_sources.shape != (selected_split.x.shape[0],):
        raise ValueError("selected source_recording_ids must have one entry per sequence")

    stream_lookup: dict[tuple[str, int], int] = {}
    for index, (source_id, alignment) in enumerate(
        zip(stream_sources, stream_split.alignment_indices)
    ):
        key = (str(source_id), int(alignment))
        if key in stream_lookup:
            raise ValueError(f"duplicate full-stream provenance key: {key}")
        stream_lookup[key] = index

    stream_indices = np.empty(selected_split.x.shape[0], dtype=np.int64)
    seen_selected_keys: set[tuple[str, int]] = set()
    for selected_index, (source_id, alignment) in enumerate(
        zip(selected_sources, selected_split.alignment_indices)
    ):
        key = (str(source_id), int(alignment))
        if key in seen_selected_keys:
            raise ValueError(f"duplicate selected provenance key: {key}")
        seen_selected_keys.add(key)
        stream_index = stream_lookup.get(key)
        if stream_index is None:
            raise ValueError(f"selected row is absent from full replay stream: {key}")
        stream_indices[selected_index] = stream_index
    return stream_indices


def provenance_score_mask(
    stream_split: SequenceSplit,
    selected_split: SequenceSplit,
) -> np.ndarray:
    """Select rows by source/alignment provenance without comparing labels."""
    stream_indices = _sequence_provenance_indices(stream_split, selected_split)
    score_mask = np.zeros(stream_split.x.shape[0], dtype=bool)
    score_mask[stream_indices] = True
    return score_mask


def exact_provenance_score_mask(
    stream_split: SequenceSplit,
    selected_split: SequenceSplit,
) -> np.ndarray:
    """Locate selected rows in a full feature stream and verify target identity."""
    stream_indices = _sequence_provenance_indices(stream_split, selected_split)
    for selected_index, stream_index in enumerate(stream_indices):
        if not np.allclose(
            stream_split.y[stream_index],
            selected_split.y[selected_index],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise ValueError(
                "selected target disagrees with full replay stream at "
                f"selected_index={selected_index}, stream_index={stream_index}"
            )
    score_mask = np.zeros(stream_split.x.shape[0], dtype=bool)
    score_mask[stream_indices] = True
    return score_mask


def exact_query_score_mask(
    stream_split: SequenceSplit,
    query_split: SequenceSplit,
) -> np.ndarray:
    """Backward-compatible name for query callers and external A/B scripts."""
    return exact_provenance_score_mask(stream_split, query_split)


def chain_evaluation_protocol_metadata() -> dict[str, str]:
    """Describe the recurrent replay and its upstream preprocessing boundary."""
    return {
        "scope": "chronological_feature_frame_recurrent_replay",
        "input_context": "all chronological windows, including rest, support, and unselected actions",
        "score_rows": "exact label-isolated held-out query target rows",
        "preprocessing_scope": (
            "offline full-recording mean removal and zero-phase filtering; not end-to-end causal raw-EMG replay"
        ),
        "session_initialization": (
            "zero state at replay start and after model or normalization selection"
        ),
        "state_policy": (
            "retain hidden state across action and repetition changes; reset before recording or explicit stream boundaries"
        ),
        "selection_role": (
            "final-query evaluation only; no source or support labels are used "
            "for checkpoint selection"
        ),
    }


def train_model(
    *,
    supervised_split: SequenceSplit,
    config,
    x_stats: dict[str, Any],
    y_stats: dict[str, Any],
    device: torch.device,
    augment_prob: float = 1.0,
    cuda_graph: bool = False,
    gpu_resident: bool = False,
    stateful: bool = False,
    stateful_stream_split: SequenceSplit | None = None,
    stateful_stream_score_mask: np.ndarray | None = None,
    initial_model: nn.Module | None = None,
    epoch_offset: int = 0,
) -> dict[str, Any]:
    """Fit for exactly ``max_epochs`` and return the final-epoch model.

    Stateful runs replay the complete chronological source stream, while the
    score mask selects every source repetition label for optimization.  There
    is intentionally no internal validation or checkpoint selection.
    """
    normalized_supervised = normalize_sequence_inputs(
        supervised_split, x_stats=x_stats, y_stats=y_stats
    )
    normalized_stateful_stream = normalized_supervised
    if stateful:
        if stateful_stream_split is not None:
            normalized_stateful_stream = normalize_sequence_inputs(
                stateful_stream_split,
                x_stats=x_stats,
                y_stats=y_stats,
            )
    model = initial_model if initial_model is not None else build_cfc_regressor(
        input_dim=normalized_supervised.x.shape[-1],
        output_dim=normalized_supervised.y.shape[-1],
        hidden_units=config.hidden_units,
        model_family=config.model_family,
        cfc_dropout=config.cfc_dropout,
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weights = config.pretrain_target_weights
    loss_fn = (WeightedMSELoss(weights) if weights else nn.MSELoss()).to(device)
    if weights and len(weights) != normalized_supervised.y.shape[-1]:
        raise ValueError("pretrain_target_weights must have one weight per output target")
    history: list[dict[str, float]] = []
    uses_cuda = device.type == "cuda"

    graphed_cfc, orig_cfc_forward, graphed_cfc_forward = None, None, None
    if cuda_graph and uses_cuda:
        graph_batch_size = config.batch_size
        if stateful:
            graph_batch_size = min(
                config.batch_size,
                len(SegmentIndex(normalized_stateful_stream)),
            )
        sample_x = torch.zeros(
            graph_batch_size,
            normalized_stateful_stream.x.shape[1],
            normalized_stateful_stream.x.shape[-1],
            device=device,
        )
        sample_hx = None
        if stateful:
            sample_hx = torch.zeros(
                graph_batch_size,
                model.cfc.state_size,
                device=device,
            )
        graphed_cfc, orig_cfc_forward, graphed_cfc_forward = build_graphed_cfc(model, sample_x, sample_hx)

    # drop_last depends on whether graph capture actually succeeded (a failed
    # capture must not silently drop the trailing partial batch).
    drop_last = graphed_cfc is not None
    if stateful:
        if not gpu_resident:
            print("  stateful training requires GPU-resident data; enabling --gpu-resident")
        segment_index = SegmentIndex(normalized_stateful_stream)
        loader = GpuResidentStatefulBatches(
            normalized_stateful_stream.x,
            normalized_stateful_stream.y,
            segment_index,
            batch_size=config.batch_size,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(config.random_seed),
            sequence_score_mask=stateful_stream_score_mask,
        )
        print(
            f"  GPU-resident stateful dataset: x={loader.x.numel() * loader.x.element_size() / 1e6:.1f} MB, "
            f"y={loader.y.numel() * loader.y.element_size() / 1e6:.1f} MB in VRAM, "
            f"segments={len(segment_index)}"
        )
    elif gpu_resident:
        loader = GpuResidentBatches(
            normalized_supervised.x,
            normalized_supervised.y,
            batch_size=config.batch_size,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(config.random_seed),
            drop_last=drop_last,
        )
        print(
            f"  GPU-resident dataset: x={loader.x.numel() * loader.x.element_size() / 1e6:.1f} MB, "
            f"y={loader.y.numel() * loader.y.element_size() / 1e6:.1f} MB in VRAM"
        )
    else:
        loader = make_train_loader(normalized_supervised, config, drop_last=drop_last)

    for epoch in range(1, config.max_epochs + 1):
        # Apply EMG augmentation during pretraining to improve cross-subject robustness
        model.train()
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        total_examples = 0
        if stateful:
            # Truncated BPTT carry-over (shared implementation in train.py).
            # Augmentation is disabled for stateful runs: per-batch circular
            # shifts would break the carried state's temporal continuity.
            loss = train_one_epoch_stateful(
                model,
                loader,
                optimizer,
                loss_fn,
                device=device,
                gradient_clip_norm=config.gradient_clip_norm,
                graphed_cfc=graphed_cfc,
            )
        else:
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=uses_cuda)
                y_batch = y_batch.to(device, non_blocking=uses_cuda)
                if torch.rand(1).item() < augment_prob:
                    x_batch = _augment_sequence_batch(x_batch)
                optimizer.zero_grad(set_to_none=True)
                if graphed_cfc is not None:
                    # CUDA graph path: deterministic CfC body in the graph, dropout and
                    # head outside so masks are re-sampled per batch (see build_graphed_cfc).
                    # Loader uses drop_last=True so every batch matches the graph shape.
                    y_sequence, _ = graphed_cfc(x_batch)
                    pred = model.head(model.dropout(y_sequence[:, -1, :]))
                else:
                    pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                if config.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                batch_size = x_batch.shape[0]
                total_loss += loss.detach().to(torch.float64) * batch_size
                total_examples += batch_size
            loss = float((total_loss / max(total_examples, 1)).item())
        history.append({"epoch": float(epoch + epoch_offset), "train_loss": float(loss)})
        print(f"    epoch {epoch + epoch_offset:3d} | train_loss={loss:.5f}", flush=True)
    if graphed_cfc is not None:
        # Return the model with the eager forward restored; downstream
        # evaluation (zero-shot, fine-tune) uses arbitrary batch shapes.
        model.cfc.forward = orig_cfc_forward
    return {"model": model, "history": history}


def freeze_for_linear_head(model: nn.Module) -> dict[str, Any]:
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_count = 0
    total_count = 0
    for name, parameter in model.named_parameters():
        should_train = name.startswith("head.")
        parameter.requires_grad = should_train
        count = int(parameter.numel())
        total_count += count
        if should_train:
            trainable_names.append(name)
            trainable_count += count
        else:
            frozen_names.append(name)
    return {
        "mode": "linear_head_only",
        "trainable_param_names": trainable_names,
        "frozen_param_names": frozen_names,
        "trainable_param_count": trainable_count,
        "total_param_count": total_count,
        "trainable_fraction": float(trainable_count / total_count) if total_count else 0.0,
    }


def _atl_training_epoch(
    *,
    source_model: nn.Module,
    target_model: nn.Module,
    dd: nn.Module,
    source_loader: Any,
    target_loader: Any,
    target_optimizer: torch.optim.Optimizer,
    dd_optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    subject_weight: float,
    device: torch.device,
    gradient_clip_norm: float | None,
    graphed_cfc: Any | None = None,
    eager_cfc: Any | None = None,
) -> dict[str, float]:
    """One causal stateful TBPTT epoch of GAN-style ATL.

    Source and target domains own independent hidden states. Each state is
    reset only by its own loader's verified stream-boundary mask, then detached
    between chunks. Warm-up and padding frames advance recurrence but never
    enter the discriminator, mapping, or supervised losses.
    """
    source_model.eval()
    target_model.train()
    dd.train()

    source_iter = source_loader.iter_training_batches()
    source_hx = torch.zeros(
        source_loader.batch_size,
        source_model.cfc.state_size,
        device=device,
    )
    target_hx = torch.zeros(
        target_loader.batch_size,
        target_model.cfc.state_size,
        device=device,
    )
    n_src_total = 0
    n_tgt_total = 0
    metric_dtype = torch.float64
    total_L_DD = torch.zeros((), dtype=metric_dtype, device=device)
    total_L_mapping = torch.zeros((), dtype=metric_dtype, device=device)
    total_L_subject = torch.zeros((), dtype=metric_dtype, device=device)
    n_src_correct = torch.zeros((), dtype=torch.int64, device=device)
    n_tgt_correct = torch.zeros((), dtype=torch.int64, device=device)
    n_batches = 0
    eps = 1e-8

    def reset_hidden(
        hx: torch.Tensor, reset_before: torch.Tensor, requires_reset: bool,
    ) -> torch.Tensor:
        reset = reset_before.to(device=device, dtype=torch.bool)
        if tuple(reset.shape) != (hx.shape[0],):
            raise ValueError(
                f"reset mask must have shape ({hx.shape[0]},), got {tuple(reset.shape)}"
            )
        if requires_reset:
            hx = hx.masked_fill(reset[:, None], 0.0)
        return hx

    for (
        target_x, target_y, target_score_mask, target_reset_before,
        target_requires_reset, n_tgt,
    ) in target_loader.iter_training_batches():
        target_hx = reset_hidden(target_hx, target_reset_before, target_requires_reset)
        target_score_mask = target_score_mask.to(device=device, dtype=torch.bool)
        if n_tgt == 0:
            # Context-only target chunks still advance New-t-net state, but do
            # not update either network because no support label is selected.
            with torch.no_grad():
                # Context has no backward pass; keep it outside the training graph.
                _, target_h_final = (eager_cfc or target_model.cfc)(target_x, target_hx)
            target_hx = target_h_final.detach()
            continue

        # Advance the independent source stream through any context-only
        # chunks until one with selected source-domain rows is available.
        while True:
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = source_loader.iter_training_batches()
                source_hx = torch.zeros_like(source_hx)
                source_batch = next(source_iter)

            (
                source_x, _source_y, source_score_mask, source_reset_before,
                source_requires_reset, n_src,
            ) = source_batch
            source_score_mask = source_score_mask.to(device=device, dtype=torch.bool)
            if getattr(source_loader, "encoder_outputs_cached", False):
                source_sequence = source_x
            else:
                source_hx = reset_hidden(source_hx, source_reset_before, source_requires_reset)
                with torch.no_grad():
                    source_sequence, source_h_final = source_model.cfc(source_x, source_hx)
                source_hx = source_h_final.detach()
            if n_src:
                F_s = source_sequence[source_score_mask]
                break

        # ---- Step 1: Train DD ----
        # The CfC body is deterministic in both supported families. Keep its
        # graph for the target step; DD sees detached features from this pass.
        if graphed_cfc is not None:
            target_sequence, target_h_final = graphed_cfc(target_x, target_hx)
        else:
            target_sequence, target_h_final = target_model.cfc(target_x, target_hx)
        F_t = target_sequence[target_score_mask]

        dd.train()
        dd_optimizer.zero_grad(set_to_none=True)
        domain_scores = dd(torch.cat([F_s, F_t.detach()], dim=0))
        dd_src = domain_scores[:n_src]
        dd_tgt = domain_scores[n_src:]
        L_DD = -(torch.log(dd_src + eps).mean()
                 + torch.log(1.0 - dd_tgt + eps).mean())
        L_DD.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(dd.parameters(), gradient_clip_norm)
        dd_optimizer.step()

        # ---- Step 2: Train New-t-net ----
        pred_t = target_model.head(target_model.dropout(target_sequence))[target_score_mask]
        scored_target_y = target_y[target_score_mask]

        target_optimizer.zero_grad(set_to_none=True)
        dd.eval()
        for parameter in dd.parameters():
            parameter.requires_grad_(False)
        try:
            L_mapping = -torch.log(dd(F_t) + eps).mean()
            L_subject = subject_weight * loss_fn(pred_t, scored_target_y)
            (L_mapping + L_subject).backward()
        finally:
            for parameter in dd.parameters():
                parameter.requires_grad_(True)
            dd.train()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(target_model.parameters(), gradient_clip_norm)
        target_optimizer.step()
        target_hx = target_h_final.detach()

        # ---- Metrics ----
        n_batches += 1
        total_L_DD += L_DD.detach().to(metric_dtype)
        total_L_mapping += L_mapping.detach().to(metric_dtype)
        total_L_subject += L_subject.detach().to(metric_dtype)
        n_src_total += n_src
        n_tgt_total += n_tgt
        with torch.no_grad():
            n_src_correct += (dd_src > 0.5).sum()
            n_tgt_correct += (dd_tgt < 0.5).sum()

    if n_batches == 0:
        raise ValueError("stateful ATL support score mask selects no sequence targets")
    return {
        "L_DD": float((total_L_DD / max(n_batches, 1)).item()),
        "L_mapping": float((total_L_mapping / max(n_batches, 1)).item()),
        "L_subject": float((total_L_subject / max(n_batches, 1)).item()),
        "dd_source_acc": float(n_src_correct.item() / max(n_src_total, 1)),
        "dd_target_acc": float(n_tgt_correct.item() / max(n_tgt_total, 1)),
        "n_source_domain_frames": float(n_src_total),
        "n_target_domain_frames": float(n_tgt_total),
        "n_supervised_frames": float(n_tgt_total),
    }


def stateful_training_protocol_metadata() -> dict[str, str]:
    """Disclose the causal-context and label-use boundary of stateful training."""
    return {
        "input_context": (
            "all chronological feature windows in each recording advance hidden state"
        ),
        "label_usage": (
            "loss uses every selected source or target-support row via exact provenance masks"
        ),
        "held_out_input_scope": (
            "held-out EMG windows may be observed only as causal unlabeled context; "
            "this is an input-transductive deployment-replay protocol"
        ),
        "gradient_scope": (
            "gradients are truncated at each TBPTT chunk; query-context chunks update hidden state "
            "without query-label loss or model selection"
        ),
        "reset_policy": "zero state at recording/explicit stream boundaries only",
    }



def fine_tune_head(
    *,
    model: nn.Module,
    support_split: SequenceSplit,
    config,
    y_stats: dict[str, Any],
    device: torch.device,
    learning_rate: float,
    epochs: int,
    # ATL (adversarial transfer learning) parameters
    enable_atl: bool = False,
    source_split: SequenceSplit | None = None,
    source_score_mask: np.ndarray | None = None,
    support_score_mask: np.ndarray | None = None,
    dd_lr: float = 1e-4,
    cfc_atl_lr: float = 1e-4,
    atl_subject_weight: float = 1.0,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:

    # ---- ATL (GAN-style, Lin & He 2024 Section 3.3) ----
    if enable_atl:
        if source_split is None:
            raise ValueError("source_split is required when enable_atl=True")
        if not hasattr(model, "forward_with_features"):
            raise ValueError(
                "ATL requires a model with forward_with_features(); "
                "use model_family='dense_cfc_linear' or 'autoncp_cfc_linear'"
            )
        for mask_name, score_mask, split in (
            ("source_score_mask", source_score_mask, source_split),
            ("support_score_mask", support_score_mask, support_split),
        ):
            if score_mask is None:
                continue
            flags = np.asarray(score_mask, dtype=bool)
            if split is None or flags.shape != (split.x.shape[0],):
                raise ValueError(f"{mask_name} must have one entry per sequence")
            if not flags.any():
                raise ValueError(f"{mask_name} selects no sequence targets")

        # Multi-s-net: frozen source feature reference
        source_model = copy.deepcopy(model).to(device)
        for param in source_model.parameters():
            param.requires_grad = False
        source_model.eval()

        # New-t-net: trainable, warm-start from pretrained weights
        target_model = copy.deepcopy(model).to(device)
        target_model.train()

        actual_hidden_units = int(target_model.cfc.state_size)
        feature_dim = int(target_model.cfc.output_size)
        dd = DomainDiscriminator(in_dim=feature_dim, hidden=128).to(device)

        trainable_count = sum(
            p.numel() for p in target_model.parameters() if p.requires_grad
        )
        total_count = sum(p.numel() for p in target_model.parameters())

        audit: dict[str, Any] = {
            "mode": "atl_gan_stateful_tbptt",
            "trainable_param_count": trainable_count,
            "total_param_count": total_count,
            "trainable_fraction": trainable_count / total_count if total_count else 0.0,
            "dd_param_count": sum(p.numel() for p in dd.parameters()),
            "subject_weight": atl_subject_weight,
            "source_state": "independent_causal_hidden_state",
            "target_state": "independent_causal_hidden_state",
            "checkpoint_selection": "none_final_epoch_returned",
            "input_context": "full_chronological_stream_context_labels_masked",
            "hidden_units": actual_hidden_units,
            "feature_dim": feature_dim,
        }

        target_optimizer = torch.optim.AdamW(
            target_model.parameters(),
            lr=cfc_atl_lr,
            weight_decay=config.weight_decay,
        )
        dd_optimizer = torch.optim.AdamW(
            dd.parameters(),
            lr=dd_lr,
            weight_decay=config.weight_decay,
        )

        loss_fn = nn.MSELoss()

        source_loader = GpuResidentStatefulBatches(
            source_split.x,
            source_split.y,
            SegmentIndex(source_split),
            batch_size=config.batch_size,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(config.random_seed),
            sequence_score_mask=source_score_mask,
        )
        target_loader = GpuResidentStatefulBatches(
            support_split.x,
            support_split.y,
            SegmentIndex(support_split),
            batch_size=config.batch_size,
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(config.random_seed + 1),
            sequence_score_mask=support_score_mask,
        )

        print("  Caching frozen source CfC features for ATL...", flush=True)
        source_loader.cache_encoder_outputs(source_model.cfc)
        graphed_cfc = orig_forward = None
        if config.cuda_graph and device.type == "cuda":
            sample_x = torch.zeros(
                target_loader.batch_size, config.seq_len, support_split.x.shape[-1],
                device=device,
            )
            sample_hx = torch.zeros(target_loader.batch_size, actual_hidden_units, device=device)
            graphed_cfc, orig_forward, _ = build_graphed_cfc(target_model, sample_x, sample_hx)
        audit.update(
            source_features_cached=source_loader.encoder_outputs_cached,
            target_forward_reused=True,
            cuda_graph_requested=bool(config.cuda_graph),
            cuda_graph_enabled=graphed_cfc is not None,
        )
        print(
            f"  ATL acceleration | source cache={audit['source_features_cached']} | "
            f"target CfC graph={audit['cuda_graph_enabled']}",
            flush=True,
        )

        history: list[dict[str, float]] = []
        try:
            for epoch in range(1, epochs + 1):
                train_metrics = _atl_training_epoch(
                    source_model=source_model,
                    target_model=target_model,
                    dd=dd,
                    source_loader=source_loader,
                    target_loader=target_loader,
                    target_optimizer=target_optimizer,
                    dd_optimizer=dd_optimizer,
                    loss_fn=loss_fn,
                    subject_weight=atl_subject_weight,
                    device=device,
                    gradient_clip_norm=config.gradient_clip_norm,
                    graphed_cfc=graphed_cfc,
                    eager_cfc=orig_forward,
                )

                entry: dict[str, float] = {
                    "epoch": float(epoch),
                    "L_DD": float(train_metrics["L_DD"]),
                    "L_mapping": float(train_metrics["L_mapping"]),
                    "L_subject": float(train_metrics["L_subject"]),
                }

                print(
                    f"  ATL epoch {epoch:3d} | "
                    f"DD={train_metrics['L_DD']:.4f} | "
                    f"Map={train_metrics['L_mapping']:.4f} | "
                    f"Reg={train_metrics['L_subject']:.4f}",
                    flush=True,
                )
                history.append(entry)
        finally:
            if orig_forward is not None:
                target_model.cfc.forward = orig_forward
        return target_model, history, audit

    # ---- Standard (non-ATL) branch ----
    # Both CfC model families expose a linear .head for head-only FT.
    adapted = copy.deepcopy(model).to(device)
    audit = freeze_for_linear_head(adapted)
    loader = make_train_loader(support_split, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            adapted,
            loader,
            optimizer,
            loss_fn,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        entry: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        history.append(entry)
    return adapted, history, audit


def _compute_doa_metrics_from_glove(
    y_true_glove: np.ndarray,
    y_pred_glove: np.ndarray,
    *,
    action_labels: np.ndarray | None = None,
) -> dict:
    """Convert glove-column predictions to 5-DoA metrics via post-hoc DOA5_W mapping."""
    y_true_doa = glove_to_doa(y_true_glove)
    y_pred_doa = glove_to_doa(y_pred_glove)
    metrics = compute_regression_metrics(y_true_doa, y_pred_doa)
    if action_labels is not None:
        metrics["per_action"] = compute_grouped_regression_metrics(
            y_true_doa, y_pred_doa, action_labels,
        )
    return metrics


def _split_sequence_split_temporal(
    split: SequenceSplit,
    *,
    train_fraction: float = 0.8,
) -> tuple[SequenceSplit, SequenceSplit]:
    """Legacy A/B helper; the supported protocol never uses this split."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    recording_ids = np.asarray(split.recording_ids)
    ordered_recording_ids = list(dict.fromkeys(recording_ids.tolist()))
    if len(ordered_recording_ids) < 2:
        raise ValueError("temporal split requires at least two recordings")

    recording_groups: list[list[str]] = []
    if split.action_labels is None:
        recording_groups.append(ordered_recording_ids)
    else:
        action_labels = np.asarray(split.action_labels)
        for action in dict.fromkeys(action_labels.tolist()):
            group = [
                recording_id
                for recording_id in ordered_recording_ids
                if np.all(action_labels[recording_ids == recording_id] == action)
            ]
            if group:
                recording_groups.append(group)

    train_recordings: list[str] = []
    held_out_recordings: list[str] = []
    for group in recording_groups:
        if len(group) == 1:
            train_recordings.extend(group)
            continue
        n_train = min(max(int(np.ceil(len(group) * train_fraction)), 1), len(group) - 1)
        train_recordings.extend(group[:n_train])
        held_out_recordings.extend(group[n_train:])
    if not held_out_recordings:
        held_out_recordings.append(train_recordings.pop())
    if not train_recordings:
        raise ValueError("temporal split left no training recording")
    return (
        copy_split_by_indices(split, np.flatnonzero(np.isin(recording_ids, train_recordings))),
        copy_split_by_indices(split, np.flatnonzero(np.isin(recording_ids, held_out_recordings))),
    )


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    additional_epochs = getattr(args, "additional_pretrain_epochs", 0)
    pretrain_weights = tuple(getattr(args, "pretrain_target_weights", ()))
    if additional_epochs < 0 or (additional_epochs and not args.resume_pretrain):
        raise ValueError("--additional-pretrain-epochs must be nonnegative and requires --resume-pretrain")
    if pretrain_weights:
        WeightedMSELoss(pretrain_weights)  # Fail before reading DB2 for invalid weights.
        if args.resume_pretrain and not additional_epochs:
            raise ValueError("--pretrain-target-weights on resume requires --additional-pretrain-epochs")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_requested = hasattr(args, "no_feature_cache") or hasattr(args, "feature_cache_dir")
    feature_cache_dir = None if getattr(args, "no_feature_cache", False) else (
        getattr(args, "feature_cache_dir", None) or args.db2_dir / ".feature_cache"
    ) if feature_cache_requested else None

    subjects = parse_csv_subjects(args.subjects)
    if not subjects or subjects[0].upper() == "ALL":
        subjects = _discover_subjects(args.db2_dir)
        if not subjects:
            raise ValueError(f"no DB2 subject directories found under {args.db2_dir}")

    exercise_raw = tuple(e.strip().upper() for e in args.exercise.split(",") if e.strip())
    if not exercise_raw or exercise_raw[0] == "ALL":
        exercises = ALL_EXERCISES
    else:
        exercises = exercise_raw

    target_subject = args.target_subject.upper()
    if target_subject not in subjects:
        raise ValueError(f"target subject {target_subject} is not in --subjects: {subjects}")

    # Resolve actions: when "all" is requested, peek at every exercise file
    # of the first subject to enumerate every non-rest action label.  Only
    # exercises that contain CyberGlove data are considered.
    if args.actions.strip().lower() == "all":
        _peek_subj = subjects[0]
        _all_actions: set[int] = set()
        for _peek_ex in exercises:
            _peek_path = args.db2_dir / f"DB2_{_peek_subj.lower()}" / f"{_peek_subj}_{_peek_ex}_A1.mat"
            if not _peek_path.is_file():
                continue
            _peek_data = load_data(str(_peek_path), variable_names=_RECORDING_CACHE_VARIABLES)
            if "glove" not in _peek_data:
                continue
            _all_actions.update(int(a) for a in _peek_data["restimulus"].flatten() if int(a) > 0)
        actions = tuple(sorted(_all_actions))
    else:
        actions = parse_csv_ints(args.actions)

    glove_columns = parse_csv_ints(args.glove_columns)
    target_contract = resolve_target_contract(args.target_mapping, glove_columns)
    emg_channels = parse_emg_channels(args.emg_channels)
    feature_order = tuple(f.strip().lower() for f in args.feature_order.split(",") if f.strip())
    if not feature_order:
        raise ValueError("--feature-order cannot be empty")
    _supported = {"mav", "mavs", "wl", "zc", "ssc", "rms"}
    _unknown = set(feature_order) - _supported
    if _unknown:
        raise ValueError(f"unsupported features in --feature-order: {sorted(_unknown)}. Supported: {sorted(_supported)}")

    config = build_best_cfc_config(
        db2_dir=args.db2_dir,
        emg_channels=emg_channels,
        target_source="glove",
        target_columns=glove_columns if not args.target_mapping else (),
        target_mapping=args.target_mapping,
        target_mapping_source="glove",
        target_mapping_version=target_contract["mapping_version"],
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        target_offset_samples=args.target_offset_samples,
        feature_order=feature_order,
        feature_normalization="mu_law",
        target_normalization="mu_law",
        mu_law_mu=args.mu_law_mu,
        seq_len=args.seq_len,
        seq_stride=1,
        hidden_units=args.hidden_units,
        model_family=args.model_family,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cfc_dropout=args.cfc_dropout,
        max_epochs=args.max_epochs,
        random_seed=args.random_seed,
        device=args.device,
        cuda_graph=args.cuda_graph,
        gpu_resident=args.gpu_resident,
        stateful=args.stateful,
        pretrain_target_weights=pretrain_weights,
    )
    model_tag = config.model_family.removesuffix("_linear")
    if pretrain_weights and len(pretrain_weights) != target_contract["output_dim"]:
        raise ValueError("--pretrain-target-weights must have one weight per output target")
    if additional_epochs and (
        output_dir / "checkpoints" / f"{target_subject}_{model_tag}_pretrain.pt"
    ).resolve() == args.resume_pretrain.resolve():
        raise ValueError("Continuation requires a separate --output-dir to preserve the input checkpoint")

    grouped = discover_target_files(config.db2_dir, config.target_source)
    missing_subjects = [subject for subject in subjects if subject not in grouped]
    if missing_subjects:
        raise ValueError(f"missing DB2 subjects required by paper protocol: {missing_subjects}")

    print("DB2 CfC joint-angle FT protocol")
    print(f"  model family   : {config.model_family}")
    print(f"  target subject : {target_subject}")
    print(f"  source subjects: {[subject for subject in subjects if subject != target_subject]}")
    print(f"  exercise/actions: {args.exercise} / {list(actions)}")
    print(
        f"  target mapping : {config.target_mapping} "
        f"(output dim={target_contract['output_dim']})"
    )
    print(f"  EMG channels   : {config.emg_channels} (input channels={len(config.emg_channels)})")
    print(f"  feature/window : {config.feature_order}, {args.window_ms} ms, stride {args.stride_ms} ms")
    print(f"  normalization  : mu-law mu={args.mu_law_mu}")
    print(f"  device  : {config.device}")
    print(f"  target_offset   : {config.target_offset_samples}")


    source_supervised_splits: list[SequenceSplit] = []
    source_stream_splits: list[SequenceSplit] = []
    source_stream_masks: list[np.ndarray] = []
    target_support_split = None
    target_query_split = None
    target_stream_split = None
    source_supervision_plan: dict[str, Any] = {}
    target_support_query_plan: dict[str, Any] | None = None

    for i, subject in enumerate(subjects, 1):
        print(f"  [{i}/{len(subjects)}] loading {subject}...", end="", flush=True)
        split_kwargs = {"actions": actions}
        if feature_cache_dir is not None:
            split_kwargs["cache_dir"] = feature_cache_dir
        split, stream_split = load_subject_splits(subject, exercises, config, **split_kwargs)
        print(f" {split.x.shape[0]} selected / {stream_split.x.shape[0]} stream seq", flush=True)
        if subject == target_subject:
            train_split, test_split, target_support_query_plan = select_repetition_split(
                split,
                actions=actions,
                train_repetitions_per_action=args.train_repetitions_per_action,
                random_seed=args.random_seed + int(subject[1:]),
            )
            target_support_split = train_split
            target_query_split = test_split
            target_stream_split = stream_split
        else:
            # Source subjects have no internal validation holdout: every
            # selected repetition label is supervised during pretraining.
            source_supervised_splits.append(split)
            source_stream_splits.append(stream_split)
            source_stream_masks.append(exact_provenance_score_mask(stream_split, split))
            source_supervision_plan[subject] = describe_all_selected_repetitions(
                split,
                actions=actions,
            )

    assert target_support_split is not None
    assert target_query_split is not None
    assert target_stream_split is not None
    assert target_support_query_plan is not None
    print("  concatenating source splits...", end="", flush=True)
    source_supervised = concat_splits(source_supervised_splits)
    source_stream = concat_splits(source_stream_splits)
    source_stream_score_mask = np.concatenate(source_stream_masks)
    print(
        f" supervised={source_supervised.x.shape[0]} "
        f"input_dim={source_supervised.x.shape[-1]}",
        flush=True,
    )

    resume_protocol = {
        "target_subject": target_subject,
        "source_subjects": [subject for subject in subjects if subject != target_subject],
        "exercises": list(exercises),
        "actions": list(actions),
        "target_support_repetitions_per_action": args.train_repetitions_per_action,
        "random_seed": args.random_seed,
        "target_support_query_plan": target_support_query_plan,
        "source_supervision_plan": source_supervision_plan,
        "source_pipeline": {
            "emg_channels": list(config.emg_channels),
            "target_source": config.target_source,
            "target_columns": list(config.target_columns),
            "target_mapping": config.target_mapping,
            "target_mapping_source": config.target_mapping_source,
            "target_mapping_version": config.target_mapping_version,
            "window_ms": config.window_ms,
            "stride_ms": config.stride_ms,
            "target_offset_samples": config.target_offset_samples,
            "feature_order": list(config.feature_order),
            "feature_normalization": config.feature_normalization,
            "target_normalization": config.target_normalization,
            "mu_law_mu": config.mu_law_mu,
            "seq_len": config.seq_len,
            "seq_stride": config.seq_stride,
            "stateful": config.stateful,
        },
    }

    device = resolve_device(config.device)
    print(f"  device: {device}", flush=True)

    if args.resume_pretrain:
        print(f"  Resuming pretrained model from {args.resume_pretrain}", flush=True)
        parent_checkpoint_sha256 = (
            hashlib.sha256(args.resume_pretrain.read_bytes()).hexdigest() if additional_epochs else None
        )
        ckpt = torch.load(args.resume_pretrain, map_location=device, weights_only=False)
        if additional_epochs and "adaptation_provenance" in ckpt:
            raise ValueError("Source continuation requires a pretrain checkpoint, not a target-adapted checkpoint")
        expected_resume_integrity = build_resume_integrity_metadata(
            x_stats=ckpt.get("feature_normalization_stats", {}),
            y_stats=ckpt.get("target_normalization_stats", {}),
            source_supervised=source_supervised,
            source_stream=source_stream,
            source_stream_score_mask=source_stream_score_mask,
            protocol=resume_protocol,
        )
        x_stats, y_stats = validate_resume_integrity_metadata(
            ckpt,
            expected_resume_integrity,
            allow_legacy_resume_without_integrity_binding=(
                args.allow_legacy_resume_without_integrity_binding
            ),
        )
        ckpt_config = ckpt.get("config", {})
        if not isinstance(ckpt_config, dict):
            raise TypeError("Checkpoint config must be a mapping to validate architecture.")
        originating_config = ckpt_config
        saved_provenance = ckpt.get("pretrain_provenance")
        pretrain_provenance = (
            saved_provenance
            if isinstance(saved_provenance, dict)
            else build_legacy_pretrain_provenance(ckpt_config)
        )
        validate_resume_architecture(ckpt_config, config)
        input_dim = resolve_resume_input_dim(
            ckpt_config,
            config,
            actual_input_dim=source_supervised.x.shape[-1],
        )
        ckpt_mapping = ckpt_config.get("target_mapping", "doa5")
        current_mapping = config.target_mapping
        if ckpt_mapping != current_mapping:
            raise ValueError(
                f"Checkpoint was trained with target_mapping='{ckpt_mapping}' "
                f"but current config uses target_mapping='{current_mapping}'. "
                f"Use matching --target-mapping or re-pretrain."
            )
        validate_checkpoint_target_contract(
            ckpt.get("target_contract"),
            target_contract,
        )
        output_dim = infer_checkpoint_output_dim(ckpt["model_state_dict"])
        expected_output_dim = int(source_supervised.y.shape[-1])
        if output_dim != expected_output_dim:
            raise ValueError(
                f"Checkpoint head has output_dim={output_dim}, but the current "
                f"target contract requires output_dim={expected_output_dim}."
            )
        pretrain_model = build_cfc_regressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=config.hidden_units,
            model_family=config.model_family,
            cfc_dropout=config.cfc_dropout,
        )
        pretrain_model.load_state_dict(ckpt["model_state_dict"])
        pretrain_model = pretrain_model.to(device)
        pretrain_model.eval()
        pretrain = {
            "model": pretrain_model,
            "history": project_pretrain_history(ckpt.get("history")),
        }
        if additional_epochs:
            if not pretrain["history"]:
                raise ValueError("Continuation requires checkpoint training history to establish the starting epoch")
            previous_epoch = int(max(entry["epoch"] for entry in pretrain["history"]))
            continuation_config = replace(config, max_epochs=additional_epochs)
            print(
                f"  Continuing source pretrain: {additional_epochs} epochs | "
                f"relative target weights={pretrain_weights or 'equal'} | new AdamW optimizer",
                flush=True,
            )
            set_random_seed(config.random_seed)
            continued = train_model(
                supervised_split=source_supervised, config=continuation_config,
                x_stats=x_stats, y_stats=y_stats, device=device,
                augment_prob=args.augment_prob, cuda_graph=args.cuda_graph,
                gpu_resident=args.gpu_resident, stateful=args.stateful,
                stateful_stream_split=source_stream,
                stateful_stream_score_mask=source_stream_score_mask,
                initial_model=pretrain_model, epoch_offset=previous_epoch,
            )
            pretrain = {"model": continued["model"], "history": pretrain["history"] + continued["history"]}
            pretrain_provenance = copy.deepcopy(pretrain_provenance)
            pretrain_provenance.setdefault("continuations", []).append({
                "checkpoint": str(args.resume_pretrain.resolve()),
                "checkpoint_sha256": parent_checkpoint_sha256,
                "config": make_jsonable(asdict(continuation_config)),
                "epoch_start": previous_epoch + 1,
                "epoch_end": previous_epoch + additional_epochs,
                "loss": "mean_normalized_weighted_mse" if pretrain_weights else "mse",
                "target_names": list(source_supervised.target_names),
                "optimizer_state": "reinitialized_adamw",
                "rng_state": "reseeded_from_config_random_seed",
            })
            originating_config = asdict(replace(config, max_epochs=previous_epoch + additional_epochs))
    else:
        originating_config = asdict(config)
        pretrain_provenance = build_pretrain_provenance(config, args)
        print("  fitting feature/target normalizers...", end="", flush=True)
        x_stats = fit_feature_normalizer(
            source_supervised.x.reshape(-1, source_supervised.x.shape[-1]),
            method=config.feature_normalization,
            mu=config.mu_law_mu,
        )
        y_stats = fit_target_normalizer(
            source_supervised.y,
            method=config.target_normalization,
            mu=config.mu_law_mu,
        )
        print(" done", flush=True)
        pretrain = train_model(
            supervised_split=source_supervised,
            config=config,
            x_stats=x_stats,
            y_stats=y_stats,
            device=device,
            augment_prob=args.augment_prob,
            cuda_graph=args.cuda_graph,
            gpu_resident=args.gpu_resident,
            stateful=args.stateful,
            stateful_stream_split=source_stream,
            stateful_stream_score_mask=source_stream_score_mask,
        )
    resume_invocation = build_resume_invocation_metadata(
        config,
        args,
        resume_checkpoint=args.resume_pretrain,
    )
    normalization_stats_path = save_feature_normalization_stats(
        output_dir,
        x_stats,
        emg_channels=config.emg_channels,
        feature_order=config.feature_order,
        target_stats=y_stats,
        target_names=source_supervised.target_names,
    )
    resume_integrity = build_resume_integrity_metadata(
        x_stats=x_stats,
        y_stats=y_stats,
        source_supervised=source_supervised,
        source_stream=source_stream,
        source_stream_score_mask=source_stream_score_mask,
        protocol=resume_protocol,
    )
    query_stream_score_mask = exact_provenance_score_mask(
        target_stream_split,
        target_query_split,
    )
    normalized_support = normalize_sequence_inputs(target_support_split, x_stats=x_stats, y_stats=y_stats)
    normalized_query = normalize_sequence_inputs(target_query_split, x_stats=x_stats, y_stats=y_stats)
    normalized_target_stream = normalize_sequence_inputs(
        target_stream_split,
        x_stats=x_stats,
        y_stats=y_stats,
    )
    zero_shot = evaluate_split(
        pretrain["model"],
        normalized_query,
        target_stats=y_stats,
        batch_size=config.batch_size,
        device=device,
    )
    # Post-hoc DoA metrics for glove_columns mode
    if config.target_mapping == "glove_columns":
        zero_shot["doa_metrics"] = _compute_doa_metrics_from_glove(
            zero_shot["y_true"], zero_shot["y_pred"],
            action_labels=normalized_query.action_labels,
        )
    support_stream_score_mask = exact_provenance_score_mask(
        normalized_target_stream,
        normalized_support,
    )
    if np.any(support_stream_score_mask & query_stream_score_mask):
        raise RuntimeError(
            "target support and query provenance overlap; query labels must not enter fine-tuning"
        )
    if args.skip_fine_tune:
        # Pretrain only — skip FT, save pretrain checkpoint, exit early
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        pretrain_checkpoint = checkpoint_dir / f"{target_subject}_{model_tag}_pretrain.pt"
        torch.save(
            {
                "model_state_dict": pretrain["model"].state_dict(),
                "config": originating_config,
                "target_contract": target_contract,
                "feature_normalization_stats": x_stats,
                "target_normalization_stats": y_stats,
                "resume_integrity": resume_integrity,
                "history": project_pretrain_history(pretrain["history"]),
                "pretrain_provenance": pretrain_provenance,
                "resume_invocation": resume_invocation,
            },
            pretrain_checkpoint,
        )

        summary = {
            "protocol": f"db2_{config.target_mapping}_{model_tag}_pretrain_only",
            "paper_reference": "reference files/fnins-18-1306050.pdf",
            "methodology": {
                "dataset": "Ninapro DB2",
                "subjects": list(subjects),
                "target_subject": target_subject,
                "source_subjects": [subject for subject in subjects if subject != target_subject],
                "exercise": args.exercise,
                "actions": list(actions),
                "target_contract": target_contract,
                "emg_channels_one_based": list(config.emg_channels),
                "feature_order": list(config.feature_order),
                "window_ms": args.window_ms,
                "stride_ms": args.stride_ms,
                "mu_law_mu": args.mu_law_mu,
                "target_support_repetitions_per_action": args.train_repetitions_per_action,
                "fine_tune": "skipped — pretrain only",
                "augment_prob": args.augment_prob,
            },
            "config": make_jsonable(originating_config),
            "invocation_config": make_jsonable(asdict(config)),
            "pretrain_provenance": make_jsonable(pretrain_provenance),
            "resume_invocation": make_jsonable(resume_invocation),
            "target_support_query_plan": target_support_query_plan,
            "source_supervision_plan": source_supervision_plan,
            "split_sizes": {
                "source_supervised": int(source_supervised.x.shape[0]),
                "source_stream_context": int(source_stream.x.shape[0]),
                "target_support": int(target_support_split.x.shape[0]),
                "target_query": int(target_query_split.x.shape[0]),
                "target_stream_context": int(target_stream_split.x.shape[0]),
            },
            "pretrain_history": make_jsonable(pretrain["history"]),
            "zero_shot_test_metrics": make_jsonable(zero_shot["metrics"]),
            "zero_shot_per_action_metrics": make_jsonable(zero_shot.get("per_action_metrics", {})),
            **({"zero_shot_doa_metrics": make_jsonable(zero_shot["doa_metrics"])} if "doa_metrics" in zero_shot else {}),
            "chain_evaluation_protocol": chain_evaluation_protocol_metadata(),
            "stateful_training_protocol": stateful_training_protocol_metadata(),
            "chain_metrics_zero_shot": _chain_metrics_dict(
                evaluate_chain(
                    pretrain["model"],
                    normalized_target_stream,
                    target_stats=y_stats,
                    device=device,
                    score_mask=query_stream_score_mask,
                )
            ),
            "artifacts": {
                "pretrain_checkpoint": str(pretrain_checkpoint),
                "feature_normalization": str(normalization_stats_path),
            },
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Pretrain-only complete. Checkpoint: {pretrain_checkpoint}")
        print(f"  Summary: {summary_path}")
        return summary

    adaptation_mode = "atl_stateful" if args.enable_atl else "head_only"
    adaptation_learning_rate = 1e-4 if args.enable_atl else args.fine_tune_learning_rate
    adaptation_provenance = build_adaptation_provenance(
        target_support_split=target_support_split,
        target_stream_split=target_stream_split,
        support_score_mask=support_stream_score_mask,
        mode=adaptation_mode,
        epochs=args.fine_tune_epochs,
        learning_rate=adaptation_learning_rate,
        atl_subject_weight=args.atl_subject_weight if args.enable_atl else None,
    )
    if args.enable_atl:
        atl_target_stream = copy.deepcopy(normalized_target_stream)
        atl_target_stream.y = np.array(atl_target_stream.y, copy=True)
        atl_target_stream.y[~support_stream_score_mask] = np.nan
        normalized_source_stream = normalize_sequence_inputs(
            source_stream,
            x_stats=x_stats,
            y_stats=y_stats,
        )
        adapted_model, ft_history, audit = fine_tune_head(
            model=pretrain["model"],
            support_split=atl_target_stream,
            config=config,
            y_stats=y_stats,
            device=device,
            learning_rate=args.fine_tune_learning_rate,
            epochs=args.fine_tune_epochs,
            enable_atl=True,
            source_split=normalized_source_stream,
            source_score_mask=source_stream_score_mask,
            support_score_mask=support_stream_score_mask,
            dd_lr=1e-4,
            cfc_atl_lr=1e-4,
            atl_subject_weight=args.atl_subject_weight,
        )
    else:
        adapted_model, ft_history, audit = fine_tune_head(
            model=pretrain["model"],
            support_split=normalized_support,
            config=config,
            y_stats=y_stats,
            device=device,
            learning_rate=args.fine_tune_learning_rate,
            epochs=args.fine_tune_epochs,
        )
    adapted = evaluate_split(
        adapted_model,
        normalized_query,
        target_stats=y_stats,
        batch_size=config.batch_size,
        device=device,
    )
    # Post-hoc DoA metrics for glove_columns mode
    if config.target_mapping == "glove_columns":
        adapted["doa_metrics"] = _compute_doa_metrics_from_glove(
            adapted["y_true"], adapted["y_pred"],
            action_labels=normalized_query.action_labels,
        )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrain_checkpoint = checkpoint_dir / f"{target_subject}_{model_tag}_pretrain.pt"
    adapted_checkpoint = checkpoint_dir / f"{target_subject}_{model_tag}_head_ft.pt"
    torch.save(
        {
            "model_state_dict": pretrain["model"].state_dict(),
            "config": originating_config,
            "target_contract": target_contract,
            "feature_normalization_stats": x_stats,
            "target_normalization_stats": y_stats,
            "resume_integrity": resume_integrity,
            "history": project_pretrain_history(pretrain["history"]),
            "pretrain_provenance": pretrain_provenance,
            "resume_invocation": resume_invocation,
        },
        pretrain_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": adapted_model.state_dict(),
            "config": originating_config,
            "target_contract": target_contract,
            "feature_normalization_stats": x_stats,
            "target_normalization_stats": y_stats,
            "resume_integrity": resume_integrity,
            "audit": audit,
            "history": project_fine_tune_history(ft_history),
            "adaptation_provenance": adaptation_provenance,
            "pretrain_provenance": pretrain_provenance,
            "resume_invocation": resume_invocation,
        },
        adapted_checkpoint,
    )

    summary = {
        "protocol": (
            f"db2_{config.target_mapping}_{model_tag}_head_finetune_no_atl"
            if not args.enable_atl
            else f"db2_{config.target_mapping}_{model_tag}_atl_stateful_tbptt"
        ),
        "paper_reference": "reference files/fnins-18-1306050.pdf",
        "methodology": {
            "dataset": "Ninapro DB2",
            "subjects": list(subjects),
            "target_subject": target_subject,
            "source_subjects": [subject for subject in subjects if subject != target_subject],
            "exercise": args.exercise,
            "actions": list(actions),
            "target_contract": target_contract,
            "emg_channels_one_based": list(config.emg_channels),
            "feature_order": list(config.feature_order),
            "window_ms": args.window_ms,
            "stride_ms": args.stride_ms,
            "mu_law_mu": args.mu_law_mu,
            "target_support_repetitions_per_action": args.train_repetitions_per_action,
            "fine_tune": (
                "head-only FT; ATL disabled"
                if not args.enable_atl
                else "causal stateful TBPTT with alternating DD and target-network training"
            ),
        },
        "config": make_jsonable(originating_config),
        "invocation_config": make_jsonable(asdict(config)),
        "pretrain_provenance": make_jsonable(pretrain_provenance),
        "resume_invocation": make_jsonable(resume_invocation),
        "target_support_query_plan": target_support_query_plan,
        "source_supervision_plan": source_supervision_plan,
        "split_sizes": {
            "source_supervised": int(source_supervised.x.shape[0]),
            "source_stream_context": int(source_stream.x.shape[0]),
            "target_support": int(target_support_split.x.shape[0]),
            "target_query": int(target_query_split.x.shape[0]),
            "target_stream_context": int(target_stream_split.x.shape[0]),
        },
        "pretrain_history": make_jsonable(pretrain["history"]) if pretrain.get("history") else [],
        "fine_tune_history": make_jsonable(ft_history),
        "parameter_audit": make_jsonable(audit),
        "zero_shot_test_metrics": {
            "metrics": make_jsonable(zero_shot["metrics"]),
            "per_action_metrics": make_jsonable(zero_shot["per_action_metrics"]),
            **({"doa_metrics": make_jsonable(zero_shot["doa_metrics"])} if "doa_metrics" in zero_shot else {}),
        },
        "adapted_test_metrics": {
            "metrics": make_jsonable(adapted["metrics"]),
            "per_action_metrics": make_jsonable(adapted["per_action_metrics"]),
            **({"doa_metrics": make_jsonable(adapted["doa_metrics"])} if "doa_metrics" in adapted else {}),
        },
        "chain_evaluation_protocol": chain_evaluation_protocol_metadata(),
        "stateful_training_protocol": stateful_training_protocol_metadata(),
        "chain_metrics_zero_shot": _chain_metrics_dict(
            evaluate_chain(
                pretrain["model"],
                normalized_target_stream,
                target_stats=y_stats,
                device=device,
                score_mask=query_stream_score_mask,
            )
        ),
        "chain_metrics_adapted": _chain_metrics_dict(
            evaluate_chain(
                adapted_model,
                normalized_target_stream,
                target_stats=y_stats,
                device=device,
                score_mask=query_stream_score_mask,
            )
        ),
        "artifacts": {
            "pretrain_checkpoint": str(pretrain_checkpoint),
            "adapted_checkpoint": str(adapted_checkpoint),
            "feature_normalization": str(normalization_stats_path),
        },
    }
    summary_path = save_summary(output_dir, summary)
    print(f"Saved DB2 joint-angle summary to: {summary_path}")
    return summary


def main() -> None:
    run_protocol(parse_args())


if __name__ == "__main__":
    main()
