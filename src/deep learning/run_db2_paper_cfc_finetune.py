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
    discover_target_files,
    make_jsonable,
    save_summary,
    subject_sort_key,
    summarize_best_epoch,
)
from train import (
    DomainDiscriminator,
    GradientReversalFunction,
    SequenceSplit,
    build_best_cfc_config,
    build_sequence_split,
    evaluate_split,
    fit_feature_normalizer,
    fit_target_normalizer,
    list_db2_files,
    load_recording_features,
    make_train_loader,
    normalize_sequence_inputs,
    resolve_device,
    train_one_epoch,
    build_cfc_regressor,
)
from datapreprocess import load_data, preprocess_emg
from SwRectify import sliding_window
from feature_extraction import extract_emg_features


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "log" / "db2_paper_cfc_finetune"

PAPER_SUBJECTS = ("S1", "S3", "S5", "S13", "S14", "S15", "S21", "S22", "S27", "S30")
PAPER_GLOVE_COLUMNS = (1, 2, 4, 5, 7, 8, 11, 12, 15, 16)
PAPER_ACTIONS = (18, 19, 20, 21, 22, 23)
PAPER_FEATURE_ORDER = ("rms",)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("integer list cannot be empty")
    return items


def parse_csv_subjects(value: str) -> tuple[str, ...]:
    subjects = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not subjects:
        raise ValueError("subject list cannot be empty")
    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DB2 protocol following Lin & He 2024: RMS-only features, 10 glove joints, "
            "cross-subject dense CfC pretrain, and FT that updates only the linear head."
        )
    )
    parser.add_argument("--db2-dir", type=Path, default=REPO_ROOT / "src" / "data" / "DB2")
    parser.add_argument("--target-subject", type=str, default="S1")
    parser.add_argument("--subjects", type=str, default=",".join(PAPER_SUBJECTS))
    parser.add_argument("--exercise", type=str, default="E2")
    parser.add_argument("--actions", type=str, default=",".join(str(value) for value in PAPER_ACTIONS))
    parser.add_argument(
        "--glove-columns",
        type=str,
        default=",".join(str(value) for value in PAPER_GLOVE_COLUMNS),
        help="Zero-based CyberGlove columns marked as red-dot joints in the paper figure.",
    )
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--stride-ms", type=float, default=50.0)
    parser.add_argument("--mu-law-mu", type=float, default=220.0)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ft-early-stopping-patience", type=int, default=5)
    parser.add_argument("--target-offset-samples", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cfc-dropout", type=float, default=0.1)
    parser.add_argument("--target-mapping", type=str, default="doa5")
    parser.add_argument("--train-repetitions-per-action", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-family",
        type=str,
        default="dense_cfc_linear",
        choices=["autoncp", "dense_cfc_linear"],
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=1.0,
        help="Probability of augmentation per batch. 1.0 = always, 0.0 = never",
    )
    parser.add_argument("--enable-atl", action="store_true", default=False)
    parser.add_argument("--atl-lambda-cap", type=float, default=1.0,
                        help="Maximum lambda for GRL gradient scaling in ATL (0.3-0.5 recommended)")
    parser.add_argument("--skip-fine-tune", action="store_true", default=False,
                        help="Pretrain only, skip fine-tuning (saves pretrain checkpoint)")
    parser.add_argument("--resume-pretrain", type=Path, default=None,
                        help="Skip pretraining; load pretrained model from this checkpoint path")
    return parser.parse_args()


def load_subject_exercise_split(subject: str, exercise: str, config) -> SequenceSplit:
    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    file_name = f"{subject}_{exercise}_A1.mat"
    if file_name not in available_files:
        raise FileNotFoundError(f"missing DB2 file for paper protocol: {file_name}")
    recording = load_recording_features(available_files[file_name], config)
    return build_sequence_split([recording], seq_len=config.seq_len, seq_stride=config.seq_stride)


def load_subject_filtered_split(
    subject: str,
    exercise: str,
    config,
    *,
    actions: tuple[int, ...],
) -> SequenceSplit:
    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    file_name = f"{subject}_{exercise}_A1.mat"
    if file_name not in available_files:
        raise FileNotFoundError(f"missing DB2 file for paper protocol: {file_name}")
    file_path = available_files[file_name]
    data = load_data(str(file_path))
    if "glove" not in data:
        raise KeyError(f"{file_name} does not contain CyberGlove data")

    emg = np.asarray(data["emg"], dtype=np.float32)
    glove = np.asarray(data["glove"], dtype=np.float32)
    restimulus = np.asarray(data["restimulus"]).reshape(-1)
    rerepetition = np.asarray(data["rerepetition"]).reshape(-1)
    n_samples = min(emg.shape[0], glove.shape[0], restimulus.shape[0], rerepetition.shape[0])
    emg = emg[:n_samples]
    glove = glove[:n_samples]
    restimulus = restimulus[:n_samples]
    rerepetition = rerepetition[:n_samples]

    # Preprocess the entire recording once — zero-phase filtering needs the full
    # temporal context to avoid edge artifacts that would appear if we filtered
    # each short action segment independently.
    emg_filtered = preprocess_emg(emg, fs=2000.0)

    selected_recordings = []
    target_columns = list(config.target_columns) if config.target_columns else []
    mapping = config.target_mapping

    for action in actions:
        for repetition in range(1, 7):
            indices = np.flatnonzero((restimulus == action) & (rerepetition == repetition))
            if indices.size == 0:
                continue
            start = int(indices[0])
            end = int(indices[-1]) + 1

            # Resolve targets: either apply 5-DoA mapping or use raw glove columns
            if mapping is not None:
                from doa_mapping import apply_linear_doa_mapping, DOA5_NAMES
                segment_targets = apply_linear_doa_mapping(glove[start:end], mapping=mapping)
                target_names = list(DOA5_NAMES)
            else:
                segment_targets = glove[start:end, target_columns]
                target_names = [f"glove_{column + 1}" for column in target_columns]

            windows = sliding_window(
                emg_filtered[start:end],
                segment_targets,
                fs=2000.0,
                window_ms=config.window_ms,
                stride_ms=config.stride_ms,
                target_mode=config.target_mode,
                target_offset_samples=config.target_offset_samples,
                target_names=target_names,
                target_prefix="glove",
            )
            feature_set = extract_emg_features(windows, feature_order=config.feature_order)
            selected_recordings.append(
                type("RecordingLike", (), {})()
            )
            selected_recordings[-1].recording_id = f"{file_name}:A{action}:R{repetition}"
            selected_recordings[-1].x_windows = np.asarray(feature_set["feature_matrix"], dtype=np.float32)
            selected_recordings[-1].y_windows = np.asarray(feature_set["target_values"], dtype=np.float32)
            selected_recordings[-1].target_alignment_indices = np.asarray(
                feature_set["target_alignment_indices"],
                dtype=np.int32,
            )
            selected_recordings[-1].feature_names = list(feature_set["channel_feature_names"])
            selected_recordings[-1].target_names = list(feature_set["target_names"] or [])
            selected_recordings[-1].fs = float(feature_set["fs"])
            selected_recordings[-1].action_labels = np.full(feature_set["n_windows"], action, dtype=np.int16)
            selected_recordings[-1].repetition_labels = np.full(feature_set["n_windows"], repetition, dtype=np.int16)
    if not selected_recordings:
        raise ValueError(f"{file_name} has no selected action/repetition windows")
    return build_sequence_split(selected_recordings, seq_len=config.seq_len, seq_stride=config.seq_stride)


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


def copy_split_by_indices(split: SequenceSplit, indices: np.ndarray) -> SequenceSplit:
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
    batch_size, seq_len, n_features = x.shape
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
    for b in range(batch_size):
        shift = int(torch.randint(-time_shift_max, time_shift_max + 1, (1,)).item())
        if shift != 0:
            x_aug[b] = torch.roll(x_aug[b], shifts=shift, dims=0)

    return x_aug


def train_model_with_validation(
    *,
    train_split: SequenceSplit,
    val_split: SequenceSplit,
    config,
    x_stats: dict[str, Any],
    y_stats: dict[str, Any],
    device: torch.device,
    augment_prob: float = 1.0,
) -> dict[str, Any]:
    normalized_train = normalize_sequence_inputs(train_split, x_stats=x_stats, y_stats=y_stats)
    normalized_val = normalize_sequence_inputs(val_split, x_stats=x_stats, y_stats=y_stats)
    loader = make_train_loader(normalized_train, config)
    model = build_cfc_regressor(
        input_dim=normalized_train.x.shape[-1],
        output_dim=normalized_train.y.shape[-1],
        hidden_units=config.hidden_units,
        model_family=config.model_family,
        cfc_dropout=config.cfc_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        # Apply EMG augmentation during pretraining to improve cross-subject robustness
        model.train()
        total_loss = 0.0
        total_examples = 0
        for x_batch, y_batch in loader:
            if torch.rand(1).item() < augment_prob:
                x_batch = _augment_sequence_batch(x_batch)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_size = x_batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
        loss = total_loss / max(total_examples, 1)
        val_eval = evaluate_split(
            model,
            normalized_val,
            target_stats=y_stats,
            batch_size=config.batch_size,
            device=device,
        )
        val_mae = float(val_eval["metrics"]["mae_mean"])
        history.append({"epoch": float(epoch), "train_loss": float(loss), "val_mae": val_mae})
        print(
            f"    epoch {epoch:3d} | train_loss={loss:.5f} | val_mae={val_mae:.4f}",
            flush=True,
        )
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
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


def _compute_dann_lambda(epoch_index: float) -> float:
    """Progressive lambda schedule for DANN domain loss weighting.

    Grows from ~0 toward 1.0, crossing 0.5 at epoch 5. Aggressive schedule
    suited for short ATL runs (10-30 epochs) where early domain pressure is
    needed before early stopping triggers.
    """
    return min(1.0, 2.0 / (1.0 + np.exp(-10.0 * (epoch_index / 10.0 - 0.5))))


def _atl_training_epoch(
    adapted: nn.Module,
    dd: nn.Module,
    source_loader: data.DataLoader,
    target_loader: data.DataLoader,
    cfc_optimizer: torch.optim.Optimizer,
    dd_optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    bce_fn: nn.Module,
    lambda_val: float,
    *,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    """Run one ATL epoch with combined regression + domain-adversarial loss."""
    adapted.train()
    dd.train()

    source_iter = iter(source_loader)
    total_mse = 0.0
    total_domain = 0.0
    n_source_correct = 0
    n_target_correct = 0
    n_source_total = 0
    n_target_total = 0
    total_batches = 0
    total_examples = 0

    for target_x, target_y in target_loader:
        target_x = target_x.to(device)
        target_y = target_y.to(device)

        try:
            source_x, source_y = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_x, source_y = next(source_iter)

        source_x = source_x.to(device)
        source_y = source_y.to(device)

        n_src = source_x.shape[0]
        n_tgt = target_x.shape[0]

        combined_x = torch.cat([source_x, target_x], dim=0)
        combined_y = torch.cat([source_y, target_y], dim=0)

        cfc_optimizer.zero_grad(set_to_none=True)
        dd_optimizer.zero_grad(set_to_none=True)

        pred, features = adapted.forward_with_features(combined_x)

        # Regression loss on all labelled data
        mse_loss = loss_fn(pred, combined_y)

        # Domain-adversarial loss via GRL
        src_feat = features[:n_src]
        tgt_feat = features[n_src:]

        src_rev = GradientReversalFunction.apply(src_feat, lambda_val)
        tgt_rev = GradientReversalFunction.apply(tgt_feat, lambda_val)

        src_domain_pred = dd(src_rev)
        tgt_domain_pred = dd(tgt_rev)

        src_domain_loss = bce_fn(src_domain_pred, torch.zeros(n_src, 1, device=device))
        tgt_domain_loss = bce_fn(tgt_domain_pred, torch.ones(n_tgt, 1, device=device))
        domain_loss = src_domain_loss + tgt_domain_loss

        total_loss = mse_loss + lambda_val * domain_loss
        total_loss.backward()

        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(adapted.parameters(), gradient_clip_norm)

        cfc_optimizer.step()
        dd_optimizer.step()

        batch_total = n_src + n_tgt
        total_mse += float(mse_loss.item()) * batch_total
        total_domain += float(domain_loss.item())
        total_examples += batch_total
        total_batches += 1

        # Track DD accuracy for diagnostics
        with torch.no_grad():
            n_source_correct += int(((src_domain_pred.cpu() < 0.5).float().sum().item()))
            n_target_correct += int(((tgt_domain_pred.cpu() > 0.5).float().sum().item()))
            n_source_total += n_src
            n_target_total += n_tgt

    return {
        "mse": total_mse / max(total_examples, 1),
        "domain_loss": total_domain / max(total_batches, 1),
        "dd_source_acc": n_source_correct / max(n_source_total, 1),
        "dd_target_acc": n_target_correct / max(n_target_total, 1),
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
    val_split: SequenceSplit | None = None,
    early_stopping_patience: int = 5,
    # ATL (adversarial transfer learning) parameters
    enable_atl: bool = False,
    source_split: SequenceSplit | None = None,
    dd_lr: float = 1e-3,
    cfc_atl_lr: float = 1e-4,
    atl_lambda_cap: float = 1.0,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    adapted = copy.deepcopy(model).to(device)

    # ---- ATL (Domain-Adversarial) branch ----
    if enable_atl:
        if source_split is None:
            raise ValueError("source_split is required when enable_atl=True")
        if not hasattr(adapted, "forward_with_features"):
            raise ValueError(
                "ATL requires a model with forward_with_features(); "
                "use model_family='dense_cfc_linear'"
            )

        # Unfreeze CfC body for domain-adversarial fine-tuning
        for param in adapted.parameters():
            param.requires_grad = True

        audit: dict[str, Any] = {
            "mode": "atl_domain_adaptation",
            "trainable_param_count": sum(p.numel() for p in adapted.parameters()),
            "total_param_count": sum(p.numel() for p in adapted.parameters()),
            "trainable_fraction": 1.0,
        }

        dd = DomainDiscriminator(in_dim=config.hidden_units, hidden=128).to(device)
        audit["dd_param_count"] = sum(p.numel() for p in dd.parameters())

        cfc_optimizer = torch.optim.AdamW(
            adapted.parameters(),
            lr=cfc_atl_lr,
            weight_decay=config.weight_decay,
        )
        dd_optimizer = torch.optim.AdamW(
            dd.parameters(),
            lr=dd_lr,
            weight_decay=config.weight_decay,
        )

        loss_fn = nn.MSELoss()
        bce_fn = nn.BCELoss()

        source_loader = make_train_loader(source_split, config)
        target_loader = make_train_loader(support_split, config)

        history: list[dict[str, float]] = []
        best_state = copy.deepcopy(adapted.state_dict())
        best_val_mae = float("inf")
        patience_counter = 0

        # Pre-mortem state
        domain_loss_window: list[float] = []

        for epoch in range(1, epochs + 1):
            lambda_val = _compute_dann_lambda(float(epoch)) * atl_lambda_cap

            train_metrics = _atl_training_epoch(
                adapted,
                dd,
                source_loader,
                target_loader,
                cfc_optimizer,
                dd_optimizer,
                loss_fn,
                bce_fn,
                lambda_val,
                device=device,
                gradient_clip_norm=config.gradient_clip_norm,
            )

            entry: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["mse"]),
                "domain_loss": float(train_metrics["domain_loss"]),
                "lambda": float(lambda_val),
                "dd_source_acc": float(train_metrics["dd_source_acc"]),
                "dd_target_acc": float(train_metrics["dd_target_acc"]),
            }

            print(
                f"  ATL epoch {epoch:3d} | "
                f"MSE={train_metrics['mse']:.5f} | "
                f"domain={train_metrics['domain_loss']:.5f} | "
                f"lambda={lambda_val:.4f} | "
                f"DD src={train_metrics['dd_source_acc']:.3f} "
                f"tgt={train_metrics['dd_target_acc']:.3f}",
                flush=True,
            )

            # Pre-mortem: DD loss too low for too long indicates perfect
            # domain separation -> cap lambda to prevent gradient explosion
            domain_loss_window.append(train_metrics["domain_loss"])
            if len(domain_loss_window) > 10:
                domain_loss_window.pop(0)
            if (
                len(domain_loss_window) >= 10
                and all(d < 0.01 for d in domain_loss_window)
                and atl_lambda_cap > 0.5
            ):
                atl_lambda_cap = 0.5
                print(
                    f"    [pre-mortem] DD loss < 0.01 for 10 epochs; "
                    f"capping lambda at {atl_lambda_cap}",
                    flush=True,
                )

            # Pre-mortem: DD overfitting on source (source acc >> target acc)
            if (
                train_metrics["dd_source_acc"] > 0.95
                and train_metrics["dd_target_acc"] < 0.5
            ):
                print(
                    "    [pre-mortem] DD overfitting on source; "
                    "increasing DD dropout not supported in current DomainDiscriminator",
                    flush=True,
                )

            # Validation: MSE-only (no domain loss)
            if val_split is not None and val_split.x.shape[0] > 0:
                adapted.eval()
                val_eval = evaluate_split(
                    adapted,
                    val_split,
                    target_stats=y_stats,
                    batch_size=config.batch_size,
                    device=device,
                )
                val_mae = float(val_eval["metrics"]["mae_mean"])
                entry["val_mae"] = val_mae

                print(f"         val MSE={val_mae:.5f}", flush=True)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = copy.deepcopy(adapted.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        entry["early_stopped"] = 1.0
                        history.append(entry)
                        break
            else:
                best_state = copy.deepcopy(adapted.state_dict())

            history.append(entry)

        adapted.load_state_dict(best_state)
        return adapted, history, audit

    # ---- Standard (non-ATL) branch ----
    # AutoNCP has no separate linear head — full-model fine-tune with lower LR.
    # DenseCfCLinearRegressor has a .head that can be isolated for head-only FT.
    if getattr(adapted, "model_family", "") == "autoncp":
        audit = {"mode": "full_model", "trainable_param_count": sum(p.numel() for p in adapted.parameters()), "total_param_count": sum(p.numel() for p in adapted.parameters()), "trainable_fraction": 1.0}
    else:
        audit = freeze_for_linear_head(adapted)
    loader = make_train_loader(support_split, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    history = []
    best_state = copy.deepcopy(adapted.state_dict())
    best_val_mae = float("inf")
    patience_counter = 0

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

        if val_split is not None and val_split.x.shape[0] > 0:
            val_eval = evaluate_split(
                adapted,
                val_split,
                target_stats=y_stats,
                batch_size=config.batch_size,
                device=device,
            )
            val_mae = float(val_eval["metrics"]["mae_mean"])
            entry["val_mae"] = val_mae

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = copy.deepcopy(adapted.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    entry["early_stopped"] = 1.0
                    history.append(entry)
                    break
        else:
            best_state = copy.deepcopy(adapted.state_dict())

        history.append(entry)

    adapted.load_state_dict(best_state)
    return adapted, history, audit


def _split_sequence_split_temporal(
    split: SequenceSplit,
    *,
    train_fraction: float = 0.8,
) -> tuple[SequenceSplit, SequenceSplit]:
    """Split one SequenceSplit into train/val by temporal position within each recording.

    Sequences are grouped by recording_id (one per action-repetition segment).
    Within each recording, the first `train_fraction` of sequences (in temporal
    order) go to train, the remainder to val.  This gives a lightweight validation
    signal for early stopping without requiring a separate held-out recording.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    recording_ids = np.asarray(split.recording_ids)
    unique_ids = np.unique(recording_ids)

    train_mask = np.zeros(split.x.shape[0], dtype=bool)
    for rec_id in unique_ids:
        rec_indices = np.flatnonzero(recording_ids == rec_id)
        # Sequences within a recording are already in temporal order
        n_train = max(1, int(np.ceil(rec_indices.shape[0] * train_fraction)))
        train_mask[rec_indices[:n_train]] = True

    val_mask = ~train_mask
    if not np.any(val_mask):
        # Fallback: use last sequence of the longest recording as val
        longest_rec = max(unique_ids, key=lambda rid: np.sum(recording_ids == rid))
        rec_indices = np.flatnonzero(recording_ids == longest_rec)
        val_mask[rec_indices[-1]] = True
        train_mask[rec_indices[-1]] = False

    return (
        copy_split_by_indices(split, np.flatnonzero(train_mask)),
        copy_split_by_indices(split, np.flatnonzero(val_mask)),
    )


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_csv_subjects(args.subjects)
    actions = parse_csv_ints(args.actions)
    glove_columns = parse_csv_ints(args.glove_columns)
    target_subject = args.target_subject.upper()
    if target_subject not in subjects:
        raise ValueError(f"target subject {target_subject} is not in --subjects")

    config = build_best_cfc_config(
        db2_dir=args.db2_dir,
        target_source="glove",
        target_columns=glove_columns if not args.target_mapping else (),
        target_mapping=args.target_mapping,
        target_mapping_source="glove",
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        target_offset_samples=args.target_offset_samples,
        feature_order=("mav", "mavs", "wl", "zc", "ssc"),
        feature_normalization="mu_law",
        target_normalization="mu_law",
        mu_law_mu=args.mu_law_mu,
        seq_len=8,
        seq_stride=1,
        hidden_units=args.hidden_units,
        model_family=args.model_family,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cfc_dropout=args.cfc_dropout,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        random_seed=args.random_seed,
        device=args.device,
    )

    grouped = discover_target_files(config.db2_dir, config.target_source)
    missing_subjects = [subject for subject in subjects if subject not in grouped]
    if missing_subjects:
        raise ValueError(f"missing DB2 subjects required by paper protocol: {missing_subjects}")

    print("DB2 paper-method dense CfC FT protocol")
    print(f"  target subject : {target_subject}")
    print(f"  source subjects: {[subject for subject in subjects if subject != target_subject]}")
    print(f"  exercise/actions: {args.exercise} / {list(actions)}")
    print(f"  glove columns  : {list(glove_columns)}")
    print(f"  feature/window : {config.feature_order}, {args.window_ms} ms, stride {args.stride_ms} ms")
    print(f"  normalization  : mu-law mu={args.mu_law_mu}")

    source_train_splits: list[SequenceSplit] = []
    source_val_splits: list[SequenceSplit] = []
    target_support_split = None
    target_query_split = None
    repetition_plan: dict[str, Any] = {}

    for i, subject in enumerate(subjects, 1):
        print(f"  [{i}/{len(subjects)}] loading {subject}...", end="", flush=True)
        split = load_subject_filtered_split(subject, args.exercise, config, actions=actions)
        print(f" {split.x.shape[0]} seq", flush=True)
        train_split, test_split, plan = select_repetition_split(
            split,
            actions=actions,
            train_repetitions_per_action=args.train_repetitions_per_action,
            random_seed=args.random_seed + int(subject[1:]),
        )
        repetition_plan[subject] = plan
        if subject == target_subject:
            target_support_split = train_split
            target_query_split = test_split
        else:
            source_train_splits.append(train_split)
            source_val_splits.append(test_split)

    assert target_support_split is not None
    assert target_query_split is not None
    print("  concatenating source splits...", end="", flush=True)
    source_train = concat_splits(source_train_splits)
    source_val = concat_splits(source_val_splits)
    print(f" train={source_train.x.shape[0]} val={source_val.x.shape[0]}", flush=True)

    print("  fitting feature normalizer...", end="", flush=True)
    x_stats = fit_feature_normalizer(
        source_train.x.reshape(-1, source_train.x.shape[-1]),
        method=config.feature_normalization,
        mu=config.mu_law_mu,
    )
    print(" done", flush=True)

    print("  fitting target normalizer...", end="", flush=True)
    y_stats = fit_target_normalizer(source_train.y, method=config.target_normalization, mu=config.mu_law_mu)
    print(" done", flush=True)

    device = resolve_device(config.device)
    print(f"  device: {device}", flush=True)

    if args.resume_pretrain:
        print(f"  Resuming pretrained model from {args.resume_pretrain}", flush=True)
        ckpt = torch.load(args.resume_pretrain, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", {})
        input_dim = len(ckpt_config.get("feature_order", config.feature_order)) * 12
        output_dim = 5
        pretrain_model = build_cfc_regressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=ckpt_config.get("hidden_units", config.hidden_units),
            model_family=ckpt_config.get("model_family", config.model_family),
            cfc_dropout=ckpt_config.get("cfc_dropout", config.cfc_dropout),
        )
        pretrain_model.load_state_dict(ckpt["model_state_dict"])
        pretrain_model.eval()
        pretrain = {
            "model": pretrain_model,
            "history": ckpt.get("history", []),
        }
    else:
        pretrain = train_model_with_validation(
            train_split=source_train,
            val_split=source_val,
            config=config,
            x_stats=x_stats,
            y_stats=y_stats,
            device=device,
            augment_prob=args.augment_prob,
        )
    normalized_support = normalize_sequence_inputs(target_support_split, x_stats=x_stats, y_stats=y_stats)
    normalized_query = normalize_sequence_inputs(target_query_split, x_stats=x_stats, y_stats=y_stats)
    zero_shot = evaluate_split(
        pretrain["model"],
        normalized_query,
        target_stats=y_stats,
        batch_size=config.batch_size,
        device=device,
    )
    ft_train_split, ft_val_split = _split_sequence_split_temporal(
        normalized_support, train_fraction=0.8,
    )
    if args.skip_fine_tune:
        # Pretrain only — skip FT, save pretrain checkpoint, exit early
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        pretrain_checkpoint = checkpoint_dir / f"{target_subject}_dense_cfc_pretrain.pt"
        torch.save({"model_state_dict": pretrain["model"].state_dict(), "config": asdict(config)}, pretrain_checkpoint)

        summary = {
            "protocol": "db2_paper_dense_cfc_pretrain_only",
            "paper_reference": "reference files/fnins-18-1306050.pdf",
            "methodology": {
                "dataset": "Ninapro DB2",
                "subjects": list(subjects),
                "target_subject": target_subject,
                "source_subjects": [subject for subject in subjects if subject != target_subject],
                "exercise": args.exercise,
                "actions": list(actions),
                "glove_columns_zero_based": list(glove_columns),
                "glove_columns_one_based": [column + 1 for column in glove_columns],
                "feature_order": list(config.feature_order),
                "window_ms": args.window_ms,
                "stride_ms": args.stride_ms,
                "mu_law_mu": args.mu_law_mu,
                "train_repetitions_per_action": args.train_repetitions_per_action,
                "fine_tune": "skipped — pretrain only",
                "augment_prob": args.augment_prob,
            },
            "config": make_jsonable(asdict(config)),
            "repetition_plan": repetition_plan,
            "split_sizes": {
                "source_train": int(source_train.x.shape[0]),
                "source_val": int(source_val.x.shape[0]),
                "target_support": int(target_support_split.x.shape[0]),
                "target_query": int(target_query_split.x.shape[0]),
            },
            "pretrain_best_epoch": make_jsonable(summarize_best_epoch(pretrain["history"])),
            "pretrain_history": make_jsonable(pretrain["history"]),
            "zero_shot_test_metrics": make_jsonable(zero_shot["metrics"]),
            "artifacts": {
                "pretrain_checkpoint": str(pretrain_checkpoint),
            },
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Pretrain-only complete. Checkpoint: {pretrain_checkpoint}")
        print(f"  Summary: {summary_path}")
        return summary
    elif args.enable_atl:
        normalized_source = normalize_sequence_inputs(source_train, x_stats=x_stats, y_stats=y_stats)
        adapted_model, ft_history, audit = fine_tune_head(
            model=pretrain["model"],
            support_split=ft_train_split,
            config=config,
            y_stats=y_stats,
            device=device,
            learning_rate=args.fine_tune_learning_rate,
            epochs=args.fine_tune_epochs,
            val_split=ft_val_split,
            early_stopping_patience=args.ft_early_stopping_patience,
            enable_atl=True,
            source_split=normalized_source,
            dd_lr=1e-3,
            cfc_atl_lr=1e-4,
            atl_lambda_cap=args.atl_lambda_cap,
        )
    else:
        adapted_model, ft_history, audit = fine_tune_head(
            model=pretrain["model"],
            support_split=ft_train_split,
            config=config,
            y_stats=y_stats,
            device=device,
            learning_rate=args.fine_tune_learning_rate,
            epochs=args.fine_tune_epochs,
            val_split=ft_val_split,
            early_stopping_patience=args.ft_early_stopping_patience,
        )
    adapted = evaluate_split(
        adapted_model,
        normalized_query,
        target_stats=y_stats,
        batch_size=config.batch_size,
        device=device,
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrain_checkpoint = checkpoint_dir / f"{target_subject}_dense_cfc_pretrain.pt"
    adapted_checkpoint = checkpoint_dir / f"{target_subject}_dense_cfc_head_ft.pt"
    torch.save({"model_state_dict": pretrain["model"].state_dict(), "config": asdict(config)}, pretrain_checkpoint)
    torch.save({"model_state_dict": adapted_model.state_dict(), "config": asdict(config), "audit": audit}, adapted_checkpoint)

    summary = {
        "protocol": "db2_paper_dense_cfc_head_finetune_no_atl" if not args.enable_atl else "db2_paper_dense_cfc_atl",
        "paper_reference": "reference files/fnins-18-1306050.pdf",
        "methodology": {
            "dataset": "Ninapro DB2",
            "subjects": list(subjects),
            "target_subject": target_subject,
            "source_subjects": [subject for subject in subjects if subject != target_subject],
            "exercise": args.exercise,
            "actions": list(actions),
            "glove_columns_zero_based": list(glove_columns),
            "glove_columns_one_based": [column + 1 for column in glove_columns],
            "feature_order": list(config.feature_order),
            "window_ms": args.window_ms,
            "stride_ms": args.stride_ms,
            "mu_law_mu": args.mu_law_mu,
            "train_repetitions_per_action": args.train_repetitions_per_action,
            "fine_tune": "head-only FT; ATL disabled" if not args.enable_atl else "ATL domain adaptation with GRL+DD",
        },
        "config": make_jsonable(asdict(config)),
        "repetition_plan": repetition_plan,
        "split_sizes": {
            "source_train": int(source_train.x.shape[0]),
            "source_val": int(source_val.x.shape[0]),
            "target_support": int(target_support_split.x.shape[0]),
            "target_query": int(target_query_split.x.shape[0]),
        },
        "pretrain_best_epoch": make_jsonable(summarize_best_epoch(pretrain["history"])) if pretrain.get("history") else None,
        "pretrain_history": make_jsonable(pretrain["history"]) if pretrain.get("history") else [],
        "fine_tune_history": make_jsonable(ft_history),
        "parameter_audit": make_jsonable(audit),
        "zero_shot_test_metrics": {
            "metrics": make_jsonable(zero_shot["metrics"]),
            "per_action_metrics": make_jsonable(zero_shot["per_action_metrics"]),
        },
        "adapted_test_metrics": {
            "metrics": make_jsonable(adapted["metrics"]),
            "per_action_metrics": make_jsonable(adapted["per_action_metrics"]),
        },
        "artifacts": {
            "pretrain_checkpoint": str(pretrain_checkpoint),
            "adapted_checkpoint": str(adapted_checkpoint),
        },
    }
    summary_path = save_summary(output_dir, summary)
    print(f"Saved DB2 paper-method summary to: {summary_path}")
    return summary


def main() -> None:
    run_protocol(parse_args())


if __name__ == "__main__":
    main()
