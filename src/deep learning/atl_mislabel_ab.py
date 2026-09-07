"""Mislabeled-target ATL control experiment.

Hypothesis under test (user-proposed): ATL trained on S1 EMG paired with
ANOTHER subject's glove angles can generalize ("别人glove + 自己EMG 可以推广").

Same pipeline/base checkpoint as atl_ft30_rerun.py; ONLY the target-side labels
change before ATL:
- truth    : S1 labels (positive control)
- template : cross-subject MEDIAN glove trajectory of the same action,
  phase-aligned to each S1 action segment (correct action + correct phase,
  wrong person). Isolates inter-subject angle differences.
- index    : S2 labels in raw row order, tiled (unpaired lower bound).

Evaluation is always against S1 TRUE query labels (cold-start + chain).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_db2_paper_cfc_finetune import (
    ALL_EXERCISES,
    _split_sequence_split_temporal,
    concat_splits,
    exact_query_score_mask,
    fine_tune_head,
    fit_feature_normalizer,
    infer_checkpoint_output_dim,
    load_subject_splits,
    provenance_score_mask,
    select_repetition_split,
)
from train import (
    CfCTrainingConfig,
    SequenceSplit,
    build_cfc_regressor,
    evaluate_chain,
    evaluate_split,
    fit_target_normalizer,
    normalize_sequence_inputs,
    resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN_DIR = REPO_ROOT / "log/db2_paper_cfc_finetune/8ch_rmszcwl_h512_b1024_atl_w10_lr1e-4_2"
BASE_CHECKPOINT = BASE_RUN_DIR / "checkpoints/S1_dense_cfc_pretrain.pt"
BASE_SUMMARY = BASE_RUN_DIR / "summary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "log/db2_paper_cfc_finetune/atl_mislabel_ab_20260814"
TUPLE_FIELDS = {"emg_channels", "feature_order", "target_columns"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--ft-epochs", type=int, default=30)
    p.add_argument("--ft-patience", type=int, default=5)
    p.add_argument("--atl-weight", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--modes", type=str, default="truth,template,index")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _segment_runs(labels, reps):
    runs = []
    n = len(labels)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start] or reps[i] != reps[start]:
            runs.append((start, i, int(labels[start]), int(reps[start])))
            start = i
    return runs


def _phase_median(trajectories, out_len):
    if not trajectories:
        raise ValueError("no source trajectories")
    if out_len == 1:
        stack = np.stack([np.asarray(t, dtype=np.float64).mean(axis=0) for t in trajectories])
        return np.median(stack, axis=0).reshape(1, -1)
    new_phase = np.linspace(0.0, 1.0, out_len)
    resampled = []
    for traj in trajectories:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.shape[0] == 1:
            resampled.append(np.repeat(traj, out_len, axis=0))
            continue
        old_phase = np.linspace(0.0, 1.0, traj.shape[0])
        resampled.append(np.column_stack([
            np.interp(new_phase, old_phase, traj[:, d]) for d in range(traj.shape[1])]))
    return np.median(np.stack(resampled), axis=0)


def build_template_labels(target: SequenceSplit, source: SequenceSplit) -> np.ndarray:
    """Relabel target rows with cross-subject median action template (phase-aligned)."""
    out = np.zeros_like(target.y, dtype=np.float32)
    if target.action_labels is None or source.action_labels is None:
        raise ValueError("template labels require action/repetition labels")
    by_action: dict[int, list[np.ndarray]] = defaultdict(list)
    for start, end, action, _rep in _segment_runs(source.action_labels, source.repetition_labels):
        by_action[action].append(source.y[start:end])
    for start, end, action, _rep in _segment_runs(target.action_labels, target.repetition_labels):
        if action not in by_action:
            raise ValueError(f"action {action} missing from source split")
        out[start:end] = _phase_median(by_action[action], end - start)
    return out


def build_index_labels(target: SequenceSplit, donor_y: np.ndarray) -> np.ndarray:
    """Raw row-order mislabeling: target row i gets donor row i (tiled)."""
    n = target.y.shape[0]
    if donor_y.shape[0] == 0:
        raise ValueError("empty donor labels")
    reps = int(np.ceil(n / donor_y.shape[0]))
    return np.tile(donor_y, (reps, 1))[:n].astype(np.float32)


def embed_selected_targets(
    stream: SequenceSplit,
    selected: SequenceSplit,
) -> SequenceSplit:
    """Copy selected (possibly intentionally wrong) labels into stream rows."""
    if stream.source_recording_ids is None or selected.source_recording_ids is None:
        raise ValueError("embedding labels requires source_recording_ids provenance")
    lookup = {
        (str(source_id), int(alignment)): index
        for index, (source_id, alignment) in enumerate(
            zip(stream.source_recording_ids, stream.alignment_indices)
        )
    }
    stream_y = stream.y.copy()
    for selected_index, (source_id, alignment) in enumerate(
        zip(selected.source_recording_ids, selected.alignment_indices)
    ):
        key = (str(source_id), int(alignment))
        if key not in lookup:
            raise ValueError(f"selected row is absent from stream: {key}")
        stream_y[lookup[key]] = selected.y[selected_index]
    return replace(stream, y=stream_y)


def build_pipeline_config(summary_config, args):
    values = {}
    for key, value in summary_config.items():
        if key not in CfCTrainingConfig.__dataclass_fields__:
            continue
        if isinstance(value, list) and key in TUPLE_FIELDS:
            value = tuple(value)
        values[key] = value
    values["db2_dir"] = Path(values["db2_dir"])
    values["batch_size"] = args.batch_size
    values["device"] = args.device
    return CfCTrainingConfig(**values)


def load_pretrain_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    input_dim = len(cfg["emg_channels"]) * len(cfg["feature_order"])
    output_dim = infer_checkpoint_output_dim(ckpt["model_state_dict"])
    model = build_cfc_regressor(
        input_dim=input_dim, output_dim=output_dim,
        hidden_units=cfg.get("hidden_units", 512),
        model_family=cfg.get("model_family", "dense_cfc_linear"),
        cfc_dropout=cfg.get("cfc_dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def metrics_summary(eval_result):
    m = eval_result["metrics"]
    return {
        "mae_mean": float(m["mae_mean"]),
        "rmse_mean": float(m["rmse_mean"]),
        "r2_mean": float(m["r2_mean"]),
        "r2_by_target": [float(v) for v in m["r2_by_target"]],
        "normalized_mae_mean": float(m["normalized_mae_mean"]),
    }


def run_mode(
    mode,
    model,
    norm_support,
    norm_query,
    norm_target_stream,
    query_score_mask,
    norm_source_stream,
    source_score_mask,
    config,
    y_stats,
    device,
    args,
):
    entry = {"mode": mode, "checkpoint": str(BASE_CHECKPOINT),
             "ft_epochs": args.ft_epochs, "ft_patience": args.ft_patience,
             "atl_subject_weight": args.atl_weight}
    print(f"\n===== mode={mode} =====", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ft_train, ft_val = _split_sequence_split_temporal(norm_support, train_fraction=0.8)
    support_stream = embed_selected_targets(norm_target_stream, norm_support)
    ft_train_score_mask = provenance_score_mask(support_stream, ft_train)
    ft_val_score_mask = provenance_score_mask(support_stream, ft_val)
    t0 = time.time()
    adapted, ft_history, audit = fine_tune_head(
        model=model, support_split=support_stream, config=config, y_stats=y_stats,
        device=device, learning_rate=1e-4, epochs=args.ft_epochs,
        val_split=support_stream, early_stopping_patience=args.ft_patience,
        enable_atl=True, source_split=norm_source_stream,
        source_score_mask=source_score_mask,
        support_score_mask=ft_train_score_mask,
        val_score_mask=ft_val_score_mask,
        dd_lr=1e-4, cfc_atl_lr=1e-4, atl_subject_weight=args.atl_weight,
    )
    adapted.eval()
    entry["fine_tune_epochs_run"] = len(ft_history)
    entry["fine_tune_time_s"] = round(time.time() - t0, 1)
    entry["parameter_audit"] = audit
    cold = evaluate_split(adapted, norm_query, target_stats=y_stats,
                          batch_size=config.batch_size, device=device)
    chain = evaluate_chain(
        adapted,
        norm_target_stream,
        target_stats=y_stats,
        device=device,
        score_mask=query_score_mask,
    )
    entry["adapted_coldstart"] = metrics_summary(cold)
    entry["adapted_chain"] = metrics_summary(chain)
    entry["adapted_chain"]["h_norm_stats"] = chain["h_norm_stats"]
    print(f"  adapted cold: MAE={entry['adapted_coldstart']['mae_mean']:.3f} "
          f"R2={entry['adapted_coldstart']['r2_mean']:.3f}", flush=True)
    print(f"  adapted chain: MAE={entry['adapted_chain']['mae_mean']:.3f} "
          f"R2={entry['adapted_chain']['r2_mean']:.3f}", flush=True)
    return entry


def main():
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    print(f"device: {device}", flush=True)

    base = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    subjects = list(base["methodology"]["subjects"])
    if args.smoke:
        subjects = subjects[: args.smoke]
    target_subject = "S1"
    actions = tuple(base["methodology"]["actions"])
    config = build_pipeline_config(base["config"], args)
    print(f"subjects: {subjects} | actions={len(actions)} | features={config.feature_order} "
          f"seq={config.seq_len} hidden={config.hidden_units}", flush=True)

    # ---- 1. Load splits (same protocol as atl_ft30_rerun.py) ----
    t_data = time.time()
    source_splits = []
    source_stream_splits = []
    source_stream_masks = []
    target_support = None
    target_query = None
    target_stream = None
    donor_y = None
    for i, subject in enumerate(subjects, 1):
        t0 = time.time()
        split, stream = load_subject_splits(
            subject,
            ALL_EXERCISES,
            config,
            actions=actions,
        )
        train_split, test_split, _plan = select_repetition_split(
            split, actions=actions, train_repetitions_per_action=4,
            random_seed=args.seed + int(subject[1:]))
        if subject == target_subject:
            target_support, target_query = train_split, test_split
            target_stream = stream
        else:
            source_splits.append(train_split)
            source_stream_splits.append(stream)
            source_stream_masks.append(exact_query_score_mask(stream, train_split))
            if subject == "S2":
                donor_y = train_split.y.copy()
        print(f"  [{i}/{len(subjects)}] {subject}: {split.x.shape[0]} seq ({time.time() - t0:.1f}s)", flush=True)
    assert target_support is not None and target_query is not None and target_stream is not None
    assert donor_y is not None
    source_train = concat_splits(source_splits)
    source_stream = concat_splits(source_stream_splits)
    source_score_mask = np.concatenate(source_stream_masks)
    query_score_mask = exact_query_score_mask(target_stream, target_query)
    print(f"  source={source_train.x.shape[0]} support={target_support.x.shape[0]} "
          f"query={target_query.x.shape[0]} | load {time.time() - t_data:.1f}s", flush=True)

    # ---- 2. Normalizers fitted on source train only ----
    x_stats = fit_feature_normalizer(
        source_train.x.reshape(-1, source_train.x.shape[-1]),
        method=config.feature_normalization, mu=config.mu_law_mu)
    y_stats = fit_target_normalizer(source_train.y, method=config.target_normalization, mu=config.mu_law_mu)
    norm_source_stream = normalize_sequence_inputs(source_stream, x_stats=x_stats, y_stats=y_stats)
    norm_query = normalize_sequence_inputs(target_query, x_stats=x_stats, y_stats=y_stats)
    norm_target_stream = normalize_sequence_inputs(target_stream, x_stats=x_stats, y_stats=y_stats)
    norm_support_truth = normalize_sequence_inputs(target_support, x_stats=x_stats, y_stats=y_stats)

    # ---- 3. Mislabeled support variants (labels replaced pre-normalization) ----
    variants = {"truth": norm_support_truth}
    if "template" in args.modes:
        t0 = time.time()
        template_y = build_template_labels(target_support, source_train)
        support_tmpl = replace(target_support, y=template_y)
        variants["template"] = normalize_sequence_inputs(support_tmpl, x_stats=x_stats, y_stats=y_stats)
        print(f"  template labels built in {time.time() - t0:.1f}s", flush=True)
    if "index" in args.modes:
        index_y = build_index_labels(target_support, donor_y)
        support_idx = replace(target_support, y=index_y)
        variants["index"] = normalize_sequence_inputs(support_idx, x_stats=x_stats, y_stats=y_stats)

    # ---- 4. Zero-shot once, then ATL per label variant ----
    torch.manual_seed(args.seed)
    base_model = load_pretrain_model(BASE_CHECKPOINT, device)
    base_model.eval()
    zero_cold = evaluate_split(base_model, norm_query, target_stats=y_stats,
                               batch_size=config.batch_size, device=device)
    zero_chain = evaluate_chain(
        base_model,
        norm_target_stream,
        target_stats=y_stats,
        device=device,
        score_mask=query_score_mask,
    )
    results = {"zero_shot": {
        "mode": "zero_shot",
        "coldstart": metrics_summary(zero_cold),
        "chain": metrics_summary(zero_chain),
        "chain_h_norm_stats": zero_chain["h_norm_stats"],
    }}
    print(f"  zero_shot: cold MAE={results['zero_shot']['coldstart']['mae_mean']:.3f} "
          f"R2={results['zero_shot']['coldstart']['r2_mean']:.3f} | "
          f"chain MAE={results['zero_shot']['chain']['mae_mean']:.3f} "
          f"R2={results['zero_shot']['chain']['r2_mean']:.3f}", flush=True)

    modes = [m for m in args.modes.split(",") if m in variants]
    for mode in modes:
        results[mode] = run_mode(
            mode,
            base_model,
            variants[mode],
            norm_query,
            norm_target_stream,
            query_score_mask,
            norm_source_stream,
            source_score_mask,
            config,
            y_stats,
            device,
            args,
        )
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- 5. Comparison table ----
    print("\n===== comparison (all eval on S1 TRUE query labels) =====")
    print(f"{'label source':<12} {'cold MAE':<10} {'cold R2':<9} {'chain MAE':<10} {'chain R2':<9}")
    z = results["zero_shot"]
    print(f"{'zero_shot':<12} {z['coldstart']['mae_mean']:<10.3f} {z['coldstart']['r2_mean']:<9.3f} "
          f"{z['chain']['mae_mean']:<10.3f} {z['chain']['r2_mean']:<9.3f}")
    for mode in modes:
        e = results[mode]
        print(f"{mode:<12} {e['adapted_coldstart']['mae_mean']:<10.3f} {e['adapted_coldstart']['r2_mean']:<9.3f} "
              f"{e['adapted_chain']['mae_mean']:<10.3f} {e['adapted_chain']['r2_mean']:<9.3f}")
    print(f"\nresults: {results_path}")


if __name__ == "__main__":
    main()
