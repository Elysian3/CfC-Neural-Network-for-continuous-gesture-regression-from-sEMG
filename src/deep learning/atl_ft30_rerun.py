"""ATL fine-tune rerun with more epochs on the rmszcwl base model.

The original run (8ch_rmszcwl_h512_b1024_atl_w10_lr1e-4[_2]) capped ATL
fine-tuning at 10 epochs (run_db2_paper_cfc_finetune --fine-tune-epochs default
= 10; early stopping never triggered since val_mae was still improving at
epoch 10).  This script re-runs the SAME pipeline on the SAME base checkpoint
(the 600-epoch pretrain from the _2 run) with --ft-epochs 30 and re-evaluates
zero-shot/adapted x coldstart/chain on the S1 query split, then compares
against the original 10-epoch numbers.

Pipeline mirrors run_db2_paper_cfc_finetune.run_protocol exactly:
  - subject list / split protocol from the original summary (repetition split,
    seed 42 + subject index, 4 train reps per action)
  - mu-law feature/target normalizers fit on source_train (x_stats verified
    against the saved feature_normalization.npz artifact)
  - fine_tune_head(enable_atl=True, atl_subject_weight=10, dd_lr=1e-4,
    cfc_atl_lr=1e-4, lr=1e-4, batch=1024)
  - evaluation: evaluate_split (coldstart) + evaluate_chain (stateful,
    mirrors firmware) on S1 query

Usage:
  python atl_ft30_rerun.py --smoke 2 --ft-epochs 2        # smoke
  python atl_ft30_rerun.py --ft-epochs 30 --ft-patience 5 # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    select_repetition_split,
)
from train import (
    CfCTrainingConfig,
    build_cfc_regressor,
    evaluate_chain,
    evaluate_split,
    fit_target_normalizer,
    normalize_sequence_inputs,
    resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_RUN_DIR = (
    REPO_ROOT / "log/db2_paper_cfc_finetune/8ch_rmszcwl_h512_b1024_atl_w10_lr1e-4_2"
)
BASE_CHECKPOINT = BASE_RUN_DIR / "checkpoints/S1_dense_cfc_pretrain.pt"
BASE_SUMMARY = BASE_RUN_DIR / "summary.json"
NORMALIZER_NPZ = BASE_RUN_DIR / "feature_normalization.npz"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "log/db2_paper_cfc_finetune/8ch_rmszcwl_h512_b1024_atl_w10_lr1e-4_2_ft30"
)

TUPLE_FIELDS = {"emg_channels", "feature_order", "target_columns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", type=int, default=0,
                        help="If >0, use only this many subjects (for smoke runs)")
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-patience", type=int, default=5)
    parser.add_argument("--atl-weight", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_pipeline_config(summary_config: dict[str, Any], args: argparse.Namespace) -> CfCTrainingConfig:
    values: dict[str, Any] = {}
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


def load_pretrain_model(path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    input_dim = len(cfg["emg_channels"]) * len(cfg["feature_order"])
    output_dim = infer_checkpoint_output_dim(ckpt["model_state_dict"])
    model = build_cfc_regressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_units=cfg.get("hidden_units", 512),
        model_family=cfg.get("model_family", "dense_cfc_linear"),
        cfc_dropout=cfg.get("cfc_dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def metrics_summary(eval_result: dict[str, Any]) -> dict[str, Any]:
    m = eval_result["metrics"]
    summary = {
        "mae_mean": float(m["mae_mean"]),
        "rmse_mean": float(m["rmse_mean"]),
        "r2_mean": float(m["r2_mean"]),
    }
    if "r2_by_target" in m:
        summary["r2_by_target"] = [float(v) for v in m["r2_by_target"]]
    if "normalized_mae_mean" in m:
        summary["normalized_mae_mean"] = float(m["normalized_mae_mean"])
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    print(f"device: {device}", flush=True)
    print(f"base checkpoint: {BASE_CHECKPOINT}", flush=True)
    print(f"output: {output_dir}", flush=True)

    base = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    subjects = list(base["methodology"]["subjects"])
    if args.smoke:
        subjects = subjects[: args.smoke]
    target_subject = "S1"
    exercises = ALL_EXERCISES
    actions = tuple(base["methodology"]["actions"])
    config = build_pipeline_config(base["config"], args)
    print(f"subjects: {subjects} (target={target_subject})", flush=True)
    print(f"features={config.feature_order} | seq_len={config.seq_len} | "
          f"hidden={config.hidden_units} | batch={config.batch_size} | "
          f"FT epochs={args.ft_epochs} patience={args.ft_patience} "
          f"atl_w={args.atl_weight}", flush=True)

    # ---- 1. Load splits once (mirrors run_protocol) ----
    t_data = time.time()
    source_train_splits: list = []
    source_stream_splits: list = []
    source_stream_masks: list[np.ndarray] = []
    target_support = None
    target_query = None
    target_stream = None
    for i, subject in enumerate(subjects, 1):
        t0 = time.time()
        split, stream = load_subject_splits(
            subject,
            exercises,
            config,
            actions=actions,
        )
        train_split, test_split, _plan = select_repetition_split(
            split,
            actions=actions,
            train_repetitions_per_action=4,
            random_seed=args.seed + int(subject[1:]),
        )
        if subject == target_subject:
            target_support, target_query = train_split, test_split
            target_stream = stream
        else:
            source_train_splits.append(train_split)
            source_stream_splits.append(stream)
            source_stream_masks.append(exact_query_score_mask(stream, train_split))
        print(f"  [{i}/{len(subjects)}] {subject}: {split.x.shape[0]} seq "
              f"({time.time() - t0:.1f}s)", flush=True)
    assert target_support is not None and target_query is not None and target_stream is not None
    source_train = concat_splits(source_train_splits)
    source_stream = concat_splits(source_stream_splits)
    source_score_mask = np.concatenate(source_stream_masks)
    query_score_mask = exact_query_score_mask(target_stream, target_query)
    print(f"  source_train={source_train.x.shape[0]} | support={target_support.x.shape[0]} "
          f"| query={target_query.x.shape[0]} | load {time.time() - t_data:.1f}s", flush=True)

    # ---- 2. Normalizers (verify x_stats against saved artifact) ----
    x_stats = fit_feature_normalizer(
        source_train.x.reshape(-1, source_train.x.shape[-1]),
        method=config.feature_normalization,
        mu=config.mu_law_mu,
    )
    y_stats = fit_target_normalizer(source_train.y, method=config.target_normalization, mu=config.mu_law_mu)
    if not args.smoke:
        saved = np.load(NORMALIZER_NPZ, allow_pickle=True)
        for key in ("center", "scale", "mu"):
            diff = float(np.max(np.abs(x_stats[key] - saved[key])))
            print(f"  x_stats[{key}] vs npz: max|diff|={diff:.3e}", flush=True)

    norm_source_stream = normalize_sequence_inputs(source_stream, x_stats=x_stats, y_stats=y_stats)
    norm_support = normalize_sequence_inputs(target_support, x_stats=x_stats, y_stats=y_stats)
    norm_query = normalize_sequence_inputs(target_query, x_stats=x_stats, y_stats=y_stats)
    norm_target_stream = normalize_sequence_inputs(target_stream, x_stats=x_stats, y_stats=y_stats)
    del source_train, source_stream, target_support, target_query, target_stream

    # ---- 3. Zero-shot eval, ATL fine-tune (30 ep), adapted eval ----
    results: dict[str, Any] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    print(f"\n===== base: {BASE_CHECKPOINT.name} =====", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t_start = time.time()

    model = load_pretrain_model(BASE_CHECKPOINT, device)
    model.eval()
    entry: dict[str, Any] = {"checkpoint": str(BASE_CHECKPOINT), "ft_epochs": args.ft_epochs,
                             "ft_patience": args.ft_patience, "atl_subject_weight": args.atl_weight}

    def evaluate_both(phase: str, m: torch.nn.Module, out: dict[str, Any]) -> None:
        out[f"{phase}_coldstart"] = metrics_summary(
            evaluate_split(m, norm_query, target_stats=y_stats,
                           batch_size=config.batch_size, device=device))
        chain = evaluate_chain(
            m,
            norm_target_stream,
            target_stats=y_stats,
            device=device,
            score_mask=query_score_mask,
        )
        out[f"{phase}_chain"] = metrics_summary(chain)
        out[f"{phase}_chain"]["h_norm_stats"] = chain["h_norm_stats"]
        print(f"  {phase}: coldstart MAE={out[f'{phase}_coldstart']['mae_mean']:.3f} "
              f"R2={out[f'{phase}_coldstart']['r2_mean']:.3f} | "
              f"chain MAE={out[f'{phase}_chain']['mae_mean']:.3f} "
              f"R2={out[f'{phase}_chain']['r2_mean']:.3f}", flush=True)

    evaluate_both("zero_shot", model, entry)

    ft_train, ft_val = _split_sequence_split_temporal(norm_support, train_fraction=0.8)
    ft_train_score_mask = exact_query_score_mask(norm_target_stream, ft_train)
    ft_val_score_mask = exact_query_score_mask(norm_target_stream, ft_val)
    t_ft = time.time()
    adapted, ft_history, audit = fine_tune_head(
        model=model,
        support_split=norm_target_stream,
        config=config,
        y_stats=y_stats,
        device=device,
        learning_rate=1e-4,
        epochs=args.ft_epochs,
        val_split=norm_target_stream,
        early_stopping_patience=args.ft_patience,
        enable_atl=True,
        source_split=norm_source_stream,
        source_score_mask=source_score_mask,
        support_score_mask=ft_train_score_mask,
        val_score_mask=ft_val_score_mask,
        dd_lr=1e-4,
        cfc_atl_lr=1e-4,
        atl_subject_weight=args.atl_weight,
    )
    adapted.eval()
    entry["fine_tune_epochs_run"] = len(ft_history)
    entry["fine_tune_time_s"] = round(time.time() - t_ft, 1)
    ft_curve = [(int(h["epoch"]), round(float(h["val_mae"]), 3)) for h in ft_history]
    print(f"  ATL done: {len(ft_history)} epochs in {entry['fine_tune_time_s']}s", flush=True)
    print(f"  FT val_mae curve: {ft_curve}", flush=True)

    evaluate_both("adapted", adapted, entry)
    entry["total_time_s"] = round(time.time() - t_start, 1)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": adapted.state_dict(),
            "config": asdict(config),
            "target_contract": base["methodology"].get("target_contract"),
            "feature_normalization_stats": x_stats,
            "target_normalization_stats": y_stats,
            "audit": audit,
            "ft_history": ft_history,
        },
        ckpt_dir / "S1_dense_cfc_head_ft30.pt",
    )
    results["ft30"] = entry
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  saved {results_path}", flush=True)

    # ---- 4. Comparison vs original 10-epoch run ----
    orig = {
        "zero_shot_coldstart": metrics_summary(base["zero_shot_test_metrics"]),
        "zero_shot_chain": metrics_summary(base["chain_metrics_zero_shot"]),
        "adapted_coldstart": metrics_summary(base["adapted_test_metrics"]),
        "adapted_chain": metrics_summary(base["chain_metrics_adapted"]),
    }
    print("\n===== S1 query split: 10-epoch (original) vs 30-epoch (this run) =====")
    print(f"{'phase':<20} {'10ep MAE':<10} {'10ep R2':<9} {'30ep MAE':<10} {'30ep R2':<9}")
    for phase, label in (("zero_shot_coldstart", "zero-shot coldstart"),
                         ("zero_shot_chain", "zero-shot chain"),
                         ("adapted_coldstart", "adapted coldstart"),
                         ("adapted_chain", "adapted chain")):
        o, n = orig[phase], entry[phase]
        print(f"{label:<20} {o['mae_mean']:<10.3f} {o['r2_mean']:<9.3f} "
              f"{n['mae_mean']:<10.3f} {n['r2_mean']:<9.3f}")
    print(f"\nresults: {results_path}")


if __name__ == "__main__":
    main()
