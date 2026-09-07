"""ATL chain-R2 A/B experiment: stateless vs stateful pretrain, ATL fine-tune, deployment-realistic eval.

Motivation (user question): CfC's closed form decays the influence of the initial
hidden state, so training stateless while deploying stateful (persistent hidden
state, as the firmware runs ``cfc_single_step``) is claimed to be harmless.  The
prior A/B summaries compared cold-start val_mae (val split) against chain MAE
(test split), which confounds split difficulty with deployment mode.  This
script measures both evaluation modes on the SAME test split, after the SAME
ATL fine-tuning, for both A/B training regimes.

Flow (mirrors run_db2_paper_cfc_finetune.run_protocol data pipeline):
  1. Load the A/B subject list + splits once (from 8ch_h512_stateless_ab summary).
  2. Fit feature/target normalizers on source train.
  3. For each pretrain checkpoint (stateless, stateful, h512 seq8):
       - zero-shot: cold-start (evaluate_split) + chain (evaluate_chain) on S1 query
       - ATL fine-tune on S1 support (enable_atl=True, source = other subjects)
       - adapted: cold-start + chain on S1 query
  4. Incrementally write results JSON; print a comparison table.

Usage:
  python atl_chain_r2_ab.py --smoke 2 --ft-epochs 2        # smoke: 2 subjects, 2 ATL epochs
  python atl_chain_r2_ab.py --ft-epochs 30 --ft-patience 5 # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
AB_SUMMARY = (
    REPO_ROOT
    / "log/db2_paper_cfc_finetune/8ch_h512_stateless_ab/summary.json"
)
CHECKPOINTS = {
    "stateless": (
        REPO_ROOT
        / "log/db2_paper_cfc_finetune/8ch_h512_stateless_ab/checkpoints/S1_dense_cfc_pretrain.pt"
    ),
    "stateful": (
        REPO_ROOT
        / "log/db2_paper_cfc_finetune/8ch_h512_stateful_ab/checkpoints/S1_dense_cfc_pretrain.pt"
    ),
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "log/db2_paper_cfc_finetune/atl_chain_r2_ab_20260808"

TUPLE_FIELDS = {"emg_channels", "feature_order", "target_columns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", type=int, default=0,
                        help="If >0, use only this many subjects (for smoke runs)")
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-patience", type=int, default=5)
    parser.add_argument("--atl-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_pipeline_config(summary_config: dict[str, Any], args: argparse.Namespace) -> CfCTrainingConfig:
    """Reconstruct the training config from the A/B summary so the data
    pipeline is identical to the runs that produced the checkpoints."""
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


def load_pretrain_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
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
    return model.to(device), cfg


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
    print(f"device: {device}")
    print(f"output: {output_dir}")

    ab = json.loads(AB_SUMMARY.read_text(encoding="utf-8"))
    subjects = list(ab["methodology"]["subjects"])
    if args.smoke:
        subjects = subjects[: args.smoke]
    target_subject = "S1"
    exercises = ALL_EXERCISES
    actions = tuple(ab["methodology"]["actions"])
    config = build_pipeline_config(ab["config"], args)
    print(f"subjects: {subjects} (target={target_subject})")
    print(f"actions: {len(actions)} | exercises: {exercises} | seq_len={config.seq_len} "
          f"| hidden={config.hidden_units} | batch={config.batch_size}")

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
        print(f"  [{i}/{len(subjects)}] {subject}: {split.x.shape[0]} seq ({time.time() - t0:.1f}s)",
              flush=True)
    assert target_support is not None and target_query is not None and target_stream is not None
    source_train = concat_splits(source_train_splits)
    source_stream = concat_splits(source_stream_splits)
    source_score_mask = np.concatenate(source_stream_masks)
    query_score_mask = exact_query_score_mask(target_stream, target_query)
    print(f"  source_train={source_train.x.shape[0]} seq | "
          f"support={target_support.x.shape[0]} | query={target_query.x.shape[0]} "
          f"| load {time.time() - t_data:.1f}s", flush=True)

    x_stats = fit_feature_normalizer(
        source_train.x.reshape(-1, source_train.x.shape[-1]),
        method=config.feature_normalization,
        mu=config.mu_law_mu,
    )
    y_stats = fit_target_normalizer(source_train.y, method=config.target_normalization, mu=config.mu_law_mu)

    norm_source_stream = normalize_sequence_inputs(source_stream, x_stats=x_stats, y_stats=y_stats)
    norm_support = normalize_sequence_inputs(target_support, x_stats=x_stats, y_stats=y_stats)
    norm_query = normalize_sequence_inputs(target_query, x_stats=x_stats, y_stats=y_stats)
    norm_target_stream = normalize_sequence_inputs(target_stream, x_stats=x_stats, y_stats=y_stats)

    # ---- 2. Per-model: zero-shot eval, ATL fine-tune, adapted eval ----
    results: dict[str, Any] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    for name, ckpt_path in CHECKPOINTS.items():
        if name in results and not args.smoke:
            print(f"[{name}] already in results, skipping", flush=True)
            continue
        print(f"\n===== {name} checkpoint: {ckpt_path} =====", flush=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        t0 = time.time()
        model, ckpt_cfg = load_pretrain_model(ckpt_path, device)
        model.eval()
        print(f"  loaded (hidden={ckpt_cfg.get('hidden_units')}, input_dim="
              f"{len(ckpt_cfg['emg_channels']) * len(ckpt_cfg['feature_order'])}) "
              f"in {time.time() - t0:.1f}s", flush=True)

        entry: dict[str, Any] = {"checkpoint": str(ckpt_path)}

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
        adapted, ft_history, _audit = fine_tune_head(
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
        print(f"  ATL done: {len(ft_history)} epochs in {entry['fine_tune_time_s']}s", flush=True)

        evaluate_both("adapted", adapted, entry)

        entry["total_time_s"] = round(time.time() - t0, 1)
        results[name] = entry
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  saved {results_path}", flush=True)

    print("\n===== comparison (S1 query split, same data for both modes) =====")
    print(f"{'model':<10} {'phase':<9} {'coldstart MAE':<14} {'coldstart R2':<12} {'chain MAE':<10} {'chain R2':<8}")
    for name in CHECKPOINTS:
        if name not in results:
            continue
        for phase in ("zero_shot", "adapted"):
            e = results[name].get(f"{phase}_coldstart")
            c = results[name].get(f"{phase}_chain")
            if e and c:
                print(f"{name:<10} {phase:<9} {e['mae_mean']:<14.3f} {e['r2_mean']:<12.3f} "
                      f"{c['mae_mean']:<10.3f} {c['r2_mean']:<8.3f}")
    print(f"\nresults: {results_path}")


if __name__ == "__main__":
    main()
