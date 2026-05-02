from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_db2_single_subject import (
    DEFAULT_OUTPUT_DIR,
    REPO_ROOT,
    make_jsonable,
    run_single_subject_experiment,
)


PASS_R2_THRESHOLD = 0.5
MIN_STABLE_VAL_R2 = 0.3
MAX_STABLE_TEST_VAL_GAP = 0.25
REQUIRED_SEMANTIC_DOFS = ("wrist_flexion_extension", "finger_flexion")


@dataclass(frozen=True)
class SemanticCandidate:
    semantic_dof: str
    target_source: str
    target_column: int
    offset_samples: int
    semantic_label: str
    semantic_confidence: str
    role: str

    @property
    def slug(self) -> str:
        dof_slug = self.semantic_dof.replace("/", "_").replace(" ", "_")
        return (
            f"{dof_slug}_{self.target_source}_target{self.target_column}"
            f"_offset{self.offset_samples}"
        )


def parse_int_list(raw_value: str) -> list[int]:
    """Parse a comma-separated integer list used by CLI matrix options."""
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_candidate_matrix(
    *,
    offsets: list[int],
    finger_columns: list[int],
    include_wrist_glove_fallback: bool,
) -> list[SemanticCandidate]:
    """Define semantic candidates before any test metric is inspected."""
    candidates: list[SemanticCandidate] = []

    for offset in offsets:
        for column in (0, 1):
            candidates.append(
                SemanticCandidate(
                    semantic_dof="wrist_flexion_extension",
                    target_source="inclin",
                    target_column=column,
                    offset_samples=offset,
                    semantic_label=f"inclin axis {column}",
                    semantic_confidence="medium: inclinometer source is the strongest available wrist-orientation proxy",
                    role="primary",
                )
            )

        for column in finger_columns:
            candidates.append(
                SemanticCandidate(
                    semantic_dof="finger_flexion",
                    target_source="glove",
                    target_column=column,
                    offset_samples=offset,
                    semantic_label=f"glove channel {column}",
                    semantic_confidence="low: DB2 glove channel is unlabeled locally; retained as finger-flexion candidate by source family",
                    role="primary",
                )
            )

        if include_wrist_glove_fallback:
            for column in finger_columns:
                candidates.append(
                    SemanticCandidate(
                        semantic_dof="wrist_flexion_extension",
                        target_source="glove",
                        target_column=column,
                        offset_samples=offset,
                        semantic_label=f"glove fallback channel {column}",
                        semantic_confidence="low: fallback only if inclin wrist candidates fail",
                        role="fallback",
                    )
                )

    return candidates


def candidate_output_dir(base_output_dir: Path, candidate: SemanticCandidate) -> Path:
    return base_output_dir / candidate.slug


def summary_path_for(base_output_dir: Path, candidate: SemanticCandidate) -> Path:
    return candidate_output_dir(base_output_dir, candidate) / "summary.json"


def metric_value(summary: dict[str, Any], split_name: str, metric_name: str) -> float | None:
    metrics = summary.get("metrics", {}).get(split_name, {})
    value = metrics.get(metric_name)
    if value is None:
        return None
    return float(value)


def best_epoch_value(summary: dict[str, Any]) -> float | None:
    value = summary.get("best_epoch", {}).get("epoch")
    if value is None:
        return None
    return float(value)


def is_stable(val_r2: float | None, test_r2: float | None) -> bool:
    if val_r2 is None or test_r2 is None:
        return False
    return val_r2 >= MIN_STABLE_VAL_R2 or (test_r2 - val_r2) <= MAX_STABLE_TEST_VAL_GAP


def summarize_candidate(
    candidate: SemanticCandidate,
    *,
    subject: str,
    split_strategy: str,
    seed: int,
    base_output_dir: Path,
) -> dict[str, Any]:
    path = summary_path_for(base_output_dir, candidate)
    row: dict[str, Any] = {
        **asdict(candidate),
        "subject": subject,
        "split_strategy": split_strategy,
        "seed": seed,
        "seq_len": None,
        "target_offset_samples": candidate.offset_samples,
        "summary_path": str(path),
        "prediction_plot": None,
        "best_epoch": None,
        "train_mae": None,
        "train_rmse": None,
        "train_r2": None,
        "val_mae": None,
        "val_rmse": None,
        "val_r2": None,
        "test_mae": None,
        "test_rmse": None,
        "test_r2": None,
        "stable": False,
        "selected_by_validation": False,
        "decision": "not_run",
        "failure_reason": "summary not found",
    }

    if not path.exists():
        return row

    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    row["prediction_plot"] = summary.get("artifacts", {}).get("prediction_plot")
    row["best_epoch"] = best_epoch_value(summary)
    row["seq_len"] = summary.get("config", {}).get("seq_len")
    for split_name in ("train", "val", "test"):
        row[f"{split_name}_mae"] = metric_value(summary, split_name, "mae_mean")
        row[f"{split_name}_rmse"] = metric_value(summary, split_name, "rmse_mean")
        row[f"{split_name}_r2"] = metric_value(summary, split_name, "r2_mean")

    row["stable"] = is_stable(row["val_r2"], row["test_r2"])
    row["decision"] = "evaluated"
    row["failure_reason"] = ""
    return row


def apply_validation_first_decisions(rows: list[dict[str, Any]]) -> None:
    """Select by validation R2 first, then use test R2 only for confirmation."""
    for semantic_dof in sorted({row["semantic_dof"] for row in rows}):
        evaluated = [
            row
            for row in rows
            if row["semantic_dof"] == semantic_dof and row["val_r2"] is not None
        ]
        if not evaluated:
            continue

        selected = select_candidate_group([row for row in evaluated if row["role"] == "primary"])
        if selected is None:
            selected = select_candidate_group(evaluated)
        if selected is not None and selected["decision"] != "pass":
            fallback = select_candidate_group([row for row in evaluated if row["role"] == "fallback"])
            if fallback is not None:
                selected["selected_by_validation"] = False
                selected = fallback
                if fallback["decision"] == "pass":
                    fallback["decision"] = "fallback_pass"
                    fallback["failure_reason"] = "primary candidates failed; fallback passed with lower semantic confidence"

        if selected is None:
            continue
        for row in evaluated:
            if row is selected:
                continue
            if row["decision"] == "evaluated":
                row["decision"] = "rejected"
                row["failure_reason"] = "not selected by validation-first ranking"


def select_candidate_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select one group by validation R2, then classify by test confirmation."""
    if not rows:
        return None

    selected = max(rows, key=lambda row: row["val_r2"])
    selected["selected_by_validation"] = True

    test_r2 = selected["test_r2"]
    if test_r2 is not None and test_r2 >= PASS_R2_THRESHOLD and selected["stable"]:
        selected["decision"] = "pass"
    elif test_r2 is not None and test_r2 >= PASS_R2_THRESHOLD:
        selected["decision"] = "unstable"
        selected["failure_reason"] = "test R2 passed but validation stability rule failed"
    else:
        selected["decision"] = "fail"
        selected["failure_reason"] = "validation-selected candidate did not confirm on test R2"
    return selected


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(payload), handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_overall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether the semantic DoF problem is solved under strict gates."""
    selected_rows = [row for row in rows if row["selected_by_validation"]]
    selected_by_dof = {row["semantic_dof"]: row for row in selected_rows}

    dof_status: dict[str, Any] = {}
    for semantic_dof in REQUIRED_SEMANTIC_DOFS:
        row = selected_by_dof.get(semantic_dof)
        if row is None:
            dof_status[semantic_dof] = {
                "decision": "missing",
                "r2_threshold_met": False,
                "strict_semantic_pass": False,
                "failure_reason": "no validation-selected candidate",
            }
            continue

        decision = row["decision"]
        dof_status[semantic_dof] = {
            "decision": decision,
            "target_source": row["target_source"],
            "target_column": row["target_column"],
            "target_offset_samples": row["target_offset_samples"],
            "val_r2": row["val_r2"],
            "test_r2": row["test_r2"],
            "stable": row["stable"],
            "semantic_confidence": row["semantic_confidence"],
            "r2_threshold_met": decision in ("pass", "fallback_pass"),
            "strict_semantic_pass": decision == "pass",
            "failure_reason": row["failure_reason"],
        }

    strict_solved_today = all(
        status["strict_semantic_pass"] for status in dof_status.values()
    )
    threshold_met_with_fallback = all(
        status["r2_threshold_met"] for status in dof_status.values()
    )

    if strict_solved_today:
        conclusion = "solved"
    elif threshold_met_with_fallback:
        conclusion = "not_solved_strictly_fallback_only"
    else:
        conclusion = "not_solved"

    return {
        "required_semantic_dofs": list(REQUIRED_SEMANTIC_DOFS),
        "strict_solved_today": strict_solved_today,
        "threshold_met_with_fallback": threshold_met_with_fallback,
        "conclusion": conclusion,
        "dof_status": dof_status,
        "selection_rule": "semantic gate, validation-first selection, test-only confirmation",
        "stability_rule": (
            f"stable when val_r2 >= {MIN_STABLE_VAL_R2} or "
            f"test_r2 - val_r2 <= {MAX_STABLE_TEST_VAL_GAP}"
        ),
        "next_step_if_not_solved": (
            "Do not promote a wrist default from glove fallback alone; inspect DB2 "
            "target semantics/plots and collect or map a stronger wrist-flexion label."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or summarize semantic DoF candidates with validation-first selection."
    )
    parser.add_argument("--db2-dir", type=Path, default=REPO_ROOT / "src" / "data" / "DB2")
    parser.add_argument("--subject", type=str, default="S1")
    parser.add_argument("--offsets", type=parse_int_list, default=[0, 200])
    parser.add_argument("--finger-columns", type=parse_int_list, default=list(range(22)))
    parser.add_argument("--include-wrist-glove-fallback", action="store_true")
    parser.add_argument("--run", action="store_true", help="Train missing candidate runs before summarizing.")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "semantic_dof_screen",
        help="Directory for per-candidate runs and matrix outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_candidate_matrix(
        offsets=args.offsets,
        finger_columns=args.finger_columns,
        include_wrist_glove_fallback=args.include_wrist_glove_fallback,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "candidate_matrix.json"
    write_json(matrix_path, [asdict(candidate) for candidate in candidates])

    if args.run:
        for candidate in candidates:
            path = summary_path_for(args.output_dir, candidate)
            if path.exists():
                continue
            run_args = argparse.Namespace(
                db2_dir=args.db2_dir,
                subject=args.subject,
                target_source=candidate.target_source,
                target_column=candidate.target_column,
                target_offset_samples=candidate.offset_samples,
                zc_threshold=None,
                ssc_threshold=None,
                feature_normalization=None,
                target_normalization=None,
                max_epochs=args.max_epochs,
                early_stopping_patience=args.early_stopping_patience,
                max_windows_per_file=args.max_windows_per_file,
                device=args.device,
                output_dir=candidate_output_dir(args.output_dir, candidate),
            )
            run_single_subject_experiment(run_args)

    rows = [
        summarize_candidate(
            candidate,
            subject=args.subject,
            split_strategy="blocked_time",
            seed=42,
            base_output_dir=args.output_dir,
        )
        for candidate in candidates
    ]
    apply_validation_first_decisions(rows)

    write_json(args.output_dir / "semantic_dof_results.json", rows)
    write_json(args.output_dir / "semantic_dof_summary.json", build_overall_summary(rows))
    write_csv(args.output_dir / "semantic_dof_results.csv", rows)
    print(f"Saved candidate matrix to: {matrix_path}")
    print(f"Saved semantic DoF results to: {args.output_dir / 'semantic_dof_results.json'}")


if __name__ == "__main__":
    main()
