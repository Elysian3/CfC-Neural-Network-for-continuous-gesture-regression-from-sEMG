import subprocess, sys
from pathlib import Path

ALL_SUBJECTS = ["S1","S2","S3","S5","S7","S8","S9","S10","S11","S12","S13","S14","S15","S16","S17","S18","S19","S20","S21","S22","S23","S24","S27","S28","S29","S30","S31","S32"]
SUBJECTS_STR = ",".join(ALL_SUBJECTS)
VENV_PYTHON = str(Path("D:/Project Antikythera/.venv/Scripts/python.exe"))

for i, target in enumerate(ALL_SUBJECTS):
    out_dir = f"log/db2_paper_cfc_finetune/loo_{target}_ft1e-5_20260524"
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(ALL_SUBJECTS)}] Target: {target}")
    print(f"{'='*60}")
    cmd = [
        VENV_PYTHON, "-u",
        "src/deep learning/run_db2_paper_cfc_finetune.py",
        "--max-epochs", "100",
        "--early-stopping-patience", "20",
        "--fine-tune-epochs", "30",
        "--fine-tune-learning-rate", "1e-5",
        "--target-subject", target,
        "--subjects", SUBJECTS_STR,
        "--output-dir", out_dir,
    ]
    result = subprocess.run(cmd, cwd="D:/Project Antikythera")
    if result.returncode != 0:
        print(f"FAILED: {target} (exit {result.returncode})")
    else:
        print(f"DONE: {target} -> {out_dir}")
print("\nALL 28 DONE")
