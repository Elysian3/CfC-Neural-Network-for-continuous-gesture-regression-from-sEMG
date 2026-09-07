"""US-005: end-to-end speed profile of optimization combinations.

Real shapes: input 16 (8ch x rms,zc), hidden 512, output 5, seq 8, batch 1024.
Combinations: eager / +cuda-graph / +gpu-resident / +both.
Reports per-batch wall vs GPU-active time and scaled epoch time for the real
training set (462,970 sequences).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_deep_learning_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, _deep_learning_dir)
sys.path.insert(0, str(Path(_deep_learning_dir).parent / "dataflow"))

from train import (
    GpuResidentBatches,
    SequenceRegressionDataset,
    build_cfc_regressor,
    build_graphed_cfc,
    resolve_device,
)

REAL_N_SEQS = 462_970


def make_iter(x, y, batch_size, device, *, gpu_resident: bool, generator_seed: int):
    if gpu_resident:
        return GpuResidentBatches(
            x, y, batch_size=batch_size, device=device,
            generator=torch.Generator(device="cpu").manual_seed(generator_seed),
            drop_last=True,
        )
    dataset = SequenceRegressionDataset(x, y)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
    )


def run_case(device, x, y, *, hidden: int, batch_size: int, use_graph: bool, use_gpu_resident: bool) -> None:
    torch.manual_seed(42)
    model = build_cfc_regressor(
        input_dim=x.shape[-1], output_dim=y.shape[-1], hidden_units=hidden,
        model_family="dense_cfc_linear", cfc_dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    graphed, orig_fwd, graph_fwd = None, None, None
    if use_graph:
        graphed, orig_fwd, graph_fwd = build_graphed_cfc(
            model, torch.zeros(batch_size, x.shape[1], x.shape[-1], device=device)
        )

    it = make_iter(x, y, batch_size, device, gpu_resident=use_gpu_resident, generator_seed=42)

    def one_epoch() -> None:
        model.train()
        for xb, yb in it:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if graphed is not None:
                ys, _ = graphed(xb)
                pred = model.head(model.dropout(ys[:, -1, :]))
            else:
                pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    one_epoch()  # warmup (also triggers graph capture)
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start_ev.record()
    one_epoch()
    end_ev.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    gpu_ms = start_ev.elapsed_time(end_ev)

    n_batches = len(it)
    per_wall = wall_ms / n_batches
    per_gpu = gpu_ms / n_batches
    epoch_s = per_wall * (REAL_N_SEQS / batch_size) / 1e3
    epoch_gpu_s = per_gpu * (REAL_N_SEQS / batch_size) / 1e3
    tag = f"graph={'Y' if use_graph else 'N'} resident={'Y' if use_gpu_resident else 'N'}"
    print(
        f"[{tag}] batches={n_batches:4d} | wall {per_wall:6.2f} ms/batch | "
        f"gpu-active {per_gpu:6.2f} ms/batch | cpu-side {max(per_wall - per_gpu, 0):6.2f} ms | "
        f"scaled epoch: wall {epoch_s:6.1f} s | gpu {epoch_gpu_s:6.1f} s"
    )


def main() -> None:
    device = resolve_device("cuda")
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"config: input 16 (8ch rms,zc), hidden 512, out 5, seq 8, batch 1024, "
          f"scaled to {REAL_N_SEQS} seqs/epoch")

    n_seq = 50_000  # 1/10 of real size; per-batch times are size-independent
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_seq, 8, 16)).astype(np.float32)
    y = rng.standard_normal((n_seq, 5)).astype(np.float32)

    for use_graph, use_resident in ((False, False), (True, False), (False, True), (True, True)):
        run_case(device, x, y, hidden=512, batch_size=1024,
                 use_graph=use_graph, use_gpu_resident=use_resident)


if __name__ == "__main__":
    main()
