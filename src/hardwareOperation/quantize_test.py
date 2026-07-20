"""Compatibility CLI for the DenseCfC hardware preflight.

The deployability check now lives in hardware_preflight.py because the current
hardware target is RMS-only DenseCfC, not AutoNCP.
"""

from hardware_preflight import (
    compute_size_stats,
    dequantize_weights,
    main,
    quantize_weights_int8,
    run_preflight,
)


__all__ = [
    "compute_size_stats",
    "dequantize_weights",
    "main",
    "quantize_weights_int8",
    "run_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
