// Legacy PC-test fixture only. Production firmware must replace this file by
// running export_weights.py on the exact trained checkpoint and normalization.
// cfc_inference.c deliberately rejects this fixture unless the PC test build
// defines CFC_ALLOW_NORMALIZED_OUTPUT_FIXTURE.
#pragma once

#define MU_LAW_FEATURE_DIM  12
#define MU_LAW_MU           255.0f
#define TARGET_NORMALIZATION_AVAILABLE 0

static const float mu_law_center[12] =
{
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
    0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
}
;

static const float mu_law_scale[12] =
{
    1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
    1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
}
;

// Usage:  scaled = (x[i] - mu_law_center[i]) / mu_law_scale[i];
//         out[i] = copysignf(log1pf(MU_LAW_MU * fabsf(scaled)) / log1pf(MU_LAW_MU), scaled);
