// cfc_inference.c — DenseCfC h=256 RMS-only per-channel INT8 inference
// INT8 weights copied from flash to SRAM at init for fast inner-loop access.
#include "weights.h"
#include "cfc_inference.h"
#include "normalization.h"
#include <string.h>  // memset

// ── LUT tables ─────────────────────────────────────────────────────────────
static float lecun_lut  [LUT_SIZE];
static float tanh_lut   [LUT_SIZE];
static float sigmoid_lut[LUT_SIZE];

// ── SRAM weight buffers (total ~163 KB, fits in ~300 KB SRAM budget) ──────
static int8_t sram_backbone_w[128 * 268];
static int8_t sram_ff1_w     [256 * 128];
static int8_t sram_ff2_w     [256 * 128];
static int8_t sram_time_a_w  [256 * 128];
static int8_t sram_time_b_w  [256 * 128];
static int8_t sram_head_w    [  5 * 256];

void cfc_lut_init(void) {
    // 1. Build activation LUTs
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = LUT_X_MIN + (float)i * LUT_DX;
        float tx = tanhf(x);
        lecun_lut[i]   = 1.7159f * tanhf(0.666f * x);
        tanh_lut[i]    = tx;
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
    }

    // 2. Copy INT8 weights flash → SRAM (fast access in inner loop)
    memcpy(sram_backbone_w, cfc_backbone_weight, sizeof(sram_backbone_w));
    memcpy(sram_ff1_w,      cfc_ff1_weight,      sizeof(sram_ff1_w));
    memcpy(sram_ff2_w,      cfc_ff2_weight,      sizeof(sram_ff2_w));
    memcpy(sram_time_a_w,   cfc_time_a_weight,   sizeof(sram_time_a_w));
    memcpy(sram_time_b_w,   cfc_time_b_weight,   sizeof(sram_time_b_w));
    memcpy(sram_head_w,     cfc_head_weight,     sizeof(sram_head_w));
}

// ── LUT lookup with linear interpolation ───────────────────────────────────

static inline float lut_lookup(const float *table, float x) {
    if (x <= LUT_X_MIN) return table[0];
    if (x >= LUT_X_MAX) return table[LUT_SIZE - 1];
    float idx_f = (x - LUT_X_MIN) / LUT_DX;
    int   idx   = (int)idx_f;
    float frac  = idx_f - (float)idx;
    return table[idx] * (1.0f - frac) + table[idx + 1] * frac;
}

// ── Per-channel INT8 fully-connected layer (reads weights from SRAM) ───────

void fc_int8(const float *input,
             const int8_t *weight_q,
             const float *bias,
             const float *scale,
             float *output,
             int in_dim,
             int out_dim)
{
    for (int j = 0; j < out_dim; j++) {
        float dot = 0.0f;
        const int8_t *w_row = weight_q + (size_t)j * in_dim;
        for (int i = 0; i < in_dim; i++) {
            dot += input[i] * (float)w_row[i];
        }
        output[j] = bias[j] + dot * scale[j];
    }
}

// ── Single CfC time-step ───────────────────────────────────────────────────

void cfc_step(const float input[CFC_INPUT_DIM],
              const float h_prev[CFC_HIDDEN_DIM],
              float h_new[CFC_HIDDEN_DIM],
              float ts)
{
    // 1. Concatenate input + hidden → backbone input
    float backbone_in[CFC_BACKBONE_IN];
    for (int i = 0; i < CFC_INPUT_DIM; i++)
        backbone_in[i] = input[i];
    for (int i = 0; i < CFC_HIDDEN_DIM; i++)
        backbone_in[CFC_INPUT_DIM + i] = h_prev[i];

    // 2. Backbone: Linear(268→128) + LeCun
    float x[CFC_BACKBONE_UNITS];
    fc_int8(backbone_in,
            sram_backbone_w, cfc_backbone_bias,
            cfc_backbone_weight_scale,
            x, CFC_BACKBONE_IN, CFC_BACKBONE_UNITS);
    for (int i = 0; i < CFC_BACKBONE_UNITS; i++)
        x[i] = lut_lookup(lecun_lut, x[i]);

    // 3. ff1: Linear(128→256) + tanh
    float ff1[CFC_HIDDEN_DIM];
    fc_int8(x, sram_ff1_w, cfc_ff1_bias, cfc_ff1_weight_scale,
            ff1, CFC_BACKBONE_UNITS, CFC_HIDDEN_DIM);
    for (int i = 0; i < CFC_HIDDEN_DIM; i++)
        ff1[i] = lut_lookup(tanh_lut, ff1[i]);

    // 4. ff2: Linear(128→256) + tanh
    float ff2[CFC_HIDDEN_DIM];
    fc_int8(x, sram_ff2_w, cfc_ff2_bias, cfc_ff2_weight_scale,
            ff2, CFC_BACKBONE_UNITS, CFC_HIDDEN_DIM);
    for (int i = 0; i < CFC_HIDDEN_DIM; i++)
        ff2[i] = lut_lookup(tanh_lut, ff2[i]);

    // 5. t_a: Linear(128→256) — no activation
    float ta[CFC_HIDDEN_DIM];
    fc_int8(x, sram_time_a_w, cfc_time_a_bias, cfc_time_a_weight_scale,
            ta, CFC_BACKBONE_UNITS, CFC_HIDDEN_DIM);

    // 6. t_b: Linear(128→256) — no activation
    float tb[CFC_HIDDEN_DIM];
    fc_int8(x, sram_time_b_w, cfc_time_b_bias, cfc_time_b_weight_scale,
            tb, CFC_BACKBONE_UNITS, CFC_HIDDEN_DIM);

    // 7. Gate interpolation
    for (int i = 0; i < CFC_HIDDEN_DIM; i++) {
        float t_interp = lut_lookup(sigmoid_lut, ta[i] * ts + tb[i]);
        h_new[i] = ff1[i] * (1.0f - t_interp) + t_interp * ff2[i];
    }
}

// ── Sequence inference ─────────────────────────────────────────────────────

void cfc_inference_int8(const float feature_seq[CFC_SEQ_LEN][CFC_INPUT_DIM],
                        float output[CFC_OUTPUT_DIM])
{
    float h[CFC_HIDDEN_DIM];
    memset(h, 0, sizeof(h));

    for (int t = 0; t < CFC_SEQ_LEN; t++) {
        float h_new[CFC_HIDDEN_DIM];
        cfc_step(feature_seq[t], h, h_new, 1.0f);
        memcpy(h, h_new, sizeof(h));
    }

    fc_int8(h,
            sram_head_w, cfc_head_bias,
            cfc_head_weight_scale,
            output, CFC_HIDDEN_DIM, CFC_OUTPUT_DIM);
}

// ── Mu-law encode ──────────────────────────────────────────────────────────

void mulaw_encode(float *features, int n) {
    for (int i = 0; i < n; i++) {
        float scaled = (features[i] - mu_law_center[i]) / mu_law_scale[i];
        float sign   = (scaled >= 0.0f) ? 1.0f : -1.0f;
        float abs_s  = sign * scaled;
        features[i]  = sign * log1pf(MU_LAW_MU * abs_s) / log1pf(MU_LAW_MU);
    }
}
