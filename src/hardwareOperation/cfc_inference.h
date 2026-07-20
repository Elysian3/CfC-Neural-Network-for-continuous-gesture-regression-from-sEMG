// cfc_inference.h — DenseCfC h=256 RMS-only per-channel INT8 inference
// Auto-generated constants included from weights.h and normalization.h
#pragma once
#include <stdint.h>
#include <math.h>

// ── Dimension constants (mirrors weights.h) ────────────────────────────────
#ifndef CFC_INPUT_DIM
#define CFC_INPUT_DIM      12
#endif
#ifndef CFC_HIDDEN_DIM
#define CFC_HIDDEN_DIM     256
#endif
#ifndef CFC_OUTPUT_DIM
#define CFC_OUTPUT_DIM     5
#endif
#ifndef CFC_BACKBONE_UNITS
#define CFC_BACKBONE_UNITS  128
#endif
#ifndef CFC_BACKBONE_IN
#define CFC_BACKBONE_IN     (CFC_INPUT_DIM + CFC_HIDDEN_DIM)
#endif
#ifndef CFC_SEQ_LEN
#define CFC_SEQ_LEN         8
#endif

// ── LUT ────────────────────────────────────────────────────────────────────
#define LUT_SIZE 256
#define LUT_X_MIN (-8.0f)
#define LUT_X_MAX  8.0f
#define LUT_DX     ((LUT_X_MAX - LUT_X_MIN) / (float)(LUT_SIZE - 1))

// ── API ────────────────────────────────────────────────────────────────────

// Initialise LUT tables.  Must be called once before any inference.
void cfc_lut_init(void);

// Per-channel INT8 fully-connected layer.
//   output[j] = bias[j] + SUM_i(input[i] * weight_q[j*in_dim + i]) * scale[j]
void fc_int8(const float *input,
             const int8_t *weight_q,
             const float *bias,
             const float *scale,
             float *output,
             int in_dim,
             int out_dim);

// Single CfC time-step.  h_new may alias h_prev for in-place update.
void cfc_step(const float input[CFC_INPUT_DIM],
              const float h_prev[CFC_HIDDEN_DIM],
              float h_new[CFC_HIDDEN_DIM],
              float ts);

// Full-sequence per-channel INT8 inference.  Writes 5-element output.
void cfc_inference_int8(const float feature_seq[CFC_SEQ_LEN][CFC_INPUT_DIM],
                        float output[CFC_OUTPUT_DIM]);

// Mu-law encode a feature vector (in-place).
void mulaw_encode(float *features, int n);
