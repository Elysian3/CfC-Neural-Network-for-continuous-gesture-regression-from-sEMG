// cfc_inference.h — DenseCfC h=256 RMS-only per-channel INT8 inference
// Dimension constants come from generated weights.h; include it first.
#pragma once
#include <stdint.h>
#include <math.h>

// ── Dimension constants (must come from generated weights.h) ───────────────
#ifndef CFC_INPUT_DIM
#error "Include generated weights.h before cfc_inference.h"
#endif
#ifndef CFC_HIDDEN_DIM
#error "Include generated weights.h before cfc_inference.h"
#endif
#ifndef CFC_OUTPUT_DIM
#error "Include generated weights.h before cfc_inference.h"
#endif
#ifndef CFC_BACKBONE_UNITS
#error "Include generated weights.h before cfc_inference.h"
#endif
#ifndef CFC_BACKBONE_IN
#error "Include generated weights.h before cfc_inference.h"
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

// Full-sequence per-channel INT8 inference (batch, for offline validation).
void cfc_inference_int8(const float feature_seq[CFC_SEQ_LEN][CFC_INPUT_DIM],
                        float output[CFC_OUTPUT_DIM]);

// Single-step RNN inference with persistent hidden state.
// Feed one RMS frame at a time; hidden state carries across calls.
// Returns a CFC_OUTPUT_DIM-element output vector. Call cfc_reset_state() to zero h.
void cfc_single_step(const float feature[CFC_INPUT_DIM],
                     float output[CFC_OUTPUT_DIM]);

// Reset persistent hidden state to zero.
void cfc_reset_state(void);

// Mu-law encode a feature vector (in-place).
void mulaw_encode(float *features, int n);

// Invert mu-law normalization in-place. Used by the output wrapper and PC
// golden tests; center and scale must each contain n values.
void cfc_inverse_mulaw(float *values,
                       int n,
                       const float *center,
                       const float *scale,
                       float mu);
